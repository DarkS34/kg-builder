import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import networkx as nx
from loguru import logger

from . import documents
from .config import BuilderConfig
from .inference import create_engine
from .json_utils import parse_json_object, parse_with_repair
from .prompts import (
    clean_graph_nodes_prompt,
    curate_graph_domains_prompt,
    extract_typed_graph_prompt,
    link_global_relations_prompt,
)
from .relations import DEFAULT_RELATION_SCHEMA, RelationSchema

MIN_SINGULARIZE_LENGTH = 3


class KnowledgeGraphBuilder:
    def __init__(
        self,
        config: BuilderConfig | None = None,
        relations: RelationSchema | None = None,
        engine=None,
        verbose: bool = True,
    ):
        logger.enable(__package__) if verbose else logger.disable(__package__)

        self.config = config or BuilderConfig()
        self.relations = relations or DEFAULT_RELATION_SCHEMA
        self.engine = engine or create_engine(self.config)
        self._converter = None

    @property
    def converter(self):
        if self._converter is None:
            self._converter = documents.build_converter()
        return self._converter

    def bootstrap(self) -> None:
        if not self.engine.is_available():
            raise RuntimeError(
                f"Cannot reach the inference engine at {self.config.ollama_host}. "
                "Make sure it is running before building."
            )
        logger.info(f"Preparing model(s): {', '.join(self.config.models)}")
        failed = [m for m in self.config.models if not self.engine.ensure_model(m)]
        if failed:
            raise RuntimeError(f"Failed to install model(s): {', '.join(failed)}")
        for model in self.config.models:
            self.engine.warmup(model)
        logger.success("All models ready")

    # EXTRACTION ----------------------------------------------------------------------------------

    def build(
        self,
        input_dir: str | Path,
        output_file_path: str | Path | None = None,
        recursive: bool = False,
    ) -> dict:
        output_file_path = output_file_path or self.config.staging_path
        files = documents.list_source_files(input_dir, recursive=recursive)
        if not files:
            logger.error(f"No supported files found in: {input_dir}")
            return {}

        logger.info(f"Found {len(files)} file(s) - extracting knowledge graph")

        concepts: set[str] = set()
        relations: set[tuple[str, str, str]] = set()
        for idx, file_path in enumerate(files, 1):
            try:
                text = documents.to_markdown(self.converter, file_path)
            except Exception as e:
                logger.exception(f"[{file_path.name}] skipped: {e}")
                continue
            chunks = documents.chunk_text(text, self.config.chunk_size)
            logger.info(f"[{idx}/{len(files)} {file_path.name}] {len(chunks)} chunk(s)")
            for ci, chunk in enumerate(chunks, 1):
                tag = f"[{idx}/{len(files)} {file_path.name} · chunk {ci}/{len(chunks)}] "
                chunk_concepts, chunk_relations = self._extract_from_chunk(chunk, tag)
                concepts.update(chunk_concepts)
                relations.update(tuple(r) for r in chunk_relations)

        if not concepts:
            logger.error("No concepts extracted from any file")
            return {}

        for source, _, target in relations:
            concepts.update((source, target))

        before = len(relations)
        relations.update(tuple(r) for r in self._link_global(sorted(concepts)))
        logger.info(f"Global linking pass added {len(relations) - before} relation(s)")

        staging = self._assemble(concepts, relations)
        documents.save_json(staging, output_file_path)
        logger.success(
            f"Staging KG written — {len(staging['entities'])} entity(ies), "
            f"{len(staging['relations'])} relation(s) → {output_file_path}"
        )
        return staging

    def _extract_from_chunk(self, chunk: str, log_prefix: str) -> tuple[list[str], list[list[str]]]:
        prompt = extract_typed_graph_prompt(chunk, self.relations)
        response = self.engine.generate(
            model=self.config.extraction_model, think=False, prompt=prompt
        ).response
        raw = self._parse_object(response, log_prefix)
        if raw is None:
            return [], []
        concepts = [c.strip() for c in raw.get("concepts", []) if isinstance(c, str) and c.strip()]
        relations = self._valid_relations(raw.get("relations", []), allowed=None)
        return concepts, relations

    def _link_global(self, inventory: list[str]) -> list[list[str]]:
        if len(inventory) < 2:
            return []
        prompt = link_global_relations_prompt(self._concepts_block(inventory), self.relations)
        response = self.engine.generate(
            model=self.config.extraction_model, think=False, prompt=prompt
        ).response
        raw = self._parse_object(response, "[global] ")
        if raw is None:
            return []
        return self._valid_relations(raw.get("relations", []), allowed=set(inventory))

    def _parse_object(self, response: str, log_prefix: str) -> dict | None:
        result, error = parse_with_repair(
            response,
            parse_json_object,
            engine=self.engine,
            repair_model=self.config.repair_model,
            max_attempts=self.config.max_repair_attempts,
            shape="object",
            log_prefix=log_prefix,
        )
        if result is None:
            logger.warning(f"{log_prefix}unrecoverable JSON: {error}")
        return result

    def _valid_relations(self, raw: list, allowed: set[str] | None) -> list[list[str]]:
        out = []
        for triple in raw or []:
            if not (isinstance(triple, list) and len(triple) == 3):
                continue
            if not all(isinstance(x, str) for x in triple):
                continue
            source, relation, target = (x.strip() for x in triple)
            if not (source and target) or source == target or relation not in self.relations:
                continue
            if allowed is not None and (source not in allowed or target not in allowed):
                continue
            out.append([source, relation, target])
        return out

    @staticmethod
    def _concepts_block(concepts: list[str]) -> str:
        return "\n".join(f"- {c}" for c in concepts)

    @staticmethod
    def _assemble(concepts: set[str], relations: set[tuple[str, str, str]]) -> dict:
        rels = sorted(list(r) for r in relations)
        return {
            "entities": sorted(concepts),
            "edges": sorted({r[1] for r in rels}),
            "relations": rels,
        }

    # CLEANUP -------------------------------------------------------------------------------------

    # clean proposes a deduplicated, denoised graph for manual review; it never
    # touches the final curated graph.
    def clean(
        self,
        staging_path: str | Path | None = None,
        output_path: str | Path | None = None,
    ) -> dict:
        staging_path = staging_path or self.config.staging_path
        output_path = output_path or self.config.cleaned_path

        graph = documents.load_json(staging_path)
        nodes = self._node_universe(graph)
        logger.info(
            f"Loaded staging KG — {len(graph['entities'])} entity(ies), "
            f"{len(graph['relations'])} relation(s), {len(nodes)} node(s) in universe"
        )

        det_map, representatives = self._deterministic_merge(nodes)
        logger.info(f"Deterministic merge — {len(nodes)} → {len(representatives)} node(s)")

        canonical, drop = self._propose_alias_mapping(representatives, graph["relations"], det_map)
        llm_map = self._llm_alias_map(canonical, set(representatives))
        node_map = self._compose_node_map(nodes, det_map, llm_map, drop)
        cleaned = self._apply_node_map(graph, node_map)

        documents.save_json(cleaned, output_path)
        logger.success(
            f"Cleaned proposal → {output_path} — "
            f"entities {len(graph['entities'])}→{len(cleaned['entities'])}, "
            f"relations {len(graph['relations'])}→{len(cleaned['relations'])}"
        )
        return cleaned

    # The node universe is entities plus every relation endpoint, so phrases that
    # only ever appear inside relations also get judged and can't stay dangling.
    @staticmethod
    def _node_universe(graph: dict) -> list[str]:
        nodes = set(graph["entities"])
        for source, _, target in graph["relations"]:
            nodes.add(source)
            nodes.add(target)
        return sorted(nodes)

    # Conservative key: casing, accents, an optional configurable qualifier and a
    # trailing plural suffix. No parenthesis stripping, to keep e.g. O(n) vs O(log n) apart.
    def _norm_key(self, name: str) -> str:
        s = name.lower().strip()
        if self.config.merge_qualifier_pattern:
            s = re.sub(self.config.merge_qualifier_pattern, "", s)
        s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
        s = re.sub(r"\s+", " ", s).strip()
        for suffix in self.config.plural_suffixes:
            if len(s) > MIN_SINGULARIZE_LENGTH and s.endswith(suffix):
                return s[: -len(suffix)]
        return s

    # Merge only mechanical variants deterministically; the survivor is a
    # capitalized, short form when available.
    def _deterministic_merge(self, nodes: list[str]) -> tuple[dict, list[str]]:
        groups = defaultdict(list)
        for n in nodes:
            groups[self._norm_key(n)].append(n)
        variant_to_canon = {}
        representatives = []
        for members in groups.values():
            canon = min(members, key=lambda x: (x[:1].islower(), len(x)))
            representatives.append(canon)
            for m in members:
                variant_to_canon[m] = canon
        return variant_to_canon, sorted(representatives)

    # Each node is listed with its outgoing relations (remapped to representatives)
    # so the model can disambiguate short or ambiguous names.
    def _nodes_block(self, nodes: list[str], relations: list[list], det_map: dict) -> str:
        outgoing = defaultdict(list)
        for source, relation, target in relations:
            outgoing[det_map.get(source, source)].append(f"{relation} {det_map.get(target, target)}")
        lines = []
        for n in nodes:
            evidence = "; ".join(outgoing[n][: self.config.max_evidence_relations])
            lines.append(f"- {n}" + (f"  [{evidence}]" if evidence else ""))
        return "\n".join(lines)

    def _propose_alias_mapping(
        self, nodes: list[str], relations: list[list], det_map: dict
    ) -> tuple[dict, set]:
        prompt = clean_graph_nodes_prompt(self._nodes_block(nodes, relations, det_map))
        response = self.engine.generate(
            model=self.config.curation_model, think=False, prompt=prompt
        ).response
        raw = self._parse_object(response, "[clean] ")
        if raw is None:
            logger.warning("Alias proposal unavailable — keeping the deterministic merge only")
            return {}, set()
        return raw.get("canonical", {}) or {}, set(raw.get("drop", []) or [])

    # Force every canonical to be an existing node (drops invented names).
    @staticmethod
    def _llm_alias_map(canonical: dict, valid: set) -> dict:
        alias_map = {}
        for canon, aliases in canonical.items():
            target = canon if canon in valid else next((a for a in aliases if a in valid), None)
            if target is None:
                continue
            for m in (canon, *aliases):
                if m in valid:
                    alias_map[m] = target
        return alias_map

    # Compose deterministic merge -> LLM merge -> drops into one node->canonical map;
    # a dropped node maps to None.
    @staticmethod
    def _compose_node_map(nodes: list[str], det_map: dict, llm_map: dict, drop: set) -> dict:
        node_map = {}
        for n in nodes:
            rep = det_map.get(n, n)
            node_map[n] = None if rep in drop else llm_map.get(rep, rep)
        return node_map

    @staticmethod
    def _apply_node_map(graph: dict, node_map: dict) -> dict:
        entities = sorted({c for c in node_map.values() if c})
        ents = set(entities)
        relations = set()
        for source, relation, target in graph["relations"]:
            canon_source, canon_target = node_map.get(source), node_map.get(target)
            # Keep a relation only when both remapped endpoints survive as entities.
            if canon_source in ents and canon_target in ents:
                relations.add((canon_source, relation, canon_target))
        return {
            "entities": entities,
            "edges": sorted({r[1] for r in relations}),
            "relations": sorted(list(r) for r in relations),
        }

    # CURATE --------------------------------------------------------------------------------------

    def curate(
        self,
        cleaned_path: str | Path | None = None,
        output_path: str | Path | None = None,
    ) -> dict:
        cleaned_path = cleaned_path or self.config.cleaned_path
        output_path = output_path or self.config.curated_path

        graph = documents.load_json(cleaned_path)
        concepts = graph["entities"]
        relations = graph["relations"]
        logger.info(f"Loaded cleaned KG — {len(concepts)} concept(s), {len(relations)} relation(s)")

        concepts_by_domains, non_taggable = self._curate_domains(concepts, relations)
        logger.info(
            f"Domains — {len(concepts_by_domains)} domain(s), "
            f"{len(non_taggable)} non-taggable concept(s)"
        )

        universe = {c for cs in concepts_by_domains.values() for c in cs}
        typed = self._build_typed_relations(relations, universe)
        typed = self._break_cycles(typed)
        logger.info(f"Relations — {len(typed)} typed group(s) over {len(universe)} concept(s)")

        curated = {
            "concepts_by_domains": concepts_by_domains,
            "generic_non_taggable_concepts": non_taggable,
            "relations": typed,
        }
        documents.save_json(curated, output_path)
        logger.success(
            f"Curated draft → {output_path} — "
            f"{len(universe)} concept(s), {len(typed)} typed relation group(s)"
        )
        return curated

    def _curate_domains(self, concepts: list[str], relations: list[list]) -> tuple[dict, list[str]]:
        prompt = curate_graph_domains_prompt(self._nodes_block(concepts, relations, {}))
        response = self.engine.generate(
            model=self.config.curation_model, think=False, prompt=prompt
        ).response
        raw = self._parse_object(response, "[domains] ") or {}
        return self._reconcile_domains(
            concepts, raw.get("domains", {}) or {}, raw.get("non_taggable", []) or []
        )

    def _reconcile_domains(
        self, concepts: list[str], domains_raw: dict, non_taggable_raw: list
    ) -> tuple[dict, list[str]]:
        valid = set(concepts)
        placed: set[str] = set()
        by_domain: dict[str, list[str]] = {}
        for domain, members in domains_raw.items():
            if not isinstance(members, list):
                continue
            kept = sorted({c for c in members if c in valid and c not in placed})
            if kept:
                placed.update(kept)
                by_domain[domain] = kept
        leftover = sorted(c for c in concepts if c not in placed)
        if leftover:
            by_domain.setdefault(self.config.unclassified_domain, []).extend(leftover)
        non_taggable = sorted({c for c in non_taggable_raw if c in valid})
        return by_domain, non_taggable

    def _build_typed_relations(self, relations: list[list], universe: set) -> list[dict]:
        buckets = {relation.key: defaultdict(list) for relation in self.relations}
        for source, key, target in relations:
            if source not in universe or target not in universe:
                continue
            if key not in self.relations:
                key = self.relations.fallback
                if key is None:
                    continue
            if target not in buckets[key][source]:
                buckets[key][source].append(target)

        typed = []
        for relation in self.relations:
            data = buckets[relation.key]
            if not data:
                continue
            relations_data = {source: sorted(data[source]) for source in sorted(data)}
            typed.append({"details": relation.details(), "relations_data": relations_data})
        return typed

    @staticmethod
    def _break_cycles(typed: list[dict]) -> list[dict]:
        for group in typed:
            details = group["details"]
            if not (details.get("acyclic") and details.get("directed")):
                continue
            graph = nx.DiGraph()
            for source, targets in group["relations_data"].items():
                graph.add_edges_from((source, target) for target in targets)
            removed = []
            while not nx.is_directed_acyclic_graph(graph):
                source, target = nx.find_cycle(graph)[-1][:2]
                graph.remove_edge(source, target)
                removed.append((source, target))
            if not removed:
                continue
            rebuilt = defaultdict(list)
            for source, target in graph.edges():
                rebuilt[source].append(target)
            group["relations_data"] = {s: sorted(rebuilt[s]) for s in sorted(rebuilt)}
            logger.warning(
                f"Broke {len(removed)} back-edge(s) in acyclic '{details['verbose']}': {removed}"
            )
        return typed
