"""Render any graph the pipeline writes into a single self-contained HTML viewer.

Nothing here touches the pipeline: it reads one of the JSON files produced by
`build` / `clean` / `curate` and folds it into one view model, which is embedded
in `assets/viewer.html`. No network calls, no extra dependency, no server — the
output is a file you can open, move or email.
"""

import json
import re
import unicodedata
import webbrowser
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from loguru import logger

from .relations import RelationSchema

TEMPLATE_PATH = Path(__file__).parent / "assets" / "viewer.html"

DATA_PLACEHOLDER = "__KG_DATA__"
TITLE_PLACEHOLDER = "__KG_TITLE__"


# VIEW MODEL --------------------------------------------------------------------------------------


def _slug(text: str) -> str:
    stripped = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"[^a-z0-9]+", "_", stripped.lower()).strip("_") or "relation"


def is_curated(graph: dict) -> bool:
    return "concepts_by_domains" in graph


def _relation_meta(key: str, schema: RelationSchema | None, details: dict | None = None) -> dict:
    details = details or {}
    relation = schema[key] if schema is not None and key in schema else None
    if relation is not None:
        meta = {"key": key, **relation.details()}
    else:
        meta = {
            "key": key,
            "verbose": details.get("verbose") or key.replace("_", " "),
            "directed": True,
            "acyclic": False,
            "use_in_embedding": True,
        }
    meta.update({k: v for k, v in details.items() if k in meta})
    return meta


def _from_curated(graph: dict, schema: RelationSchema | None) -> tuple[list, list, dict, list]:
    """(node names, groups, name -> group, typed edges) out of the curated schema."""
    by_verbose = {r.verbose: r.key for r in schema} if schema is not None else {}

    group_of: dict[str, str] = {}
    for domain, concepts in graph.get("concepts_by_domains", {}).items():
        for concept in concepts:
            group_of.setdefault(concept, domain)

    edges: list[tuple[str, str, str]] = []
    relations: list[dict] = []
    for group in graph.get("relations", []):
        details = group.get("details", {}) or {}
        verbose = details.get("verbose", "")
        key = group.get("key") or by_verbose.get(verbose) or _slug(verbose)
        relations.append(_relation_meta(key, schema, details))
        for source, targets in (group.get("relations_data", {}) or {}).items():
            edges.extend((source, key, target) for target in targets)

    names = set(group_of)
    for source, _, target in edges:
        names.update((source, target))
    return sorted(names), relations, group_of, edges


def _from_flat(graph: dict, schema: RelationSchema | None) -> tuple[list, list, dict, list]:
    edges = [tuple(triple) for triple in graph.get("relations", []) if len(triple) == 3]
    names = set(graph.get("entities", []))
    for source, _, target in edges:
        names.update((source, target))

    keys = list(dict.fromkeys(list(graph.get("edges", [])) + [key for _, key, _ in edges]))
    relations = [_relation_meta(key, schema) for key in keys]
    return sorted(names), relations, _communities(sorted(names), edges), edges


# A flat graph has no domains, so clusters stand in for them: they are what the
# eye is looking for anyway, and they make the layout readable before `curate`
# has ever run. Each cluster is named after its best-connected member.
def _communities(names: list[str], edges: list[tuple[str, str, str]]) -> dict[str, str]:
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(names)
    graph.add_edges_from((source, target) for source, _, target in edges)

    try:
        groups = nx.community.louvain_communities(graph, seed=0)
    except (AttributeError, ImportError):  # pragma: no cover - very old networkx
        groups = list(nx.connected_components(graph))

    degrees = dict(graph.degree())
    group_of: dict[str, str] = {}
    for members in sorted(groups, key=len, reverse=True):
        hub = max(members, key=lambda n: (degrees.get(n, 0), -len(n), n))
        label = f"Cluster · {hub}"
        for member in members:
            group_of[member] = label
    return group_of


def build_view_model(
    graph: dict,
    schema: RelationSchema | None = None,
    title: str = "Knowledge graph",
    source: str = "",
) -> dict:
    """Fold a staging, cleaned or curated graph into what the viewer needs.

    Nodes and links are emitted as positional arrays: they dominate the payload,
    and the graph is re-read on every page load.
    """
    if not is_curated(graph) and not ("entities" in graph or "relations" in graph):
        raise ValueError(
            "Unrecognised graph JSON — expected a curated graph ('concepts_by_domains') "
            "or a flat one ('entities' / 'relations')"
        )

    reader = _from_curated if is_curated(graph) else _from_flat
    names, relations, group_of, edges = reader(graph, schema)

    unknown = "—"
    sizes: dict[str, int] = defaultdict(int)
    for name in names:
        sizes[group_of.get(name, unknown)] += 1
    groups = [
        {"name": name, "count": count}
        for name, count in sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    group_index = {group["name"]: i for i, group in enumerate(groups)}

    node_index = {name: i for i, name in enumerate(names)}
    relation_index = {relation["key"]: i for i, relation in enumerate(relations)}

    symmetric = {i for i, relation in enumerate(relations) if not relation["directed"]}

    links, seen = [], set()
    counts: dict[str, int] = defaultdict(int)
    degree = [0] * len(names)
    for source, key, target in edges:
        s, t, r = node_index.get(source), node_index.get(target), relation_index.get(key)
        # Self-loops have nothing to draw; a symmetric relation stated in both
        # directions is one edge, not two lines on top of each other.
        if s is None or t is None or r is None or s == t:
            continue
        signature = (min(s, t), max(s, t), r) if r in symmetric else (s, t, r)
        if signature in seen:
            continue
        seen.add(signature)
        links.append([s, t, r])
        counts[key] += 1
        degree[s] += 1
        degree[t] += 1

    non_taggable = set(graph.get("generic_non_taggable_concepts", []))
    nodes = [
        [name, group_index[group_of.get(name, unknown)], 1 if name in non_taggable else 0]
        for name in names
    ]

    for relation in relations:
        relation["count"] = counts[relation["key"]]

    return {
        "meta": {
            "title": title,
            "source": source,
            "kind": "curated" if is_curated(graph) else "flat",
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "isolated": sum(1 for d in degree if d == 0),
        },
        "relations": relations,
        "groups": groups,
        "nodes": nodes,
        "links": links,
    }


# RENDERING ---------------------------------------------------------------------------------------


def render_html(model: dict) -> str:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    # `<` is escaped so the payload can never terminate its own <script> tag.
    payload = json.dumps(model, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    title = model["meta"]["title"].replace("&", "&amp;").replace("<", "&lt;")
    # One pass, so neither substitution can rewrite what the other just inserted.
    values = {DATA_PLACEHOLDER: payload, TITLE_PLACEHOLDER: title}
    return re.sub(
        "|".join(re.escape(token) for token in values),
        lambda m: values[m.group()],
        template,
    )


def visualize(
    graph: dict,
    output_path: str | Path,
    schema: RelationSchema | None = None,
    title: str = "Knowledge graph",
    source: str = "",
    open_browser: bool = False,
) -> Path:
    model = build_view_model(graph, schema, title=title, source=source)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(model), encoding="utf-8")

    logger.success(
        f"Viewer written — {len(model['nodes'])} node(s), {len(model['links'])} link(s), "
        f"{len(model['groups'])} group(s), {len(model['relations'])} relation type(s) "
        f"→ {output_path}"
    )
    if open_browser:
        webbrowser.open(output_path.resolve().as_uri())
    return output_path
