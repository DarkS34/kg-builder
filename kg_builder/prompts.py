from .relations import RelationSchema

_LANGUAGE_RULE = """\
# LANGUAGE
Write every concept name in the SAME LANGUAGE as the source material. Do not translate it, do not normalise it to English, do not transliterate it. Keep the original wording, accents and casing of the subject terminology.
The relation type keys listed below are FIXED IDENTIFIERS, not words of the text: emit them exactly as written, whatever the language of the source."""


def _type_preference_rule(schema: RelationSchema) -> str:
    if schema.fallback is None:
        return (
            "- Emit a relation only when one of the types above clearly applies. "
            "If none of them does, do not emit the relation at all."
        )
    specific = ", ".join(f"`{key}`" for key in schema.specific_keys())
    return (
        f"- Prefer the SPECIFIC type ({specific}) whenever its reading is clearly true; "
        f"reserve `{schema.fallback}` for real associations that fit none of the others. "
        f"Do not force a specific type when in doubt, but do not use `{schema.fallback}` "
        "as a default dumping ground either."
    )


def json_repair_prompt(broken_output: str, error_msg: str, shape: str = "object") -> str:
    return f"""\
The previous output could not be parsed as valid JSON, or does not satisfy the required schema.

Your task: produce a corrected JSON {shape} that (1) parses as valid JSON, and (2) preserves the original information as faithfully as possible.

# ERROR FROM THE PREVIOUS ATTEMPT
{error_msg}

# BROKEN OUTPUT TO REPAIR
{broken_output}

# RULES
- Return a single JSON {shape}. Nothing before it, nothing after it.
- No ```json, no backticks, no comments, no explanations.
- Do not translate any content: keep every value in its original language.
- Correctly escape line breaks (`\\n`) and inner quotes (`\\"`) inside strings.
- If the broken output is unrecoverable, return `{{}}`.

JSON:"""


def extract_typed_graph_prompt(source_text: str, schema: RelationSchema) -> str:
    return f"""\
Extract a KNOWLEDGE GRAPH from a fragment of source material on ANY subject. Identify the CONCEPTS of the subject and the TYPED RELATIONS between them, directly in the schema given below.

# WHAT COUNTS AS A VALID CONCEPT
A concept NAMES an idea of the subject: a term that could be an entry in a glossary or an index (a thing, technique, category, structure, phenomenon or named entity). It is NOT a phrase that describes or predicates something.
- Glossary test: if you would NOT put it as an entry in an index of the subject, it is NOT a concept.
- Do NOT extract: document metadata (section titles, bibliography, licences, authors), incidental scenarios from examples (objects, characters or specific situations that merely illustrate), or fragments that read as part of a sentence (they start with a verb, contain a conjugated verb, or express a condition or an action).
- Use a concise, canonical name for each concept (a noun or noun phrase), as it would appear in an index. Do not repeat the same concept with casing or plural variants.

{_LANGUAGE_RULE}

# RELATION TYPES (respect the SOURCE → TARGET direction)
Each relation is a triple [source, type, target]. Direction matters: choose the order that makes the stated reading true.
{schema.catalog_block()}

# RELATION RULES
- Source and target must be DIFFERENT, and both must appear in your `concepts` list. Relating a concept to itself is forbidden.
{_type_preference_rule(schema)}
- Extract only relations SUPPORTED by the text of the fragment, not by outside knowledge.

# OUTPUT
A single JSON object with exactly this shape:
{{
  "concepts": ["<concept>", "..."],
  "relations": [["<source>", "<type>", "<target>"], "..."]
}}
- `type` is one of: {schema.key_list()}. Nothing else.
- Every source and target in `relations` must appear in `concepts`.
- If the fragment yields no extractable concepts, return {{"concepts": [], "relations": []}}.
- No text before or after, no backticks, no comments.

# FRAGMENT
{source_text}

JSON:"""


def link_global_relations_prompt(concepts_block: str, schema: RelationSchema) -> str:
    return f"""\
You are given the COMPLETE INVENTORY of concepts of a knowledge graph, extracted from the whole body of material of a single subject. Extraction ran fragment by fragment, so relations between concepts that never appeared together in the same fragment are missing.

Your task: propose the TYPED RELATIONS that structure the subject at a GLOBAL level — above all the backbone that cuts across different parts of the syllabus: chains of dependencies, hierarchies and compositions between concepts that may have been explained in different sections.

{_LANGUAGE_RULE}

# RELATION TYPES (respect the SOURCE → TARGET direction)
{schema.catalog_block()}

# RULES
- Source and target must be DIFFERENT and both must appear LITERALLY in the inventory. Do not invent new concepts and do not rewrite their names.
- Propose relations that are true for the subject as a whole; focus on those connecting different parts of the syllabus, not on restating the obvious inside one subtopic.
{_type_preference_rule(schema)}
- Do not propose a relation from a concept to itself.

# OUTPUT
A single JSON object with exactly this shape:
{{
  "relations": [["<source>", "<type>", "<target>"], "..."]
}}
- `type` is one of: {schema.key_list()}.
- Every source and target must be in the inventory.
- No text before or after, no backticks, no comments.

# CONCEPT INVENTORY
{concepts_block}

JSON:"""


def clean_graph_nodes_prompt(nodes_block: str) -> str:
    return f"""\
You are given the NODES of a knowledge graph automatically extracted from a corpus, each with its outgoing relations as evidence. The extraction is noisy: there are duplicates, variants, metadata and fragments that are not concepts.

Your task: propose a cleanup as a MAPPING, without losing conceptual information.

# WHAT COUNTS AS A VALID NODE
A valid node NAMES a concept of the domain: a term that could be an entry in a glossary or an index of the subject (a thing, idea, technique, category or named entity). It is NOT a phrase that describes, explains or predicates something.

# WHAT TO DO
- MERGE variants of the same concept under a single canonical name: casing differences, singular/plural, articles, parenthetical annotations, or context suffixes (the same term with and without the name of a system or tool).
- REMOVE anything that does NOT name a concept:
  · document metadata (section titles, table of contents, bibliography, licences) and proper names of authors or works;
  · incidental scenarios from examples (objects, characters or specific situations that merely illustrate);
  · FRAGMENTS: descriptive phrases, clauses or predicates that read as part of a sentence instead of naming a concept (they start with a verb, contain a conjugated verb, or express a condition, property or action).
- REMOVE concepts clearly foreign to the domain (infer the domain from the set of nodes).

# WHEN IN DOUBT
- Glossary test: if you would NOT put it as an entry in an index of the subject, it goes to "drop".
- With domain CONCEPTS, keep: do not remove a term just because it is short, generic or infrequent.
- With FRAGMENTS, remove even when in doubt.
- Between two concepts, prefer merging over removing.

# CANONICAL NAME
- The canonical name MUST be one of the names in the input list. Do not invent new names, do not translate them, do not fix their spelling.
- Choose the most general and complete form; NEVER merge a general concept into a more specific one.
- Use the relations to disambiguate short names.

# OUTPUT
A single JSON object with exactly this shape:
{{
  "canonical": {{"<canonical name>": ["<alias>", "..."]}},
  "drop": ["<node to remove>", "..."]
}}
- Every input node appears exactly once: as a canonical key, inside an alias list, or in "drop".
- A concept without variants goes in as a canonical key with an empty alias list.
- No text before or after, no backticks, no comments.

# NODES
{nodes_block}

JSON:"""


def curate_graph_domains_prompt(nodes_block: str) -> str:
    return f"""\
You are given the already cleaned CONCEPTS of a knowledge graph, each with its outgoing relations as evidence. They come from a single corpus on one subject.

Your task: (1) group ALL the concepts into coherent thematic DOMAINS, and (2) mark which ones are too generic to be used as labels.

# DOMAINS
- A domain is a thematic block of the subject (in the style of the main topics or units of a syllabus), not a fine-grained tag.
- Propose FEW domains (as a guideline, between 3 and 8), each with a reasonable mass of concepts.
- Use the relation evidence: a concept that many others point to, or that aggregates many parts, usually NAMES a domain or sits close to one.
- COMPLETE AND MANDATORY PARTITION: the output must contain EACH AND EVERY concept of the input, exactly once. Go through them one by one and place them all; do not skip any out of haste or doubt, and do not repeat any in two domains.
- NO CATCH-ALL: if a concept does not fit clearly, assign it to the MOST RELATED domain according to its theme or its relations. It is FORBIDDEN to leave a concept without a domain, and FORBIDDEN to create a generic dumping-ground domain such as "Other", "Various", "Miscellaneous" or "Unclassified".

# NON-TAGGABLE CONCEPTS (non_taggable)
A non-taggable concept is too universal to identify what a specific item is about: names of the subject itself or of umbrella topics, names of languages or tools, and terms so transversal that they would appear in items of almost any subtopic.
- Test: if seeing that concept in an item does NOT let you tell what is being practised in particular (because it fits almost everything), it is non-taggable.
- They still belong to their domain; they are only listed apart so they can be excluded from labelling.
- When in doubt, do NOT mark it: a specific domain concept must remain taggable.

# NAMES
- Use the EXACT input names, both in the domains and in non_taggable. Do not invent, rename, translate or fix spelling.
- The DOMAIN NAMES are yours to write: short and descriptive, in the SAME LANGUAGE as the concepts.

# OUTPUT
A single JSON object with exactly this shape:
{{
  "domains": {{"<Domain name>": ["<concept>", "..."]}},
  "non_taggable": ["<concept>", "..."]
}}
- Every input concept appears exactly once inside "domains".
- Before answering, check that the number of concepts spread across "domains" matches the number of concepts in the input: if any is missing, place it in its most related domain.
- "non_taggable" is a subset of the input concepts (it may be empty).
- No text before or after, no backticks, no comments.

# CONCEPTS
{nodes_block}

JSON:"""
