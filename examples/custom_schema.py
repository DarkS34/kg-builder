from kg_builder import (
    DEFAULT_RELATION_SCHEMA,
    BuilderConfig,
    KnowledgeGraphBuilder,
    RelationSchema,
    RelationType,
)

CAUSES = RelationType(
    key="causes",
    verbose="causes",
    definition="the SOURCE brings about, triggers or produces the TARGET.",
    reading="SOURCE causes TARGET",
    examples=(
        ("Insulin resistance", "Type 2 diabetes"),
        ("Deforestation", "Soil erosion"),
    ),
    directed=True,
    acyclic=True,
    use_in_embedding=True,
)

TREATS = RelationType(
    key="treats",
    verbose="treats",
    definition="the SOURCE is an intervention applied against the TARGET condition.",
    reading="SOURCE is used to treat TARGET",
    examples=(("Metformin", "Type 2 diabetes"),),
    directed=True,
    acyclic=False,
    use_in_embedding=True,
)

CLINICAL_SCHEMA = RelationSchema(
    types=(*DEFAULT_RELATION_SCHEMA.types, CAUSES, TREATS),
    fallback="related_to",
)

config = BuilderConfig(
    extraction_model="gemma3:27b",
    curation_model="gemma3:27b",
    chunk_size=8_000,
    merge_qualifier_pattern=r"\s+in (humans|animals)\b",
    output_dir="output",
)

if __name__ == "__main__":
    from kg_builder.cli import setup_logging

    setup_logging()

    builder = KnowledgeGraphBuilder(config=config, relations=CLINICAL_SCHEMA)
    builder.bootstrap()
    builder.build("corpus")
    builder.clean()
    builder.curate()
