from .builder import KnowledgeGraphBuilder
from .config import BuilderConfig
from .inference import InferenceError, OllamaEngine, create_engine
from .relations import (
    BUILTIN_SCHEMAS,
    DEFAULT_RELATION_SCHEMA,
    DEFAULT_RELATION_SCHEMA_EN,
    DEFAULT_RELATION_SCHEMA_ES,
    RelationSchema,
    RelationType,
)

__version__ = "0.1.0"

__all__ = [
    "BUILTIN_SCHEMAS",
    "DEFAULT_RELATION_SCHEMA",
    "DEFAULT_RELATION_SCHEMA_EN",
    "DEFAULT_RELATION_SCHEMA_ES",
    "BuilderConfig",
    "InferenceError",
    "KnowledgeGraphBuilder",
    "OllamaEngine",
    "RelationSchema",
    "RelationType",
    "create_engine",
]
