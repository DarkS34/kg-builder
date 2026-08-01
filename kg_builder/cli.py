import argparse
import logging
import sys
import warnings
from pathlib import Path

from loguru import logger

from .builder import KnowledgeGraphBuilder
from .config import NOISY_LOGGERS, NOISY_WARNING_MODULES, BuilderConfig
from .relations import BUILTIN_SCHEMAS, DEFAULT_RELATION_SCHEMA, RelationSchema


def setup_logging(quiet: bool = False) -> None:
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)
    for pattern in NOISY_WARNING_MODULES:
        warnings.filterwarnings("ignore", module=pattern)

    logger.remove()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logger.level("DEBUG", color="<blue><dim>")
    logger.level("INFO", color="<white>")
    logger.level("SUCCESS", color="<bold><green>")
    logger.level("WARNING", color="<yellow>")
    logger.level("ERROR", color="<bold><red>")

    logger.add(
        sys.stdout,
        level="WARNING" if quiet else "DEBUG",
        format="[{time:HH:mm:ss}] <level>{level: <8}</level> | <cyan>{module}</cyan> >> {message}",
        colorize=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kg-builder",
        description=(
            "Schema-guided knowledge graph extraction from documents, using local LLMs. "
            "The relation vocabulary is data, not code: point --relations at a JSON file "
            "to change it without touching a single prompt."
        ),
    )
    parser.add_argument(
        "--relations",
        metavar="NAME|FILE",
        help=(
            f"relation schema: a built-in name ({', '.join(sorted(BUILTIN_SCHEMAS))}) "
            "or a path to a JSON file describing your own (default: en)"
        ),
    )
    parser.add_argument("--host", help="Ollama host, e.g. localhost:11434 (env: OLLAMA_HOST)")
    parser.add_argument("--extraction-model", help="model used for extraction and global linking")
    parser.add_argument("--curation-model", help="model used for node cleanup and domain curation")
    parser.add_argument("--repair-model", help="model used to repair malformed JSON")
    parser.add_argument("--chunk-size", type=int, help="characters per extraction chunk")
    parser.add_argument(
        "--merge-qualifier-pattern",
        metavar="REGEX",
        help=r"regex stripped from node names before comparing them, e.g. '\s+in (python|java)\b'",
    )
    parser.add_argument(
        "--unclassified-domain", help="name of the bucket for concepts the model failed to place"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="directory for the default output filenames"
    )
    parser.add_argument("--quiet", action="store_true", help="only log warnings and errors")
    parser.add_argument(
        "--no-pull", action="store_true", help="skip the model availability check and warmup"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="extract a staging graph from a source directory")
    build.add_argument("input_dir", type=Path)
    build.add_argument("-o", "--output", type=Path)
    build.add_argument("-r", "--recursive", action="store_true", help="walk subdirectories")

    clean = subparsers.add_parser("clean", help="deduplicate and denoise a staging graph")
    clean.add_argument("staging", type=Path, nargs="?")
    clean.add_argument("-o", "--output", type=Path)

    curate = subparsers.add_parser("curate", help="shape a cleaned graph into the curated schema")
    curate.add_argument("cleaned", type=Path, nargs="?")
    curate.add_argument("-o", "--output", type=Path)

    run_all = subparsers.add_parser("all", help="run build, then clean, then curate")
    run_all.add_argument("input_dir", type=Path)
    run_all.add_argument("-r", "--recursive", action="store_true", help="walk subdirectories")

    subparsers.add_parser("relations", help="print the active relation schema as the model sees it")

    return parser


def config_from_args(args: argparse.Namespace) -> BuilderConfig:
    overrides = {
        "extraction_model": args.extraction_model,
        "curation_model": args.curation_model,
        "repair_model": args.repair_model,
        "ollama_host": args.host,
        "chunk_size": args.chunk_size,
        "merge_qualifier_pattern": args.merge_qualifier_pattern,
        "unclassified_domain": args.unclassified_domain,
        "output_dir": args.output_dir,
    }
    return BuilderConfig(**{k: v for k, v in overrides.items() if v is not None})


def schema_from_args(args: argparse.Namespace) -> RelationSchema:
    if args.relations is None:
        return DEFAULT_RELATION_SCHEMA
    if args.relations in BUILTIN_SCHEMAS:
        return BUILTIN_SCHEMAS[args.relations]
    return RelationSchema.from_json(Path(args.relations))


def print_relations(schema: RelationSchema) -> None:
    print(f"{len(schema)} relation type(s), fallback: {schema.fallback or '(none)'}\n")
    print(schema.catalog_block())
    print("\nGraph semantics written to `details` in the curated output:")
    for relation in schema:
        print(
            f"  {relation.key:<20} verbose={relation.verbose!r} "
            f"directed={relation.directed} acyclic={relation.acyclic} "
            f"use_in_embedding={relation.use_in_embedding}"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.quiet)

    try:
        schema = schema_from_args(args)
    except (OSError, ValueError, KeyError) as e:
        logger.error(f"Could not load the relation schema: {e}")
        return 2

    if args.command == "relations":
        print_relations(schema)
        return 0

    builder = KnowledgeGraphBuilder(config=config_from_args(args), relations=schema)

    try:
        if not args.no_pull:
            builder.bootstrap()

        if args.command == "build":
            builder.build(args.input_dir, args.output, recursive=args.recursive)
        elif args.command == "clean":
            builder.clean(args.staging, args.output)
        elif args.command == "curate":
            builder.curate(args.cleaned, args.output)
        elif args.command == "all":
            builder.build(args.input_dir, recursive=args.recursive)
            builder.clean()
            builder.curate()
    except (RuntimeError, OSError, KeyError) as e:
        logger.error(str(e))
        return 1

    return 0
