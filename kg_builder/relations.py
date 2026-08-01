import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RelationType:
    key: str
    verbose: str
    definition: str
    reading: str = ""
    examples: tuple[tuple[str, str], ...] = ()
    directed: bool = True
    acyclic: bool = False
    use_in_embedding: bool = True

    def __post_init__(self):
        if not self.key or self.key.split() != [self.key]:
            raise ValueError(f"Relation key must be a single whitespace-free token: {self.key!r}")
        if not self.verbose:
            raise ValueError(f"Relation '{self.key}' needs a verbose label")
        if not self.definition:
            raise ValueError(f"Relation '{self.key}' needs a definition (it is shown to the model)")
        object.__setattr__(self, "examples", tuple(tuple(pair) for pair in self.examples))

    def details(self) -> dict:
        return {
            "verbose": self.verbose,
            "directed": self.directed,
            "acyclic": self.acyclic,
            "use_in_embedding": self.use_in_embedding,
        }

    def catalog_entry(self) -> str:
        lines = [f"- `{self.key}` — {self.definition}"]
        if self.reading:
            lines.append(f'  Reading: "{self.reading}".')
        lines.append(
            "  Direction: "
            + (
                "directed — SOURCE and TARGET are not interchangeable."
                if self.directed
                else "symmetric — the order of SOURCE and TARGET does not matter."
            )
        )
        if self.examples:
            lines.append("  Examples:")
            lines.extend(f'    · ["{s}", "{self.key}", "{t}"]' for s, t in self.examples)
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict) -> "RelationType":
        return cls(
            key=data["key"],
            verbose=data["verbose"],
            definition=data["definition"],
            reading=data.get("reading", ""),
            examples=tuple(tuple(pair) for pair in data.get("examples", ())),
            directed=data.get("directed", True),
            acyclic=data.get("acyclic", False),
            use_in_embedding=data.get("use_in_embedding", True),
        )


@dataclass(frozen=True)
class RelationSchema:
    types: tuple[RelationType, ...]
    fallback: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "types", tuple(self.types))
        if not self.types:
            raise ValueError("A relation schema needs at least one relation type")
        keys = [relation.key for relation in self.types]
        duplicated = {key for key in keys if keys.count(key) > 1}
        if duplicated:
            raise ValueError(f"Duplicated relation key(s): {', '.join(sorted(duplicated))}")
        if self.fallback is not None and self.fallback not in keys:
            raise ValueError(
                f"fallback '{self.fallback}' is not one of the declared keys: {', '.join(keys)}"
            )

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(relation.key for relation in self.types)

    def __iter__(self):
        return iter(self.types)

    def __len__(self) -> int:
        return len(self.types)

    def __contains__(self, key: object) -> bool:
        return key in self.keys

    def __getitem__(self, key: str) -> RelationType:
        for relation in self.types:
            if relation.key == key:
                return relation
        raise KeyError(key)

    def get(self, key: str) -> RelationType | None:
        return self.get(key, None)

    def specific_keys(self) -> tuple[str, ...]:
        return tuple(key for key in self.keys if key != self.fallback)

    def catalog_block(self) -> str:
        return "\n".join(relation.catalog_entry() for relation in self.types)

    def key_list(self) -> str:
        return ", ".join(f'"{key}"' for key in self.keys)

    def to_dict(self) -> dict:
        return {
            "fallback": self.fallback,
            "types": [
                {
                    "key": relation.key,
                    "verbose": relation.verbose,
                    "definition": relation.definition,
                    "reading": relation.reading,
                    "examples": [list(pair) for pair in relation.examples],
                    "directed": relation.directed,
                    "acyclic": relation.acyclic,
                    "use_in_embedding": relation.use_in_embedding,
                }
                for relation in self.types
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RelationSchema":
        return cls(
            types=tuple(RelationType.from_dict(entry) for entry in data["types"]),
            fallback=data.get("fallback"),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "RelationSchema":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


DEFAULT_RELATION_SCHEMA_EN = RelationSchema(
    fallback="related_to",
    types=(
        RelationType(
            key="prerequisite",
            verbose="has as a prerequisite",
            definition=(
                "the SOURCE presupposes or needs the TARGET; the TARGET must be mastered "
                "BEFORE the SOURCE."
            ),
            reading="to learn SOURCE you must first know TARGET",
            examples=(
                ("Binary search", "Sorted list"),
                ("Multiplication", "Addition"),
                ("Integral calculus", "Derivatives"),
            ),
            directed=True,
            acyclic=True,
            use_in_embedding=False,
        ),
        RelationType(
            key="is_a",
            verbose="is a kind of",
            definition="the SOURCE is a TYPE, case or subclass of the TARGET.",
            reading="SOURCE is a kind of TARGET",
            examples=(
                ("Whale", "Mammal"),
                ("Sonnet", "Poem"),
                ("Equilateral triangle", "Triangle"),
            ),
            directed=True,
            acyclic=False,
            use_in_embedding=True,
        ),
        RelationType(
            key="part_of",
            verbose="is part of",
            definition="the SOURCE is a COMPONENT of the TARGET; the TARGET is the whole that contains it.",
            reading="SOURCE is part of TARGET",
            examples=(
                ("Nucleus", "Cell"),
                ("Chorus", "Song"),
                ("Engine", "Car"),
            ),
            directed=True,
            acyclic=False,
            use_in_embedding=True,
        ),
        RelationType(
            key="related_to",
            verbose="is related to",
            definition=(
                "a genuine semantic association that does not cleanly fit any of the types above."
            ),
            reading="SOURCE and TARGET are semantically associated",
            examples=(
                ("Supply", "Demand"),
                ("Photosynthesis", "Cellular respiration"),
            ),
            directed=False,
            acyclic=False,
            use_in_embedding=True,
        ),
    ),
)


# Spanish counterpart. `key`, `verbose` and `examples` are translated because they
# surface in the output and in the consumer's own prompts; `definition` and `reading`
# stay English, since they are embedded in this package's English prompt scaffolding.
DEFAULT_RELATION_SCHEMA_ES = RelationSchema(
    fallback="relacionado",
    types=(
        RelationType(
            key="prerrequisito",
            verbose="tiene como prerrequisito",
            definition=(
                "the SOURCE presupposes or needs the TARGET; the TARGET must be mastered "
                "BEFORE the SOURCE."
            ),
            reading="to learn SOURCE you must first know TARGET",
            examples=(
                ("Búsqueda binaria", "Lista ordenada"),
                ("Multiplicación", "Suma"),
                ("Cálculo integral", "Derivadas"),
            ),
            directed=True,
            acyclic=True,
            use_in_embedding=False,
        ),
        RelationType(
            key="es_un",
            verbose="es un tipo de",
            definition="the SOURCE is a TYPE, case or subclass of the TARGET.",
            reading="SOURCE is a kind of TARGET",
            examples=(
                ("Ballena", "Mamífero"),
                ("Soneto", "Poema"),
                ("Triángulo equilátero", "Triángulo"),
            ),
            directed=True,
            acyclic=False,
            use_in_embedding=True,
        ),
        RelationType(
            key="parte_de",
            verbose="es parte de",
            definition="the SOURCE is a COMPONENT of the TARGET; the TARGET is the whole that contains it.",
            reading="SOURCE is part of TARGET",
            examples=(
                ("Núcleo", "Célula"),
                ("Estribillo", "Canción"),
                ("Motor", "Automóvil"),
            ),
            directed=True,
            acyclic=False,
            use_in_embedding=True,
        ),
        RelationType(
            key="relacionado",
            verbose="se relaciona con",
            definition=(
                "a genuine semantic association that does not cleanly fit any of the types above."
            ),
            reading="SOURCE and TARGET are semantically associated",
            examples=(
                ("Oferta", "Demanda"),
                ("Fotosíntesis", "Respiración celular"),
            ),
            directed=False,
            acyclic=False,
            use_in_embedding=True,
        ),
    ),
)


BUILTIN_SCHEMAS = {
    "en": DEFAULT_RELATION_SCHEMA_EN,
    "es": DEFAULT_RELATION_SCHEMA_ES,
}

DEFAULT_RELATION_SCHEMA = DEFAULT_RELATION_SCHEMA_ES

