"""FtM JSONL streaming, filtering, and name extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .datasets import DEFAULT_SANCTIONS_DATASETS
from .scripts import detect_scripts


@dataclass(frozen=True, slots=True)
class NameRecord:
    text: str
    scripts: frozenset[str]
    source_property: str  # "name", "alias", "previousName", "weakAlias"


@dataclass(frozen=True, slots=True)
class EntityRecord:
    entity_id: str
    datasets: tuple[str, ...]
    names: tuple[NameRecord, ...]


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    schema_filter: str = "Person"
    name_properties: tuple[str, ...] = ("name", "alias", "previousName", "weakAlias")
    sanctions_datasets: frozenset[str] | None = None  # None = use DEFAULT_SANCTIONS_DATASETS
    split_separator: str = " / "
    min_name_length: int = 2

    @property
    def effective_datasets(self) -> frozenset[str]:
        return self.sanctions_datasets if self.sanctions_datasets is not None else DEFAULT_SANCTIONS_DATASETS


def _clean_names(
    props: dict,
    config: ExtractionConfig,
) -> tuple[NameRecord, ...]:
    """Extract, clean, split, and deduplicate names from entity properties."""
    seen_texts: set[str] = set()
    records: list[NameRecord] = []

    for prop_key in config.name_properties:
        for raw_value in props.get(prop_key, []):
            # Split on separator (e.g. " / " for multi-name values)
            parts = raw_value.split(config.split_separator) if config.split_separator else [raw_value]
            for part in parts:
                text = part.strip()
                # Filter: must have >= 1 alpha char and meet min length
                if len(text) < config.min_name_length:
                    continue
                if not any(ch.isalpha() for ch in text):
                    continue
                # Dedup: exact text match, first occurrence wins
                if text in seen_texts:
                    continue
                seen_texts.add(text)
                records.append(NameRecord(
                    text=text,
                    scripts=detect_scripts(text),
                    source_property=prop_key,
                ))

    return tuple(records)


def stream_entities(
    path: str | Path,
    config: ExtractionConfig | None = None,
) -> Iterator[EntityRecord]:
    """Stream FtM JSONL, filtering by schema + datasets, extracting names.

    Yields EntityRecord for each entity that:
    - Matches the schema filter
    - Belongs to at least one sanctions dataset
    - Has at least one valid name after cleaning
    """
    if config is None:
        config = ExtractionConfig()
    target_datasets = config.effective_datasets

    with open(path, "r") as f:
        for line in f:
            entity = json.loads(line)
            if entity.get("schema") != config.schema_filter:
                continue
            entity_ds = set(entity.get("datasets", []))
            if not (entity_ds & target_datasets):
                continue

            props = entity.get("properties", {})
            names = _clean_names(props, config)
            if not names:
                continue

            yield EntityRecord(
                entity_id=entity["id"],
                datasets=tuple(sorted(entity_ds & target_datasets)),
                names=names,
            )


def extract_all(
    path: str | Path,
    config: ExtractionConfig | None = None,
) -> list[EntityRecord]:
    """Convenience: materialize all entities into a list."""
    return list(stream_entities(path, config))
