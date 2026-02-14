"""Candidate pair generation with classification and capping."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from itertools import combinations
from typing import Iterator

from .extract import EntityRecord, NameRecord
from .scripts import dominant_script


@dataclass(frozen=True, slots=True)
class NamePair:
    pair_id: str
    entity_id: str
    name_a: str
    script_a: str
    property_a: str
    name_b: str
    script_b: str
    property_b: str
    pair_category: str  # "cross_script", "latin_latin", "non_latin"
    source_datasets: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PairConfig:
    per_entity_cap: int = 100
    rng_seed: int = 42
    include_categories: frozenset[str] = frozenset({"cross_script", "latin_latin", "non_latin"})


def _make_pair_id(entity_id: str, name_a: str, name_b: str) -> str:
    """Deterministic pair ID: truncated SHA-256 of entity_id + canonical name pair."""
    payload = f"{entity_id}\0{name_a}\0{name_b}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _classify_pair(dom_a: str, dom_b: str) -> str:
    """Classify a pair based on dominant scripts."""
    if dom_a == dom_b:
        return "latin_latin" if dom_a == "Latin" else "non_latin"
    return "cross_script"


def _canonical_order(
    nr_a: NameRecord, dom_a: str,
    nr_b: NameRecord, dom_b: str,
    category: str,
) -> tuple[NameRecord, str, NameRecord, str]:
    """Canonical ordering for deterministic pair IDs.

    Cross-script with a Latin side: Latin first.
    Otherwise: alphabetical by dominant script, then by text.
    """
    if category == "cross_script":
        if dom_a == "Latin":
            return nr_a, dom_a, nr_b, dom_b
        if dom_b == "Latin":
            return nr_b, dom_b, nr_a, dom_a
    # Fallback: alphabetical by (dominant_script, text)
    key_a = (dom_a, nr_a.text)
    key_b = (dom_b, nr_b.text)
    if key_a <= key_b:
        return nr_a, dom_a, nr_b, dom_b
    return nr_b, dom_b, nr_a, dom_a


def _generate_entity_pairs(
    entity: EntityRecord,
    config: PairConfig,
) -> list[NamePair]:
    """Generate all C(n,2) pairs for an entity, classify, cap, and return."""
    names = entity.names
    if len(names) < 2:
        return []

    # Precompute dominant scripts
    dominants = {nr: dominant_script(nr.scripts) for nr in names}

    # Generate all pairs, classify, canonicalize
    by_category: dict[str, list[NamePair]] = {
        "cross_script": [],
        "latin_latin": [],
        "non_latin": [],
    }

    for nr_a, nr_b in combinations(names, 2):
        dom_a, dom_b = dominants[nr_a], dominants[nr_b]
        category = _classify_pair(dom_a, dom_b)

        if category not in config.include_categories:
            continue

        # Canonical ordering
        c_a, c_dom_a, c_b, c_dom_b = _canonical_order(nr_a, dom_a, nr_b, dom_b, category)

        pair = NamePair(
            pair_id=_make_pair_id(entity.entity_id, c_a.text, c_b.text),
            entity_id=entity.entity_id,
            name_a=c_a.text,
            script_a=c_dom_a,
            property_a=c_a.source_property,
            name_b=c_b.text,
            script_b=c_dom_b,
            property_b=c_b.source_property,
            pair_category=category,
            source_datasets=entity.datasets,
        )
        by_category[category].append(pair)

    # Cap with priority: cross_script first, then latin_latin, then non_latin
    budget = config.per_entity_cap
    selected: list[NamePair] = []

    # Entity-seeded RNG for deterministic selection.
    # Derive seed from entity_id via SHA-256 (not hash(), which is salted per-process).
    entity_hash = int(hashlib.sha256(entity.entity_id.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(config.rng_seed ^ entity_hash)

    for tier in ("cross_script", "latin_latin", "non_latin"):
        candidates = by_category.get(tier, [])
        if not candidates or budget <= 0:
            continue
        if len(candidates) <= budget:
            selected.extend(candidates)
            budget -= len(candidates)
        else:
            rng.shuffle(candidates)
            selected.extend(candidates[:budget])
            budget = 0

    return selected


def generate_pairs(
    entities: Iterator[EntityRecord] | list[EntityRecord],
    config: PairConfig | None = None,
) -> Iterator[NamePair]:
    """Generate within-entity name pairs across all entities.

    Yields NamePair instances with classification, canonical ordering,
    and per-entity capping applied.
    """
    if config is None:
        config = PairConfig()

    for entity in entities:
        yield from _generate_entity_pairs(entity, config)
