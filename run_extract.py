#!/usr/bin/env python3
"""CLI runner for FtM phonetic name pair extraction."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from time import perf_counter

from dd_ftm.core.datasets import DEFAULT_SANCTIONS_DATASETS, OPENSANCTIONS_URL
from dd_ftm.core.extract import ExtractionConfig, stream_entities
from dd_ftm.core.pairs import PairConfig, generate_pairs


def download_data(dest: Path) -> None:
    """Download OpenSanctions default collection.

    NOTE: OpenSanctions data is licensed CC BY-NC 4.0.
    Commercial use of the data or derived outputs requires a separate license.
    See https://www.opensanctions.org/licensing/
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {OPENSANCTIONS_URL}")
    print(f"  → {dest}")
    urllib.request.urlretrieve(OPENSANCTIONS_URL, dest)
    print(f"  Done ({dest.stat().st_size / 1e9:.1f} GB)")


def run_stats_only(data_path: Path, config: ExtractionConfig) -> None:
    """Profile without writing pairs — fast sanity check."""
    t0 = perf_counter()
    entity_count = 0
    name_count = 0
    pair_eligible = 0
    script_counts: Counter[str] = Counter()

    for entity in stream_entities(data_path, config):
        entity_count += 1
        name_count += len(entity.names)
        if len(entity.names) >= 2:
            pair_eligible += 1
        for nr in entity.names:
            for s in nr.scripts:
                script_counts[s] += 1

    elapsed = perf_counter() - t0
    print(f"\n{'='*60}")
    print(f"Stats-only profile ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  Entities:       {entity_count:>10,}")
    print(f"  Names:          {name_count:>10,}")
    print(f"  Pair-eligible:  {pair_eligible:>10,}")
    print(f"\n  Scripts:")
    for script, count in script_counts.most_common(20):
        print(f"    {script:20s} {count:>8,}")


def run_full(data_path: Path, output_path: Path, extract_config: ExtractionConfig, pair_config: PairConfig) -> None:
    """Full extraction: stream entities, generate pairs, write Parquet."""
    import pandas as pd

    t0 = perf_counter()
    print("Streaming entities and generating pairs...")

    rows: list[dict] = []
    entity_count = 0
    name_count = 0
    category_counts: Counter[str] = Counter()
    script_counts: Counter[str] = Counter()
    entity_pair_counts: Counter[str] = Counter()

    for entity in stream_entities(data_path, extract_config):
        entity_count += 1
        name_count += len(entity.names)
        entity_pairs = 0

        for pair in generate_pairs([entity], pair_config):
            rows.append({
                "pair_id": pair.pair_id,
                "entity_id": pair.entity_id,
                "name_a": pair.name_a,
                "script_a": pair.script_a,
                "property_a": pair.property_a,
                "name_b": pair.name_b,
                "script_b": pair.script_b,
                "property_b": pair.property_b,
                "pair_category": pair.pair_category,
                "source_datasets": "|".join(pair.source_datasets),
            })
            category_counts[pair.pair_category] += 1
            script_counts[pair.script_a] += 1
            script_counts[pair.script_b] += 1
            entity_pairs += 1

        if entity_pairs > 0:
            entity_pair_counts[entity.entity_id] = entity_pairs

    t_extract = perf_counter() - t0
    print(f"Extraction done in {t_extract:.1f}s")

    # Write Parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    file_size_mb = output_path.stat().st_size / 1e6

    # Summary
    print(f"\n{'='*60}")
    print(f"Extraction Summary")
    print(f"{'='*60}")
    print(f"  Entities:       {entity_count:>10,}")
    print(f"  Names:          {name_count:>10,}")
    print(f"  Total pairs:    {len(rows):>10,}")
    print(f"\n  Pairs by category:")
    for cat in ("cross_script", "latin_latin", "non_latin"):
        print(f"    {cat:20s} {category_counts[cat]:>8,}")
    print(f"\n  Script distribution (across pair endpoints):")
    for script, count in script_counts.most_common(15):
        print(f"    {script:20s} {count:>8,}")

    # Top entities by pair count
    top_entities = sorted(entity_pair_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    if top_entities:
        print(f"\n  Top 10 entities by pair count:")
        for eid, count in top_entities:
            print(f"    {eid:45s} {count:>6,}")

    print(f"\n  Output: {output_path} ({file_size_mb:.1f} MB)")
    print(f"  Time:   {perf_counter() - t0:.1f}s total")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract phonetic name pairs from OpenSanctions FtM data",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/targets.nested.json"),
        help="Path to FtM JSONL file (default: data/targets.nested.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/pairs.parquet"),
        help="Output Parquet path (default: output/pairs.parquet)",
    )
    parser.add_argument(
        "--per-entity-cap",
        type=int,
        default=100,
        help="Max pairs per entity (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for deterministic capping (default: 42)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download data if missing",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Profile without writing pairs",
    )
    args = parser.parse_args()

    if not args.data_path.exists():
        if args.download:
            download_data(args.data_path)
        else:
            print(f"Data file not found: {args.data_path}", file=sys.stderr)
            print("Use --download to fetch it.", file=sys.stderr)
            sys.exit(1)

    extract_config = ExtractionConfig(
        sanctions_datasets=DEFAULT_SANCTIONS_DATASETS,
    )

    if args.stats_only:
        run_stats_only(args.data_path, extract_config)
    else:
        pair_config = PairConfig(
            per_entity_cap=args.per_entity_cap,
            rng_seed=args.seed,
        )
        run_full(args.data_path, args.output, extract_config, pair_config)


if __name__ == "__main__":
    main()
