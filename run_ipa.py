#!/usr/bin/env python3
"""CLI runner for name-to-IPA transcription."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

from dd_name_ipa import GenerationConfig, generate_batch


def _make_progress(t0_ref: list[float]):
    """Create a progress callback that reports rate and ETA."""
    def progress(done: int, total: int) -> None:
        elapsed = perf_counter() - t0_ref[0]
        rate = done / elapsed if elapsed > 0 else 0
        eta_h = (total - done) / rate / 3600 if rate > 0 else float("inf")
        print(f"  {done:>6,}/{total:,} ({rate:.2f} names/s, ETA {eta_h:.1f}h)", flush=True)
    return progress


def run_from_names_file(
    names_path: Path,
    output_path: Path,
    config: GenerationConfig,
) -> None:
    """Read names from a text file (one per line), generate IPA, write Parquet."""
    import pandas as pd

    with open(names_path) as f:
        names = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(names):,} names from {names_path}", flush=True)

    t0 = [perf_counter()]
    results = generate_batch(names, config, progress_callback=_make_progress(t0))
    elapsed = perf_counter() - t0[0]

    rows = [{
        "name": r.name,
        "ipa": r.ipa,
        "confidence": r.confidence,
        "n_candidates": len(r.candidates),
    } for r in results]
    df = pd.DataFrame(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")

    has_ipa = df["ipa"] != ""
    high = df["confidence"] >= 0.5
    print(f"\n{'='*50}", flush=True)
    print(f"Done in {elapsed/3600:.1f}h ({elapsed:.0f}s)", flush=True)
    print(f"  Names:            {len(df):>8,}", flush=True)
    print(f"  IPA generated:    {has_ipa.sum():>8,}", flush=True)
    print(f"  High conf (>=50%): {(has_ipa & high).sum():>8,}", flush=True)
    print(f"  Low conf (<50%):  {(has_ipa & ~high).sum():>8,}", flush=True)
    print(f"  Failed:           {(~has_ipa).sum():>8,}", flush=True)
    print(f"  Output: {output_path}", flush=True)


def run_from_parquet(
    input_path: Path,
    output_path: Path,
    name_columns: list[str],
    config: GenerationConfig,
) -> None:
    """Read names from Parquet, generate IPA, write augmented Parquet."""
    import pandas as pd

    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} rows from {input_path}", flush=True)

    for col in name_columns:
        if col not in df.columns:
            print(f"Column {col!r} not found in {input_path}", file=sys.stderr)
            sys.exit(1)

        unique_names = df[col].dropna().unique().tolist()
        print(f"\nProcessing column {col!r}: {len(unique_names):,} unique names", flush=True)

        t0 = [perf_counter()]
        results = generate_batch(unique_names, config, progress_callback=_make_progress(t0))
        elapsed = perf_counter() - t0[0]

        ipa_map = {r.name: r.ipa for r in results}
        conf_map = {r.name: r.confidence for r in results}

        df[f"{col}_ipa"] = df[col].map(ipa_map)
        df[f"{col}_ipa_confidence"] = df[col].map(conf_map)

        has_ipa = df[f"{col}_ipa"].notna() & (df[f"{col}_ipa"] != "")
        high_conf = df[f"{col}_ipa_confidence"] >= 0.5
        print(f"  Done in {elapsed:.1f}s ({len(unique_names) / elapsed:.1f} names/s)", flush=True)
        print(f"  IPA generated:       {has_ipa.sum():,}/{len(df):,}", flush=True)
        print(f"  High confidence (>=0.5): {(has_ipa & high_conf).sum():,}", flush=True)
        print(f"  Low confidence (<0.5):  {(has_ipa & ~high_conf).sum():,}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"\nOutput: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)", flush=True)


def run_interactive(config: GenerationConfig) -> None:
    """Interactive mode: type names, get IPA."""
    print("Interactive IPA transcription (Ctrl+D to exit)")
    print(f"Model: {config.model} | N={config.n} | T={config.temperature}")
    print()
    try:
        while True:
            name = input("Name: ").strip()
            if not name:
                continue
            results = generate_batch([name], config)
            r = results[0]
            print(f"  IPA:        {r.ipa}")
            print(f"  Confidence: {r.confidence:.0%} ({int(r.confidence * config.n)}/{config.n} agree)")
            if r.confidence < 1.0:
                from collections import Counter
                from dd_name_ipa.generate import _normalize_ipa
                counts = Counter(_normalize_ipa(c) for c in r.candidates)
                for ipa, count in counts.most_common():
                    marker = " ◀" if ipa == r.ipa else ""
                    print(f"    {count:>2d}× /{ipa}/{marker}")
            print()
    except (EOFError, KeyboardInterrupt):
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate IPA transcriptions for names using best-of-N LLM inference",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input Parquet file with name columns",
    )
    parser.add_argument(
        "--names-file",
        type=Path,
        help="Input text file with one name per line",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/pairs_ipa.parquet"),
        help="Output Parquet path (default: output/pairs_ipa.parquet)",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=["name_a", "name_b"],
        help="Name columns to transcribe (default: name_a name_b)",
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:8355/v1",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--model",
        default="nvidia/Qwen3-235B-A22B-FP4",
        help="Model name",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=10,
        help="Number of candidates per name (default: 10)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Max concurrent API requests (default: 8)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode: type names, get IPA",
    )
    args = parser.parse_args()

    config = GenerationConfig(
        api_base=args.api_base,
        model=args.model,
        n=args.n,
        temperature=args.temperature,
        concurrent_requests=args.concurrency,
    )

    if args.interactive:
        run_interactive(config)
    elif args.names_file:
        run_from_names_file(args.names_file, args.output, config)
    elif args.input:
        run_from_parquet(args.input, args.output, args.columns, config)
    else:
        parser.error("Provide --input, --names-file, or --interactive")


if __name__ == "__main__":
    main()
