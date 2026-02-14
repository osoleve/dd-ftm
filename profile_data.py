"""Profile the OpenSanctions default collection for the phonetic pairs pipeline."""

import json
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations

DATA_PATH = "data/targets.nested.json"


def detect_scripts(name: str) -> set[str]:
    """Detect Unicode script blocks present in a name."""
    scripts = set()
    for ch in name:
        if ch.isalpha():
            block = unicodedata.name(ch, "").split()[0] if unicodedata.name(ch, "") else "UNKNOWN"
            # Map to coarser script categories
            if block in ("LATIN",):
                scripts.add("Latin")
            elif block in ("CYRILLIC",):
                scripts.add("Cyrillic")
            elif block in ("ARABIC", "ARABIC-INDIC"):
                scripts.add("Arabic")
            elif block in ("CJK", "KANGXI", "IDEOGRAPHIC"):
                scripts.add("CJK")
            elif block in ("HANGUL",):
                scripts.add("Hangul")
            elif block in ("DEVANAGARI",):
                scripts.add("Devanagari")
            elif block in ("HIRAGANA",):
                scripts.add("Hiragana")
            elif block in ("KATAKANA",):
                scripts.add("Katakana")
            elif block in ("THAI",):
                scripts.add("Thai")
            elif block in ("GEORGIAN",):
                scripts.add("Georgian")
            elif block in ("ARMENIAN",):
                scripts.add("Armenian")
            elif block in ("HEBREW",):
                scripts.add("Hebrew")
            elif block in ("BENGALI",):
                scripts.add("Bengali")
            elif block in ("GURMUKHI",):
                scripts.add("Gurmukhi")
            elif block in ("GUJARATI",):
                scripts.add("Gujarati")
            elif block in ("TAMIL",):
                scripts.add("Tamil")
            elif block in ("TELUGU",):
                scripts.add("Telugu")
            elif block in ("KANNADA",):
                scripts.add("Kannada")
            elif block in ("MALAYALAM",):
                scripts.add("Malayalam")
            elif block in ("MYANMAR",):
                scripts.add("Myanmar")
            elif block in ("KHMER",):
                scripts.add("Khmer")
            elif block in ("TIBETAN",):
                scripts.add("Tibetan")
            elif block in ("ETHIOPIC",):
                scripts.add("Ethiopic")
            elif block in ("GREEK",):
                scripts.add("Greek")
            else:
                scripts.add(f"Other({block})")
    return scripts


def extract_names(entity: dict) -> list[str]:
    """Extract all name strings from a Person entity."""
    names = []
    props = entity.get("properties", {})
    for key in ("name", "alias", "previousName", "weakAlias"):
        names.extend(props.get(key, []))
    return [n for n in names if n and n.strip()]


def main():
    # --- Load and filter ---
    print("Loading data...")
    persons = []
    total_entities = 0
    schema_counts = Counter()

    with open(DATA_PATH, "r") as f:
        for line in f:
            entity = json.loads(line)
            total_entities += 1
            schema_counts[entity.get("schema", "?")] += 1
            if entity.get("schema") == "Person":
                persons.append(entity)

    print(f"Total entities: {total_entities:,}")
    print(f"\nSchema distribution (top 15):")
    for schema, count in schema_counts.most_common(15):
        print(f"  {schema:30s} {count:>8,}")

    print(f"\nPerson entities: {len(persons):,}")

    # --- Name/alias counts ---
    alias_counts = []
    persons_with_multi_name = 0
    total_names = 0

    for p in persons:
        names = extract_names(p)
        n = len(names)
        alias_counts.append(n)
        total_names += n
        if n >= 2:
            persons_with_multi_name += 1

    alias_counts.sort()
    print(f"\nTotal names across all persons: {total_names:,}")
    print(f"Persons with ≥2 names (pair-eligible): {persons_with_multi_name:,} ({100*persons_with_multi_name/len(persons):.1f}%)")

    print(f"\nAlias count distribution:")
    print(f"  Min:    {alias_counts[0]}")
    print(f"  p25:    {alias_counts[len(alias_counts)//4]}")
    print(f"  Median: {alias_counts[len(alias_counts)//2]}")
    print(f"  p75:    {alias_counts[3*len(alias_counts)//4]}")
    print(f"  p95:    {alias_counts[int(len(alias_counts)*0.95)]}")
    print(f"  p99:    {alias_counts[int(len(alias_counts)*0.99)]}")
    print(f"  Max:    {alias_counts[-1]}")

    # Histogram buckets
    buckets = Counter()
    for c in alias_counts:
        if c == 1:
            buckets["1 (no pairs)"] += 1
        elif c <= 3:
            buckets["2-3"] += 1
        elif c <= 5:
            buckets["4-5"] += 1
        elif c <= 10:
            buckets["6-10"] += 1
        elif c <= 20:
            buckets["11-20"] += 1
        elif c <= 50:
            buckets["21-50"] += 1
        else:
            buckets["51+"] += 1

    print(f"\n  Bucket distribution:")
    for bucket in ["1 (no pairs)", "2-3", "4-5", "6-10", "11-20", "21-50", "51+"]:
        if bucket in buckets:
            print(f"    {bucket:15s} {buckets[bucket]:>8,}")

    # --- Script analysis ---
    print(f"\nScript analysis...")
    script_counts = Counter()  # how many names per script
    person_script_combos = Counter()  # how many persons have names in which script combos
    cross_script_persons = 0

    for p in persons:
        names = extract_names(p)
        entity_scripts = set()
        for name in names:
            scripts = detect_scripts(name)
            for s in scripts:
                script_counts[s] += 1
            entity_scripts.update(scripts)

        combo = frozenset(entity_scripts)
        person_script_combos[combo] += 1
        if len(entity_scripts) > 1:
            cross_script_persons += 1

    print(f"\nNames by script (a name may appear in multiple):")
    for script, count in script_counts.most_common(20):
        print(f"  {script:20s} {count:>8,}")

    print(f"\nPersons with names in multiple scripts: {cross_script_persons:,} ({100*cross_script_persons/len(persons):.1f}%)")

    print(f"\nTop 20 script combinations per entity:")
    for combo, count in person_script_combos.most_common(20):
        label = " + ".join(sorted(combo)) if combo else "(none)"
        print(f"  {label:45s} {count:>8,}")

    # --- Pair estimates ---
    print(f"\nPair estimates...")
    total_all_pairs = 0
    total_cross_script_pairs = 0
    total_latin_latin_pairs = 0

    for p in persons:
        names = extract_names(p)
        if len(names) < 2:
            continue

        # Compute scripts for each name
        name_scripts = [(n, detect_scripts(n)) for n in names]
        pairs = list(combinations(name_scripts, 2))
        total_all_pairs += len(pairs)

        for (n1, s1), (n2, s2) in pairs:
            has_latin_1 = "Latin" in s1
            has_latin_2 = "Latin" in s2
            non_latin_1 = s1 - {"Latin"}
            non_latin_2 = s2 - {"Latin"}

            if (has_latin_1 and non_latin_2) or (non_latin_1 and has_latin_2):
                total_cross_script_pairs += 1
            elif has_latin_1 and has_latin_2 and not non_latin_1 and not non_latin_2:
                total_latin_latin_pairs += 1

    print(f"  Total (all-pairs, all types):   {total_all_pairs:>10,}")
    print(f"  Cross-script (Latin ↔ non-Latin): {total_cross_script_pairs:>10,}")
    print(f"  Latin ↔ Latin:                    {total_latin_latin_pairs:>10,}")
    print(f"  Other (non-Latin ↔ non-Latin):    {total_all_pairs - total_cross_script_pairs - total_latin_latin_pairs:>10,}")

    # --- Top heavy-alias entities ---
    print(f"\nTop 15 entities by alias count:")
    by_alias = sorted(persons, key=lambda p: len(extract_names(p)), reverse=True)
    for p in by_alias[:15]:
        names = extract_names(p)
        scripts = set()
        for n in names:
            scripts.update(detect_scripts(n))
        eid = p.get("id", "?")
        primary = p.get("properties", {}).get("name", ["?"])[0]
        print(f"  {len(names):>3d} names | {', '.join(sorted(scripts)):30s} | {primary[:60]}")

    # --- Source list distribution ---
    print(f"\nSource dataset distribution (top 20):")
    source_counts = Counter()
    for p in persons:
        for ds in p.get("datasets", []):
            source_counts[ds] += 1
    for src, count in source_counts.most_common(20):
        print(f"  {src:45s} {count:>8,}")


if __name__ == "__main__":
    main()
