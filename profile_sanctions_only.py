"""Profile OpenSanctions data filtered to sanctions-list sources only."""

import json
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations

DATA_PATH = "data/targets.nested.json"

# Actual sanctions list datasets (not PEP, not Wikidata, not wanted lists)
# Identified by naming convention and known sources
SANCTIONS_KEYWORDS = {
    "sanctions", "sdn", "ofac", "consolidated", "hmt", "seco",
    "dfat", "mfat", "nbctf", "nsdc", "nacp",
    "terror", "freeze", "designated", "debarment",
}

# Explicit includes for known sanctions datasets that might not match keywords
SANCTIONS_EXPLICIT = {
    "us_ofac_sdn", "us_ofac_cons", "un_sc_sanctions",
    "eu_fsf",  "eu_cor_members", "gb_hmt_sanctions",
    "ch_seco_sanctions", "au_dfat_sanctions", "nz_mfat_sanctions",
    "ca_dfatd_sema_sanctions", "jp_mof_sanctions",
    "ua_nsdc_sanctions", "ua_nacp_sanctions",
    "il_mod_terrorists", "za_fic_sanctions",
    "qa_nctc_sanctions", "kz_afmrk_sanctions",
    "kg_fiu_national", "ru_nsd_isin",
    "us_trade_csl", "us_bis_denied",
}

# Explicit excludes
NON_SANCTIONS = {
    "wikidata", "wd_categories", "wd_peps", "wd_oligarchs",
    "ann_pep_positions", "ann_graph_topics",
    "everypolitician", "br_pep", "fr_maires",
    "es_mayors_councillors", "lt_pep_declarations",
    "pl_wanted", "sg_gov_dir", "co_funcion_publica",
    "ge_declarations", "ng_chipper_peps",
    "us_sam_exclusions", "us_hhs_exclusions", "us_ca_med_exclusions",
}


def is_sanctions_dataset(ds_name: str) -> bool:
    if ds_name in SANCTIONS_EXPLICIT:
        return True
    if ds_name in NON_SANCTIONS:
        return False
    lower = ds_name.lower()
    return any(kw in lower for kw in SANCTIONS_KEYWORDS)


def detect_scripts(name: str) -> set[str]:
    scripts = set()
    for ch in name:
        if ch.isalpha():
            uname = unicodedata.name(ch, "")
            if not uname:
                continue
            block = uname.split()[0]
            if block == "LATIN":
                scripts.add("Latin")
            elif block == "CYRILLIC":
                scripts.add("Cyrillic")
            elif block in ("ARABIC", "ARABIC-INDIC"):
                scripts.add("Arabic")
            elif block in ("CJK", "KANGXI", "IDEOGRAPHIC"):
                scripts.add("CJK")
            elif block == "HANGUL":
                scripts.add("Hangul")
            elif block == "DEVANAGARI":
                scripts.add("Devanagari")
            elif block in ("HIRAGANA", "KATAKANA"):
                scripts.add(block.capitalize())
            elif block in ("THAI", "GEORGIAN", "ARMENIAN", "HEBREW",
                           "BENGALI", "GURMUKHI", "GUJARATI", "TAMIL",
                           "TELUGU", "KANNADA", "MALAYALAM", "MYANMAR",
                           "KHMER", "TIBETAN", "ETHIOPIC", "GREEK"):
                scripts.add(block.capitalize())
            else:
                scripts.add(f"Other({block})")
    return scripts


def extract_names(entity: dict) -> list[str]:
    names = []
    props = entity.get("properties", {})
    for key in ("name", "alias", "previousName", "weakAlias"):
        names.extend(props.get(key, []))
    return [n for n in names if n and n.strip()]


def main():
    # --- Pass 1: discover all datasets ---
    print("Pass 1: discovering datasets...")
    all_datasets = Counter()
    with open(DATA_PATH, "r") as f:
        for line in f:
            entity = json.loads(line)
            if entity.get("schema") == "Person":
                for ds in entity.get("datasets", []):
                    all_datasets[ds] += 1

    sanctions_ds = set()
    non_sanctions_ds = set()
    unknown_ds = set()

    for ds in all_datasets:
        if is_sanctions_dataset(ds):
            sanctions_ds.add(ds)
        elif ds in NON_SANCTIONS:
            non_sanctions_ds.add(ds)
        else:
            unknown_ds.add(ds)

    print(f"\nClassified sanctions datasets ({len(sanctions_ds)}):")
    for ds in sorted(sanctions_ds):
        print(f"  ✓ {ds:45s} {all_datasets[ds]:>8,} persons")

    if unknown_ds:
        print(f"\nUnclassified datasets ({len(unknown_ds)}) — NOT included:")
        for ds in sorted(unknown_ds):
            print(f"  ? {ds:45s} {all_datasets[ds]:>8,} persons")

    print(f"\nExcluded non-sanctions datasets ({len(non_sanctions_ds)}):")
    for ds in sorted(non_sanctions_ds):
        print(f"  ✗ {ds:45s} {all_datasets[ds]:>8,} persons")

    # --- Pass 2: filter and profile ---
    print(f"\n{'='*60}")
    print("Pass 2: profiling sanctions-only persons...")

    persons = []
    with open(DATA_PATH, "r") as f:
        for line in f:
            entity = json.loads(line)
            if entity.get("schema") != "Person":
                continue
            entity_ds = set(entity.get("datasets", []))
            if entity_ds & sanctions_ds:
                persons.append(entity)

    print(f"\nSanctions-list persons: {len(persons):,}")

    # --- Name/alias counts ---
    alias_counts = []
    pair_eligible = 0
    total_names = 0

    for p in persons:
        names = extract_names(p)
        n = len(names)
        alias_counts.append(n)
        total_names += n
        if n >= 2:
            pair_eligible += 1

    alias_counts.sort()
    print(f"Total names: {total_names:,}")
    print(f"Pair-eligible (≥2 names): {pair_eligible:,} ({100*pair_eligible/len(persons):.1f}%)")

    print(f"\nAlias count distribution:")
    print(f"  Min:    {alias_counts[0]}")
    print(f"  p25:    {alias_counts[len(alias_counts)//4]}")
    print(f"  Median: {alias_counts[len(alias_counts)//2]}")
    print(f"  p75:    {alias_counts[3*len(alias_counts)//4]}")
    print(f"  p95:    {alias_counts[int(len(alias_counts)*0.95)]}")
    print(f"  p99:    {alias_counts[int(len(alias_counts)*0.99)]}")
    print(f"  Max:    {alias_counts[-1]}")

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
    script_counts = Counter()
    cross_script_persons = 0

    for p in persons:
        names = extract_names(p)
        entity_scripts = set()
        for name in names:
            scripts = detect_scripts(name)
            for s in scripts:
                script_counts[s] += 1
            entity_scripts.update(scripts)
        if len(entity_scripts) > 1:
            cross_script_persons += 1

    print(f"\nNames by script:")
    for script, count in script_counts.most_common(20):
        print(f"  {script:20s} {count:>8,}")

    print(f"\nPersons with multi-script names: {cross_script_persons:,} ({100*cross_script_persons/len(persons):.1f}%)")

    # --- Pair counts ---
    print(f"\nPair counts...")
    total_pairs = 0
    cross_script_pairs = 0
    latin_latin_pairs = 0

    for p in persons:
        names = extract_names(p)
        if len(names) < 2:
            continue
        name_scripts = [(n, detect_scripts(n)) for n in names]
        pairs = list(combinations(name_scripts, 2))
        total_pairs += len(pairs)

        for (n1, s1), (n2, s2) in pairs:
            l1, l2 = "Latin" in s1, "Latin" in s2
            nl1, nl2 = s1 - {"Latin"}, s2 - {"Latin"}
            if (l1 and nl2) or (nl1 and l2):
                cross_script_pairs += 1
            elif l1 and l2 and not nl1 and not nl2:
                latin_latin_pairs += 1

    print(f"  Total all-pairs:                  {total_pairs:>10,}")
    print(f"  Cross-script (Latin ↔ non-Latin):  {cross_script_pairs:>10,}")
    print(f"  Latin ↔ Latin:                     {latin_latin_pairs:>10,}")
    print(f"  Non-Latin ↔ non-Latin:             {total_pairs - cross_script_pairs - latin_latin_pairs:>10,}")

    # --- Top entities ---
    print(f"\nTop 15 entities by alias count:")
    by_alias = sorted(persons, key=lambda p: len(extract_names(p)), reverse=True)
    for p in by_alias[:15]:
        names = extract_names(p)
        scripts = set()
        for n in names:
            scripts.update(detect_scripts(n))
        primary = p.get("properties", {}).get("name", ["?"])[0]
        print(f"  {len(names):>3d} names | {', '.join(sorted(scripts)):30s} | {primary[:60]}")

    # --- Sample some cross-script pairs ---
    print(f"\n{'='*60}")
    print("Sample cross-script pairs (first 30 from distinct entities):")
    seen = 0
    for p in persons:
        if seen >= 30:
            break
        names = extract_names(p)
        if len(names) < 2:
            continue
        name_scripts = [(n, detect_scripts(n)) for n in names]
        for (n1, s1), (n2, s2) in combinations(name_scripts, 2):
            if seen >= 30:
                break
            l1, l2 = "Latin" in s1, "Latin" in s2
            nl1, nl2 = s1 - {"Latin"}, s2 - {"Latin"}
            if (l1 and nl2) or (nl1 and l2):
                s1_label = "+".join(sorted(s1))
                s2_label = "+".join(sorted(s2))
                print(f"  [{s1_label:>12s}] {n1[:40]:40s} ↔ [{s2_label:>12s}] {n2[:40]}")
                seen += 1
                break  # one pair per entity


if __name__ == "__main__":
    main()
