# Contrastive Phonetic Name Pairs Dataset — Pipeline Design

## Overview

A large-scale, cross-script phonetic name pair corpus bootstrapped from public sanctions data, expanded synthetically via local LLM generation and judging. The dataset enables contrastive pretraining experiments on phonological structure in proper names, exploiting the fact that names are semantically ungrounded — any learned structure must be phonological.

## Motivation

Proper names are ideal for studying learned phonological representations because they carry no semantic content out of context. A model trained contrastively on name pairs cannot rely on distributional semantics; similarity can only reflect phonological structure. This makes names a natural control for isolating syllable segmentation, onset structure, stress patterns, and other prosodic features in learned representations.

Sanctions list aliases are expert-curated phonetic equivalence classes produced at enormous cost by government analysts. An entry on the OFAC SDN list may include the original script form, a UN-standard romanization, a passport romanization, and several known media spellings. These are effectively gold-standard cross-script phonetic pairs, and they are public domain.

## Pipeline Stages

### Stage 0: Source Acquisition

Download the OpenSanctions `default` collection in FollowTheMoney nested JSON format:

```
https://data.opensanctions.org/datasets/latest/default/targets.nested.json
```

This single file provides all individuals across OFAC SDN, UN Consolidated, EU Consolidated, UK HMT, and dozens of additional lists, pre-deduplicated, with aliases nested inside each entity record. The FollowTheMoney schema distinguishes entity types natively, so filtering for individuals is a schema check (`schema: Person`), not a classification task.

**Supplementary sources** (for coverage of underrepresented phonological systems): Olympic athlete rosters, UN personnel directories, academic author databases (DBLP, ORCID).

**Output:** Filtered person records with fields: entity ID, primary name, alias list, script(s), source list(s).

**Licensing:** OpenSanctions data is CC 4.0 Attribution-NonCommercial for non-commercial use. Attribution in the datasheet is required.

### Stage 1: Pair Extraction and Script Detection

For each individual, generate all (primary, alias) and (alias, alias) pairs. Tag each name with detected script(s) via Unicode block analysis.

Retain pairs where at least one name is in Latin script and the other is non-Latin, OR both are Latin but sourced from different romanization standards (e.g., UN transliteration vs. passport spelling). FollowTheMoney properties may provide script metadata directly, reducing the need for independent detection.

**Output:** Candidate pair table: `(pair_id, name_a, script_a, name_b, script_b, source, pair_type)`.

**Expected yield:** ~30–50k candidate pairs from sanctions data alone.

### Stage 2: Pair Classification (Local LLM Judge)

Not all alias pairs are phonetic. Rather than discard non-phonetic pairs, we label all pairs by relationship type. Non-phonetic pairs (semantic aliases, cover identities, married names) become labeled negatives, useful both as hard negatives in contrastive training and as a standalone phonetic/non-phonetic classification benchmark.

**Model:** Qwen3-70B on dual-Spark configuration, temperature 0.

**Prompt design:** Present the pair with script labels. Classify as: `phonetic_transliteration`, `phonetic_variant` (same name, different romanization convention), `semantic_alias` (different name entirely), or `ambiguous`. Return a confidence score, a phonetic similarity score on a continuous scale, and a one-line rationale.

**Confidence handling:** Pairs classified at confidence ≥ 0.8 are accepted with their label. Pairs marked `ambiguous` or falling below the confidence threshold go to a second round with a rephrased prompt. If the second round also returns `ambiguous`, the pair is retained with label `ambiguous` — downstream users can filter or include these at their discretion.

**Diagnostic feedback loop:** Rationales from disagreement cases serve as diagnostics for improving syllabification heuristics downstream. The judge's mistakes become signal about where the Sonority Sequencing Principle breaks down for specific language families.

**Output:** Fully labeled pair table with columns: `pair_type` (`phonetic_transliteration`, `phonetic_variant`, `semantic_alias`, `ambiguous`), judge confidence, phonetic similarity score, and rationale.

### Stage 3: Latin-Side Syllabification

Syllabify all Latin-script names across all pair types using NLTK's `SyllableTokenizer`, which implements sonority sequencing with the Maximal Onset Principle. Semantic alias pairs get syllabified too — their syllable-level distance profiles are useful metadata and provide a natural sanity check (semantic aliases should show high syllable edit distance, and cases where they don't may indicate judge misclassification).

**Process:**

1. Normalize Latin names: strip diacritics to ASCII for syllabification, retain originals as metadata.
2. Run NLTK SSP-based syllabification on normalized forms.
3. Store syllabified form as a list of syllable strings.

**Known limitations:** SSP will mishandle certain clusters (Georgian-origin names with complex onsets, word-initial /str/ in English, languages that systematically violate SSP). Flag names where syllabification produces a syllable with no vowel nucleus for review. Accept imperfection for v1 — the metric space construction is tolerant of occasional errors.

**Output:** Syllabified pair table: `(pair_id, name_a, syllables_a, name_b, script_b, ...)`.

### Stage 4: Metric Space Construction

Build a metric tree over syllabified Latin forms using syllable-level edit distance.

**Distance function:** Levenshtein distance computed on syllable sequences, not character sequences. Each syllable is an atomic unit. Optionally weight substitution cost by sonority distance between syllable nuclei for a more phonologically informed metric.

**Tree type:** Vantage-point tree or BK-tree indexed on the syllable Levenshtein distance.

**Output:** Serialized metric tree and lexicographically sorted index over syllabified forms.

### Stage 5: Hard Negative Mining

For each phonetic pair `(a, b)`, hard negatives come from two sources:

1. **Metric tree neighbors:** Query the tree for names within syllable edit distance 1–2 of `a` that are not phonetic variants of `a`. These are phonological near-misses — names sharing syllable template but differing in segments, mirroring the concept of minimal pairs in distinctive feature theory.
2. **Semantic alias pairs:** Pairs already labeled `semantic_alias` in Stage 2 are natural hard negatives when one member is phonologically close to an anchor. These are especially valuable because they represent cases where surface similarity is coincidental rather than phonetic.

**Output:** Triplets `(anchor, positive, hard_negative)` for triplet loss setups, and quadruplets `(a1, a2, b1, b2)` for pairs-of-pairs relational contrastive setups. Each negative is tagged with its source (`metric_neighbor` or `semantic_alias`).

### Stage 6: Synthetic Expansion

Sort the phonetic subset of the corpus (pairs labeled `phonetic_transliteration` or `phonetic_variant`) lexicographically by syllabified form. Identify sparse regions: consecutive entries where syllable edit distance exceeds a tunable threshold (default: 3).

**Generation:**

1. For each gap, prompt the local LLM to propose K candidate names that fall between the bounding entries in phonological space.
2. Prompt specifies: the bounding names, their syllable structures, the apparent source language/phonological system, and a request for phonotactically valid interpolations.
3. Candidates go through multi-round QA (3 rounds, majority vote, temperature 0):
   - Round 1: Is this a plausible proper name in any natural language?
   - Round 2: Does it phonologically fall between the bounding entries?
   - Round 3: Is it phonotactically valid for its apparent source language?
4. Accepted candidates are inserted into the sorted index; the metric tree is rebuilt incrementally.

**Iteration:** Each round of insertion creates new gaps. Run until marginal acceptance rate drops below 20% of proposals accepted.

**Output:** Expanded corpus with provenance: `source: opensanctions` vs. `source: synthetic, rounds_passed: 3/3`.

### Stage 7: Feature Annotation

For each name in the final corpus, compute and store:

- Syllable count and syllable weight profile (light/heavy based on coda presence)
- Onset complexity per syllable (consonant count in onset)
- Estimated stress pattern (penultimate default, adjustable per detected source language)
- Phonological feature vectors per segment (±voice, ±continuant, ±sonorant, place, height, backness, rounding)
- Language family tag (best guess from script + phonotactic profile, LLM-assisted)

### Stage 8: Dataset Packaging

**Format:** Parquet.

**Schema:**

- `pair_id`, `anchor_name`, `anchor_syllabified`, `anchor_script`, `positive_name`, `positive_script`, `pair_type` (`phonetic_transliteration` | `phonetic_variant` | `semantic_alias` | `ambiguous`), `source`, `judge_confidence`, `phonetic_similarity_score`, `judge_rationale`
- Hard negative columns: `negative_name`, `negative_syllabified`, `syllable_edit_distance`, `negative_source` (`metric_neighbor` | `semantic_alias`)
- Feature annotation columns per name

**Splits:** None predefined. Downstream users split by language family or phonological feature to test generalization.

**Datasheet metadata:** OpenSanctions collection version and download date, judge model identity and version, NLTK version and syllabifier config, acceptance thresholds, expansion iteration count, coverage statistics by script and estimated language family.

**Release:** HuggingFace Datasets with full datasheet. DOI via Zenodo.

## Hardware Allocation

| Stage | Compute | Location |
|---|---|---|
| 0–1 | CPU-bound download/parsing | Single Spark, CPU only |
| 2 | GPU — LLM inference | Dual Spark (70B judge) |
| 3 | CPU-bound syllabification | Single Spark, CPU only |
| 4–5 | CPU-bound tree ops | Single Spark, CPU only |
| 6 | GPU — LLM generation + judging | Dual Spark (70B judge) |
| 7 | CPU + minor LLM for language ID | Single Spark |
| 8 | CPU-bound serialization | Single Spark |

CPU-bound stages (0–1, 3–5) can overlap with GPU-bound work. The long pole is Stage 6 (iterative synthetic expansion), bounded by judge throughput. Stage 2 validation is a long unattended batch run suitable for overnight execution.

## Open Questions

- **Sonority scale granularity:** Coarse (obstruent < nasal < liquid < glide < vowel) or fine-grained Parker-style? Coarser generalizes better cross-linguistically; finer improves syllabification for complex onsets.
- **Expansion termination:** Fixed iteration count vs. marginal acceptance rate vs. target corpus size?
- **Non-Latin anchor pairs:** Worth adding a parallel track for Cyrillic↔Arabic or Devanagari↔CJK pairs? This requires a different syllabification strategy or segment-level IPA via eSpeak-ng, bypassing Latin-anchored SSP entirely.
- **Phonaesthetic confounds:** Names are not fully ungrounded — languages exhibit statistical tendencies in name phonology by gender, era, and social context. Control for this within-language/within-era in v1, then explore as a feature in v2.
- **Versioning:** Pin to a single OpenSanctions snapshot for v1. Consider an update mechanism for future versions tied to their delta files.

## Downstream Experiments

The dataset is designed to support three contrastive pretraining configurations on a diffusion model:

1. **Hard negative pairs + contrastive loss:** Tests phonological neighborhood discrimination. Semantic alias pairs provide additional naturally occurring negatives.
2. **Triplet loss (anchor, positive, hard negative):** Tests whether syllable structure and segmental features occupy distinct regions of the learned space.
3. **Pairs-of-pairs relational loss:** Tests whether the model discovers analogical phonological structure (e.g., voicing alternations, vowel shifts) — effectively asking if it can rediscover distinctive features.

The denoising trajectory of the diffusion model provides a natural probe: if syllable boundaries crystallize at a different noise level than segmental features, this constitutes evidence for hierarchical phonological representation emerging without supervision.

Additionally, the full label set (`phonetic_transliteration`, `phonetic_variant`, `semantic_alias`, `ambiguous`) makes the dataset usable as a standalone benchmark for phonetic pair classification — a task directly relevant to sanctions screening and entity resolution systems.