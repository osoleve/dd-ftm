"""Unicode script detection for name classification."""

from __future__ import annotations

import unicodedata
from collections import Counter

# Maps the first word of unicodedata.name() to a human-readable script label.
# Covers all scripts observed in OpenSanctions Person entities plus common edge cases.
_SCRIPT_MAP: dict[str, str] = {
    "LATIN": "Latin",
    "CYRILLIC": "Cyrillic",
    "ARABIC": "Arabic",
    "ARABIC-INDIC": "Arabic",
    "CJK": "CJK",
    "KANGXI": "CJK",
    "IDEOGRAPHIC": "CJK",
    "HANGUL": "Hangul",
    "DEVANAGARI": "Devanagari",
    "HIRAGANA": "Hiragana",
    "KATAKANA": "Katakana",
    "KATAKANA-HIRAGANA": "Katakana",
    "FULLWIDTH": "CJK",
    "HALFWIDTH": "CJK",
    "THAI": "Thai",
    "GEORGIAN": "Georgian",
    "ARMENIAN": "Armenian",
    "HEBREW": "Hebrew",
    "BENGALI": "Bengali",
    "GURMUKHI": "Gurmukhi",
    "GUJARATI": "Gujarati",
    "TAMIL": "Tamil",
    "TELUGU": "Telugu",
    "KANNADA": "Kannada",
    "MALAYALAM": "Malayalam",
    "MYANMAR": "Myanmar",
    "KHMER": "Khmer",
    "TIBETAN": "Tibetan",
    "ETHIOPIC": "Ethiopic",
    "GREEK": "Greek",
}

# Per-character memoization: char -> script label or None (non-alpha / unmapped).
# unicodedata.name() is expensive (~10k unique chars across 240k names).
_char_cache: dict[str, str | None] = {}


def _classify_char(ch: str) -> str | None:
    """Return script label for an alphabetic character, or None."""
    cached = _char_cache.get(ch)
    if cached is not None:
        return cached
    # Sentinel: distinguish "not cached" from "cached as None"
    if ch in _char_cache:
        return None

    result = None
    if ch.isalpha():
        uname = unicodedata.name(ch, "")
        if uname:
            block = uname.split()[0]
            result = _SCRIPT_MAP.get(block)
            if result is None:
                result = f"Other({block})"
    _char_cache[ch] = result
    return result


def detect_scripts(text: str) -> frozenset[str]:
    """Return all Unicode scripts present in alphabetic characters of text."""
    scripts: set[str] = set()
    for ch in text:
        label = _classify_char(ch)
        if label is not None:
            scripts.add(label)
    return frozenset(scripts)


def dominant_script(scripts: frozenset[str]) -> str:
    """Collapse a set of scripts to a single label for pair classification.

    Rules:
    - Empty → "Unknown"
    - Single script → return it
    - Latin + exactly one non-Latin → return the non-Latin
      (handles stray Latin chars in otherwise non-Latin names)
    - Multiple non-Latin → return whichever appears (deterministic via sort)
    """
    if not scripts:
        return "Unknown"
    if len(scripts) == 1:
        return next(iter(scripts))
    non_latin = scripts - {"Latin"}
    if non_latin:
        # If there's exactly one non-Latin script, prefer it
        # (even if Latin is also present — handles mixed-script data entry)
        if len(non_latin) == 1:
            return next(iter(non_latin))
        # Multiple non-Latin: pick alphabetically first for determinism
        return sorted(non_latin)[0]
    # Only Latin variants? Shouldn't happen with current map, but be safe
    return "Latin"


def dominant_script_weighted(text: str) -> str:
    """Determine dominant script by character frequency in text.

    More accurate than dominant_script(detect_scripts(text)) for names
    with mixed-script characters, since it weights by count.
    """
    counts: Counter[str] = Counter()
    for ch in text:
        label = _classify_char(ch)
        if label is not None:
            counts[label] += 1
    if not counts:
        return "Unknown"
    # Most frequent script wins
    return counts.most_common(1)[0][0]
