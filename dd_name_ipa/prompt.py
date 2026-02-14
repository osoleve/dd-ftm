"""Few-shot prompt construction for name-to-IPA transcription."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are an expert phonetician and linguist specializing in proper name \
transcription. Given a personal name, provide its broad IPA transcription.

Rules:
- Infer the most likely source language from orthography and script.
- Use broad (phonemic) transcription in /slashes/.
- For multi-word names, separate words with spaces inside the slashes.
- Preserve phonological features of the source language (tones, \
pharyngeals, retroflex, gemination, etc.).
- Return ONLY the IPA transcription. No explanation, no alternatives.\
"""

# Diverse, high-quality few-shot examples covering the major scripts and
# language families in the OpenSanctions data. Each chosen to demonstrate
# a specific phonological or orthographic challenge.
FEW_SHOT_EXAMPLES: list[tuple[str, str]] = [
    # --- Cyrillic (the largest non-Latin block in sanctions data) ---
    # Russian: palatalization, vowel reduction, voiced/voiceless assimilation
    ("Владимир Путин", "/vlɐˈdʲimʲɪr ˈputʲɪn/"),
    # Ukrainian: distinct from Russian — /ɦ/ not /ɡ/, /ɪ/ patterns, soft consonants
    ("Олександр Костенко", "/ɔlɛkˈsɑndr kɔsˈtɛnkɔ/"),
    # Belarusian: dzekanne/tsekanne, distinct vowel system
    ("Аляксандр Лукашэнка", "/alʲakˈsandr lukaˈʂɛnka/"),

    # --- Arabic (second largest non-Latin) ---
    # Pharyngeals, emphatics, long vowels, sun-letter assimilation
    ("محمد بن سلمان", "/muˈħammad bin salˈmaːn/"),
    # Ayin, definite article assimilation
    ("عبد الرحمن", "/ʕabd ar.raħˈmaːn/"),

    # --- CJK ---
    # Mandarin: aspirated/unaspirated distinction, tonal (not marked in broad IPA)
    ("习近平", "/ɕi tɕin pʰiŋ/"),
    # Japanese (Kanji): pitch accent, mora-timed, flap /ɾ/
    ("山本太郎", "/jamamato taɾoː/"),

    # --- Hangul ---
    # Korean: tensification, nasalization, three-way laryngeal contrast
    ("김정은", "/kim tɕɔŋ ɯn/"),

    # --- Devanagari ---
    # Hindi: retroflex consonants, breathy voiced stops, schwa deletion
    ("नरेन्द्र मोदी", "/nəˈɾeːndɾə ˈmoːdiː/"),

    # --- Hebrew ---
    # Stress patterns, pharyngeals reduced in Modern Hebrew
    ("בנימין נתניהו", "/binjaˈmin netanˈjahu/"),

    # --- Georgian ---
    # Ejectives, harmonic clusters
    ("ბიძინა ივანიშვილი", "/biˈd͡zina ivaniʃˈvili/"),

    # --- Armenian ---
    # Aspirated/ejective stops, Western vs Eastern pronunciation
    ("Նիկոլ Փաշինյան", "/niˈkɔl pʰaʃinˈjan/"),

    # --- Thai ---
    # Aspiration contrast, tonal (not marked in broad IPA for names)
    ("ทักษิณ ชินวัตร", "/tʰaksin tɕʰinnaˈwat/"),

    # --- Latin script, various source languages ---
    # French: nasalized vowels, silent consonants, uvular /ʁ/
    ("Jean-Pierre Dupont", "/ʒɑ̃ pjɛʁ dyˈpɔ̃/"),
    # German: uvular /ʁ/, front rounded vowels, final devoicing
    ("Friedrich Müller", "/ˈfʁiːdʁɪç ˈmʏlɐ/"),
    # Spanish: fricative allophones, tap /ɾ/, trill /r/
    ("José García López", "/xoˈse ɣaɾˈθi.a ˈlopeθ/"),
    # Polish: palatal series, specific sibilant inventory
    ("Wojciech Kowalski", "/ˈvɔjtɕɛx kɔˈvalskʲi/"),
    # English: stress-timed, rhotic
    ("Robert Johnson", "/ˈɹɑbɚt ˈdʒɑnsən/"),
    # Romanized Arabic name in Latin script — model must infer Arabic phonology
    ("Abdulaziz al-Rashid", "/ʕabdulʕaˈziːz ar.raˈʃiːd/"),
    # Romanized Chinese name — model must infer Mandarin phonology
    ("Zhang Wei", "/tʂɑŋ weɪ/"),
]


def build_messages(name: str) -> list[dict[str, str]]:
    """Build the chat messages for a name-to-IPA request."""
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example_name, example_ipa in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": example_name})
        messages.append({"role": "assistant", "content": example_ipa})
    messages.append({"role": "user", "content": name})
    return messages
