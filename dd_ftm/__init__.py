"""FtM phonetic name pair extraction module."""

from .core import (
    DEFAULT_SANCTIONS_DATASETS,
    OPENSANCTIONS_URL,
    EntityRecord,
    ExtractionConfig,
    NamePair,
    NameRecord,
    PairConfig,
    detect_scripts,
    dominant_script,
    extract_all,
    generate_pairs,
    stream_entities,
)

__all__ = [
    "DEFAULT_SANCTIONS_DATASETS",
    "OPENSANCTIONS_URL",
    "EntityRecord",
    "ExtractionConfig",
    "NamePair",
    "NameRecord",
    "PairConfig",
    "detect_scripts",
    "dominant_script",
    "extract_all",
    "generate_pairs",
    "stream_entities",
]
