"""Core extraction and pairing logic for FtM phonetic name pairs."""

from .datasets import DEFAULT_SANCTIONS_DATASETS, OPENSANCTIONS_URL
from .extract import EntityRecord, ExtractionConfig, NameRecord, extract_all, stream_entities
from .pairs import NamePair, PairConfig, generate_pairs
from .scripts import detect_scripts, dominant_script

__all__ = [
    "DEFAULT_SANCTIONS_DATASETS",
    "OPENSANCTIONS_URL",
    "EntityRecord",
    "ExtractionConfig",
    "NameRecord",
    "NamePair",
    "PairConfig",
    "detect_scripts",
    "dominant_script",
    "extract_all",
    "generate_pairs",
    "stream_entities",
]
