"""Name-to-IPA transcription via best-of-N LLM generation."""

from .generate import GenerationConfig, IPAResult, generate_batch
from .prompt import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT, build_messages

__all__ = [
    "FEW_SHOT_EXAMPLES",
    "GenerationConfig",
    "IPAResult",
    "SYSTEM_PROMPT",
    "build_messages",
    "generate_batch",
]
