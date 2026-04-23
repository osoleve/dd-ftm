"""Configuration for the NeMo Data Designer name-to-IPA processor plugin."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from data_designer.config.base import ProcessorConfig


class NameIpaProcessorConfig(ProcessorConfig):
    """Append IPA transcription columns for one or more name columns."""

    processor_type: Literal["name-ipa"] = "name-ipa"
    name_columns: list[str] = Field(
        default_factory=lambda: ["name"],
        min_length=1,
        description="Columns whose values should be transcribed to IPA.",
    )
    output_suffix: str = Field(default="_ipa", description="Suffix for IPA output columns.")
    confidence_suffix: str = Field(
        default="_ipa_confidence",
        description="Suffix for confidence output columns.",
    )
    api_base: str = Field(default="http://localhost:8355/v1", description="OpenAI-compatible API base URL.")
    model: str = Field(default="nvidia/Qwen3-235B-A22B-FP4", description="Model name to call.")
    n: int = Field(default=10, ge=1, description="Number of candidates to sample per name.")
    temperature: float = Field(default=0.6, ge=0.0, description="Sampling temperature.")
    max_tokens: int = Field(default=128, ge=1, description="Maximum tokens per completion.")
    concurrent_requests: int = Field(default=8, ge=1, description="Maximum concurrent API calls.")
    disable_thinking: bool = Field(default=True, description="Disable model thinking blocks when supported.")
