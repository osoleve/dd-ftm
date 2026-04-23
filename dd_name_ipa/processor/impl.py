"""NeMo Data Designer processor that adds IPA transcription columns."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from data_designer.engine.processing.processors.base import Processor

from dd_name_ipa.generate import GenerationConfig, generate_batch

from .config import NameIpaProcessorConfig

logger = logging.getLogger(__name__)


class NameIpaProcessor(Processor[NameIpaProcessorConfig]):
    """Generate IPA columns for configured name fields after dataset generation."""

    def process_after_generation(self, data: pd.DataFrame) -> pd.DataFrame:
        output = data.copy()
        generation_config = GenerationConfig(
            api_base=self.config.api_base,
            model=self.config.model,
            n=self.config.n,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            concurrent_requests=self.config.concurrent_requests,
            disable_thinking=self.config.disable_thinking,
        )

        for column in self.config.name_columns:
            if column not in output.columns:
                raise ValueError(f"Configured name column {column!r} is not present in the dataset")

            ipa_column = f"{column}{self.config.output_suffix}"
            confidence_column = f"{column}{self.config.confidence_suffix}"
            collisions = [name for name in (ipa_column, confidence_column) if name in output.columns]
            if collisions:
                raise ValueError(f"Refusing to overwrite existing columns: {', '.join(collisions)}")

            normalized_names = output[column].map(_normalize_name)
            unique_names = list(dict.fromkeys(name for name in normalized_names.tolist() if name is not None))

            if not unique_names:
                output[ipa_column] = pd.NA
                output[confidence_column] = pd.NA
                continue

            logger.info("Generating IPA for %d unique values in column %s", len(unique_names), column)
            results = generate_batch(unique_names, generation_config)
            ipa_map = {result.name: (result.ipa or pd.NA) for result in results}
            confidence_map = {
                result.name: (result.confidence if result.ipa else pd.NA)
                for result in results
            }

            output[ipa_column] = normalized_names.map(ipa_map)
            output[confidence_column] = normalized_names.map(confidence_map)

        return output


def _normalize_name(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = value.strip() if isinstance(value, str) else str(value).strip()
    return text or None
