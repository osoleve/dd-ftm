# dd-ftm

Utilities for building name-focused datasets from OpenSanctions FollowTheMoney data, plus an installable NeMo Data Designer processor plugin for LLM-based name-to-IPA transcription.

## What's in this repo

- `dd_ftm`: extraction and pairing helpers for OpenSanctions FollowTheMoney person records
- `dd_name_ipa`: best-of-N IPA generation against an OpenAI-compatible API
- `dd_name_ipa.processor`: a NeMo Data Designer processor plugin that appends IPA and confidence columns to generated datasets

## NeMo Data Designer plugin status

This repository now follows the current Data Designer plugin requirements:

- packaged with a `pyproject.toml`
- plugin discovery via the `data_designer.plugins` entry-point group
- plugin implementation split into `config.py`, `impl.py`, and `plugin.py`
- compatible with `data-designer>=0.5.7`

The exposed plugin entry point is `name-ipa`.

## Installation

For local development and plugin discovery:

```bash
uv pip install -e .
```

Or with pip:

```bash
pip install -e .
```

## Using the Data Designer plugin

```python
from dd_name_ipa.processor import NameIpaProcessorConfig

builder.add_processor(
    NameIpaProcessorConfig(
        name="name_ipa",
        name_columns=["name", "alias"],
        api_base="http://localhost:8355/v1",
        model="nvidia/Qwen3-235B-A22B-FP4",
        n=10,
        temperature=0.6,
        concurrent_requests=8,
    )
)
```

The processor runs after generation and adds two columns for each configured source column:

- `<column>_ipa`
- `<column>_ipa_confidence`

## Library usage

### Extract FollowTheMoney name pairs

```python
from dd_ftm import ExtractionConfig, PairConfig, generate_pairs, stream_entities

entities = stream_entities("data/targets.nested.json", ExtractionConfig())
pairs = list(generate_pairs(entities, PairConfig(per_entity_cap=100)))
```

### Generate IPA directly

```python
from dd_name_ipa import GenerationConfig, generate_batch

results = generate_batch(
    ["Vladimir Putin", "محمد بن سلمان"],
    GenerationConfig(api_base="http://localhost:8355/v1"),
)
```

## Included scripts

- `python /absolute/path/to/run_extract.py --help`
- `python /absolute/path/to/run_ipa.py --help`

## Data and licensing notes

- Code in this repository is MIT licensed.
- OpenSanctions data is licensed separately under CC BY-NC 4.0.
- Commercial use of OpenSanctions data or derived outputs may require a separate OpenSanctions license.

See `sanctions_pipeline_overview.md` for the broader dataset design context.
