#!/usr/bin/env bash
# Run the 25k IPA subsample (~22h at 0.31 names/s with batch_size=16).
# Usage: nohup ./run_ipa_25k.sh > output/ipa_25k.log 2>&1 &
#    or: screen -S ipa ./run_ipa_25k.sh

set -euo pipefail
cd "$(dirname "$0")"

exec python3 -u run_ipa.py \
    --names-file output/ipa_subsample_25k.txt \
    --output output/ipa_25k.parquet \
    --concurrency 8 \
    -n 10 \
    --temperature 0.6
