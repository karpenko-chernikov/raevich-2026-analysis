#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mplconfig}"
mkdir -p "$MPLCONFIGDIR" figures data
python src/fetch_results.py --out data
python src/prepare_data.py
python src/plot_analysis.py
