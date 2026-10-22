#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p data
for name in emotional-reactions-reddit interpretations-reddit explorations-reddit; do
    curl -L --fail "https://raw.githubusercontent.com/behavioral-data/Empathy-Mental-Health/master/dataset/$name.csv" -o "data/$name.csv"
done
