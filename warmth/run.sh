#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/.deps${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$PWD/warmth/.matplotlib"
exec /home/agent/venv/bin/python warmth/pipeline.py "$@"
