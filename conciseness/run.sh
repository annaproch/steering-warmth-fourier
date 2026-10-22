#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/.deps${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$PWD/conciseness/cache/matplotlib"
exec /home/agent/venv/bin/python conciseness/pilot.py "$@"
