#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
if [ -n "${PYTHON_BIN:-}" ]; then
    warmth_python="$PYTHON_BIN"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    warmth_python="$VIRTUAL_ENV/bin/python"
elif [ -x "$PWD/.venv/bin/python" ]; then
    warmth_python="$PWD/.venv/bin/python"
elif [ -x /home/agent/venv/bin/python ]; then
    warmth_python=/home/agent/venv/bin/python
else
    warmth_python=python3
fi
# Workspace-local dependencies belong to the container, not a copied Mac venv.
if [ "$warmth_python" = /home/agent/venv/bin/python ]; then
    export PYTHONPATH="$PWD/.deps${PYTHONPATH:+:$PYTHONPATH}"
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/warmth/.matplotlib}"
exec "$warmth_python" warmth/matched.py "$@"
