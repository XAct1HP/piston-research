#!/usr/bin/env bash

# See run.bat: stale bytecode in a synced folder is
# indistinguishable from a folder that never updated.
export PYTHONDONTWRITEBYTECODE=1
# No arguments: start the browser front end.
# Any arguments: pass them straight through to the command line tool.
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for candidate in python3 python "py -3"; do
  if $candidate --version >/dev/null 2>&1; then PY="$candidate"; break; fi
done

if [ -z "$PY" ]; then
  echo
  echo "  No Python interpreter found. Install Python 3.10 or newer."
  echo
  exit 1
fi

if ! $PY -c "import numpy" >/dev/null 2>&1; then
  echo
  echo "  Found $($PY --version 2>&1) but its packages are not installed yet."
  echo "  Run this once:"
  echo "      $PY -m pip install -r requirements.txt"
  echo
  exit 1
fi

if [ $# -eq 0 ]; then
  if ! $PY -c "import fastapi, uvicorn" >/dev/null 2>&1; then
    echo
    echo "  The front end needs fastapi and uvicorn:"
    echo "      $PY -m pip install -r requirements.txt"
    echo
    echo "  Or use the command line, which does not need them:"
    echo "      ./run.sh loads examples/ls3.json"
    echo
    exit 1
  fi
  $PY -c "import cadquery" >/dev/null 2>&1 || echo \
    "  Note: cadquery missing, so the 3D viewport will not load. $PY -m pip install cadquery"
  if ! "$PY" -c "import skfem, tetgen" >/dev/null 2>&1; then
    echo "  Note: scikit-fem/tetgen missing, so the Stress button and the fea"
    echo "  command will not work. $PY -m pip install scikit-fem tetgen scipy"
  fi
  if [ ! -f .env ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo
    echo "  Note: no Anthropic API key, so the Assistant pane will be off."
    echo "  To turn it on:  cp .env.example .env  and put your key in it."
  fi
  echo
  echo "  Piston System Research Tool -- http://127.0.0.1:8000/  (Ctrl-C to stop)"
  echo
  $PY -m psrt serve examples/ls3.json
else
  $PY -m psrt "$@"
fi
