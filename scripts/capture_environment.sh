#!/usr/bin/env bash
# Thin wrapper — 01 §12 steps 2-9. The capture itself lives in capture_environment.py so it
# can be exercised against mocked hardware; this file only chooses the frozen interpreter.
#
#   bash scripts/capture_environment.sh [--out DIR]
#
# Publication is transactional: a failed capture writes no manifest at all.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
PY="$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )"
exec "$PY" scripts/capture_environment.py --root "$ROOT" "$@"
