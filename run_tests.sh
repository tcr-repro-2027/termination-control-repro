#!/usr/bin/env bash
# CPU-only test-suite: no GPU, no model weights, no dataset needed (~30 s).
# Each suite has its own conftest.py, so they are run one directory at a time.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONDONTWRITEBYTECODE=1
status=0
for suite in events token_orbit extraction evaluation boundary steering motif support_probe data paper; do
    echo "=== tests/$suite"
    python -m pytest -q -p no:cacheprovider "tests/$suite" "$@" || status=1
done
exit $status
