# coding=utf-8
"""Put the sibling projects on sys.path before any test imports them.

E2 borrows `tcr.evaluation.quality` (the support definition) and `tcr.evaluation.registry`
(the model list) rather than keeping second copies, so the tests need the same
path setup the CLIs do.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
