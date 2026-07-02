"""
Import this before any `from config import ...` / `from prompts import ...`.

Adds the sibling config/ and data/ folders to sys.path so scripts living in
src/ can import config.py and prompts.py by their bare module names despite
living in a different directory.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _sub in ("config", "data"):
    _p = str(_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
