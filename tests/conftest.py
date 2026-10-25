"""Use the stub SDK only when the real one is not installed (e.g. a sandbox without access to
the course package index)."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "source"))
try:
    import vis_nav_sdk  # noqa: F401
except ImportError:
    sys.path.insert(0, str(HERE / "stubs"))
