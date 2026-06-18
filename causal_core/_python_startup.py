import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[1]
_repo_str = str(_repo)
if _repo_str not in sys.path:
    sys.path.insert(0, _repo_str)

from causal_core.transformers_fork import ensure_all_forks

ensure_all_forks()
