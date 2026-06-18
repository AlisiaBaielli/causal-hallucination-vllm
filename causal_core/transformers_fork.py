"""Mount repo fork modules under ``transformers.models.*``.

Stock HuggingFace ``transformers`` (>=5) provides the base package. Our fork
under ``transformers/src/transformers/models/`` carries ONLY/chall attention
patches for Qwen3-VL and InternVL; those files must replace or extend the
corresponding ``transformers.models`` submodules at import time.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FORK_MODELS = REPO / "transformers" / "src" / "transformers" / "models"

_QWEN3_VL_DONE = False
_INTERNVL_DONE = False
_LLAMA_DONE = False


def _purge_modules(prefix: str) -> None:
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            del sys.modules[name]


def _load_module(full_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(full_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load fork module {full_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


def ensure_qwen3_vl_fork() -> None:
    global _QWEN3_VL_DONE
    if _QWEN3_VL_DONE:
        return

    fork_dir = FORK_MODELS / "qwen3_vl"
    if not fork_dir.is_dir():
        raise ImportError(f"Missing Qwen3-VL fork at {fork_dir}")

    import transformers.models  # noqa: F401 — parent must exist (transformers>=5)

    pkg_name = "transformers.models.qwen3_vl"
    _purge_modules(pkg_name)

    spec = importlib.util.spec_from_file_location(
        pkg_name,
        fork_dir / "__init__.py",
        submodule_search_locations=[str(fork_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create package spec for {pkg_name}")
    pkg = importlib.util.module_from_spec(spec)
    pkg.__path__ = [str(fork_dir)]  # type: ignore[attr-defined]
    sys.modules[pkg_name] = pkg
    spec.loader.exec_module(pkg)

    _QWEN3_VL_DONE = True


def ensure_internvl_fork() -> None:
    global _INTERNVL_DONE
    if _INTERNVL_DONE:
        return

    fork_dir = FORK_MODELS / "internvl"
    if not fork_dir.is_dir():
        raise ImportError(f"Missing InternVL fork at {fork_dir}")

    importlib.import_module("transformers.models.internvl")

    for stem in ("modeling_qwen3_te", "modeling_internvl_real"):
        full_name = f"transformers.models.internvl.{stem}"
        file_path = fork_dir / f"{stem}.py"
        if not file_path.is_file():
            raise ImportError(f"Missing fork file {file_path}")
        if full_name in sys.modules:
            del sys.modules[full_name]
        _load_module(full_name, file_path)

    _INTERNVL_DONE = True


def ensure_llama_fork() -> None:
    """Mount the ONLY-aware fork over ``transformers.models.llama.modeling_llama``.

    Stock HF Llama (>=5) does not carry the ONLY contrastive-decoding (CD) branch
    that LLaVA's ``use_only`` path relies on. This replaces the ``modeling_llama``
    submodule with the fork (stock TF5 Llama + CD branch) so that
    ``from transformers.models.llama.modeling_llama import LlamaModel`` picks it up.
    """
    global _LLAMA_DONE
    if _LLAMA_DONE:
        return

    fork_dir = FORK_MODELS / "llama"
    file_path = fork_dir / "modeling_llama.py"
    if not file_path.is_file():
        raise ImportError(f"Missing Llama fork file {file_path}")

    importlib.import_module("transformers.models.llama")  # parent must exist

    full_name = "transformers.models.llama.modeling_llama"
    if full_name in sys.modules:
        del sys.modules[full_name]
    module = _load_module(full_name, file_path)

    # Rebind the package + top-level references so callers that did
    # ``from transformers import LlamaModel`` before the fork still resolve it.
    import transformers
    import transformers.models.llama as _llama_pkg
    for cls in ("LlamaModel", "LlamaForCausalLM", "LlamaPreTrainedModel"):
        obj = getattr(module, cls, None)
        if obj is not None:
            setattr(_llama_pkg, cls, obj)
            setattr(transformers, cls, obj)

    _LLAMA_DONE = True


def ensure_all_forks() -> None:
    ensure_qwen3_vl_fork()
    ensure_internvl_fork()
    ensure_llama_fork()
