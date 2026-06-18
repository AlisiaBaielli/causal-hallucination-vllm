# transformers/

A minimal patched copy of HuggingFace `transformers`, containing only the file we modify:

```
transformers/src/transformers/models/llama/modeling_llama.py
```

## Why only LLaMA?

- **LLaVA-v1.5-7B** uses **LLaMA-2-7B** as its language backbone. We patch
  `modeling_llama.py` to expose attention weights and install a forward-pass hook at the
  EIC-selected intervention layer.
- **Qwen3-VL-8B** and **InternVL3.5-8B-HF** ship with their own modelling code that already
  exposes the required attention tensors. We use the stock transformers package and
  install hooks at runtime (see `causal_core/hooks.py`) — no source-level patch needed.

## Applying the patch

Replace the corresponding file in your transformers installation, or install this fork:

```bash
pip uninstall transformers
cd transformers && pip install -e .
```
