"""Offline EIC calibration: compute per-head TVER across K environments,
derive C-scores, and select the intervention layer.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext

import torch
from tqdm import tqdm

from causal_core.envs import EnvMaker, BaseExample, ENV_LIST_DEFAULT
from causal_core.scores import RunningStats, tver_from_attn, compute_C, choose_intervention_layer
from causal_core.models import internvl as internvl_adapter

def pick_adapter(name: str):
    name = name.lower()
    if name == "llava":
        from causal_core.models import llava_adapter
        return llava_adapter
    if name in {"qwen", "qwen3", "qwen-vl-3", "qwen3-vl"}:
        from causal_core.models import qwen3_adapter
        return qwen3_adapter
    raise ValueError(f"Unsupported model_type: {name}")

def _ensure_repo_paths():
    """Add causal-hallucination-vlm repo root to sys.path so vendored packages (llava/) are importable."""
    here = os.path.abspath(__file__)

    repo_root = os.path.dirname(os.path.dirname(here))

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    return repo_root

def extract_attentions(out):
    """Return attentions as a tuple/list of tensors [B,H,Q,K] for each layer.

    Supports:
    - HF ModelOutput with `.attentions`
    - LLaVA forks returning tuples/lists where attentions are included as a trailing element
    """

    if hasattr(out, "attentions") and out.attentions is not None:
        return out.attentions

    if isinstance(out, (tuple, list)):

        for item in out:
            if hasattr(item, "attentions") and getattr(item, "attentions") is not None:
                return item.attentions

        for item in reversed(out):
            if (
                isinstance(item, (tuple, list))
                and len(item) > 0
                and torch.is_tensor(item[0])
                and item[0].dim() == 4
            ):
                return item

        for item in reversed(out):
            if torch.is_tensor(item) and item.dim() == 4:
                return (item,)

    raise RuntimeError(
        f"Could not extract attentions. Output type={type(out)}. "
        "Make sure output_attentions=True is supported by this model/fork."
    )

def _finalize_stats_from_global(stats_by_layer):
    means = []
    vars_ = []
    for st in stats_by_layer:
        m, v = st.finalize()
        means.append(m)
        vars_.append(v)
    return means, vars_

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument(
        "--model_type",
        choices=["llava", "qwen3", "internvl"],
        required=True,
        help="LLaVA-v1.5 and Qwen3-VL share the main calibration path. "
             "InternVL uses a different image-preprocessing pipeline and is "
             "dispatched to a dedicated branch.",
    )
    ap.add_argument("--layer", type=int, default=0, help="If --all_layers is set, this is ignored")
    ap.add_argument("--all_layers", action="store_true", help="Compute stats for all layers and select l~")
    ap.add_argument("--n_samples", type=int, default=1000)
    ap.add_argument("--envs", nargs="+", default=ENV_LIST_DEFAULT)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--question_file",
        type=str,
        required=True,
        help="MME-style jsonl with fields: question_id, image, question/text",
    )
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--conv_mode", type=str, default="llava_v1")
    ap.add_argument("--model_base", type=str, default=None)
    ap.add_argument(
        "--variance_mode",
        type=str,
        default="env_per_example",
        choices=["env_per_example", "global"],
        help=(
            "How to estimate variance for C-score. "
            "env_per_example computes Var_env(TVER) per example then averages across examples "
            "(recommended for invariance signal). "
            "global reproduces legacy variance over all (example,env) points."
        ),
    )
    ap.add_argument(
        "--amp_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "none"],
        help="Autocast dtype during calibration. bf16 is usually most stable on A100.",
    )
    ap.add_argument(
        "--seed0",
        type=int,
        default=0,
        help="Base random seed for environment perturbations (default 0).",
    )
    args = ap.parse_args()

    if args.model_type == "internvl":
        return _internvl_main(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ensure_repo_paths()

    qpath = os.path.expanduser(args.question_file)
    ipath = os.path.expanduser(args.image_folder)
    questions = [json.loads(q) for q in open(qpath, "r")]

    base_examples = []
    for i, line in enumerate(questions):
        image_file = line.get("image")
        qid = str(line.get("question_id", i))

        text = line.get("question") or line.get("text")
        if image_file is None or text is None:
            continue
        base_examples.append(
            BaseExample(
                idx=i,
                example_id=qid,
                image_path=os.path.join(ipath, image_file),
                text=text,
            )
        )

    if len(base_examples) == 0:
        raise RuntimeError(f"No examples loaded from {qpath}")

    env_maker = EnvMaker(base_examples, seed0=args.seed0)
    adapter = pick_adapter(args.model_type)

    model_type = args.model_type.lower()

    if model_type == "llava":
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from llava.utils import disable_torch_init

        disable_torch_init()
        model_path = os.path.expanduser(args.model_name)
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)

        if hasattr(model, "config"):
            try:
                model.config.output_attentions = True
                model.config.return_dict = True
            except Exception:
                pass

        bundle = {
            "tokenizer": tokenizer,
            "image_processor": image_processor,
            "conv_mode": args.conv_mode,
            "model": model,
        }

        model.to(device)
        model.eval()

    elif model_type in {"qwen", "qwen3"}:
        from transformers import AutoProcessor
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

        model_path = os.path.expanduser(args.model_name)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

        if hasattr(model, "config"):
            try:
                model.config.output_attentions = True
                model.config.return_dict = True
            except Exception:
                pass

        try:
            model.config._attn_implementation = "eager"
            if hasattr(model.config, "text_config"):
                model.config.text_config._attn_implementation = "eager"
            if hasattr(model, "model") and hasattr(model.model, "config"):
                model.model.config._attn_implementation = "eager"
        except Exception:
            pass
        bundle = {
            "processor": processor,
            "model": model,
        }

        model.eval()

    else:
        raise RuntimeError(f"Unsupported model_type={args.model_type}. Supported: llava, qwen3, internvl.")

    total_examples = min(args.n_samples, len(base_examples))
    total_steps = total_examples * len(args.envs)
    step = 0

    stats_by_layer = None

    sum_mu_by_layer = None
    sum_var_by_layer = None
    n_examples_agg = 0

    with torch.no_grad():
        pbar = tqdm(total=total_steps, desc="calibrating (example x env)", dynamic_ncols=True)
        t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        t1 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        if t0 is not None:
            t0.record()

        for ex in base_examples[:total_examples]:

            ex_tver_by_layer = None

            for env in args.envs:
                img, text = env_maker.get(ex, env)
                inputs = adapter.build_inputs(bundle, img, text, device)

                if device.type == "cuda" and args.amp_dtype != "none":
                    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
                    autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype)
                else:
                    autocast_ctx = nullcontext()

                with autocast_ctx:
                    out = model(**inputs, output_attentions=True, use_cache=False, return_dict=True)

                attentions = extract_attentions(out)
                vision_mask, text_mask = adapter.get_kv_masks(model, inputs)

                if args.all_layers:
                    L = len(attentions)
                    if ex_tver_by_layer is None:
                        ex_tver_by_layer = [[] for _ in range(L)]

                    if args.variance_mode == "global" and stats_by_layer is None:
                        num_heads = attentions[0].shape[1]
                        stats_by_layer = [RunningStats.create(num_heads, device="cpu") for _ in range(L)]

                    for l in range(L):
                        attn_last = attentions[l][:, :, -1, :]
                        tver = tver_from_attn(attn_last, text_mask=text_mask, vision_mask=vision_mask)
                        tver_h = tver.mean(dim=0).cpu()
                        ex_tver_by_layer[l].append(tver_h)
                        if args.variance_mode == "global":
                            stats_by_layer[l].update(tver_h)
                else:
                    if ex_tver_by_layer is None:
                        ex_tver_by_layer = [[]]

                    if args.variance_mode == "global" and stats_by_layer is None:
                        num_heads = attentions[args.layer].shape[1]
                        stats_by_layer = [RunningStats.create(num_heads, device="cpu")]

                    attn_last = attentions[args.layer][:, :, -1, :]
                    tver = tver_from_attn(attn_last, text_mask=text_mask, vision_mask=vision_mask)
                    tver_h = tver.mean(dim=0).cpu()
                    ex_tver_by_layer[0].append(tver_h)
                    if args.variance_mode == "global":
                        stats_by_layer[0].update(tver_h)

                step += 1
                pbar.update(1)
                if device.type == "cuda" and (step % 50 == 0 or step == total_steps):
                    t1.record()
                    torch.cuda.synchronize()
                    ms = t0.elapsed_time(t1)
                    it_s = (ms / 1000.0) / max(1, step)
                    eta_min = (total_steps - step) * it_s / 60.0
                    pbar.set_postfix_str(f"{it_s:.3f}s/it, ETA {eta_min:.1f}m")

            if args.variance_mode == "env_per_example":
                if ex_tver_by_layer is None:
                    continue

                if sum_mu_by_layer is None:
                    sum_mu_by_layer = []
                    sum_var_by_layer = []
                    for vals in ex_tver_by_layer:
                        if len(vals) == 0:
                            raise RuntimeError("No TVER values collected for one layer")
                        h_shape = vals[0].shape
                        sum_mu_by_layer.append(torch.zeros(h_shape, dtype=vals[0].dtype))
                        sum_var_by_layer.append(torch.zeros(h_shape, dtype=vals[0].dtype))

                for l, vals in enumerate(ex_tver_by_layer):
                    env_stack = torch.stack(vals, dim=0)
                    mu_ex = env_stack.mean(dim=0)
                    var_env_ex = env_stack.var(dim=0, unbiased=False)
                    sum_mu_by_layer[l] += mu_ex
                    sum_var_by_layer[l] += var_env_ex

                n_examples_agg += 1

        pbar.close()

    if args.variance_mode == "global":
        if stats_by_layer is None:
            raise RuntimeError("No stats collected in global mode")
        means, vars_ = _finalize_stats_from_global(stats_by_layer)
    else:
        if sum_mu_by_layer is None or sum_var_by_layer is None or n_examples_agg == 0:
            raise RuntimeError("No stats collected in env_per_example mode")
        means = [m / n_examples_agg for m in sum_mu_by_layer]
        vars_ = [v / n_examples_agg for v in sum_var_by_layer]

    if args.all_layers:
        mu_by_layer = torch.stack(means, dim=0)
        var_by_layer = torch.stack(vars_, dim=0)

        chosen_layer = choose_intervention_layer(mu_by_layer)
        mean = mu_by_layer[chosen_layer]
        var = var_by_layer[chosen_layer]
        C = compute_C(mean, var)
    else:
        chosen_layer = args.layer
        mean = means[0]
        var = vars_[0]
        C = compute_C(mean, var)

    payload = {
        "model_name": args.model_name,
        "model_type": args.model_type,
        "chosen_layer": int(chosen_layer),
        "envs": args.envs,
        "variance_mode": args.variance_mode,
        "n_examples": int(total_examples),
        "mean": mean,
        "var": var,
        "C": C,
    }

    if args.all_layers:
        payload["mean_by_layer"] = mu_by_layer
        payload["var_by_layer"] = var_by_layer

    nan_keys = ["mean", "var", "C"]
    for k in nan_keys:
        t = payload.get(k)
        if torch.is_tensor(t) and (torch.isnan(t).any() or torch.isinf(t).any()):
            raise RuntimeError(
                f"Calibration produced NaN/Inf in {k}. Try --amp_dtype bf16 (default) or --amp_dtype none."
            )

    torch.save(payload, args.out)
    print(f"Saved C-scores to {args.out}")

if __name__ == "__main__":
    main()

def _internvl_main(args=None):
    if args is None:
        ap = argparse.ArgumentParser()

        args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ensure_repo_paths()

    qpath = os.path.expanduser(args.question_file)
    ipath = os.path.expanduser(args.image_folder)
    questions = [json.loads(q) for q in open(qpath, "r")]

    base_examples = []
    for i, line in enumerate(questions):
        image_file = line.get("image")
        qid = str(line.get("question_id", i))
        text = line.get("question") or line.get("text")
        if image_file is None or text is None:
            continue
        base_examples.append(
            BaseExample(
                idx=i,
                example_id=qid,
                image_path=os.path.join(ipath, image_file),
                text=text,
            )
        )

    if len(base_examples) == 0:
        raise RuntimeError(f"No examples loaded from {qpath}")

    env_maker = EnvMaker(base_examples)

    from transformers import AutoProcessor, InternVLForConditionalGeneration

    model_path = os.path.expanduser(args.model_name)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=False)
    model = InternVLForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
        trust_remote_code=False,
    )

    if hasattr(model, "config"):
        try:
            model.config.output_attentions = True
            model.config.return_dict = True
        except Exception:
            pass

    try:
        model.config._attn_implementation = "eager"
        if hasattr(model.config, "text_config"):
            model.config.text_config._attn_implementation = "eager"
        if hasattr(model, "model") and hasattr(model.model, "config"):
            model.model.config._attn_implementation = "eager"
        if hasattr(model.model, "language_model") and hasattr(model.model.language_model, "config"):
            model.model.language_model.config._attn_implementation = "eager"
    except Exception:
        pass

    bundle = {"processor": processor, "model": model}
    model.eval()

    total_examples = min(args.n_samples, len(base_examples))
    total_steps = total_examples * len(args.envs)
    step = 0

    stats_by_layer = None
    sum_mu_by_layer = None
    sum_var_by_layer = None
    n_examples_agg = 0

    with torch.no_grad():
        pbar = tqdm(total=total_steps, desc="calibrating internvl (example x env)", dynamic_ncols=True)
        t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        t1 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        if t0 is not None:
            t0.record()

        for ex in base_examples[:total_examples]:
            ex_tver_by_layer = None

            for env in args.envs:
                img, text = env_maker.get(ex, env)
                inputs = internvl_adapter.build_inputs(bundle, img, text, device)

                if device.type == "cuda" and args.amp_dtype != "none":
                    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
                    autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype)
                else:
                    autocast_ctx = nullcontext()

                with autocast_ctx:
                    out = model(**inputs, output_attentions=True, use_cache=False, return_dict=True)

                attentions = extract_attentions(out)
                vision_mask, text_mask = internvl_adapter.get_kv_masks(model, inputs)

                if args.all_layers:
                    L = len(attentions)
                    if ex_tver_by_layer is None:
                        ex_tver_by_layer = [[] for _ in range(L)]

                    if args.variance_mode == "global" and stats_by_layer is None:
                        num_heads = attentions[0].shape[1]
                        stats_by_layer = [RunningStats.create(num_heads, device="cpu") for _ in range(L)]

                    for l in range(L):
                        attn_last = attentions[l][:, :, -1, :]
                        tver = tver_from_attn(attn_last, text_mask=text_mask, vision_mask=vision_mask)
                        tver_h = tver.mean(dim=0).cpu()
                        ex_tver_by_layer[l].append(tver_h)
                        if args.variance_mode == "global":
                            stats_by_layer[l].update(tver_h)
                else:
                    if ex_tver_by_layer is None:
                        ex_tver_by_layer = [[]]

                    if args.variance_mode == "global" and stats_by_layer is None:
                        num_heads = attentions[args.layer].shape[1]
                        stats_by_layer = [RunningStats.create(num_heads, device="cpu")]

                    attn_last = attentions[args.layer][:, :, -1, :]
                    tver = tver_from_attn(attn_last, text_mask=text_mask, vision_mask=vision_mask)
                    tver_h = tver.mean(dim=0).cpu()
                    ex_tver_by_layer[0].append(tver_h)
                    if args.variance_mode == "global":
                        stats_by_layer[0].update(tver_h)

                step += 1
                pbar.update(1)
                if device.type == "cuda" and (step % 50 == 0 or step == total_steps):
                    t1.record()
                    torch.cuda.synchronize()
                    ms = t0.elapsed_time(t1)
                    it_s = (ms / 1000.0) / max(1, step)
                    eta_min = (total_steps - step) * it_s / 60.0
                    pbar.set_postfix_str(f"{it_s:.3f}s/it, ETA {eta_min:.1f}m")

            if args.variance_mode == "env_per_example":
                if ex_tver_by_layer is None:
                    continue

                if sum_mu_by_layer is None:
                    sum_mu_by_layer, sum_var_by_layer = [], []
                    for vals in ex_tver_by_layer:
                        if len(vals) == 0:
                            raise RuntimeError("No TVER values collected for one layer")
                        h_shape = vals[0].shape
                        sum_mu_by_layer.append(torch.zeros(h_shape, dtype=vals[0].dtype))
                        sum_var_by_layer.append(torch.zeros(h_shape, dtype=vals[0].dtype))

                for l, vals in enumerate(ex_tver_by_layer):
                    env_stack = torch.stack(vals, dim=0)
                    mu_ex = env_stack.mean(dim=0)
                    var_env_ex = env_stack.var(dim=0, unbiased=False)
                    sum_mu_by_layer[l] += mu_ex
                    sum_var_by_layer[l] += var_env_ex

                n_examples_agg += 1

        pbar.close()

    if args.variance_mode == "global":
        if stats_by_layer is None:
            raise RuntimeError("No stats collected in global mode")
        means, vars_ = _finalize_stats_from_global(stats_by_layer)
    else:
        if sum_mu_by_layer is None or sum_var_by_layer is None or n_examples_agg == 0:
            raise RuntimeError("No stats collected in env_per_example mode")
        means = [m / n_examples_agg for m in sum_mu_by_layer]
        vars_ = [v / n_examples_agg for v in sum_var_by_layer]

    if args.all_layers:
        mu_by_layer = torch.stack(means, dim=0)
        var_by_layer = torch.stack(vars_, dim=0)
        chosen_layer = choose_intervention_layer(mu_by_layer)
        mean = mu_by_layer[chosen_layer]
        var = var_by_layer[chosen_layer]
        C = compute_C(mean, var)
    else:
        chosen_layer = args.layer
        mean = means[0]
        var = vars_[0]
        C = compute_C(mean, var)

    payload = {
        "model_name": args.model_name,
        "model_type": "internvl",
        "chosen_layer": int(chosen_layer),
        "envs": args.envs,
        "variance_mode": args.variance_mode,
        "n_examples": int(total_examples),
        "mean": mean,
        "var": var,
        "C": C,
    }
    if args.all_layers:
        payload["mean_by_layer"] = mu_by_layer
        payload["var_by_layer"] = var_by_layer

    for k in ("mean", "var", "C"):
        t = payload.get(k)
        if torch.is_tensor(t) and (torch.isnan(t).any() or torch.isinf(t).any()):
            raise RuntimeError(
                f"Calibration produced NaN/Inf in {k}. Try --amp_dtype bf16 (default) or --amp_dtype none."
            )

    torch.save(payload, args.out)
    print(f"Saved C-scores to {args.out} (chosen_layer={int(chosen_layer)})")

