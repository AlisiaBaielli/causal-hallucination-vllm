"""
MME evaluation for LLaVA-v1.5-7B.

Supports: vanilla, chall (ours), ONLY, VCD, M3ID.
"""
import argparse
import json
import logging
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers.generation.logits_process import LogitsProcessorList

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "transformers" / "src"))
sys.path.insert(0, str(REPO))
_self_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _self_dir]

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from causal_core.eval_common import load_c_scores, resolve_method
from causal_core.models.llava_sampling import evolve_only_sampling
from causal_core.monitor import CausalLogitsProcessor, CausalMonitor
from causal_core.only_eic import inject_eic_for_only
from causal_core.vcd import add_diffusion_noise

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

evolve_only_sampling()

def parse_args():
    p = argparse.ArgumentParser(description="MME eval for LLaVA-v1.5-7B")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--image_folder", type=str, required=True)
    p.add_argument("--question_file", type=str, required=True)
    p.add_argument("--answers_file", type=str, required=True)
    p.add_argument("--conv_mode", type=str, default="llava_v1")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--do_sample", type=bool, default=True)
    p.add_argument("--c_scores_path", type=str, default=None)
    p.add_argument("--layer_index", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--img_start", type=int, default=35)
    p.add_argument("--img_len", type=int, default=576)
    p.add_argument("--method_name", type=str, default=None)
    p.add_argument("--no_hook", action="store_true")
    p.add_argument("--use_only", action="store_true")
    p.add_argument("--use_eic_heads", action="store_true",
                   help="With --use_only: use offline EIC head set in CD branch (ONLY+EIC)")
    p.add_argument("--use_vcd", action="store_true")
    p.add_argument("--use_m3id", action="store_true")
    p.add_argument("--noise_step", type=int, default=500)
    p.add_argument("--js_gamma", type=float, default=0.2)
    p.add_argument("--ritual_alpha_pos", type=float, default=3.0)
    p.add_argument("--ritual_alpha_neg", type=float, default=1.0)
    p.add_argument("--ritual_beta", type=float, default=0.1)
    return p.parse_args()

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    method, needs_scores = resolve_method(args)
    # `--method_name` is an output label only (the answer path is explicit via
    # --answers_file); decoding behavior must key off the canonical `method`.
    if needs_scores and not args.c_scores_path:
        raise ValueError(f"--c_scores_path is required for method={method}")

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    # CHALL's grounding monitor needs real attention weights (output_attentions),
    # which the sdpa kernel returns as None; force eager for chall only.
    tokenizer, model, image_processor, _context_len = load_pretrained_model(
        model_path, None, model_name,
        attn_implementation=("eager" if method == "chall" else None),
    )

    monitor = None
    orig_fwd = None
    processors = LogitsProcessorList([])
    layer_for_only = args.layer_index

    if needs_scores:
        c_scores = load_c_scores(args.c_scores_path, args.layer_index)
        log.info(
            f"C-scores layer={args.layer_index}: "
            f"nonzero={int((c_scores > 0).sum())}/{len(c_scores)}"
        )
        if method == "chall":
            monitor = CausalMonitor(
                model, args.layer_index, c_scores,
                img_start=args.img_start, img_len=args.img_len,
            )
            orig_fwd = monitor.install_qk_hook()
            processors = LogitsProcessorList([
                CausalLogitsProcessor(monitor, alpha=args.alpha)
            ])
        elif method == "only_eic":
            layer_for_only = inject_eic_for_only(
                model=model,
                scores_path=args.c_scores_path,
                layer_index=args.layer_index,
                pure_eic=True,
            )

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    log.info(f"MME method={method} alpha={args.alpha} n={len(questions)} -> {args.answers_file}")

    os.makedirs(os.path.dirname(args.answers_file) or ".", exist_ok=True)

    with open(args.answers_file, "w") as ans_file:
        for line in tqdm(questions, desc=f"MME-{method}"):
            idx = line["question_id"]
            image_file = line["image"]
            cur_prompt = line["question"]
            qs = DEFAULT_IMAGE_TOKEN + "\n" + cur_prompt

            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            image_tensor = image_tensor.unsqueeze(0).half().to(model.device)

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt",
            ).unsqueeze(0).to(model.device)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

            image_neg = None
            if method == "vcd":
                image_neg = add_diffusion_noise(image_tensor, args.noise_step)

            gen_kwargs = dict(
                images=image_tensor,
                images_neg=image_neg,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                use_only=(method in ("only", "only_eic")),
                use_vcd=(method == "vcd"),
                use_m3id=(method == "m3id"),
                enhance_layer_index=layer_for_only,
                js_gamma=args.js_gamma,
                ritual_alpha_pos=args.ritual_alpha_pos,
                ritual_alpha_neg=args.ritual_alpha_neg,
                ritual_beta=args.ritual_beta,
            )
            if processors:
                gen_kwargs["logits_processor"] = processors

            with torch.inference_mode():
                output_ids = model.generate(input_ids, **gen_kwargs)
            if isinstance(output_ids, tuple):
                output_ids = output_ids[0]

            outputs = tokenizer.batch_decode(
                output_ids[:, input_ids.shape[1]:], skip_special_tokens=True,
            )[0].strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)].strip()

            ans_file.write(json.dumps({
                "question_id": idx,
                "prompt": cur_prompt,
                "text": outputs,
                "model_id": model_name,
                "image": image_file,
                "metadata": {},
            }) + "\n")
            ans_file.flush()

    if monitor is not None and orig_fwd is not None:
        monitor.restore(orig_fwd)
    log.info(f"[done] Saved answers to {args.answers_file}")

if __name__ == "__main__":
    main()
