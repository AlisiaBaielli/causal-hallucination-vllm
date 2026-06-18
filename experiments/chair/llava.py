"""
CHAIR evaluation for LLaVA-v1.5-7B.

Supports: vanilla, chall (ours), ONLY, ONLY+EIC, VCD, M3ID.
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
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "transformers" / "src"))
sys.path.insert(0, str(REPO))
_self_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _self_dir]

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.model import LlavaLlamaForCausalLM

from causal_core.eval_common import caption_output_path, load_c_scores, resolve_method
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

def parse_args():
    p = argparse.ArgumentParser(description="CHAIR eval for LLaVA-v1.5-7B")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--anno_path", type=str, required=True)
    p.add_argument("--out_path", type=str, required=True,
                   help="Output directory or .jsonl file path")
    p.add_argument("--num_eval_samples", type=int, default=500)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--conv_mode", type=str, default="v1")

    p.add_argument("--c_scores_path", type=str, default=None)
    p.add_argument("--layer_index", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--img_start", type=int, default=35)
    p.add_argument("--img_len", type=int, default=576)
    p.add_argument("--method_name", type=str, default=None,
                   help="Override output method tag (default: inferred from flags)")

    p.add_argument("--no_hook", action="store_true",
                   help="Disable CHALL monitor (vanilla or ONLY/VCD/M3ID)")
    p.add_argument("--use_only", action="store_true", help="ONLY baseline")
    p.add_argument("--use_eic_heads", action="store_true",
                   help="With --use_only: use offline EIC head set in CD branch")
    p.add_argument("--use_vcd", action="store_true", help="VCD baseline")
    p.add_argument("--use_m3id", action="store_true", help="M3ID baseline")
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
    if args.method_name:
        method = args.method_name
    if needs_scores and not args.c_scores_path:
        raise ValueError(f"--c_scores_path is required for method={method}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=model.device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    evolve_only_sampling()

    monitor = None
    orig_fwd = None
    processors = LogitsProcessorList([])
    layer_for_only = args.layer_index

    if needs_scores:
        if not args.c_scores_path:
            raise ValueError(f"--c_scores_path is required for method={method}")
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

    with open(args.anno_path) as f:
        coco = json.load(f)
    images = coco["images"]
    random.shuffle(images)
    images = images[: args.num_eval_samples]

    out_file = caption_output_path(args.out_path, method)
    log.info(f"CHAIR method={method} alpha={args.alpha} n={len(images)} -> {out_file}")

    results = []
    for img_info in tqdm(images, total=len(images)):
        img_id = img_info["id"]
        img_path = os.path.join(args.data_path, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image_tensor = image_tensor.unsqueeze(0).half().to(model.device)

        qs = DEFAULT_IMAGE_TOKEN + "\nPlease describe this image in detail."
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt",
        ).unsqueeze(0).to(model.device)

        image_neg = None
        if method == "vcd":
            image_neg = add_diffusion_noise(image_tensor, args.noise_step)

        gen_kwargs = dict(
            images=image_tensor,
            images_neg=image_neg,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
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

        output_text = tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:], skip_special_tokens=True,
        )[0].strip()
        results.append({"image_id": img_id, "caption": output_text})

    if monitor is not None and orig_fwd is not None:
        monitor.restore(orig_fwd)

    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    log.info(f"[done] Saved {len(results)} captions to {out_file}")

if __name__ == "__main__":
    main()
