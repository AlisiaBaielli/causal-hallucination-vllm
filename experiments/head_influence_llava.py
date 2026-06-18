"""
Mechanistic validation for LLaVA-v1.5-7B (thesis Section 4.2).
"""
import os
import sys
import json
import argparse
import random
import warnings
import logging
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from scipy import stats

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_repo, "transformers", "src"))
sys.path.insert(0, _repo)

from llava.model import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from transformers import AutoTokenizer

from causal_core.models.llava_sampling import evolve_only_sampling
from causal_core.eval_common import load_c_scores

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--coco_dir", type=str, required=True)
    p.add_argument("--pope_file", type=str, required=True)
    p.add_argument("--scores_path", type=str, required=True)
    p.add_argument("--enhance_layer_index", type=int, default=1)
    p.add_argument("--n_generative", type=int, default=200)
    p.add_argument("--n_yesno", type=int, default=200)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    C = load_c_scores(args.scores_path, args.enhance_layer_index).float()
    n_heads = int(C.shape[0])
    log.info(f"C-scores at L{args.enhance_layer_index}: nonzero={int((C > 0).sum())}/{n_heads}")
    log.info(f"Top-5 heads by C: {C.argsort(descending=True)[:5].tolist()}")

    log.info(f"Loading LLaVA: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=model.device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    # ONLY contrastive sampling (provides the CD branch at the enhance layer).
    evolve_only_sampling()

    target_attn = model.model.layers[args.enhance_layer_index].self_attn
    head_dim = target_attn.head_dim
    log.info(f"Layer {args.enhance_layer_index}: num_heads={n_heads}, head_dim={head_dim}")

    # Hook o_proj at the enhance layer. With use_only, o_proj is called twice
    # per step: once for the CD branch, once for the real branch. Capture the
    # per-head pre-projection vectors at the last token for both calls. The
    # ||real - cd|| norm is symmetric in call order.
    orig_o_proj_forward = target_attn.o_proj.forward
    per_head_buffer = {}

    def hooked_o_proj(input_tensor):
        bsz, q_len, hidden = input_tensor.shape
        per_head_last = input_tensor.view(bsz, q_len, n_heads, head_dim)[:, -1, :, :]
        call_idx = per_head_buffer.get("_call_count", 0)
        if call_idx == 0:
            per_head_buffer["a"] = per_head_last.detach().float().cpu()
        elif call_idx == 1:
            per_head_buffer["b"] = per_head_last.detach().float().cpu()
        per_head_buffer["_call_count"] = call_idx + 1
        return orig_o_proj_forward(input_tensor)

    target_attn.o_proj.forward = hooked_o_proj

    coco_files = sorted([f for f in os.listdir(args.coco_dir) if f.endswith(".jpg")])
    random.shuffle(coco_files)
    gen_files = coco_files[:args.n_generative]

    with open(args.pope_file) as f:
        pope = [json.loads(l) for l in f]
    pope = pope[:args.n_yesno]

    GEN_PROMPT = "Please describe this image in detail."

    def build_input(image_path, prompt_text):
        image = Image.open(image_path).convert("RGB")
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        conv = conv_templates["v1"].copy()
        qu = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
        conv.append_message(conv.roles[0], qu)
        conv.append_message(conv.roles[1], None)
        prompt_full = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_full, tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors="pt").unsqueeze(0).cuda()
        return input_ids, image_tensor.unsqueeze(0).half().cuda()

    def measure_per_head_deltas(image_path, prompt_text):
        input_ids, image_tensor = build_input(image_path, prompt_text)
        per_head_buffer.clear()
        per_head_buffer["_call_count"] = 0
        with torch.inference_mode():
            model.generate(
                input_ids,
                images=image_tensor,
                images_pos=None,
                images_neg=None,
                do_sample=False,
                max_new_tokens=1,
                use_cache=False,
                use_ritual=False,
                use_vcd=False,
                use_m3id=False,
                use_only=True,
                ritual_alpha_pos=3.0,
                ritual_alpha_neg=1.0,
                ritual_beta=0.1,
                js_gamma=0.2,
                enhance_layer_index=args.enhance_layer_index,
            )
        if "a" not in per_head_buffer or "b" not in per_head_buffer:
            raise RuntimeError("CD branch did not run (o_proj called <2x); use_only path inactive")
        real_ph = per_head_buffer["a"][0]
        cd_ph = per_head_buffer["b"][0]
        return (real_ph - cd_ph).norm(dim=-1).numpy()

    log.info(f"=== Running {args.n_generative} generative prompts ===")
    gen_deltas = []
    for f in tqdm(gen_files, desc="Generative"):
        try:
            gen_deltas.append(measure_per_head_deltas(os.path.join(args.coco_dir, f), GEN_PROMPT))
        except Exception as e:
            log.warning(f"Skip {f}: {e}")
    gen_deltas = np.stack(gen_deltas) if gen_deltas else np.zeros((0, n_heads))
    gen_mean = gen_deltas.mean(axis=0) if len(gen_deltas) > 0 else np.zeros(n_heads)

    log.info(f"=== Running {args.n_yesno} POPE yes/no prompts ===")
    yn_deltas = []
    for q in tqdm(pope, desc="Yes/No"):
        img_path = os.path.join(args.coco_dir, q["image"])
        if not os.path.exists(img_path):
            continue
        try:
            yn_deltas.append(measure_per_head_deltas(img_path, q["text"]))
        except Exception as e:
            log.warning(f"Skip {q.get('image')}: {e}")
    yn_deltas = np.stack(yn_deltas) if yn_deltas else np.zeros((0, n_heads))
    yn_mean = yn_deltas.mean(axis=0) if len(yn_deltas) > 0 else np.zeros(n_heads)

    target_attn.o_proj.forward = orig_o_proj_forward

    C_np = C.numpy()

    def corr(C, d):
        return {
            "pearson_r": float(stats.pearsonr(C, d)[0]),
            "pearson_p": float(stats.pearsonr(C, d)[1]),
            "spearman_rho": float(stats.spearmanr(C, d)[0]),
            "spearman_p": float(stats.spearmanr(C, d)[1]),
        }
    gen_corr = corr(C_np, gen_mean)
    yn_corr = corr(C_np, yn_mean)

    print("\n=== LLaVA mechanistic correlation (per-head correction strength) ===")
    print(f"  Generative: Pearson r={gen_corr['pearson_r']:.3f} (p={gen_corr['pearson_p']:.4g})  "
          f"Spearman rho={gen_corr['spearman_rho']:.3f} (p={gen_corr['spearman_p']:.4g})")
    print(f"  Yes/No:     Pearson r={yn_corr['pearson_r']:.3f} (p={yn_corr['pearson_p']:.4g})  "
          f"Spearman rho={yn_corr['spearman_rho']:.3f} (p={yn_corr['spearman_p']:.4g})")

    out = {
        "model": "LLaVA-v1.5-7B",
        "layer": args.enhance_layer_index,
        "method": "contrastive_correction",
        "C_scores": C_np.tolist(),
        "n_generative": int(len(gen_deltas)),
        "n_yesno": int(len(yn_deltas)),
        "gen_mean_delta": gen_mean.tolist(),
        "yn_mean_delta": yn_mean.tolist(),
        "correlation": {
            "gen_pearson_r": gen_corr["pearson_r"],
            "gen_pearson_p": gen_corr["pearson_p"],
            "gen_spearman_rho": gen_corr["spearman_rho"],
            "gen_spearman_p": gen_corr["spearman_p"],
            "yn_pearson_r": yn_corr["pearson_r"],
            "yn_pearson_p": yn_corr["pearson_p"],
            "yn_spearman_rho": yn_corr["spearman_rho"],
            "yn_spearman_p": yn_corr["spearman_p"],
        },
    }
    out_path = os.path.join(args.output_dir, "mechanistic_results.json")
    with open(out_path, "w") as fp:
        json.dump(out, fp, indent=2)
    log.info(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
