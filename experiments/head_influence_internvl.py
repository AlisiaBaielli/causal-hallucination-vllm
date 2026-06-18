"""
Head-ablation mechanistic validation for InternVL3.5-8B-HF.
"""
import os, sys, json, argparse, warnings, logging, random
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from causal_core.transformers_fork import ensure_internvl_fork
ensure_internvl_fork()

from transformers import AutoProcessor
from transformers.models.internvl.modeling_internvl_real import InternVLForConditionalGeneration
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str,
                   default=str(REPO / "data/models/InternVL3_5-8B-HF"))
    p.add_argument("--coco_dir", type=str, default=str(REPO / "data/coco/val2014"))
    p.add_argument("--pope_file", type=str, default=str(REPO / "data/POPE/coco/coco_pope_popular.json"))
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

    payload = torch.load(args.scores_path, map_location="cpu", weights_only=False)
    C = payload["C"].float() if "C" in payload else payload["scores"].float()
    n_heads = int(C.shape[0])
    log.info(f"C-scores at L{args.enhance_layer_index}: nonzero={int((C>0).sum())}/{n_heads}")
    log.info(f"Top-5 heads by C: {C.argsort(descending=True)[:5].tolist()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Loading InternVL: {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = InternVLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        target_attn = model.model.language_model.layers[args.enhance_layer_index].self_attn
    elif hasattr(model, "language_model"):
        target_attn = model.language_model.layers[args.enhance_layer_index].self_attn
    else:
        raise ValueError("Could not resolve InternVL attention layer")
    head_dim = target_attn.head_dim
    log.info(f"Layer {args.enhance_layer_index}: num_query_heads={n_heads}, head_dim={head_dim}")

    orig_o_proj_forward = target_attn.o_proj.forward
    head_to_zero = {"idx": -1}

    def hooked_o_proj(input_tensor):
        if head_to_zero["idx"] >= 0:
            bsz, q_len, hidden = input_tensor.shape
            per_head = input_tensor.view(bsz, q_len, n_heads, head_dim).clone()
            per_head[:, :, head_to_zero["idx"], :] = 0
            input_tensor = per_head.view(bsz, q_len, hidden)
        return orig_o_proj_forward(input_tensor)

    target_attn.o_proj.forward = hooked_o_proj

    coco_files = sorted([f for f in os.listdir(args.coco_dir) if f.endswith(".jpg")])
    random.shuffle(coco_files)
    gen_files = coco_files[:args.n_generative]

    with open(args.pope_file) as f:
        pope = [json.loads(l) for l in f]
    pope = pope[:args.n_yesno]

    GEN_PROMPT = "Describe this image in detail."

    def _gen_one_step_scores(inputs):
        out = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            use_cache=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
        return out.scores[0][0].detach().float().cpu()

    def measure_per_head_deltas(image_path, prompt_text):

        image = Image.open(image_path).convert("RGB")
        msgs = [{
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}],
        }]
        inputs = processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device)

        head_to_zero["idx"] = -1
        with torch.inference_mode():
            ref_logits = _gen_one_step_scores(inputs)

        deltas = np.zeros(n_heads)
        for h in range(n_heads):
            head_to_zero["idx"] = h
            with torch.inference_mode():
                ab_logits = _gen_one_step_scores(inputs)
            deltas[h] = (ref_logits - ab_logits).norm(dim=-1).item()

        head_to_zero["idx"] = -1
        return deltas

    log.info(f"=== Running {args.n_generative} generative prompts ===")
    gen_deltas = []
    for f in tqdm(gen_files, desc="Generative"):
        try:
            d = measure_per_head_deltas(os.path.join(args.coco_dir, f), GEN_PROMPT)
            gen_deltas.append(d)
        except Exception as e:
            log.warning(f"Skipping {f}: {e}")
    gen_deltas = np.stack(gen_deltas) if gen_deltas else np.zeros((0, n_heads))
    gen_mean = gen_deltas.mean(axis=0) if len(gen_deltas) > 0 else np.zeros(n_heads)

    log.info(f"=== Running {args.n_yesno} POPE yes/no prompts ===")
    yn_deltas = []
    for q in tqdm(pope, desc="Yes/No"):
        img_file = q["image"]
        img_path = os.path.join(args.coco_dir, img_file)
        if not os.path.exists(img_path):
            continue
        question = q["text"] + " Answer yes or no only."
        try:
            d = measure_per_head_deltas(img_path, question)
            yn_deltas.append(d)
        except Exception as e:
            log.warning(f"Skipping {img_file}: {e}")
    yn_deltas = np.stack(yn_deltas) if yn_deltas else np.zeros((0, n_heads))
    yn_mean = yn_deltas.mean(axis=0) if len(yn_deltas) > 0 else np.zeros(n_heads)

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

    print("\n=== InternVL §4.1 head-ablation correlation ===")
    print(f"  Generative: Pearson r={gen_corr['pearson_r']:.3f} (p={gen_corr['pearson_p']:.4g})  "
          f"Spearman ρ={gen_corr['spearman_rho']:.3f} (p={gen_corr['spearman_p']:.4g})")
    print(f"  Yes/No:     Pearson r={yn_corr['pearson_r']:.3f} (p={yn_corr['pearson_p']:.4g})  "
          f"Spearman ρ={yn_corr['spearman_rho']:.3f} (p={yn_corr['spearman_p']:.4g})")

    out = {
        "model": "InternVL3.5-8B-HF",
        "layer": args.enhance_layer_index,
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
    out_path = os.path.join(args.output_dir, "mechanistic_results_internvl.json")
    with open(out_path, "w") as fp:
        json.dump(out, fp, indent=2)
    log.info(f"Saved → {out_path}")

if __name__ == "__main__":
    main()
