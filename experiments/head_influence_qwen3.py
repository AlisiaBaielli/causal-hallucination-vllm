
"""
Mechanistic analysis for Qwen3-VL: per-head contrastive correction strength.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo)
sys.path.insert(0, os.path.join(_repo, 'experiments'))

from causal_core.transformers_fork import ensure_qwen3_vl_fork
ensure_qwen3_vl_fork()

from transformers import AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from causal_core.models.qwen3 import evolve_only_sampling_qwen3

from PIL import Image
from tqdm import tqdm
from scipy import stats

evolve_only_sampling_qwen3()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--coco_dir", type=str, required=True)
    parser.add_argument("--pope_file", type=str, required=True)
    parser.add_argument("--scores_path", type=str, required=True)
    parser.add_argument("--enhance_layer_index", type=int, default=0)
    parser.add_argument("--n_generative", type=int, default=200)
    parser.add_argument("--n_yesno", type=int, default=200)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(42)

    payload = torch.load(args.scores_path, map_location="cpu")
    if "C" in payload:
        C = payload["C"].float()
    elif "scores" in payload:
        C = payload["scores"].float()
    else:
        raise ValueError(f"Cannot find C-scores in {args.scores_path}")

    cmin, cmax = C.min(), C.max()
    if cmax > cmin:
        C = (C - cmin) / (cmax - cmin)

    num_heads = C.shape[0]
    print(f"C-scores at L{args.enhance_layer_index}: {C.tolist()}")
    print(f"Nonzero: {(C > 0).sum().item()}/{num_heads}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype='auto',
        device_map='auto',
        trust_remote_code=True,
    )

    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        target_attn = model.model.language_model.layers[args.enhance_layer_index].self_attn
    elif hasattr(model, 'language_model'):
        target_attn = model.language_model.layers[args.enhance_layer_index].self_attn
    else:
        raise ValueError('Could not resolve Qwen3 attention layer')

    head_dim = target_attn.head_dim
    print(f"Layer {args.enhance_layer_index}: num_heads={num_heads}, head_dim={head_dim}")

    orig_o_proj_forward = target_attn.o_proj.forward
    per_head_buffer = {}

    def hooked_o_proj(input_tensor):
        bsz, q_len, hidden_size = input_tensor.shape
        per_head = input_tensor.view(bsz, q_len, num_heads, head_dim)
        per_head_last = per_head[:, -1, :, :]

        call_idx = per_head_buffer.get('_call_count', 0)
        if call_idx == 0:
            per_head_buffer['real'] = per_head_last.detach().cpu().float()
        elif call_idx == 1:
            per_head_buffer['cd'] = per_head_last.detach().cpu().float()
        per_head_buffer['_call_count'] = call_idx + 1

        return orig_o_proj_forward(input_tensor)

    target_attn.o_proj.forward = hooked_o_proj

    coco_images = sorted([f for f in os.listdir(args.coco_dir) if f.endswith('.jpg')])[:args.n_generative]

    with open(args.pope_file) as f:
        pope_data = [json.loads(line) for line in f][:args.n_yesno]

    gen_query = "Please describe this image in detail."

    print(f"\n=== Running {len(coco_images)} generative + {len(pope_data)} yes/no prompts ===\n")

    all_results = []

    def run_prompt(image, prompt_text):
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': image},
                    {'type': 'text', 'text': prompt_text},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors='pt',
        ).to(model.device)

        per_head_buffer.clear()
        per_head_buffer['_call_count'] = 0

        with torch.inference_mode():
            model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                use_only=True,
                enhance_layer_index=args.enhance_layer_index,
                ritual_alpha_pos=3.0,
                ritual_alpha_neg=1.0,
                ritual_beta=0.1,
                js_gamma=0.1,
            )

        if 'real' in per_head_buffer and 'cd' in per_head_buffer:
            real_ph = per_head_buffer['real'][0]
            cd_ph = per_head_buffer['cd'][0]
            delta_ph = real_ph - cd_ph
            return {
                'delta_norms': delta_ph.norm(dim=-1).numpy().tolist(),
                'real_norms': real_ph.norm(dim=-1).numpy().tolist(),
                'cd_norms': cd_ph.norm(dim=-1).numpy().tolist(),
            }
        return None

    for img_file in tqdm(coco_images, desc="Generative"):
        img_path = os.path.join(args.coco_dir, img_file)
        raw_image = Image.open(img_path).convert("RGB")

        result = run_prompt(raw_image, gen_query)
        if result:
            result['type'] = 'generative'
            result['image'] = img_file
            all_results.append(result)

    for item in tqdm(pope_data, desc="Yes/No"):
        img_file = item["image"]
        question = item["text"]

        img_path = os.path.join(args.coco_dir, img_file)
        if not os.path.exists(img_path):
            continue
        raw_image = Image.open(img_path).convert("RGB")

        result = run_prompt(raw_image, question)
        if result:
            result['type'] = 'yesno'
            result['image'] = img_file
            result['question'] = question
            all_results.append(result)

    target_attn.o_proj.forward = orig_o_proj_forward

    C_np = C.numpy()

    gen_deltas = np.array([r['delta_norms'] for r in all_results if r['type'] == 'generative'])
    yn_deltas = np.array([r['delta_norms'] for r in all_results if r['type'] == 'yesno'])

    print(f"\n{'='*60}")
    print(f"MECHANISTIC ANALYSIS RESULTS (Qwen3-VL)")
    print(f"{'='*60}")
    print(f"Generative prompts: {len(gen_deltas)}")
    print(f"Yes/No prompts:     {len(yn_deltas)}")
    print(f"C-scores (L{args.enhance_layer_index}): min={C_np.min():.3f} mean={C_np.mean():.3f} max={C_np.max():.3f}")
    print(f"Nonzero C heads: {np.where(C_np > 0)[0].tolist()}")

    gen_mean = gen_deltas.mean(axis=0) if len(gen_deltas) > 0 else np.zeros(num_heads)
    yn_mean = yn_deltas.mean(axis=0) if len(yn_deltas) > 0 else np.zeros(num_heads)

    print(f"\n--- Per-head mean correction strength (||real - CD||) ---")
    print(f"{'Head':>4}  {'C-score':>7}  {'Gen delta':>9}  {'YN delta':>9}")
    for h in range(num_heads):
        print(f"{h:4d}  {C_np[h]:7.3f}  {gen_mean[h]:9.4f}  {yn_mean[h]:9.4f}")

    if len(gen_deltas) > 0:
        r_gen, p_gen = stats.pearsonr(C_np, gen_mean)
        rho_gen, p_rho_gen = stats.spearmanr(C_np, gen_mean)
    else:
        r_gen, p_gen, rho_gen, p_rho_gen = 0, 1, 0, 1

    if len(yn_deltas) > 0:
        r_yn, p_yn = stats.pearsonr(C_np, yn_mean)
        rho_yn, p_rho_yn = stats.spearmanr(C_np, yn_mean)
    else:
        r_yn, p_yn, rho_yn, p_rho_yn = 0, 1, 0, 1

    print(f"\n--- Correlation: C-score vs correction strength ---")
    print(f"Generative:  Pearson r={r_gen:.4f} (p={p_gen:.4f}), Spearman rho={rho_gen:.4f} (p={p_rho_gen:.4f})")
    print(f"Yes/No:      Pearson r={r_yn:.4f} (p={p_yn:.4f}), Spearman rho={rho_yn:.4f} (p={p_rho_yn:.4f})")

    high_c_mask = C_np > 0
    low_c_mask = C_np == 0
    n_high = high_c_mask.sum()
    n_low = low_c_mask.sum()

    if n_high > 0 and n_low > 0:
        gen_high_mean = gen_mean[high_c_mask].mean()
        gen_low_mean = gen_mean[low_c_mask].mean()
        yn_high_mean = yn_mean[high_c_mask].mean()
        yn_low_mean = yn_mean[low_c_mask].mean()

        gen_snr = gen_high_mean / (gen_low_mean + 1e-8)
        yn_snr = yn_high_mean / (yn_low_mean + 1e-8)

        print(f"\n--- Signal vs Noise (high-C vs low-C heads) ---")
        print(f"High-C heads ({n_high}): {np.where(high_c_mask)[0].tolist()}")
        print(f"Low-C heads  ({n_low}): {np.where(low_c_mask)[0].tolist()}")
        print(f"Generative:  high-C mean={gen_high_mean:.4f}, low-C mean={gen_low_mean:.4f}, SNR={gen_snr:.2f}")
        print(f"Yes/No:      high-C mean={yn_high_mean:.4f}, low-C mean={yn_low_mean:.4f}, SNR={yn_snr:.2f}")

    output = {
        'C_scores': C_np.tolist(),
        'scores_path': args.scores_path,
        'enhance_layer': args.enhance_layer_index,
        'num_generative': len(gen_deltas),
        'num_yesno': len(yn_deltas),
        'gen_mean_delta': gen_mean.tolist(),
        'yn_mean_delta': yn_mean.tolist(),
        'correlation': {
            'gen_pearson_r': float(r_gen), 'gen_pearson_p': float(p_gen),
            'gen_spearman_rho': float(rho_gen), 'gen_spearman_p': float(p_rho_gen),
            'yn_pearson_r': float(r_yn), 'yn_pearson_p': float(p_yn),
            'yn_spearman_rho': float(rho_yn), 'yn_spearman_p': float(p_rho_yn),
        },
        'per_prompt': all_results,
    }
    out_path = os.path.join(args.output_dir, 'mechanistic_results_qwen3.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
