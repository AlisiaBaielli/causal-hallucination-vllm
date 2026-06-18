"""
MMBench evaluation for Qwen3-VL-8B-Instruct (vanilla / CHALL / ONLY / VCD / M3ID).
"""
import argparse, os, json, math, re, io, base64, logging, sys
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
from tqdm import tqdm
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
_self_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _self_dir]
sys.path.insert(0, str(REPO / "transformers" / "src"))
sys.path.insert(0, str(REPO))

from causal_core.transformers_fork import ensure_qwen3_vl_fork
ensure_qwen3_vl_fork()

import torch
from transformers import AutoProcessor
from transformers.generation.logits_process import LogitsProcessorList
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

from causal_core.models.qwen3 import evolve_only_sampling_qwen3
from causal_core.monitor import CausalMonitorQwen3, CausalLogitsProcessor

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

all_options = ['A', 'B', 'C', 'D']

def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() in ('nan', 'none'):
        return True
    return False

def get_options(row, options):
    parsed = []
    for opt in options:
        val = row[opt]
        if is_none(val):
            break
        parsed.append(val)
    return parsed

def load_image_from_base64(image_str):
    return Image.open(io.BytesIO(base64.b64decode(image_str))).convert("RGB")

def extract_answer(text):
    text = text.strip()
    if text and text[0] in "ABCD":
        return text[0]
    m = re.search(r'(?:answer|option)\s*(?:is|:)?\s*([ABCD])', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    for c in text:
        if c in "ABCD":
            return c
    return None

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str,
                   default=str(REPO / "data/models/Qwen3-VL-8B-Instruct"))
    p.add_argument("--question_file", type=str,
                   default=str(REPO / "data/mmbench/mmbench_dev_20230712.tsv"))
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--all_rounds", action="store_true", default=True,
                   help="CircularEval: test all option rotations")
    p.add_argument("--single_pred_prompt", action="store_true", default=True)
    p.add_argument("--method", type=str, required=True,
                   choices=["vanilla", "only", "only_eic", "chall", "vcd", "m3id"])
    p.add_argument("--c_scores_path", type=str, default=str(REPO / "scores/qwen3_eic.pt"))
    p.add_argument("--noise_step", type=int, default=500)
    p.add_argument("--cd_alpha", type=float, default=1.0)
    p.add_argument("--cd_beta", type=float, default=0.1)
    p.add_argument("--layer_index", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of questions (validation only; default = full set).")
    return p.parse_args()

def main():
    args = parse_args()
    import random, numpy as np
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    random.seed(42); np.random.seed(42)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype="auto", device_map="auto", trust_remote_code=True,
    )
    model.eval()
    image_token_id = model.config.image_token_id

    evolve_only_sampling_qwen3()

    use_only = args.method in ("only", "only_eic")
    use_vcd = args.method == "vcd"
    use_m3id = args.method == "m3id"

    if args.method == "only_eic":
        from causal_core.only_eic import inject_eic_for_only
        inject_eic_for_only(model=model, scores_path=args.c_scores_path,
                            layer_index=args.layer_index, pure_eic=False, require_match=False)

    monitor = None
    processors = LogitsProcessorList()
    if args.method == "chall":
        payload = torch.load(args.c_scores_path, map_location="cpu")
        c_scores = payload.get("C", payload.get("scores", next(iter(payload.values()))))
        if c_scores.dim() == 2:
            c_scores = c_scores[args.layer_index]
        c_scores = c_scores.float()
        monitor = CausalMonitorQwen3(model, args.layer_index, c_scores, image_token_id)
        monitor.install_hook()
        processors = LogitsProcessorList([CausalLogitsProcessor(monitor, alpha=args.alpha)])

    questions = pd.read_table(os.path.expanduser(args.question_file))
    if args.limit:
        questions = questions.head(args.limit)
    log.info(f"Loaded {len(questions)} MMBench questions, method={args.method}")

    os.makedirs(args.out_dir, exist_ok=True)
    results = []

    for index, row in tqdm(questions.iterrows(), total=len(questions),
                           desc=f"MMBench ({args.method})"):
        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]
        num_rounds = len(options) if args.all_rounds else 1

        for round_idx in range(num_rounds):
            idx = row['index']
            question = row['question']
            hint = row['hint']
            image = load_image_from_base64(row['image'])

            if not is_none(hint):
                question = hint + '\n' + question
            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            if args.single_pred_prompt:
                question = question + '\n' + "Answer with the option's letter from the given choices directly."

            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]}]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)

            if monitor is not None:
                monitor.set_img_positions(inputs.input_ids, image_token_id)

            with torch.inference_mode():
                if use_vcd or use_m3id:
                    from causal_core.eval_common import import_vcd_baseline
                    contrastive_generate, add_diffusion_noise = import_vcd_baseline("qwen3")
                    neg_inputs = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                                  for k, v in inputs.items()}
                    if use_vcd:
                        neg_inputs["pixel_values"] = add_diffusion_noise(inputs["pixel_values"], args.noise_step)
                    else:
                        neg_inputs["pixel_values"] = torch.zeros_like(inputs["pixel_values"])
                    out = contrastive_generate(
                        model, dict(inputs), neg_inputs,
                        max_new_tokens=args.max_new_tokens, do_sample=True,
                        temperature=1.0, top_p=1.0, cd_alpha=args.cd_alpha, cd_beta=args.cd_beta,
                    )
                else:
                    out = model.generate(
                        **inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
                        temperature=1.0, top_p=1.0,
                        use_only=use_only, enhance_layer_index=args.layer_index,
                        ritual_alpha_pos=3.0, ritual_alpha_neg=1.0, ritual_beta=0.1, js_gamma=0.1,
                        logits_processor=processors,
                    )
            output_ids = out[0] if isinstance(out, tuple) else out
            outputs = processor.batch_decode(
                output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
            )[0].strip()

            results.append({
                "question_id": int(idx),
                "round_id": round_idx,
                "text": outputs,
                "options": options,
                "option_char": cur_option_char,
            })

            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]

    if monitor is not None:
        monitor.restore()

    out_file = os.path.join(args.out_dir, f"mmbench_{args.method}_raw.jsonl")
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    gt_answers = {row['index']: row['answer'] for _, row in questions.iterrows()
                  if not is_none(row.get('answer'))}

    if gt_answers:
        preds_by_q = defaultdict(list)
        for r in results:
            pred = extract_answer(r["text"])
            round_id = r["round_id"]
            n_opts = len(r["option_char"])
            if pred and pred in all_options[:n_opts]:
                pred_idx = all_options.index(pred)
                orig_idx = (pred_idx + round_id) % n_opts
                orig_pred = all_options[orig_idx]
            else:
                orig_pred = pred
            preds_by_q[r["question_id"]].append(orig_pred)

        correct = total = 0
        cat_correct = Counter(); cat_total = Counter()
        for _, row in questions.iterrows():
            idx = row['index']
            gt = row.get('answer')
            cat = row.get('l2-category', row.get('category', 'unknown'))
            if is_none(gt) or idx not in preds_by_q:
                continue
            round_preds = preds_by_q[idx]
            all_correct = all(p == gt for p in round_preds)
            total += 1; cat_total[cat] += 1
            if all_correct:
                correct += 1; cat_correct[cat] += 1

        acc = correct / total * 100 if total > 0 else 0
        log.info(f"\n{'='*60}")
        log.info(f"  MMBench-Dev ({args.method}) — CircularEval Accuracy: {acc:.2f}% ({correct}/{total})")
        log.info(f"{'='*60}")

        summary = {
            "method": args.method, "accuracy": round(acc, 2),
            "correct": correct, "total": total,
            "per_category": {cat: round(cat_correct[cat]/cat_total[cat]*100, 2)
                             for cat in sorted(cat_total.keys())},
        }
        with open(os.path.join(args.out_dir, f"mmbench_{args.method}.json"), "w") as f:
            json.dump(summary, f, indent=2)
    else:
        log.info(f"No ground truth — saved {len(results)} predictions for submission")

    log.info(f"Saved to {out_file}")

if __name__ == "__main__":
    main()
