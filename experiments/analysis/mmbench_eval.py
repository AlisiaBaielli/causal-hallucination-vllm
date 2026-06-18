"""
MMBench evaluation for LLaVA-v1.5-7B.
"""
import argparse, torch, os, json, math, re
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import Counter

REPO = Path(__file__).resolve().parents[2]

_self_dir = str(Path(__file__).resolve().parent)
import sys
sys.path = [p for p in sys.path if p != _self_dir]
sys.path.insert(0, str(REPO / "transformers" / "src"))
sys.path.insert(0, str(REPO))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model import LlavaLlamaForCausalLM
from llava.mm_utils import tokenizer_image_token, process_images
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList
from causal_core.models.llava_sampling import evolve_only_sampling
from causal_core.vcd import add_diffusion_noise

from PIL import Image
import io, base64, logging

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
    return Image.open(io.BytesIO(base64.b64decode(image_str)))

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
                   default=str(REPO / "data/models/llava-v1.5-7b"))
    p.add_argument("--question_file", type=str,
                   default=str(REPO / "data/mmbench/mmbench_dev_20230712.tsv"))
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--conv_mode", type=str, default="v1")
    p.add_argument("--all_rounds", action="store_true", default=True,
                   help="CircularEval: test all option rotations")
    p.add_argument("--single_pred_prompt", action="store_true", default=True)
    p.add_argument("--method", type=str, required=True,
                   choices=["vanilla", "only", "chall", "vcd", "m3id"])
    p.add_argument("--c_scores_path", type=str,
                   default=str(REPO / "scores/llava_eic.pt"))
    p.add_argument("--noise_step", type=int, default=500)
    p.add_argument("--layer_index", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--img_start", type=int, default=35)
    p.add_argument("--img_len", type=int, default=576)
    p.add_argument("--gamma", type=float, default=0.2)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--alpha_pos", type=float, default=3.0)
    p.add_argument("--alpha_neg", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of questions (validation only; default = full set).")
    return p.parse_args()

def main():
    args = parse_args()

    import random, numpy as np
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(42)
    np.random.seed(42)

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
    processors = LogitsProcessorList()
    use_only = (args.method == "only")

    if args.method == "chall":
        from causal_core.monitor import CausalMonitor, CausalLogitsProcessor
        payload = torch.load(args.c_scores_path, map_location="cpu")
        c_scores = payload.get("C", payload.get("scores", next(iter(payload.values()))))
        if c_scores.dim() == 2:
            c_scores = c_scores[args.layer_index]
        c_scores = c_scores.float()
        monitor = CausalMonitor(model, args.layer_index, c_scores,
                               img_start=args.img_start, img_len=args.img_len)
        orig_fwd = monitor.install_qk_hook()
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

            qs = question
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            if args.single_pred_prompt:
                qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
            ).unsqueeze(0).cuda()

            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_neg = add_diffusion_noise(image_tensor, args.noise_step) if args.method == "vcd" else None

            gen_kwargs = dict(
                images=image_tensor.unsqueeze(0).half().cuda(),
                images_pos=None,
                images_neg=(image_neg.unsqueeze(0).half().cuda() if image_neg is not None else None),
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                max_new_tokens=1024,
                use_cache=True,
                use_only=use_only,
                use_vcd=(args.method == "vcd"),
                use_m3id=(args.method == "m3id"),
                use_ritual=False,
                ritual_alpha_pos=args.alpha_pos,
                ritual_alpha_neg=args.alpha_neg,
                ritual_beta=args.beta,
                js_gamma=args.gamma,
                enhance_layer_index=args.layer_index,
            )

            if len(processors) > 0:
                gen_kwargs["logits_processor"] = processors

            with torch.inference_mode():
                out = model.generate(input_ids, **gen_kwargs)
                output_ids = out[0] if isinstance(out, tuple) else out

            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True
            )[0].strip()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)].strip()

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
        monitor.restore(orig_fwd)

    out_file = os.path.join(args.out_dir, f"mmbench_{args.method}_raw.jsonl")
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    gt_answers = {row['index']: row['answer'] for _, row in questions.iterrows()
                  if not is_none(row.get('answer'))}

    if gt_answers:
        from collections import defaultdict
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

        correct = 0
        total = 0
        cat_correct = Counter()
        cat_total = Counter()

        for _, row in questions.iterrows():
            idx = row['index']
            gt = row.get('answer')
            cat = row.get('l2-category', row.get('category', 'unknown'))
            if is_none(gt) or idx not in preds_by_q:
                continue

            round_preds = preds_by_q[idx]
            all_correct = all(p == gt for p in round_preds)

            total += 1
            cat_total[cat] += 1
            if all_correct:
                correct += 1
                cat_correct[cat] += 1

        acc = correct / total * 100 if total > 0 else 0
        log.info(f"\n{'='*60}")
        log.info(f"  MMBench-Dev ({args.method}) — CircularEval Accuracy: {acc:.2f}% ({correct}/{total})")
        log.info(f"{'='*60}")

        log.info("\nPer L2-category:")
        for cat in sorted(cat_total.keys()):
            c = cat_correct[cat]
            t = cat_total[cat]
            log.info(f"  {cat:45s}: {c/t*100:5.1f}% ({c}/{t})")

        summary = {
            "method": args.method,
            "accuracy": round(acc, 2),
            "correct": correct,
            "total": total,
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
