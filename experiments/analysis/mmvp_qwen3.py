"""
MMVP evaluation for Qwen3-VL-8B-Instruct (vanilla / CHALL / ONLY / VCD / M3ID).
"""
import os, sys, json, argparse, logging, warnings, re, csv
from pathlib import Path
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
_self_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _self_dir]
sys.path.insert(0, str(REPO / "transformers" / "src"))
sys.path.insert(0, str(REPO))

from causal_core.transformers_fork import ensure_qwen3_vl_fork
ensure_qwen3_vl_fork()

import torch
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download

from transformers import AutoProcessor
from transformers.generation.logits_process import LogitsProcessorList
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

from causal_core.models.qwen3 import evolve_only_sampling_qwen3
from causal_core.monitor import CausalMonitorQwen3, CausalLogitsProcessor

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--model_path", type=str,
                   default=str(REPO / "data/models/Qwen3-VL-8B-Instruct"))
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--c_scores_path", type=str, required=True)
    p.add_argument("--layer_index", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--max_new_tokens", type=int, default=20)
    p.add_argument("--method_name", type=str, default="chall")
    p.add_argument("--use_only", action="store_true", help="ONLY baseline")
    p.add_argument("--use_eic_heads", action="store_true",
                   help="With --use_only: use offline EIC head set in the CD branch")
    p.add_argument("--use_vcd", action="store_true", help="VCD baseline")
    p.add_argument("--use_m3id", action="store_true", help="M3ID baseline")
    p.add_argument("--noise_step", type=int, default=500)
    p.add_argument("--cd_alpha", type=float, default=1.0)
    p.add_argument("--cd_beta", type=float, default=0.1)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of questions (validation only; default = full set).")
    return p.parse_args()

LETTER_RE = re.compile(r"\(?\s*([abAB])\s*\)?")

def parse_letter(text: str):
    """Extract (a)/(b) choice; fall back to first letter; return 'a'/'b' or None."""
    text = text.strip().lower()
    if text.startswith("(a)") or text.startswith("a)") or text.startswith("a."):
        return "a"
    if text.startswith("(b)") or text.startswith("b)") or text.startswith("b."):
        return "b"
    m = LETTER_RE.search(text)
    if m:
        return m.group(1).lower()
    return None

def main():
    args = parse_args()
    import random, numpy as np
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed); np.random.seed(args.seed)

    log.info(f"[load] model from {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype="auto", device_map="auto", trust_remote_code=True,
    )
    model.eval()
    image_token_id = model.config.image_token_id

    evolve_only_sampling_qwen3()

    payload = torch.load(args.c_scores_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        c_scores = payload.get("C", payload.get("scores", next(iter(payload.values()))))
    else:
        c_scores = payload
    if c_scores.dim() == 2:
        c_scores = c_scores[args.layer_index]
    c_scores = c_scores.float()

    if args.use_only and args.use_eic_heads:
        from causal_core.only_eic import inject_eic_for_only
        inject_eic_for_only(model=model, scores_path=args.c_scores_path,
                            layer_index=args.layer_index, pure_eic=False, require_match=False)
        args.method_name = "only_eic"
    elif args.use_only:
        args.method_name = "only"

    monitor = None
    processors = LogitsProcessorList([])
    if args.use_only or args.use_vcd or args.use_m3id:
        log.info(f"[{args.method_name}] no monitor")
    elif args.alpha > 0:
        monitor = CausalMonitorQwen3(model, args.layer_index, c_scores, image_token_id)
        monitor.install_hook()
        processors = LogitsProcessorList([CausalLogitsProcessor(monitor, alpha=args.alpha)])
        log.info(f"[CHALL] layer={args.layer_index} alpha={args.alpha}")
    else:
        log.info("[Vanilla]")

    log.info("[data] downloading MMVP from HuggingFace")
    qcsv = hf_hub_download("MMVP/MMVP", "Questions.csv", repo_type="dataset")
    img_dir = snapshot_download("MMVP/MMVP", repo_type="dataset",
                                allow_patterns=["MMVP Images/*"])
    img_dir = Path(img_dir) / "MMVP Images"
    questions = list(csv.DictReader(open(qcsv)))
    if args.limit:
        questions = questions[:args.limit]
    log.info(f"[data] {len(questions)} questions, images in {img_dir}")

    os.makedirs(args.out_path, exist_ok=True)
    out_file = os.path.join(args.out_path, f"{args.method_name}.jsonl")

    results = []
    for row in tqdm(questions, desc=f"MMVP-{args.method_name}"):
        idx = int(row["Index"])
        question = row["Question"].strip()
        options = row["Options"].strip()
        gt = row["Correct Answer"].strip().lower().strip("()")

        image = Image.open(img_dir / f"{idx}.jpg").convert("RGB")
        prompt_text = f"{question}\n{options}\nAnswer with the letter only."
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)

        if monitor is not None:
            monitor.set_img_positions(inputs.input_ids, image_token_id)

        with torch.inference_mode():
            if args.use_vcd or args.use_m3id:
                from causal_core.eval_common import import_vcd_baseline
                contrastive_generate, add_diffusion_noise = import_vcd_baseline("qwen3")
                neg_inputs = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                              for k, v in inputs.items()}
                if args.use_vcd:
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
                    use_only=bool(args.use_only), enhance_layer_index=args.layer_index,
                    ritual_alpha_pos=3.0, ritual_alpha_neg=1.0, ritual_beta=0.1, js_gamma=0.1,
                    logits_processor=processors,
                )
        output_ids = out[0] if isinstance(out, tuple) else out
        response = processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
        )[0].strip()

        pred = parse_letter(response)
        results.append(dict(index=idx, question=question, options=options,
                            gt=gt, response=response, pred=pred, correct=(pred == gt)))

    if monitor is not None:
        monitor.restore()

    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    by_idx = {r["index"]: r for r in results}
    n = len(results)
    single_acc = sum(r["correct"] for r in results) / n if n else 0.0
    pairs_correct = n_pairs = 0
    for i in range(1, 301, 2):
        if i in by_idx and (i + 1) in by_idx:
            n_pairs += 1
            if by_idx[i]["correct"] and by_idx[i + 1]["correct"]:
                pairs_correct += 1
    pair_acc = pairs_correct / n_pairs if n_pairs else 0.0

    log.info(f"\n=== MMVP {args.method_name} (Qwen3) ===")
    log.info(f"Single-image accuracy: {single_acc*100:.2f}%  ({sum(r['correct'] for r in results)}/{n})")
    log.info(f"Pair accuracy:          {pair_acc*100:.2f}%  ({pairs_correct}/{n_pairs})")

    summary = dict(method=args.method_name, alpha=args.alpha, layer=args.layer_index,
                   n=n, n_pairs=n_pairs, single_acc=single_acc, pair_acc=pair_acc,
                   single_correct=sum(r["correct"] for r in results), pair_correct=pairs_correct)
    with open(os.path.join(args.out_path, f"{args.method_name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
