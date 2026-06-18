"""
MMVP evaluation for LLaVA-v1.5-7B (vanilla and causal steering).
"""
import os, sys, json, argparse, logging, warnings, re, csv
from pathlib import Path
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]

_self_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _self_dir]
sys.path.insert(0, str(REPO / "transformers" / "src"))
sys.path.insert(0, str(REPO))

import torch
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download

from llava.model import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList

from causal_core.monitor import CausalMonitor, CausalLogitsProcessor
from causal_core.models.llava_sampling import evolve_only_sampling
from causal_core.vcd import add_diffusion_noise

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--model_path", type=str, default=str(REPO / "data/models/llava-v1.5-7b"))
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--c_scores_path", type=str, required=True)
    p.add_argument("--layer_index", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--img_start", type=int, default=35)
    p.add_argument("--img_len", type=int, default=576)
    p.add_argument("--max_new_tokens", type=int, default=20)
    p.add_argument("--method_name", type=str, default="chall")
    p.add_argument("--use_only", action="store_true",
                   help="Run ONLY baseline instead of CHALL.")
    p.add_argument("--only_alpha_pos", type=float, default=3.0)
    p.add_argument("--only_alpha_neg", type=float, default=1.0)
    p.add_argument("--only_beta", type=float, default=0.1)
    p.add_argument("--only_gamma", type=float, default=0.25)
    p.add_argument("--use_vcd", action="store_true",
                   help="Run VCD baseline (diffusion-noised negative image, contrastive decoding).")
    p.add_argument("--noise_step", type=int, default=500)
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
    if text.startswith("yes"):
        return None
    return None

def main():
    args = parse_args()
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    import random, numpy as np
    random.seed(args.seed); np.random.seed(args.seed)

    log.info(f"[load] model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()
    vt = model.get_vision_tower()
    if not vt.is_loaded:
        vt.load_model()
    vt.to(device=model.device, dtype=torch.float16)
    image_processor = vt.image_processor

    evolve_only_sampling()

    log.info("[data] downloading MMVP from HuggingFace")
    qcsv = hf_hub_download("MMVP/MMVP", "Questions.csv", repo_type="dataset")
    img_dir = snapshot_download("MMVP/MMVP", repo_type="dataset",
                                allow_patterns=["MMVP Images/*"])
    img_dir = Path(img_dir) / "MMVP Images"
    questions = list(csv.DictReader(open(qcsv)))
    if args.limit:
        questions = questions[:args.limit]
    log.info(f"[data] {len(questions)} questions, images in {img_dir}")

    payload = torch.load(args.c_scores_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        c_scores = payload.get("C", payload.get("scores", next(iter(payload.values()))))
    else:
        c_scores = payload
    if c_scores.dim() == 2:
        c_scores = c_scores[args.layer_index]
    c_scores = c_scores.float()

    causal_processor = None
    if args.use_only:
        log.info(f"[ONLY active] layer={args.layer_index} "
                 f"alpha_pos={args.only_alpha_pos} alpha_neg={args.only_alpha_neg} "
                 f"beta={args.only_beta} gamma={args.only_gamma}")
    elif args.use_vcd:
        log.info(f"[VCD active] noise_step={args.noise_step}")
    elif args.alpha > 0:
        monitor = CausalMonitor(model, args.layer_index, c_scores,
                               img_start=args.img_start, img_len=args.img_len)
        monitor.install_qk_hook()
        causal_processor = CausalLogitsProcessor(monitor, alpha=args.alpha)
        log.info(f"[CHALL active] layer={args.layer_index} alpha={args.alpha}")
    else:
        log.info("[Vanilla mode]")

    os.makedirs(args.out_path, exist_ok=True)
    out_file = os.path.join(args.out_path, f"{args.method_name}.jsonl")

    results = []
    for row in tqdm(questions, desc=f"MMVP-{args.method_name}"):
        idx = int(row["Index"])
        question = row["Question"].strip()
        options = row["Options"].strip()
        gt = row["Correct Answer"].strip().lower().strip("()")

        img_path = img_dir / f"{idx}.jpg"
        image = Image.open(img_path).convert("RGB")
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image_tensor = image_tensor.unsqueeze(0).half().to(model.device)

        prompt_text = f"{question}\n{options}\nAnswer with the letter only."
        qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
        conv = conv_templates["v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(model.device)

        gen_kwargs = dict(
            images=image_tensor,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            max_new_tokens=args.max_new_tokens,
            use_only=args.use_only,
            use_vcd=args.use_vcd,
            enhance_layer_index=args.layer_index,
        )
        if args.use_only:
            gen_kwargs.update(dict(
                images_pos=None, images_neg=None,
                ritual_alpha_pos=args.only_alpha_pos,
                ritual_alpha_neg=args.only_alpha_neg,
                ritual_beta=args.only_beta,
                js_gamma=args.only_gamma,
            ))
        elif args.use_vcd:
            gen_kwargs.update(dict(
                images_pos=None,
                images_neg=add_diffusion_noise(image_tensor, args.noise_step),
            ))
        if causal_processor is not None:
            gen_kwargs["logits_processor"] = LogitsProcessorList([causal_processor])

        with torch.inference_mode():
            out = model.generate(input_ids, **gen_kwargs)
        output_ids = out[0] if isinstance(out, tuple) else out

        response = tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
        )[0].strip()

        pred = parse_letter(response)
        correct = (pred == gt)
        results.append(dict(
            index=idx, question=question, options=options,
            gt=gt, response=response, pred=pred, correct=correct,
        ))

    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    by_idx = {r["index"]: r for r in results}
    n = len(results)
    single_acc = sum(r["correct"] for r in results) / n

    pairs_correct = 0
    n_pairs = 0
    for i in range(1, 301, 2):
        if i in by_idx and (i + 1) in by_idx:
            n_pairs += 1
            if by_idx[i]["correct"] and by_idx[i + 1]["correct"]:
                pairs_correct += 1
    pair_acc = pairs_correct / n_pairs if n_pairs else 0

    log.info(f"\n=== MMVP {args.method_name} ===")
    log.info(f"Single-image accuracy: {single_acc*100:.2f}%  ({sum(r['correct'] for r in results)}/{n})")
    log.info(f"Pair accuracy:          {pair_acc*100:.2f}%  ({pairs_correct}/{n_pairs})")

    summary = dict(
        method=args.method_name, alpha=args.alpha, layer=args.layer_index,
        n=n, n_pairs=n_pairs,
        single_acc=single_acc, pair_acc=pair_acc,
        single_correct=sum(r["correct"] for r in results), pair_correct=pairs_correct,
    )
    with open(os.path.join(args.out_path, f"{args.method_name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
