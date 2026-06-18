"""
AMBER evaluation for Qwen3-VL-8B.
"""
import os, sys, json, argparse, logging, warnings, random
from pathlib import Path
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from causal_core.transformers_fork import ensure_qwen3_vl_fork
ensure_qwen3_vl_fork()

import torch
import numpy as np
from PIL import Image

from transformers import AutoProcessor
from transformers.generation.logits_process import LogitsProcessorList
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from causal_core.models.qwen3 import evolve_only_sampling_qwen3

from causal_core.monitor import CausalMonitorQwen3, CausalLogitsProcessor

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def main():
    p = argparse.ArgumentParser(description="AMBER eval for Qwen3-VL")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_path", type=str,
                   default=str(REPO / "data/models/Qwen3-VL-8B-Instruct"))
    p.add_argument("--amber_query", type=str, required=True)
    p.add_argument("--amber_image_dir", type=str, required=True)
    p.add_argument("--output_file", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--c_scores_path", type=str, required=True)
    p.add_argument("--layer_index", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--do_sample", type=str2bool, default=True)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--no_hook", action="store_true",
                   help="Disable CHALL monitor (vanilla / ONLY / VCD / M3ID)")
    p.add_argument("--use_only", action="store_true", help="ONLY baseline")
    p.add_argument("--use_eic_heads", action="store_true",
                   help="With --use_only: use offline EIC head set in the CD branch")
    p.add_argument("--use_vcd", action="store_true", help="VCD baseline")
    p.add_argument("--use_m3id", action="store_true", help="M3ID baseline")
    p.add_argument("--noise_step", type=int, default=500)
    p.add_argument("--cd_alpha", type=float, default=1.0)
    p.add_argument("--cd_beta", type=float, default=0.1)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    proc_kwargs = dict(trust_remote_code=True)
    if args.max_pixels is not None:
        proc_kwargs["max_pixels"] = args.max_pixels
    processor = AutoProcessor.from_pretrained(args.model_path, **proc_kwargs)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype="auto", device_map="auto", trust_remote_code=True,
    )
    model.eval()
    image_token_id = model.config.image_token_id

    evolve_only_sampling_qwen3()

    payload = torch.load(args.c_scores_path, map_location="cpu")
    if isinstance(payload, dict):
        c_scores = payload.get("scores", payload.get("C", None))
        if c_scores is None:
            c_scores = next(iter(payload.values()))
    else:
        c_scores = payload
    if c_scores.dim() == 2:
        c_scores = c_scores[args.layer_index]
    c_scores = c_scores.float()
    log.info(f"C-scores: {c_scores.shape}, nonzero={int((c_scores>0).sum())}/{len(c_scores)}")

    if args.use_only and args.use_eic_heads:
        from causal_core.only_eic import inject_eic_for_only
        inject_eic_for_only(
            model=model, scores_path=args.c_scores_path,
            layer_index=args.layer_index, pure_eic=False, require_match=False,
        )

    if args.no_hook or args.use_only or getattr(args, 'use_vcd', False) or getattr(args, 'use_m3id', False):
        monitor = None
        processors = LogitsProcessorList([])
    else:
        monitor = CausalMonitorQwen3(model, args.layer_index, c_scores, image_token_id)
        monitor.install_hook()
        causal_processor = CausalLogitsProcessor(monitor, alpha=args.alpha)
        processors = LogitsProcessorList([causal_processor])

    with open(args.amber_query) as f:
        queries = json.load(f)
    log.info(f"Loaded {len(queries)} AMBER generative queries")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    results = []
    for item in tqdm(queries, desc="AMBER generative"):
        qid = item["id"]
        img_file = item["image"]
        query = item["query"]

        img_path = os.path.join(args.amber_image_dir, img_file)
        raw_image = Image.open(img_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": raw_image},
                    {"type": "text", "text": query},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        if monitor is not None:
            monitor.set_img_positions(inputs["input_ids"], image_token_id)

        try:
            with torch.inference_mode():
                if getattr(args, "use_vcd", False) or getattr(args, "use_m3id", False):
                    from causal_core.eval_common import import_vcd_baseline
                    contrastive_generate, add_diffusion_noise = import_vcd_baseline("qwen3")
                    neg_inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                    if args.use_vcd:
                        neg_inputs["pixel_values"] = add_diffusion_noise(inputs["pixel_values"], args.noise_step)
                    else:
                        neg_messages = [{"role": "user", "content": [{"type": "text", "text": query}]}]
                        neg_inputs = processor.apply_chat_template(
                            neg_messages, tokenize=True, add_generation_prompt=True,
                            return_dict=True, return_tensors="pt").to(model.device)
                    output_ids = contrastive_generate(
                        model, dict(inputs), neg_inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample, temperature=args.temperature, top_p=args.top_p,
                        cd_alpha=args.cd_alpha, cd_beta=args.cd_beta)
                else:
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        use_only=bool(args.use_only),
                        enhance_layer_index=args.layer_index,
                        logits_processor=processors,
                    )
        except torch.cuda.OutOfMemoryError:
            log.warning(f"[OOM] Skipping {img_file}, clearing cache")
            torch.cuda.empty_cache()
            results.append({"id": qid, "response": ""})
            continue

        gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        output_text = processor.batch_decode(
            [gen_ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        results.append({"id": qid, "response": output_text})

        if len(results) % 100 == 0:
            log.info(f"  [{len(results)}/{len(queries)}] "
                     f"Last: {output_text[:80]}...")

    if monitor is not None:
        monitor.restore()

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved {len(results)} results to {args.output_file}")

if __name__ == "__main__":
    main()
