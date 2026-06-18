"""
MME Evaluation for Qwen3-VL-8B.
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
from qwen_vl_utils import process_vision_info

from transformers import AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from causal_core.models.qwen3 import evolve_only_sampling_qwen3
from causal_core.monitor import CausalMonitorQwen3, CausalLogitsProcessor

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

evolve_only_sampling_qwen3()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_path", type=str,
                   default=str(REPO / "data/models/Qwen3-VL-8B-Instruct"))
    p.add_argument("--image_folder", type=str,
                   default=str(REPO / "data/MME/MME_Benchmark_release_version/MME_Benchmark"))
    p.add_argument("--question_file", type=str,
                   default=str(REPO / "data/MME/test_merged_final.jsonl"))
    p.add_argument("--answers_file", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--do_sample", type=bool, default=True)

    p.add_argument("--c_scores_path", type=str, required=True)
    p.add_argument("--layer_index", type=int, default=16)
    p.add_argument("--alpha", type=float, default=0.3)
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

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype="auto", device_map="auto", trust_remote_code=True)
    model.eval()

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

    from transformers.generation.logits_process import LogitsProcessorList
    if args.no_hook or args.use_only or getattr(args, "use_vcd", False) or getattr(args, "use_m3id", False):
        monitor = None
        processors = LogitsProcessorList([])
    else:
        monitor = CausalMonitorQwen3(model, args.layer_index, c_scores,
                                    image_token_id=model.config.image_token_id)
        causal_processor = CausalLogitsProcessor(monitor, alpha=args.alpha)
        processors = LogitsProcessorList([causal_processor])

    questions = [json.loads(q) for q in open(args.question_file)]
    log.info(f"MME: {len(questions)} questions, alpha={args.alpha}")

    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    ans_file = open(args.answers_file, "w")

    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["question"]

        img_path = os.path.join(args.image_folder, image_file)

        messages = [{"role": "user", "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": qs},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)

        if monitor is not None:
            monitor.set_img_positions(inputs["input_ids"], model.config.image_token_id)

        with torch.inference_mode():
            if getattr(args, "use_vcd", False) or getattr(args, "use_m3id", False):
                from causal_core.eval_common import import_vcd_baseline
                contrastive_generate, add_diffusion_noise = import_vcd_baseline("qwen3")
                neg_inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                if args.use_vcd:
                    neg_inputs["pixel_values"] = add_diffusion_noise(inputs["pixel_values"], args.noise_step)
                else:
                    neg_messages = [{"role": "user", "content": [{"type": "text", "text": qs}]}]
                    neg_inputs = processor.apply_chat_template(neg_messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
                output_ids = contrastive_generate(model, dict(inputs), neg_inputs, max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, temperature=args.temperature, top_p=args.top_p, cd_alpha=args.cd_alpha, cd_beta=args.cd_beta)
            else:
              output_ids = model.generate(
                **inputs,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                use_only=bool(args.use_only),
                enhance_layer_index=args.layer_index,
                logits_processor=processors,
            )

        gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        output_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": qs,
            "text": output_text,
            "model_id": "qwen3-vl-8b",
            "image": image_file,
            "metadata": {}
        }) + "\n")
        ans_file.flush()

    ans_file.close()
    if monitor is not None:
        monitor.restore()
    log.info(f"Saved answers to {args.answers_file}")

if __name__ == "__main__":
    main()
