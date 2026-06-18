"""
POPE Evaluation for Qwen3-VL-8B.
"""
import os, sys, json, argparse, logging, warnings, random
from pathlib import Path
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

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

def recorder(text):
    NEG_WORDS = ["No", "not", "no", "NO"]
    line = text.split("\n")[0].replace(".", "").replace(",", "")
    words = line.split(" ")
    if any(w in NEG_WORDS for w in words) or any(w.endswith("n't") for w in words):
        return 0
    return 1

def compute_metrics(pred_list, label_list):
    TP = TN = FP = FN = 0
    for pred, label in zip(pred_list, label_list):
        if pred == 1 and label == 1: TP += 1
        elif pred == 1 and label == 0: FP += 1
        elif pred == 0 and label == 0: TN += 1
        elif pred == 0 and label == 1: FN += 1
    total = TP + TN + FP + FN
    acc = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    yes_ratio = pred_list.count(1) / len(pred_list) if len(pred_list) > 0 else 0
    return acc, precision, recall, f1, yes_ratio

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_path", type=str,
                   default=str(REPO / "data/models/Qwen3-VL-8B-Instruct"))
    p.add_argument("--data_path", type=str,
                   default=str(REPO / "data/coco/val2014"))
    p.add_argument("--pope_path", type=str, required=True)
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--do_sample", type=bool, default=True)
    p.add_argument("--c_scores_path", type=str, required=True)
    p.add_argument("--layer_index", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--type", type=str, default="random")
    p.add_argument("--dataset_name", type=str, default="coco")
    p.add_argument("--no_hook", action="store_true",
                   help="Disable causal hook (vanilla run through same script)")
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
        log.info("NO MONITOR mode — vanilla / ONLY / VCD / M3ID")
        monitor = None
        processors = LogitsProcessorList([])
    else:
        monitor = CausalMonitorQwen3(model, args.layer_index, c_scores,
                                    image_token_id=model.config.image_token_id)
        causal_processor = CausalLogitsProcessor(monitor, alpha=args.alpha)
        processors = LogitsProcessorList([causal_processor])

    pope_entries = [json.loads(l) for l in open(args.pope_path)]
    log.info(f"POPE: {len(pope_entries)} questions, {args.dataset_name}/{args.type}, alpha={args.alpha}")

    os.makedirs(args.out_path, exist_ok=True)
    pred_list, label_list = [], []

    for entry in tqdm(pope_entries, total=len(pope_entries)):
        image_file = entry["image"]
        question = entry["text"]
        label = 1 if entry["label"].lower() == "yes" else 0

        img_path = os.path.join(args.data_path, image_file)
        if not os.path.exists(img_path):
            continue

        messages = [{"role": "user", "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": "Answer yes or no only. " + question},
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
                    neg_messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
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

        pred = recorder(output_text)
        pred_list.append(pred)
        label_list.append(label)

    acc, precision, recall, f1, yes_ratio = compute_metrics(pred_list, label_list)
    acc = round(acc * 100, 2)
    precision = round(precision * 100, 2)
    recall = round(recall * 100, 2)
    f1 = round(f1 * 100, 2)
    yes_ratio = round(yes_ratio * 100, 2)

    log.info(f"POPE {args.dataset_name}/{args.type} alpha={args.alpha}")
    log.info(f"Acc: {acc}, P: {precision}, R: {recall}, F1: {f1}, Yes%: {yes_ratio}")
    print(f"POPE_{args.type}: Acc={acc} P={precision} R={recall} F1={f1} Yes={yes_ratio}")

    result = {"dataset": args.dataset_name, "split": args.type, "alpha": args.alpha,
              "accuracy": acc, "precision": precision, "recall": recall,
              "f1": f1, "yes_ratio": yes_ratio, "n_samples": len(pred_list)}
    out_file = os.path.join(args.out_path, f"pope_{args.dataset_name}_{args.type}_a{args.alpha}.json")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    if monitor is not None:
        monitor.restore()

if __name__ == "__main__":
    main()
