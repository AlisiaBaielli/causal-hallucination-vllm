"""
POPE Evaluation for LLaVA-v1.5-7B.
"""
import os, sys, json, argparse, logging, warnings, random
from pathlib import Path
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "transformers" / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))
_self_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _self_dir]

import torch
import numpy as np
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model import LlavaLlamaForCausalLM
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList

from causal_core.eval_common import load_c_scores, resolve_method
from causal_core.models.llava_sampling import evolve_only_sampling
from causal_core.monitor import CausalMonitor, CausalLogitsProcessor
from causal_core.only_eic import inject_eic_for_only
from causal_core.vcd import add_diffusion_noise

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

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
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True,
                   help="COCO val2014 directory")
    p.add_argument("--pope_path", type=str, required=True,
                   help="POPE jsonl (e.g. coco_pope_random.json)")
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--do_sample", action="store_true", default=True)
    p.add_argument("--conv_mode", type=str, default="llava_v1")
    p.add_argument("--c_scores_path", type=str, default=None,
                   help="Required for CHALL (default); ignored if --no_hook")
    p.add_argument("--layer_index", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--img_start", type=int, default=35)
    p.add_argument("--img_len", type=int, default=576)
    p.add_argument("--type", type=str, default="random")
    p.add_argument("--dataset_name", type=str, default="coco")
    p.add_argument("--no_hook", action="store_true",
                   help="Vanilla baseline (no monitor, no processor)")
    p.add_argument("--use_only", action="store_true", help="ONLY baseline")
    p.add_argument("--use_eic_heads", action="store_true",
                   help="With --use_only: offline EIC heads in CD branch (ONLY+EIC)")
    p.add_argument("--use_vcd", action="store_true", help="VCD baseline")
    p.add_argument("--use_m3id", action="store_true", help="M3ID baseline")
    p.add_argument("--noise_step", type=int, default=500,
                   help="VCD diffusion noise step")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.no_hook:
        method = "vanilla"
    elif args.use_only:
        method = "only_eic" if args.use_eic_heads else "only"
    elif args.use_vcd:
        method = "vcd"
    elif args.use_m3id:
        method = "m3id"
    else:
        method = "chall"
        if args.c_scores_path is None:
            raise ValueError("--c_scores_path is required for CHALL (default mode)")

    needs_scores = method in ("chall", "only_eic")
    if needs_scores and args.c_scores_path is None:
        raise ValueError(f"--c_scores_path is required for method={method}")

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
    processors = LogitsProcessorList([])
    layer_for_only = args.layer_index
    orig_fwd = None

    if needs_scores:
        payload_scores = load_c_scores(args.c_scores_path, args.layer_index)
        log.info(f"C-scores: nonzero={int((payload_scores > 0).sum())}/{len(payload_scores)}")

        if method == "chall":
            monitor = CausalMonitor(model, args.layer_index, payload_scores,
                                   img_start=args.img_start, img_len=args.img_len)
            orig_fwd = monitor.install_qk_hook()
            causal_processor = CausalLogitsProcessor(monitor, alpha=args.alpha)
            processors = LogitsProcessorList([causal_processor])
        elif method == "only_eic":
            layer_for_only = inject_eic_for_only(
                model=model,
                scores_path=args.c_scores_path,
                layer_index=args.layer_index,
                pure_eic=True,
            )

    pope_entries = [json.loads(l) for l in open(args.pope_path)]
    log.info(f"POPE: {len(pope_entries)} questions, "
             f"{args.dataset_name}/{args.type}, method={method}, alpha={args.alpha}")

    os.makedirs(args.out_path, exist_ok=True)
    pred_list, label_list = [], []

    for entry in tqdm(pope_entries, total=len(pope_entries)):
        image_file = entry["image"]
        question = entry["text"]
        label = 1 if entry["label"].lower() == "yes" else 0

        img_path = os.path.join(args.data_path, image_file)
        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image_tensor = image_tensor.unsqueeze(0).half().to(model.device)

        qs = DEFAULT_IMAGE_TOKEN + "\nAnswer yes or no only. " + question
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(model.device)

        image_neg = None
        if method == "vcd":
            image_neg = add_diffusion_noise(image_tensor, args.noise_step)

        gen_kwargs = dict(
            images=image_tensor,
            images_neg=image_neg,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            use_only=(method in ("only", "only_eic")),
            use_vcd=(method == "vcd"),
            use_m3id=(method == "m3id"),
            enhance_layer_index=layer_for_only,
        )
        if processors and len(processors) > 0:
            gen_kwargs["logits_processor"] = processors

        with torch.inference_mode():
            output_ids = model.generate(input_ids, **gen_kwargs)

        if isinstance(output_ids, tuple):
            output_ids = output_ids[0]
        gen_ids = output_ids[0, input_ids.shape[1]:]
        output_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        pred_list.append(recorder(output_text))
        label_list.append(label)

    acc, precision, recall, f1, yes_ratio = compute_metrics(pred_list, label_list)
    acc = round(acc * 100, 2)
    precision = round(precision * 100, 2)
    recall = round(recall * 100, 2)
    f1 = round(f1 * 100, 2)
    yes_ratio = round(yes_ratio * 100, 2)

    log.info(f"POPE {args.dataset_name}/{args.type} method={method} alpha={args.alpha}")
    log.info(f"Acc: {acc}, P: {precision}, R: {recall}, F1: {f1}, Yes%: {yes_ratio}")
    print(f"POPE_{args.type}: Acc={acc} P={precision} R={recall} F1={f1} Yes={yes_ratio}")

    result = {"dataset": args.dataset_name, "split": args.type,
              "method": method, "alpha": args.alpha,
              "accuracy": acc, "precision": precision, "recall": recall,
              "f1": f1, "yes_ratio": yes_ratio, "n_samples": len(pred_list)}
    out_file = os.path.join(
        args.out_path,
        f"pope_{args.dataset_name}_{args.type}_{method}_a{args.alpha}.json",
    )
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    if monitor is not None:
        monitor.restore(orig_fwd)

if __name__ == "__main__":
    main()
