"""
Ablation: EIC head selection + static head suppression, POPE.
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
from llava.conversation import Conversation, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from causal_core.models.llava_sampling import evolve_only_sampling
from pope_loader import POPEDataSet

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _eic_suppress_core import install_eic_suppress, restore_eic_suppress, load_c_scores

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)
evolve_only_sampling()

def recorder(out, pred_list):
    NEG_WORDS = ["No", "not", "no", "NO"]
    for line in out.split('\n'):
        line = line.replace('.', '').replace(',', '')
        words = line.split(' ')
        if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
            pred_list.append(0)
        else:
            pred_list.append(1)
        break
    return pred_list

def print_acc(pred_list, label_list):
    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == 1 and label == 1: TP += 1
        elif pred == 1 and label == 0: FP += 1
        elif pred == 0 and label == 0: TN += 1
        elif pred == 0 and label == 1: FN += 1
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    yes_ratio = pred_list.count(1) / len(pred_list) if pred_list else 0
    return acc, precision, recall, f1, yes_ratio

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--pope_path", type=str, required=True)
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--do_sample", type=bool, default=True)
    p.add_argument("--c_scores_path", type=str, required=True)
    p.add_argument("--layer_index", type=int, default=1)
    p.add_argument("--type", type=str, default="random")
    p.add_argument("--dataset_name", type=str, default="coco")
    args = p.parse_args()

    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed); np.random.seed(args.seed)

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name)

    c_scores = load_c_scores(args.c_scores_path, args.layer_index)
    orig_fwd, head_mask = install_eic_suppress(model, args.layer_index, c_scores)
    log.info(f"[EIC-suppress] layer={args.layer_index}  kept heads={int(head_mask.sum())}/{len(head_mask)}")

    pope_dataset = POPEDataSet(pope_path=args.pope_path, data_path=args.data_path,
                               trans=image_processor, model="llava")
    pope_loader = torch.utils.data.DataLoader(
        pope_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    pred_list, label_list = [], []
    log.info(f"POPE eval (EIC-suppress): {args.dataset_name}/{args.type}")
    for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):
        image = data["image"][0]
        qs = data["query"][0]
        label = data["label"]
        label_list += list(label)

        conv_out = Conversation(
            system="A chat between a curious human and an artificial intelligence assistant. "
                   "The assistant gives helpful, detailed, and polite answers to the human's questions.",
            roles=("USER", "ASSISTANT"),
            version="v1", messages=[], offset=0,
            sep_style=SeparatorStyle.TWO, sep=" ", sep2="</s>",
        )
        qu_out = DEFAULT_IMAGE_TOKEN + '\n' + qs
        conv_out.append_message(conv_out.roles[0], qu_out)
        conv_out.append_message(conv_out.roles[1], None)
        prompt_out = conv_out.get_prompt()
        input_ids = tokenizer_image_token(prompt_out, tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors='pt').unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image.unsqueeze(0).half().cuda(),
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                use_only=False,
                enhance_layer_index=args.layer_index,
            )
        if isinstance(output_ids, tuple):
            output_ids = output_ids[0]

        outputs = tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
        )[0].strip()
        if outputs.endswith("</s>"):
            outputs = outputs[:-4].strip()
        pred_list = recorder(outputs, pred_list)

    acc, precision, recall, f1, yes_ratio = print_acc(pred_list, label_list)
    acc = round(acc * 100, 2); precision = round(precision * 100, 2)
    recall = round(recall * 100, 2); f1 = round(f1 * 100, 2); yes_ratio = round(yes_ratio * 100, 2)
    log.info(f"POPE {args.dataset_name}/{args.type} (EIC-suppress)")
    log.info(f"Acc: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}, Yes%: {yes_ratio}")

    os.makedirs(args.out_path, exist_ok=True)
    result = {"dataset": args.dataset_name, "split": args.type, "intervention": "eic_suppress",
              "accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
              "yes_ratio": yes_ratio, "n_samples": len(pred_list)}
    out_file = os.path.join(args.out_path, f"pope_{args.dataset_name}_{args.type}_eic_suppress.json")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Saved to {out_file}")
    print(f"POPE_{args.type}: Acc={acc} P={precision} R={recall} F1={f1} Yes={yes_ratio}")
    restore_eic_suppress(model, args.layer_index, orig_fwd)

if __name__ == "__main__":
    main()
