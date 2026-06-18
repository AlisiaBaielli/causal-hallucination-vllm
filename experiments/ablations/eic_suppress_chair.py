"""
Ablation: EIC head selection + static head suppression (no adaptive sharpening), CHAIR.

"""
import os, sys, json, argparse, logging, warnings
from pathlib import Path
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "transformers" / "src"))
sys.path.insert(0, str(REPO))
_self_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _self_dir]

import torch
from PIL import Image

from llava.model import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from transformers import AutoTokenizer
from causal_core.models.llava_sampling import evolve_only_sampling

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _eic_suppress_core import install_eic_suppress, restore_eic_suppress, load_c_scores

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--anno_path", type=str, required=True)
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--num_eval_samples", type=int, default=500)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--c_scores_path", type=str, required=True)
    p.add_argument("--layer_index", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--method_name", type=str, default="eic_suppress")
    return p.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    import random, numpy as np
    random.seed(args.seed); np.random.seed(args.seed)

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

    c_scores = load_c_scores(args.c_scores_path, args.layer_index)
    orig_fwd, head_mask = install_eic_suppress(model, args.layer_index, c_scores)
    log.info(f"[EIC-suppress] layer={args.layer_index}  kept heads={int(head_mask.sum())}/{len(head_mask)}: "
             f"{torch.where(head_mask > 0)[0].tolist()}")

    with open(args.anno_path) as f:
        coco = json.load(f)
    images = coco["images"]
    import random as _rng
    _rng.seed(args.seed); _rng.shuffle(images)
    images = images[:args.num_eval_samples]

    os.makedirs(args.out_path, exist_ok=True)
    out_file = os.path.join(args.out_path, f"{args.method_name}.jsonl")
    results = []

    log.info("Start eval (EIC heads + static suppression) ...")
    for img_info in tqdm(images, total=len(images)):
        img_path = os.path.join(args.data_path, img_info["file_name"])
        if not os.path.exists(img_path):
            continue
        image = Image.open(img_path).convert("RGB")
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image_tensor = image_tensor.unsqueeze(0).half().to(model.device)
        qs = DEFAULT_IMAGE_TOKEN + "\nPlease describe this image in detail."
        conv = conv_templates["v1"].copy()
        conv.append_message(conv.roles[0], qs); conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt",
        ).unsqueeze(0).to(model.device)

        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            use_only=False,
            enhance_layer_index=args.layer_index,
        )
        if isinstance(output_ids, tuple):
            output_ids = output_ids[0]
        output_text = tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:], skip_special_tokens=True,
        )[0].strip()
        results.append({"image_id": img_info["id"], "caption": output_text})

    restore_eic_suppress(model, args.layer_index, orig_fwd)
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    log.info(f"[done] Saved {len(results)} captions to {out_file}")

if __name__ == "__main__":
    main()
