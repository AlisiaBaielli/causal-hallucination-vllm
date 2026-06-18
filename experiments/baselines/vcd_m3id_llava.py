"""VCD and M3ID baselines for LLaVA-v1.5.
"""
import os, sys, json, argparse, logging, warnings, time
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

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

def add_diffusion_noise(pixel_values, noise_step=500):
    num_steps = 1000
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alpha_bar = alphas_prod[noise_step].to(pixel_values.device, pixel_values.dtype)
    noise = torch.randn_like(pixel_values)
    return torch.sqrt(alpha_bar) * pixel_values + torch.sqrt(1 - alpha_bar) * noise

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_path", type=str, required=True)

    p.add_argument("--benchmark", type=str, required=True,
                   choices=["chair", "amber", "pope", "mme"])

    p.add_argument("--data_path", type=str, help="COCO val2014 images dir")
    p.add_argument("--anno_path", type=str, help="COCO instances annotation json")
    p.add_argument("--num_eval_samples", type=int, default=500)

    p.add_argument("--amber_query", type=str)
    p.add_argument("--amber_image_dir", type=str)

    p.add_argument("--method", type=str, required=True, choices=["vcd", "m3id"])
    p.add_argument("--noise_step", type=int, default=500,
                   help="Diffusion noise step for VCD")
    p.add_argument("--cd_alpha", type=float, default=1.0)
    p.add_argument("--cd_beta", type=float, default=0.1)

    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--do_sample", action="store_true", default=True)
    p.add_argument("--out_path", type=str, required=True)
    return p.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import random, numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

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

    use_vcd = (args.method == "vcd")
    use_m3id = (args.method == "m3id")

    if args.benchmark == "chair":
        results = run_chair(args, model, tokenizer, image_processor, use_vcd, use_m3id)
    elif args.benchmark == "amber":
        results = run_amber(args, model, tokenizer, image_processor, use_vcd, use_m3id)
    else:
        raise NotImplementedError(f"Benchmark {args.benchmark} not yet supported in this script")

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(args.out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    log.info(f"Saved {len(results)} results to {args.out_path}")

def _generate_one(model, tokenizer, image_processor, image_path, question,
                   args, use_vcd, use_m3id):
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    image_tensor = image_tensor.unsqueeze(0).half().to(model.device)

    image_neg = None
    if use_vcd:
        image_neg = add_diffusion_noise(image_tensor, args.noise_step)
    elif use_m3id:
        pass

    qs = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output_ids, _ = model.generate(
            input_ids,
            images=image_tensor,
            images_pos=None,
            images_neg=image_neg,
            do_sample=args.do_sample,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            use_vcd=use_vcd,
            use_m3id=use_m3id,
            use_only=False,
            use_ritual=False,
            ritual_alpha_pos=3.0,
            ritual_alpha_neg=args.cd_alpha,
            ritual_beta=args.cd_beta,
            js_gamma=0.25,
            enhance_layer_index=0,
        )

    output_text = tokenizer.batch_decode(
        output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
    )[0].strip()
    return output_text

def run_chair(args, model, tokenizer, image_processor, use_vcd, use_m3id):
    with open(args.anno_path) as f:
        coco = json.load(f)
    images = coco["images"]

    import random as _rng
    _rng.seed(args.seed)
    _rng.shuffle(images)
    images = images[:args.num_eval_samples]

    results = []
    for img_info in tqdm(images, desc=f"CHAIR {args.method}"):
        img_path = os.path.join(args.data_path, img_info["file_name"])
        if not os.path.exists(img_path):
            continue
        caption = _generate_one(
            model, tokenizer, image_processor, img_path,
            "Please describe this image in detail.",
            args, use_vcd, use_m3id,
        )
        results.append({"image_id": img_info["id"], "caption": caption})
    return results

def run_amber(args, model, tokenizer, image_processor, use_vcd, use_m3id):
    with open(args.amber_query) as f:
        queries = json.load(f)

    results = []
    for item in tqdm(queries, desc=f"AMBER {args.method}"):
        if item.get("type") != "generative":
            continue
        img_path = os.path.join(args.amber_image_dir, item["image"])
        if not os.path.exists(img_path):
            continue
        answer = _generate_one(
            model, tokenizer, image_processor, img_path,
            item["query"],
            args, use_vcd, use_m3id,
        )
        result = {k: v for k, v in item.items()}
        result["response"] = answer
        results.append(result)
    return results

if __name__ == "__main__":
    main()
