"""
Ablation: TVER head selection + adaptive sharpening (instead of EIC head set).

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

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

class TVERMonitor:
    """Per-step LOW-TVER head selection + adaptive sharpening sensor."""

    def __init__(self, model, layer_idx, img_start=35, img_len=576):
        self.layer = model.model.layers[layer_idx]
        self.num_heads = self.layer.self_attn.config.num_attention_heads
        self.img_start = img_start
        self.img_len = img_len
        self.grounding_score = 1.0
        log.info(f"[TVER-select] layer={layer_idx}  num_heads={self.num_heads}  "
                 f"img_tokens=[{img_start}:{img_start+img_len}]")

    def install_hook(self):
        layer = self.layer
        monitor = self
        original_forward = layer.self_attn.forward

        def hooked_forward(*args, **kwargs):
            kw = dict(kwargs); kw["output_attentions"] = True
            out = original_forward(*args, **kw)
            if len(out) >= 2 and out[1] is not None:
                attn = out[1]
                B, H, Q, KV = attn.shape
                if KV > monitor.img_start + monitor.img_len:
                    img_end = monitor.img_start + monitor.img_len
                    last_q = attn[:, :, -1, :]

                    text_mask = torch.ones(KV, dtype=torch.bool, device=attn.device)
                    text_mask[monitor.img_start:img_end] = False
                    text_mask[0] = False
                    img_mask = torch.zeros(KV, dtype=torch.bool, device=attn.device)
                    img_mask[monitor.img_start:img_end] = True

                    aT = last_q[:, :, text_mask].float()
                    aV = last_q[:, :, img_mask].float()
                    aT = torch.where(
                        aT > aT.mean(-1, keepdim=True) + aT.std(-1, keepdim=True),
                        torch.zeros_like(aT), aT,
                    )
                    aV = torch.where(
                        aV > aV.mean(-1, keepdim=True) + aV.std(-1, keepdim=True),
                        torch.zeros_like(aV), aV,
                    )
                    aT = aT / aT.sum(-1, keepdim=True).clamp_min(1e-8)
                    aV = aV / aV.sum(-1, keepdim=True).clamp_min(1e-8)
                    HT = -(aT * (aT + 1e-10).log()).sum(-1)
                    HV = -(aV * (aV + 1e-10).log()).sum(-1)
                    tver = (HT / HV.clamp_min(1e-8)).squeeze(0)

                    low_tver_mask = tver < tver.mean()
                    if not low_tver_mask.any():
                        monitor.grounding_score = 1.0
                        return out
                    head_idx = torch.where(low_tver_mask)[0]

                    img_attn = attn[:, head_idx, -1, monitor.img_start:img_end]
                    img_sum = img_attn.sum(-1, keepdim=True).clamp_min(1e-8)
                    img_norm = img_attn / img_sum
                    ent = -(img_norm * (img_norm + 1e-10).log()).sum(-1)
                    max_ent = torch.log(torch.tensor(float(monitor.img_len),
                                                     device=ent.device))
                    norm_ent = ent / max_ent
                    monitor.grounding_score = float(1.0 - norm_ent.mean().item())
            return out

        layer.self_attn.forward = hooked_forward
        return original_forward

    def restore(self, original_forward):
        self.layer.self_attn.forward = original_forward

class AdaptiveSharpenProcessor:
    def __init__(self, monitor, alpha=0.7):
        self.monitor = monitor
        self.alpha = alpha

    def __call__(self, input_ids, scores):
        gs = self.monitor.grounding_score
        t = max(1.0 - self.alpha * (1.0 - gs), 0.3)
        if t < 0.99:
            scores = scores / t
        return scores

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--anno_path", type=str, required=True)
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--num_eval_samples", type=int, default=500)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--layer_index", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--img_start", type=int, default=35)
    p.add_argument("--img_len", type=int, default=576)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--method_name", type=str, default="tver_head_select")
    return p.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    import random, numpy as np
    random.seed(args.seed); np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    vt = model.get_vision_tower()
    if not vt.is_loaded:
        vt.load_model()
    vt.to(device=model.device, dtype=torch.float16)
    image_processor = vt.image_processor
    evolve_only_sampling()

    monitor = TVERMonitor(model, args.layer_index,
                              img_start=args.img_start, img_len=args.img_len)
    orig_fwd = monitor.install_hook()
    processor = AdaptiveSharpenProcessor(monitor, alpha=args.alpha)

    with open(args.anno_path) as f:
        coco = json.load(f)
    images = coco["images"]
    import random as _rng
    _rng.seed(args.seed); _rng.shuffle(images)
    images = images[:args.num_eval_samples]

    os.makedirs(args.out_path, exist_ok=True)
    out_file = os.path.join(args.out_path, f"{args.method_name}.jsonl")
    results = []

    log.info("Start eval (TVER head selector + adaptive sharpening) ...")
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

        from transformers.generation.logits_process import LogitsProcessorList
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            use_only=False,
            enhance_layer_index=args.layer_index,
            logits_processor=LogitsProcessorList([processor]),
        )
        if isinstance(output_ids, tuple):
            output_ids = output_ids[0]
        output_text = tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:], skip_special_tokens=True,
        )[0].strip()
        results.append({"image_id": img_info["id"], "caption": output_text})

    monitor.restore(orig_fwd)
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    log.info(f"[done] Saved {len(results)} captions to {out_file}")

if __name__ == "__main__":
    main()
