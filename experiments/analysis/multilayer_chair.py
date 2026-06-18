"""
Multi-layer CHAIR eval for LLaVA.
"""
import os, sys, json, argparse, logging, warnings, random, time
from pathlib import Path
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]

_self_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _self_dir]

sys.path.insert(0, str(REPO / "transformers" / "src"))
sys.path.insert(0, str(REPO))

import torch
import numpy as np
from PIL import Image

from llava.model import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList
from causal_core.models.llava_sampling import evolve_only_sampling
from causal_core.monitor import CausalMonitor, CausalLogitsProcessor

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

class MultiLayerMonitor:
    def __init__(self, monitors):
        self.monitors = monitors

    @property
    def grounding_score(self):
        return sum(m.grounding_score for m in self.monitors) / len(self.monitors)

    @property
    def mean_entropy(self):
        return sum(m.mean_entropy for m in self.monitors) / len(self.monitors)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--model_path", type=str, default=str(REPO / "data/models/llava-v1.5-7b"))
    p.add_argument("--data_path", type=str, default=str(REPO / "data/coco/val2014"))
    p.add_argument("--anno_path", type=str,
                   default=str(REPO / "data/coco/annotations/instances_val2014.json"))
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--num_eval_samples", type=int, default=500)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--c_scores_paths", type=str, nargs="+", required=True)
    p.add_argument("--layer_indices", type=int, nargs="+", required=True)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--img_start", type=int, default=35)
    p.add_argument("--img_len", type=int, default=576)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    args = p.parse_args()

    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed); np.random.seed(args.seed)

    log.info(f"Loading LLaVA: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()
    vt = model.get_vision_tower()
    if not vt.is_loaded: vt.load_model()
    vt.to(device=model.device, dtype=torch.float16)
    image_processor = vt.image_processor

    evolve_only_sampling()

    monitors = []
    orig_fwds = []
    for c_path, layer_idx in zip(args.c_scores_paths, args.layer_indices):
        payload = torch.load(c_path, map_location="cpu", weights_only=False)
        c_scores = payload["C"].float()
        log.info(f"L={layer_idx}: nonzero={int((c_scores>0).sum())}/{c_scores.shape[0]}")
        m = CausalMonitor(model, layer_idx, c_scores,
                        img_start=args.img_start, img_len=args.img_len)
        orig_fwd = m.install_qk_hook()
        monitors.append(m)
        orig_fwds.append(orig_fwd)

    multi_monitor = MultiLayerMonitor(monitors)
    proc_chall = CausalLogitsProcessor(multi_monitor, alpha=args.alpha)

    with open(args.anno_path) as f:
        coco = json.load(f)
    images = coco["images"]
    random.shuffle(images)
    images = images[:args.num_eval_samples]

    os.makedirs(args.out_path, exist_ok=True)
    out_file = os.path.join(args.out_path, "chall_multilayer.jsonl")
    fout = open(out_file, "w")

    for img_info in tqdm(images, total=len(images)):
        img_id = img_info["id"]
        img_path = os.path.join(args.data_path, img_info["file_name"])
        if not os.path.exists(img_path):
            continue
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            image_tensor = image_tensor.unsqueeze(0).half().to(model.device)

            qs = DEFAULT_IMAGE_TOKEN + "\nPlease describe this image in detail."
            conv = conv_templates["v1"].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(model.device)

            output_ids = model.generate(
                input_ids, images=image_tensor,
                do_sample=True, temperature=args.temperature, top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                use_only=False, enhance_layer_index=args.layer_indices[0],
                logits_processor=LogitsProcessorList([proc_chall]),
            )
            if isinstance(output_ids, tuple):
                output_ids = output_ids[0]
            gen_ids = output_ids[0, input_ids.shape[1]:]
            caption = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            fout.write(json.dumps({"image_id": img_id, "caption": caption}) + "\n")
            fout.flush()
        except Exception as e:
            log.warning(f"Skip image {img_id}: {e}")
            continue

    fout.close()

    for m, fwd in zip(monitors, orig_fwds):
        m.restore(fwd)

    log.info(f"[done] Saved to {out_file}")

if __name__ == "__main__":
    main()
