"""
Multi-Layer Grounding Score Analysis
"""
import os, sys, json, argparse, logging, warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

_self_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _self_dir]

sys.path.insert(0, str(REPO / "transformers" / "src"))
sys.path.insert(0, str(REPO))

from tqdm import tqdm
import torch
import numpy as np
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

class MultiLayerMonitor:
    def __init__(self, model, layer_indices, c_scores, img_start=35, img_len=576):
        self.layer_indices = layer_indices
        self.img_start = img_start
        self.img_len = img_len

        C = c_scores.float()
        self.high_c_mask = (C > 0)
        self.high_c_indices = torch.where(self.high_c_mask)[0]
        self.c_weights = C[self.high_c_mask]
        self.c_weights = self.c_weights / (self.c_weights.sum() + 1e-8)

        self.layers = {idx: model.model.layers[idx] for idx in layer_indices}
        self.original_forwards = {}

        self._current = {idx: [] for idx in layer_indices}
        self.all_samples = []

    def install(self):
        for idx in self.layer_indices:
            layer = self.layers[idx]
            original_forward = layer.self_attn.forward
            self.original_forwards[idx] = original_forward

            monitor = self
            layer_idx = idx

            def make_hook(orig_fwd, lidx):
                def hooked_forward(*args, **kwargs):
                    kwargs_copy = dict(kwargs)
                    kwargs_copy["output_attentions"] = True
                    result = orig_fwd(*args, **kwargs_copy)

                    if len(result) >= 2 and result[1] is not None:
                        attn_weights = result[1]
                        B, H, Q, KV = attn_weights.shape

                        if Q == 1 and KV > monitor.img_start + monitor.img_len:
                            img_end = monitor.img_start + monitor.img_len
                            img_attn = attn_weights[:, monitor.high_c_indices.to(attn_weights.device), :,
                                                    monitor.img_start:img_end]
                            img_attn = img_attn.squeeze(2)
                            img_attn_sum = img_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                            img_attn_norm = img_attn / img_attn_sum
                            entropy = -(img_attn_norm * (img_attn_norm + 1e-10).log()).sum(dim=-1)
                            max_entropy = torch.log(torch.tensor(float(monitor.img_len)))
                            norm_entropy = entropy / max_entropy
                            c_w = monitor.c_weights.to(norm_entropy.device)
                            mean_norm_entropy = (norm_entropy * c_w.unsqueeze(0)).sum(dim=-1).mean().item()
                            gs = 1.0 - mean_norm_entropy
                            monitor._current[lidx].append(round(gs, 6))

                    return result
                return hooked_forward

            layer.self_attn.forward = make_hook(original_forward, idx)

    def start_sample(self):
        self._current = {idx: [] for idx in self.layer_indices}

    def finish_sample(self):
        self.all_samples.append({str(idx): list(self._current[idx]) for idx in self.layer_indices})

    def restore(self):
        for idx in self.layer_indices:
            self.layers[idx].self_attn.forward = self.original_forwards[idx]

def identify_hal(caption, gt_objects):
    SYNONYMS = {
        "person": {"man","woman","boy","girl","child","people","person","kid","lady","player","rider"},
        "car": {"car","vehicle","automobile"}, "dog": {"dog","puppy"}, "cat": {"cat","kitten"},
        "chair": {"chair","seat"}, "dining table": {"table","desk"}, "tv": {"tv","television","monitor","screen"},
        "couch": {"couch","sofa"}, "bed": {"bed"}, "bottle": {"bottle"}, "cup": {"cup","mug"},
        "bowl": {"bowl"}, "laptop": {"laptop","computer"}, "cell phone": {"phone","cellphone"},
        "book": {"book"}, "clock": {"clock"}, "vase": {"vase"}, "potted plant": {"plant","flower","flowers"},
        "bicycle": {"bicycle","bike"}, "motorcycle": {"motorcycle","motorbike"},
        "bus": {"bus"}, "truck": {"truck"}, "boat": {"boat","ship"},
        "bird": {"bird"}, "horse": {"horse"}, "cow": {"cow"}, "sheep": {"sheep"},
        "elephant": {"elephant"}, "bear": {"bear"}, "zebra": {"zebra"}, "giraffe": {"giraffe"},
        "umbrella": {"umbrella"}, "handbag": {"handbag","purse","bag"}, "backpack": {"backpack"},
        "knife": {"knife"}, "fork": {"fork"}, "spoon": {"spoon"}, "pizza": {"pizza"},
        "cake": {"cake"}, "banana": {"banana"}, "apple": {"apple"}, "sandwich": {"sandwich"},
        "bench": {"bench"}, "toilet": {"toilet"}, "sink": {"sink"}, "refrigerator": {"refrigerator","fridge"},
        "oven": {"oven","stove"}, "microwave": {"microwave"},
    }
    gt_words = set()
    for obj in gt_objects:
        obj_l = obj.lower()
        gt_words.add(obj_l)
        for cat, syns in SYNONYMS.items():
            if obj_l == cat or obj_l in syns:
                gt_words.update(syns)
                gt_words.add(cat)

    words = caption.lower().split()
    hal_pos = []
    for i, w in enumerate(words):
        w_clean = w.strip(".,;:!?\"'()")
        for cat, syns in SYNONYMS.items():
            if w_clean in syns and cat not in [g.lower() for g in gt_objects]:
                is_gt = False
                for gc in gt_objects:
                    gc_l = gc.lower()
                    if w_clean in SYNONYMS.get(gc_l, {gc_l}):
                        is_gt = True
                        break
                if not is_gt:
                    hal_pos.append(i)
                    break
    return hal_pos, words

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--model_path", type=str, default=str(REPO/"data/models/llava-v1.5-7b"))
    p.add_argument("--data_path", type=str, default=str(REPO/"data/coco/val2014"))
    p.add_argument("--anno_path", type=str, default=str(REPO/"data/coco/annotations/instances_val2014.json"))
    p.add_argument("--c_scores_path", type=str, default=str(REPO/"scores/llava_eic.pt"))
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--num_eval_samples", type=int, default=500)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--layers", type=str, default="0,1,2,4,8,16,24,31")
    p.add_argument("--img_start", type=int, default=35)
    p.add_argument("--img_len", type=int, default=576)
    p.add_argument("--alpha", type=float, default=0.3)
    args = p.parse_args()

    layer_indices = [int(x) for x in args.layers.split(",")]

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
        attn_implementation="eager")
    model.eval()
    vt = model.get_vision_tower()
    if not vt.is_loaded:
        vt.load_model()
    vt.to(device=model.device, dtype=torch.float16)
    image_processor = vt.image_processor
    evolve_only_sampling()

    payload = torch.load(args.c_scores_path, map_location="cpu")
    if isinstance(payload, dict):
        c_scores = payload.get("scores", payload.get("C", next(iter(payload.values()))))
    else:
        c_scores = payload
    if c_scores.dim() == 2:
        c_scores = c_scores[1]
    c_scores = c_scores.float()
    log.info(f"C-scores: {c_scores.shape}, nonzero={int((c_scores>0).sum())}")

    monitor = MultiLayerMonitor(model, layer_indices, c_scores, args.img_start, args.img_len)
    monitor.install()

    with open(args.anno_path) as f:
        coco = json.load(f)
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    img_to_cats = {}
    for ann in coco["annotations"]:
        iid = ann["image_id"]
        cat = cat_id_to_name.get(ann["category_id"], "")
        img_to_cats.setdefault(iid, set()).add(cat)

    images = coco["images"]
    random.shuffle(images)
    images = images[:args.num_eval_samples]

    records = []
    tau_floor = 0.3

    for img_info in tqdm(images, desc="Generating"):
        img_id = img_info["id"]
        img_path = os.path.join(args.data_path, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().to(model.device)

        qs = DEFAULT_IMAGE_TOKEN + "\nPlease describe this image in detail."
        conv = conv_templates["v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(model.device)

        monitor.start_sample()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids, images=image_tensor,
                do_sample=True, temperature=1.0, top_p=1.0,
                max_new_tokens=args.max_new_tokens,
                use_only=False, enhance_layer_index=1,
            )
        monitor.finish_sample()
        if isinstance(output_ids, tuple):
            output_ids = output_ids[0]

        gen_ids = output_ids[0, input_ids.shape[1]:]
        caption = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        gt_objects = list(img_to_cats.get(img_id, set()))
        hal_pos, words = identify_hal(caption, gt_objects)

        sample_gs = monitor.all_samples[-1]

        records.append({
            "image_id": img_id,
            "caption": caption,
            "n_words": len(words),
            "hal_pos": hal_pos,
            "has_hal": len(hal_pos) > 0,
            "gt_objects": gt_objects,
            "gs_per_layer": sample_gs,
        })

    monitor.restore()

    os.makedirs(args.out_path, exist_ok=True)
    with open(os.path.join(args.out_path, "multilayer_gs.json"), "w") as f:
        json.dump(records, f)
    log.info(f"Saved {len(records)} records")

    log.info("\n=== Multi-Layer GS Analysis ===")
    alpha = args.alpha

    for L in layer_indices:
        Ls = str(L)
        trigger_counts = []
        total_sharpening = []
        gs_stds = []
        gs_means_hal = []
        gs_means_nonhal = []

        for rec in records:
            gs_steps = rec["gs_per_layer"].get(Ls, [])
            if not gs_steps:
                continue

            gs_arr = np.array(gs_steps)
            gs_stds.append(np.std(gs_arr))

            tau_eff = np.maximum(1.0 - alpha * (1.0 - gs_arr), tau_floor)
            n_trigger = np.sum(tau_eff < 0.95)
            trigger_counts.append(n_trigger / len(gs_arr))
            total_sharpening.append(np.sum(1.0 - tau_eff))

            if rec["has_hal"]:
                gs_means_hal.append(np.mean(gs_arr))
            else:
                gs_means_nonhal.append(np.mean(gs_arr))

        mean_trigger = np.mean(trigger_counts) if trigger_counts else 0
        mean_sharpening = np.mean(total_sharpening) if total_sharpening else 0
        mean_gs_std = np.mean(gs_stds) if gs_stds else 0

        log.info(f"Layer {L:2d}: trigger_rate={mean_trigger:.3f}  "
                 f"total_sharpening={mean_sharpening:.2f}  "
                 f"gs_std={mean_gs_std:.5f}  "
                 f"gs_hal={np.mean(gs_means_hal):.5f}  "
                 f"gs_nonhal={np.mean(gs_means_nonhal):.5f}")

    summary = {}
    for L in layer_indices:
        Ls = str(L)
        all_gs = []
        for rec in records:
            all_gs.extend(rec["gs_per_layer"].get(Ls, []))
        if all_gs:
            gs_arr = np.array(all_gs)
            tau_arr = np.maximum(1.0 - alpha * (1.0 - gs_arr), tau_floor)
            summary[Ls] = {
                "mean_gs": round(float(np.mean(gs_arr)), 5),
                "std_gs": round(float(np.std(gs_arr)), 5),
                "min_gs": round(float(np.min(gs_arr)), 5),
                "max_gs": round(float(np.max(gs_arr)), 5),
                "trigger_rate": round(float(np.mean(tau_arr < 0.95)), 4),
                "mean_tau_eff": round(float(np.mean(tau_arr)), 5),
                "n_steps": len(all_gs),
            }

    with open(os.path.join(args.out_path, "multilayer_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary → {args.out_path}/multilayer_summary.json")

if __name__ == "__main__":
    main()
