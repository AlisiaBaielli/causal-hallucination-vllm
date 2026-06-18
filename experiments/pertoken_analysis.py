"""
Per-Token Grounding Analysis
"""
import os, sys, json, argparse, logging, warnings
from pathlib import Path
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "transformers" / "src"))
sys.path.insert(0, str(REPO))

import torch
from PIL import Image
import numpy as np

from llava.model import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from transformers import AutoTokenizer
from causal_core.models.llava_sampling import evolve_only_sampling

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

from causal_core.monitor import CausalMonitor, CausalLogitsProcessor

class GroundingLogger(CausalLogitsProcessor):

    def __init__(self, monitor, alpha=0.3):
        super().__init__(monitor, alpha)
        self.token_scores = []
        self._current_sample = []

    def __call__(self, input_ids, scores):
        gs = self.monitor.grounding_score
        ent = self.monitor.mean_entropy
        self._current_sample.append({
            "grounding_score": round(gs, 4),
            "entropy": round(ent, 4),
        })
        return super().__call__(input_ids, scores)

    def start_sample(self):
        self._current_sample = []

    def finish_sample(self):
        self.token_scores.append(self._current_sample)
        self._current_sample = []

def identify_hallucinated_words(caption, coco_objects):
    import re
    SYNONYMS = {
        "person": {"man", "woman", "boy", "girl", "child", "people", "person",
                   "kid", "lady", "gentleman", "player", "rider", "skier",
                   "surfer", "snowboarder", "pedestrian"},
        "car": {"car", "vehicle", "automobile", "suv", "sedan"},
        "truck": {"truck", "pickup"},
        "bicycle": {"bicycle", "bike", "cycle"},
        "motorcycle": {"motorcycle", "motorbike"},
        "bus": {"bus"},
        "train": {"train", "locomotive"},
        "airplane": {"airplane", "plane", "jet", "aircraft"},
        "boat": {"boat", "ship", "vessel", "sailboat", "yacht"},
        "dog": {"dog", "puppy", "pup"},
        "cat": {"cat", "kitten", "kitty"},
        "horse": {"horse", "pony"},
        "cow": {"cow", "cattle", "bull", "ox"},
        "sheep": {"sheep", "lamb"},
        "bird": {"bird", "parrot", "eagle", "pigeon", "seagull"},
        "elephant": {"elephant"},
        "bear": {"bear"},
        "giraffe": {"giraffe"},
        "zebra": {"zebra"},
        "bench": {"bench"},
        "chair": {"chair", "seat"},
        "couch": {"couch", "sofa"},
        "bed": {"bed"},
        "dining table": {"table", "desk"},
        "toilet": {"toilet"},
        "tv": {"tv", "television", "monitor", "screen"},
        "laptop": {"laptop", "computer", "notebook"},
        "cell phone": {"phone", "cellphone", "smartphone"},
        "refrigerator": {"refrigerator", "fridge"},
        "oven": {"oven", "stove"},
        "microwave": {"microwave"},
        "sink": {"sink"},
        "clock": {"clock"},
        "vase": {"vase"},
        "book": {"book"},
        "cup": {"cup", "mug"},
        "bottle": {"bottle"},
        "bowl": {"bowl"},
        "knife": {"knife"},
        "fork": {"fork"},
        "spoon": {"spoon"},
        "pizza": {"pizza"},
        "cake": {"cake"},
        "sandwich": {"sandwich"},
        "hot dog": {"hotdog"},
        "donut": {"donut", "doughnut"},
        "banana": {"banana"},
        "apple": {"apple"},
        "orange": {"orange"},
        "broccoli": {"broccoli"},
        "carrot": {"carrot"},
        "umbrella": {"umbrella"},
        "handbag": {"handbag", "purse", "bag"},
        "tie": {"tie", "necktie"},
        "suitcase": {"suitcase", "luggage"},
        "frisbee": {"frisbee"},
        "skis": {"ski", "skis"},
        "snowboard": {"snowboard"},
        "sports ball": {"ball", "baseball", "football", "soccer", "tennis"},
        "kite": {"kite"},
        "baseball bat": {"bat"},
        "baseball glove": {"glove", "mitt"},
        "skateboard": {"skateboard"},
        "surfboard": {"surfboard"},
        "tennis racket": {"racket", "racquet"},
        "backpack": {"backpack"},
        "potted plant": {"plant", "flower", "flowers"},
        "scissors": {"scissors"},
        "teddy bear": {"teddy"},
        "toothbrush": {"toothbrush"},
        "hair drier": {"hairdryer", "dryer"},
        "remote": {"remote"},
        "keyboard": {"keyboard"},
        "mouse": {"mouse"},
        "toaster": {"toaster"},
        "fire hydrant": {"hydrant"},
        "stop sign": {"sign"},
        "parking meter": {"meter"},
        "traffic light": {"traffic"},
        "wine glass": {"glass", "wine"},
    }

    gt_words = set()
    for obj in coco_objects:
        obj_lower = obj.lower()
        gt_words.add(obj_lower)
        for cat, syns in SYNONYMS.items():
            if obj_lower in syns or obj_lower == cat:
                gt_words.update(syns)
                gt_words.add(cat)

    all_coco_nouns = set()
    for cat, syns in SYNONYMS.items():
        all_coco_nouns.update(syns)
        all_coco_nouns.add(cat)

    words = re.findall(r'[a-z]+', caption.lower())
    hallucinated_positions = []
    for i, w in enumerate(words):
        if w in all_coco_nouns and w not in gt_words:
            hallucinated_positions.append(i)

    return hallucinated_positions, words

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--model_path", type=str,
                   default=str(REPO / "data/models/llava-v1.5-7b"))
    p.add_argument("--data_path", type=str,
                   default=str(REPO / "data/coco/val2014"))
    p.add_argument("--anno_path", type=str,
                   default=str(REPO / "data/coco/annotations/instances_val2014.json"))
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--num_eval_samples", type=int, default=500)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--c_scores_path", type=str, required=True)
    p.add_argument("--layer_index", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--img_start", type=int, default=35)
    p.add_argument("--img_len", type=int, default=576)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import random
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

    monitor = CausalMonitor(model, args.layer_index, c_scores,
                          img_start=args.img_start, img_len=args.img_len)
    orig_fwd = monitor.install_qk_hook()

    gs_logger = GroundingLogger(monitor, alpha=args.alpha)

    with open(args.anno_path) as f:
        coco = json.load(f)
    images = coco["images"]
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    img_to_cats = {}
    for ann in coco["annotations"]:
        iid = ann["image_id"]
        cat = cat_id_to_name.get(ann["category_id"], "")
        if iid not in img_to_cats:
            img_to_cats[iid] = set()
        img_to_cats[iid].add(cat)

    import random as _rng
    _rng.seed(args.seed)
    _rng.shuffle(images)
    images = images[:args.num_eval_samples]

    os.makedirs(args.out_path, exist_ok=True)
    from transformers.generation.logits_process import LogitsProcessorList

    all_records = []

    for img_info in tqdm(images, total=len(images)):
        img_id = img_info["id"]
        img_path = os.path.join(args.data_path, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

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

        gs_logger.start_sample()
        processors = LogitsProcessorList([gs_logger])

        output_ids = model.generate(
            input_ids, images=image_tensor,
            do_sample=True, temperature=args.temperature, top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            use_only=False, enhance_layer_index=args.layer_index,
            logits_processor=processors,
        )
        if isinstance(output_ids, tuple):
            output_ids = output_ids[0]

        gs_logger.finish_sample()

        gen_ids = output_ids[0, input_ids.shape[1]:]
        tokens = [tokenizer.decode([tid], skip_special_tokens=False) for tid in gen_ids]
        caption = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        gt_objects = list(img_to_cats.get(img_id, set()))
        hal_positions, words = identify_hallucinated_words(caption, gt_objects)

        gs_data = gs_logger.token_scores[-1]
        record = {
            "image_id": img_id,
            "caption": caption,
            "tokens": tokens[:len(gs_data)],
            "grounding_scores": [s["grounding_score"] for s in gs_data],
            "entropies": [s["entropy"] for s in gs_data],
            "gt_objects": gt_objects,
            "hallucinated_word_positions": hal_positions,
            "words": words,
        }
        all_records.append(record)

    monitor.restore(orig_fwd)

    out_file = os.path.join(args.out_path, "pertoken_grounding.json")
    with open(out_file, "w") as f:
        json.dump(all_records, f, indent=2)
    log.info(f"Saved {len(all_records)} records to {out_file}")

    all_gs_hal = []
    all_gs_nonhal = []
    n_hal_samples = 0

    for rec in all_records:
        gs_list = rec["grounding_scores"]
        words = rec["words"]
        hal_pos = set(rec["hallucinated_word_positions"])

        if len(hal_pos) > 0:
            n_hal_samples += 1
        n_tokens = len(gs_list)
        n_words = len(words)

        if n_words == 0 or n_tokens == 0:
            continue

        tokens_per_word = n_tokens / n_words

        for wi, w in enumerate(words):
            ti_start = int(wi * tokens_per_word)
            ti_end = min(int((wi + 1) * tokens_per_word), n_tokens)
            if ti_start >= n_tokens:
                break
            avg_gs = np.mean([gs_list[t] for t in range(ti_start, max(ti_end, ti_start+1))])

            if wi in hal_pos:
                all_gs_hal.append(avg_gs)
            else:
                all_gs_nonhal.append(avg_gs)

    log.info("=" * 60)
    log.info("GROUNDING SCORE ANALYSIS")
    log.info("=" * 60)
    log.info(f"Total samples: {len(all_records)}")
    log.info(f"Samples with hallucinations: {n_hal_samples}")
    log.info(f"Hallucinated words: {len(all_gs_hal)}")
    log.info(f"Non-hallucinated words: {len(all_gs_nonhal)}")

    if len(all_gs_hal) > 0 and len(all_gs_nonhal) > 0:
        mean_hal = np.mean(all_gs_hal)
        mean_nonhal = np.mean(all_gs_nonhal)
        std_hal = np.std(all_gs_hal)
        std_nonhal = np.std(all_gs_nonhal)

        log.info(f"Mean grounding (hallucinated):     {mean_hal:.4f} +/- {std_hal:.4f}")
        log.info(f"Mean grounding (non-hallucinated): {mean_nonhal:.4f} +/- {std_nonhal:.4f}")
        log.info(f"Difference: {mean_nonhal - mean_hal:.4f}")

        from scipy import stats
        t_stat, p_value = stats.ttest_ind(all_gs_nonhal, all_gs_hal, equal_var=False)
        log.info(f"Welch's t-test: t={t_stat:.4f}, p={p_value:.6f}")

        pooled_std = np.sqrt((std_hal**2 + std_nonhal**2) / 2)
        cohens_d = (mean_nonhal - mean_hal) / (pooled_std + 1e-8)
        log.info(f"Cohen's d: {cohens_d:.4f}")

        summary = {
            "n_samples": len(all_records),
            "n_hal_samples": n_hal_samples,
            "n_hal_words": len(all_gs_hal),
            "n_nonhal_words": len(all_gs_nonhal),
            "mean_gs_hallucinated": round(mean_hal, 4),
            "mean_gs_nonhallucinated": round(mean_nonhal, 4),
            "std_gs_hallucinated": round(std_hal, 4),
            "std_gs_nonhallucinated": round(std_nonhal, 4),
            "difference": round(mean_nonhal - mean_hal, 4),
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "cohens_d": round(cohens_d, 4),
        }
        with open(os.path.join(args.out_path, "grounding_analysis.json"), "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"Saved summary to grounding_analysis.json")
    else:
        log.info("Not enough data for statistical analysis")

if __name__ == "__main__":
    main()
