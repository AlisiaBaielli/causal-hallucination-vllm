"""
Per-token grounding analysis for InternVL3.5-8B-HF.
"""
import os, sys, re, json, argparse, logging, warnings, random, time
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from transformers import AutoProcessor, InternVLForConditionalGeneration
from transformers.generation.logits_process import LogitsProcessorList
from causal_core.models.internvl import evolve_only_sampling_internvl

from causal_core.monitor import CausalMonitorInternVL, CausalLogitsProcessor, parse_image_id

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

class GroundingLogger(CausalLogitsProcessor):

    def __init__(self, monitor, alpha=0.3):
        super().__init__(monitor, alpha)
        self.token_scores = []
        self._current_sample = []

    def __call__(self, input_ids, scores):
        gs = self.monitor.grounding_score
        ent = self.monitor.mean_entropy
        self._current_sample.append({
            "grounding_score": round(float(gs), 4),
            "entropy": round(float(ent), 4),
        })
        return super().__call__(input_ids, scores)

    def start_sample(self):
        self._current_sample = []

    def finish_sample(self):
        self.token_scores.append(self._current_sample)
        self._current_sample = []

SYNONYMS = {
    "person": {"man","woman","boy","girl","child","people","person","kid","lady","gentleman",
               "player","rider","skier","surfer","snowboarder","pedestrian"},
    "car":     {"car","vehicle","automobile","suv","sedan"},
    "truck":   {"truck","pickup"},
    "bicycle": {"bicycle","bike","cycle"},
    "motorcycle": {"motorcycle","motorbike"},
    "bus":     {"bus"},
    "train":   {"train","locomotive"},
    "airplane": {"airplane","plane","jet","aircraft"},
    "boat":    {"boat","ship","vessel","sailboat","yacht"},
    "dog":     {"dog","puppy","pup"},
    "cat":     {"cat","kitten","kitty"},
    "horse":   {"horse","pony"},
    "sheep":   {"sheep","lamb"},
    "cow":     {"cow","cattle","bull"},
    "elephant": {"elephant"},
    "bear":    {"bear"},
    "zebra":   {"zebra"},
    "giraffe": {"giraffe"},
    "bird":    {"bird","seagull","duck","goose","sparrow","pigeon","eagle","owl"},
    "bottle":  {"bottle"},
    "cup":     {"cup","mug"},
    "wine glass": {"glass","wineglass"},
    "fork":    {"fork"},
    "knife":   {"knife"},
    "spoon":   {"spoon"},
    "bowl":    {"bowl"},
    "banana":  {"banana"},
    "apple":   {"apple"},
    "sandwich": {"sandwich"},
    "orange":  {"orange"},
    "broccoli": {"broccoli"},
    "carrot":  {"carrot"},
    "hot dog": {"hotdog"},
    "pizza":   {"pizza"},
    "donut":   {"donut","doughnut"},
    "cake":    {"cake"},
    "chair":   {"chair"},
    "couch":   {"couch","sofa"},
    "potted plant": {"plant"},
    "bed":     {"bed"},
    "dining table": {"table"},
    "toilet":  {"toilet"},
    "tv":      {"tv","television"},
    "laptop":  {"laptop","computer"},
    "mouse":   {"mouse"},
    "remote":  {"remote"},
    "keyboard": {"keyboard"},
    "cell phone": {"phone","cellphone"},
    "microwave": {"microwave"},
    "oven":    {"oven"},
    "toaster": {"toaster"},
    "sink":    {"sink"},
    "refrigerator": {"refrigerator","fridge"},
    "book":    {"book"},
    "clock":   {"clock"},
    "vase":    {"vase"},
    "scissors": {"scissors"},
    "teddy bear": {"teddy"},
    "hair drier": {"hairdryer"},
    "toothbrush": {"toothbrush"},
    "umbrella": {"umbrella"},
    "handbag": {"handbag","purse","bag"},
    "tie":     {"tie","necktie"},
    "suitcase": {"suitcase","luggage"},
    "frisbee": {"frisbee"},
    "skis":    {"skis","ski"},
    "snowboard": {"snowboard"},
    "sports ball": {"ball"},
    "kite":    {"kite"},
    "baseball bat": {"bat"},
    "baseball glove": {"glove"},
    "skateboard": {"skateboard"},
    "surfboard": {"surfboard"},
    "tennis racket": {"racket","racquet"},
    "stop sign": {"sign"},
    "parking meter": {"meter"},
    "bench":   {"bench"},
    "fire hydrant": {"hydrant"},
    "traffic light": {"light","stoplight"},
    "backpack": {"backpack"},
}

def identify_hallucinated_words(caption, coco_objects):
    gt_words = set()
    for cat in coco_objects:
        gt_words.add(cat)
        for syn in SYNONYMS.get(cat, set()):
            gt_words.add(syn)
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
                   default=str(REPO / "data/models/InternVL3_5-8B-HF"))
    p.add_argument("--data_path", type=str, default=str(REPO / "data/coco/val2014"))
    p.add_argument("--anno_path", type=str,
                   default=str(REPO / "data/coco/annotations/instances_val2014.json"))
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--num_eval_samples", type=int, default=500)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--c_scores_path", type=str, required=True)
    p.add_argument("--layer_index", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    log.info(f"Loading InternVL: {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = InternVLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    image_token_id = model.config.image_token_id

    try:
        model.config._attn_implementation = "eager"
        if hasattr(model.config, "text_config"):
            model.config.text_config._attn_implementation = "eager"
        if hasattr(model.model, "language_model") and hasattr(model.model.language_model, "config"):
            model.model.language_model.config._attn_implementation = "eager"
    except Exception:
        pass

    evolve_only_sampling_internvl()

    payload = torch.load(args.c_scores_path, map_location="cpu", weights_only=False)
    c_scores = payload["C"].float()
    log.info(f"C-scores at L{args.layer_index}: nonzero={int((c_scores>0).sum())}/{c_scores.shape[0]}")

    monitor = CausalMonitorInternVL(model, args.layer_index, c_scores, image_token_id)
    monitor.install_hook()

    gs_logger = GroundingLogger(monitor, alpha=args.alpha)

    with open(args.anno_path) as f:
        coco = json.load(f)
    images = coco["images"]
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    img_to_cats = {}
    for ann in coco["annotations"]:
        iid = ann["image_id"]
        cat = cat_id_to_name.get(ann["category_id"], "")
        img_to_cats.setdefault(iid, set()).add(cat)

    random.shuffle(images)
    images = images[:args.num_eval_samples]

    os.makedirs(args.out_path, exist_ok=True)
    all_records = []

    for img_info in tqdm(images, total=len(images)):
        img_id = img_info["id"]
        img_path = os.path.join(args.data_path, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Please describe this image in detail."},
                ],
            }]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)

            monitor.set_img_positions(inputs.input_ids, image_token_id)

            gs_logger.start_sample()
            processors = LogitsProcessorList([gs_logger])

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    do_sample=True, temperature=args.temperature, top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    use_only=False,
                    enhance_layer_index=args.layer_index,
                    logits_processor=processors,
                )

            gs_logger.finish_sample()

            gen_ids = output_ids[0, inputs.input_ids.shape[1]:]
            tokens = [processor.tokenizer.decode([tid], skip_special_tokens=False) for tid in gen_ids]
            caption = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

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

        except Exception as e:
            log.warning(f"Skip image {img_id}: {e}")
            continue

    out_file = os.path.join(args.out_path, "pertoken_grounding.json")
    with open(out_file, "w") as f:
        json.dump(all_records, f, indent=2)
    log.info(f"Saved {len(all_records)} records → {out_file}")

if __name__ == "__main__":
    main()
