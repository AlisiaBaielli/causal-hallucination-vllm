"""
Compute CLIPScore for caption files against source images.
CLIPScore = cosine similarity between CLIP image and text embeddings.
"""
import os, sys, json, argparse, logging
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]

def load_jsonl_captions(path):
    """Load captions from JSONL (CHAIR format): {image_id, caption}"""
    results = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            results.append({
                "image_id": d.get("image_id", d.get("id")),
                "caption": d.get("caption", d.get("text", "")),
            })
    return results

def load_json_captions(path):
    """Load captions from JSON (AMBER format): [{id, response}]"""
    with open(path) as f:
        data = json.load(f)
    return [{"image_id": d.get("id"), "caption": d.get("response", "")} for d in data]

def compute_clipscore(model, processor, image_dir, captions, id_to_filename, device):
    """Compute mean CLIPScore for a set of captions."""
    scores = []
    for item in tqdm(captions, desc="CLIPScore", leave=False):
        img_id = item["image_id"]
        caption = item["caption"]

        if not caption:
            continue

        filename = id_to_filename.get(img_id)
        if filename is None:
            continue

        img_path = os.path.join(image_dir, filename)
        if not os.path.exists(img_path):
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(text=[caption], images=[image], return_tensors="pt",
                             padding=True, truncation=True, max_length=77).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                img_emb = outputs.image_embeds
                txt_emb = outputs.text_embeds
                score = torch.nn.functional.cosine_similarity(img_emb, txt_emb).item()
                scores.append(score)
        except Exception as e:
            continue

    return float(np.mean(scores)) if scores else 0.0, len(scores)

def parse_caption_files(pairs):
    """Parse 'name:path' pairs into dict."""
    out = {}
    for pair in pairs:
        name, path = pair.split(":", 1)
        out[name] = path
    return out

def main():
    from transformers import CLIPModel, CLIPProcessor

    p = argparse.ArgumentParser()
    p.add_argument("--caption_files", nargs="+", required=True,
                   help="Space-separated name:path pairs for caption files")
    p.add_argument("--image_dir", type=str, required=True,
                   help="Directory containing source images")
    p.add_argument("--anno_path", type=str, required=True,
                   help="Annotation JSON with 'images' list (COCO-style) or AMBER query JSON")
    p.add_argument("--format", type=str, default="jsonl", choices=["jsonl", "json"],
                   help="Caption file format: jsonl (CHAIR) or json (AMBER)")
    p.add_argument("--out_path", type=str, default=None,
                   help="Path to save results JSON")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading CLIP model on {device}...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.eval()

    with open(args.anno_path) as f:
        anno = json.load(f)

    if "images" in anno:
        id_to_file = {img["id"]: img["file_name"] for img in anno["images"]}
    else:
        id_to_file = {q["id"]: q["image"] for q in anno}

    methods = parse_caption_files(args.caption_files)
    loader = load_jsonl_captions if args.format == "jsonl" else load_json_captions

    results = {}
    for method, path in methods.items():
        if not os.path.exists(path):
            log.warning(f"  {method}: file not found at {path}")
            continue
        captions = loader(path)
        score, n = compute_clipscore(clip_model, clip_processor, args.image_dir,
                                     captions, id_to_file, device)
        results[method] = {"clipscore": round(score, 4), "n": n}
        log.info(f"  {method:10s}: CLIPScore={score:.4f} (n={n})")

    if args.out_path:
        os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
        with open(args.out_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Saved results to {args.out_path}")

if __name__ == "__main__":
    main()
