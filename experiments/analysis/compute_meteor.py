"""
Compute CIDEr and METEOR on CHAIR captions using pycocoevalcap.
"""
import os, sys, json, argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from pycocotools.coco import COCO
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

def load_jsonl(path):
    results = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            results.append({
                "image_id": int(d.get("image_id", d.get("id"))),
                "caption": d.get("caption", d.get("text", "")),
            })
    return results

def evaluate_captions(coco_gt, captions):
    valid_ids = set(coco_gt.getImgIds())
    filtered = [c for c in captions if c["image_id"] in valid_ids]
    img_ids = [c["image_id"] for c in filtered]

    gts = {}
    res = {}
    for c in filtered:
        img_id = c["image_id"]
        res[img_id] = [{"caption": c["caption"]}]
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        gts[img_id] = [{"caption": a["caption"]} for a in anns]

    tokenizer = PTBTokenizer()
    gts_tok = tokenizer.tokenize(gts)
    res_tok = tokenizer.tokenize(res)

    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts_tok, res_tok)

    meteor_scorer = Meteor()
    meteor_score, _ = meteor_scorer.compute_score(gts_tok, res_tok)

    return {"CIDEr": cider_score, "METEOR": meteor_score, "n": len(filtered)}

def parse_caption_files(pairs):
    """Parse 'name:path' pairs into dict."""
    out = {}
    for pair in pairs:
        name, path = pair.split(":", 1)
        out[name] = path
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--caption_files", nargs="+", required=True,
                   help="Space-separated name:path pairs for caption JSONL files")
    p.add_argument("--coco_captions", type=str, required=True,
                   help="Path to captions_val2014.json")
    p.add_argument("--out_path", type=str, default=None,
                   help="Path to save results JSON")
    args = p.parse_args()

    print(f"Loading COCO captions from {args.coco_captions}")
    coco_gt = COCO(args.coco_captions)

    methods = parse_caption_files(args.caption_files)
    all_results = {}
    for method, path in methods.items():
        if not os.path.exists(path):
            print(f"  {method}: file not found")
            continue
        captions = load_jsonl(path)
        scores = evaluate_captions(coco_gt, captions)
        all_results[method] = scores
        print(f"  {method:10s}: CIDEr={scores['CIDEr']:.4f}  METEOR={scores['METEOR']:.4f}  (n={scores['n']})")

    if args.out_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
        with open(args.out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved to {args.out_path}")

if __name__ == "__main__":
    main()
