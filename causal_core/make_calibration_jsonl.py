"""Build the unlabelled EIC calibration set from COCO annotations.
"""
import argparse, json, random

PROMPTS = [
    "Describe this image.",
    "What is happening in this picture?",
    "What do you see in this image?",
    "Can you describe what's in this photo?",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True,
                    help="COCO instances_*.json (provides image ids + file names).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="Limit to N images (0 = all).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    images = json.load(open(args.instances))["images"]
    random.seed(args.seed)
    random.shuffle(images)
    if args.n > 0:
        images = images[:args.n]

    with open(args.out, "w") as f:
        for i, im in enumerate(images):
            f.write(json.dumps({
                "question_id": im["id"],
                "image": im["file_name"],
                "text": PROMPTS[i % len(PROMPTS)],
            }) + "\n")
    print(f"wrote {len(images)} calibration entries -> {args.out}")

if __name__ == "__main__":
    main()
