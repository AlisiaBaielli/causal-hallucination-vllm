"""
Standard MME scorer.

Computes the official MME metric (per-category accuracy + accuracy+, summed into
Perception and Cognition totals) from a model answers file and the question file.
"""
import argparse, json
from collections import defaultdict

PERCEPTION = ["existence", "count", "position", "color", "posters", "celebrity",
              "scene", "landmark", "artwork", "OCR"]
COGNITION = ["commonsense_reasoning", "numerical_calculation",
             "text_translation", "code_reasoning"]
# Thesis MME "total (out of 800)" = the 4 object-hallucination categories only.
HALLUCINATION = ["existence", "count", "position", "color"]


def parse_yes_no(text: str):
    t = text.strip().lower()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    if "yes" in t and "no" not in t:
        return "yes"
    if "no" in t and "yes" not in t:
        return "no"
    return "no"


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--answers_file", required=True)
    p.add_argument("--question_file", required=True)
    p.add_argument("--out_path", default=None)
    args = p.parse_args()

    gt = {}
    for r in load_jsonl(args.question_file):
        qid = str(r["question_id"])
        gt[qid] = {
            "answer": str(r["answer"]).strip().lower(),
            "category": r.get("category") or qid.split("/")[0],
            "image": r.get("image", qid.rsplit("/", 1)[0]),
        }

    preds = {str(r["question_id"]): parse_yes_no(r.get("text", "")) for r in load_jsonl(args.answers_file)}

    # per-category bookkeeping
    cat_q_total = defaultdict(int)
    cat_q_correct = defaultdict(int)
    img_correct = defaultdict(lambda: [0, 0])  # (category, image) -> [n_questions, n_correct]

    for qid, g in gt.items():
        if qid not in preds:
            continue
        cat = g["category"]
        correct = int(preds[qid] == g["answer"])
        cat_q_total[cat] += 1
        cat_q_correct[cat] += correct
        key = (cat, g["image"])
        img_correct[key][0] += 1
        img_correct[key][1] += correct

    # acc+ : images where all questions correct
    cat_img_total = defaultdict(int)
    cat_img_allcorrect = defaultdict(int)
    for (cat, _img), (nq, nc) in img_correct.items():
        cat_img_total[cat] += 1
        if nq == nc:
            cat_img_allcorrect[cat] += 1

    per_cat = {}
    for cat in cat_q_total:
        acc = cat_q_correct[cat] / cat_q_total[cat] if cat_q_total[cat] else 0.0
        accp = cat_img_allcorrect[cat] / cat_img_total[cat] if cat_img_total[cat] else 0.0
        per_cat[cat] = {"acc": round(acc * 100, 2), "acc_plus": round(accp * 100, 2),
                        "score": round((acc + accp) * 100, 2),
                        "n_q": cat_q_total[cat], "n_img": cat_img_total[cat]}

    perception_total = round(sum(per_cat[c]["score"] for c in PERCEPTION if c in per_cat), 2)
    cognition_total = round(sum(per_cat[c]["score"] for c in COGNITION if c in per_cat), 2)
    # Thesis-reported MME total: existence + count + position + color, out of 800.
    hallucination_total = round(sum(per_cat[c]["score"] for c in HALLUCINATION if c in per_cat), 2)

    summary = {
        "mme_total": hallucination_total,
        "perception_total": perception_total,
        "cognition_total": cognition_total,
        "overall_total": round(perception_total + cognition_total, 2),
        "per_category": per_cat,
    }

    print(f"{'Category':<26} {'Acc':>7} {'Acc+':>7} {'Score':>8}")
    for group, cats in (("PERCEPTION", PERCEPTION), ("COGNITION", COGNITION)):
        print(f"-- {group} --")
        for c in cats:
            if c in per_cat:
                d = per_cat[c]
                print(f"{c:<26} {d['acc']:>7} {d['acc_plus']:>7} {d['score']:>8}")
    print(f"{'PERCEPTION TOTAL':<26} {'':>7} {'':>7} {perception_total:>8}")
    print(f"{'COGNITION TOTAL':<26} {'':>7} {'':>7} {cognition_total:>8}")
    print(f"{'MME TOTAL (thesis, /800)':<26} {'':>7} {'':>7} {hallucination_total:>8}")

    if args.out_path:
        with open(args.out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved {args.out_path}")


if __name__ == "__main__":
    main()
