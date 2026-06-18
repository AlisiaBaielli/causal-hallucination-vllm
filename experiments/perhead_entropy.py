"""
Per-Head Entropy Analysis for Causal
"""
import os, sys, json, argparse, logging, warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "transformers" / "src"))
sys.path.insert(0, str(REPO))

import torch
import numpy as np

from llava.model import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from transformers import AutoTokenizer
from causal_core.models.llava_sampling import evolve_only_sampling
from tqdm import tqdm
from PIL import Image

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

class PerHeadMonitor:
    """Hooks attention layer, logs normalized entropy for EVERY head per decode step."""

    def __init__(self, model, layer_idx, img_start=35, img_len=576):
        self.layer      = model.model.layers[layer_idx]
        self.num_heads  = self.layer.self_attn.config.num_attention_heads
        self.img_start  = img_start
        self.img_len    = img_len
        self._current   = []
        self.all_samples = []

    def install(self):
        layer   = self.layer
        monitor = self
        orig    = layer.self_attn.forward

        def hooked(*args, **kwargs):
            kwargs_c = dict(kwargs)
            kwargs_c["output_attentions"] = True
            result = orig(*args, **kwargs_c)

            if len(result) >= 2 and result[1] is not None:
                attn_w = result[1]
                B, H, Q, KV = attn_w.shape
                img_end = monitor.img_start + monitor.img_len

                if Q == 1 and KV > img_end:
                    img_attn = attn_w[0, :, 0, monitor.img_start:img_end]
                    s = img_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                    p = img_attn / s
                    ent = -(p * (p + 1e-10).log()).sum(dim=-1)
                    max_ent = torch.log(torch.tensor(float(monitor.img_len),
                                                     device=ent.device))
                    norm_ent = (ent / max_ent).clamp(0, 1)
                    monitor._current.append(norm_ent.detach().cpu().float().numpy())

            return result

        layer.self_attn.forward = hooked
        return orig

    def restore(self, orig):
        self.layer.self_attn.forward = orig

    def start_sample(self):
        self._current = []

    def finish_sample(self):
        self.all_samples.append(list(self._current))
        self._current = []

SYNONYMS = {
    "person": {"man","woman","boy","girl","child","people","person","kid","lady",
               "gentleman","player","rider","skier","surfer","snowboarder","pedestrian"},
    "car": {"car","vehicle","automobile","suv","sedan"},
    "truck": {"truck","pickup"},
    "bicycle": {"bicycle","bike","cycle"},
    "motorcycle": {"motorcycle","motorbike"},
    "bus": {"bus"},
    "train": {"train","locomotive"},
    "airplane": {"airplane","plane","jet","aircraft"},
    "boat": {"boat","ship","vessel","sailboat","yacht"},
    "dog": {"dog","puppy","pup"},
    "cat": {"cat","kitten","kitty"},
    "horse": {"horse","pony"},
    "cow": {"cow","cattle","bull","ox"},
    "sheep": {"sheep","lamb"},
    "bird": {"bird","parrot","eagle","pigeon","seagull"},
    "elephant": {"elephant"},
    "bear": {"bear"},
    "giraffe": {"giraffe"},
    "zebra": {"zebra"},
    "bench": {"bench"},
    "chair": {"chair","seat"},
    "couch": {"couch","sofa"},
    "bed": {"bed"},
    "dining table": {"table","desk"},
    "toilet": {"toilet"},
    "tv": {"tv","television","monitor","screen"},
    "laptop": {"laptop","computer","notebook"},
    "cell phone": {"phone","cellphone","smartphone"},
    "refrigerator": {"refrigerator","fridge"},
    "oven": {"oven","stove"},
    "microwave": {"microwave"},
    "sink": {"sink"},
    "clock": {"clock"},
    "vase": {"vase"},
    "book": {"book"},
    "cup": {"cup","mug"},
    "bottle": {"bottle"},
    "bowl": {"bowl"},
    "knife": {"knife"},
    "fork": {"fork"},
    "spoon": {"spoon"},
    "pizza": {"pizza"},
    "cake": {"cake"},
    "sandwich": {"sandwich"},
    "hot dog": {"hotdog"},
    "donut": {"donut","doughnut"},
    "banana": {"banana"},
    "apple": {"apple"},
    "orange": {"orange"},
    "broccoli": {"broccoli"},
    "carrot": {"carrot"},
    "umbrella": {"umbrella"},
    "handbag": {"handbag","purse","bag"},
    "tie": {"tie","necktie"},
    "suitcase": {"suitcase","luggage"},
    "frisbee": {"frisbee"},
    "skis": {"ski","skis"},
    "snowboard": {"snowboard"},
    "sports ball": {"ball","baseball","football","soccer","tennis"},
    "kite": {"kite"},
    "baseball bat": {"bat"},
    "baseball glove": {"glove","mitt"},
    "skateboard": {"skateboard"},
    "surfboard": {"surfboard"},
    "tennis racket": {"racket","racquet"},
    "backpack": {"backpack"},
    "potted plant": {"plant","flower","flowers"},
    "scissors": {"scissors"},
    "teddy bear": {"teddy"},
    "toothbrush": {"toothbrush"},
    "hair drier": {"hairdryer","dryer"},
    "remote": {"remote"},
    "keyboard": {"keyboard"},
    "mouse": {"mouse"},
    "toaster": {"toaster"},
    "fire hydrant": {"hydrant"},
    "stop sign": {"sign"},
    "parking meter": {"meter"},
    "traffic light": {"traffic"},
    "wine glass": {"glass","wine"},
}

ALL_COCO_NOUNS = set()
for cat, syns in SYNONYMS.items():
    ALL_COCO_NOUNS.update(syns)
    ALL_COCO_NOUNS.add(cat)

import re
def identify_hal(caption, gt_objects):
    gt_words = set()
    for obj in gt_objects:
        ol = obj.lower()
        gt_words.add(ol)
        for cat, syns in SYNONYMS.items():
            if ol in syns or ol == cat:
                gt_words.update(syns)
                gt_words.add(cat)
    words = re.findall(r'[a-z]+', caption.lower())
    hal = [i for i, w in enumerate(words) if w in ALL_COCO_NOUNS and w not in gt_words]
    return hal, words

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",            type=int,   default=3407)
    p.add_argument("--model_path",      type=str,   default=str(REPO/"data/models/llava-v1.5-7b"))
    p.add_argument("--data_path",       type=str,   default=str(REPO/"data/coco/val2014"))
    p.add_argument("--anno_path",       type=str,   default=str(REPO/"data/coco/annotations/instances_val2014.json"))
    p.add_argument("--c_scores_path",   type=str,   default=str(REPO/"scores/llava_eic.pt"))
    p.add_argument("--out_path",        type=str,   required=True)
    p.add_argument("--num_eval_samples",type=int,   default=500)
    p.add_argument("--max_new_tokens",  type=int,   default=128)
    p.add_argument("--layer_index",     type=int,   default=1)
    p.add_argument("--img_start",       type=int,   default=35)
    p.add_argument("--img_len",         type=int,   default=576)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import random, numpy as _np
    random.seed(args.seed)
    _np.random.seed(args.seed)

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
        c_scores = c_scores[args.layer_index]
    c_scores = c_scores.float().numpy()
    log.info(f"C-scores shape: {c_scores.shape}, layer {args.layer_index}")
    log.info(f"Top-5 C-score heads: {np.argsort(c_scores)[::-1][:5]}")

    monitor = PerHeadMonitor(model, args.layer_index, args.img_start, args.img_len)
    orig_fwd = monitor.install()

    with open(args.anno_path) as f:
        coco = json.load(f)
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    img_to_cats = {}
    for ann in coco["annotations"]:
        iid = ann["image_id"]
        cat = cat_id_to_name.get(ann["category_id"], "")
        img_to_cats.setdefault(iid, set()).add(cat)

    import random as _rng
    _rng.seed(args.seed)
    images = list(coco["images"])
    _rng.shuffle(images)
    images = images[:args.num_eval_samples]

    os.makedirs(args.out_path, exist_ok=True)
    records = []

    for img_info in tqdm(images):
        img_id   = img_info["id"]
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

        monitor.start_sample()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids, images=image_tensor,
                do_sample=True, temperature=1.0, top_p=1.0,
                max_new_tokens=args.max_new_tokens,
                use_only=False, enhance_layer_index=args.layer_index,
            )
        monitor.finish_sample()
        if isinstance(output_ids, tuple):
            output_ids = output_ids[0]

        gen_ids = output_ids[0, input_ids.shape[1]:]
        caption = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        tokens  = [tokenizer.decode([tid]) for tid in gen_ids]

        gt_objects = list(img_to_cats.get(img_id, set()))
        hal_pos, words = identify_hal(caption, gt_objects)

        steps = monitor.all_samples[-1]
        records.append({
            "image_id":   img_id,
            "caption":    caption,
            "n_tokens":   len(tokens),
            "n_words":    len(words),
            "words":      words,
            "hal_pos":    hal_pos,
            "gt_objects": gt_objects,
            "per_head_entropy": [s.tolist() for s in steps],
        })

    monitor.restore(orig_fwd)

    out_file = os.path.join(args.out_path, "perhead_entropy.json")
    with open(out_file, "w") as f:
        json.dump(records, f)
    log.info(f"Saved {len(records)} records → {out_file}")

    log.info("Running per-head analysis...")
    _run_analysis(records, c_scores, args.out_path)

def _run_analysis(records, c_scores, out_path):
    from sklearn.metrics import roc_curve, auc as sk_auc
    from scipy import stats

    num_heads = len(c_scores)

    def interp(arr1d, n_out):
        """Interpolate a 1-D array to n_out points."""
        arr = np.array(arr1d, dtype=float)
        if len(arr) == 0 or n_out == 0:
            return np.full(n_out, np.nan)
        xi = np.linspace(0, len(arr) - 1, n_out)
        left  = np.floor(xi).astype(int)
        right = np.minimum(left + 1, len(arr) - 1)
        frac  = xi - left
        return arr[left] * (1 - frac) + arr[right] * frac

    y_true  = [[] for _ in range(num_heads)]
    y_score = [[] for _ in range(num_heads)]

    for rec in records:
        steps   = rec["per_head_entropy"]
        n_words = rec["n_words"]
        hals    = set(rec["hal_pos"])
        if n_words == 0 or len(steps) == 0:
            continue

        for h in range(num_heads):
            h_ent = np.array([step[h] for step in steps])
            h_w   = interp(h_ent, n_words)
            for i, e in enumerate(h_w):
                if not np.isnan(e):
                    y_true[h].append(1 if i in hals else 0)
                    y_score[h].append(e)
    aucs = []
    for h in range(num_heads):
        yt = np.array(y_true[h])
        ys = np.array(y_score[h])
        if yt.sum() < 5:
            aucs.append(0.5)
            continue
        try:
            fpr, tpr, _ = roc_curve(yt, ys)
            aucs.append(sk_auc(fpr, tpr))
        except Exception:
            aucs.append(0.5)

    aucs = np.array(aucs)

    spearman = stats.spearmanr(c_scores, aucs)
    log.info(f"Spearman r(C-score, AUC) = {spearman.statistic:.4f}, p = {spearman.pvalue:.4f}")

    order = np.argsort(c_scores)[::-1]
    log.info("\nHead | C-score | AUC  | high-C?")
    high_c = c_scores > 0
    for h in order:
        log.info(f"  {h:2d} | {c_scores[h]:7.4f} | {aucs[h]:.4f} | {'YES' if high_c[h] else 'no'}")

    auc_high = aucs[high_c]
    auc_low  = aucs[~high_c]
    t, p = stats.ttest_ind(auc_high, auc_low, equal_var=False)
    log.info(f"\nMean AUC high-C heads: {auc_high.mean():.4f} (n={len(auc_high)})")
    log.info(f"Mean AUC low-C  heads: {auc_low.mean():.4f}  (n={len(auc_low)})")
    log.info(f"Welch t={t:.3f}, p={p:.4f}")

    summary = {
        "spearman_r": round(float(spearman.statistic), 4),
        "spearman_p": round(float(spearman.pvalue), 4),
        "mean_auc_high_c": round(float(auc_high.mean()), 4),
        "mean_auc_low_c":  round(float(auc_low.mean()), 4),
        "t_stat": round(float(t), 4),
        "p_value": round(float(p), 4),
        "per_head": [
            {"head": int(h), "c_score": round(float(c_scores[h]), 4),
             "auc": round(float(aucs[h]), 4), "high_c": bool(high_c[h])}
            for h in range(num_heads)
        ],
    }
    with open(os.path.join(out_path, "perhead_analysis.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary → perhead_analysis.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        ax = axes[0]
        colors = ["#DC2626" if high_c[h] else "#6B7280" for h in range(num_heads)]
        ax.scatter(c_scores, aucs, c=colors, s=60, alpha=0.8, edgecolors="none")
        ax.axhline(0.5, color="black", linestyle="--", lw=1, alpha=0.5, label="Random (0.5)")
        ax.set_xlabel("C-score", fontsize=11)
        ax.set_ylabel("Hallucination AUC", fontsize=11)
        ax.set_title(f"C-score vs. Per-Head AUC\n(Spearman r={spearman.statistic:.3f}, p={spearman.pvalue:.3f})", fontsize=11)
        from matplotlib.lines import Line2D
        legend_elems = [
            Line2D([0],[0], marker='o', color='w', markerfacecolor='#DC2626', markersize=8, label='High-C heads'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='#6B7280', markersize=8, label='Low-C heads'),
        ]
        ax.legend(handles=legend_elems, fontsize=9)

        ax2 = axes[1]
        means = [auc_high.mean(), auc_low.mean()]
        sems  = [auc_high.std()/np.sqrt(len(auc_high)), auc_low.std()/np.sqrt(len(auc_low))]
        bars  = ax2.bar(["High-C heads\n(C-score > 0)", "Low-C heads\n(C-score ≤ 0)"],
                        means, yerr=sems, capsize=5,
                        color=["#DC2626", "#6B7280"], alpha=0.8, width=0.4)
        ax2.axhline(0.5, color="black", linestyle="--", lw=1, alpha=0.5)
        ax2.set_ylabel("Mean Hallucination AUC", fontsize=11)
        ax2.set_title(f"High-C vs Low-C Heads\n(t={t:.2f}, p={p:.3f})", fontsize=11)
        ax2.set_ylim(0.48, max(means) + 0.03)

        plt.tight_layout()
        fig.savefig(os.path.join(out_path, "perhead_auc.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(out_path, "perhead_auc.png"), dpi=150, bbox_inches="tight")
        plt.close()
        log.info("Saved perhead_auc.pdf")
    except Exception as e:
        log.warning(f"Plot failed: {e}")

if __name__ == "__main__":
    main()
