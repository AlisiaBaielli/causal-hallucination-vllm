"""
Efficiency benchmark: Vanilla / VCD / M3ID / ONLY / Ours on LLaVA-1.5-7B.

Measures per-instance latency (mean ± std) and peak GPU memory for each
method on the same 50 CHAIR images. Greedy decoding, max_new_tokens=128,
seed=3407.
"""
import os, sys, json, time, argparse, warnings, random, math
from pathlib import Path
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]

_self_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _self_dir]

sys.path.insert(0, str(REPO / "transformers" / "src"))
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from llava.model import LlavaLlamaForCausalLM
from llava.conversation import conv_templates, SeparatorStyle, Conversation
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList

from causal_core.models.llava_sampling import evolve_only_sampling
from causal_core.vcd import add_diffusion_noise
from causal_core.monitor import CausalMonitor, CausalLogitsProcessor

warnings.filterwarnings("ignore")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--model_path", type=str, default=str(REPO / "data/models/llava-v1.5-7b"))
    p.add_argument("--data_path", type=str, default=str(REPO / "data/coco/val2014"))
    p.add_argument("--anno_path", type=str, default=str(REPO / "data/coco/annotations/instances_val2014.json"))
    p.add_argument("--c_scores_path", type=str, default=str(REPO / "results/calibration/llava_zscore_K7.pt"))
    p.add_argument("--layer_index", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--num_eval_samples", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, default=3, help="warmup generations per method (excluded from stats)")
    p.add_argument("--out_path", type=str, required=True)
    return p.parse_args()

def load_chair_images(anno_path, data_path, n, seed=3407):
    """Same selection as the main CHAIR runs."""
    anno = json.load(open(anno_path))
    images = anno["images"]
    rng = random.Random(); rng.seed(seed)
    rng.shuffle(images)
    return images[:n]

def prepare_prompt(tokenizer):
    """LLaVA-1.5 'Please describe this image in detail.' prompt (CHAIR standard)."""
    qs = DEFAULT_IMAGE_TOKEN + "\n" + "Please describe this image in detail."
    conv = Conversation(
        system="A chat between a curious human and an artificial intelligence assistant. "
               "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("USER", "ASSISTANT"),
        version="v1",
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.TWO,
        sep=" ",
        sep2="</s>",
    )
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

def time_method(name, model, tokenizer, image_processor, images_meta, args,
                gen_kwargs_builder, logits_processor=None, prepare_neg=None):
    """
    Run `name` method over images_meta; return per-instance latency list and peak memory.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    input_ids = prepare_prompt(tokenizer)

    times = []
    iter_meta = images_meta
    if args.warmup > 0:
        iter_meta = images_meta[:args.warmup] + images_meta
    is_warmup = lambda i: i < args.warmup

    for i, meta in enumerate(tqdm(iter_meta, desc=name)):
        img_path = os.path.join(args.data_path, meta["file_name"])
        raw = Image.open(img_path).convert("RGB")
        image = image_processor.preprocess(raw, return_tensors="pt")["pixel_values"][0]
        image_tensor = image.unsqueeze(0).half().cuda()

        image_neg = None
        if prepare_neg is not None:
            image_neg = prepare_neg(image_tensor)

        gen_kwargs = gen_kwargs_builder(image_tensor, image_neg)
        if logits_processor is not None:
            gen_kwargs["logits_processor"] = LogitsProcessorList([logits_processor])

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.inference_mode():
            _ = model.generate(input_ids, **gen_kwargs)
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        if not is_warmup(i):
            times.append(t2 - t1)

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return times, peak_mem

def main():
    args = parse_args()
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed); np.random.seed(args.seed)

    print(f"[load] model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()
    vt = model.get_vision_tower()
    if not vt.is_loaded:
        vt.load_model()
    vt.to(device=model.device, dtype=torch.float16)
    image_processor = vt.image_processor

    evolve_only_sampling()
    print(f"[load] model ready (device={model.device})")

    images_meta = load_chair_images(args.anno_path, args.data_path,
                                    args.num_eval_samples, seed=args.seed)
    print(f"[data] {len(images_meta)} CHAIR images (seed={args.seed})")

    payload = torch.load(args.c_scores_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        c_scores = payload.get("C", payload.get("scores", next(iter(payload.values()))))
    else:
        c_scores = payload
    if c_scores.dim() == 2:
        c_scores = c_scores[args.layer_index]
    c_scores = c_scores.float()

    results = {}

    common = dict(
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
    )

    def gen_vanilla(image, _):
        return dict(images=image, images_pos=None, images_neg=None, **common)
    print("\n[run] Vanilla")
    times_v, mem_v = time_method("Vanilla", model, tokenizer, image_processor,
                                  images_meta, args, gen_vanilla)
    results["Vanilla"] = dict(times=times_v, peak_mem_gib=mem_v)

    def prep_vcd_neg(img):
        return add_diffusion_noise(img, noise_step=500)
    def gen_vcd(image, image_neg):
        return dict(images=image, images_pos=None, images_neg=image_neg,
                    use_vcd=True,
                    ritual_alpha_pos=3.0, ritual_alpha_neg=1.0, ritual_beta=0.1,
                    js_gamma=0.0, **common)
    print("\n[run] VCD")
    times_vcd, mem_vcd = time_method("VCD", model, tokenizer, image_processor,
                                      images_meta, args, gen_vcd, prepare_neg=prep_vcd_neg)
    results["VCD"] = dict(times=times_vcd, peak_mem_gib=mem_vcd)

    def prep_m3id_neg(img):
        return torch.zeros_like(img)
    def gen_m3id(image, image_neg):
        return dict(images=image, images_pos=None, images_neg=image_neg,
                    use_m3id=True,
                    ritual_alpha_pos=3.0, ritual_alpha_neg=1.0, ritual_beta=0.1,
                    js_gamma=0.0, **common)
    print("\n[run] M3ID")
    times_m3id, mem_m3id = time_method("M3ID", model, tokenizer, image_processor,
                                        images_meta, args, gen_m3id, prepare_neg=prep_m3id_neg)
    results["M3ID"] = dict(times=times_m3id, peak_mem_gib=mem_m3id)

    def gen_only(image, _):
        return dict(images=image, images_pos=None, images_neg=None,
                    use_only=True, enhance_layer_index=1,
                    ritual_alpha_pos=3.0, ritual_alpha_neg=1.0,
                    ritual_beta=0.1, js_gamma=0.25, **common)
    print("\n[run] ONLY")
    times_only, mem_only = time_method("ONLY", model, tokenizer, image_processor,
                                        images_meta, args, gen_only)
    results["ONLY"] = dict(times=times_only, peak_mem_gib=mem_only)

    monitor = CausalMonitor(model, args.layer_index, c_scores, img_start=35, img_len=576)
    orig_fwd = monitor.install_qk_hook()
    proc = CausalLogitsProcessor(monitor, alpha=args.alpha)
    def gen_chall(image, _):
        return dict(images=image, images_pos=None, images_neg=None,
                    use_only=False, enhance_layer_index=args.layer_index,
                    **common)
    print("\n[run] Ours (Causal)")
    times_s, mem_s = time_method("Ours", model, tokenizer, image_processor,
                                  images_meta, args, gen_chall, logits_processor=proc)
    monitor.restore(orig_fwd)
    results["Ours"] = dict(times=times_s, peak_mem_gib=mem_s)

    print("\n" + "=" * 70)
    print(f"{'Method':<10s} {'mean (s)':>10s} {'std (s)':>10s} {'min (s)':>10s} {'max (s)':>10s} {'mem (GiB)':>12s}")
    print("-" * 70)
    summary = {}
    for name, r in results.items():
        t = np.array(r["times"])
        summary[name] = dict(
            n=len(t),
            mean=float(t.mean()), std=float(t.std()),
            min=float(t.min()), max=float(t.max()),
            peak_mem_gib=float(r["peak_mem_gib"]),
        )
        print(f"{name:<10s} {t.mean():>10.4f} {t.std():>10.4f} {t.min():>10.4f} {t.max():>10.4f} {r['peak_mem_gib']:>12.2f}")
    print("=" * 70)

    vt = summary["Vanilla"]["mean"]
    vm = summary["Vanilla"]["peak_mem_gib"]
    print(f"\n{'Method':<10s} {'time × Vanilla':>16s} {'mem × Vanilla':>16s}")
    print("-" * 50)
    for name in ["Vanilla", "VCD", "M3ID", "ONLY", "Ours"]:
        s = summary[name]
        print(f"{name:<10s} {s['mean']/vt:>15.3f}× {s['peak_mem_gib']/vm:>15.3f}×")

    os.makedirs(args.out_path, exist_ok=True)
    with open(os.path.join(args.out_path, "efficiency_results.json"), "w") as f:
        json.dump(dict(
            config=dict(
                seed=args.seed,
                n_images=args.num_eval_samples,
                max_new_tokens=args.max_new_tokens,
                warmup=args.warmup,
                alpha=args.alpha,
                layer_index=args.layer_index,
            ),
            summary=summary,
            raw=results,
        ), f, indent=2)
    print(f"\nSaved {args.out_path}/efficiency_results.json")

if __name__ == "__main__":
    main()
