"""
Ablation: EIC head selection + static head suppression, MME.
"""
import os, sys, json, argparse, logging, warnings, random
from pathlib import Path
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "transformers" / "src"))
sys.path.insert(0, str(REPO))
_self_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _self_dir]

import torch
import numpy as np
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from causal_core.models.llava_sampling import evolve_only_sampling

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _eic_suppress_core import install_eic_suppress, restore_eic_suppress, load_c_scores

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)
evolve_only_sampling()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--image_folder", type=str, required=True)
    p.add_argument("--question_file", type=str, required=True)
    p.add_argument("--answers_file", type=str, required=True)
    p.add_argument("--conv_mode", type=str, default="llava_v1")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--do_sample", type=bool, default=True)
    p.add_argument("--c_scores_path", type=str, required=True)
    p.add_argument("--layer_index", type=int, default=1)
    args = p.parse_args()

    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed); np.random.seed(args.seed)

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name)

    c_scores = load_c_scores(args.c_scores_path, args.layer_index)
    orig_fwd, head_mask = install_eic_suppress(model, args.layer_index, c_scores)
    log.info(f"[EIC-suppress] layer={args.layer_index}  kept heads={int(head_mask.sum())}/{len(head_mask)}")

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    ans_file = open(args.answers_file, "w")

    log.info(f"MME eval (EIC-suppress): {len(questions)} questions")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["question"]
        cur_prompt = qs
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_tensor = image_tensor.to('cuda')

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                use_only=False,
                enhance_layer_index=args.layer_index,
            )
        if isinstance(output_ids, tuple):
            output_ids = output_ids[0]

        outputs = tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
        )[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)].strip()
        ans_file.write(json.dumps({
            "question_id": idx, "prompt": cur_prompt, "text": outputs,
            "model_id": model_name, "image": image_file, "metadata": {},
        }) + "\n")
        ans_file.flush()

    ans_file.close()
    restore_eic_suppress(model, args.layer_index, orig_fwd)
    log.info(f"Saved answers to {args.answers_file}")

if __name__ == "__main__":
    main()
