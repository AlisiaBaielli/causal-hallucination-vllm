
"""
Environment generation for Causal-ONLY calibration.
"""
from __future__ import annotations
import hashlib
import random
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

from PIL import Image, ImageFilter, ImageEnhance, ImageDraw

ENV_LIST_K1 = ["orig"]
ENV_LIST_K3 = ["orig", "img_mismatch", "mask"]
ENV_LIST_K5 = ["orig", "img_mismatch", "mask", "appearance", "paraphrase"]
ENV_LIST_K7 = ENV_LIST_K5 + ["neg_conflict", "ctx_rephrase"]
ENV_LIST_DEFAULT = ENV_LIST_K7

def _stable_int(seed_str: str, mod: int = 2**31 - 1) -> int:
    h = hashlib.sha256(seed_str.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % mod

def _rng_for(example_id: str, env: str, seed0: int = 0) -> random.Random:
    s = _stable_int(f"{seed0}|{example_id}|{env}")
    return random.Random(s)

def _apply_mask(
    img: Image.Image,
    rng: random.Random,
    num_blocks: int = 1,
    block_frac: float = 0.20,
) -> Image.Image:
    """
    Apply random gray block masking to image (E3 in paper).

    Paper spec: m=1 square block, side ratio r=0.2, neutral fill.
    Purpose: weaken background cues while keeping global semantics.
    Constraint: r≤0.25 to preserve semantics.

    Args:
        img: Input PIL Image
        rng: Random number generator for reproducibility
        num_blocks: Number of mask blocks (default: 1, per paper)
        block_frac: Block size as fraction of min(width, height) (default: 0.20, per paper)
    """
    img = img.copy()
    w, h = img.size
    block = int(min(w, h) * block_frac)

    draw = ImageDraw.Draw(img)
    fill = (128, 128, 128)

    for _ in range(num_blocks):
        x0 = rng.randint(0, max(0, w - block))
        y0 = rng.randint(0, max(0, h - block))
        x1 = min(w, x0 + block)
        y1 = min(h, y0 + block)
        draw.rectangle([x0, y0, x1, y1], fill=fill)

    return img

def _apply_appearance(img: Image.Image, rng: random.Random) -> Image.Image:
    """
    Apply appearance perturbations (E4 in paper).

    Paper spec: mild color jitter (0.2), Gaussian blur (σ≈1), downsample-upsample (scale 0.5).
    Purpose: alter low-level statistics without changing object identity.

    Research justification:
    - ColorJitter(0.2) is standard practice (PyTorch docs, typical augmentation)
    - SimCLR uses σ∈[0.1, 2.0] for blur; σ=1 is mid-range
    - Scale 0.5 downsample loses fine details while preserving global structure

    Args:
        img: Input PIL Image
        rng: Random number generator for reproducibility
    """
    img2 = img.copy()

    img2 = img2.filter(ImageFilter.GaussianBlur(radius=1.0))

    enh = ImageEnhance.Brightness(img2)
    img2 = enh.enhance(rng.uniform(0.8, 1.2))
    enh = ImageEnhance.Contrast(img2)
    img2 = enh.enhance(rng.uniform(0.8, 1.2))
    enh = ImageEnhance.Color(img2)
    img2 = enh.enhance(rng.uniform(0.8, 1.2))

    w, h = img2.size
    nw, nh = max(1, int(w * 0.5)), max(1, int(h * 0.5))
    img2 = img2.resize((nw, nh), resample=Image.BILINEAR).resize((w, h), resample=Image.BILINEAR)

    return img2

_PARAPHRASE_RULES = [

    ("Are there any ", "Can you spot "),
    ("Are there ", "Can you see any "),
    ("Is there a ", "Can you see a "),
    ("Is there an ", "Can you see an "),
    ("Is there ", "Do you see "),

    ("What color is ", "What is the color of "),
    ("What colour is ", "What is the colour of "),
    ("What is ", "Identify "),
    ("What are ", "Identify the "),
    ("How many ", "What number of "),
    ("Where is ", "In what location is "),
    ("Where are ", "In what location are "),
    ("Which ", "What "),

    ("Does the ", "Is the "),
    ("Do the ", "Are the "),
    ("Can you see ", "Is there "),
    ("Describe ", "Give a description of "),
    ("Explain ", "Describe "),
    ("Is this ", "Does this show "),
    ("Is that ", "Does that appear to be "),
]

_ENDING_RULES = [
    (" in the image?", " in this picture?"),
    (" in this image?", " shown here?"),
    (" in the picture?", " in the photo?"),
    (" in the photo?", " visible in the image?"),
    (" visible?", " present?"),
    (" shown?", " displayed?"),
]

_PARAPHRASE_BIDIRECTIONAL = [
    (" the image", " this picture"),
    (" the picture", " the photo"),
    (" the photo", " the image"),
    (" a ", " one "),
]

def _paraphrase(text: str, rng: random.Random) -> str:
    """
    Apply semantic-preserving paraphrasing to text.

    Uses expanded rule set covering common POPE/MME question formats.
    Applies rules probabilistically for diversity.

    Args:
        text: Input text
        rng: Random number generator for reproducibility
    """
    t = text

    for a, b in _PARAPHRASE_RULES:
        if a in t and rng.random() < 0.8:
            t = t.replace(a, b, 1)
            break

    for a, b in _ENDING_RULES:
        if a in t and rng.random() < 0.6:
            t = t.replace(a, b, 1)
            break

    if rng.random() < 0.5:
        rule = rng.choice(_PARAPHRASE_BIDIRECTIONAL)
        a, b = rule if rng.random() < 0.5 else (rule[1], rule[0])
        if a in t:
            t = t.replace(a, b, 1)

    return t

_LOC_SWAP = [
    ("on the beach", "in a dim indoor room"),
    ("in a dim indoor room", "on the beach"),
    ("on the street", "in a dark room"),
    ("in a kitchen", "outdoors"),
    ("outdoors", "in a kitchen"),
    ("in a park", "inside a building"),
    ("inside", "outside"),
    ("outside", "inside"),
    ("indoors", "outdoors"),
    ("outdoors", "indoors"),
    ("during the day", "at night"),
    ("at night", "during the day"),
]

_CTX_MODIFICATIONS = [
    ("in the image", "in this scene"),
    ("in this image", "in the photograph"),
    ("in the picture", "in the image"),
    ("in the photo", "in this picture"),
]

def _ctx_rephrase(text: str, rng: random.Random) -> str:
    """
    Modify scene descriptors and locatives (E7 in paper).

    Paper spec: modify scene descriptors and locatives (e.g., "on the beach" →
    "in a dim indoor room") while keeping the task verbatim.
    Purpose: alter environment hints without changing the core question.

    Args:
        text: Input text
        rng: Random number generator for reproducibility
    """
    t = text

    pairs = _LOC_SWAP.copy()
    rng.shuffle(pairs)
    for a, b in pairs:
        if a in t:
            return t.replace(a, b, 1)

    mods = _CTX_MODIFICATIONS.copy()
    rng.shuffle(mods)
    for a, b in mods:
        if a in t and rng.random() < 0.8:
            return t.replace(a, b, 1)

    if "image" in t and rng.random() < 0.8:
        t = t.replace("image", "picture", 1)
    elif "picture" in t and rng.random() < 0.8:
        t = t.replace("picture", "image", 1)

    return t

_OBJECT_COLLOCATION_SWAPS = {

    "crab": "spider",
    "fish": "bird",
    "dolphin": "horse",
    "whale": "elephant",
    "shark": "wolf",
    "seagull": "crow",
    "turtle": "lizard",
    "octopus": "cat",

    "tree": "lamp",
    "flower": "vase",
    "grass": "carpet",
    "mountain": "bookshelf",
    "beach": "bedroom",
    "ocean": "bathtub",
    "sky": "ceiling",
    "sun": "lightbulb",
    "cloud": "pillow",

    "apple": "ball",
    "banana": "phone",
    "orange": "clock",
    "cake": "box",
    "pizza": "frisbee",
    "sandwich": "book",

    "car": "couch",
    "bus": "bed",
    "bicycle": "chair",
    "motorcycle": "table",
    "airplane": "desk",
    "boat": "bench",

    "dog": "teddy bear",
    "cat": "pillow",
    "bird": "kite",
    "horse": "motorcycle",
    "cow": "truck",
    "sheep": "cloud",
    "elephant": "car",
    "bear": "couch",
    "zebra": "fence",
    "giraffe": "crane",

    "person": "mannequin",
    "man": "statue",
    "woman": "sculpture",
    "child": "doll",
    "face": "mask",
    "hand": "glove",
}

def _apply_collocation_swap(text: str, rng: random.Random) -> Optional[str]:
    """Swap a full object word/phrase with a conflicting collocation.

    Uses word-boundary regex matching to avoid accidental substring edits
    (e.g., replacing "car" inside "carpet").
    """

    keys = list(_OBJECT_COLLOCATION_SWAPS.keys())
    rng.shuffle(keys)

    for obj in keys:
        pattern = re.compile(rf"\b{re.escape(obj)}\b", flags=re.IGNORECASE)
        m = pattern.search(text)
        if m is None:
            continue

        replacement = _OBJECT_COLLOCATION_SWAPS[obj]
        token = m.group(0)
        if token and token[0].isupper():
            replacement = replacement[0].upper() + replacement[1:]

        return text[: m.start()] + replacement + text[m.end() :]

    return None

def _neg_conflict(text: str, rng: random.Random) -> str:
    """
    Inject controlled negation/conflict to probe shortcut reliance (E6 in paper).

    Paper spec: inject "no <object>" OR swap background-object collocations.
    Purpose: deliberately misalign textual priors with image to probe shortcuts.

    Two mechanisms (applied probabilistically):
    1. Object collocation swap (50% chance): e.g., "crab" → "spider"
       - Breaks statistical co-occurrence patterns from training data
    2. Negation injection: e.g., "Is there a dog" → "Is there no dog"
       - Creates awkward phrasing that conflicts with typical distributions

    Research justification:
    - Counterfactual VQA (Niu et al., CVPR 2021) shows language bias probing
      via negation reveals shortcut reliance in VQA models.
    - The awkward "no <object>" phrasing is intentional - it creates a
      textual pattern that conflicts with typical training distributions.

    Args:
        text: Input text
        rng: Random number generator for reproducibility
    """
    t = text

    if rng.random() < 0.5:
        swapped = _apply_collocation_swap(t, rng)
        if swapped is not None:
            return swapped

    if t.startswith("Is there a ") and " no " not in t:
        return t.replace("Is there a ", "Is there no ", 1)
    if t.startswith("Is there an ") and " no " not in t:
        return t.replace("Is there an ", "Is there no ", 1)
    if t.startswith("Is there ") and " no " not in t:
        return t.replace("Is there ", "Is there no ", 1)

    if t.startswith("Are there any ") and " no " not in t:
        return t.replace("Are there any ", "Are there no ", 1)
    if t.startswith("Are there ") and " no " not in t:
        return t.replace("Are there ", "Are there no ", 1)

    if t.startswith("Can you see a ") and " no " not in t:
        return t.replace("Can you see a ", "Can you see no ", 1)
    if t.startswith("Can you see ") and " no " not in t:
        return t.replace("Can you see ", "Can you see no ", 1)

    if t.startswith("Do you see ") and " no " not in t:
        return t.replace("Do you see ", "Do you see no ", 1)

    swapped = _apply_collocation_swap(t, rng)
    if swapped is not None:
        return swapped

    return _ctx_rephrase(t, rng)

@dataclass
class BaseExample:
    idx: int
    example_id: str
    image_path: str
    text: str

class EnvMaker:
    """
    Generates deterministic environment views for a base dataset (K=7 per paper).

    Environments (Section 2.2):
    - E1 orig: Original image and text (baseline)
    - E2 img_mismatch: Different image, same text (tests image dependence)
    - E3 mask: Gray block masking (r=0.2)
    - E4 appearance: Blur + color jitter + downsampling
    - E5 paraphrase: Semantic-preserving text rewrite
    - E6 neg_conflict: Negation injection + object collocation swap
    - E7 ctx_rephrase: Context/framing modification

    All perturbations are deterministic given the example ID and seed.
    """

    def __init__(
        self,
        base_examples: List[BaseExample],
        seed0: int = 0,
        mismatch_stride: int = 997,
    ):
        """
        Args:
            base_examples: List of BaseExample objects
            seed0: Base random seed for reproducibility
            mismatch_stride: Offset for img_mismatch (should be coprime to len)
        """
        self.base = base_examples
        self.seed0 = seed0
        self.stride = mismatch_stride

    def get(self, ex: BaseExample, env: str) -> Tuple[Image.Image, str]:
        """
        Get the (image, text) pair for a given example and environment.

        Args:
            ex: The base example
            env: Environment name

        Returns:
            Tuple of (PIL.Image, str)
        """
        rng = _rng_for(ex.example_id, env, self.seed0)

        if env == "img_mismatch":
            j = (ex.idx + self.stride) % len(self.base)
            img_path = self.base[j].image_path
            text = ex.text
        else:
            img_path = ex.image_path
            text = ex.text

        img = Image.open(img_path).convert("RGB")

        if env == "orig":
            return img, text
        if env == "img_mismatch":
            return img, text
        if env == "mask":
            return _apply_mask(img, rng), text
        if env == "appearance":
            return _apply_appearance(img, rng), text

        if env == "paraphrase":
            return img, _paraphrase(text, rng)
        if env == "neg_conflict":
            return img, _neg_conflict(text, rng)
        if env == "ctx_rephrase":
            return img, _ctx_rephrase(text, rng)

        raise ValueError(f"Unknown env: {env}")

    @staticmethod
    def available_envs() -> List[str]:
        """Return list of all supported environment names (K=7)."""
        return ENV_LIST_K7
