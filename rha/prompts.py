import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

NEEDLE_START = "[NEEDLE_START]"
NEEDLE_END = "[NEEDLE_END]"

WORDS = [
    "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit",
    "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore", "et", "dolore",
    "magna", "aliqua", "enim", "ad", "minim", "veniam", "quis", "nostrud",
    "exercitation", "ullamco", "laboris", "nisi", "aliquip", "ex", "ea",
    "commodo", "consequat", "duis", "aute", "irure", "reprehenderit",
    "voluptate", "velit", "esse", "cillum", "eu", "fugiat", "nulla",
    "pariatur", "excepteur", "sint", "occaecat", "cupidatat", "non", "proident",
]

NAMES = ["Ari", "Bela", "Cato", "Dina", "Emil", "Faye", "Galen", "Hera"]
OBJECTS = ["red folder", "blue ledger", "silver case", "green dossier", "amber file"]
CODENAMES = ["KITE", "ORCHID", "RIVER", "EMBER", "NOVA", "QUARTZ"]


@dataclass(frozen=True)
class RetrievalExample:
    prompt: str
    gold: str
    variant: str
    seed: int
    target_tokens: int
    needle_frac: float

    def as_dict(self) -> Dict:
        return {
            "prompt": self.prompt,
            "gold": self.gold,
            "variant": self.variant,
            "seed": self.seed,
            "target_tokens": self.target_tokens,
            "needle_frac": self.needle_frac,
        }


def seeded_secret(seed: int) -> str:
    rng = random.Random(seed)
    return f"{rng.randint(100000, 999999)}"


def seeded_decoy(seed: int) -> str:
    rng = random.Random(seed + 10_000_019)
    return f"{rng.randint(100000, 999999)}"


def filler_words(n_words: int, seed: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(max(0, n_words)))


def _needle_sentence(variant: str, secret: str, decoy: str, seed: int) -> Tuple[str, str]:
    rng = random.Random(seed)
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    codename = rng.choice(CODENAMES)

    if variant == "literal":
        return f"{NEEDLE_START} The access code is {secret}. {NEEDLE_END}", "What is the access code?"
    if variant == "alias":
        return f"{NEEDLE_START} Codename {codename} is assigned key {secret}. {NEEDLE_END}", f"What key is assigned to codename {codename}?"
    if variant == "paraphrase":
        return f"{NEEDLE_START} To unlock the archive, use the six digit sequence {secret}. {NEEDLE_END}", "Which six digit sequence unlocks the archive?"
    if variant == "relational":
        return (
            f"{NEEDLE_START} {name} owns the {obj}. {name}'s badge number is {secret}. {NEEDLE_END}",
            f"What is the badge number of the person who owns the {obj}?",
        )
    if variant == "distractor_heavy":
        return (
            f"{NEEDLE_START} Ignore ticket {decoy}; the access code for {codename} is {secret}. {NEEDLE_END}",
            f"What is the access code for {codename}?",
        )
    raise ValueError(f"Unknown retrieval variant: {variant}")


def build_prompt_with_filler(total_filler_words: int, needle_frac: float, variant: str, secret: str, decoy: str, seed: int) -> str:
    needle, question = _needle_sentence(variant, secret, decoy, seed)
    before_n = int(round(total_filler_words * needle_frac))
    after_n = max(0, total_filler_words - before_n)
    before = filler_words(before_n, seed + 1)
    after = filler_words(after_n, seed + 2)

    return (
        "You are reading a long document. Answer the final question using only the relevant fact from the document.\n\n"
        "Document:\n"
        f"{before}\n"
        f"{needle}\n"
        f"{after}\n\n"
        f"Distractor note: the number {decoy} is unrelated unless the document explicitly says otherwise.\n\n"
        f"Question: {question}\n"
        "Answer with only the six digits."
    )


def prompt_token_len(tokenizer, prompt: str) -> int:
    return len(tokenizer(prompt, add_special_tokens=False).input_ids)


def calibrate_filler_words(
    tokenizer,
    target_tokens: int,
    needle_frac: float,
    variant: str,
    secret: str,
    decoy: str,
    seed: int,
    max_words: Optional[int] = None,
) -> int:
    hi = max_words or max(256, target_tokens * 2)
    lo = 0
    best_n = 0
    best_err = float("inf")

    for _ in range(32):
        mid = (lo + hi) // 2
        prompt = build_prompt_with_filler(mid, needle_frac, variant, secret, decoy, seed)
        length = prompt_token_len(tokenizer, prompt)
        err = abs(length - target_tokens)
        if err < best_err:
            best_n = mid
            best_err = err
        if length < target_tokens:
            lo = mid + 1
        else:
            hi = mid - 1
        if lo > hi:
            break

    return best_n


def make_retrieval_example(
    tokenizer,
    target_tokens: int,
    needle_frac: float,
    seed: int,
    variant: str = "literal",
) -> RetrievalExample:
    secret = seeded_secret(seed)
    decoy = seeded_decoy(seed)
    n_words = calibrate_filler_words(
        tokenizer=tokenizer,
        target_tokens=target_tokens,
        needle_frac=needle_frac,
        variant=variant,
        secret=secret,
        decoy=decoy,
        seed=seed,
    )
    prompt = build_prompt_with_filler(n_words, needle_frac, variant, secret, decoy, seed)
    return RetrievalExample(
        prompt=prompt,
        gold=secret,
        variant=variant,
        seed=seed,
        target_tokens=target_tokens,
        needle_frac=needle_frac,
    )


def build_dataset(
    tokenizer,
    variants: List[str],
    n_per_variant: int,
    target_tokens: int,
    needle_frac: float,
    seed_base: int,
) -> List[Dict]:
    rows = []
    for variant_idx, variant in enumerate(variants):
        for i in range(n_per_variant):
            seed = seed_base + variant_idx * 100_000 + i
            rows.append(
                make_retrieval_example(
                    tokenizer=tokenizer,
                    target_tokens=target_tokens,
                    needle_frac=needle_frac,
                    seed=seed,
                    variant=variant,
                ).as_dict()
            )
    return rows
