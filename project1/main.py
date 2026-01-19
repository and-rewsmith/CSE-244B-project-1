from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from nltk.tokenize import word_tokenize
import jieba

# DOCUMENTATION: OPENSUBTITLES LANGUAGE SELECTION CHOICE
"""
OpenSubtitles Language Selection Choice. Reasoning for picking zh-CN variant was to prioritize 1) Simplified Chinese characters (avoiding traditional) and 2) Mainland China locale (broadly used).

Chinese language/locale options explanation:

- Chinese (zh):
  Generic Chinese language without region or script specified.
  May resolve to Simplified or Traditional depending on system defaults.

- Chinese (China) (zh-CN / zh-cn):
  Mainland China locale.
  Uses Simplified Chinese characters.

- Chinese (Taiwan) (zh-TW):
  Taiwan locale.
  Uses Traditional Chinese characters.

- Chinese (ZH):
  Same as "zh"; uppercase variant with no semantic difference.

- Chinese (EN):
  Invalid / non-standard locale entry (Chinese is not English).

- Chinese (ZE):
  Invalid / non-standard locale entry; likely a misconfiguration.
"""

# PART 1: Load parallel data

DATA_DIR = Path(__file__).resolve().parents[1] / "language_downloads"
EN_PATH = DATA_DIR / "OpenSubtitles.en-zh.en"
ZH_PATH = DATA_DIR / "OpenSubtitles.en-zh.zh"


def read_parallel_lines(
    en_path: Path,
    zh_path: Path,
    limit: int | None = None,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with (
        en_path.open("r", encoding="utf-8") as en_file,
        zh_path.open("r", encoding="utf-8") as zh_file,
    ):
        for i, (en_line, zh_line) in enumerate(zip(en_file, zh_file)):
            if limit is not None and i >= limit:
                break
            en = en_line.strip()
            zh = zh_line.strip()
            if not en or not zh:
                continue  # we skip here the lines that are empty for either language (big simplification)
            pairs.append((zh, en))
    return pairs


def assert_parallel_integrity(pairs: Iterable[tuple[str, str]]) -> None:
    """
    Makes sure we don't have empty sentences in the pairs.
    """
    pairs = list(pairs)
    assert pairs, "No sentence pairs loaded."
    for zh, en in pairs[:5]:
        assert zh, "Empty source sentence found."
        assert en, "Empty target sentence found."


def part1_load_parallel(limit: int = 10_000) -> list[tuple[str, str]]:
    if not EN_PATH.exists() or not ZH_PATH.exists():
        raise FileNotFoundError(f"Expected data files at {EN_PATH} and {ZH_PATH}.")
    pairs = read_parallel_lines(EN_PATH, ZH_PATH, limit=limit)
    assert_parallel_integrity(pairs)
    return pairs


# PART 2: Preprocess data and tokenize

NULL_TOKEN = "NULL"


def tokenize_english(text: str) -> list[str]:
    """
    NLTK's tokenizer works well for English.
    """
    text = text.lower()
    return word_tokenize(text)  # type: ignore


def tokenize_chinese(text: str) -> list[str]:
    """
    Chinese does not have spaces between words. We use a Chinese word
    segmenter (jieba) to create word-like units, rather than treating
    the entire sentence as a single token.
    """
    return [token for token in jieba.lcut(text) if token.strip()]


def preprocess_pairs(
    pairs: Iterable[tuple[str, str]],
) -> list[tuple[list[str], list[str]]]:
    """
    Tokenize and normalize (zh, en) sentence pairs.

    We lowercase English to reduce sparsity, and we prepend a NULL token
    to the Chinese side so the model can align English words to "nothing"
    when needed.

    This NULL token is not used on the target side because we are modeling:
    _target words aligning to some source word or to nothing_"

    We return data of type:
    list[tuple[list[str], list[str]]]
    """
    tokenized: list[tuple[list[str], list[str]]] = []
    for zh, en in pairs:
        zh_tokens = [NULL_TOKEN] + tokenize_chinese(zh)
        en_tokens = tokenize_english(en)
        if not zh_tokens or not en_tokens:
            continue
        tokenized.append((zh_tokens, en_tokens))
    assert_preprocessing_integrity(tokenized)
    return tokenized


def assert_preprocessing_integrity(
    tokenized_pairs: Iterable[tuple[list[str], list[str]]],
) -> None:
    tokenized_pairs = list(tokenized_pairs)
    assert tokenized_pairs, "No tokenized pairs produced."
    for zh_tokens, en_tokens in tokenized_pairs[:5]:
        assert zh_tokens[0] == NULL_TOKEN, "Missing NULL token on source side."
        assert all(token.strip() for token in zh_tokens), "Empty source token."
        assert all(token.strip() for token in en_tokens), "Empty target token."


# PART 3: Train IBM model 1 with EM


def initialize_translation_table(
    tokenized_pairs: Iterable[tuple[list[str], list[str]]],
) -> dict[str, dict[str, float]]:
    """
    Initialize t(e|f) uniformly over all target words that co-occur with f
    in the same sentence pair. This keeps the parameter set compact and
    gives non-zero probability mass to observed translation candidates.
    """
    co_occurring_targets: dict[str, set[str]] = defaultdict(set)
    for zh_tokens, en_tokens in tokenized_pairs:
        for f in zh_tokens:
            co_occurring_targets[f].update(en_tokens)

    t: dict[str, dict[str, float]] = {}
    for f, e_set in co_occurring_targets.items():
        if not e_set:
            continue
        uniform_prob = 1.0 / len(e_set)
        t[f] = {e: uniform_prob for e in e_set}

    # NOTE: indexing here is just a storage choice
    # i.e. prob = t[f][e] is t(e|f)

    return t


def em_train_ibm1(
    tokenized_pairs: Iterable[tuple[list[str], list[str]]],
    iterations: int = 5,
) -> dict[str, dict[str, float]]:
    """
    Train IBM model 1.
    """
    tokenized_pairs = list(tokenized_pairs)
    table = initialize_translation_table(tokenized_pairs)

    for _ in range(iterations):
        # initialize
        count: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        total_f: dict[str, float] = defaultdict(float)

        for zh_tokens, en_tokens in tokenized_pairs:
            # compute normalization
            s_total: dict[str, float] = {}
            for e in en_tokens:
                s_total_e = 0.0
                for f in zh_tokens:
                    s_total_e += table.get(f, {}).get(e, 0.0)
                s_total[e] = s_total_e

            # collect counts
            for e in en_tokens:
                denom = s_total[e]
                if denom == 0.0:
                    continue
                for f in zh_tokens:
                    frac = table.get(f, {}).get(e, 0.0) / denom
                    count[f][e] += frac
                    total_f[f] += frac

        # estimate probabilities
        for f, e_dict in count.items():
            total = total_f[f]
            if total == 0.0:  # i assume we skip this case
                continue
            table[f] = {e: c / total for e, c in e_dict.items()}

    assert_model_integrity(table)
    return table


def assert_model_integrity(t: dict[str, dict[str, float]]) -> None:
    """Sanity-check that a few f distributions sum to ~1."""
    checked = 0
    for f, e_dict in t.items():
        total = sum(e_dict.values())
        assert 0.99 <= total <= 1.01, f"Probabilities for {f} sum to {total}"
        checked += 1
        if checked >= 5:
            break


# PART 4: Report translation table

# PART 5: Eval with perplexity

# MAIN FUNCTION


def main() -> None:
    # Part 1: load parallel data (ZH -> EN)
    pairs = part1_load_parallel(limit=10_000)
    print(f"Loaded {len(pairs)} sentence pairs.")

    # Part 2: preprocess and tokenize
    tokenized_pairs = preprocess_pairs(pairs)
    print(f"Tokenized {len(tokenized_pairs)} sentence pairs.")

    # Part 3: train IBM Model 1 (EM)
    t = em_train_ibm1(tokenized_pairs, iterations=5)
    print(f"Trained translation table for {len(t)} source tokens.")


if __name__ == "__main__":
    main()
