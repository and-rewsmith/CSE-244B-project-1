from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
import random
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


def top_source_words(
    tokenized_pairs: Iterable[tuple[list[str], list[str]]],
    top_n: int = 10,
) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for zh_tokens, _ in tokenized_pairs:
        for token in zh_tokens:
            if token == NULL_TOKEN:
                continue
            counter[token] += 1
    return counter.most_common(top_n)


def report_top_translations(
    t: dict[str, dict[str, float]],
    tokenized_pairs: Iterable[tuple[list[str], list[str]]],
    top_f: int = 10,
    top_e: int = 5,
) -> None:
    top_sources = top_source_words(tokenized_pairs, top_n=top_f)
    print("\ntop translations for frequent source words:")
    for f, count in top_sources:
        translations = t.get(f, {})
        # reverse because we want the most probable translations first!
        # TODO: slice before sorting to save time (runs fast so I am ignoring this for now)
        top_translations = sorted(
            translations.items(), key=lambda item: item[1], reverse=True
        )[:top_e]
        formatted = ", ".join(f"{e}:{prob:.4f}" for e, prob in top_translations)
        print(f"{f} (count={count}) -> {formatted}")


# PART 5: Eval with perplexity


def build_target_vocab(
    tokenized_pairs: Iterable[tuple[list[str], list[str]]],
) -> list[str]:
    vocab: set[str] = set()
    for _, en_tokens in tokenized_pairs:
        vocab.update(en_tokens)
    return sorted(vocab)


def sentence_log_prob(
    f_tokens: list[str],
    e_tokens: list[str],
    t: dict[str, dict[str, float]],
) -> float:
    """
    Compute log P(e | f) under IBM Model 1.
    """
    if not f_tokens or not e_tokens:
        ValueError("f_tokens and e_tokens cannot be empty")

    log_prob = 0.0
    for e in e_tokens:
        s_total = 0.0
        for f in f_tokens:
            s_total += t.get(f, {}).get(e, 0.0)
        s_total = max(s_total, 1e-12)
        log_prob += math.log(s_total)

    return log_prob


def perplexity_from_log_prob(log_prob: float, length: int) -> float:
    if length <= 0:
        return float("inf")
    return math.exp(-log_prob / length)


def sample_random_sentence(vocab: list[str], length: int) -> list[str]:
    return [random.choice(vocab) for _ in range(length)]


def evaluate_perplexity(
    tokenized_pairs: list[tuple[list[str], list[str]]],
    table: dict[str, dict[str, float]],
    trials: int = 5,
) -> None:
    """
    Compare perplexity of real translations vs random target sentences
    of the same length.
    """
    vocab = build_target_vocab(tokenized_pairs)
    if not vocab:
        raise ValueError("Target vocabulary is empty.")

    random.seed(0)
    print("\nPerplexity comparison (real vs random):")
    for i in range(trials):
        f_tokens, e_tokens = random.choice(tokenized_pairs)
        random_e = sample_random_sentence(vocab, len(e_tokens))

        log_p_real = sentence_log_prob(f_tokens, e_tokens, table)
        log_p_rand = sentence_log_prob(f_tokens, random_e, table)

        ppl_real = perplexity_from_log_prob(log_p_real, len(e_tokens))
        ppl_rand = perplexity_from_log_prob(log_p_rand, len(e_tokens))

        print(f"Trial {i + 1}: real={ppl_real:.4f}, random={ppl_rand:.4f}")


# Main Function


def main() -> None:
    # Part 1: load parallel data (ZH -> EN)
    pairs = part1_load_parallel(limit=10_000)
    print(f"Loaded {len(pairs)} sentence pairs.")

    # Part 2: preprocess and tokenize
    tokenized_pairs = preprocess_pairs(pairs)
    print(f"Tokenized {len(tokenized_pairs)} sentence pairs.")

    # Part 3: train IBM Model 1 (EM)
    table = em_train_ibm1(tokenized_pairs, iterations=5)
    print(f"Trained translation table for {len(table)} source tokens.")

    # Part 4: report translation table
    report_top_translations(table, tokenized_pairs, top_f=10, top_e=5)

    # Part 5: evaluate perplexity
    evaluate_perplexity(tokenized_pairs, table, trials=5)


if __name__ == "__main__":
    main()
