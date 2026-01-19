from __future__ import annotations

from collections import Counter
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "language_downloads"
EN_PATH = DATA_DIR / "OpenSubtitles.en-zh.en"
ZH_PATH = DATA_DIR / "OpenSubtitles.en-zh.zh"


def read_parallel(
    en_path: Path,
    zh_path: Path,
    limit: int | None = None,
) -> list[tuple[str, str]]:
    """Read parallel EN/ZH lines into sentence pairs."""
    pairs: list[tuple[str, str]] = []
    with en_path.open("r", encoding="utf-8") as en_file, zh_path.open(
        "r", encoding="utf-8"
    ) as zh_file:
        for i, (en_line, zh_line) in enumerate(zip(en_file, zh_file)):
            if limit is not None and i >= limit:
                break
            en = en_line.strip()
            zh = zh_line.strip()
            if not en or not zh:
                continue
            pairs.append((zh, en))
    return pairs


def summarize_pairs(pairs: list[tuple[str, str]], sample: int = 3) -> None:
    """Print dataset summary and a few samples."""
    print(f"Loaded pairs: {len(pairs)}")
    if not pairs:
        return

    zh_lengths = [len(zh.split()) for zh, _ in pairs]
    en_lengths = [len(en.split()) for _, en in pairs]
    print(
        "Avg tokenized lengths (whitespace): "
        f"zh={sum(zh_lengths)/len(zh_lengths):.2f}, "
        f"en={sum(en_lengths)/len(en_lengths):.2f}"
    )

    print("\nSample pairs (ZH -> EN):")
    for zh, en in pairs[:sample]:
        print(f"ZH: {zh}")
        print(f"EN: {en}")
        print("---")

    zh_counter = Counter(" ".join(zh for zh, _ in pairs).split())
    en_counter = Counter(" ".join(en for _, en in pairs).split())
    print("Top 10 zh tokens:", zh_counter.most_common(10))
    print("Top 10 en tokens:", en_counter.most_common(10))


def main() -> None:
    if not EN_PATH.exists() or not ZH_PATH.exists():
        raise FileNotFoundError(
            f"Expected data files at {EN_PATH} and {ZH_PATH}."
        )

    # Default to Chinese -> English direction.
    pairs = read_parallel(EN_PATH, ZH_PATH, limit=10000)
    summarize_pairs(pairs)


if __name__ == "__main__":
    main()