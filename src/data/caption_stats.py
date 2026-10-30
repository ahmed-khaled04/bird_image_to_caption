"""Caption statistics for EDA: problems in the raw `description` column and the effect of cleaning.

Word statistics come from the train split; test is only used to measure how many of its
words a train vocabulary would miss.
"""
import re
from collections import Counter

import pandas as pd
from spellchecker import SpellChecker

from src.data.caption_cleaning import TYPO_FIXES, clean_caption


def split_lines(descriptions: list) -> pd.Series:
    """Every line of every description exactly as stored, including the empty last line."""
    return pd.Series([line for description in descriptions for line in description.split("\n")])


def per_image_counts(descriptions: list, cleaned: list) -> pd.DataFrame:
    """Per image: raw lines, empty lines, duplicate lines (ignoring case) and captions kept by cleaning."""
    rows = []
    for description, captions in zip(descriptions, cleaned):
        lines = description.split("\n")
        non_empty = [line.strip().lower() for line in lines if line.strip()]
        rows.append({
            "lines": len(lines),
            "empty": len(lines) - len(non_empty),
            "duplicates": len(non_empty) - len(set(non_empty)),
            "kept": len(captions),
        })
    return pd.DataFrame(rows)


def cleaning_funnel(lines: pd.Series, cleaned: list) -> pd.Series:
    """Caption count after each cleaning stage: split, drop empty, drop duplicates."""
    non_empty = sum(1 for line in lines if clean_caption(line))
    return pd.Series({
        "lines after split": len(lines),
        "non-empty": non_empty,
        "unique (kept)": sum(len(captions) for captions in cleaned),
    })


def formatting_issues(lines: pd.Series) -> pd.Series:
    """How many raw lines show each formatting problem."""
    checks = {
        "has uppercase": lines.str.contains(r"[A-Z]"),
        "double spaces": lines.str.contains(r"\s{2,}"),
        "leading/trailing space": lines != lines.str.strip(),
        "has digits": lines.str.contains(r"\d"),
        "non-ASCII characters": ~lines.map(str.isascii),
        "ends with period": lines.str.strip().str.endswith("."),
    }
    return pd.Series({name: int(mask.sum()) for name, mask in checks.items()})


def symbol_counts(lines: pd.Series) -> Counter:
    """Frequency of every character that is neither a letter, a digit nor whitespace."""
    return Counter(ch for line in lines for ch in line if not ch.isalnum() and not ch.isspace())


def word_counts(texts) -> Counter:
    """Word frequencies after lowercasing and turning every non-letter into a space."""
    return Counter(word for text in texts for word in re.sub(r"[^a-z\s]", " ", text.lower()).split())


def unknown_words(words: Counter) -> list:
    """(word, count) for words the English spell checker does not know: typos and bird terms."""
    unknown = SpellChecker().unknown(words.keys())
    return sorted(((word, words[word]) for word in unknown), key=lambda item: (-item[1], item[0]))


def fixed_typos(words: Counter) -> list:
    """(typo, count) for the TYPO_FIXES entries that occur in the raw captions."""
    found = ((typo, words[typo]) for typo in TYPO_FIXES if typo in words)
    return sorted(found, key=lambda item: (-item[1], item[0]))


def vocabulary_coverage(train_words: Counter, test_words: Counter, max_min_count: int = 20) -> pd.DataFrame:
    """Keeping only train words seen at least `min_count` times: vocabulary size, share of train
    word occurrences it covers, and share of test word occurrences that would become <unk>."""
    train_total, test_total = sum(train_words.values()), sum(test_words.values())
    rows = []
    for min_count in range(1, max_min_count + 1):
        vocab = {word for word, count in train_words.items() if count >= min_count}
        rows.append({
            "min_count": min_count,
            "vocab_size": len(vocab),
            "train_coverage": sum(train_words[word] for word in vocab) / train_total,
            "test_unk_rate": sum(count for word, count in test_words.items() if word not in vocab) / test_total,
        })
    return pd.DataFrame(rows)


def caption_problems(cleaned: list) -> dict:
    """Problems left after cleaning, one count each. Every value should be 0."""
    valid = re.compile(r"[a-z]+( [a-z]+)*")  # lowercase words separated by single spaces
    return {
        "images without captions": sum(1 for captions in cleaned if not captions),
        "captions that are not lowercase letters and single spaces":
            sum(1 for captions in cleaned for caption in captions if not valid.fullmatch(caption)),
        "images with a repeated caption": sum(1 for captions in cleaned if len(set(captions)) != len(captions)),
    }
