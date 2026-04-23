from __future__ import annotations

import re
from typing import Iterable


def whitespace_normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_math_answer(text: str) -> str:
    text = text.strip()
    text = text.replace("\\boxed{", "")
    text = text.replace("}", "")
    text = text.replace("$", "")
    text = text.replace("\n", " ")
    text = re.sub(r"^answer\s*[:：]\s*", "", text, flags=re.I)
    return whitespace_normalize(text)


def keyword_tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_\-\+\\/\.]+", text.lower())


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def numbered_block(items: Iterable[str]) -> str:
    return "\n".join(f"{idx}. {item}" for idx, item in enumerate(items, start=1))
