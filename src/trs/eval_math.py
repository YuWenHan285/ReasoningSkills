from __future__ import annotations

from trs.utils.text import normalize_math_answer


def exact_match(prediction: str, answer: str | None) -> bool | None:
    if answer is None:
        return None
    return normalize_math_answer(prediction) == normalize_math_answer(answer)
