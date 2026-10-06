"""Detect PDF text layers that extract as garbage (broken font encodings)."""

from __future__ import annotations

import unicodedata


def text_quality(text: str) -> dict:
    """Return simple readability statistics for extracted text."""

    chars = [ch for ch in text if not ch.isspace()]
    total = len(chars)

    if total == 0:
        return {"chars": 0, "letter_ratio": 0.0, "control_ratio": 0.0}

    letters = sum(1 for ch in chars if ch.isalpha())
    control = sum(
        1
        for ch in chars
        if unicodedata.category(ch) in {"Cc", "Co", "Cn"}
    )

    return {
        "chars": total,
        "letter_ratio": letters / total,
        "control_ratio": control / total,
    }


def is_garbled(text: str) -> bool:
    """True when text is mostly control characters / non-letters."""

    stats = text_quality(text)

    if stats["chars"] < 20:
        return False

    return stats["control_ratio"] > 0.03 or stats["letter_ratio"] < 0.5


def strip_control_chars(text: str) -> str:
    """Remove stray control characters but keep newlines and tabs."""

    return "".join(
        ch
        for ch in text
        if ch in "\n\t" or unicodedata.category(ch) not in {"Cc", "Co", "Cn"}
    )
