"""Build a lightweight, paper-type-agnostic profile of a document.

The profile is used for two things:

1. Deciding which evidence categories are even applicable to this
   paper (e.g. a pure literature-analysis paper may have no
   "Theoretical Framework" section, and that is not a defect).
2. Describing the paper in plain language for the feedback summary.

Everything here is derived generically from section labels and a
small set of domain-neutral phrase cues -- nothing is tied to any
specific paper or field.
"""

from __future__ import annotations

import re
from collections import Counter

_METHODS_SECTIONS = {"methods", "methodology"}
_RESULTS_SECTIONS = {"results", "discussion"}
_THEORY_HINTS = (
    "theoretical framework",
    "conceptual framework",
    "theory of",
    "grounded in",
    "framework for",
    "propose a model",
    "propose a theory",
)
_EMPIRICAL_HINTS = (
    "we conducted",
    "we performed",
    "we ran",
    "participants",
    "randomized",
    "was measured",
    "experiment",
    "we trained",
    "we tested",
    "dataset",
)
_REVIEW_HINTS = (
    "we reviewed",
    "we examined",
    "we surveyed",
    "literature review",
    "systematic review",
    "we analyzed",
    "papers were",
    "articles were",
    "studies were",
)


def _get(chunk, key, default=None):
    if isinstance(chunk, dict):
        return chunk.get(key, default)

    return getattr(chunk, key, default)


def build_profile(chunks: list) -> dict:
    """Summarise a document's structure from its chunks."""

    total = len(chunks)

    sections = Counter(
        (_get(chunk, "section") or "unspecified")
        for chunk in chunks
    )

    all_text = " ".join(
        (_get(chunk, "text") or "").lower()
        for chunk in chunks
    )

    def frac(names: set) -> float:
        if total == 0:
            return 0.0

        return sum(sections[name] for name in names) / total

    def hits(phrases) -> int:
        return sum(1 for phrase in phrases if phrase in all_text)

    methods_frac = frac(_METHODS_SECTIONS)
    results_frac = frac(_RESULTS_SECTIONS)
    theory_hits = hits(_THEORY_HINTS)
    empirical_hits = hits(_EMPIRICAL_HINTS)
    review_hits = hits(_REVIEW_HINTS)

    has_methods_section = methods_frac > 0
    has_results_section = results_frac > 0

    if review_hits >= 2 and review_hits >= empirical_hits:
        paper_type = "literature review / survey"
    elif empirical_hits >= 2 or (has_methods_section and has_results_section):
        paper_type = "empirical study"
    elif theory_hits >= 1 and not has_methods_section:
        paper_type = "conceptual / theoretical paper"
    else:
        paper_type = "general research paper"

    return {
        "paper_type": paper_type,
        "total_chunks": total,
        "sections_present": dict(sections),
        "has_methods_section": has_methods_section,
        "has_results_section": has_results_section,
        "has_theory_language": theory_hits >= 1,
        "is_review_style": review_hits >= 2 and review_hits >= empirical_hits,
    }


# Categories whose absence can be a genuine feature of the paper's
# type rather than a gap. A category is only ever downgraded to
# "not applicable" when BOTH the structural signal is absent AND the
# retrieved evidence for it is weak -- so real evidence always wins.
_CONDITIONALLY_APPLICABLE = {
    "Theoretical Framework": lambda profile: (
        profile["has_theory_language"]
        or not (
            profile["has_methods_section"]
            and profile["has_results_section"]
        )
    ),
    "Research Areas / Themes": lambda profile: (
        profile["is_review_style"]
        or not (
            profile["has_methods_section"]
            and profile["has_results_section"]
        )
    ),
    "Methods Discussed": lambda profile: (
        profile["has_methods_section"]
        or profile["has_results_section"]
    ),
}


def structurally_applicable(category: str, profile: dict) -> bool:
    """True unless the paper's structure suggests this category doesn't fit."""

    check = _CONDITIONALLY_APPLICABLE.get(category)

    if check is None:
        return True

    return bool(check(profile))
