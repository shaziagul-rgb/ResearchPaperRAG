"""Plain-language feedback for an /api/analyze result.

This turns the category table into a short paragraph a person can
read without decoding scores and status labels. Nothing here is
specific to any one paper's content -- it only reacts to the status
labels, the inferred paper type, and section presence.
"""

from __future__ import annotations


def _clause_for(item: dict) -> str | None:
    label = item["category"]
    status = item["status"]

    if status == "found" and item.get("page"):
        return f"{label} is clearly supported (page {item['page']})"

    if status == "partial" and item.get("page"):
        return f"{label} has some support but is thin (page {item['page']})"

    if status == "missing":
        return f"{label} could not be found"

    return None


def build_feedback(
    profile: dict,
    category_results: list[dict],
    coverage: float,
) -> str:
    """Compose a short, readable summary of the analysis."""

    found = [c for c in category_results if c["status"] == "found"]
    partial = [c for c in category_results if c["status"] == "partial"]
    missing = [c for c in category_results if c["status"] == "missing"]
    not_applicable = [
        c for c in category_results if c["status"] == "not_applicable"
    ]

    sentences = []

    article = "an" if profile["paper_type"][0].lower() in "aeiou" else "a"
    sentences.append(
        f"This looks like {article} {profile['paper_type']} "
        f"({profile['total_chunks']} text chunks analysed)."
    )

    if found:
        names = ", ".join(item["category"] for item in found)
        sentences.append(f"Well-supported: {names}.")

    if partial:
        names = ", ".join(item["category"] for item in partial)
        sentences.append(
            f"Partially supported (present but with weaker evidence): {names}."
        )

    if missing:
        names = ", ".join(item["category"] for item in missing)
        sentences.append(
            f"No solid evidence was found for: {names}. "
            "This may mean the paper genuinely doesn't cover it, "
            "or that the wording differs from what was searched for."
        )

    if not_applicable:
        names = ", ".join(item["category"] for item in not_applicable)
        sentences.append(
            f"Treated as not applicable to this kind of paper: {names}."
        )

    sentences.append(f"Overall estimated evidence coverage: {coverage}%.")

    return " ".join(sentences)
