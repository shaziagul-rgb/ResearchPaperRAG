from __future__ import annotations

from .config import MAX_EVIDENCE_PER_CATEGORY
from .retrieval import CATEGORY_CONFIG, EvidenceRetriever


CATEGORIES = list(CATEGORY_CONFIG.keys())


def confidence_from_evidence(
    evidence: list[dict],
) -> str:
    """
    Convert retrieval strength into a simple confidence label.

    This is a retrieval confidence indicator, not a claim that
    the source itself is complete or reproducible.
    """

    if not evidence:
        return "none"

    best_score = max(
        float(item.get("score", 0.0))
        for item in evidence
    )

    if best_score >= 0.75:
        return "strong"

    if best_score >= 0.50:
        return "moderate"

    return "weak"


def classify_status(
    evidence: list[dict],
) -> str:
    """
    Classify a category based on the amount and strength of
    retrieved evidence.
    """

    if not evidence:
        return "missing"

    best_score = max(
        float(item.get("score", 0.0))
        for item in evidence
    )

    if best_score >= 0.50:
        return "found"

    return "partial"


def _evidence_to_dict(item) -> dict:
    """
    Convert RetrievedEvidence into a JSON-safe dictionary.
    """

    return {
        "page": item.page,
        "text": item.text,
        "score": round(
            float(item.score),
            3,
        ),
        "section": item.section,
        "semantic_score": round(
            float(item.semantic_score),
            3,
        ),
        "keyword_score": round(
            float(item.keyword_score),
            3,
        ),
    }


def _is_scope_evidence(
    item: dict,
) -> bool:
    """
    Identify evidence that describes the chapter's scope,
    purpose, coverage, or organisation.

    Encyclopedia/reference chapters often do not contain an
    explicit 'Research Aim' heading, so scope must also be
    detected from introductory prose.
    """

    text = item["text"].lower()
    section = (item.get("section") or "").lower()

    if section not in {
        "abstract",
        "introduction",
        "background",
        "overview",
        "scope",
    }:
        return False

    scope_terms = [
        "this chapter",
        "this entry",
        "this article",
        "this contribution",
        "the chapter",
        "the article",
        "the present chapter",
        "the present article",
        "we discuss",
        "we examine",
        "we review",
        "we provide",
        "focuses on",
        "focuses upon",
        "aims to",
        "aims at",
        "objective",
        "purpose",
        "overview",
        "reviews the",
        "reviews research",
        "research on localization",
        "localization research",
    ]

    return any(
        term in text
        for term in scope_terms
    )


def _retrieve_scope_evidence(
    chunks: list[dict],
    retriever: EvidenceRetriever,
) -> list:
    """
    Retrieve scope evidence using both semantic retrieval
    and explicit introductory-scope detection.
    """

    query = (
        "chapter aim purpose scope overview "
        "what this chapter discusses examines reviews "
        "research on localization"
    )

    candidates = retriever.retrieve_question(
        chunks,
        query,
        top_k=8,
    )

    candidate_dicts = [
        _evidence_to_dict(item)
        for item in candidates
    ]

    explicit = [
        item
        for item in candidate_dicts
        if _is_scope_evidence(item)
    ]

    if explicit:
        explicit.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return explicit[
            :MAX_EVIDENCE_PER_CATEGORY
        ]

    return candidates[
        :MAX_EVIDENCE_PER_CATEGORY
    ]


def analyse(
    chunks: list[dict],
) -> dict:
    """
    Analyse a document by retrieving evidence for each
    predefined research-analysis category.
    """

    retriever = EvidenceRetriever()

    categories = []

    for category in CATEGORIES:

        if category == "Research Aim / Scope":
            evidence_objects = _retrieve_scope_evidence(
                chunks,
                retriever,
            )
        else:
            evidence_objects = retriever.retrieve_category(
                chunks,
                category,
                top_k=MAX_EVIDENCE_PER_CATEGORY,
            )

        evidence = [
            _evidence_to_dict(item)
            for item in evidence_objects
        ]

        status = classify_status(
            evidence,
        )

        confidence = confidence_from_evidence(
            evidence,
        )

        primary = (
            evidence[0]
            if evidence
            else None
        )

        alternatives = (
            evidence[1:]
            if len(evidence) > 1
            else []
        )

        categories.append(
            {
                "category": category,
                "status": status,
                "confidence": confidence,
                "page": (
                    primary["page"]
                    if primary
                    else None
                ),
                "section": (
                    primary["section"]
                    if primary
                    else None
                ),
                "evidence": (
                    primary["text"]
                    if primary
                    else None
                ),
                "score": (
                    primary["score"]
                    if primary
                    else None
                ),
                "alternatives": alternatives,
            }
        )

    applicable_categories = len(
        [
            item
            for item in categories
            if item["status"] != "not_applicable"
        ]
    )

    found = len(
        [
            item
            for item in categories
            if item["status"] == "found"
        ]
    )

    partial = len(
        [
            item
            for item in categories
            if item["status"] == "partial"
        ]
    )

    missing = len(
        [
            item
            for item in categories
            if item["status"] == "missing"
        ]
    )

    evidence_coverage = round(
        (
            (found + (partial * 0.5))
            / applicable_categories
        )
        * 100
    ) if applicable_categories else 0

    evidence_gaps = [
        item["category"]
        for item in categories
        if item["status"] == "missing"
    ]

    return {
        "categories": categories,
        "summary": {
            "total_categories": len(CATEGORIES),
            "applicable_categories": applicable_categories,
            "found": found,
            "partial": partial,
            "missing": missing,
            "not_applicable": 0,
            "evidence_coverage": evidence_coverage,
        },
        "evidence_gaps": evidence_gaps,
    }