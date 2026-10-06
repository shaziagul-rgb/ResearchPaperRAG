from __future__ import annotations

import re

from .config import (
    DEFAULT_TOP_K,
    FOUND_SCORE,
    MAX_EVIDENCE_PER_CATEGORY,
    MISSING_SCORE,
)
from .feedback import build_feedback
from .paper_profile import build_profile, structurally_applicable
from .retrieval import CATEGORY_CONFIG, EvidenceRetriever


CATEGORIES = list(CATEGORY_CONFIG.keys())


def confidence_from_evidence(evidence: list) -> str:
    if not evidence:
        return "none"

    best_score = max(item.score for item in evidence)

    if best_score >= 0.60:
        return "strong"

    if best_score >= FOUND_SCORE:
        return "moderate"

    return "weak"


def classify_status(
    primary,
    category: str,
    doc_best_score: float,
) -> str:
    """
    Status of a category, judged relative to the best evidence found
    anywhere else in this same document.

    Fixed absolute score thresholds don't transfer across documents:
    a short, sparse paper and a dense one produce very different raw
    score ranges. Judging each category against the strongest score
    seen elsewhere in the *same* document adapts to that automatically.
    An absolute floor/ceiling still applies so that a document where
    nothing scores well doesn't get an artificial "found".
    """

    if primary is None:
        return "missing"

    # An explicit statement of purpose/aim in the abstract or introduction
    # is direct evidence of scope even when its retrieval score is low.
    if (
        category == "Research Aim / Scope"
        and _is_scope_evidence(primary)
        and (primary.section or "").lower() in {"abstract", "introduction"}
    ):
        return "found"

    score = float(primary.score)
    relative = (score / doc_best_score) if doc_best_score > 0 else 0.0

    if score >= FOUND_SCORE and relative >= 0.55:
        return "found"

    if score < MISSING_SCORE or relative < 0.22:
        return "missing"

    return "partial"


def _evidence_to_dict(evidence) -> dict:
    return {
        "page": evidence.page,
        "text": evidence.text,
        "score": round(float(evidence.score), 3),
        "section": evidence.section,
        "semantic_score": round(float(evidence.semantic_score), 3),
        "keyword_score": round(float(evidence.keyword_score), 3),
    }


_CITATION_NOTICE_CONTEXT = re.compile(
    r"\bcit(?:e|ing|ed)\b[^.]{0,15}\bas\b",
    re.I,
)


def _contains_scope_phrase(text: str, phrases: list) -> bool:
    """
    Whether an explicit scope phrase ("this paper", "this article", ...)
    appears in a context that actually states the paper's purpose,
    rather than in a citation notice such as "Please cite this
    article as: ...", which is publisher boilerplate repeated on
    every page and would otherwise look identical to real scope
    language under plain substring matching.
    """

    lower = text.lower()

    if _CITATION_NOTICE_CONTEXT.search(lower):
        return False

    return any(phrase in lower for phrase in phrases)


# Phrases a paper commonly uses to state its own purpose or scope.
# Humanities/review-style papers tend to say "this study aims to..."
# or "this paper reviews...", while CS/engineering papers more often
# say "we introduce...", "we present...", or "in this work, we...".
# Both styles need to be covered for this to generalise across fields.
_SCOPE_PHRASES = [
    "this chapter",
    "this study",
    "this paper",
    "this research",
    "this article",
    "this work",
    "focuses on",
    "focus on",
    "aims to",
    "aim of",
    "purpose of",
    "objective",
    "we review",
    "we examine",
    "we analyse",
    "we analyze",
    "we introduce",
    "we present",
    "we propose",
    "we develop",
    "we describe",
    "we design",
    "we address",
    "we tackle",
    "we investigate",
    "in this work",
    "in this paper",
    "in this study",
    "our goal is",
    "our contribution",
    "we contribute",
    "the chapter discusses",
    "the chapter examines",
    "the article discusses",
    "the article examines",
]


def _is_scope_evidence(evidence) -> bool:
    if not evidence.text:
        return False

    text = evidence.text.lower()
    section = evidence.section.lower() if evidence.section else ""

    has_scope_term = _contains_scope_phrase(text, _SCOPE_PHRASES)

    scope_sections = [
        "abstract",
        "introduction",
        "research aim",
    ]

    in_scope_section = any(
        section_name in section
        for section_name in scope_sections
    )

    return has_scope_term or in_scope_section


def _is_theory_evidence(evidence) -> bool:
    if not evidence.text:
        return False

    text = evidence.text.lower()
    section = evidence.section.lower() if evidence.section else ""

    theory_terms = [
        "theoretical framework",
        "theoretical approach",
        "theoretical paradigm",
        "theory",
        "theoretical",
        "epistemological",
        "conceptual framework",
        "conceptual approach",
        "conceptualize",
        "conceptualise",
        "defined as",
        "definition",
        "paradigm",
    ]

    theory_sections = [
        "theoretical framework",
        "theory",
        "background",
        "conceptual framework",
    ]

    has_theory_term = any(
        term in text
        for term in theory_terms
    )

    in_theory_section = any(
        section_name in section
        for section_name in theory_sections
    )

    return has_theory_term or in_theory_section


def _is_research_direction_evidence(evidence) -> bool:
    if not evidence.text:
        return False

    text = evidence.text.lower()
    section = evidence.section.lower() if evidence.section else ""

    direction_terms = [
        "future research",
        "future work",
        "research directions",
        "research direction",
        "potential areas of research",
        "areas for future research",
        "further research",
        "further studies",
        "should be investigated",
        "should be explored",
        "needs further research",
        "need for further research",
        "research agenda",
    ]

    direction_sections = [
        "research directions",
        "future work",
        "future research",
        "conclusion",
        "conclusions",
    ]

    has_direction_term = any(
        term in text
        for term in direction_terms
    )

    in_direction_section = any(
        section_name in section
        for section_name in direction_sections
    )

    return has_direction_term or in_direction_section


def _is_method_evidence(evidence) -> bool:
    if not evidence.text:
        return False

    text = evidence.text.lower()

    method_terms = [
        "method",
        "methodology",
        "approach",
        "methodological",
        "empirical",
        "experimental",
        "corpus",
        "case study",
        "ethnographic",
        "survey",
        "interview",
        "analysis",
        "data collection",
    ]

    return any(
        term in text
        for term in method_terms
    )


def _is_study_evidence(evidence) -> bool:
    if not evidence.text:
        return False

    text = evidence.text.lower()

    study_terms = [
        "study",
        "studies",
        "empirical study",
        "empirical studies",
        "research",
        "experiment",
        "case study",
        "findings",
        "evidence",
        "results",
        "participants",
        "data",
        "corpus",
    ]

    return any(
        term in text
        for term in study_terms
    )


def _select_primary_evidence(category: str, evidence: list):
    if not evidence:
        return None

    candidates = list(evidence)

    def selection_score(item):
        score = float(item.score)

        section = item.section.lower() if item.section else ""
        text = item.text.lower() if item.text else ""

        if category == "Research Aim / Scope":
            if _is_scope_evidence(item):
                score += 0.06

            if any(
                section_name in section
                for section_name in [
                    "abstract",
                    "introduction",
                    "research aim",
                ]
            ):
                score += 0.03

            if _contains_scope_phrase(
                text,
                _SCOPE_PHRASES,
            ):
                score += 0.10

            if _is_scope_evidence(item):
                if item.page <= 2:
                    score += 0.06
                elif item.page <= 4:
                    score += 0.03

        elif category == "Theoretical Framework":
            if _is_theory_evidence(item):
                score += 0.20

            if "theoretical framework" in section:
                score += 0.15

            elif "theory" in section:
                score += 0.10

            if any(
                term in text
                for term in [
                    "theoretical framework",
                    "theoretical approach",
                    "theoretical paradigm",
                    "conceptual framework",
                ]
            ):
                score += 0.10

        elif category == "Research Areas / Themes":
            if "research areas" in section:
                score += 0.20

            if any(
                term in text
                for term in [
                    "research areas",
                    "research themes",
                    "areas of research",
                    "research directions",
                    "area of interest",
                ]
            ):
                score += 0.15

        elif category == "Methods Discussed":
            if _is_method_evidence(item):
                score += 0.15

            if any(
                section_name in section
                for section_name in [
                    "methods",
                    "methodology",
                    "research methods",
                ]
            ):
                score += 0.15

        elif category == "Evidence / Studies Reviewed":
            if _is_study_evidence(item):
                score += 0.15

            if any(
                term in text
                for term in [
                    "empirical studies",
                    "empirical study",
                    "experimental study",
                    "studies",
                    "evidence",
                    "findings",
                ]
            ):
                score += 0.10

        elif category == "Conclusions / Research Directions":
            explicit_direction = any(
                term in text
                for term in [
                    "future research",
                    "research directions",
                    "research direction",
                    "potential areas of research",
                    "areas for future research",
                    "further research",
                    "further studies",
                    "should be investigated",
                    "should be explored",
                    "needs further research",
                    "need for further research",
                    "research agenda",
                ]
            )

            if "research directions" in section:
                score += 0.03

            elif "future research" in section:
                score += 0.025

            elif "future work" in section:
                score += 0.025

            elif (
                "conclusion" in section
                or "conclusions" in section
            ):
                score += 0.02

            if explicit_direction:
                score += 0.10

            if _is_research_direction_evidence(item):
                score += 0.02

        return score

    candidates.sort(
        key=selection_score,
        reverse=True,
    )

    return candidates[0]


def _retrieve_scope_evidence(
    retriever: EvidenceRetriever,
    chunks: list,
):
    evidence = retriever.retrieve_category(
        chunks,
        "Research Aim / Scope",
        top_k=10,
    )

    early_chunks = []

    for chunk in chunks:
        if isinstance(chunk, dict):
            page = chunk.get("page")
        else:
            page = getattr(chunk, "page", None)

        if page is not None and page <= 3:
            early_chunks.append(chunk)

    if early_chunks:
        early_evidence = retriever.retrieve_category(
            early_chunks,
            "Research Aim / Scope",
            top_k=min(
                5,
                len(early_chunks),
            ),
        )

        evidence.extend(early_evidence)

    unique = {}

    for item in evidence:
        key = (
            item.page,
            item.text.strip(),
        )

        if key not in unique:
            unique[key] = item
        elif item.score > unique[key].score:
            unique[key] = item

    evidence = list(unique.values())

    explicit = [
        item
        for item in evidence
        if _is_scope_evidence(item)
    ]

    if explicit:
        evidence = explicit

    evidence.sort(
        key=lambda item: float(item.score),
        reverse=True,
    )

    return evidence


def analyse(chunks: list) -> dict:
    retriever = EvidenceRetriever()

    profile = build_profile(chunks)

    # ---------------------------------------------------------
    # Pass 1: retrieve evidence for every category and remember the
    # best score seen anywhere, so status can be judged relative to
    # what this specific document actually offers.
    # ---------------------------------------------------------

    raw_results = []

    for category in CATEGORIES:
        if category == "Research Aim / Scope":
            evidence = _retrieve_scope_evidence(
                retriever,
                chunks,
            )

        else:
            evidence = retriever.retrieve_category(
                chunks,
                category,
                top_k=max(
                    MAX_EVIDENCE_PER_CATEGORY,
                    DEFAULT_TOP_K,
                ),
            )

        primary = _select_primary_evidence(
            category,
            evidence,
        )

        raw_results.append(
            {
                "category": category,
                "evidence": evidence,
                "primary": primary,
            }
        )

    doc_best_score = max(
        (
            float(item["primary"].score)
            for item in raw_results
            if item["primary"] is not None
        ),
        default=0.0,
    )

    # ---------------------------------------------------------
    # Pass 2: classify status relative to doc_best_score, apply
    # paper-type-aware "not applicable" gating, and build the
    # user-facing category rows.
    # ---------------------------------------------------------

    category_results = []

    for item in raw_results:
        category = item["category"]
        evidence = item["evidence"]
        primary = item["primary"]

        status = classify_status(
            primary,
            category,
            doc_best_score,
        )

        # Only ever downgrade to "not applicable": a category with
        # genuinely strong evidence is never suppressed just because
        # the document's structure looked atypical.
        if (
            status in {"missing", "partial"}
            and not structurally_applicable(category, profile)
        ):
            status = "not_applicable"

        confidence = confidence_from_evidence(
            evidence
        )

        if status == "found" and confidence == "weak":
            confidence = "moderate"

        alternatives = [
            item2
            for item2 in evidence
            if item2 is not primary
        ]

        category_results.append(
            {
                "category": category,
                "status": status,
                "confidence": (
                    confidence
                    if evidence
                    else None
                ),
                "page": (
                    primary.page
                    if primary
                    else None
                ),
                "section": (
                    primary.section
                    if primary
                    else None
                ),
                "evidence": (
                    primary.text
                    if primary
                    else None
                ),
                "score": (
                    round(
                        float(primary.score),
                        3,
                    )
                    if primary
                    else None
                ),
                "alternatives": [
                    _evidence_to_dict(item2)
                    for item2 in alternatives
                ],
            }
        )

    found = sum(
        1
        for item in category_results
        if item["status"] == "found"
    )

    partial = sum(
        1
        for item in category_results
        if item["status"] == "partial"
    )

    missing = sum(
        1
        for item in category_results
        if item["status"] == "missing"
    )

    not_applicable = sum(
        1
        for item in category_results
        if item["status"] == "not_applicable"
    )

    applicable_categories = (
        len(category_results)
        - not_applicable
    )

    if applicable_categories:
        coverage = (
            (
                found
                + (partial * 0.5)
            )
            / applicable_categories
        ) * 100
    else:
        coverage = 0

    coverage = round(
        min(
            max(coverage, 0.0),
            100.0,
        ),
        1,
    )

    evidence_gaps = [
        item["category"]
        for item in category_results
        if item["status"] == "missing"
    ]

    return {
        "paper_type": profile["paper_type"],
        "categories": category_results,
        "summary": {
            "total_categories": len(category_results),
            "applicable_categories": applicable_categories,
            "found": found,
            "partial": partial,
            "missing": missing,
            "not_applicable": not_applicable,
            "evidence_coverage": coverage,
        },
        "evidence_gaps": evidence_gaps,
        "feedback": build_feedback(
            profile,
            category_results,
            coverage,
        ),
    }