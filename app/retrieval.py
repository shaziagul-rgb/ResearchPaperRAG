from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import MIN_EVIDENCE_SCORE, MODEL_NAME


@dataclass
class RetrievedEvidence:
    page: int
    text: str
    score: float
    section: str | None
    semantic_score: float
    keyword_score: float


CATEGORY_CONFIG = {
    "Research Aim / Scope": {
        "queries": [
            "research aim purpose objective scope",
            "purpose of this chapter research focus",
            "overview of localization research",
        ],
        "preferred_sections": [
            "introduction",
            "research areas",
            "research directions",
        ],
        "excluded_sections": [
            "references",
            "bibliography",
            "figure",
            "caption",
        ],
        "keywords": [
            "aim",
            "purpose",
            "objective",
            "scope",
            "focus",
            "research",
            "chapter",
            "overview",
        ],
    },
    "Key Concepts / Definitions": {
        "queries": [
            "definition of localization",
            "what is localization",
            "localization definition concept",
            "meaning of localization",
        ],
        "preferred_sections": [
            "definitions",
            "theoretical framework",
        ],
        "excluded_sections": [
            "references",
            "bibliography",
            "figure",
            "caption",
        ],
        "keywords": [
            "definition",
            "defined",
            "refers",
            "concept",
            "localization",
            "locale",
        ],
    },
    "Theoretical Framework": {
        "queries": [
            "theoretical framework localization",
            "theories approaches localization",
            "conceptualization of localization",
            "translation theory localization",
        ],
        "preferred_sections": [
            "theoretical framework",
            "theoretical approaches",
        ],
        "excluded_sections": [
            "references",
            "bibliography",
            "figure",
            "caption",
        ],
        "keywords": [
            "theoretical",
            "theory",
            "framework",
            "approach",
            "conceptual",
            "translation",
            "modality",
        ],
    },
    "Research Areas / Themes": {
        "queries": [
            "main research areas in localization research",
            "research areas localization",
            "research themes localization studies",
            "areas of research in localization",
        ],
        "preferred_sections": [
            "research areas",
            "research directions",
        ],
        "excluded_sections": [
            "references",
            "bibliography",
            "figure",
            "caption",
        ],
        "keywords": [
            "research",
            "areas",
            "themes",
            "studies",
            "approaches",
            "empirical",
            "experimental",
            "cognitive",
            "sociological",
            "ethnographic",
        ],
    },
    "Methods Discussed": {
        "queries": [
            "research methods localization studies",
            "methods used in localization research",
            "methodology localization research",
            "empirical methods localization",
        ],
        "preferred_sections": [
            "methods",
            "methodology",
            "research areas",
            "research directions",
        ],
        "excluded_sections": [
            "references",
            "bibliography",
            "figure",
            "caption",
        ],
        "keywords": [
            "method",
            "methodology",
            "empirical",
            "experimental",
            "corpus",
            "survey",
            "study",
            "analysis",
            "data",
        ],
    },
    "Evidence / Studies Reviewed": {
        "queries": [
            "studies reviewed localization research",
            "research evidence localization",
            "empirical studies localization",
            "previous research localization",
        ],
        "preferred_sections": [
            "research areas",
            "research directions",
            "theoretical framework",
        ],
        "excluded_sections": [
            "references",
            "bibliography",
            "figure",
            "caption",
        ],
        "keywords": [
            "study",
            "studies",
            "research",
            "empirical",
            "evidence",
            "findings",
            "results",
            "literature",
        ],
    },
    "Conclusions / Research Directions": {
        "queries": [
            "conclusions localization research",
            "future research directions localization",
            "research gaps localization",
            "future of localization research",
        ],
        "preferred_sections": [
            "conclusion",
            "conclusions",
            "research directions",
        ],
        "excluded_sections": [
            "references",
            "bibliography",
            "figure",
            "caption",
        ],
        "keywords": [
            "conclusion",
            "future",
            "directions",
            "research",
            "gaps",
            "challenges",
            "opportunities",
        ],
    },
}


class EvidenceRetriever:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
    ):
        self.model = SentenceTransformer(
            model_name
        )

    def _encode(
        self,
        texts: list[str],
    ) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
        )

    @staticmethod
    def _normalise_text(
        text: str,
    ) -> str:
        return re.sub(
            r"\s+",
            " ",
            text.lower(),
        ).strip()

    @staticmethod
    def _tokenise(
        text: str,
    ) -> set[str]:
        return set(
            re.findall(
                r"\b[a-zA-Z][a-zA-Z0-9-]{2,}\b",
                text.lower(),
            )
        )

    def _keyword_score(
        self,
        text: str,
        keywords: list[str],
    ) -> float:
        if not keywords:
            return 0.0

        normalised = self._normalise_text(
            text
        )

        matches = sum(
            1
            for keyword in keywords
            if keyword.lower() in normalised
        )

        return min(
            matches / max(
                len(keywords) * 0.25,
                1,
            ),
            1.0,
        )

    def _query_overlap_score(
        self,
        query: str,
        text: str,
    ) -> float:
        query_tokens = self._tokenise(
            query
        )

        text_tokens = self._tokenise(
            text
        )

        if not query_tokens:
            return 0.0

        overlap = (
            query_tokens
            & text_tokens
        )

        return (
            len(overlap)
            / len(query_tokens)
        )

    @staticmethod
    def _section_matches(
        section: str | None,
        names: list[str],
    ) -> bool:
        if not section:
            return False

        section_lower = section.lower()

        return any(
            name.lower() in section_lower
            for name in names
        )

    @staticmethod
    def _is_excluded_section(
        section: str | None,
        excluded_sections: list[str],
    ) -> bool:
        if not section:
            return False

        section_lower = section.lower()

        return any(
            excluded.lower() in section_lower
            for excluded in excluded_sections
        )

    @staticmethod
    def _is_reference_like(
        text: str,
    ) -> bool:
        text_lower = text.lower()

        year_count = len(
            re.findall(
                r"\b(?:19|20)\d{2}\b",
                text,
            )
        )

        citation_patterns = len(
            re.findall(
                r"\b(?:doi|https?://|vol\.|pp?\.|"
                r"journal|proceedings|publisher)\b",
                text_lower,
            )
        )

        return (
            year_count >= 5
            and citation_patterns >= 1
        )

    def _question_keywords(
        self,
        question: str,
    ) -> list[str]:
        words = re.findall(
            r"\b[a-zA-Z][a-zA-Z0-9-]{2,}\b",
            question.lower(),
        )

        stopwords = {
            "what",
            "are",
            "the",
            "main",
            "how",
            "does",
            "why",
            "which",
            "where",
            "when",
            "this",
            "that",
            "from",
            "with",
            "about",
            "into",
            "used",
            "discussed",
            "identified",
            "defined",
        }

        return [
            word
            for word in words
            if word not in stopwords
        ]

    def _detect_question_intent(
        self,
        question: str,
    ) -> str:
        q = question.lower()

        if any(
            phrase in q
            for phrase in [
                "future research",
                "future directions",
                "research directions",
                "research gaps",
                "what future",
                "identified for future",
                "future of",
            ]
        ):
            return "future"

        if any(
            word in q
            for word in [
                "method",
                "methodology",
                "methods",
            ]
        ):
            return "methods"

        if any(
            phrase in q
            for phrase in [
                "theoretical approach",
                "theoretical approaches",
                "theoretical framework",
                "theory",
                "theories",
            ]
        ):
            return "theory"

        if any(
            phrase in q
            for phrase in [
                "research areas",
                "research area",
                "research themes",
                "main areas",
                "areas discussed",
            ]
        ):
            return "research_areas"

        if any(
            phrase in q
            for phrase in [
                "what is",
                "what are",
                "define",
                "definition",
                "meaning of",
            ]
        ):
            return "definition"

        if any(
            word in q
            for word in [
                "study",
                "studies",
                "evidence",
                "findings",
                "research",
            ]
        ):
            return "evidence"

        return "general"

    @staticmethod
    def _intent_multiplier(
        intent: str,
        section: str | None,
        text: str,
    ) -> float:
        section_lower = (
            section.lower()
            if section
            else ""
        )

        text_lower = text.lower()

        if intent == "definition":
            if "definition" in section_lower:
                return 1.45

            if "theoretical framework" in section_lower:
                return 1.15

            return 0.90

        if intent == "research_areas":
            if "research areas" in section_lower:
                return 1.45

            if "research directions" in section_lower:
                return 1.15

            return 0.90

        if intent == "theory":
            if "theoretical framework" in section_lower:
                return 1.45

            if "theoretical approach" in section_lower:
                return 1.35

            return 0.90

        if intent == "methods":
            if (
                "method" in section_lower
                or "methodology" in section_lower
            ):
                return 1.45

            method_terms = [
                "method",
                "methodology",
                "empirical",
                "experimental",
                "corpus",
                "survey",
                "analysis",
            ]

            if any(
                term in text_lower
                for term in method_terms
            ):
                return 1.15

            return 0.90

        if intent == "future":
            if "research directions" in section_lower:
                return 1.65

            if (
                "conclusion" in section_lower
                or "conclusions" in section_lower
            ):
                return 1.35

            if "research areas" in section_lower:
                return 0.65

            return 0.80

        if intent == "evidence":
            if (
                "research areas" in section_lower
                or "research directions" in section_lower
                or "theoretical framework" in section_lower
            ):
                return 1.15

            return 0.95

        return 1.0

    def retrieve_category(
        self,
        chunks: list[dict],
        category: str,
        top_k: int = 5,
    ) -> list[RetrievedEvidence]:

        config = CATEGORY_CONFIG[
            category
        ]

        queries = config["queries"]

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        if not texts:
            return []

        text_embeddings = self._encode(
            texts
        )

        query_embeddings = self._encode(
            queries
        )

        semantic_scores = np.max(
            query_embeddings
            @ text_embeddings.T,
            axis=0,
        )

        scored = []

        for index, chunk in enumerate(
            chunks
        ):
            text = chunk["text"]
            section = chunk.get(
                "section"
            )

            if self._is_excluded_section(
                section,
                config["excluded_sections"],
            ):
                continue

            if self._is_reference_like(
                text
            ):
                continue

            keyword_score = (
                self._keyword_score(
                    text,
                    config["keywords"],
                )
            )

            score = (
                0.75
                * float(
                    semantic_scores[index]
                )
                + 0.25
                * keyword_score
            )

            if self._section_matches(
                section,
                config["preferred_sections"],
            ):
                score *= 1.15

            scored.append(
                RetrievedEvidence(
                    page=int(
                        chunk["page"]
                    ),
                    text=text,
                    score=score,
                    section=section,
                    semantic_score=float(
                        semantic_scores[index]
                    ),
                    keyword_score=keyword_score,
                )
            )

        scored.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        results = []
        seen = set()

        for item in scored:
            key = (
                item.page,
                self._normalise_text(
                    item.text
                )[:250],
            )

            if key in seen:
                continue

            seen.add(key)

            if (
                item.score
                >= MIN_EVIDENCE_SCORE
            ):
                results.append(item)

            if len(results) >= top_k:
                break

        return results

    def retrieve_question(
        self,
        chunks: list[dict],
        question: str,
        top_k: int = 5,
    ) -> list[RetrievedEvidence]:

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        if not texts:
            return []

        text_embeddings = self._encode(
            texts
        )

        question_embedding = self._encode(
            [question]
        )[0]

        semantic_scores = (
            text_embeddings
            @ question_embedding
        )

        question_keywords = (
            self._question_keywords(
                question
            )
        )

        intent = (
            self._detect_question_intent(
                question
            )
        )

        scored = []

        for index, chunk in enumerate(
            chunks
        ):
            text = chunk["text"]
            section = chunk.get(
                "section"
            )

            excluded_sections = {
                "references",
                "bibliography",
                "figure",
                "caption",
            }

            section_lower = (
                section.lower()
                if section
                else ""
            )

            if any(
                excluded in section_lower
                for excluded in excluded_sections
            ):
                continue

            if self._is_reference_like(
                text
            ):
                continue

            keyword_score = (
                self._keyword_score(
                    text,
                    question_keywords,
                )
            )

            query_overlap = (
                self._query_overlap_score(
                    question,
                    text,
                )
            )

            base_score = (
                0.70
                * float(
                    semantic_scores[index]
                )
                + 0.20
                * keyword_score
                + 0.10
                * query_overlap
            )

            multiplier = (
                self._intent_multiplier(
                    intent,
                    section,
                    text,
                )
            )

            score = (
                base_score
                * multiplier
            )

            scored.append(
                RetrievedEvidence(
                    page=int(
                        chunk["page"]
                    ),
                    text=text,
                    score=score,
                    section=section,
                    semantic_score=float(
                        semantic_scores[index]
                    ),
                    keyword_score=keyword_score,
                )
            )

        scored.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        results = []
        seen = set()

        for item in scored:
            key = (
                item.page,
                self._normalise_text(
                    item.text
                )[:300],
            )

            if key in seen:
                continue

            seen.add(key)

            if (
                item.score
                >= MIN_EVIDENCE_SCORE
            ):
                results.append(item)

            if len(results) >= top_k:
                break

        return results