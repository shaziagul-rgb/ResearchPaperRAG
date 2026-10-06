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
            "aim and scope of this chapter",
            "purpose and scope of the review",
            "focus of this research",
            "what this chapter examines",
            "research objectives",
        ],
        "preferred_sections": [
            "abstract",
            "introduction",
            "research aim",
            "scope",
        ],
        "excluded_sections": [
            "references",
            "bibliography",
            "figure",
            "figure_caption",
            "caption",
        ],
        "keywords": [
            "aim",
            "purpose",
            "objective",
            "scope",
            "focus",
            "this chapter",
            "this study",
            "this research",
            "this review",
            "we review",
            "we examine",
            "we analyse",
            "we analyze",
            "focuses on",
            "research on",
        ],
    },

    "Key Concepts / Definitions": {
        "queries": [
            "definition of key concepts",
            "concepts and definitions",
            "meaning and definition of important terms",
            "term is defined as",
            "conceptual definition",
            "key terminology used in this study",
        ],
        "preferred_sections": [
            "definitions",
            "background",
            "theoretical framework",
            "introduction",
        ],
        "excluded_sections": [
            "references",
            "bibliography",
            "figure",
            "figure_caption",
            "caption",
        ],
        "keywords": [
            "defined",
            "definition",
            "refers to",
            "means",
            "concept",
            "term",
            "definition of",
            "is defined",
        ],
    },

    "Theoretical Framework": {
        "queries": [
            "theoretical framework theory theoretical perspective",
            "theoretical foundations",
            "conceptual framework",
            "theoretical model",
            "theoretical approach",
            "theoretical paradigm",
            "epistemological framework",
        ],
        "preferred_sections": [
            "theoretical framework",
            "theory",
            "theoretical",
            "background",
            "introduction",
        ],
        "excluded_sections": [
            "references",
            "bibliography",
            "figure",
            "figure_caption",
            "caption",
        ],
        "keywords": [
            "theoretical",
            "theory",
            "framework",
            "conceptual",
            "paradigm",
            "model",
            "perspective",
            "epistemological",
            "theoretical framework",
            "theoretical approach",
        ],
    },

    "Research Areas / Themes": {
        "queries": [
            "research areas themes fields topics",
            "areas of research",
            "research themes",
            "main research topics",
            "subareas of research",
            "areas of study discussed",
        ],
        "preferred_sections": [
            "research areas",
            "related work",
            "literature review",
            "introduction",
        ],
        "excluded_sections": [
            "references",
            "bibliography",
            "figure",
            "figure_caption",
            "caption",
        ],
        "keywords": [
            "research area",
            "research areas",
            "research theme",
            "research themes",
            "field",
            "fields",
            "topic",
            "topics",
            "subarea",
            "areas of research",
        ],
    },

    "Methods Discussed": {
        "queries": [
            "research methods methodology approaches",
            "methods used in research",
            "methodological approaches",
            "research design procedures",
            "experimental methods",
            "methods and approaches discussed in previous studies",
            "research methodologies used in previous studies",
        ],
        "preferred_sections": [
            "methods",
            "methodology",
            "research methods",
            "research design",
            "experimental",
            "approaches",
            "research areas",
        ],
        "excluded_sections": [
            "references",
            "bibliography",
            "figure",
            "figure_caption",
            "caption",
        ],
        "keywords": [
            "method",
            "methods",
            "methodology",
            "methodological",
            "approach",
            "approaches",
            "procedure",
            "procedures",
            "experiment",
            "experimental",
            "corpus",
            "corpus studies",
            "dataset",
            "data collection",
            "analysis",
            "sample",
            "participants",
            "survey",
            "interview",
            "case study",
            "empirical",
            "comparative",
            "contrastive",
        ],
    },

    "Evidence / Studies Reviewed": {
        "queries": [
            "evidence studies reviewed previous research",
            "empirical evidence studies",
            "previous studies and findings",
            "research evidence literature",
            "studies reviewed in the literature",
            "findings from previous studies",
            "evidence from previous research",
        ],
        "preferred_sections": [
            "literature review",
            "related work",
            "evidence",
            "studies",
            "results",
            "research areas",
        ],
        "excluded_sections": [
            "references",
            "bibliography",
            "figure",
            "figure_caption",
            "caption",
        ],
        "keywords": [
            "study",
            "studies",
            "evidence",
            "findings",
            "results",
            "research",
            "investigation",
            "empirical",
            "literature",
            "previous studies",
            "researchers found",
            "found that",
            "reported",
            "demonstrated",
            "observed",
        ],
    },

    "Conclusions / Research Directions": {
        "queries": [
            "conclusions research directions future work",
            "future research directions",
            "conclusions implications",
            "areas for future research",
            "developing research approaches",
            "future areas of research",
            "research gaps and future directions",
            "recommendations for further research",
        ],
        "preferred_sections": [
            "conclusion",
            "conclusions",
            "future work",
            "research directions",
            "discussion",
            "developing approaches",
        ],
        "excluded_sections": [
            "references",
            "bibliography",
            "figure",
            "figure_caption",
            "caption",
        ],
        "keywords": [
            "conclusion",
            "conclusions",
            "future",
            "future research",
            "research directions",
            "implications",
            "further research",
            "future work",
            "developing",
            "emerging",
            "areas for future",
            "research gaps",
            "gap",
            "gaps",
        ],
    },
}


class EvidenceRetriever:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    # ---------------------------------------------------------
    # Chunk helpers
    # ---------------------------------------------------------

    @staticmethod
    def _get_chunk_value(chunk, key, default=None):
        """
        Support both ResearchChunk objects and dictionaries.

        main.py currently passes dictionaries, while some tests may
        pass ResearchChunk instances.
        """
        if isinstance(chunk, dict):
            return chunk.get(key, default)

        return getattr(chunk, key, default)

    # ---------------------------------------------------------
    # Embeddings
    # ---------------------------------------------------------

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings

    # ---------------------------------------------------------
    # Keyword scoring
    # ---------------------------------------------------------

    @staticmethod
    def _keyword_score(text: str, keywords: list[str]) -> float:
        """
        Estimate lexical relevance.

        We deliberately do not require a large percentage of all
        category keywords. A passage can be highly relevant while
        naturally containing only a few category-specific terms.
        """
        text_lower = text.lower()

        matches = 0

        for keyword in keywords:
            keyword_lower = keyword.lower().strip()

            if not keyword_lower:
                continue

            if keyword_lower in text_lower:
                matches += 1

        # Four meaningful matches = full lexical score.
        return min(matches / 4.0, 1.0)

    # ---------------------------------------------------------
    # Query overlap
    # ---------------------------------------------------------

    @staticmethod
    def _query_overlap_score(text: str, query: str) -> float:
        text_tokens = set(
            re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        )

        query_tokens = set(
            re.findall(r"\b[a-zA-Z]{3,}\b", query.lower())
        )

        if not query_tokens:
            return 0.0

        overlap = len(text_tokens & query_tokens)

        return min(
            overlap / max(len(query_tokens), 1),
            1.0,
        )

    # ---------------------------------------------------------
    # Section matching
    # ---------------------------------------------------------

    @staticmethod
    def _section_matches(
        section: str | None,
        preferred_sections: list[str],
    ) -> bool:
        if not section:
            return False

        section_lower = section.lower().strip()

        for preferred in preferred_sections:
            if preferred.lower() in section_lower:
                return True

        return False

    # ---------------------------------------------------------
    # Excluded sections
    # ---------------------------------------------------------

    @staticmethod
    def _is_excluded_section(
        section: str | None,
        excluded_sections: list[str],
    ) -> bool:
        if not section:
            return False

        section_lower = section.lower().strip()

        for excluded in excluded_sections:
            if excluded.lower() in section_lower:
                return True

        return False

    # ---------------------------------------------------------
    # Reference-like text
    # ---------------------------------------------------------

    @staticmethod
    def _is_reference_like(text: str) -> bool:
        """
        Detect bibliography/reference-like text even if the section
        detector failed to label it as references.
        """
        years = re.findall(
            r"\b(?:19|20)\d{2}\b",
            text,
        )

        citation_patterns = [
            r"\bdoi\b",
            r"https?://",
            r"\bvol\.\b",
            r"\bpp\.\b",
            r"\bjournal\b",
            r"\bproceedings\b",
            r"\bpublisher\b",
            r"\bpress\b",
        ]

        citation_matches = sum(
            bool(re.search(pattern, text, re.I))
            for pattern in citation_patterns
        )

        return len(years) >= 5 and citation_matches >= 1

    # ---------------------------------------------------------
    # Question processing
    # ---------------------------------------------------------

    @staticmethod
    def _question_keywords(question: str) -> list[str]:
        stopwords = {
            "what",
            "which",
            "where",
            "when",
            "why",
            "how",
            "does",
            "did",
            "the",
            "this",
            "that",
            "these",
            "those",
            "are",
            "is",
            "was",
            "were",
            "and",
            "or",
            "for",
            "with",
            "from",
            "into",
            "about",
            "can",
            "could",
            "would",
            "should",
            "paper",
            "chapter",
        }

        words = re.findall(
            r"\b[a-zA-Z]{3,}\b",
            question.lower(),
        )

        return [
            word
            for word in words
            if word not in stopwords
        ]

    @staticmethod
    def _detect_question_intent(question: str) -> str:
        question_lower = question.lower()

        if any(
            phrase in question_lower
            for phrase in [
                "future research",
                "future work",
                "research direction",
                "further research",
                "what should be studied",
                "what remains",
            ]
        ):
            return "future"

        if any(
            phrase in question_lower
            for phrase in [
                "method",
                "methodology",
                "approach",
                "how was",
                "how were",
                "research design",
                "experiment",
            ]
        ):
            return "methods"

        if any(
            phrase in question_lower
            for phrase in [
                "theory",
                "theoretical",
                "framework",
                "paradigm",
                "conceptual",
            ]
        ):
            return "theory"

        if any(
            phrase in question_lower
            for phrase in [
                "research area",
                "research areas",
                "themes",
                "topics",
                "fields",
            ]
        ):
            return "research_areas"

        if any(
            phrase in question_lower
            for phrase in [
                "define",
                "definition",
                "what is",
                "what does",
                "meaning",
            ]
        ):
            return "definition"

        if any(
            phrase in question_lower
            for phrase in [
                "evidence",
                "studies",
                "findings",
                "results",
                "researchers found",
            ]
        ):
            return "evidence"

        return "general"

    @staticmethod
    def _intent_multiplier(
        intent: str,
        section: str | None,
    ) -> float:
        if not section:
            return 1.0

        section_lower = section.lower()

        if intent == "future":
            if any(
                term in section_lower
                for term in [
                    "future",
                    "research directions",
                    "conclusion",
                    "discussion",
                ]
            ):
                return 1.10

        if intent == "methods":
            if any(
                term in section_lower
                for term in [
                    "method",
                    "methodology",
                    "research design",
                    "experimental",
                    "research areas",
                ]
            ):
                return 1.10

        if intent == "theory":
            if any(
                term in section_lower
                for term in [
                    "theoretical",
                    "theory",
                    "framework",
                    "background",
                ]
            ):
                return 1.10

        if intent == "research_areas":
            if "research areas" in section_lower:
                return 1.10

        if intent == "definition":
            if any(
                term in section_lower
                for term in [
                    "definition",
                    "definitions",
                    "background",
                ]
            ):
                return 1.10

        if intent == "evidence":
            if any(
                term in section_lower
                for term in [
                    "evidence",
                    "studies",
                    "results",
                    "literature review",
                    "research areas",
                ]
            ):
                return 1.10

        return 1.0

    # ---------------------------------------------------------
    # Category retrieval
    # ---------------------------------------------------------

    def retrieve_category(
        self,
        chunks: list,
        category: str,
        top_k: int = 3,
    ) -> list[RetrievedEvidence]:

        config = CATEGORY_CONFIG[category]

        valid_chunks = []

        for chunk in chunks:
            text = self._get_chunk_value(
                chunk,
                "text",
                "",
            )

            if not text or not text.strip():
                continue

            section = self._get_chunk_value(
                chunk,
                "section",
                None,
            )

            if self._is_excluded_section(
                section,
                config["excluded_sections"],
            ):
                continue

            if self._is_reference_like(text):
                continue

            valid_chunks.append(chunk)

        if not valid_chunks:
            return []

        queries = config["queries"]

        query_embeddings = self._encode(queries)

        chunk_texts = [
            self._get_chunk_value(
                chunk,
                "text",
                "",
            )
            for chunk in valid_chunks
        ]

        chunk_embeddings = self._encode(
            chunk_texts
        )

        candidates = []

        for index, chunk in enumerate(valid_chunks):

            text = self._get_chunk_value(
                chunk,
                "text",
                "",
            )

            section = self._get_chunk_value(
                chunk,
                "section",
                None,
            )

            page = self._get_chunk_value(
                chunk,
                "page",
                None,
            )

            similarities = np.dot(
                query_embeddings,
                chunk_embeddings[index],
            )

            semantic_score = float(
                np.max(similarities)
            )

            semantic_score = max(
                0.0,
                min(semantic_score, 1.0),
            )

            keyword_score = self._keyword_score(
                text,
                config["keywords"],
            )

            score = (
                0.75 * semantic_score
                + 0.25 * keyword_score
            )

            # Small additive section bonus.
            if self._section_matches(
                section,
                config["preferred_sections"],
            ):
                score += 0.05

            score = max(
                0.0,
                min(score, 1.0),
            )

            candidates.append(
                RetrievedEvidence(
                    page=(
                        int(page)
                        if page is not None
                        else 0
                    ),
                    text=text,
                    score=round(score, 3),
                    section=section,
                    semantic_score=round(
                        semantic_score,
                        3,
                    ),
                    keyword_score=round(
                        keyword_score,
                        3,
                    ),
                )
            )

        # -----------------------------------------------------
        # Do NOT filter candidates using MIN_EVIDENCE_SCORE here.
        # We want analysis.py to see the strongest evidence even
        # when its score is below the global threshold.
        # -----------------------------------------------------

        candidates.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        # -----------------------------------------------------
        # Deduplicate highly similar text chunks.
        # -----------------------------------------------------

        selected = []

        for candidate in candidates:

            candidate_words = set(
                candidate.text.lower().split()
            )

            duplicate = False

            for existing in selected:

                existing_words = set(
                    existing.text.lower().split()
                )

                if not candidate_words or not existing_words:
                    continue

                overlap = (
                    len(
                        candidate_words
                        & existing_words
                    )
                    / len(
                        candidate_words
                        | existing_words
                    )
                )

                if overlap > 0.70:
                    duplicate = True
                    break

            if duplicate:
                continue

            selected.append(candidate)

            if len(selected) >= top_k:
                break

        return selected

    # ---------------------------------------------------------
    # Question retrieval
    # ---------------------------------------------------------

    def retrieve_question(
        self,
        chunks: list,
        question: str,
        top_k: int = 5,
    ) -> list[RetrievedEvidence]:

        if not question.strip():
            return []

        valid_chunks = []

        for chunk in chunks:

            text = self._get_chunk_value(
                chunk,
                "text",
                "",
            )

            if not text or not text.strip():
                continue

            section = self._get_chunk_value(
                chunk,
                "section",
                None,
            )

            if self._is_excluded_section(
                section,
                [
                    "references",
                    "bibliography",
                    "figure",
                    "figure_caption",
                    "caption",
                ],
            ):
                continue

            if self._is_reference_like(text):
                continue

            valid_chunks.append(chunk)

        if not valid_chunks:
            return []

        question_embedding = self._encode(
            [question]
        )[0]

        chunk_texts = [
            self._get_chunk_value(
                chunk,
                "text",
                "",
            )
            for chunk in valid_chunks
        ]

        chunk_embeddings = self._encode(
            chunk_texts
        )

        question_keywords = self._question_keywords(
            question
        )

        intent = self._detect_question_intent(
            question
        )

        candidates = []

        for index, chunk in enumerate(valid_chunks):

            text = self._get_chunk_value(
                chunk,
                "text",
                "",
            )

            section = self._get_chunk_value(
                chunk,
                "section",
                None,
            )

            page = self._get_chunk_value(
                chunk,
                "page",
                None,
            )

            semantic_score = float(
                np.dot(
                    question_embedding,
                    chunk_embeddings[index],
                )
            )

            semantic_score = max(
                0.0,
                min(semantic_score, 1.0),
            )

            keyword_score = 0.0

            if question_keywords:

                text_lower = text.lower()

                matches = sum(
                    1
                    for keyword in question_keywords
                    if keyword in text_lower
                )

                keyword_score = min(
                    matches / 4.0,
                    1.0,
                )

            overlap_score = self._query_overlap_score(
                text,
                question,
            )

            score = (
                0.70 * semantic_score
                + 0.20 * keyword_score
                + 0.10 * overlap_score
            )

            score *= self._intent_multiplier(
                intent,
                section,
            )

            score = max(
                0.0,
                min(score, 1.0),
            )

            candidates.append(
                RetrievedEvidence(
                    page=(
                        int(page)
                        if page is not None
                        else 0
                    ),
                    text=text,
                    score=round(score, 3),
                    section=section,
                    semantic_score=round(
                        semantic_score,
                        3,
                    ),
                    keyword_score=round(
                        keyword_score,
                        3,
                    ),
                )
            )

        candidates.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        # -----------------------------------------------------
        # Apply the minimum evidence score threshold here.
        #
        # Unlike retrieve_category (used by /api/analyze, which
        # wants to see the best available evidence even when it is
        # weak, so it can report partial/missing coverage),
        # retrieve_question feeds evidence straight to the LLM for
        # /api/ask. If nothing clears the bar, the caller should
        # get an empty list back and return an "insufficient
        # evidence" response instead of asking the model to answer
        # from irrelevant passages.
        # -----------------------------------------------------

        candidates = [
            candidate
            for candidate in candidates
            if candidate.score >= MIN_EVIDENCE_SCORE
        ]

        selected = []

        for candidate in candidates:

            candidate_words = set(
                candidate.text.lower().split()
            )

            duplicate = False

            for existing in selected:

                existing_words = set(
                    existing.text.lower().split()
                )

                if not candidate_words or not existing_words:
                    continue

                overlap = (
                    len(
                        candidate_words
                        & existing_words
                    )
                    / len(
                        candidate_words
                        | existing_words
                    )
                )

                if overlap > 0.70:
                    duplicate = True
                    break

            if duplicate:
                continue

            selected.append(candidate)

            if len(selected) >= top_k:
                break

        return selected