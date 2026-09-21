from dataclasses import dataclass

import numpy as np

from .chunking import ResearchChunk


@dataclass
class RankedEvidence:
    chunk: ResearchChunk
    score: float


def cosine_similarity(
    query: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    """
    Calculate cosine similarity between one query
    vector and all document vectors.
    """

    query_norm = np.linalg.norm(query)

    matrix_norms = np.linalg.norm(
        matrix,
        axis=1,
    )

    if query_norm == 0:
        return np.zeros(
            len(matrix)
        )

    denominator = (
        query_norm * matrix_norms
    )

    denominator[
        denominator == 0
    ] = 1

    return (
        matrix @ query
    ) / denominator


def rank_chunks(
    query_embedding: np.ndarray,
    chunks: list[ResearchChunk],
    embeddings: np.ndarray,
    preferred_sections: list[str],
    top_k: int = 5,
) -> list[RankedEvidence]:
    """
    Rank document chunks by semantic similarity.

    A small structural bonus is applied when a chunk
    belongs to a section that is particularly relevant
    to the research category being analysed.
    """

    scores = cosine_similarity(
        query_embedding,
        embeddings,
    )

    adjusted: list[
        tuple[float, int]
    ] = []

    for index, chunk in enumerate(
        chunks
    ):
        score = float(
            scores[index]
        )

        if (
            chunk.section
            in preferred_sections
        ):
            score += 0.035

        adjusted.append(
            (
                score,
                index,
            )
        )

    adjusted.sort(
        reverse=True
    )

    return [
        RankedEvidence(
            chunk=chunks[index],
            score=score,
        )
        for score, index in adjusted[
            :top_k
        ]
    ]