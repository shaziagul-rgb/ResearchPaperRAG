from dataclasses import dataclass

from .chunking import ResearchChunk
from .retrieval import rank_chunks


@dataclass
class AnalysisDefinition:
    label: str
    queries: list[str]
    why: str
    threshold: float
    preferred_sections: list[str]


ANALYSIS_DEFINITIONS = [
    AnalysisDefinition(
        label="Research problem",
        queries=[
            "What research problem or research gap does this paper address?",
            "What question or objective motivates this study?",
            "What problem is the proposed research trying to solve?",
        ],
        why=(
            "A clear research problem defines what the study "
            "is trying to investigate or solve."
        ),
        threshold=0.31,
        preferred_sections=[
            "abstract",
            "introduction",
        ],
    ),
    AnalysisDefinition(
        label="Dataset",
        queries=[
            "What dataset or data source was used in the study?",
            "What corpus, participants, images, documents, "
            "or experimental data were used?",
            "How was the research data collected or selected?",
        ],
        why=(
            "Dataset information helps explain what data was "
            "used and whether the experiment can be reproduced."
        ),
        threshold=0.30,
        preferred_sections=[
            "dataset",
            "method",
            "training",
        ],
    ),
    AnalysisDefinition(
        label="Method / model",
        queries=[
            "What method, algorithm, model, or technical "
            "approach does the paper propose?",
            "How is the proposed system or research method constructed?",
            "What architecture or methodology is used in the experiment?",
        ],
        why=(
            "The method describes how the proposed system "
            "or experiment was constructed."
        ),
        threshold=0.32,
        preferred_sections=[
            "method",
            "introduction",
        ],
    ),
    AnalysisDefinition(
        label="Training",
        queries=[
            "How was the machine learning model trained?",
            "What training procedure and configuration were used?",
            "What optimizer, learning rate, epochs, batch size, "
            "loss function, or training settings were used?",
        ],
        why=(
            "Training details help explain how a machine-learning "
            "model was learned and configured."
        ),
        threshold=0.32,
        preferred_sections=[
            "training",
            "method",
            "evaluation",
        ],
    ),
    AnalysisDefinition(
        label="Evaluation",
        queries=[
            "How was the proposed method evaluated?",
            "What evaluation metrics or benchmarks were used?",
            "How did the researchers measure performance or error?",
        ],
        why=(
            "Evaluation criteria show how researchers measured "
            "the performance of their approach."
        ),
        threshold=0.31,
        preferred_sections=[
            "evaluation",
            "results",
            "method",
        ],
    ),
    AnalysisDefinition(
        label="Results",
        queries=[
            "What experimental results and findings are reported?",
            "What performance did the proposed method achieve?",
            "What were the main outcomes of the experiments?",
        ],
        why=(
            "Results provide the empirical evidence used "
            "to support the findings of the study."
        ),
        threshold=0.31,
        preferred_sections=[
            "results",
            "evaluation",
            "discussion",
        ],
    ),
    AnalysisDefinition(
        label="Limitations / future work",
        queries=[
            "What limitations or weaknesses does the paper identify?",
            "What unresolved problems remain after this study?",
            "What future research directions or future work are proposed?",
        ],
        why=(
            "Limitations and future work identify unresolved "
            "problems and opportunities for further research."
        ),
        threshold=0.30,
        preferred_sections=[
            "limitations",
            "discussion",
            "conclusion",
        ],
    ),
]


def analyse(
    chunks: list[ResearchChunk],
    embeddings,
    embedder,
) -> list[dict]:
    """
    Analyse the paper against several research-method categories.

    Each category uses multiple semantic questions. The strongest
    retrieved passages are combined, then weak matches are removed.
    """

    results = []

    for definition in ANALYSIS_DEFINITIONS:
        candidates = {}

        for query in definition.queries:
            query_embedding = embedder.encode(
                query,
                normalize_embeddings=True,
            )

            ranked = rank_chunks(
                query_embedding,
                chunks,
                embeddings,
                definition.preferred_sections,
                top_k=5,
            )

            for result in ranked:
                chunk = result.chunk

                evidence = {
                    "page": chunk.page,
                    "text": clean_text(chunk.text),
                    "score": float(result.score),
                    "section": chunk.section,
                }

                existing = candidates.get(
                    chunk.id
                )

                if (
                    existing is None
                    or evidence["score"]
                    > existing["score"]
                ):
                    candidates[chunk.id] = evidence

        ordered = sorted(
            candidates.values(),
            key=lambda item: item["score"],
            reverse=True,
        )

        strong_matches = [
            item
            for item in ordered
            if item["score"] >= definition.threshold
        ]

        best = (
            strong_matches[0]
            if strong_matches
            else None
        )

        if best:
            confidence = (
                "strong"
                if best["score"]
                >= definition.threshold + 0.10
                else "moderate"
            )

            status = "found"

        else:
            confidence = None
            status = "missing"

        results.append(
            {
                "label": definition.label,
                "status": status,
                "confidence": confidence,
                "evidence": best,
                "alternatives": strong_matches[1:3],
                "why": definition.why,
                "queries": definition.queries,
            }
        )

    return results


def clean_text(text: str) -> str:
    return " ".join(
        text.split()
    ).strip()