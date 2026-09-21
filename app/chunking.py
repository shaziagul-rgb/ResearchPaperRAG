from dataclasses import dataclass


@dataclass
class ResearchChunk:
    id: str
    page: int
    text: str
    start: int
    end: int
    section: str | None = None


def create_chunks(
    pages: list[dict],
    chunk_size: int = 650,
    overlap: int = 100,
) -> list[ResearchChunk]:
    """
    Split each page into smaller overlapping passages.

    Keeping the page number means retrieved evidence
    can later be shown to the user with a page reference.
    """

    chunks: list[ResearchChunk] = []

    for page in pages:
        text = page["text"].strip()

        if not text:
            continue

        section = detect_section(text)

        start = 0
        chunk_number = 0

        while start < len(text):
            end = min(
                start + chunk_size,
                len(text),
            )

            chunk_text = text[
                start:end
            ].strip()

            if chunk_text:
                chunks.append(
                    ResearchChunk(
                        id=(
                            f"page-{page['page']}"
                            f"-chunk-{chunk_number}"
                        ),
                        page=page["page"],
                        text=chunk_text,
                        start=start,
                        end=end,
                        section=section,
                    )
                )

            if end >= len(text):
                break

            start = max(
                end - overlap,
                start + 1,
            )

            chunk_number += 1

    return chunks


def detect_section(
    text: str,
) -> str | None:
    """
    Look for common research-paper section names.

    This is deliberately lightweight. It gives semantic
    retrieval a little structural information without
    requiring a complicated document parser.
    """

    lower = text.lower()

    section_terms = [
        (
            "abstract",
            ["abstract"],
        ),
        (
            "introduction",
            [
                "introduction",
                "background",
                "motivation",
            ],
        ),
        (
            "related work",
            [
                "related work",
                "literature review",
            ],
        ),
        (
            "dataset",
            [
                "dataset",
                "data collection",
                "participants",
                "corpus",
            ],
        ),
        (
            "method",
            [
                "method",
                "methodology",
                "approach",
                "architecture",
            ],
        ),
        (
            "training",
            [
                "training",
                "implementation",
                "experimental setup",
            ],
        ),
        (
            "evaluation",
            [
                "evaluation",
                "metrics",
                "benchmark",
                "experiments",
            ],
        ),
        (
            "results",
            [
                "results",
                "findings",
            ],
        ),
        (
            "discussion",
            [
                "discussion",
            ],
        ),
        (
            "limitations",
            [
                "limitations",
                "limitations and future work",
            ],
        ),
        (
            "conclusion",
            [
                "conclusion",
                "future work",
            ],
        ),
    ]

    for section, terms in section_terms:
        if any(
            term in lower
            for term in terms
        ):
            return section

    return None