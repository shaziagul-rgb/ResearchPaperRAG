from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ResearchChunk:
    id: str
    page: int
    text: str
    start: int
    end: int
    section: str | None = None


SECTION_PATTERNS = [
    ("abstract", re.compile(r"^(?:\d+[\.\)]\s*)?abstract$", re.I)),
    ("introduction", re.compile(r"^(?:\d+[\.\)]\s*)?introduction$", re.I)),
    ("background", re.compile(r"^(?:\d+[\.\)]\s*)?background$", re.I)),
    (
        "literature review",
        re.compile(r"^(?:\d+[\.\)]\s*)?literature\s+review$",
        re.I),
    ),
    (
        "related work",
        re.compile(r"^(?:\d+[\.\)]\s*)?related\s+work$",
        re.I),
    ),
    (
        "methods",
        re.compile(
            r"^(?:\d+[\.\)]\s*)?(?:research\s+)?methods?$",
            re.I,
        ),
    ),
    (
        "methodology",
        re.compile(
            r"^(?:\d+[\.\)]\s*)?methodology$",
            re.I,
        ),
    ),
    (
        "results",
        re.compile(r"^(?:\d+[\.\)]\s*)?results?$", re.I),
    ),
    (
        "discussion",
        re.compile(r"^(?:\d+[\.\)]\s*)?discussion$", re.I),
    ),
    (
        "conclusion",
        re.compile(r"^(?:\d+[\.\)]\s*)?conclusions?$", re.I),
    ),
    (
        "future work",
        re.compile(
            r"^(?:\d+[\.\)]\s*)?future\s+work$",
            re.I,
        ),
    ),
    (
        "references",
        re.compile(
            r"^(?:\d+[\.\)]\s*)?references$",
            re.I,
        ),
    ),
]


SPECIAL_HEADINGS = {
    "defining localization": "definitions",
    "the localization paradigm and its use in other areas of ts":
        "theoretical framework",
    "the localization paradigm and its use in other areas of translation studies":
        "theoretical framework",
    "overview of localization research": "research areas",
    "potential areas of research": "research directions",
}


# Actual heading variants that may appear after PDF extraction.
INLINE_SPECIAL_HEADINGS = [
    (
        re.compile(
            r"the\s*[\"“”‘’']?\s*localization\s+paradigm"
            r"\s+and\s+its\s+use\s+in\s+other\s+areas\s+of\s+"
            r"(?:ts|translation\s+studies)",
            re.I,
        ),
        "theoretical framework",
    ),
    (
        re.compile(
            r"defining\s+localization",
            re.I,
        ),
        "definitions",
    ),
    (
        re.compile(
            r"overview\s+of\s+localization\s+research",
            re.I,
        ),
        "research areas",
    ),
    (
        re.compile(
            r"potential\s+areas\s+of\s+research",
            re.I,
        ),
        "research directions",
    ),
]


FIGURE_CAPTION_PATTERNS = [
    re.compile(
        r"^(?:figure|fig\.?)\s*\d+",
        re.I,
    ),
    re.compile(
        r"^different\s+areas\s+of\s+research\s+within\s+localization",
        re.I,
    ),
    re.compile(
        r"^areas\s+of\s+research\s+within\s+localization",
        re.I,
    ),
    re.compile(
        r"^interdisciplinarity\s+in\s+the\s+study\s+of\s+localization",
        re.I,
    ),
    re.compile(
        r"^adapted\s+from\b",
        re.I,
    ),
]


def _clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalise_heading_key(text: str) -> str:
    text = _clean_text(text)

    text = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )

    text = re.sub(
        r'["\']',
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text.lower(),
    )

    return text.strip(" .:;-")


def normalise_heading(
    value: str,
) -> str | None:

    value = _clean_text(value)

    if not value:
        return None

    key = _normalise_heading_key(
        value
    )

    if key in SPECIAL_HEADINGS:
        return SPECIAL_HEADINGS[key]

    for section, pattern in SECTION_PATTERNS:

        if hasattr(
            pattern,
            "fullmatch",
        ):
            matched = pattern.fullmatch(
                value
            )
        else:
            matched = re.fullmatch(
                pattern,
                value,
                flags=re.I,
            )

        if not matched:
            continue

        if section in {
            "methods",
            "methodology",
        }:
            return "methods"

        if section in {
            "conclusion",
            "future work",
        }:
            return "research directions"

        return section

    return None


def is_figure_caption(
    text: str,
) -> bool:

    value = _clean_text(text)

    if not value:
        return False

    return any(
        pattern.search(value)
        for pattern in FIGURE_CAPTION_PATTERNS
    )


def _find_inline_special_heading(
    text: str,
) -> tuple[int, int, str] | None:

    for pattern, section in INLINE_SPECIAL_HEADINGS:

        match = pattern.search(text)

        if match is None:
            continue

        position = match.start()
        end = match.end()

        # Don't interpret a heading at character zero
        # as an inline heading. It will be handled normally.
        if position == 0:
            return (
                position,
                end,
                section,
            )

        # The character before the heading should indicate
        # a natural boundary.
        before = text[position - 1]

        if before not in " \n\t.:;-–—":
            continue

        return (
            position,
            end,
            section,
        )

    return None


def _find_heading_boundary(
    text: str,
) -> tuple[int, int, str] | None:

    result = _find_inline_special_heading(
        text
    )

    if result is not None:
        return result

    return None


def _looks_like_heading(
    text: str,
) -> bool:

    value = _clean_text(text)

    if not value:
        return False

    if is_figure_caption(value):
        return False

    if normalise_heading(value) is not None:
        return True

    words = value.split()

    if len(words) > 12:
        return False

    if len(value) > 100:
        return False

    if value.endswith(
        (".", ",", ";", ":")
    ):
        return False

    if re.match(
        r"^\d+(?:\.\d+)*[\.\)]?\s+\S+",
        value,
    ):
        return True

    if len(words) <= 6:
        return True

    title_case_words = sum(
        1
        for word in words
        if word[:1].isupper()
    )

    if title_case_words >= max(
        2,
        int(len(words) * 0.6),
    ):
        return True

    return False


def _split_text_at_heading(
    text: str,
    current_section: str | None,
) -> list[tuple[str, str | None]]:

    text = _clean_text(text)

    if not text:
        return []

    results = []
    remaining = text
    section = current_section

    while remaining:

        match = _find_heading_boundary(
            remaining
        )

        if match is None:
            results.append(
                (
                    remaining,
                    section,
                )
            )
            break

        position, end, new_section = match

        before = _clean_text(
            remaining[:position]
        )

        after = _clean_text(
            remaining[end:]
        )

        if before:
            results.append(
                (
                    before,
                    section,
                )
            )

        section = new_section

        remaining = after

    return results


def split_into_sections(
    pages: list[dict],
) -> list[dict]:

    sections = []

    current_section: str | None = None

    for page_data in pages:

        page_number = int(
            page_data.get(
                "page",
                1,
            )
        )

        raw_text = page_data.get(
            "text",
            "",
        )

        raw_text = (
            raw_text
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )

        lines = [
            _clean_text(line)
            for line in raw_text.split("\n")
        ]

        lines = [
            line
            for line in lines
            if line
        ]

        current_block = []

        def flush_block():

            nonlocal current_block

            if not current_block:
                return

            text = _clean_text(
                " ".join(
                    current_block
                )
            )

            current_block = []

            if not text:
                return

            split_blocks = _split_text_at_heading(
                text,
                current_section,
            )

            for block_text, block_section in split_blocks:

                if not block_text:
                    continue

                sections.append(
                    {
                        "page": page_number,
                        "text": block_text,
                        "section": block_section,
                    }
                )

        for line in lines:

            if is_figure_caption(line):

                flush_block()

                sections.append(
                    {
                        "page": page_number,
                        "text": line,
                        "section": "figure_caption",
                    }
                )

                continue

            heading_section = normalise_heading(
                line
            )

            if heading_section is not None:

                flush_block()

                current_section = heading_section

                continue

            inline_heading = (
                _find_inline_special_heading(
                    line
                )
            )

            if inline_heading is not None:

                position, end, new_section = (
                    inline_heading
                )

                before = _clean_text(
                    line[:position]
                )

                after = _clean_text(
                    line[end:]
                )

                flush_block()

                if before:
                    sections.append(
                        {
                            "page": page_number,
                            "text": before,
                            "section": current_section,
                        }
                    )

                current_section = new_section

                if after:
                    current_block.append(
                        after
                    )

                continue

            if _looks_like_heading(line):

                flush_block()

                detected = normalise_heading(
                    line
                )

                if detected is not None:
                    current_section = detected

                continue

            current_block.append(line)

        flush_block()

    return sections


def _merge_short_heading_blocks(
    sections: list[dict],
) -> list[dict]:

    if not sections:
        return []

    merged = []

    for item in sections:

        text = item["text"].strip()

        if (
            merged
            and len(text.split()) <= 3
            and item["section"]
            == merged[-1]["section"]
        ):
            merged[-1]["text"] = (
                merged[-1]["text"]
                + " "
                + text
            ).strip()

        else:
            merged.append(
                {
                    "page": item["page"],
                    "text": item["text"],
                    "section": item["section"],
                }
            )

    return merged


def create_chunks(
    pages: list[dict],
    chunk_size: int = 180,
    overlap: int = 40,
) -> list[ResearchChunk]:

    sections = split_into_sections(
        pages
    )

    sections = _merge_short_heading_blocks(
        sections
    )

    chunks = []

    chunk_counter = 0

    for section in sections:

        page = int(
            section["page"]
        )

        text = _clean_text(
            section["text"]
        )

        section_name = section.get(
            "section"
        )

        if not text:
            continue

        if section_name == "figure_caption":
            continue

        words = text.split()

        if not words:
            continue

        start = 0

        while start < len(words):

            end = min(
                start + chunk_size,
                len(words),
            )

            chunk_text = " ".join(
                words[start:end]
            ).strip()

            if chunk_text:

                chunks.append(
                    ResearchChunk(
                        id=f"chunk-{chunk_counter}",
                        page=page,
                        text=chunk_text,
                        start=start,
                        end=end,
                        section=section_name,
                    )
                )

                chunk_counter += 1

            if end >= len(words):
                break

            next_start = end - overlap

            if next_start <= start:
                next_start = end

            start = next_start

    return chunks