from __future__ import annotations

import re
from dataclasses import dataclass


"""Section-aware chunking utilities for extracted PDF text."""


@dataclass
class ResearchChunk:
    """A chunk of document text with its source page and section."""

    id: str
    page: int
    text: str
    start: int
    end: int
    section: str | None = None


SECTION_PATTERNS = [
    (
        "abstract",
        re.compile(
            r"^(?:(?:\d+|[IVXLC]+|[A-Z])[\.\)]\s*)?abstract$",
            re.I,
        ),
    ),
    (
        "introduction",
        re.compile(
            r"^(?:(?:\d+|[IVXLC]+|[A-Z])[\.\)]\s*)?introduction$",
            re.I,
        ),
    ),
    (
        "background",
        re.compile(
            r"^(?:(?:\d+|[IVXLC]+|[A-Z])[\.\)]\s*)?background$",
            re.I,
        ),
    ),
    (
        "literature review",
        re.compile(
            r"^(?:(?:\d+|[IVXLC]+|[A-Z])[\.\)]\s*)?literature\s+review$",
            re.I,
        ),
    ),
    (
        "related work",
        re.compile(
            r"^(?:(?:\d+|[IVXLC]+|[A-Z])[\.\)]\s*)?related\s+work$",
            re.I,
        ),
    ),
    (
        "methods",
        re.compile(
            r"^(?:(?:\d+|[IVXLC]+|[A-Z])[\.\)]\s*)?(?:research\s+)?methods?$",
            re.I,
        ),
    ),
    (
        "methods",
        re.compile(
            r"^(?:(?:\d+|[IVXLC]+|[A-Z])[\.\)]\s*)?(?:materials?\s+and\s+methods?"
            r"|data\s+and\s+methods?"
            r"|research\s+design\s+and\s+methods?)$",
            re.I,
        ),
    ),
    (
        "methodology",
        re.compile(
            r"^(?:(?:\d+|[IVXLC]+|[A-Z])[\.\)]\s*)?methodology$",
            re.I,
        ),
    ),
    (
        "results",
        re.compile(
            r"^(?:(?:\d+|[IVXLC]+|[A-Z])[\.\)]\s*)?(?:results?|findings?)$",
            re.I,
        ),
    ),
    (
        "results",
        re.compile(
            r"^(?:(?:\d+|[IVXLC]+|[A-Z])[\.\)]\s*)?results?\s+and\s+discussion$",
            re.I,
        ),
    ),
    (
        "discussion",
        re.compile(
            r"^(?:(?:\d+|[IVXLC]+|[A-Z])[\.\)]\s*)?discussion$",
            re.I,
        ),
    ),
    (
        "discussion",
        re.compile(
            r"^(?:(?:\d+|[IVXLC]+|[A-Z])[\.\)]\s*)?limitations?"
            r"(?:\s+and\s+future\s+work)?$",
            re.I,
        ),
    ),
    (
        "conclusion",
        re.compile(
            r"^(?:(?:\d+|[IVXLC]+|[A-Z])[\.\)]\s*)?conclusions?$",
            re.I,
        ),
    ),
    (
        "conclusion",
        re.compile(
            r"^(?:(?:\d+|[IVXLC]+|[A-Z])[\.\)]\s*)?(?:concluding\s+remarks"
            r"|discussion\s+and\s+conclusions?"
            r"|summary\s+and\s+conclusions?)$",
            re.I,
        ),
    ),
    (
        "future work",
        re.compile(
            r"^(?:(?:\d+|[IVXLC]+|[A-Z])[\.\)]\s*)?future\s+(?:work|research|directions)$",
            re.I,
        ),
    ),
    (
        "references",
        re.compile(
            r"^(?:(?:\d+|[IVXLC]+|[A-Z])[\.\)]\s*)?(?:references|bibliography"
            r"|acknowledge?ments?|appendix(?:\s+[a-z0-9]+)?)$",
            re.I,
        ),
    ),
]


# Headings that are specific to a particular document can be added here
# by a caller if needed, but the pipeline does not depend on any by
# default so that it generalises to arbitrary papers.
SPECIAL_HEADINGS: dict[str, str] = {}


# Some PDF extractors place a heading in the middle of a text line
# instead of on its own line. No document-specific patterns are
# registered by default; this list exists as an extension point.
INLINE_SPECIAL_HEADINGS: list[tuple[re.Pattern, str]] = []


MIN_CHUNK_WORDS = 20


FIGURE_CAPTION_PATTERNS = [
    re.compile(
        r"^(?:figure|fig\.?)\s*(?:\d+|[ivxlc]+)\s*(?:$|[:.\-–—|])",
        re.I,
    ),
    re.compile(
        r"^table\s*(?:\d+|[ivxlc]+)\s*(?:$|[:.\-–—|])",
        re.I,
    ),
    re.compile(
        r"^adapted\s+from\b",
        re.I,
    ),
    re.compile(
        r"^source\s*:",
        re.I,
    ),
    # A full all-caps line (a table/figure caption's own sentence(s),
    # e.g. "RECALL@1 (R@1) AND RECALL@10 ... BOLD IS THE BEST RESULT.").
    # Requires at least one lowercase-free sentence of real length, so
    # it won't catch short headings like "RESULTS." which are handled
    # separately by heading detection instead.
    re.compile(r"^(?=.{25,}$)[^a-z]*\.$"),
]


_NOISE_LINE = re.compile(
    r"^(?:\d{1,3}|[IVXLC]{1,6}\.?)$"
    r"|©|copyright\s*\(?c?\)?\s*\d{4}|\bISBN\b|\b97[89]-[\d-]+"
    r"|/\$\d+\.\d{2}"
    # Publisher running-header/footer boilerplate (Elsevier, Springer,
    # Wiley, etc.) repeated on every page of a preprint/typeset PDF.
    # This is not part of the paper's actual content.
    r"|please cite this article as"
    r"|how to cite this article"
    r"|this is a pdf file of an unedited manuscript"
    r"|this article (?:is protected by|has been accepted)"
    r"|accepted manuscript"
    r"|downloaded from http"
    # End-of-paper declarations (competing interests, funding, data
    # availability, ethics/consent) are administrative boilerplate,
    # not part of the paper's conclusions, even though they often sit
    # directly under a Conclusion heading with no heading of their own.
    r"|competing (?:financial )?interests?"
    r"|conflicts? of interest"
    r"|this (?:work|research|study) was (?:funded|supported) by"
    r"|data availability statement"
    r"|the data (?:that support|used in)"
    r"|informed consent was obtained"
    r"|this (?:article|study) was approved by",
    re.I,
)


_LIGATURES = {
    0xFB00: "ff",
    0xFB01: "fi",
    0xFB02: "fl",
    0xFB03: "ffi",
    0xFB04: "ffl",
}


def _is_noise_line(line: str) -> bool:
    """Page numbers, lone roman numerals and publisher footers."""

    return bool(_NOISE_LINE.search(line.strip()))


def _clean_text(text: str) -> str:
    """Normalise whitespace and remove invisible characters."""

    text = text.translate(_LIGATURES)

    # Soft hyphen (U+00AD): a discretionary line-break point inside a
    # word, meant to be invisible unless a renderer actually breaks
    # the line there. Some PDF extractors leave it in as a literal
    # character (often with the wrapped word's remainder separated
    # by whitespace, e.g. "guid\xadance" or "guid\xad ance"), which
    # otherwise splits ordinary words apart throughout the document
    # and breaks keyword matching on them.
    text = re.sub(r"\u00ad\s*(?=[a-z])", "", text)
    text = text.replace("\u00ad", "")

    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    # A first-page author-contact / equal-contribution footnote can get
    # read mid-sentence into the main text, because two-column PDF
    # extraction interleaves a column-bottom footnote with the next
    # column's body text (e.g. "...awareness at *e-mail: a@b.edu.
    # First Last and First Last contributed equally to this research.").
    # Strip the footnote marker and the sentence naming it, without
    # touching the real sentence it landed inside of.
    text = re.sub(
        r"\*\s*e-?mail\s*:\s*\S+@\S+\.?\s*", " ", text, flags=re.I
    )
    text = re.sub(
        r"[A-Z][\w.'’-]+(?:\s+[A-Z][\w.'’-]+){0,4}\s+and\s+"
        r"[A-Z][\w.'’-]+(?:\s+[A-Z][\w.'’-]+){0,4}\s+contributed\s+"
        r"equally\s+to\s+this\s+(?:research|work|paper|study)\.?",
        "",
        text,
    )

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _normalise_heading_key(text: str) -> str:
    """Create a comparable key for special heading matching."""

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


_KEYWORD_SECTIONS = [
    ("references", ("reference", "bibliograph", "acknowledg", "appendix")),
    ("research directions", ("conclusion", "future work", "future research", "concluding")),
    ("abstract", ("abstract",)),
    ("introduction", ("introduction",)),
    ("literature review", ("literature review", "related work", "prior work")),
    ("background", ("background", "preliminar")),
    ("results", ("result", "finding", "evaluation")),
    ("discussion", ("discussion", "limitation")),
    ("methods", ("method", "approach", "implementation", "system design",
                 "materials", "experiment", "setup", "procedure")),
]


def _keyword_section(value: str) -> str | None:
    """Map a short heading to a canonical section by keyword."""

    stripped = re.sub(
        r"^(?:(?:\d+(?:\.\d+)*|[IVXLC]+|[A-Z])[\.\)]\s*)",
        "",
        value.strip(),
    ).lower()

    if not stripped or len(stripped.split()) > 8:
        return None

    # Headings are not sentences or questions.
    if stripped.endswith((".", "?", "!", ",", ";", ":")):
        return None

    lead = " ".join(stripped.split()[:4])

    raw = value.strip()
    numbered = re.match(r"^(?:\d+(?:\.\d+)*|[IVXLC]+)[\.\)]\s+\S", raw)
    shouting = raw.upper() == raw and any(ch.isalpha() for ch in raw)

    # Ordinary short lines (table cells, axis labels) must not flip sections.
    if not (numbered or shouting):
        return None

    for section, keys in _KEYWORD_SECTIONS:
        if section == "references":
            if stripped.startswith(keys):
                return section

            continue

        if any(key in lead for key in keys):
            return section

    return None


def normalise_heading(
    value: str,
) -> str | None:
    """Return the normalised section name for a detected heading."""

    value = _clean_text(value)

    if not value:
        return None

    key = _normalise_heading_key(value)

    if key in SPECIAL_HEADINGS:
        return SPECIAL_HEADINGS[key]

    for section, pattern in SECTION_PATTERNS:
        if hasattr(pattern, "fullmatch"):
            matched = pattern.fullmatch(value)
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
    """Return True when text appears to be a figure caption."""

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
    """Find a known heading embedded inside an extracted text line."""

    for pattern, section in INLINE_SPECIAL_HEADINGS:
        match = pattern.search(text)

        if match is None:
            continue

        position = match.start()
        end = match.end()

        # A heading at the start of a line is handled normally.
        if position == 0:
            return (
                position,
                end,
                section,
            )

        # Require a natural text boundary before an inline heading.
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
    """Find the next heading boundary inside a text block."""

    return _find_inline_special_heading(text)


def _looks_like_heading(
    text: str,
) -> bool:
    """Use conservative heuristics to identify likely headings."""

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
        (".", ",", ";", ":"),
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
    """Split a text block when a known inline heading is encountered."""

    text = _clean_text(text)

    if not text:
        return []

    results = []
    remaining = text
    section = current_section

    while remaining:
        match = _find_heading_boundary(
            remaining,
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
            remaining[:position],
        )

        after = _clean_text(
            remaining[end:],
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
    """Convert extracted PDF pages into section-labelled text blocks."""

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

        # Join words split by a line-break hyphen, but keep hyphens that
        # belong to a compound ("state-of-\nthe-art" -> "state-of-the-art").
        raw_text = re.sub(
            r"(\b[a-z]+(?:-[a-z]+)+)-\n(?=[a-z])", r"\1-", raw_text
        )
        raw_text = re.sub(
            r"(?<=[a-z])-\n(?=[a-z])", "", raw_text
        )

        # Same, but for a soft hyphen (U+00AD) at a line break, which
        # some PDF exports leave as a literal character right at the
        # wrap point (e.g. "cup\xad\nboard"). This must happen here,
        # before the text is split into individual lines below --
        # once split, the two halves of the word are cleaned as
        # separate strings and can no longer be rejoined correctly.
        raw_text = re.sub(
            r"\u00ad\n(?=[a-z])", "", raw_text
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
                " ".join(current_block),
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

        previous_was_noise = False

        for line in lines:
            stripped_line = line.strip()

            if _is_noise_line(line):
                previous_was_noise = True
                continue

            # A boilerplate sentence hard-wrapped across a PDF line break
            # (e.g. "no known competing financial interests or personal\n
            # relationships that could have appeared to influence...")
            # only has its first half matched by _is_noise_line. Its
            # continuation looks like ordinary text on its own, so it
            # is dropped too when it directly follows a dropped noise
            # line and reads like a continuation (starts lowercase, no
            # sentence-ending punctuation yet been reached at its
            # start) rather than the start of a new sentence.
            if (
                previous_was_noise
                and stripped_line
                and stripped_line[0].islower()
            ):
                previous_was_noise = False
                continue

            previous_was_noise = False

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

            inline_abstract = re.match(
                r"^abstract\s*[—–:.\-]\s*(.+)$", line, re.I
            )

            if inline_abstract:
                flush_block()
                current_section = "abstract"
                current_block.append(inline_abstract.group(1))
                continue

            heading_section = normalise_heading(line)

            if heading_section is not None:
                flush_block()
                current_section = heading_section
                continue

            inline_heading = _find_inline_special_heading(line)

            if inline_heading is not None:
                position, end, new_section = inline_heading

                before = _clean_text(
                    line[:position],
                )

                after = _clean_text(
                    line[end:],
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
                    current_block.append(after)

                continue

            if _looks_like_heading(line):
                flush_block()

                detected = normalise_heading(line) or _keyword_section(line)

                if detected is not None:
                    current_section = detected

                continue

            current_block.append(line)

        flush_block()

    return sections


def _merge_short_heading_blocks(
    sections: list[dict],
) -> list[dict]:
    """Merge very short adjacent blocks belonging to the same section."""

    if not sections:
        return []

    merged = []

    for item in sections:
        text = item["text"].strip()

        if (
            merged
            and len(text.split()) <= 3
            and item["section"] == merged[-1]["section"]
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


_NOT_SENTENCE_END = {
    "al.", "e.g.", "i.e.", "fig.", "figs.", "eq.", "vs.", "etc.",
    "no.", "dr.", "cf.", "approx.",
}


def _snap_to_sentence_end(
    words: list[str],
    start: int,
    end: int,
) -> int:
    """Move a chunk end back to the nearest sentence end, if one is close."""

    lowest = start + int((end - start) * 0.6)

    for index in range(end - 1, lowest - 1, -1):
        word = words[index]

        if word.endswith((".", "?", "!")) and word.lower() not in _NOT_SENTENCE_END:
            return index + 1

    return end


def create_chunks(
    pages: list[dict],
    chunk_size: int = 180,
    overlap: int = 40,
) -> list[ResearchChunk]:
    """Create overlapping word-based chunks from section-labelled text."""

    sections = split_into_sections(pages)

    sections = _merge_short_heading_blocks(
        sections,
    )

    chunks = []
    chunk_counter = 0

    for section in sections:
        page = int(section["page"])

        text = _clean_text(
            section["text"],
        )

        section_name = section.get(
            "section",
        )

        if not text:
            continue

        if section_name == "figure_caption":
            continue

        words = text.split()

        if len(words) < MIN_CHUNK_WORDS:
            continue

        start = 0

        while start < len(words):
            end = min(
                start + chunk_size,
                len(words),
            )

            if end < len(words):
                end = _snap_to_sentence_end(words, start, end)

            chunk_text = " ".join(
                words[start:end],
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