"""Second-chance text extraction for PDFs whose fonts have no Unicode map.

Some PDFs (old Acrobat Distiller / publisher files) use Type 1 fonts with
custom encodings. PyMuPDF then returns control characters, while pypdf can
still recover the glyph codes as tokens like ``/C101`` (= "e"). This module
decodes those tokens and repairs word spacing, without needing OCR.
"""

from __future__ import annotations

import io
import re

# Adobe StandardEncoding codes above 127 that appear in text.
_STANDARD_HIGH = {
    161: "¡", 162: "¢", 163: "£", 164: "⁄", 165: "¥", 166: "ƒ", 167: "§",
    168: "¤", 169: "'", 170: "“", 171: "«", 172: "‹", 173: "›", 174: "ﬁ",
    175: "ﬂ", 177: "–", 178: "†", 179: "‡", 180: "·", 182: "¶", 183: "•",
    184: "‚", 185: "„", 186: "”", 187: "»", 188: "…", 189: "‰", 191: "¿",
    193: "`", 194: "´", 195: "ˆ", 196: "˜", 197: "¯", 198: "˘", 199: "˙",
    200: "¨", 202: "˚", 203: "¸", 205: "˝", 206: "˛", 207: "ˇ", 208: "—",
    225: "Æ", 227: "ª", 232: "Ł", 233: "Ø", 234: "Œ", 235: "º", 241: "æ",
    245: "ı", 248: "ł", 249: "ø", 250: "œ", 251: "ß",
}

_GLYPH_CODE = re.compile(r"/C(\d{1,3})(?:\.[\w.]*)?")
_GLYPH_UNI = re.compile(r"/uni([0-9A-Fa-f]{4})")

# TJ gaps at least this negative are treated as word gaps and widened so
# pypdf's space detection (which assumes a full-width space) sees them.
_WORD_GAP_MIN = 150.0
_WORD_GAP_BOOST = 1.3


def _decode_code(match: re.Match) -> str:
    code = int(match.group(1))

    if code == 39:
        return "’"

    if code == 96:
        return "‘"

    if code < 128:
        return chr(code)

    return _STANDARD_HIGH.get(code, "")


def decode_glyph_tokens(text: str) -> str:
    """Turn ``/C101`` style glyph tokens into real characters."""

    text = _GLYPH_CODE.sub(_decode_code, text)
    text = _GLYPH_UNI.sub(lambda m: chr(int(m.group(1), 16)), text)

    return text


def _widen_gaps_callback(operator, operands, cm_matrix, tm_matrix):
    """pypdf visitor: widen word-sized TJ gaps so spaces are detected."""

    if operator != b"TJ" or not operands:
        return

    array = operands[0]

    try:
        for index, value in enumerate(array):
            if isinstance(value, (bytes, str)):
                continue

            number = float(value)

            if number <= -_WORD_GAP_MIN:
                array[index] = type(value)(number * _WORD_GAP_BOOST) \
                    if isinstance(value, float) else number * _WORD_GAP_BOOST
    except Exception:
        return


def extract_pages_with_pypdf(file_bytes: bytes) -> list[str] | None:
    """Return one decoded text string per page, or None if pypdf is absent."""

    try:
        from pypdf import PdfReader
    except ImportError:
        return None

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception:
        return None

    texts: list[str] = []

    for page in reader.pages:
        try:
            raw = page.extract_text(
                visitor_operand_before=_widen_gaps_callback,
            ) or ""

            texts.append(decode_glyph_tokens(raw))
        except Exception:
            texts.append("")

    return texts
