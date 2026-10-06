from __future__ import annotations

from dataclasses import asdict
import os

import fitz
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .analysis import analyse
from .chunking import create_chunks
from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_FILE_SIZE_MB,
    OCR_DPI,
    OCR_LANGUAGE,
    QUESTION_TOP_K,
)
from .fallback_extract import extract_pages_with_pypdf
from .generation import generate_answer
from .retrieval import EvidenceRetriever
from .text_quality import is_garbled, strip_control_chars, text_quality


app = FastAPI(
    title="ResearchPaperRAG Evidence API",
    version="0.5.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_TESSDATA_CANDIDATES = [
    "/opt/homebrew/share/tessdata",
    "/usr/local/share/tessdata",
    "/usr/share/tesseract-ocr/5/tessdata",
    "/usr/share/tesseract-ocr/4.00/tessdata",
    "/usr/share/tessdata",
]


def _find_tessdata() -> str | None:
    env = os.environ.get("TESSDATA_PREFIX")

    if env and os.path.isdir(env):
        return env

    for path in _TESSDATA_CANDIDATES:
        if os.path.isdir(path):
            return path

    return None


def _ocr_page(page) -> str | None:
    """OCR one page with PyMuPDF + Tesseract. None if OCR is unavailable."""

    try:
        kwargs = {
            "language": OCR_LANGUAGE,
            "dpi": OCR_DPI,
            "full": True,
        }

        tessdata = _find_tessdata()

        if tessdata:
            kwargs["tessdata"] = tessdata

        textpage = page.get_textpage_ocr(**kwargs)

        return page.get_text(
            "text",
            textpage=textpage,
        ).strip()

    except Exception:
        return None


def extract_pages(
    file_bytes: bytes,
    strict: bool = True,
) -> list[dict]:
    try:
        document = fitz.open(
            stream=file_bytes,
            filetype="pdf",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read PDF: {exc}",
        ) from exc

    pages = []
    ocr_available = True
    fallback_texts: list[str] | None = None
    fallback_tried = False

    for page_number, page in enumerate(
        document,
        start=1,
    ):
        text = page.get_text(
            "text"
        ).strip()

        used_ocr = False
        used_fallback = False
        garbled = is_garbled(text)
        empty_scan = (
            not text
            and bool(page.get_images())
        )

        # Second chance: decode glyph codes with pypdf (no OCR needed).
        if garbled:
            if not fallback_tried:
                fallback_tried = True
                fallback_texts = extract_pages_with_pypdf(file_bytes)

            if fallback_texts and page_number <= len(fallback_texts):
                candidate = fallback_texts[page_number - 1].strip()

                if candidate and not is_garbled(candidate):
                    text = candidate
                    garbled = False
                    used_fallback = True

        if (garbled or empty_scan) and ocr_available:
            ocr_text = _ocr_page(page)

            if ocr_text is None:
                ocr_available = False
            elif ocr_text and not is_garbled(ocr_text):
                text = ocr_text
                garbled = False
                used_ocr = True

        if not garbled:
            text = strip_control_chars(text)

        pages.append(
            {
                "page": page_number,
                "text": text,
                "garbled": garbled,
                "ocr": used_ocr,
                "fallback": used_fallback,
            }
        )

    document.close()

    bad_pages = sum(
        1 for item in pages if item["garbled"]
    )

    if strict and pages and bad_pages > len(pages) / 2:
        raise HTTPException(
            status_code=422,
            detail=(
                f"The text in this PDF is unreadable on {bad_pages} of "
                f"{len(pages)} pages (its fonts have no Unicode mapping, "
                "so text extracts as symbols). "
                + (
                    "OCR was attempted but did not help."
                    if ocr_available
                    else "Install Tesseract for OCR "
                    "(macOS: brew install tesseract) and try again, "
                    "or re-export the PDF (e.g. Print to PDF)."
                )
            ),
        )

    return pages


def validate_pdf(
    file_bytes: bytes,
) -> None:
    max_bytes = (
        MAX_FILE_SIZE_MB
        * 1024
        * 1024
    )

    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"PDF is larger than the "
                f"{MAX_FILE_SIZE_MB} MB limit."
            ),
        )

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded PDF is empty.",
        )


def validate_question(
    question: str,
) -> str:
    """
    Validate and normalize a user question.

    Swagger/OpenAPI uses 'string' as the default
    placeholder for string form fields. Reject it
    so it is never sent to the retrieval pipeline.
    """

    question = question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    if question.lower() == "string":
        raise HTTPException(
            status_code=400,
            detail="Please enter a real question.",
        )

    return question


def create_document_chunks(
    pages: list[dict],
) -> list[dict]:
    chunks = create_chunks(
        pages,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    return [
        asdict(chunk)
        for chunk in chunks
    ]


@app.get("/")
def root() -> dict:
    return {
        "name": "ResearchPaperRAG Evidence API",
        "version": "0.5.0",
        "status": "running",
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
    }


@app.post("/api/debug-pdf")
async def debug_pdf(
    file: UploadFile = File(...),
) -> dict:
    file_bytes = await file.read()

    validate_pdf(file_bytes)

    pages = extract_pages(
        file_bytes,
        strict=False,
    )

    chunks = create_chunks(
        pages,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    cleaned_preview_by_page = {}

    for chunk in chunks:
        if chunk.page not in cleaned_preview_by_page:
            cleaned_preview_by_page[chunk.page] = chunk.text

    return {
        "filename": file.filename,
        "pages": len(pages),
        "characters": sum(
            len(page["text"])
            for page in pages
        ),
        "pages_ocr": sum(
            1 for page in pages if page["ocr"]
        ),
        "pages_decoded_with_fallback": sum(
            1 for page in pages if page["fallback"]
        ),
        "pages_unreadable": sum(
            1 for page in pages if page["garbled"]
        ),
        "preview": [
            {
                "page": page["page"],
                "raw_text": page["text"][:500],
                "cleaned_text_after_chunking": (
                    cleaned_preview_by_page.get(
                        page["page"],
                        "(no chunk produced for this page)",
                    )[:500]
                ),
                "quality": text_quality(page["text"]),
                "ocr": page["ocr"],
            }
            for page in pages[:3]
        ],
    }


@app.post("/api/analyze")
async def analyze_pdf(
    file: UploadFile = File(...),
) -> dict:
    file_bytes = await file.read()

    validate_pdf(file_bytes)

    pages = extract_pages(
        file_bytes
    )

    total_text = "".join(
        page["text"]
        for page in pages
    ).strip()

    if not total_text:
        raise HTTPException(
            status_code=400,
            detail=(
                "No readable text was found "
                "in the PDF."
            ),
        )

    chunk_dicts = create_document_chunks(
        pages
    )

    result = analyse(
        chunk_dicts
    )

    return {
        "id": "analysis",
        "filename": file.filename,
        "pages": len(pages),
        "chunks": len(chunk_dicts),
        **result,
    }


@app.post("/api/ask")
async def ask_question(
    file: UploadFile = File(...),
    question: str = Form(...),
) -> dict:
    # Validate the question before doing any
    # PDF processing, embedding, or LLM inference.
    question = validate_question(
        question
    )

    file_bytes = await file.read()

    validate_pdf(file_bytes)

    pages = extract_pages(
        file_bytes
    )

    total_text = "".join(
        page["text"]
        for page in pages
    ).strip()

    if not total_text:
        raise HTTPException(
            status_code=400,
            detail=(
                "No readable text was found "
                "in the PDF."
            ),
        )

    chunk_dicts = create_document_chunks(
        pages
    )

    retriever = EvidenceRetriever()

    evidence = retriever.retrieve_question(
        chunk_dicts,
        question,
        top_k=QUESTION_TOP_K,
    )

    if not evidence:
        return {
            "question": question,
            "answer": (
                "The provided evidence does not contain "
                "enough information to answer this question."
            ),
            "evidence": [],
        }

    try:
        generated = generate_answer(
            question,
            evidence,
        )

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    return {
        "question": question,
        "answer": generated.answer,
        "evidence": generated.evidence,
    }