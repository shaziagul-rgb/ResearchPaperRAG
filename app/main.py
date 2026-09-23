from __future__ import annotations

from dataclasses import asdict

import fitz
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .analysis import analyse
from .chunking import create_chunks
from .config import MAX_FILE_SIZE_MB
from .generation import generate_answer
from .retrieval import EvidenceRetriever


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


def extract_pages(
    file_bytes: bytes,
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

    for page_number, page in enumerate(
        document,
        start=1,
    ):
        text = page.get_text(
            "text"
        ).strip()

        pages.append(
            {
                "page": page_number,
                "text": text,
            }
        )

    document.close()

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
        chunk_size=180,
        overlap=40,
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
        file_bytes
    )

    return {
        "filename": file.filename,
        "pages": len(pages),
        "characters": sum(
            len(page["text"])
            for page in pages
        ),
        "preview": [
            {
                "page": page["page"],
                "text": page["text"][:500],
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
        top_k=5,
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