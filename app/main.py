from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

from .analysis import analyse
from .chunking import create_chunks
from .config import MAX_FILE_SIZE_MB, MODEL_NAME
from .pdf_parser import extract_pdf_pages
from .schemas import AnalysisResponse


app = FastAPI(
    title="Research Evidence API",
    version="0.1.0",
    description=(
        "API for extracting and retrieving "
        "evidence from research papers."
    ),
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_model = None


def get_model():
    """
    Load the embedding model once and reuse it.

    The first analysis takes longer because the model
    needs to be downloaded and loaded locally.
    """

    global _model

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    return _model


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "research-evidence-api",
    }


@app.post(
    "/api/analyze",
    response_model=AnalysisResponse,
)
async def analyze_paper(
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Please provide a PDF file.",
        )

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    pdf_bytes = await file.read()

    max_bytes = (
        MAX_FILE_SIZE_MB
        * 1024
        * 1024
    )

    if len(pdf_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"PDF must be smaller than "
                f"{MAX_FILE_SIZE_MB} MB."
            ),
        )

    try:
        pages = extract_pdf_pages(
            pdf_bytes
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="The PDF could not be read.",
        ) from exc

    non_empty_pages = [
        page
        for page in pages
        if page["text"].strip()
    ]

    if not non_empty_pages:
        raise HTTPException(
            status_code=422,
            detail=(
                "No selectable text was found. "
                "Please use a text-based PDF rather "
                "than a scanned image-only PDF."
            ),
        )

    chunks = create_chunks(
        non_empty_pages
    )

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail=(
                "No usable text passages "
                "were found in the PDF."
            ),
        )

    model = get_model()

    texts = [
        chunk.text
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    results = analyse(
        chunks,
        embeddings,
        model,
    )

    found_count = sum(
        1
        for item in results
        if item["status"] == "found"
    )

    evidence_coverage = round(
        (
            found_count
            / len(results)
        )
        * 100
    )

    evidence_gaps = [
        item["label"]
        for item in results
        if item["status"] == "missing"
    ]

    return AnalysisResponse(
        filename=file.filename,
        pages=len(pages),
        chunks=len(chunks),
        evidence_coverage=evidence_coverage,
        analysis=results,
        evidence_gaps=evidence_gaps,
    )