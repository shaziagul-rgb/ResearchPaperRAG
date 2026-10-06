"""Configuration for the ResearchPaperRAG application."""

# Embedding model used for semantic retrieval.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Document chunking (word counts per chunk).
CHUNK_SIZE = 180
CHUNK_OVERLAP = 40

# Upload limits.
MAX_FILE_SIZE_MB = 25

# Retrieval settings.
DEFAULT_TOP_K = 8       # evidence pool size per category in /api/analyze
QUESTION_TOP_K = 5      # evidence pool size for /api/ask
MAX_EVIDENCE_PER_CATEGORY = 3

# Minimum combined retrieval score required for evidence.
MIN_EVIDENCE_SCORE = 0.30   # /api/ask cutoff
FOUND_SCORE = 0.30          # /api/analyze: score needed for 'found'
MISSING_SCORE = 0.18        # /api/analyze: below this the category is 'missing'

# Local Ollama generation service.
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"

# OCR fallback for PDFs whose text layer is missing or unreadable.
# Requires the Tesseract program (macOS: brew install tesseract).
# For other languages use e.g. "eng+urd" (needs the language data installed).
OCR_LANGUAGE = "eng"
OCR_DPI = 200
