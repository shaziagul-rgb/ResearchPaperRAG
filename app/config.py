MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 650
CHUNK_OVERLAP = 100

MAX_FILE_SIZE_MB = 25

DEFAULT_TOP_K = 8
MAX_EVIDENCE_PER_CATEGORY = 3

# A candidate must reach this combined score before
# it can normally be treated as evidence.
MIN_EVIDENCE_SCORE = 0.38

# Local Ollama model.
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"