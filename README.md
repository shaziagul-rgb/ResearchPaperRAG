# ResearchPaperRAG
# ResearchPaperRAG

**Evidence-Grounded Research Paper Analysis with RAG**

ResearchPaperRAG is a research paper analysis API built with **Python and FastAPI**. It extracts text from research papers, retrieves relevant evidence using semantic search, and uses a **local LLM** to generate short explanations grounded in the retrieved passages.

The main goal is to help answer a practical research question:

> **What evidence does a research paper provide about its methodology, and how reproducible is that information?**

Rather than asking an LLM to analyse an entire paper directly, ResearchPaperRAG first retrieves relevant evidence from the paper and then uses the LLM to explain that evidence.

---

## What it does

ResearchPaperRAG analyses a research paper across several methodology categories:

* Research Problem
* Dataset
* Method / Model
* Training
* Evaluation
* Results
* Limitations / Future Work

For each category, the system attempts to identify relevant passages and records:

* Evidence from the paper
* Page number
* Detected section
* Retrieval score
* Confidence level
* Evidence status
* Alternative evidence passages

It also calculates an **evidence coverage** score showing how many of the analysis categories have sufficient supporting evidence.

---

## How the RAG pipeline works

```text
Research Paper (PDF)
        │
        ▼
   PDF Extraction
        │
        ▼
      Chunking
        │
        ▼
 Semantic Embeddings
        │
        ▼
 Section-Aware Retrieval
        │
        ▼
 Evidence Validation
        │
        ▼
 Retrieved Evidence
        │
        ▼
      Local LLM
        │
        ▼
 Grounded Explanation
```

The retrieval stage happens **before** the LLM is used.

This is important because the LLM is instructed to explain only the evidence retrieved from the paper rather than inventing missing methodology details.

---

## Technology Stack

### Backend

* Python
* FastAPI
* Uvicorn
* PyMuPDF
* Sentence Transformers
* NumPy
* HTTPX

### AI / NLP

* Semantic embeddings
* Section-aware retrieval
* Retrieval-Augmented Generation (RAG)
* Local LLM inference with Ollama
* Qwen2.5 3B

### API

The backend provides a REST API for uploading and analysing PDF research papers.

---

## Example Analysis

A paper can be uploaded through the API and analysed across the seven categories.

Example output:

```text
Research Problem
Status: Found
Confidence: Strong
Page: 2

Dataset
Status: Found
Confidence: Moderate
Page: 4

Method / Model
Status: Found
Confidence: Strong
Page: 6

Training
Status: Partial
Confidence: Weak
Page: 8

Evaluation
Status: Found
Confidence: Moderate
Page: 10

Results
Status: Found
Confidence: Strong
Page: 11

Limitations / Future Work
Status: Partial
Confidence: Weak
Page: 15

Evidence coverage: 86%
```

The coverage value represents **evidence coverage across the defined categories**. It should not be interpreted as a guarantee that the paper is fully reproducible.

---

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/shaziagul-rgb/ResearchPaperRAG.git
cd ResearchPaperRAG
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it:

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the API

Start the FastAPI development server:

```bash
python -m uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

The Swagger interface can be used to upload a PDF and test the analysis endpoint directly.

---

## Local LLM

ResearchPaperRAG uses **Ollama** for local LLM inference.

Install Ollama and download the model:

```bash
ollama pull qwen2.5:3b
```

Make sure Ollama is running before using the LLM-powered explanation stage.

The project is designed so that the core evidence retrieval pipeline does not depend on a paid cloud API.

---

## API Endpoint

### `POST /api/analyze`

Upload a research paper as a PDF.

Example using `curl`:

```bash
curl -X POST \
  http://127.0.0.1:8000/api/analyze \
  -F "file=@paper.pdf"
```

The response contains the paper metadata, retrieved evidence, category analysis, evidence coverage, and identified evidence gaps.

---

## Project Structure

```text
ResearchPaperRAG/
│
├── app/
│   ├── __init__.py
│   ├── analysis.py
│   ├── chunking.py
│   ├── config.py
│   ├── llm.py
│   ├── main.py
│   ├── pdf_parser.py
│   ├── retrieval.py
│   └── schemas.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Design Principles

### Evidence before generation

The LLM is not responsible for discovering evidence in the paper. Retrieval happens first, and the retrieved passages are supplied to the LLM.

### Grounded explanations

The LLM is instructed to use only the supplied evidence and avoid adding information that is not supported by the paper.

### Page-level evidence

Retrieved passages retain their source page number so that users can return to the original paper and verify the information.

### Reproducibility focus

The analysis is organised around information that researchers commonly need when trying to understand or reproduce a study, such as datasets, methodology, training, evaluation and results.

---

## Current Limitations

ResearchPaperRAG is an experimental project and the retrieval system is not perfect.

In particular:

* PDF text extraction can be affected by document formatting.
* Section detection is currently heuristic.
* Semantic retrieval can return passages that are related to a topic but do not contain the exact evidence required.
* Evidence coverage does not mean that a study is reproducible.
* Local LLM performance depends on the model and available hardware.
* Tables, figures and complex mathematical notation require additional processing for complete coverage.

These limitations are part of the reason for keeping the evidence retrieval stage separate from the LLM generation stage.

---

## Future Development

Possible improvements include:

* Better section and heading detection
* Hybrid keyword + semantic retrieval
* Reranking retrieved passages
* Improved table and figure extraction
* Claim-to-evidence mapping
* Multi-paper comparison
* Reproducibility checklist generation
* Evidence citations linked directly to PDF pages
* Evaluation of retrieval quality
* Support for larger local language models
* Web-based frontend for interactive analysis

---

## Why this project?

ResearchPaperRAG combines several areas of practical AI and software engineering:

**Python + FastAPI + NLP + Semantic Search + RAG + LLMs + Information Retrieval**

The project is intended as a practical example of building an AI application where retrieval, evidence and generation are separate and testable components.

---

## License

MIT License
