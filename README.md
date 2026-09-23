# ResearchPaperRAG

Evidence-grounded question answering for research papers using section-aware retrieval, semantic search, and a local language model.

## Overview

ResearchPaperRAG is a local Retrieval-Augmented Generation (RAG) system for asking questions about research documents.

The system extracts text from PDF documents, identifies document sections, creates section-aware chunks, retrieves relevant evidence using semantic and keyword signals, detects the intent of the question, and generates an answer using a local Qwen2.5:3b model through Ollama.

The main goal is to keep generated answers grounded in the supplied document rather than relying on unsupported model knowledge.

## Architecture

```text
PDF
 │
 ▼
PyMuPDF
 │
 ▼
Section-aware document chunking
 │
 ▼
Semantic + keyword retrieval
 │
 ▼
Question intent detection
 │
 ▼
Evidence selection
 │
 ▼
Qwen2.5:3b via Ollama
 │
 ▼
Grounded answer + page citation
```

## Key Features

* PDF text extraction with PyMuPDF
* Section-aware document chunking
* Academic section and heading detection
* Semantic retrieval using Sentence Transformers
* Keyword-based retrieval
* Question-intent detection
* Section-aware retrieval weighting
* Evidence filtering
* Reference and figure-caption filtering
* Grounded answer generation
* Page-level citations
* Insufficient-evidence handling
* FastAPI REST API
* Local LLM inference through Ollama

## Retrieval Approach

The retrieval pipeline combines several signals rather than relying only on embedding similarity.

### Semantic Similarity

Document chunks and queries are embedded using Sentence Transformers.

Semantic similarity provides the main signal for identifying conceptually relevant passages.

### Keyword Matching

Important terms from the question are compared with candidate chunks.

This helps preserve terminology that may be important to the question even when semantic similarity alone is not sufficient.

### Query Overlap

For direct question answering, the system also considers overlap between meaningful question terms and candidate evidence.

### Section Awareness

The retriever gives additional weight to sections that are particularly relevant to the question.

For example:

* definition questions favour definitions and theoretical framework sections
* research-area questions favour research areas and research directions
* methods questions favour methods and methodology-related evidence
* theory questions favour theoretical framework and theoretical approaches
* future-research questions favour research directions and conclusions

### Question Intent

The system detects several question types, including:

* definitions
* research areas
* research methods
* theoretical approaches
* future research
* evidence and studies
* general questions

The detected intent influences evidence ranking before the selected passages are sent to the language model.

## Evidence Grounding

The language model receives only the evidence retrieved for the question.

The generation prompt instructs the model to:

* use only the supplied evidence
* avoid unsupported claims
* avoid adding outside knowledge
* distinguish research methods from research topics
* avoid reproducing academic citations
* provide concise answers
* return an insufficient-evidence response when the evidence is not sufficient

Page citations are added by the application based on the retrieved evidence.

## Local LLM

Answer generation uses:

```text
Qwen2.5:3b
```

through a locally running Ollama server.

No external LLM API is required for answer generation.

## Example

### Question

> What research methods are discussed in localization research?

### Example Answer

The chapter discusses several methodological approaches, including descriptive and theoretical studies, empirical and corpus-based research, experimental approaches, case studies, error analysis, sociological and ethnographic approaches, and social network analysis. [p. 10; p. 11; p. 12]

The exact answer depends on the evidence retrieved from the supplied document.

## API

### `GET /`

Returns basic information about the API.

### `GET /health`

Returns the API health status.

### `POST /api/analyze`

Analyzes the supplied PDF and reports document structure and evidence coverage across predefined research categories.

### `POST /api/ask`

Answers a question using evidence retrieved from the supplied PDF.

The response contains:

* generated answer
* supporting evidence
* page numbers
* section information
* retrieval scores

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/shaziagul-rgb/ResearchPaperRAG.git
cd ResearchPaperRAG
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Install Ollama separately and make sure the local Ollama service is running.

Pull the required model:

```bash
ollama pull qwen2.5:3b
```

Verify the model:

```bash
ollama list
```

The application is configured to use:

```text
qwen2.5:3b
```

## Running the API

Start the FastAPI application:

```bash
python3 -m uvicorn app.main:app --reload
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

FastAPI provides interactive API documentation through Swagger UI.

## Project Structure

```text
ResearchPaperRAG/
│
├── app/
│   ├── analysis.py
│   ├── chunking.py
│   ├── config.py
│   ├── generation.py
│   ├── main.py
│   └── retrieval.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

### `chunking.py`

Extracted PDF text is converted into section-aware chunks.

The chunking stage recognises academic sections and handles heading variations that can occur during PDF text extraction.

### `retrieval.py`

Implements:

* semantic retrieval
* keyword scoring
* query-term overlap
* section weighting
* question-intent detection
* evidence filtering

### `analysis.py`

Uses the retrieval system to analyse a document across predefined research categories.

### `generation.py`

Builds the grounded prompt, sends the selected evidence to Ollama, cleans the generated response, and attaches page citations.

### `main.py`

Provides the FastAPI application and REST endpoints.

### `config.py`

Contains model configuration, retrieval thresholds, and the Ollama endpoint.

## Design Decisions

### Section-Aware Retrieval

Academic documents contain meaningful structural information.

A passage from a theoretical framework, methods section, or research-directions section can have a different relevance to a question even when the text has similar semantic similarity.

The retriever therefore uses document sections as an additional ranking signal.

### Hybrid Retrieval

Semantic similarity is useful for identifying conceptually related passages, while keyword matching helps preserve important terminology.

The system combines both signals instead of relying on a single retrieval method.

### Question-Intent Detection

Different questions require different evidence.

For example, a question about future research should favour research-directions and conclusion sections, while a definition question should favour definitions and theoretical framework sections.

Intent detection provides an additional ranking signal for this purpose.

### Evidence Threshold

Retrieved passages must reach a minimum combined score before being treated as supporting evidence.

This allows the system to return an insufficient-evidence response instead of forcing the language model to answer every question.

### Local Inference

The project uses Ollama and Qwen2.5:3b for local answer generation.

This keeps the generation pipeline independent of an external hosted LLM API.

## Evaluation

The system was tested using questions covering several information needs:

| Test Area              | Example Question                                                     |
| ---------------------- | -------------------------------------------------------------------- |
| Definitions            | What is localization?                                                |
| Research Areas         | What are the main research areas discussed in localization research? |
| Research Methods       | What research methods are discussed in localization research?        |
| Theoretical Approaches | What theoretical approaches are discussed in the chapter?            |
| Future Research        | What future research directions are identified?                      |
| Web Accessibility      | What role does web accessibility play in localization research?      |
| Localization Training  | What does the chapter say about localization training?               |
| Unsupported Topics     | What does the chapter say about quantum computing?                   |

The unsupported-topic test is particularly important for a grounded RAG system.

When the supplied document does not contain relevant evidence, the system returns:

```text
The provided evidence does not contain enough information to answer this question.
```

This prevents the system from intentionally generating an answer when relevant evidence cannot be retrieved.

## Limitations

The current implementation is designed primarily for research and engineering experimentation rather than production-scale document processing.

Current limitations include:

* PDF extraction quality depends on the source document
* heading detection uses heuristics
* retrieval thresholds are manually configured
* documents are currently processed in memory
* answer generation depends on the locally available Ollama model
* evaluation currently uses a small set of representative questions
* the system does not yet provide a formal benchmark against multiple RAG architectures

## Future Improvements

Potential extensions include:

* automated retrieval evaluation datasets
* retrieval precision and recall measurements
* larger document collections
* persistent vector storage
* document management and indexing
* improved section and heading detection
* streaming model responses
* richer citation mapping
* frontend integration
* experiment tracking
* automated regression tests for retrieval behaviour

## Technology Stack

* Python
* FastAPI
* PyMuPDF
* Sentence Transformers
* NumPy
* Ollama
* Qwen2.5:3b

## Status

ResearchPaperRAG is an experimental research and engineering project exploring evidence-grounded question answering over academic documents.

The current implementation focuses on document structure, hybrid evidence retrieval, question intent, grounded generation, and local LLM inference.
