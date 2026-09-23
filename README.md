# ResearchPaperRAG

Evidence-grounded question answering for research papers using
section-aware retrieval and a local language model.

## Overview

ResearchPaperRAG is a local RAG system that extracts research
documents, creates section-aware chunks, retrieves relevant
evidence, and generates grounded answers using Qwen2.5:3b
through Ollama.

## Architecture

PDF
 ↓
PyMuPDF
 ↓
Section-aware chunking
 ↓
Semantic + keyword retrieval
 ↓
Question intent detection
 ↓
Evidence selection
 ↓
Qwen2.5:3b / Ollama
 ↓
Grounded answer + page citations

## Features

- PDF text extraction
- Section-aware document chunking
- Document structure detection
- Semantic retrieval
- Keyword-based retrieval
- Question intent detection
- Evidence filtering
- Grounded answer generation
- Page-level citations
- Insufficient-evidence handling
- FastAPI REST API
- Local LLM inference with Ollama

## Example

Question:
What research methods are discussed in localization research?

Answer:
...

[p. 10; p. 12; p. 11]

## Tech Stack

Python
FastAPI
PyMuPDF
Sentence Transformers
Ollama
Qwen2.5:3b
NumPy

## Project Structure

app/
├── analysis.py
├── chunking.py
├── config.py
├── generation.py
├── main.py
└── retrieval.py

## Installation

...

## Running the API

...

## API Endpoints

GET  /
GET  /health
POST /api/analyze
POST /api/ask

## Design Decisions

### Section-aware retrieval
...

### Evidence grounding
...

### Local inference
...

## Evaluation

The system was tested against questions covering:

- definitions
- research areas
- research methods
- theoretical approaches
- future research
- web accessibility
- localization training
- unsupported topics

The unsupported-topic test verifies that the system returns
an insufficient-evidence response rather than generating an
unsupported answer.


