# MathPaper-Agent

MathPaper-Agent is a small retrieval-augmented generation prototype for mathematical papers.

The goal of this project is to help users ask questions about a PDF paper by retrieving relevant excerpts from the paper and generating a prompt that can be used with a large language model.

This project is currently a minimal local RAG system. It does not require an OpenAI API key for retrieval, because embeddings are computed locally using `sentence-transformers`.

## Features

Current version: `v0.2`

* Extract text from a PDF paper
* Split the extracted text into overlapping chunks
* Compute local embeddings using `sentence-transformers`
* Retrieve relevant chunks for a user question
* Generate a prompt containing the question and relevant paper excerpts
* Support arbitrary PDF file paths from the command line
* Run the full local RAG pipeline with a single command

## Current Pipeline

```text
PDF
→ text extraction
→ chunking
→ local embedding
→ semantic retrieval
→ prompt generation
→ LLM answer
```

## Project Structure

```text
mathpaper-agent/
├── data/
│   └── papers/
│       └── your_paper.pdf
├── src/
│   ├── read_pdf.py
│   ├── chunk_text.py
│   ├── embed_chunks.py
│   ├── search_chunks.py
│   ├── make_prompt.py
│   └── run_rag.py
├── .gitignore
├── README.md
└── requirements.txt
```

Generated files such as extracted text, chunks, embeddings, and prompts are ignored by Git.

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows CMD:

```bat
.venv\Scripts\activate.bat
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Put a PDF paper in:

```text
data/papers/
```

For example:

```text
data/papers/test.pdf
```

Then run the full pipeline:

```bash
python src/run_rag.py data/papers/test.pdf
```

The program will:

1. Extract text from the PDF
2. Split the text into chunks
3. Compute local embeddings
4. Ask for a user question
5. Retrieve relevant chunks
6. Generate a RAG prompt

The generated prompt will be saved as:

```text
data/papers/test_answer_prompt.md
```

You can copy the content of this file into ChatGPT or another LLM to get an answer grounded in the retrieved paper excerpts.

## Example

Run:

```bash
python src/run_rag.py data/papers/test.pdf --query "What is the main theorem of this paper?"
```

To automatically open the generated prompt file with Notepad:

```bash
python src/run_rag.py data/papers/test.pdf --query "What is the main theorem of this paper?" --open
```

## Individual Scripts

The full pipeline can be run with `run_rag.py`, but each step can also be executed separately.

Extract text from a PDF:

```bash
python src/read_pdf.py data/papers/test.pdf
```

Split extracted text into chunks:

```bash
python src/chunk_text.py data/papers/test.txt
```

Compute local embeddings:

```bash
python src/embed_chunks.py data/papers/test_chunks
```

Search relevant chunks:

```bash
python src/search_chunks.py data/papers/test_embeddings.json --query "What is the main theorem of this paper?"
```

Generate a RAG prompt:

```bash
python src/make_prompt.py data/papers/test_embeddings.json --query "What is the main theorem of this paper?"
```

## Main Components

### PDF Text Extraction

`read_pdf.py` uses `pypdf` to extract text from PDF pages and save it as a `.txt` file.

### Chunking

`chunk_text.py` splits a long paper into overlapping text chunks. Overlap is used to reduce the risk of cutting a theorem, lemma, or proof in the middle.

### Embedding

`embed_chunks.py` uses a local sentence-transformers model to convert each text chunk into a vector representation.

The default model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

### Semantic Search

`search_chunks.py` embeds the user query and compares it with all chunk embeddings using cosine similarity.

Since embeddings are normalized, the dot product between two vectors is used as the similarity score.

### Prompt Generation

`make_prompt.py` retrieves the most relevant chunks and formats them into a prompt for a large language model.

The generated prompt asks the LLM to answer only using the retrieved paper excerpts.

## Motivation

Mathematical papers are often long, technical, and highly structured. This project explores how retrieval-based LLM tools can help with mathematical literature reading, theorem lookup, proof dependency analysis, and research note generation.

## Planned Improvements

* Add automatic LLM answer generation
* Add page-level citations
* Improve chunking for theorem and proof environments
* Support multiple papers in one index
* Add arXiv metadata search
* Add BibTeX generation
* Add a simple web interface
* Add evaluation examples for mathematical question answering
* Replace JSON storage with a vector database such as FAISS or Chroma

## Notes

This project does not upload PDFs, extracted text, embeddings, generated chunks, or private keys to GitHub by default. These files are excluded through `.gitignore`.
