\# MathPaper-Agent



MathPaper-Agent is a small retrieval-augmented generation prototype for mathematical papers.



The goal of this project is to help users ask questions about a PDF paper by retrieving relevant excerpts from the paper and generating a prompt that can be used with a large language model.



This is currently a minimal local RAG system.



\## Features



Current version: `v0.1`



\* Extract text from a PDF paper

\* Split the extracted text into overlapping chunks

\* Compute local embeddings using `sentence-transformers`

\* Retrieve relevant chunks for a user question

\* Generate a prompt containing the question and relevant paper excerpts

\* Use the generated prompt with ChatGPT or another LLM



\## Project Structure



```text

mathpaper-agent/

├── data/

│   └── papers/

│       ├── test.pdf

│       ├── test.txt

│       ├── chunks/

│       ├── chunks\_with\_embeddings.json

│       └── answer\_prompt.md

├── src/

│   ├── read\_pdf.py

│   ├── chunk\_text.py

│   ├── embed\_chunks.py

│   ├── search\_chunks.py

│   ├── make\_prompt.py

│   └── run\_rag.py

├── .gitignore

├── README.md

└── requirements.txt

```



\## Installation



Create a virtual environment:



```bash

python -m venv .venv

```



Activate it on Windows CMD:



```bat

.venv\\Scripts\\activate.bat

```



Install dependencies:



```bash

pip install -r requirements.txt

```



\## Usage



Put a PDF paper at:



```text

data/papers/test.pdf

```



Extract text from the PDF:



```bash

python src/read\_pdf.py

```



Split the extracted text into chunks:



```bash

python src/chunk\_text.py

```



Compute local embeddings:



```bash

python src/embed\_chunks.py

```



Run the RAG prompt generator:



```bash

python src/run\_rag.py

```



Then enter a question, for example:



```text

What is the main theorem of this paper?

```



The program will generate:



```text

data/papers/answer\_prompt.md

```



Copy the content of this file into ChatGPT or another LLM to get an answer grounded in the retrieved paper excerpts.



\## Current Pipeline



```text

PDF

→ text extraction

→ chunking

→ local embedding

→ semantic retrieval

→ prompt generation

→ LLM answer

```



\## Motivation



Mathematical papers are often long, technical, and highly structured. This project explores how retrieval-based LLM tools can help with mathematical literature reading, theorem lookup, proof dependency analysis, and research note generation.



\## Planned Improvements



\* Add automatic LLM answer generation

\* Add page-level citations

\* Improve chunking for theorem and proof environments

\* Support multiple papers

\* Add arXiv metadata search

\* Add BibTeX generation

\* Add a simple web interface

\* Add evaluation examples for mathematical question answering



\## Notes



This project does not upload the PDF or embeddings to GitHub by default. The `.gitignore` file should exclude local data files, virtual environments, and private keys.



