import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer


# Make sure Python can import files from src/
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from read_pdf import read_pdf
from chunk_text import chunk_text
from embed_chunks import load_chunks, DEFAULT_EMBEDDING_MODEL


UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"


@st.cache_resource
def get_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Load and cache the embedding model.

    Streamlit reruns the script after interactions.
    Without caching, the model may be loaded repeatedly.
    """
    return SentenceTransformer(model_name)


def save_uploaded_pdf(uploaded_file) -> Path:
    """
    Save the uploaded PDF file to data/uploads/.

    A short hash is added to the filename so that different PDFs
    with the same original filename do not overwrite each other.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()[:12]

    original_path = Path(uploaded_file.name)
    safe_stem = original_path.stem.replace(" ", "_")
    suffix = original_path.suffix.lower()

    pdf_path = UPLOAD_DIR / f"{safe_stem}_{file_hash}{suffix}"
    pdf_path.write_bytes(file_bytes)

    return pdf_path


def save_chunks(text_path: Path, chunks_dir: Path, chunk_size: int, overlap: int) -> int:
    """
    Read a text file, split it into chunks, and save chunk files.
    """
    text = text_path.read_text(encoding="utf-8")

    chunks = chunk_text(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    chunks_dir.mkdir(parents=True, exist_ok=True)

    for i, chunk in enumerate(chunks):
        out_path = chunks_dir / f"chunk_{i:04d}.txt"
        out_path.write_text(chunk, encoding="utf-8")

    return len(chunks)


def save_embeddings(
    chunks_dir: Path,
    embeddings_path: Path,
    model: SentenceTransformer,
    source_pdf_name: str,
) -> int:
    """
    Load chunks, compute embeddings, and save them to a JSON file.

    Each chunk also stores the source PDF name, so that search results
    can tell us which paper the chunk comes from.
    """
    chunks = load_chunks(str(chunks_dir))

    if len(chunks) == 0:
        raise ValueError(f"No chunk_*.txt files found in {chunks_dir}")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()
        chunk["source_pdf"] = source_pdf_name

    embeddings_path.write_text(
        json.dumps(chunks, ensure_ascii=False),
        encoding="utf-8",
    )

    return len(chunks)


def build_document_index(
    pdf_path: Path,
    chunk_size: int,
    overlap: int,
    model: SentenceTransformer,
    model_name: str,
    force_rebuild: bool = False,
) -> dict:
    """
    Build or reuse the local index for one PDF document.

    Cached files are reused when possible:
    - extracted text
    - chunks
    - embeddings
    """
    paper_dir = pdf_path.parent
    paper_stem = pdf_path.stem

    model_key = (
        model_name.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )

    text_path = paper_dir / f"{paper_stem}.txt"
    chunks_dir = paper_dir / f"{paper_stem}_chunks_cs{chunk_size}_ov{overlap}"
    embeddings_path = paper_dir / (
        f"{paper_stem}_embeddings_cs{chunk_size}_ov{overlap}_{model_key}.json"
    )

    used_text_cache = False
    used_chunks_cache = False
    used_embeddings_cache = False

    # Step 1: extract text, or reuse cached text
    if text_path.exists() and not force_rebuild:
        text = text_path.read_text(encoding="utf-8")
        used_text_cache = True
    else:
        text = read_pdf(str(pdf_path))
        text_path.write_text(text, encoding="utf-8")

    # Step 2: create chunks, or reuse cached chunks
    existing_chunks = list(chunks_dir.glob("chunk_*.txt"))

    if existing_chunks and not force_rebuild:
        num_chunks = len(existing_chunks)
        used_chunks_cache = True
    else:
        num_chunks = save_chunks(
            text_path=text_path,
            chunks_dir=chunks_dir,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    # Step 3: compute embeddings, or reuse cached embeddings
    if embeddings_path.exists() and not force_rebuild:
        used_embeddings_cache = True
    else:
        save_embeddings(
            chunks_dir=chunks_dir,
            embeddings_path=embeddings_path,
            model=model,
            source_pdf_name=pdf_path.name,
        )

    return {
        "pdf_path": pdf_path,
        "text_path": text_path,
        "chunks_dir": chunks_dir,
        "embeddings_path": embeddings_path,
        "num_characters": len(text),
        "num_chunks": num_chunks,
        "used_text_cache": used_text_cache,
        "used_chunks_cache": used_chunks_cache,
        "used_embeddings_cache": used_embeddings_cache,
    }


def load_all_chunks(document_infos: list[dict]) -> list[dict]:
    """
    Load chunks from multiple document embedding files.
    """
    all_chunks = []

    for info in document_infos:
        embeddings_path = info["embeddings_path"]
        chunks = json.loads(embeddings_path.read_text(encoding="utf-8"))

        for chunk in chunks:
            if "source_pdf" not in chunk:
                chunk["source_pdf"] = info["pdf_path"].name

            all_chunks.append(chunk)

    return all_chunks


def search_across_documents(
    query: str,
    chunks: list[dict],
    model: SentenceTransformer,
    top_k: int = 5,
) -> list[dict]:
    """
    Search relevant chunks across multiple documents.
    """
    if len(chunks) == 0:
        raise ValueError("No chunks available for search.")

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    results = []

    for chunk in chunks:
        chunk_embedding = np.array(chunk["embedding"])

        # Since embeddings are normalized, dot product equals cosine similarity.
        score = float(np.dot(query_embedding, chunk_embedding))

        results.append(
            {
                "id": chunk["id"],
                "path": chunk["path"],
                "source_pdf": chunk.get("source_pdf", "unknown"),
                "score": score,
                "text": chunk["text"],
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:top_k]


def build_prompt_from_results(query: str, results: list[dict]) -> str:
    """
    Build a RAG prompt from retrieved chunks.
    """
    context_parts = []

    for i, result in enumerate(results, start=1):
        context_parts.append(
            f"""
[Chunk {i}]
Source PDF: {result["source_pdf"]}
ID: {result["id"]}
Path: {result["path"]}
Score: {result["score"]:.4f}

{result["text"]}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a careful mathematical research assistant.

Answer the user's question using only the provided paper excerpts.

Rules:
1. Do not invent facts not supported by the excerpts.
2. If the excerpts are insufficient, say so clearly.
3. When possible, refer to the source PDF and chunk IDs.
4. Give a mathematically precise answer.
5. If the question asks about a theorem, lemma, proof, or assumption, identify the exact statement or condition appearing in the excerpts.
6. If multiple papers are relevant, compare them clearly.

User question:
{query}

Paper excerpts:
{context}

Answer:
"""

    return prompt.strip()


def run_multi_paper_pipeline(
    pdf_paths: list[Path],
    query: str,
    chunk_size: int,
    overlap: int,
    top_k: int,
    model_name: str,
    force_rebuild: bool = False,
) -> tuple[str, dict]:
    """
    Run the local RAG pipeline over multiple PDF papers.
    """
    model = get_embedding_model(model_name)

    document_infos = []

    for pdf_path in pdf_paths:
        info = build_document_index(
            pdf_path=pdf_path,
            chunk_size=chunk_size,
            overlap=overlap,
            model=model,
            model_name=model_name,
            force_rebuild=force_rebuild,
        )
        document_infos.append(info)

    all_chunks = load_all_chunks(document_infos)

    results = search_across_documents(
        query=query,
        chunks=all_chunks,
        model=model,
        top_k=top_k,
    )

    prompt = build_prompt_from_results(
        query=query,
        results=results,
    )

    prompt_path = UPLOAD_DIR / "multi_paper_answer_prompt.md"
    prompt_path.write_text(prompt, encoding="utf-8")

    info = {
        "document_infos": document_infos,
        "num_documents": len(document_infos),
        "num_total_chunks": len(all_chunks),
        "retrieved_results": results,
        "prompt_path": prompt_path,
    }

    return prompt, info


st.set_page_config(
    page_title="MathPaper-Agent",
    page_icon="📄",
    layout="wide",
)

st.title("MathPaper-Agent")
st.write(
    "A local RAG prototype for mathematical papers. "
    "Upload one or more PDFs, ask a question, and generate a prompt grounded in retrieved paper excerpts."
)

uploaded_files = st.file_uploader(
    "Upload PDF papers",
    type=["pdf"],
    accept_multiple_files=True,
)

query = st.text_input(
    "Question",
    placeholder="Which paper discusses Brouwer's conjecture for split graphs?",
)

with st.expander("Advanced settings"):
    chunk_size = st.number_input(
        "Chunk size",
        min_value=200,
        max_value=5000,
        value=1200,
        step=100,
    )

    overlap = st.number_input(
        "Overlap",
        min_value=0,
        max_value=2000,
        value=200,
        step=50,
    )

    top_k = st.number_input(
        "Top-k retrieved chunks",
        min_value=1,
        max_value=30,
        value=8,
        step=1,
    )

    model_name = st.text_input(
        "Embedding model",
        value=DEFAULT_EMBEDDING_MODEL,
    )

    force_rebuild = st.checkbox(
        "Force rebuild cache",
        value=False,
        help="If checked, the app will recompute text, chunks, and embeddings.",
    )

run_button = st.button("Generate RAG Prompt")

if run_button:
    if not uploaded_files:
        st.error("Please upload at least one PDF file.")
    elif not query.strip():
        st.error("Please enter a question.")
    elif overlap >= chunk_size:
        st.error("Overlap must be smaller than chunk size.")
    else:
        with st.spinner("Running local multi-paper RAG pipeline..."):
            try:
                pdf_paths = [save_uploaded_pdf(file) for file in uploaded_files]

                prompt, info = run_multi_paper_pipeline(
                    pdf_paths=pdf_paths,
                    query=query,
                    chunk_size=int(chunk_size),
                    overlap=int(overlap),
                    top_k=int(top_k),
                    model_name=model_name,
                    force_rebuild=force_rebuild,
                )

                st.success("Prompt generated successfully.")

                st.subheader("Pipeline summary")
                st.write(f"Number of documents: {info['num_documents']}")
                st.write(f"Total number of chunks: {info['num_total_chunks']}")
                st.write(f"Prompt path: `{info['prompt_path']}`")

                st.subheader("Document cache summary")
                for doc_info in info["document_infos"]:
                    with st.expander(f"{doc_info['pdf_path'].name}"):
                        st.write(f"Extracted characters: {doc_info['num_characters']}")
                        st.write(f"Number of chunks: {doc_info['num_chunks']}")
                        st.write(f"Text cache: {doc_info['used_text_cache']}")
                        st.write(f"Chunks cache: {doc_info['used_chunks_cache']}")
                        st.write(f"Embeddings cache: {doc_info['used_embeddings_cache']}")
                        st.write(f"Embeddings path: `{doc_info['embeddings_path']}`")

                st.subheader("Retrieved chunks")
                for i, result in enumerate(info["retrieved_results"], start=1):
                    with st.expander(
                        f"Rank {i}: {result['source_pdf']} | {result['id']} | Score: {result['score']:.4f}"
                    ):
                        st.write(f"Path: `{result['path']}`")
                        st.text(result["text"][:2500])

                st.subheader("Generated prompt")
                st.text_area(
                    "Copy this prompt into ChatGPT or another LLM",
                    value=prompt,
                    height=500,
                )

                st.download_button(
                    label="Download prompt",
                    data=prompt,
                    file_name="multi_paper_answer_prompt.md",
                    mime="text/markdown",
                )

            except Exception as e:
                st.error(f"Error: {e}")