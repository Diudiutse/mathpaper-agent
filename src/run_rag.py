import argparse
import subprocess
from pathlib import Path

from read_pdf import read_pdf
from chunk_text import chunk_text
from embed_chunks import load_chunks, DEFAULT_EMBEDDING_MODEL
from make_prompt import build_prompt

from sentence_transformers import SentenceTransformer
import json


def open_file_with_notepad(path: Path) -> None:
    subprocess.run(["notepad", str(path)])


def save_chunks(text_path: Path, chunks_dir: Path, chunk_size: int, overlap: int) -> None:
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

    print(f"Created {len(chunks)} chunks.")
    print(f"Saved chunks to {chunks_dir}")


def save_embeddings(chunks_dir: Path, embeddings_path: Path, model_name: str) -> None:
    chunks = load_chunks(str(chunks_dir))

    if len(chunks) == 0:
        raise ValueError(f"No chunk_*.txt files found in {chunks_dir}")

    print(f"Loaded {len(chunks)} chunks.")
    print(f"Loading embedding model: {model_name}")

    model = SentenceTransformer(model_name)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()

    embeddings_path.write_text(
        json.dumps(chunks, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved embeddings to {embeddings_path}")
    print(f"Embedding dimension: {len(chunks[0]['embedding'])}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full MathPaper-Agent local RAG pipeline."
    )

    parser.add_argument(
        "pdf_path",
        help="Path to the input PDF paper.",
    )

    parser.add_argument(
        "--query",
        default=None,
        help="User question. If omitted, the program will ask interactively.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Maximum number of characters in each chunk.",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Number of overlapping characters between neighboring chunks.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of relevant chunks to retrieve.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Sentence-transformers embedding model name.",
    )

    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated prompt file with Notepad.",
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    paper_dir = pdf_path.parent
    paper_stem = pdf_path.stem

    text_path = paper_dir / f"{paper_stem}.txt"
    chunks_dir = paper_dir / f"{paper_stem}_chunks"
    embeddings_path = paper_dir / f"{paper_stem}_embeddings.json"
    prompt_path = paper_dir / f"{paper_stem}_answer_prompt.md"

    print("MathPaper-Agent v0.2")
    print("=" * 60)

    print("\nStep 1: Extracting text from PDF...")
    text = read_pdf(str(pdf_path))
    text_path.write_text(text, encoding="utf-8")
    print(f"Extracted {len(text)} characters.")
    print(f"Saved text to {text_path}")

    print("\nStep 2: Splitting text into chunks...")
    save_chunks(
        text_path=text_path,
        chunks_dir=chunks_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    print("\nStep 3: Computing embeddings...")
    save_embeddings(
        chunks_dir=chunks_dir,
        embeddings_path=embeddings_path,
        model_name=args.model,
    )

    if args.query is None:
        query = input("\nAsk a question about the paper: ")
    else:
        query = args.query

    print("\nStep 4: Generating RAG prompt...")
    prompt = build_prompt(
        query=query,
        embeddings_path=str(embeddings_path),
        top_k=args.top_k,
        model_name=args.model,
    )

    prompt_path.write_text(prompt, encoding="utf-8")

    print("\nPrompt generated successfully.")
    print(f"Saved to: {prompt_path}")

    if args.open:
        print("\nOpening prompt file with Notepad...")
        open_file_with_notepad(prompt_path)


if __name__ == "__main__":
    main()