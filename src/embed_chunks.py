import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(chunks_dir: str) -> list[dict]:
    """
    Load all chunk_*.txt files from a directory.

    Parameters
    ----------
    chunks_dir:
        Directory containing chunk text files.

    Returns
    -------
    list[dict]
        A list of dictionaries, each containing chunk id, path, and text.
    """
    chunks = []

    for path in sorted(Path(chunks_dir).glob("chunk_*.txt")):
        text = path.read_text(encoding="utf-8")
        chunks.append(
            {
                "id": path.stem,
                "path": str(path),
                "text": text,
            }
        )

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute local embeddings for text chunks."
    )

    parser.add_argument(
        "chunks_dir",
        help="Directory containing chunk_*.txt files.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Path to the output JSON file. If omitted, save next to the chunks directory.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Sentence-transformers embedding model name.",
    )

    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir)

    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")

    chunks = load_chunks(str(chunks_dir))

    if len(chunks) == 0:
        raise ValueError(f"No chunk_*.txt files found in {chunks_dir}")

    print(f"Loaded {len(chunks)} chunks.")
    print(f"Loading embedding model: {args.model}")

    model = SentenceTransformer(args.model)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()

    if args.output is None:
        out_path = chunks_dir.parent / "chunks_with_embeddings.json"
    else:
        out_path = Path(args.output)

    out_path.write_text(
        json.dumps(chunks, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved embeddings to {out_path}")
    print(f"Embedding dimension: {len(chunks[0]['embedding'])}")


if __name__ == "__main__":
    main()