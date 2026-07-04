import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(chunks_dir: str) -> list[dict]:
    chunks = []

    for path in sorted(Path(chunks_dir).glob("chunk_*.txt")):
        text = path.read_text(encoding="utf-8")
        chunks.append({
            "id": path.stem,
            "path": str(path),
            "text": text,
        })

    return chunks


if __name__ == "__main__":
    chunks_dir = "data/papers/chunks"
    chunks = load_chunks(chunks_dir)

    print(f"Loaded {len(chunks)} chunks.")
    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()

    out_path = Path("data/papers/chunks_with_embeddings.json")
    out_path.write_text(
        json.dumps(chunks, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved embeddings to {out_path}")
    print(f"Embedding dimension: {len(chunks[0]['embedding'])}")