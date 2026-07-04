import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def search(query: str, top_k: int = 5) -> list[dict]:
    data_path = Path("data/papers/chunks_with_embeddings.json")
    chunks = json.loads(data_path.read_text(encoding="utf-8"))

    model = SentenceTransformer(EMBEDDING_MODEL)

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    results = []

    for chunk in chunks:
        chunk_embedding = np.array(chunk["embedding"])

        # 因为我们已经 normalize_embeddings=True，
        # 所以点积就是 cosine similarity。
        score = float(np.dot(query_embedding, chunk_embedding))

        results.append({
            "id": chunk["id"],
            "path": chunk["path"],
            "score": score,
            "text": chunk["text"],
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    query = input("Ask a question about the paper: ")

    results = search(query, top_k=5)

    print("\nTop relevant chunks:\n")

    for i, result in enumerate(results, start=1):
        print("=" * 80)
        print(f"Rank {i}")
        print(f"Chunk: {result['id']}")
        print(f"Score: {result['score']:.4f}")
        print("-" * 80)
        print(result["text"][:1500])
        print()