import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def search(
    query: str,
    embeddings_path: str,
    top_k: int = 5,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> list[dict]:
    """
    Search for the most relevant chunks for a user query.

    Parameters
    ----------
    query:
        User question.
    embeddings_path:
        Path to the JSON file containing chunks and their embeddings.
    top_k:
        Number of top results to return.
    model_name:
        Sentence-transformers model used to embed the query.

    Returns
    -------
    list[dict]
        Top-k relevant chunks, sorted by similarity score.
    """
    data_path = Path(embeddings_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {data_path}")

    chunks = json.loads(data_path.read_text(encoding="utf-8"))

    if len(chunks) == 0:
        raise ValueError(f"No chunks found in {data_path}")

    print(f"Loaded {len(chunks)} chunks from {data_path}")
    print(f"Loading embedding model: {model_name}")

    model = SentenceTransformer(model_name)

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    results = []

    for chunk in chunks:
        chunk_embedding = np.array(chunk["embedding"])

        # Since both query and chunk embeddings are normalized,
        # the dot product equals cosine similarity.
        score = float(np.dot(query_embedding, chunk_embedding))

        results.append(
            {
                "id": chunk["id"],
                "path": chunk["path"],
                "score": score,
                "text": chunk["text"],
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search relevant chunks for a user question."
    )

    parser.add_argument(
        "embeddings_path",
        help="Path to the chunks_with_embeddings.json file.",
    )

    parser.add_argument(
        "--query",
        default=None,
        help="User question. If omitted, the program will ask interactively.",
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

    args = parser.parse_args()

    if args.query is None:
        query = input("Ask a question about the paper: ")
    else:
        query = args.query

    results = search(
        query=query,
        embeddings_path=args.embeddings_path,
        top_k=args.top_k,
        model_name=args.model,
    )

    print("\nTop relevant chunks:\n")

    for i, result in enumerate(results, start=1):
        print("=" * 80)
        print(f"Rank {i}")
        print(f"Chunk: {result['id']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Path: {result['path']}")
        print("-" * 80)
        print(result["text"][:1500])
        print()


if __name__ == "__main__":
    main()