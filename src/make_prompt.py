import argparse
from pathlib import Path

from search_chunks import search, DEFAULT_EMBEDDING_MODEL


def build_prompt(
    query: str,
    embeddings_path: str,
    top_k: int = 5,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> str:
    """
    Build a RAG prompt using the top-k retrieved paper chunks.

    Parameters
    ----------
    query:
        User question.
    embeddings_path:
        Path to the chunks_with_embeddings.json file.
    top_k:
        Number of relevant chunks to retrieve.
    model_name:
        Sentence-transformers model used to embed the query.

    Returns
    -------
    str
        A prompt that can be pasted into ChatGPT or another LLM.
    """
    results = search(
        query=query,
        embeddings_path=embeddings_path,
        top_k=top_k,
        model_name=model_name,
    )

    context_parts = []

    for i, result in enumerate(results, start=1):
        context_parts.append(
            f"""
[Chunk {i}]
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
3. When possible, refer to the chunk IDs.
4. Give a mathematically precise answer.
5. If the question asks about a theorem, lemma, proof, or assumption, identify the exact statement or condition appearing in the excerpts.

User question:
{query}

Paper excerpts:
{context}

Answer:
"""

    return prompt.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a RAG prompt from retrieved paper chunks."
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
        "--output",
        default="data/papers/answer_prompt.md",
        help="Path to save the generated prompt.",
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

    prompt = build_prompt(
        query=query,
        embeddings_path=args.embeddings_path,
        top_k=args.top_k,
        model_name=args.model,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(prompt, encoding="utf-8")

    print("\nPrompt generated successfully.")
    print(f"Saved to: {out_path}")
    print("\nYou can now open this file and paste its content into ChatGPT.")


if __name__ == "__main__":
    main()