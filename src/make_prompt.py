from pathlib import Path

from search_chunks import search


def build_prompt(query: str, top_k: int = 5) -> str:
    results = search(query, top_k=top_k)

    context_parts = []

    for i, result in enumerate(results, start=1):
        context_parts.append(
            f"""
[Chunk {i}]
ID: {result["id"]}
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


if __name__ == "__main__":
    query = input("Ask a question about the paper: ")

    prompt = build_prompt(query, top_k=5)

    out_path = Path("data/papers/answer_prompt.md")
    out_path.write_text(prompt, encoding="utf-8")

    print("\nPrompt generated successfully.")
    print(f"Saved to: {out_path}")
    print("\nYou can now open this file and paste its content into ChatGPT.")