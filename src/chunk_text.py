from pathlib import Path


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


if __name__ == "__main__":
    input_path = Path("data/papers/test.txt")
    text = input_path.read_text(encoding="utf-8")

    chunks = chunk_text(text)

    out_dir = Path("data/papers/chunks")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, chunk in enumerate(chunks):
        out_path = out_dir / f"chunk_{i:04d}.txt"
        out_path.write_text(chunk, encoding="utf-8")

    print(f"Created {len(chunks)} chunks.")
    print(f"Saved chunks to {out_dir}")