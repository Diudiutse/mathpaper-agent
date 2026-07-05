import argparse
from pathlib import Path


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    """
    Split a long text into overlapping chunks.

    Parameters
    ----------
    text:
        The input long text.
    chunk_size:
        The maximum number of characters in each chunk.
    overlap:
        The number of overlapping characters between neighboring chunks.

    Returns
    -------
    list[str]
        A list of text chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0:
        raise ValueError("overlap must be non-negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a text file into overlapping chunks."
    )

    parser.add_argument(
        "input_txt",
        help="Path to the input txt file.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save chunk files. If omitted, use a folder named chunks next to the input file.",
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

    args = parser.parse_args()

    input_path = Path(args.input_txt)

    if not input_path.exists():
        raise FileNotFoundError(f"Input txt file not found: {input_path}")

    text = input_path.read_text(encoding="utf-8")

    chunks = chunk_text(
        text,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    if args.output_dir is None:
        out_dir = input_path.parent / "chunks"
    else:
        out_dir = Path(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    for i, chunk in enumerate(chunks):
        out_path = out_dir / f"chunk_{i:04d}.txt"
        out_path.write_text(chunk, encoding="utf-8")

    print(f"Input file: {input_path}")
    print(f"Created {len(chunks)} chunks.")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Overlap: {args.overlap}")
    print(f"Saved chunks to {out_dir}")


if __name__ == "__main__":
    main()