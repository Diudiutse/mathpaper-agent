import argparse
from pathlib import Path

from pypdf import PdfReader


def read_pdf(path: str) -> str:
    """
    Read a PDF file and extract all text.

    Parameters
    ----------
    path:
        Path to the input PDF file.

    Returns
    -------
    str
        Extracted text, with page markers.
    """
    reader = PdfReader(path)
    texts = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        texts.append(f"\n\n[Page {i + 1}]\n{text}")

    return "\n".join(texts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text from a PDF paper."
    )

    parser.add_argument(
        "pdf_path",
        help="Path to the input PDF file.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Path to the output txt file. If omitted, use the same name as the PDF.",
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    text = read_pdf(str(pdf_path))

    if args.output is None:
        out_path = pdf_path.with_suffix(".txt")
    else:
        out_path = Path(args.output)

    out_path.write_text(text, encoding="utf-8")

    print(f"Extracted {len(text)} characters.")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()