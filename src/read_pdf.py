from pathlib import Path
from pypdf import PdfReader


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        texts.append(f"\n\n[Page {i + 1}]\n{text}")

    return "\n".join(texts)


if __name__ == "__main__":
    pdf_path = "data/papers/test.pdf"
    text = read_pdf(pdf_path)

    out_path = Path("data/papers/test.txt")
    out_path.write_text(text, encoding="utf-8")

    print(f"Extracted {len(text)} characters.")
    print(f"Saved to {out_path}")