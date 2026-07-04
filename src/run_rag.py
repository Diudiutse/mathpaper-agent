import subprocess
from pathlib import Path

from make_prompt import build_prompt


def open_file_with_notepad(path: Path) -> None:
    subprocess.run(["notepad", str(path)])


if __name__ == "__main__":
    print("MathPaper-Agent v0.1")
    print("=" * 60)

    query = input("Ask a question about the paper: ")

    prompt = build_prompt(query, top_k=5)

    out_path = Path("data/papers/answer_prompt.md")
    out_path.write_text(prompt, encoding="utf-8")

    print()
    print("Prompt generated successfully.")
    print(f"Saved to: {out_path}")
    print()
    print("Opening the prompt file now...")

    open_file_with_notepad(out_path)