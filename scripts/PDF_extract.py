import pymupdf
from pathlib import Path
import re

PDF_PATH = Path("/Users/denizterziler/Downloads/Gulyabani-WEB-1.pdf")
OUTPUT_PATH = Path("book_texts/B01_clean_candidate.txt")

def clean_page_text(text: str) -> str:
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove lines that are only page numbers
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()

        if not stripped:
            lines.append("")
            continue

        if stripped.isdigit():
            continue

        lines.append(stripped)

    text = "\n".join(lines)

    # Collapse too many blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


doc = pymupdf.open(PDF_PATH)
pages = []

for page_num, page in enumerate(doc, start=1):
    text = page.get_text("text")
    text = clean_page_text(text)

    if text:
        pages.append(f"\n\n[PAGE {page_num}]\n{text}")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text("\n".join(pages).strip(), encoding="utf-8")

print(f"Saved to {OUTPUT_PATH}")