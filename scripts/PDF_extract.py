import pymupdf
from pathlib import Path
import re


PDF_PATH = Path("/Users/denizterziler/PycharmProjects/BookCharacterProject/bookpdfs_phase1/Cadı-WEB-1.pdf")
OUTPUT_PATH = Path("book_texts/B01_clean_candidate.txt")


TURKISH_LETTERS = "A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû"
LOWER_TURKISH = "a-zçğıöşüâîû"


def clean_pdf_artifacts(text: str) -> str:
    """
    Cleans common PDF extraction artifacts.
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove soft hyphen artifacts
    text = text.replace("[SHY]", "")
    text = text.replace("\u00ad", "")

    # Remove invisible Unicode artifacts
    text = text.replace("\ufeff", "")  # BOM
    text = text.replace("\u200b", "")  # zero-width space
    text = text.replace("\u200c", "")  # zero-width non-joiner
    text = text.replace("\u200d", "")  # zero-width joiner

    # Fix visible line-break hyphenation:
    # sevinç-\nle -> sevinçle
    # karak-\nter -> karakter
    #
    # This is safer than merging all line breaks.
    text = re.sub(
        rf"([{TURKISH_LETTERS}])-\s*\n\s*([{LOWER_TURKISH}])",
        r"\1\2",
        text
    )

    return text


def remove_inline_footnote_markers(text: str) -> str:
    """
    Removes inline footnote markers such as:
    nıza125 -> nıza
    vazedebileceğim?126 -> vazedebileceğim?
    sâire128 -> sâire

    It does not remove standalone numbers like 1908.
    """

    # Remove numbers immediately after letters
    text = re.sub(
        rf"(?<=[{TURKISH_LETTERS}])\d{{1,4}}",
        "",
        text
    )

    # Remove numbers immediately after punctuation
    text = re.sub(
        r"(?<=[\.\,\;\:\!\?\”\’\'\"\)])\d{1,4}",
        "",
        text
    )

    return text


def clean_page_text(text: str) -> str:
    text = clean_pdf_artifacts(text)
    text = remove_inline_footnote_markers(text)

    # Remove excessive spaces/tabs, but keep line breaks
    text = re.sub(r"[ \t]+", " ", text)

    lines = []

    for line in text.split("\n"):
        stripped = line.strip()

        if not stripped:
            lines.append("")
            continue

        # Remove lines that are only page numbers
        if re.fullmatch(r"\d+", stripped):
            continue

        # Remove Roman numeral page numbers
        # Example: I, II, III, IV, V, VI...
        if re.fullmatch(r"[IVXLCDMivxlcdm]+", stripped):
            continue

        # Remove page headers like:
        # 40 | Gulyabani
        if re.fullmatch(r"\d+\s*\|\s*.+", stripped):
            continue

        # Remove reversed page headers like:
        # Gulyabani | 40
        if re.fullmatch(r".+\s*\|\s*\d+", stripped):
            continue

        # Remove footnote/glossary lines like:
        # 125 nazargâh-ı istiğrap: şaşkın bakış
        #
        # Safer rule: only removes lines starting with 2+ digit numbers.
        if re.match(r"^\d{2,}\s+.+", stripped):
            continue

        lines.append(stripped)

    text = "\n".join(lines)

    # Collapse too many blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF file not found: {PDF_PATH}")

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


if __name__ == "__main__":
    main()