from pathlib import Path
import json
import re

INPUT_PATH = Path("book_texts/B02_clean_candidate.txt")
OUTPUT_PATH = Path("book_texts/B02_segments.json")

BOOK_ID = "B02"
TITLE = "Efsuncu Baba"

WINDOW_SIZE = 300
OVERLAP = 50   # safer for dialogue/context


text = INPUT_PATH.read_text(encoding="utf-8")

# Remove page markers, but keep the narrative text
text = re.sub(r"\[PAGE \d+\]", "\n", text)

# Normalize dialogue dash spacing: —Yenge -> — Yenge
text = re.sub(r"—\s*", "— ", text)

# Normalize spaces
text = re.sub(r"[ \t]+", " ", text)
text = re.sub(r"\n+", "\n", text)

words = text.split()

segments = []
step = WINDOW_SIZE - OVERLAP

for i in range(0, len(words), step):
    segment_words = words[i:i + WINDOW_SIZE]

    if len(segment_words) < 30:
        continue

    segment_text = " ".join(segment_words)

    segments.append({
        "segment_id": f"{BOOK_ID}_S{len(segments) + 1:04d}",
        "start_word": i,
        "end_word": i + len(segment_words),
        "text": segment_text
    })

data = {
    "book_id": BOOK_ID,
    "title": TITLE,
    "window_size_words": WINDOW_SIZE,
    "overlap_words": OVERLAP,
    "segments": segments
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH.write_text(
    json.dumps(data, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print(f"Saved {len(segments)} segments to {OUTPUT_PATH}")