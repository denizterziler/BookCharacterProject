import json
import re
import itertools
from pathlib import Path
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BOOK_ID = "B05"

SEGMENTS_PATH = PROJECT_ROOT / "scripts" / "book_texts" / f"{BOOK_ID}" /f"{BOOK_ID}_segments.json"
CHARACTERS_PATH = PROJECT_ROOT / "inputs" / f"{BOOK_ID}_characters.json"
OUTPUT_PATH = PROJECT_ROOT / "predictions" / "name_only" / f"{BOOK_ID}" / f"{BOOK_ID}_window_presence_predictions.json"


def normalize_text(text: str) -> str:
    """
    Basic normalization for Turkish text matching.
    Keeps Turkish characters, lowercases text, and normalizes apostrophes.
    """
    text = text.lower()
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    return text


def load_segments(segments_path: Path) -> list[dict]:
    """
    Reads segment JSON created from 300-word windows.

    Expected format:
    {
      "book_id": "B02",
      "title": "Efsuncu Baba",
      "window_size_words": 300,
      "overlap_words": 50,
      "segments": [
        {
          "segment_id": "B02_S0001",
          "start_word": 0,
          "end_word": 300,
          "text": "..."
        }
      ]
    }
    """
    print("Reading segments from:", segments_path)
    print("File exists:", segments_path.exists())

    if not segments_path.exists():
        raise FileNotFoundError(f"Segments file not found: {segments_path}")

    data = json.loads(segments_path.read_text(encoding="utf-8"))

    if "segments" not in data:
        raise ValueError("Segments JSON must contain a 'segments' field.")

    print(f"Segments loaded: {len(data['segments'])}")

    return data["segments"]


def load_character_aliases(characters_path: Path) -> dict:
    """
    Reads character aliases from character inventory JSON.

    Expected format:
    {
      "book_id": "B02",
      "title": "Efsuncu Baba",
      "characters": [
        {
          "char_id": "B02_C001",
          "canonical_name": "Agop",
          "aliases": ["Agop"]
        }
      ]
    }
    """
    print("Reading character inventory from:", characters_path)
    print("File exists:", characters_path.exists())

    if not characters_path.exists():
        raise FileNotFoundError(f"Character inventory not found: {characters_path}")

    data = json.loads(characters_path.read_text(encoding="utf-8"))

    character_data = {}

    for character in data["characters"]:
        char_id = character["char_id"]
        canonical_name = character["canonical_name"]
        aliases = character.get("aliases", [])

        all_aliases = list(dict.fromkeys([canonical_name] + aliases))

        character_data[char_id] = {
            "canonical_name": canonical_name,
            "aliases": all_aliases
        }

    print(f"Characters loaded: {len(character_data)}")

    return character_data


def alias_to_regex(alias: str) -> re.Pattern:
    """
    Creates a regex pattern for an alias.

    It matches full words/phrases and allows Turkish suffixes after apostrophes,
    e.g. Agop'un, Enverî'nin.
    """
    alias_norm = normalize_text(alias)
    escaped = re.escape(alias_norm)

    pattern = rf"(?<!\w){escaped}(?:'[a-zçğıöşü]+)?(?!\w)"

    return re.compile(pattern, flags=re.IGNORECASE)


def detect_characters_in_segment(segment_text: str, character_data: dict) -> tuple[set, dict]:
    """
    Detects characters in one 300-word segment using exact alias matching.

    Returns:
        detected_characters: set of char_ids
        matched_aliases: dict of char_id -> matched aliases
    """
    segment_norm = normalize_text(segment_text)

    detected_characters = set()
    matched_aliases = defaultdict(list)

    for char_id, data in character_data.items():
        for alias in data["aliases"]:
            pattern = alias_to_regex(alias)

            if pattern.search(segment_norm):
                detected_characters.add(char_id)
                matched_aliases[char_id].append(alias)

    return detected_characters, dict(matched_aliases)


def build_cooccurrence_edges(segment_predictions: list[dict]) -> list[dict]:
    """
    Creates undirected co-occurrence edges.
    If two characters are detected in the same 300-word segment,
    edge weight increases by 1.
    """
    edge_weights = defaultdict(int)
    evidence_segments = defaultdict(list)

    for item in segment_predictions:
        segment_id = item["segment_id"]
        chars = sorted(item["characters_present"])

        for c1, c2 in itertools.combinations(chars, 2):
            edge_key = tuple(sorted([c1, c2]))
            edge_weights[edge_key] += 1
            evidence_segments[edge_key].append(segment_id)

    edges = []

    for (source, target), weight in sorted(edge_weights.items()):
        edges.append({
            "source": source,
            "target": target,
            "weight": weight,
            "evidence_segments": evidence_segments[(source, target)]
        })

    return edges


def run_name_only_window_baseline():
    segments = load_segments(SEGMENTS_PATH)
    character_data = load_character_aliases(CHARACTERS_PATH)

    segment_predictions = []

    for segment in segments:
        segment_id = segment["segment_id"]
        segment_text = segment["text"]

        detected_chars, matched_aliases = detect_characters_in_segment(
            segment_text,
            character_data
        )

        segment_predictions.append({
            "segment_id": segment_id,
            "characters_present": sorted(detected_chars),
            "matched_aliases": matched_aliases
        })

    edges = build_cooccurrence_edges(segment_predictions)

    output = {
        "book_id": BOOK_ID,
        "system": "name_only_baseline",
        "description": "Dictionary/alias-based baseline using exact alias matching over 300-word overlapping windows.",
        "unit": "300-word overlapping segment",
        "segment_presence": segment_predictions,
        "edges": edges
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Saved output to: {OUTPUT_PATH}")
    print(f"Segments processed: {len(segment_predictions)}")
    print(f"Edges created: {len(edges)}")


if __name__ == "__main__":
    run_name_only_window_baseline()