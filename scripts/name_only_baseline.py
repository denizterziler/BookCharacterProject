import json
import re
import itertools
from pathlib import Path
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STORY_PATH = PROJECT_ROOT / "corpus" / "T01_story.txt"
GOLD_PATH = PROJECT_ROOT / "gold" / "T01_gold.json"
OUTPUT_PATH = PROJECT_ROOT / "predictions" / "name_only" / "T01_output.json"


def normalize_text(text: str) -> str:
    """
    Basic normalization for Turkish text matching.
    Keeps Turkish characters, lowercases text, and normalizes apostrophes.
    """
    text = text.lower()
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    return text


def load_story_paragraphs(story_path: Path) -> dict:
    """
    Reads story file with paragraph IDs like:

    [P001]
    paragraph text...

    [P002]
    paragraph text...

    Returns:
        {
          "P001": "paragraph text",
          "P002": "paragraph text"
        }
    """
    print("Reading story from:", story_path)
    print("File exists:", story_path.exists())

    if not story_path.exists():
        raise FileNotFoundError(f"Story file not found: {story_path}")

    text = story_path.read_text(encoding="utf-8")

    print("First 150 characters of story file:")
    print(repr(text[:150]))

    paragraphs = {}
    current_id = None
    current_lines = []

    paragraph_id_pattern = re.compile(r"^\[(P\d{3})\]\s*$")

    for line in text.splitlines():
        stripped = line.strip()

        match = paragraph_id_pattern.match(stripped)

        if match:
            # Save previous paragraph before starting a new one
            if current_id is not None:
                paragraphs[current_id] = "\n".join(current_lines).strip()

            current_id = match.group(1)
            current_lines = []
        else:
            if current_id is not None:
                current_lines.append(line)

    # Save the final paragraph
    if current_id is not None:
        paragraphs[current_id] = "\n".join(current_lines).strip()

    if not paragraphs:
        raise ValueError(
            "No paragraphs found. Make sure your story uses IDs like [P001], [P002], ..."
        )

    print("Paragraph IDs found:", list(paragraphs.keys()))

    return paragraphs

def load_character_aliases(gold_path: Path) -> dict:
    """
    Reads character aliases from gold JSON.

    Returns:
        {
          "C01": {
            "canonical_name": "Kurt",
            "aliases": ["kurt", "canavar"]
          }
        }
    """
    print("Reading gold annotation from:", gold_path)
    print("File exists:", gold_path.exists())

    if not gold_path.exists():
        raise FileNotFoundError(f"Gold JSON file not found: {gold_path}")

    gold = json.loads(gold_path.read_text(encoding="utf-8"))

    character_data = {}

    for character in gold["characters"]:
        char_id = character["char_id"]
        canonical_name = character["canonical_name"]
        aliases = character.get("aliases", [])

        # Include canonical name as an alias too.
        all_aliases = list(dict.fromkeys([canonical_name] + aliases))

        character_data[char_id] = {
            "canonical_name": canonical_name,
            "aliases": all_aliases
        }

    return character_data


def alias_to_regex(alias: str) -> re.Pattern:
    """
    Creates a regex pattern for an alias.

    This tries to match full words/phrases and allows Turkish suffixes
    after apostrophes, e.g. Kurt'un, Keloğlan'ın.

    It also works for simple lowercase forms such as kurt, köylüler, sıpası.
    """
    alias_norm = normalize_text(alias)
    escaped = re.escape(alias_norm)

    # (?<!\w) and (?!\w) prevent matching inside larger words.
    # Optional apostrophe suffix allows forms like Kurt'un.
    pattern = rf"(?<!\w){escaped}(?:'[a-zçğıöşü]+)?(?!\w)"

    return re.compile(pattern, flags=re.IGNORECASE)


def detect_characters_in_paragraph(paragraph_text: str, character_data: dict) -> tuple[set, dict]:
    """
    Detects characters in one paragraph by alias matching.

    Returns:
        detected_characters: set of char_ids
        matched_aliases: dict of char_id -> matched aliases
    """
    paragraph_norm = normalize_text(paragraph_text)

    detected_characters = set()
    matched_aliases = defaultdict(list)

    for char_id, data in character_data.items():
        for alias in data["aliases"]:
            pattern = alias_to_regex(alias)

            if pattern.search(paragraph_norm):
                detected_characters.add(char_id)
                matched_aliases[char_id].append(alias)

    return detected_characters, dict(matched_aliases)


def build_cooccurrence_edges(paragraph_predictions: list[dict]) -> list[dict]:
    """
    Creates undirected co-occurrence edges.
    If two characters are detected in the same paragraph, edge weight increases by 1.
    """
    edge_weights = defaultdict(int)
    evidence_paragraphs = defaultdict(list)

    for item in paragraph_predictions:
        paragraph_id = item["paragraph_id"]
        chars = sorted(item["characters_present"])

        for c1, c2 in itertools.combinations(chars, 2):
            edge_key = tuple(sorted([c1, c2]))
            edge_weights[edge_key] += 1
            evidence_paragraphs[edge_key].append(paragraph_id)

    edges = []

    for (source, target), weight in sorted(edge_weights.items()):
        edges.append({
            "source": source,
            "target": target,
            "weight": weight,
            "evidence_paragraphs": evidence_paragraphs[(source, target)]
        })

    return edges


def run_name_only_baseline():
    paragraphs = load_story_paragraphs(STORY_PATH)
    character_data = load_character_aliases(GOLD_PATH)

    paragraph_predictions = []

    for paragraph_id, paragraph_text in paragraphs.items():
        detected_chars, matched_aliases = detect_characters_in_paragraph(
            paragraph_text,
            character_data
        )

        paragraph_predictions.append({
            "paragraph_id": paragraph_id,
            "characters_present": sorted(detected_chars),
            "matched_aliases": matched_aliases
        })

    edges = build_cooccurrence_edges(paragraph_predictions)

    output = {
        "text_id": "T01",
        "system": "name_only_baseline",
        "description": "Dictionary/alias-based baseline using exact alias matching over paragraph units.",
        "paragraph_presence": paragraph_predictions,
        "edges": edges
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Saved output to: {OUTPUT_PATH}")
    print(f"Paragraphs processed: {len(paragraph_predictions)}")
    print(f"Edges created: {len(edges)}")


if __name__ == "__main__":
    run_name_only_baseline()