import json
import re
import itertools
from pathlib import Path
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STORY_PATH = PROJECT_ROOT / "corpus" / "T01_story.txt"
GOLD_PATH = PROJECT_ROOT / "gold" / "T01_gold.json"

# Change this name for each model you want to test.
MODEL_NAME = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"

# Try 1, 2, and 3 separately.
TOP_K = 2

# Output name is created from model name safely.
SAFE_MODEL_NAME = MODEL_NAME.replace("/", "__")
OUTPUT_PATH = PROJECT_ROOT / "predictions" / "sentence_transformer" / f"T01_{SAFE_MODEL_NAME}_top{TOP_K}_output.json"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_story_paragraphs(story_path: Path) -> dict:
    """
    Reads story file with paragraph IDs like:

    [P001]
    paragraph text...

    [P002]
    paragraph text...
    """
    if not story_path.exists():
        raise FileNotFoundError(f"Story file not found: {story_path}")

    text = story_path.read_text(encoding="utf-8")

    paragraphs = {}
    current_id = None
    current_lines = []

    paragraph_id_pattern = re.compile(r"^\[(P\d{3})\]\s*$")

    for line in text.splitlines():
        stripped = line.strip()
        match = paragraph_id_pattern.match(stripped)

        if match:
            if current_id is not None:
                paragraphs[current_id] = "\n".join(current_lines).strip()

            current_id = match.group(1)
            current_lines = []
        else:
            if current_id is not None:
                current_lines.append(line)

    if current_id is not None:
        paragraphs[current_id] = "\n".join(current_lines).strip()

    if not paragraphs:
        raise ValueError("No paragraphs found. Use IDs like [P001], [P002], ...")

    return paragraphs


def build_character_profiles(gold: dict) -> dict:
    """
    Builds one textual profile for each character.

    The model does not discover characters by itself.
    It receives the predefined character inventory and ranks candidate characters.
    """
    profiles = {}

    for character in gold["characters"]:
        char_id = character["char_id"]
        canonical_name = character["canonical_name"]
        character_type = character.get("character_type", "")
        aliases = character.get("aliases", [])
        notes = character.get("notes", "")

        aliases_text = ", ".join(aliases)

        profile_text = (
            f"Karakter adı: {canonical_name}. "
            f"Karakter türü: {character_type}. "
            f"Aliaslar: {aliases_text}. "
            f"Açıklama: {notes}"
        )

        profiles[char_id] = {
            "canonical_name": canonical_name,
            "profile_text": profile_text
        }

    return profiles


def build_top_k_presence_predictions(
    paragraph_ids: list[str],
    paragraph_embeddings: np.ndarray,
    character_ids: list[str],
    character_embeddings: np.ndarray,
    top_k: int
) -> list[dict]:
    """
    Selects the top-k most similar characters for each paragraph.
    """
    similarity_matrix = cosine_similarity(paragraph_embeddings, character_embeddings)

    predictions = []

    for p_idx, paragraph_id in enumerate(paragraph_ids):
        scores = []

        for c_idx, char_id in enumerate(character_ids):
            score = float(similarity_matrix[p_idx][c_idx])
            scores.append((char_id, score))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        selected = scores[:top_k]

        predictions.append({
            "paragraph_id": paragraph_id,
            "characters_present": sorted([char_id for char_id, _ in selected]),
            "similarities": {
                char_id: round(score, 4)
                for char_id, score in scores
            },
            "top_characters": [
                {
                    "char_id": char_id,
                    "similarity": round(score, 4)
                }
                for char_id, score in selected
            ],
            "selection_method": f"top_{top_k}"
        })

    return predictions


def build_top_k_inspection(
    paragraph_ids: list[str],
    paragraph_embeddings: np.ndarray,
    character_ids: list[str],
    character_embeddings: np.ndarray,
    top_k: int = 3
) -> list[dict]:
    """
    Gives top-k candidates for manual inspection.
    """
    similarity_matrix = cosine_similarity(paragraph_embeddings, character_embeddings)

    inspection = []

    for p_idx, paragraph_id in enumerate(paragraph_ids):
        scores = []

        for c_idx, char_id in enumerate(character_ids):
            score = float(similarity_matrix[p_idx][c_idx])
            scores.append((char_id, score))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        top_items = scores[:top_k]

        inspection.append({
            "paragraph_id": paragraph_id,
            "top_characters": [
                {
                    "char_id": char_id,
                    "similarity": round(score, 4)
                }
                for char_id, score in top_items
            ]
        })

    return inspection


def build_cooccurrence_edges(paragraph_predictions: list[dict]) -> list[dict]:
    """
    Creates undirected co-occurrence edges from predicted paragraph presence.
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


def main():
    print("Loading story...")
    paragraphs = load_story_paragraphs(STORY_PATH)

    print("Loading gold character inventory...")
    gold = load_json(GOLD_PATH)
    character_profiles = build_character_profiles(gold)

    paragraph_ids = list(paragraphs.keys())
    paragraph_texts = [paragraphs[p_id] for p_id in paragraph_ids]

    character_ids = list(character_profiles.keys())
    character_texts = [character_profiles[c_id]["profile_text"] for c_id in character_ids]

    print(f"Paragraph count: {len(paragraph_texts)}")
    print(f"Character count: {len(character_texts)}")

    print(f"Loading sentence-transformer model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Embedding paragraphs...")
    paragraph_embeddings = model.encode(
        paragraph_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("Embedding character profiles...")
    character_embeddings = model.encode(
        character_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print(f"Building paragraph predictions using top-{TOP_K} selection...")
    paragraph_predictions = build_top_k_presence_predictions(
        paragraph_ids=paragraph_ids,
        paragraph_embeddings=paragraph_embeddings,
        character_ids=character_ids,
        character_embeddings=character_embeddings,
        top_k=TOP_K
    )

    print("Building top-k inspection output...")
    top_k_inspection = build_top_k_inspection(
        paragraph_ids=paragraph_ids,
        paragraph_embeddings=paragraph_embeddings,
        character_ids=character_ids,
        character_embeddings=character_embeddings,
        top_k=3
    )

    edges = build_cooccurrence_edges(paragraph_predictions)

    output = {
        "text_id": gold["text_id"],
        "system": "sentence_transformer_topk_baseline",
        "model_name": MODEL_NAME,
        "description": (
            "Pretrained sentence-transformer embedding baseline. "
            "The model receives the predefined character inventory from the gold JSON, "
            "builds character profile embeddings, compares them with paragraph embeddings, "
            "and selects the top-k most similar character candidates for each paragraph. "
            "No model training or fine-tuning is performed."
        ),
        "selection_method": f"top_{TOP_K}",
        "top_k": TOP_K,
        "character_profiles": character_profiles,
        "paragraph_presence": paragraph_predictions,
        "top_k_inspection": top_k_inspection,
        "edges": edges
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Saved output to: {OUTPUT_PATH}")
    print(f"Edges created: {len(edges)}")


if __name__ == "__main__":
    main()