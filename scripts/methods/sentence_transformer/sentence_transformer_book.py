import json
import itertools
from pathlib import Path
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BOOK_ID = "B05"

SEGMENTS_PATH = PROJECT_ROOT / "scripts" / "book_texts" / f"{BOOK_ID}" / f"{BOOK_ID}_segments.json"
CHARACTERS_PATH = PROJECT_ROOT / "inputs" / f"{BOOK_ID}_characters.json"

# Change this name for each model you want to test.
#MODEL_NAME = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
#MODEL_NAME = "trmteb/turkish-embedding-model"
MODEL_NAME = "mertcobanov/mpnet-base-all-nli-triplet-turkish-v2"

# Try 1, 2, and 3 separately.
TOP_K = 3

SAFE_MODEL_NAME = MODEL_NAME.replace("/", "__")
OUTPUT_PATH = (
    PROJECT_ROOT
    / "predictions"
    / "sentence_transformer"
    / f"{BOOK_ID}_{SAFE_MODEL_NAME}_window_presence_top{TOP_K}.json"
)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_segments(segments_path: Path) -> list[dict]:
    """
    Reads 300-word window/segment JSON.

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
    data = load_json(segments_path)

    if "segments" not in data:
        raise ValueError("Segments JSON must contain a 'segments' field.")

    return data["segments"]


def build_character_profiles(character_inventory: dict) -> dict:
    """
    Builds one textual profile for each character.

    The model does not discover characters by itself.
    It receives the predefined character inventory and ranks candidate characters.
    """
    profiles = {}

    for character in character_inventory["characters"]:
        char_id = character["char_id"]
        canonical_name = character["canonical_name"]
        aliases = character.get("aliases", [])
        gender = character.get("gender", "")
        role = character.get("role", "")
        notes = character.get("notes", "")

        aliases_text = ", ".join(aliases)

        profile_text = (
            f"Karakter adı: {canonical_name}. "
            f"Diğer adları: {aliases_text}. "
            f"Cinsiyet: {gender}. "
            f"Rol: {role}. "
            f"Açıklama: {notes}"
        )

        profiles[char_id] = {
            "canonical_name": canonical_name,
            "profile_text": profile_text
        }

    return profiles


def build_top_k_presence_predictions(
    segment_ids: list[str],
    segment_embeddings: np.ndarray,
    character_ids: list[str],
    character_embeddings: np.ndarray,
    top_k: int
) -> list[dict]:
    """
    Selects the top-k most similar characters for each 300-word segment.
    """
    similarity_matrix = cosine_similarity(segment_embeddings, character_embeddings)

    predictions = []

    for s_idx, segment_id in enumerate(segment_ids):
        scores = []

        for c_idx, char_id in enumerate(character_ids):
            score = float(similarity_matrix[s_idx][c_idx])
            scores.append((char_id, score))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        selected = scores[:top_k]

        predictions.append({
            "segment_id": segment_id,
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
    segment_ids: list[str],
    segment_embeddings: np.ndarray,
    character_ids: list[str],
    character_embeddings: np.ndarray,
    top_k: int = 5
) -> list[dict]:
    """
    Gives top-k candidates for manual inspection.
    """
    similarity_matrix = cosine_similarity(segment_embeddings, character_embeddings)

    inspection = []

    for s_idx, segment_id in enumerate(segment_ids):
        scores = []

        for c_idx, char_id in enumerate(character_ids):
            score = float(similarity_matrix[s_idx][c_idx])
            scores.append((char_id, score))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        top_items = scores[:top_k]

        inspection.append({
            "segment_id": segment_id,
            "top_characters": [
                {
                    "char_id": char_id,
                    "similarity": round(score, 4)
                }
                for char_id, score in top_items
            ]
        })

    return inspection


def build_cooccurrence_edges(segment_predictions: list[dict]) -> list[dict]:
    """
    Creates undirected co-occurrence edges from predicted segment presence.
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


def main():
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("SEGMENTS_PATH:", SEGMENTS_PATH)
    print("CHARACTERS_PATH:", CHARACTERS_PATH)

    print("Loading segments...")
    segments = load_segments(SEGMENTS_PATH)

    print("Loading character inventory...")
    character_inventory = load_json(CHARACTERS_PATH)
    character_profiles = build_character_profiles(character_inventory)

    segment_ids = [segment["segment_id"] for segment in segments]
    segment_texts = [segment["text"] for segment in segments]

    character_ids = list(character_profiles.keys())
    character_texts = [character_profiles[c_id]["profile_text"] for c_id in character_ids]

    print(f"Segment count: {len(segment_texts)}")
    print(f"Character count: {len(character_texts)}")

    print(f"Loading sentence-transformer model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Embedding segments...")
    segment_embeddings = model.encode(
        segment_texts,
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

    print(f"Building segment predictions using top-{TOP_K} selection...")
    segment_predictions = build_top_k_presence_predictions(
        segment_ids=segment_ids,
        segment_embeddings=segment_embeddings,
        character_ids=character_ids,
        character_embeddings=character_embeddings,
        top_k=TOP_K
    )

    print("Building top-k inspection output...")
    top_k_inspection = build_top_k_inspection(
        segment_ids=segment_ids,
        segment_embeddings=segment_embeddings,
        character_ids=character_ids,
        character_embeddings=character_embeddings,
        top_k=5
    )

    edges = build_cooccurrence_edges(segment_predictions)

    output = {
        "book_id": BOOK_ID,
        "title": character_inventory.get("title", ""),
        "system": "sentence_transformer_topk_baseline",
        "model_name": MODEL_NAME,
        "description": (
            "Pretrained sentence-transformer embedding baseline. "
            "The model receives the predefined character inventory, "
            "builds character profile embeddings, compares them with 300-word segment embeddings, "
            "and selects the top-k most similar character candidates for each segment. "
            "No model training or fine-tuning is performed."
        ),
        "unit": "300-word overlapping segment",
        "selection_method": f"top_{TOP_K}",
        "top_k": TOP_K,
        "character_profiles": character_profiles,
        "segment_presence": segment_predictions,
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