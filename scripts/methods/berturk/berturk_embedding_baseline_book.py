import json
import itertools
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BOOK_ID = "B02"

SEGMENTS_PATH = PROJECT_ROOT / "scripts" / "book_texts" / f"{BOOK_ID}" /f"{BOOK_ID}_segments.json"
CHARACTERS_PATH = PROJECT_ROOT / "inputs" / f"{BOOK_ID}_characters.json"
OUTPUT_PATH = PROJECT_ROOT / "predictions" / "berturk" / f"{BOOK_ID}_window_presence_top3.json"

MODEL_NAME = "dbmdz/bert-base-turkish-cased"

# Try TOP_K = 1, 2, 3 separately and compare results.
TOP_K = 3


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

    BERTurk does not discover characters by itself here.
    It receives predefined candidate character profiles and compares them with segments.
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


def mean_pooling(model_output, attention_mask):
    """
    Mean pooling over token embeddings, ignoring padding tokens.
    """
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    summed_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    summed_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    return summed_embeddings / summed_mask


def embed_texts(texts: list[str], tokenizer, model, device, batch_size: int = 8) -> np.ndarray:
    """
    Converts a list of texts into embeddings using pretrained BERTurk.
    No training/fine-tuning is performed.
    """
    all_embeddings = []

    model.eval()

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            encoded = {key: value.to(device) for key, value in encoded.items()}

            model_output = model(**encoded)
            embeddings = mean_pooling(model_output, encoded["attention_mask"])

            all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


def build_top_k_presence_predictions(
    segment_ids: list[str],
    segment_embeddings: np.ndarray,
    character_ids: list[str],
    character_embeddings: np.ndarray,
    top_k: int = 3
) -> list[dict]:
    """
    Predicts the top-k most similar characters for each 300-word segment.

    This is a weak semantic baseline, not a true NER/coreference system.
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
    Creates a separate inspection output showing top-k candidates.
    This is useful for manually checking model behavior.
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

    print(f"Loading pretrained model: {MODEL_NAME}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    print("Embedding segments...")
    segment_embeddings = embed_texts(segment_texts, tokenizer, model, device)

    print("Embedding character profiles...")
    character_embeddings = embed_texts(character_texts, tokenizer, model, device)

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
        "system": "berturk_embedding_topk_baseline",
        "model_name": MODEL_NAME,
        "description": (
            "Pretrained BERTurk embedding similarity baseline. "
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