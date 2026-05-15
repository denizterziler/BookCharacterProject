import json
import re
import itertools
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STORY_PATH = PROJECT_ROOT / "corpus" / "T01_story.txt"
GOLD_PATH = PROJECT_ROOT / "gold" / "T01_gold.json"
OUTPUT_PATH = PROJECT_ROOT / "predictions" / "berturk" / "T01_output_top3.json"

MODEL_NAME = "dbmdz/bert-base-turkish-cased"

# Try TOP_K = 1, 2, 3 separately and compare results.
TOP_K = 3


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

    Important:
    BERTurk does not discover characters by itself here.
    It receives candidate character profiles and compares them with paragraphs.
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
    paragraph_ids: list[str],
    paragraph_embeddings: np.ndarray,
    character_ids: list[str],
    character_embeddings: np.ndarray,
    top_k: int = 2
) -> list[dict]:
    """
    Predicts the top-k most similar characters for each paragraph.

    This is better than fixed thresholding for raw BERTurk embeddings because
    cosine similarities tend to be high for many paragraph-character pairs.
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
    Creates a separate inspection output showing top-k candidates.
    This is useful for manually checking model behavior.
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

    print(f"Loading pretrained model: {MODEL_NAME}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    print("Embedding paragraphs...")
    paragraph_embeddings = embed_texts(paragraph_texts, tokenizer, model, device)

    print("Embedding character profiles...")
    character_embeddings = embed_texts(character_texts, tokenizer, model, device)

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
        "system": "berturk_embedding_topk_baseline",
        "model_name": MODEL_NAME,
        "description": (
            "Pretrained BERTurk embedding similarity baseline. "
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