import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLD_PATH = PROJECT_ROOT / "gold" / "T01_gold.json"
"""
PRED_PATH = PROJECT_ROOT / "predictions" / "name_only" / "T01_output.json"
RESULT_PATH = PROJECT_ROOT / "results" / "T01_name_only_eval.json"
PRED_PATH = PROJECT_ROOT / "predictions" / "berturk" / "T01_output.json"
RESULT_PATH = PROJECT_ROOT / "results" / "T01_berturk_eval.json"
"""
PRED_PATH = PROJECT_ROOT / "predictions" / "sentence_transformer" / "T01_emrecan__bert-base-turkish-cased-mean-nli-stsb-tr_top2_output.json"
RESULT_PATH = PROJECT_ROOT / "results" / "T01_sentence_transformer_top2_eval.json"

def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def get_presence_pairs(paragraph_presence: list[dict]) -> set[tuple[str, str]]:
    """
    Converts paragraph presence list into pairs:

    {
      ("P001", "C01"),
      ("P002", "C01"),
      ("P002", "C02")
    }
    """
    pairs = set()

    for item in paragraph_presence:
        paragraph_id = item["paragraph_id"]
        for char_id in item["characters_present"]:
            pairs.add((paragraph_id, char_id))

    return pairs


def calculate_scores(gold_pairs: set, pred_pairs: set) -> dict:
    true_positives = gold_pairs & pred_pairs
    false_positives = pred_pairs - gold_pairs
    false_negatives = gold_pairs - pred_pairs

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "true_positive_count": tp,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": sorted(list(true_positives)),
        "false_positives": sorted(list(false_positives)),
        "false_negatives": sorted(list(false_negatives))
    }


def main():
    gold = load_json(GOLD_PATH)
    pred = load_json(PRED_PATH)

    gold_pairs = get_presence_pairs(gold["paragraph_presence"])
    pred_pairs = get_presence_pairs(pred["paragraph_presence"])

    scores = calculate_scores(gold_pairs, pred_pairs)

    output = {
        "text_id": gold["text_id"],
        "system": pred["system"],
        "task": "paragraph_level_character_presence",
        "scores": scores
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("Evaluation completed.")
    print(f"Precision: {scores['precision']}")
    print(f"Recall:    {scores['recall']}")
    print(f"F1:        {scores['f1']}")
    print(f"False positives: {scores['false_positive_count']}")
    print(f"False negatives: {scores['false_negative_count']}")
    print(f"Saved result to: {RESULT_PATH}")


if __name__ == "__main__":
    main()