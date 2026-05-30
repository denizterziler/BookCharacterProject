import json
from pathlib import Path


def find_project_root() -> Path:
    """
    Finds BookCharacterProject root by walking upward until expected folders are found.
    This avoids parents[1], parents[2], parents[3] path mistakes.
    """
    current = Path(__file__).resolve()

    for parent in [current.parent] + list(current.parents):
        if (parent / "inputs").exists() and (parent / "predictions").exists():
            return parent

    raise RuntimeError("Could not find project root. Make sure 'inputs' and 'predictions' folders exist.")


PROJECT_ROOT = find_project_root()

BOOK_ID = "B05"

GOLD_PATH = (
    PROJECT_ROOT
    / "gold"
    / f"{BOOK_ID}_gold.json"
)
TOP_K = 1
# Change this path for each method you want to evaluate.
PRED_PATH = (
    PROJECT_ROOT
    / "predictions"
    / "local_llm"
    / f"B05_hf_co_ytu-ce-cosmos_Turkish-Gemma-9b-T1-GGUF_Q4_K_M_candidate_based_window_presence_validation_output.json"
)

RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / f"{BOOK_ID}_validation"
    / f"B05_ytu-cosmos_eval.json"
)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def get_gold_presence_items(gold: dict) -> list[dict]:
    """
    Expected gold format:

    {
      "book_id": "B02",
      "title": "Efsuncu Baba",
      "gold_annotations": [
        {
          "segment_id": "B02_S0001",
          "characters_present": ["B02_C001", "B02_C002"]
        }
      ]
    }
    """
    if "gold_annotations" not in gold:
        raise ValueError("Gold file must contain 'gold_annotations'.")

    return gold["gold_annotations"]


def get_prediction_presence_items(pred: dict) -> list[dict]:
    """
    Expected prediction format:

    {
      "book_id": "B02",
      "system": "...",
      "segment_presence": [
        {
          "segment_id": "B02_S0001",
          "characters_present": ["B02_C001", "B02_C002"]
        }
      ]
    }
    """
    if "segment_presence" not in pred:
        raise ValueError("Prediction file must contain 'segment_presence'.")

    return pred["segment_presence"]


def filter_predictions_to_gold_segments(
    pred_items: list[dict],
    gold_items: list[dict]
) -> list[dict]:
    """
    Keeps only predictions for the selected gold validation segments.

    This is important because some methods may produce predictions for all book segments,
    but validation should be calculated only on the manually annotated 10 segments.
    """
    gold_segment_ids = {
        item["segment_id"]
        for item in gold_items
    }

    return [
        item
        for item in pred_items
        if item["segment_id"] in gold_segment_ids
    ]


def get_presence_pairs(presence_items: list[dict]) -> set[tuple[str, str]]:
    """
    Converts segment-level presence list into pairs:

    {
      ("B02_S0001", "B02_C001"),
      ("B02_S0001", "B02_C002"),
      ("B02_S0002", "B02_C001")
    }
    """
    pairs = set()

    for item in presence_items:
        segment_id = item["segment_id"]

        for char_id in item.get("characters_present", []):
            pairs.add((segment_id, char_id))

    return pairs


def calculate_scores(gold_pairs: set[tuple[str, str]], pred_pairs: set[tuple[str, str]]) -> dict:
    true_positives = gold_pairs & pred_pairs
    false_positives = pred_pairs - gold_pairs
    false_negatives = gold_pairs - pred_pairs

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

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


def calculate_segment_level_details(
    gold_items: list[dict],
    pred_items: list[dict]
) -> list[dict]:
    """
    Gives per-segment TP/FP/FN for manual inspection.
    """
    gold_by_segment = {
        item["segment_id"]: set(item.get("characters_present", []))
        for item in gold_items
    }

    pred_by_segment = {
        item["segment_id"]: set(item.get("characters_present", []))
        for item in pred_items
    }

    all_segment_ids = sorted(gold_by_segment.keys())

    details = []

    for segment_id in all_segment_ids:
        gold_chars = gold_by_segment.get(segment_id, set())
        pred_chars = pred_by_segment.get(segment_id, set())

        true_positives = gold_chars & pred_chars
        false_positives = pred_chars - gold_chars
        false_negatives = gold_chars - pred_chars

        details.append({
            "segment_id": segment_id,
            "gold": sorted(gold_chars),
            "predicted": sorted(pred_chars),
            "true_positives": sorted(true_positives),
            "false_positives": sorted(false_positives),
            "false_negatives": sorted(false_negatives)
        })

    return details


def main():
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("GOLD_PATH:", GOLD_PATH)
    print("PRED_PATH:", PRED_PATH)
    print("RESULT_PATH:", RESULT_PATH)

    gold = load_json(GOLD_PATH)
    pred = load_json(PRED_PATH)

    gold_items = get_gold_presence_items(gold)
    pred_items = get_prediction_presence_items(pred)

    pred_items = filter_predictions_to_gold_segments(
        pred_items=pred_items,
        gold_items=gold_items
    )

    print(f"Gold validation segments: {len(gold_items)}")
    print(f"Prediction segments used: {len(pred_items)}")

    gold_segment_ids = {item["segment_id"] for item in gold_items}
    pred_segment_ids = {item["segment_id"] for item in pred_items}

    missing_prediction_segments = gold_segment_ids - pred_segment_ids

    if missing_prediction_segments:
        print("WARNING: Missing predictions for these gold segments:")
        print(sorted(missing_prediction_segments))

    gold_pairs = get_presence_pairs(gold_items)
    pred_pairs = get_presence_pairs(pred_items)

    scores = calculate_scores(gold_pairs, pred_pairs)
    segment_details = calculate_segment_level_details(gold_items, pred_items)

    output = {
        "book_id": gold.get("book_id", BOOK_ID),
        "title": gold.get("title", ""),
        "system": pred.get("system", ""),
        "model_name": pred.get("model_name", ""),
        "prompt_type": pred.get("prompt_type", ""),
        "selection_method": pred.get("selection_method", ""),
        "top_k": pred.get("top_k", None),
        "task": "300_word_window_level_character_presence",
        "unit": "300-word overlapping segment",
        "gold_path": str(GOLD_PATH),
        "prediction_path": str(PRED_PATH),
        "gold_segment_count": len(gold_items),
        "prediction_segment_count_used": len(pred_items),
        "missing_prediction_segments": sorted(missing_prediction_segments),
        "scores": scores,
        "segment_details": segment_details
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    RESULT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\nEvaluation completed.")
    print(f"Book:      {output['book_id']}")
    print(f"System:    {output['system']}")
    print(f"Model:     {output['model_name']}")
    print(f"Prompt:    {output['prompt_type']}")
    print(f"Top-k:     {output['top_k']}")
    print(f"Precision: {scores['precision']}")
    print(f"Recall:    {scores['recall']}")
    print(f"F1:        {scores['f1']}")
    print(f"TP:        {scores['true_positive_count']}")
    print(f"FP:        {scores['false_positive_count']}")
    print(f"FN:        {scores['false_negative_count']}")
    print(f"Saved result to: {RESULT_PATH}")


if __name__ == "__main__":
    main()