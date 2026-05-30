import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BOOK_ID = "B05"

SEGMENTS_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "book_texts"
    / BOOK_ID
    / f"{BOOK_ID}_segments.json"
)

CHARACTER_INVENTORY_PATH = (
    PROJECT_ROOT
    / "inputs"
    / f"{BOOK_ID}_characters.json"
)

GOLD_PATH = (
    PROJECT_ROOT
    / "gold"
    / f"{BOOK_ID}_gold.json"
)

OUTPUT_DIR = PROJECT_ROOT / "predictions" / "local_llm"

MODELS = [
    #"qwen2.5:14b",
    #"hf.co/ogulcanaydogan/Turkish-LLM-14B-Instruct-GGUF:Q4_K_M",
    "hf.co/ytu-ce-cosmos/Turkish-Gemma-9b-T1-GGUF:Q4_K_M",
]

PROMPT_TYPE = "candidate_based_window_presence_validation"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_validation_segment_ids(gold_path: Path) -> set[str]:
    """
    Reads the gold validation file and returns only the selected validation segment IDs.

    Expected format:
    {
      "book_id": "B05",
      "title": "...",
      "gold_annotations": [
        {
          "segment_id": "B05_S0001",
          "characters_present": [...]
        }
      ]
    }
    """
    gold = load_json(gold_path)

    if "gold_annotations" not in gold:
        raise ValueError("Gold validation JSON must contain 'gold_annotations'.")

    return {
        item["segment_id"]
        for item in gold["gold_annotations"]
    }


def build_candidate_prompt(segment: dict, character_inventory: dict) -> str:
    segment_id = segment["segment_id"]
    segment_text = segment["text"]

    example_char_ids = [
        character["char_id"]
        for character in character_inventory["characters"][:2]
    ]

    return f"""
Sen Türkçe romanlar üzerinde çalışan dikkatli bir NLP anotasyon asistanısın.

Sana iki girdi verilecek:

1. 300 kelimelik bir roman segmenti
2. Aday karakter envanteri JSON'u

Görevin, verilen aday karakterler arasından bu segmentte hangi karakterlerin var olduğunu belirlemektir.

ÖNEMLİ KURALLAR:
- Sadece geçerli JSON döndür.
- Markdown kullanma.
- Açıklama yazma.
- JSON dışında hiçbir metin üretme.
- Yeni karakter ekleme.
- Sadece character_inventory içindeki char_id değerlerini kullan.
- characters_present içinde sadece char_id değerleri olsun.
- Segment ID değerini değiştirme.
- Başka kitaba ait char_id kullanma.
- Bu kitap için book_id: {BOOK_ID}
- Tüm karakter ID'leri {BOOK_ID}_C ile başlamalıdır.

KARAKTER VARLIĞI KURALI:
Bir karakter şu durumlarda segmentte var kabul edilir:
- karakterin adı geçiyorsa,
- alias kullanılıyorsa,
- karakter konuşuyorsa,
- karaktere hitap ediliyorsa,
- karakter açıkça tartışılıyorsa,
- karakter zamirle veya bağlamla açıkça belirtiliyorsa,
- önceki örtüşen bağlamdan gelen açık özne devamlılığı varsa.

Ancak:
- Sadece mekân adı geçmesi karakter varlığı oluşturmaz.
- Yeni karakter uydurma.
- Emin değilsen karakteri ekleme.

ÇIKTI ŞEMASI:
Aşağıdaki yapıda geçerli JSON üret:

{{
  "segment_id": "{segment_id}",
  "characters_present": {json.dumps(example_char_ids, ensure_ascii=False)}
}}

SEGMENT:
<<<
{segment_text}
>>>

CHARACTER_INVENTORY_JSON:
<<<
{json.dumps(character_inventory, ensure_ascii=False, indent=2)}
>>>
""".strip()


def run_ollama_model(model_name: str, prompt: str) -> str:
    print(f"Running model: {model_name}")

    result = subprocess.run(
        ["ollama", "run", model_name],
        input=prompt,
        text=True,
        capture_output=True,
        encoding="utf-8"
    )

    if result.returncode != 0:
        print("STDERR:")
        print(result.stderr)
        raise RuntimeError(f"Ollama model failed: {model_name}")

    return result.stdout.strip()


def extract_json(raw_output: str) -> str:
    """
    Extracts the final JSON object from model output.

    Handles outputs like:
    <think>
    reasoning...
    {bad/example json}
    </think>
    {
      "segment_id": "...",
      "characters_present": [...]
    }
    """
    cleaned = raw_output.strip()

    # Remove thinking block if model produced one.
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")

    return cleaned[start:end + 1]


def validate_prediction(
    parsed: dict,
    valid_char_ids: set[str],
    expected_segment_id: str,
    book_id: str
) -> dict:
    """
    Cleans model output so evaluation will not crash.

    Also fixes wrong book-prefix errors such as:
    B02_C001 -> B05_C001
    if B05_C001 exists in the valid character inventory.
    """
    segment_id = parsed.get("segment_id", expected_segment_id)

    if segment_id != expected_segment_id:
        print(f"Corrected segment ID: {segment_id} -> {expected_segment_id}")
        segment_id = expected_segment_id

    characters_present = parsed.get("characters_present", [])

    if not isinstance(characters_present, list):
        characters_present = []

    cleaned_characters = []

    for char_id in characters_present:
        if not isinstance(char_id, str):
            continue

        # Case 1: already valid.
        if char_id in valid_char_ids:
            cleaned_characters.append(char_id)
            continue

        # Case 2: wrong book prefix, e.g. B02_C001 instead of B05_C001.
        if "_C" in char_id:
            suffix = char_id.split("_C", 1)[1]
            corrected_char_id = f"{book_id}_C{suffix}"

            if corrected_char_id in valid_char_ids:
                print(f"Corrected character ID: {char_id} -> {corrected_char_id}")
                cleaned_characters.append(corrected_char_id)
            else:
                print(f"Ignored invalid character ID: {char_id}")
        else:
            print(f"Ignored invalid character ID: {char_id}")

    cleaned_characters = sorted(set(cleaned_characters))

    return {
        "segment_id": segment_id,
        "characters_present": cleaned_characters
    }


def safe_model_name(model_name: str) -> str:
    return (
        model_name
        .replace(":", "_")
        .replace("/", "_")
        .replace(".", "_")
    )


def save_outputs(model_name: str, output: dict, raw_outputs: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = safe_model_name(model_name)

    json_path = OUTPUT_DIR / f"{BOOK_ID}_{safe_name}_{PROMPT_TYPE}_output.json"
    raw_path = OUTPUT_DIR / f"{BOOK_ID}_{safe_name}_{PROMPT_TYPE}_raw.json"

    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    raw_path.write_text(
        json.dumps(raw_outputs, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Saved parsed JSON to: {json_path}")
    print(f"Saved raw outputs to: {raw_path}")


def main():
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("SEGMENTS_PATH:", SEGMENTS_PATH)
    print("CHARACTER_INVENTORY_PATH:", CHARACTER_INVENTORY_PATH)
    print("GOLD_PATH:", GOLD_PATH)

    segments_data = load_json(SEGMENTS_PATH)
    character_inventory = load_json(CHARACTER_INVENTORY_PATH)

    all_segments = segments_data["segments"]

    validation_segment_ids = load_validation_segment_ids(GOLD_PATH)

    segments = [
        segment
        for segment in all_segments
        if segment["segment_id"] in validation_segment_ids
    ]

    segments = sorted(
        segments,
        key=lambda segment: segment["segment_id"]
    )

    print(f"All segments in book: {len(all_segments)}")
    print(f"Validation segment IDs: {sorted(validation_segment_ids)}")
    print(f"Validation segments selected: {len(segments)}")

    if len(segments) != len(validation_segment_ids):
        found_ids = {segment["segment_id"] for segment in segments}
        missing_ids = validation_segment_ids - found_ids
        raise ValueError(f"Some validation segments were not found: {sorted(missing_ids)}")

    valid_char_ids = {
        character["char_id"]
        for character in character_inventory["characters"]
    }

    for model_name in MODELS:
        segment_predictions = []
        raw_outputs = {}

        for segment in segments:
            segment_id = segment["segment_id"]

            print(f"\nProcessing {segment_id} with {model_name}")

            prompt = build_candidate_prompt(
                segment=segment,
                character_inventory=character_inventory
            )

            raw_output = run_ollama_model(model_name, prompt)
            raw_outputs[segment_id] = raw_output

            try:
                json_text = extract_json(raw_output)
                parsed = json.loads(json_text)

                cleaned_prediction = validate_prediction(
                    parsed=parsed,
                    valid_char_ids=valid_char_ids,
                    expected_segment_id=segment_id,
                    book_id=BOOK_ID
                )

            except Exception as e:
                print(f"Could not parse valid JSON for {segment_id}.")
                print("Error:", e)

                cleaned_prediction = {
                    "segment_id": segment_id,
                    "characters_present": []
                }

            segment_predictions.append(cleaned_prediction)

        output = {
            "book_id": BOOK_ID,
            "title": character_inventory.get("title", ""),
            "system": "local_llm_candidate_based",
            "model_name": model_name,
            "prompt_type": PROMPT_TYPE,
            "unit": "300-word overlapping segment",
            "mode": "validation_only",
            "gold_file": str(GOLD_PATH),
            "description": (
                "Local LLM candidate-based character presence extraction. "
                "This validation run processes only the manually selected gold validation segments. "
                "The model receives one 300-word segment and the predefined character inventory, "
                "then returns only the character IDs present in that segment. "
                "Post-processing corrects wrong book-prefix character IDs when the corresponding "
                "character ID exists in the current book inventory."
            ),
            "segment_presence": segment_predictions
        }

        save_outputs(
            model_name=model_name,
            output=output,
            raw_outputs=raw_outputs
        )


if __name__ == "__main__":
    main()