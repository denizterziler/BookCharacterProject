import json
import re
import time
from pathlib import Path
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BOOK_ID = "B05"

MODEL = "gpt-4.1-mini"
PROMPT_TYPE = "candidate_based_window_presence_full"

USE_EXACT_ALIAS = True
SLEEP_SECONDS = 0.3

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

GPT_OUTPUT_DIR = PROJECT_ROOT / "predictions" / BOOK_ID / "llm"
EXACT_ALIAS_OUTPUT_DIR = PROJECT_ROOT / "predictions"/ BOOK_ID / "exact_alias"
ENSEMBLE_OUTPUT_DIR = PROJECT_ROOT / "predictions" / BOOK_ID / "ensemble"

GPT_OUTPUT_PATH = (
    GPT_OUTPUT_DIR
    / f"{BOOK_ID}_gpt-4_1-mini_{PROMPT_TYPE}_output.json"
)

GPT_RAW_OUTPUT_PATH = (
    GPT_OUTPUT_DIR
    / f"{BOOK_ID}_gpt-4_1-mini_{PROMPT_TYPE}_raw.json"
)

EXACT_ALIAS_OUTPUT_PATH = (
    EXACT_ALIAS_OUTPUT_DIR
    / f"{BOOK_ID}_exact_alias_window_presence_full_output.json"
)

ENSEMBLE_OUTPUT_PATH = (
    ENSEMBLE_OUTPUT_DIR
    / f"{BOOK_ID}_gpt_alias_ensemble_window_presence.json"
)


client = OpenAI()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    text = text.replace("`", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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


def run_openai_model(prompt: str) -> str:
    response = client.responses.create(
        model=MODEL,
        input=prompt,
        temperature=0
    )

    return response.output_text.strip()


def extract_json(raw_output: str) -> str:
    cleaned = raw_output.strip()

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

        if char_id in valid_char_ids:
            cleaned_characters.append(char_id)
            continue

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

    return {
        "segment_id": segment_id,
        "characters_present": sorted(set(cleaned_characters))
    }


def create_alias_patterns(character_inventory: dict) -> dict:
    alias_patterns = {}

    for character in character_inventory["characters"]:
        char_id = character["char_id"]
        aliases = character.get("aliases", [])

        patterns = []

        for alias in aliases:
            alias = alias.strip()

            if not alias:
                continue

            normalized_alias = normalize_text(alias)
            escaped_alias = re.escape(normalized_alias)

            pattern = rf"(?<!\w){escaped_alias}(?:'[a-zçğıöşüâîû]+)?(?!\w)"

            patterns.append(re.compile(pattern, flags=re.IGNORECASE))

        alias_patterns[char_id] = patterns

    return alias_patterns


def run_exact_alias_baseline(
    segments_data: dict,
    character_inventory: dict
) -> dict:
    alias_patterns = create_alias_patterns(character_inventory)

    segment_presence = []

    for segment in segments_data["segments"]:
        segment_id = segment["segment_id"]
        text = normalize_text(segment["text"])

        characters_present = []

        for char_id, patterns in alias_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    characters_present.append(char_id)
                    break

        segment_presence.append({
            "segment_id": segment_id,
            "characters_present": sorted(set(characters_present))
        })

    return {
        "book_id": BOOK_ID,
        "title": character_inventory.get("title", ""),
        "system": "exact_alias_baseline",
        "unit": "300-word overlapping segment",
        "mode": "full_book",
        "description": (
            "Exact alias matching baseline. A character is marked as present "
            "if one of the predefined aliases is found in the segment text."
        ),
        "segment_presence": segment_presence
    }


def presence_to_map(presence_data: dict) -> dict[str, set[str]]:
    presence_map = {}

    for item in presence_data["segment_presence"]:
        segment_id = item["segment_id"]
        characters = set(item.get("characters_present", []))
        presence_map[segment_id] = characters

    return presence_map


def build_ensemble_presence(
    segments_data: dict,
    character_inventory: dict,
    gpt_presence_data: dict,
    exact_alias_presence_data: dict
) -> dict:
    valid_char_ids = {
        character["char_id"]
        for character in character_inventory["characters"]
    }

    gpt_map = presence_to_map(gpt_presence_data)
    exact_alias_map = presence_to_map(exact_alias_presence_data)

    segment_presence = []

    for segment in segments_data["segments"]:
        segment_id = segment["segment_id"]

        gpt_chars = gpt_map.get(segment_id, set())
        exact_alias_chars = exact_alias_map.get(segment_id, set())

        if USE_EXACT_ALIAS:
            final_chars = gpt_chars | exact_alias_chars
        else:
            final_chars = gpt_chars

        final_chars = {
            char_id
            for char_id in final_chars
            if char_id in valid_char_ids
        }

        segment_presence.append({
            "segment_id": segment_id,
            "characters_present": sorted(final_chars),
            "sources": {
                "gpt4_1_mini": sorted(gpt_chars),
                "exact_alias": sorted(exact_alias_chars)
            }
        })

    return {
        "book_id": BOOK_ID,
        "title": character_inventory.get("title", ""),
        "system": (
            "gpt4_1_mini_plus_exact_alias"
            if USE_EXACT_ALIAS
            else "gpt4_1_mini_only"
        ),
        "unit": "300-word overlapping segment",
        "mode": "full_book",
        "presence_rule": (
            "final_chars = gpt4_1_mini_chars ∪ exact_alias_chars"
            if USE_EXACT_ALIAS
            else "final_chars = gpt4_1_mini_chars"
        ),
        "source_files": {
            "segments": str(SEGMENTS_PATH),
            "character_inventory": str(CHARACTER_INVENTORY_PATH),
            "gpt_presence": str(GPT_OUTPUT_PATH),
            "exact_alias_presence": str(EXACT_ALIAS_OUTPUT_PATH)
        },
        "segment_presence": segment_presence
    }


def run_gpt_presence(
    segments_data: dict,
    character_inventory: dict
) -> tuple[dict, dict]:
    segments = sorted(
        segments_data["segments"],
        key=lambda segment: segment["segment_id"]
    )

    valid_char_ids = {
        character["char_id"]
        for character in character_inventory["characters"]
    }

    segment_predictions = []
    raw_outputs = {}

    for index, segment in enumerate(segments, start=1):
        segment_id = segment["segment_id"]

        print(f"\n[{index}/{len(segments)}] Processing {segment_id} with {MODEL}")

        prompt = build_candidate_prompt(
            segment=segment,
            character_inventory=character_inventory
        )

        raw_output = run_openai_model(prompt)
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

        except Exception as error:
            print(f"Could not parse valid JSON for {segment_id}.")
            print("Error:", error)

            cleaned_prediction = {
                "segment_id": segment_id,
                "characters_present": []
            }

        segment_predictions.append(cleaned_prediction)

        time.sleep(SLEEP_SECONDS)

    output = {
        "book_id": BOOK_ID,
        "title": character_inventory.get("title", ""),
        "system": "gpt_candidate_based_character_presence",
        "model_name": MODEL,
        "prompt_type": PROMPT_TYPE,
        "unit": "300-word overlapping segment",
        "mode": "full_book",
        "description": (
            "GPT-4.1-mini candidate-based character presence extraction. "
            "This full-book run processes all 300-word overlapping segments. "
            "The model receives one segment and the predefined character inventory, "
            "then returns only the character IDs present in that segment."
        ),
        "segment_presence": segment_predictions
    }

    return output, raw_outputs


def main():
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("BOOK_ID:", BOOK_ID)
    print("MODEL:", MODEL)
    print("SEGMENTS_PATH:", SEGMENTS_PATH)
    print("CHARACTER_INVENTORY_PATH:", CHARACTER_INVENTORY_PATH)
    print("GPT_OUTPUT_PATH:", GPT_OUTPUT_PATH)
    print("EXACT_ALIAS_OUTPUT_PATH:", EXACT_ALIAS_OUTPUT_PATH)
    print("ENSEMBLE_OUTPUT_PATH:", ENSEMBLE_OUTPUT_PATH)

    segments_data = load_json(SEGMENTS_PATH)
    character_inventory = load_json(CHARACTER_INVENTORY_PATH)

    print("\nRunning GPT-4.1-mini full-book character presence extraction...")
    gpt_presence_data, raw_outputs = run_gpt_presence(
        segments_data=segments_data,
        character_inventory=character_inventory
    )

    save_json(gpt_presence_data, GPT_OUTPUT_PATH)
    save_json(raw_outputs, GPT_RAW_OUTPUT_PATH)

    print("\nRunning exact alias baseline...")
    exact_alias_presence_data = run_exact_alias_baseline(
        segments_data=segments_data,
        character_inventory=character_inventory
    )

    save_json(exact_alias_presence_data, EXACT_ALIAS_OUTPUT_PATH)

    print("\nBuilding GPT + exact alias ensemble...")
    ensemble_presence_data = build_ensemble_presence(
        segments_data=segments_data,
        character_inventory=character_inventory,
        gpt_presence_data=gpt_presence_data,
        exact_alias_presence_data=exact_alias_presence_data
    )

    save_json(ensemble_presence_data, ENSEMBLE_OUTPUT_PATH)

    print("\nCompleted.")
    print(f"Book:                {BOOK_ID}")
    print(f"Segments:            {len(segments_data['segments'])}")
    print(f"GPT output:          {GPT_OUTPUT_PATH}")
    print(f"GPT raw output:      {GPT_RAW_OUTPUT_PATH}")
    print(f"Exact alias output:  {EXACT_ALIAS_OUTPUT_PATH}")
    print(f"Ensemble output:     {ENSEMBLE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()