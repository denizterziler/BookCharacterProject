import json
import time
from pathlib import Path
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BOOK_ID = "B05"

MODEL = "gpt-4.1-mini"

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

PRESENCE_PATH = (
    PROJECT_ROOT
    / "predictions"
    / BOOK_ID
    / "ensemble"
    / f"{BOOK_ID}_gpt_alias_ensemble_window_presence.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "character_features"
    / f"{BOOK_ID}_character_features.json"
)


MAX_SEGMENTS_PER_CHARACTER = 12
SLEEP_SECONDS = 0.5


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


def extract_json(raw_output: str) -> dict:
    cleaned = raw_output.strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No valid JSON object found:\n{raw_output}")

    json_text = cleaned[start:end + 1]
    return json.loads(json_text)


def build_segment_lookup(segments_data: dict) -> dict:
    return {
        segment["segment_id"]: segment["text"]
        for segment in segments_data["segments"]
    }


def build_character_presence_map(presence_data: dict) -> dict:
    """
    Returns:
    {
      "B01_C001": ["B01_S0001", "B01_S0002"]
    }
    """
    char_to_segments = {}

    for segment in presence_data["segment_presence"]:
        segment_id = segment["segment_id"]

        for char_id in segment.get("characters_present", []):
            if char_id not in char_to_segments:
                char_to_segments[char_id] = []

            char_to_segments[char_id].append(segment_id)

    return char_to_segments


def select_evidence_segments(segment_ids: list[str], max_segments: int) -> list[str]:
    """
    Keeps the feature extraction cheap and manageable.
    Uses first, middle, and last segments when there are many.
    """
    if len(segment_ids) <= max_segments:
        return segment_ids

    selected = []

    # First part
    selected.extend(segment_ids[:4])

    # Middle part
    middle_start = max(0, len(segment_ids) // 2 - 2)
    selected.extend(segment_ids[middle_start:middle_start + 4])

    # Last part
    selected.extend(segment_ids[-4:])

    # Remove duplicates while preserving order
    seen = set()
    unique_selected = []

    for segment_id in selected:
        if segment_id not in seen:
            unique_selected.append(segment_id)
            seen.add(segment_id)

    return unique_selected[:max_segments]


def create_prompt(
    book_id: str,
    character: dict,
    evidence_segments: list[dict]
) -> str:
    character_info = {
        "char_id": character["char_id"],
        "canonical_name": character.get("canonical_name", ""),
        "aliases": character.get("aliases", []),
        "gender": character.get("gender", ""),
        "role": character.get("role", ""),
        "notes": character.get("notes", "")
    }

    return f"""
Sen Türkçe edebî metinlerden karakter özellikleri çıkaran bir bilgi çıkarımı sistemisin.

Kitap ID: {book_id}

Karakter hakkında önceden bilinen bilgiler:
{json.dumps(character_info, ensure_ascii=False, indent=2)}

Bu karakterin geçtiği kanıt segmentleri:
{json.dumps(evidence_segments, ensure_ascii=False, indent=2)}

Görev:
Sadece aşağıdaki üç ek özelliği çıkar:

1. occupation
2. wealth_status
3. age_group

occupation:
- Serbest metin olabilir.
- Örnekler: "hizmetçi", "öğrenci", "memur", "asker", "tüccar", "ev hanımı"
- Metinde açık kanıt yoksa "unknown" yaz.

wealth_status için izin verilen değerler:
- poor
- lower_middle
- middle
- upper_middle
- wealthy
- unknown

age_group için izin verilen değerler:
- child_youth
- teen
- adult
- old
- unknown

age_group açıklamaları:
- child_youth: çocuk veya genç yaşta olduğu açıkça anlaşılan karakter
- teen: ergenlik / delikanlılık / genç kızlık dönemi açıkça anlaşılan karakter
- adult: yetişkin olduğu anlaşılan karakter
- old: yaşlı olduğu açıkça belirtilen veya bağlamdan güçlü biçimde anlaşılan karakter
- unknown: yaş grubu için yeterli kanıt yok

Kurallar:
- Sadece verilen karakter bilgilerini ve kanıt segmentlerini kullan.
- Tahmin yapma.
- Kanıt açık değilse "unknown" kullan.
- Confidence değeri 0 ile 100 arasında bir tam sayı olmalıdır.
- Değer "unknown" ise confidence mutlaka 0 olmalıdır.
- "unknown" olmayan her değer için en az bir evidence segment ID verilmelidir.
- Çıktı sadece geçerli JSON olmalıdır.
- JSON dışında açıklama yazma.
- Segmentlerde karakterle ilgisi olmayan bilgileri kullanma.
- Özelliği bütün kitap için değil, yalnızca bu karakter için çıkar.

Çıktıyı tam olarak şu JSON yapısında döndür:

{{
  "char_id": "{character["char_id"]}",
  "canonical_name": "{character.get("canonical_name", "")}",
  "occupation": {{
    "value": "",
    "confidence": 0,
    "evidence_segments": []
  }},
  "wealth_status": {{
    "value": "",
    "confidence": 0,
    "evidence_segments": []
  }},
  "age_group": {{
    "value": "",
    "confidence": 0,
    "evidence_segments": []
  }}
}}
""".strip()

def call_gpt(prompt: str) -> dict:
    response = client.responses.create(
        model=MODEL,
        input=prompt,
        temperature=0
    )

    raw_output = response.output_text
    return extract_json(raw_output)


def validate_feature_output(
    result: dict,
    character: dict,
    valid_segment_ids: set[str]
) -> dict:
    allowed_wealth = {
        "poor",
        "lower_middle",
        "middle",
        "upper_middle",
        "wealthy",
        "unknown"
    }

    allowed_age = {
        "child_youth",
        "teen",
        "adult",
        "old",
        "unknown"
    }

    char_id = character["char_id"]
    canonical_name = character.get("canonical_name", "")

    cleaned = {
        "char_id": char_id,
        "canonical_name": canonical_name
    }

    fields = {
        "occupation": None,
        "wealth_status": allowed_wealth,
        "age_group": allowed_age
    }

    for field, allowed_values in fields.items():
        item = result.get(field, {})

        value = item.get("value", "unknown")
        confidence = item.get("confidence", 0)
        evidence_segments = item.get("evidence_segments", [])

        if not isinstance(value, str):
            value = "unknown"

        value = value.strip()

        if field == "occupation":
            if not value:
                value = "unknown"
        else:
            if value not in allowed_values:
                value = "unknown"

        if not isinstance(confidence, int):
            try:
                confidence = int(confidence)
            except Exception:
                confidence = 0

        confidence = max(0, min(100, confidence))

        if value == "unknown":
            confidence = 0
            evidence_segments = []

        if not isinstance(evidence_segments, list):
            evidence_segments = []

        evidence_segments = [
            segment_id
            for segment_id in evidence_segments
            if isinstance(segment_id, str) and segment_id in valid_segment_ids
        ]

        if value != "unknown" and not evidence_segments:
            value = "unknown"
            confidence = 0
            evidence_segments = []

        cleaned[field] = {
            "value": value,
            "confidence": confidence,
            "evidence_segments": evidence_segments
        }

    return cleaned


def main():
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("BOOK_ID:", BOOK_ID)
    print("MODEL:", MODEL)
    print("SEGMENTS_PATH:", SEGMENTS_PATH)
    print("CHARACTER_INVENTORY_PATH:", CHARACTER_INVENTORY_PATH)
    print("PRESENCE_PATH:", PRESENCE_PATH)
    print("OUTPUT_PATH:", OUTPUT_PATH)

    segments_data = load_json(SEGMENTS_PATH)
    character_inventory = load_json(CHARACTER_INVENTORY_PATH)
    presence_data = load_json(PRESENCE_PATH)

    segment_lookup = build_segment_lookup(segments_data)
    char_to_segments = build_character_presence_map(presence_data)

    all_features = []

    characters = character_inventory["characters"]

    for index, character in enumerate(characters, start=1):
        char_id = character["char_id"]
        canonical_name = character.get("canonical_name", "")

        print(f"\n[{index}/{len(characters)}] Extracting features for {char_id} - {canonical_name}")

        segment_ids = char_to_segments.get(char_id, [])

        selected_segment_ids = select_evidence_segments(
            segment_ids=segment_ids,
            max_segments=MAX_SEGMENTS_PER_CHARACTER
        )

        evidence_segments = [
            {
                "segment_id": segment_id,
                "text": segment_lookup[segment_id]
            }
            for segment_id in selected_segment_ids
            if segment_id in segment_lookup
        ]

        valid_segment_ids = {
            item["segment_id"]
            for item in evidence_segments
        }

        if not evidence_segments:
            print("No evidence segments found. Marking as unknown.")

            feature_result = {
                "char_id": char_id,
                "canonical_name": canonical_name,
                "occupation": {
                    "value": "unknown",
                    "confidence": 0,
                    "evidence_segments": []
                },
                "wealth_status": {
                    "value": "unknown",
                    "confidence": 0,
                    "evidence_segments": []
                },
                "age_group": {
                    "value": "unknown",
                    "confidence": 0,
                    "evidence_segments": []
                }
            }

            all_features.append(feature_result)
            continue

        prompt = create_prompt(
            book_id=BOOK_ID,
            character=character,
            evidence_segments=evidence_segments
        )

        try:
            raw_result = call_gpt(prompt)

            feature_result = validate_feature_output(
                result=raw_result,
                character=character,
                valid_segment_ids=valid_segment_ids
            )

        except Exception as error:
            print(f"ERROR for {char_id}: {error}")

            feature_result = {
                "char_id": char_id,
                "canonical_name": canonical_name,
                "occupation": {
                    "value": "unknown",
                    "confidence": 0,
                    "evidence_segments": []
                },
                "wealth_status": {
                    "value": "unknown",
                    "confidence": 0,
                    "evidence_segments": []
                },
                "age_group": {
                    "value": "unknown",
                    "confidence": 0,
                    "evidence_segments": []
                },
                "error": str(error)
            }

        all_features.append(feature_result)

        time.sleep(SLEEP_SECONDS)

    output = {
        "book_id": BOOK_ID,
        "model": MODEL,
        "source_presence_file": str(PRESENCE_PATH),
        "input_character_fields": [
            "canonical_name",
            "aliases",
            "gender",
            "role",
            "notes"
        ],
        "extracted_character_features": [
            "occupation",
            "wealth_status",
            "age_group"
        ],
        "wealth_status_labels": [
            "poor",
            "lower_middle",
            "middle",
            "upper_middle",
            "wealthy",
            "unknown"
        ],
        "age_group_labels": [
            "child_youth",
            "teen",
            "adult",
            "old",
            "unknown"
        ],
        "features": all_features
    }

    save_json(output, OUTPUT_PATH)

    print("\nCompleted.")
    print(f"Characters processed: {len(all_features)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()