import json
import time
from pathlib import Path
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BOOK_ID = "B05"

MODEL = "gpt-4.1-mini"

MIN_WEIGHT_FLOOR = 10
MIN_RELATIVE_WEIGHT = 0.10

MAX_EVIDENCE_SEGMENTS_PER_PAIR = 10
SLEEP_SECONDS = 0.5


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

CHARACTER_FEATURES_PATH = (
    PROJECT_ROOT
    / "results"
    / "character_features"
    / f"{BOOK_ID}_character_features.json"
)

COOCCURRENCE_PATH = (
    PROJECT_ROOT
    / "results"
    / "cooccurrences"
    / f"{BOOK_ID}_cooccurrences.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "relations"
    / f"{BOOK_ID}_relations.json"
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


def build_character_lookup(character_inventory: dict) -> dict:
    return {
        character["char_id"]: character
        for character in character_inventory["characters"]
    }


def build_feature_lookup(character_features: dict) -> dict:
    lookup = {}

    for item in character_features["features"]:
        char_id = item["char_id"]

        lookup[char_id] = {
            "occupation": item.get("occupation", {}).get("value", "unknown"),
            "occupation_confidence": item.get("occupation", {}).get("confidence", 0),

            "wealth_status": item.get("wealth_status", {}).get("value", "unknown"),
            "wealth_status_confidence": item.get("wealth_status", {}).get("confidence", 0),

            "age_group": item.get("age_group", {}).get("value", "unknown"),
            "age_group_confidence": item.get("age_group", {}).get("confidence", 0),
        }

    return lookup


def select_evidence_segments(segment_ids: list[str], max_segments: int) -> list[str]:
    """
    If a pair co-occurs in many segments, we do not send all segments to GPT.
    We select first, middle, and last evidence segments.
    """
    if len(segment_ids) <= max_segments:
        return segment_ids

    selected = []

    selected.extend(segment_ids[:3])

    middle_start = max(0, len(segment_ids) // 2 - 2)
    selected.extend(segment_ids[middle_start:middle_start + 4])

    selected.extend(segment_ids[-3:])

    seen = set()
    unique_selected = []

    for segment_id in selected:
        if segment_id not in seen:
            unique_selected.append(segment_id)
            seen.add(segment_id)

    return unique_selected[:max_segments]


def calculate_relation_threshold(total_segments: int) -> int:
    return max(
        MIN_WEIGHT_FLOOR,
        round(total_segments * MIN_RELATIVE_WEIGHT)
    )


def create_prompt(
    book_id: str,
    pair: dict,
    source_character: dict,
    target_character: dict,
    source_features: dict,
    target_features: dict,
    evidence_segments: list[dict]
) -> str:
    source_info = {
        "char_id": pair["source"],
        "canonical_name": pair.get("source_name", ""),
        "aliases": source_character.get("aliases", []),
        "gender": source_character.get("gender", ""),
        "role": source_character.get("role", ""),
        "notes": source_character.get("notes", ""),
        "features": source_features
    }

    target_info = {
        "char_id": pair["target"],
        "canonical_name": pair.get("target_name", ""),
        "aliases": target_character.get("aliases", []),
        "gender": target_character.get("gender", ""),
        "role": target_character.get("role", ""),
        "notes": target_character.get("notes", ""),
        "features": target_features
    }

    return f"""
Sen Türkçe edebî metinlerde iki karakter arasındaki ilişki türünü çıkaran dikkatli bir bilgi çıkarımı sistemisin.

Kitap ID: {book_id}

Kaynak karakter:
{json.dumps(source_info, ensure_ascii=False, indent=2)}

Hedef karakter:
{json.dumps(target_info, ensure_ascii=False, indent=2)}

Bu iki karakterin birlikte geçtiği kanıt segmentleri:
{json.dumps(evidence_segments, ensure_ascii=False, indent=2)}

Görev:
Bu iki karakter arasındaki en uygun ilişki türünü belirle.

İzin verilen relation_type değerleri:
- family
- romantic
- friend
- enemy
- servant_employer
- authority_subordinate
- acquaintance
- same_household
- social_circle
- unknown

İzin verilen direction değerleri:
- undirected
- source_to_target
- target_to_source
- unknown

Açıklamalar:
- family: akrabalık, evlilik, ebeveyn-çocuk, kardeşlik gibi aile ilişkileri
- romantic: aşk, sevgili, nişanlılık, evlilik niyeti veya romantik ilgi
- friend: açık arkadaşlık veya dostluk
- enemy: düşmanlık, çatışma, rekabet, nefret
- servant_employer: hizmetçi-efendi, çalışan-işveren, cariye/sahip gibi hizmet ilişkileri
- authority_subordinate: amir-memur, asker-komutan, yönetici-ast gibi otorite ilişkileri
- acquaintance: tanışıklık var ama güçlü ilişki türü belirsiz
- same_household: aynı ev/konak/aile çevresinde bulunma
- social_circle: aynı sosyal çevrede bulunma, fakat ilişki daha özel değil
- unknown: verilen kanıtla ilişki açıkça çıkarılamıyor

Kurallar:
- Sadece verilen karakter bilgilerini ve kanıt segmentlerini kullan.
- Tahmin yapma.
- Kanıt açık değilse relation_type için "unknown" kullan.
- Confidence 0 ile 100 arasında tam sayı olmalıdır.
- relation_type "unknown" ise confidence 0 olmalıdır.
- relation_type "unknown" değilse evidence_segments içinde en az bir segment ID olmalıdır.
- JSON dışında açıklama yazma.
- Sadece geçerli JSON döndür.
- Segmentlerde başka karakterlerle ilgili bilgileri bu iki karaktere ilişki olarak yazma.

Çıktıyı tam olarak şu JSON yapısında döndür:

{{
  "source": "{pair["source"]}",
  "source_name": "{pair.get("source_name", "")}",
  "target": "{pair["target"]}",
  "target_name": "{pair.get("target_name", "")}",
  "cooccurrence_weight": {pair["weight"]},
  "relation_type": "",
  "direction": "",
  "confidence": 0,
  "evidence_segments": []
}}
""".strip()


def call_gpt(prompt: str) -> dict:
    response = client.responses.create(
        model=MODEL,
        input=prompt,
        temperature=0
    )

    return extract_json(response.output_text)


def validate_relation_output(
    result: dict,
    pair: dict,
    valid_segment_ids: set[str]
) -> dict:
    allowed_relations = {
        "family",
        "romantic",
        "friend",
        "enemy",
        "servant_employer",
        "authority_subordinate",
        "acquaintance",
        "same_household",
        "social_circle",
        "unknown"
    }

    allowed_directions = {
        "undirected",
        "source_to_target",
        "target_to_source",
        "unknown"
    }

    relation_type = result.get("relation_type", "unknown")
    direction = result.get("direction", "unknown")
    confidence = result.get("confidence", 0)
    evidence_segments = result.get("evidence_segments", [])

    if relation_type not in allowed_relations:
        relation_type = "unknown"

    if direction not in allowed_directions:
        direction = "unknown"

    if not isinstance(confidence, int):
        try:
            confidence = int(confidence)
        except Exception:
            confidence = 0

    confidence = max(0, min(100, confidence))

    if not isinstance(evidence_segments, list):
        evidence_segments = []

    evidence_segments = [
        segment_id
        for segment_id in evidence_segments
        if isinstance(segment_id, str) and segment_id in valid_segment_ids
    ]

    if relation_type == "unknown":
        confidence = 0
        direction = "unknown"
        evidence_segments = []

    if relation_type != "unknown" and not evidence_segments:
        relation_type = "unknown"
        confidence = 0
        direction = "unknown"
        evidence_segments = []

    return {
        "source": pair["source"],
        "source_name": pair.get("source_name", ""),
        "target": pair["target"],
        "target_name": pair.get("target_name", ""),
        "cooccurrence_weight": pair["weight"],
        "relation_type": relation_type,
        "direction": direction,
        "confidence": confidence,
        "evidence_segments": evidence_segments
    }


def main():
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("BOOK_ID:", BOOK_ID)
    print("MODEL:", MODEL)
    print("MIN_WEIGHT_FLOOR:", MIN_WEIGHT_FLOOR)
    print("MIN_RELATIVE_WEIGHT:", MIN_RELATIVE_WEIGHT)
    print("SEGMENTS_PATH:", SEGMENTS_PATH)
    print("CHARACTER_INVENTORY_PATH:", CHARACTER_INVENTORY_PATH)
    print("CHARACTER_FEATURES_PATH:", CHARACTER_FEATURES_PATH)
    print("COOCCURRENCE_PATH:", COOCCURRENCE_PATH)
    print("OUTPUT_PATH:", OUTPUT_PATH)

    segments_data = load_json(SEGMENTS_PATH)
    character_inventory = load_json(CHARACTER_INVENTORY_PATH)
    character_features = load_json(CHARACTER_FEATURES_PATH)
    cooccurrence_data = load_json(COOCCURRENCE_PATH)

    total_segments = len(segments_data["segments"])
    relation_threshold = calculate_relation_threshold(total_segments)

    print(f"\nTotal segments: {total_segments}")
    print(f"Relation extraction threshold: weight >= {relation_threshold}")

    selected_pairs = [
        pair
        for pair in cooccurrence_data["cooccurrences"]
        if pair["weight"] >= relation_threshold
    ]

    print(f"Total co-occurrence pairs: {len(cooccurrence_data['cooccurrences'])}")
    print(f"Selected pairs for relation extraction: {len(selected_pairs)}")

    segment_lookup = build_segment_lookup(segments_data)
    character_lookup = build_character_lookup(character_inventory)
    feature_lookup = build_feature_lookup(character_features)

    relations = []

    for index, pair in enumerate(selected_pairs, start=1):
        source = pair["source"]
        target = pair["target"]

        print(
            f"\n[{index}/{len(selected_pairs)}] "
            f"{pair.get('source_name', source)} — {pair.get('target_name', target)} "
            f"| weight={pair['weight']}"
        )

        selected_segment_ids = select_evidence_segments(
            pair.get("evidence_segments", []),
            MAX_EVIDENCE_SEGMENTS_PER_PAIR
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

        source_character = character_lookup.get(source, {})
        target_character = character_lookup.get(target, {})

        source_features = feature_lookup.get(source, {})
        target_features = feature_lookup.get(target, {})

        if not evidence_segments:
            relation_result = {
                "source": source,
                "source_name": pair.get("source_name", ""),
                "target": target,
                "target_name": pair.get("target_name", ""),
                "cooccurrence_weight": pair["weight"],
                "relation_type": "unknown",
                "direction": "unknown",
                "confidence": 0,
                "evidence_segments": []
            }

            relations.append(relation_result)
            continue

        prompt = create_prompt(
            book_id=BOOK_ID,
            pair=pair,
            source_character=source_character,
            target_character=target_character,
            source_features=source_features,
            target_features=target_features,
            evidence_segments=evidence_segments
        )

        try:
            raw_result = call_gpt(prompt)

            relation_result = validate_relation_output(
                result=raw_result,
                pair=pair,
                valid_segment_ids=valid_segment_ids
            )

        except Exception as error:
            print(f"ERROR for pair {source} - {target}: {error}")

            relation_result = {
                "source": source,
                "source_name": pair.get("source_name", ""),
                "target": target,
                "target_name": pair.get("target_name", ""),
                "cooccurrence_weight": pair["weight"],
                "relation_type": "unknown",
                "direction": "unknown",
                "confidence": 0,
                "evidence_segments": [],
                "error": str(error)
            }

        relations.append(relation_result)

        time.sleep(SLEEP_SECONDS)

    output = {
        "book_id": BOOK_ID,
        "model": MODEL,
        "source_cooccurrence_file": str(COOCCURRENCE_PATH),
        "source_character_features_file": str(CHARACTER_FEATURES_PATH),
        "unit": "300-word overlapping segment",
        "relation_selection_rule": (
            "Pairs are selected if cooccurrence_weight >= "
            "max(MIN_WEIGHT_FLOOR, round(total_segments * MIN_RELATIVE_WEIGHT))."
        ),
        "total_segments": total_segments,
        "min_weight_floor": MIN_WEIGHT_FLOOR,
        "min_relative_weight": MIN_RELATIVE_WEIGHT,
        "relation_threshold": relation_threshold,
        "total_cooccurrence_pairs": len(cooccurrence_data["cooccurrences"]),
        "selected_pair_count": len(selected_pairs),
        "relation_labels": [
            "family",
            "romantic",
            "friend",
            "enemy",
            "servant_employer",
            "authority_subordinate",
            "acquaintance",
            "same_household",
            "social_circle",
            "unknown"
        ],
        "relations": relations
    }

    save_json(output, OUTPUT_PATH)

    print("\nCompleted.")
    print(f"Book:               {BOOK_ID}")
    print(f"Total segments:     {total_segments}")
    print(f"Threshold:          {relation_threshold}")
    print(f"Relations extracted:{len(relations)}")
    print(f"Saved to:           {OUTPUT_PATH}")


if __name__ == "__main__":
    main()