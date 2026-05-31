import json
from pathlib import Path
from itertools import combinations


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BOOK_ID = "B05"

PRESENCE_PATH = (
    PROJECT_ROOT
    / "predictions"
    / BOOK_ID
    / "ensemble"
    / f"{BOOK_ID}_gpt_alias_ensemble_window_presence.json"
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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "cooccurrences"
    / f"{BOOK_ID}_cooccurrences.json"
)


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


def get_character_name(character_lookup: dict, char_id: str) -> str:
    character = character_lookup.get(char_id, {})
    return character.get("canonical_name", "")


def get_character_features(feature_lookup: dict, char_id: str) -> dict:
    return feature_lookup.get(char_id, {
        "occupation": "unknown",
        "occupation_confidence": 0,
        "wealth_status": "unknown",
        "wealth_status_confidence": 0,
        "age_group": "unknown",
        "age_group_confidence": 0,
    })


def build_cooccurrences(
    presence_data: dict,
    character_inventory: dict,
    character_features: dict
) -> dict:
    character_lookup = build_character_lookup(character_inventory)
    feature_lookup = build_feature_lookup(character_features)

    valid_char_ids = set(character_lookup.keys())

    cooccurrence_map = {}

    for segment in presence_data["segment_presence"]:
        segment_id = segment["segment_id"]

        characters = sorted(set(segment.get("characters_present", [])))

        characters = [
            char_id
            for char_id in characters
            if char_id in valid_char_ids
        ]

        for source, target in combinations(characters, 2):
            source, target = sorted([source, target])
            pair_key = (source, target)

            if pair_key not in cooccurrence_map:
                cooccurrence_map[pair_key] = {
                    "source": source,
                    "source_name": get_character_name(character_lookup, source),
                    "source_features": get_character_features(feature_lookup, source),

                    "target": target,
                    "target_name": get_character_name(character_lookup, target),
                    "target_features": get_character_features(feature_lookup, target),

                    "weight": 0,
                    "evidence_segments": []
                }

            cooccurrence_map[pair_key]["weight"] += 1
            cooccurrence_map[pair_key]["evidence_segments"].append(segment_id)

    cooccurrences = sorted(
        cooccurrence_map.values(),
        key=lambda item: (-item["weight"], item["source"], item["target"])
    )

    return {
        "book_id": BOOK_ID,
        "title": character_inventory.get("title", ""),
        "source_presence_file": str(PRESENCE_PATH),
        "source_character_features_file": str(CHARACTER_FEATURES_PATH),
        "unit": "300-word overlapping segment",
        "definition": (
            "Two characters are considered co-occurring if they are both present "
            "in the same 300-word overlapping segment. The weight equals the "
            "number of segments in which the pair co-occurs."
        ),
        "cooccurrence_count": len(cooccurrences),
        "cooccurrences": cooccurrences
    }


def main():
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("BOOK_ID:", BOOK_ID)
    print("PRESENCE_PATH:", PRESENCE_PATH)
    print("CHARACTER_INVENTORY_PATH:", CHARACTER_INVENTORY_PATH)
    print("CHARACTER_FEATURES_PATH:", CHARACTER_FEATURES_PATH)
    print("OUTPUT_PATH:", OUTPUT_PATH)

    presence_data = load_json(PRESENCE_PATH)
    character_inventory = load_json(CHARACTER_INVENTORY_PATH)
    character_features = load_json(CHARACTER_FEATURES_PATH)

    cooccurrence_data = build_cooccurrences(
        presence_data=presence_data,
        character_inventory=character_inventory,
        character_features=character_features
    )

    save_json(cooccurrence_data, OUTPUT_PATH)

    print("\nCompleted.")
    print(f"Book:             {BOOK_ID}")
    print(f"Co-occurrences:   {cooccurrence_data['cooccurrence_count']}")
    print(f"Saved to:         {OUTPUT_PATH}")

    print("\nTop 10 co-occurrences:")
    for item in cooccurrence_data["cooccurrences"][:10]:
        print(
            f"{item['source_name']} — {item['target_name']} "
            f"| weight={item['weight']}"
        )


if __name__ == "__main__":
    main()