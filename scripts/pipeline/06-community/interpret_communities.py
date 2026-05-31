import json
import time
from pathlib import Path
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BOOK_ID = "B05"

MODEL = "gpt-4.1-mini"
SLEEP_SECONDS = 0.5

GRAPH_PATH = (
    PROJECT_ROOT
    / "results"
    / "graphs"
    / f"{BOOK_ID}_graph.json"
)

COMMUNITIES_PATH = (
    PROJECT_ROOT
    / "results"
    / "communities"
    / f"{BOOK_ID}_louvain_communities.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "community_interpretations"
    / f"{BOOK_ID}_community_interpretations.json"
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

    return json.loads(cleaned[start:end + 1])


def build_edge_lookup(graph_data: dict) -> dict:
    edge_lookup = {}

    for edge in graph_data.get("edges", []):
        source = edge["source"]
        target = edge["target"]
        key = tuple(sorted([source, target]))
        edge_lookup[key] = edge

    return edge_lookup


def get_internal_edges_for_community(
    community: dict,
    edge_lookup: dict
) -> list[dict]:
    member_ids = [
        member["char_id"]
        for member in community.get("members", [])
    ]

    member_set = set(member_ids)

    internal_edges = []

    for (source, target), edge in edge_lookup.items():
        if source in member_set and target in member_set:
            internal_edges.append({
                "source": source,
                "source_name": edge.get("source_name", ""),
                "target": target,
                "target_name": edge.get("target_name", ""),
                "weight": edge.get("weight", 0),
                "relation_type": edge.get("relation_type", "not_extracted"),
                "relation_confidence": edge.get("relation_confidence", 0)
            })

    internal_edges = sorted(
        internal_edges,
        key=lambda item: (-item["weight"], item["source_name"], item["target_name"])
    )

    return internal_edges


def create_prompt(
    book_id: str,
    title: str,
    community: dict,
    internal_edges: list[dict]
) -> str:
    community_input = {
        "community_id": community["community_id"],
        "size": community.get("size", 0),
        "internal_edge_weight": community.get("internal_edge_weight", 0),
        "external_edge_weight": community.get("external_edge_weight", 0),
        "gender_distribution": community.get("gender_distribution", {}),
        "occupation_distribution": community.get("occupation_distribution", {}),
        "wealth_status_distribution": community.get("wealth_status_distribution", {}),
        "age_group_distribution": community.get("age_group_distribution", {}),
        "internal_relation_distribution": community.get("internal_relation_distribution", {}),
        "members": community.get("members", []),
        "strong_internal_edges": internal_edges[:15]
    }

    return f"""
Sen Türkçe roman karakter ağlarını yorumlayan dikkatli bir edebî analiz asistanısın.

Kitap ID: {book_id}
Kitap başlığı: {title}

Aşağıda Louvain algoritmasıyla tespit edilmiş bir karakter topluluğu verilmiştir.

Topluluk verisi:
{json.dumps(community_input, ensure_ascii=False, indent=2)}

Görev:
Bu topluluğun romandaki muhtemel anlatısal/sosyal işlevini yorumla.

Önemli:
- Toplulukları sen belirlemiyorsun. Topluluklar Louvain algoritmasıyla zaten tespit edildi.
- Senin görevin sadece verilen topluluğu açıklamak ve adlandırmak.
- Sadece verilen üyeler, özellikler, edge weight değerleri ve ilişki türlerini kullan.
- Uydurma olay veya ilişki ekleme.
- Kanıt yetersizse bunu belirt.
- Çok iddialı yorum yapma.
- JSON dışında açıklama yazma.

community_label:
- Kısa ve açıklayıcı olmalı.
- Örnekler:
  - "ana romantik çatışma çevresi"
  - "aile ve hane çevresi"
  - "hizmet/otorite ilişkileri çevresi"
  - "periferik sosyal çevre"
  - "belirsiz karakter kümesi"

dominant_social_logic için örnekler:
- family_household
- romantic_conflict
- class_or_service_relation
- authority_relation
- friendship_or_social_circle
- mixed_or_unclear

Çıktıyı tam olarak şu JSON yapısında döndür:

{{
  "community_id": "{community["community_id"]}",
  "community_label": "",
  "dominant_social_logic": "",
  "interpretation": "",
  "dominant_character_features": [],
  "important_characters": [],
  "important_relations": [],
  "uncertainties": [],
  "confidence": 0
}}
""".strip()


def call_gpt(prompt: str) -> dict:
    response = client.responses.create(
        model=MODEL,
        input=prompt,
        temperature=0
    )

    return extract_json(response.output_text)


def validate_interpretation_output(
    result: dict,
    community: dict
) -> dict:
    allowed_social_logic = {
        "family_household",
        "romantic_conflict",
        "class_or_service_relation",
        "authority_relation",
        "friendship_or_social_circle",
        "mixed_or_unclear"
    }

    community_id = community["community_id"]

    community_label = result.get("community_label", "")
    dominant_social_logic = result.get("dominant_social_logic", "mixed_or_unclear")
    interpretation = result.get("interpretation", "")
    dominant_character_features = result.get("dominant_character_features", [])
    important_characters = result.get("important_characters", [])
    important_relations = result.get("important_relations", [])
    uncertainties = result.get("uncertainties", [])
    confidence = result.get("confidence", 0)

    if not isinstance(community_label, str) or not community_label.strip():
        community_label = "belirsiz karakter kümesi"

    if dominant_social_logic not in allowed_social_logic:
        dominant_social_logic = "mixed_or_unclear"

    if not isinstance(interpretation, str):
        interpretation = ""

    for field_name, value in [
        ("dominant_character_features", dominant_character_features),
        ("important_characters", important_characters),
        ("important_relations", important_relations),
        ("uncertainties", uncertainties),
    ]:
        if not isinstance(value, list):
            if field_name == "dominant_character_features":
                dominant_character_features = []
            elif field_name == "important_characters":
                important_characters = []
            elif field_name == "important_relations":
                important_relations = []
            elif field_name == "uncertainties":
                uncertainties = []

    if not isinstance(confidence, int):
        try:
            confidence = int(confidence)
        except Exception:
            confidence = 0

    confidence = max(0, min(100, confidence))

    return {
        "community_id": community_id,
        "community_label": community_label.strip(),
        "dominant_social_logic": dominant_social_logic,
        "interpretation": interpretation.strip(),
        "dominant_character_features": dominant_character_features,
        "important_characters": important_characters,
        "important_relations": important_relations,
        "uncertainties": uncertainties,
        "confidence": confidence
    }


def main():
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("BOOK_ID:", BOOK_ID)
    print("MODEL:", MODEL)
    print("GRAPH_PATH:", GRAPH_PATH)
    print("COMMUNITIES_PATH:", COMMUNITIES_PATH)
    print("OUTPUT_PATH:", OUTPUT_PATH)

    graph_data = load_json(GRAPH_PATH)
    community_data = load_json(COMMUNITIES_PATH)

    edge_lookup = build_edge_lookup(graph_data)

    interpretations = []

    for index, community in enumerate(community_data.get("communities", []), start=1):
        print(
            f"\n[{index}/{community_data.get('community_count', 0)}] "
            f"Interpreting {community['community_id']}"
        )

        internal_edges = get_internal_edges_for_community(
            community=community,
            edge_lookup=edge_lookup
        )

        prompt = create_prompt(
            book_id=BOOK_ID,
            title=graph_data.get("title", ""),
            community=community,
            internal_edges=internal_edges
        )

        try:
            raw_result = call_gpt(prompt)

            interpretation = validate_interpretation_output(
                result=raw_result,
                community=community
            )

        except Exception as error:
            print(f"ERROR for {community['community_id']}: {error}")

            interpretation = {
                "community_id": community["community_id"],
                "community_label": "belirsiz karakter kümesi",
                "dominant_social_logic": "mixed_or_unclear",
                "interpretation": "",
                "dominant_character_features": [],
                "important_characters": [],
                "important_relations": [],
                "uncertainties": [str(error)],
                "confidence": 0
            }

        interpretations.append(interpretation)

        time.sleep(SLEEP_SECONDS)

    output = {
        "book_id": BOOK_ID,
        "title": graph_data.get("title", ""),
        "model": MODEL,
        "source_graph_file": str(GRAPH_PATH),
        "source_communities_file": str(COMMUNITIES_PATH),
        "description": (
            "GPT-based interpretation of Louvain-detected character communities. "
            "GPT is used only to interpret communities, not to detect them."
        ),
        "interpretations": interpretations
    }

    save_json(output, OUTPUT_PATH)

    print("\nCompleted.")
    print(f"Book: {BOOK_ID}")
    print(f"Communities interpreted: {len(interpretations)}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nCommunity labels:")
    for item in interpretations:
        print(
            f"{item['community_id']} | "
            f"{item['community_label']} | "
            f"{item['dominant_social_logic']} | "
            f"confidence={item['confidence']}"
        )


if __name__ == "__main__":
    main()