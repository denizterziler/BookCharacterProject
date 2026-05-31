import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BOOK_ID = "B05"

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

RELATIONS_PATH = (
    PROJECT_ROOT
    / "results"
    / "relations"
    / f"{BOOK_ID}_relations.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "graphs"
    / f"{BOOK_ID}_graph.json"
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


def build_feature_lookup(character_features: dict) -> dict:
    lookup = {}

    for item in character_features.get("features", []):
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


def build_relation_lookup(relations_data: dict) -> dict:
    """
    Returns:
    {
        ("B01_C001", "B01_C002"): {
            "relation_type": "romantic",
            "direction": "undirected",
            "confidence": 85,
            "evidence_segments": [...]
        }
    }

    The pair key is sorted because co-occurrence edges are undirected.
    """
    lookup = {}

    for relation in relations_data.get("relations", []):
        source = relation["source"]
        target = relation["target"]

        pair_key = tuple(sorted([source, target]))

        lookup[pair_key] = {
            "relation_type": relation.get("relation_type", "unknown"),
            "direction": relation.get("direction", "unknown"),
            "relation_confidence": relation.get("confidence", 0),
            "relation_evidence_segments": relation.get("evidence_segments", [])
        }

    return lookup


def build_nodes(
    character_inventory: dict,
    feature_lookup: dict
) -> list[dict]:
    nodes = []

    for character in character_inventory.get("characters", []):
        char_id = character["char_id"]

        features = feature_lookup.get(char_id, {
            "occupation": "unknown",
            "occupation_confidence": 0,
            "wealth_status": "unknown",
            "wealth_status_confidence": 0,
            "age_group": "unknown",
            "age_group_confidence": 0,
        })

        node = {
            "id": char_id,
            "label": character.get("canonical_name", ""),
            "canonical_name": character.get("canonical_name", ""),
            "aliases": character.get("aliases", []),
            "gender": character.get("gender", "unknown"),
            "role": character.get("role", ""),
            "notes": character.get("notes", ""),

            "occupation": features["occupation"],
            "occupation_confidence": features["occupation_confidence"],

            "wealth_status": features["wealth_status"],
            "wealth_status_confidence": features["wealth_status_confidence"],

            "age_group": features["age_group"],
            "age_group_confidence": features["age_group_confidence"],
        }

        nodes.append(node)

    return nodes


def build_edges(
    cooccurrence_data: dict,
    relation_lookup: dict
) -> list[dict]:
    edges = []

    for pair in cooccurrence_data.get("cooccurrences", []):
        source = pair["source"]
        target = pair["target"]

        pair_key = tuple(sorted([source, target]))

        relation_info = relation_lookup.get(pair_key, {
            "relation_type": "not_extracted",
            "direction": "unknown",
            "relation_confidence": 0,
            "relation_evidence_segments": []
        })

        edge = {
            "source": source,
            "source_name": pair.get("source_name", ""),

            "target": target,
            "target_name": pair.get("target_name", ""),

            "edge_type": "undirected",
            "weight": pair.get("weight", 0),
            "cooccurrence_evidence_segments": pair.get("evidence_segments", []),

            "relation_type": relation_info["relation_type"],
            "relation_direction": relation_info["direction"],
            "relation_confidence": relation_info["relation_confidence"],
            "relation_evidence_segments": relation_info["relation_evidence_segments"]
        }

        edges.append(edge)

    edges = sorted(
        edges,
        key=lambda edge: (-edge["weight"], edge["source"], edge["target"])
    )

    return edges


def build_graph(
    character_inventory: dict,
    character_features: dict,
    cooccurrence_data: dict,
    relations_data: dict
) -> dict:
    feature_lookup = build_feature_lookup(character_features)
    relation_lookup = build_relation_lookup(relations_data)

    nodes = build_nodes(
        character_inventory=character_inventory,
        feature_lookup=feature_lookup
    )

    edges = build_edges(
        cooccurrence_data=cooccurrence_data,
        relation_lookup=relation_lookup
    )

    graph = {
        "book_id": BOOK_ID,
        "title": character_inventory.get("title", ""),
        "graph_type": "feature_enriched_character_cooccurrence_graph",
        "unit": "300-word overlapping segment",
        "is_directed": False,
        "description": (
            "This graph represents characters as nodes and segment-level "
            "co-occurrences as undirected weighted edges. Node attributes include "
            "manually provided character metadata and GPT-extracted character "
            "features. Edge weights indicate the number of 300-word overlapping "
            "segments in which the two characters co-occur. Relation labels are "
            "included only for high-frequency co-occurring pairs selected during "
            "relation extraction."
        ),
        "source_files": {
            "character_inventory": str(CHARACTER_INVENTORY_PATH),
            "character_features": str(CHARACTER_FEATURES_PATH),
            "cooccurrences": str(COOCCURRENCE_PATH),
            "relations": str(RELATIONS_PATH)
        },
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges
    }

    return graph


def main():
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("BOOK_ID:", BOOK_ID)
    print("CHARACTER_INVENTORY_PATH:", CHARACTER_INVENTORY_PATH)
    print("CHARACTER_FEATURES_PATH:", CHARACTER_FEATURES_PATH)
    print("COOCCURRENCE_PATH:", COOCCURRENCE_PATH)
    print("RELATIONS_PATH:", RELATIONS_PATH)
    print("OUTPUT_PATH:", OUTPUT_PATH)

    character_inventory = load_json(CHARACTER_INVENTORY_PATH)
    character_features = load_json(CHARACTER_FEATURES_PATH)
    cooccurrence_data = load_json(COOCCURRENCE_PATH)
    relations_data = load_json(RELATIONS_PATH)

    graph = build_graph(
        character_inventory=character_inventory,
        character_features=character_features,
        cooccurrence_data=cooccurrence_data,
        relations_data=relations_data
    )

    save_json(graph, OUTPUT_PATH)

    print("\nCompleted.")
    print(f"Book:       {BOOK_ID}")
    print(f"Nodes:      {graph['node_count']}")
    print(f"Edges:      {graph['edge_count']}")
    print(f"Saved to:   {OUTPUT_PATH}")

    print("\nTop 10 edges:")
    for edge in graph["edges"][:10]:
        print(
            f"{edge['source_name']} — {edge['target_name']} "
            f"| weight={edge['weight']} "
            f"| relation={edge['relation_type']} "
            f"| confidence={edge['relation_confidence']}"
        )


if __name__ == "__main__":
    main()