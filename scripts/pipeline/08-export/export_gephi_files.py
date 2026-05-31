import json
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BOOK_ID = "B01"

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

INTERPRETATIONS_PATH = (
    PROJECT_ROOT
    / "results"
    / "community_interpretations"
    / f"{BOOK_ID}_community_interpretations.json"
)

NODES_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "gephi"
    / f"{BOOK_ID}_nodes.csv"
)

EDGES_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "gephi"
    / f"{BOOK_ID}_edges.csv"
)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_community_lookup(community_data: dict) -> dict:
    lookup = {}

    for community in community_data.get("communities", []):
        community_id = community["community_id"]

        for member in community.get("members", []):
            char_id = member["char_id"]
            lookup[char_id] = community_id

    return lookup


def build_interpretation_lookup(interpretation_data: dict) -> dict:
    lookup = {}

    for item in interpretation_data.get("interpretations", []):
        community_id = item["community_id"]

        lookup[community_id] = {
            "community_label": item.get("community_label", ""),
            "dominant_social_logic": item.get("dominant_social_logic", ""),
            "community_interpretation_confidence": item.get("confidence", 0)
        }

    return lookup


def save_nodes_csv(
    graph_data: dict,
    community_lookup: dict,
    interpretation_lookup: dict,
    output_path: Path
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "Id",
        "Label",
        "CanonicalName",
        "Gender",
        "Role",
        "Occupation",
        "OccupationConfidence",
        "WealthStatus",
        "WealthStatusConfidence",
        "AgeGroup",
        "AgeGroupConfidence",
        "Community",
        "CommunityLabel",
        "DominantSocialLogic",
        "CommunityInterpretationConfidence"
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for node in graph_data.get("nodes", []):
            char_id = node["id"]
            community_id = community_lookup.get(char_id, "")
            interpretation = interpretation_lookup.get(community_id, {})

            writer.writerow({
                "Id": char_id,
                "Label": node.get("label", ""),
                "CanonicalName": node.get("canonical_name", ""),
                "Gender": node.get("gender", "unknown"),
                "Role": node.get("role", ""),
                "Occupation": node.get("occupation", "unknown"),
                "OccupationConfidence": node.get("occupation_confidence", 0),
                "WealthStatus": node.get("wealth_status", "unknown"),
                "WealthStatusConfidence": node.get("wealth_status_confidence", 0),
                "AgeGroup": node.get("age_group", "unknown"),
                "AgeGroupConfidence": node.get("age_group_confidence", 0),
                "Community": community_id,
                "CommunityLabel": interpretation.get("community_label", ""),
                "DominantSocialLogic": interpretation.get("dominant_social_logic", ""),
                "CommunityInterpretationConfidence": interpretation.get(
                    "community_interpretation_confidence",
                    0
                )
            })


def save_edges_csv(graph_data: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "Source",
        "Target",
        "Type",
        "Weight",
        "RelationType",
        "RelationDirection",
        "RelationConfidence"
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for edge in graph_data.get("edges", []):
            writer.writerow({
                "Source": edge["source"],
                "Target": edge["target"],
                "Type": "Undirected",
                "Weight": edge.get("weight", 0),
                "RelationType": edge.get("relation_type", "not_extracted"),
                "RelationDirection": edge.get("relation_direction", "unknown"),
                "RelationConfidence": edge.get("relation_confidence", 0)
            })


def main():
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("BOOK_ID:", BOOK_ID)
    print("GRAPH_PATH:", GRAPH_PATH)
    print("COMMUNITIES_PATH:", COMMUNITIES_PATH)
    print("INTERPRETATIONS_PATH:", INTERPRETATIONS_PATH)
    print("NODES_OUTPUT_PATH:", NODES_OUTPUT_PATH)
    print("EDGES_OUTPUT_PATH:", EDGES_OUTPUT_PATH)

    graph_data = load_json(GRAPH_PATH)
    community_data = load_json(COMMUNITIES_PATH)
    interpretation_data = load_json(INTERPRETATIONS_PATH)

    community_lookup = build_community_lookup(community_data)
    interpretation_lookup = build_interpretation_lookup(interpretation_data)

    save_nodes_csv(
        graph_data=graph_data,
        community_lookup=community_lookup,
        interpretation_lookup=interpretation_lookup,
        output_path=NODES_OUTPUT_PATH
    )

    save_edges_csv(
        graph_data=graph_data,
        output_path=EDGES_OUTPUT_PATH
    )

    print("\nCompleted.")
    print(f"Nodes CSV saved to: {NODES_OUTPUT_PATH}")
    print(f"Edges CSV saved to: {EDGES_OUTPUT_PATH}")


if __name__ == "__main__":
    main()