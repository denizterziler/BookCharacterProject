import json
import csv
from collections import Counter
from pathlib import Path


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent] + list(current.parents):
        if (parent / "results").exists() and (parent / "inputs").exists():
            return parent
    raise RuntimeError("Could not find project root.")


PROJECT_ROOT = find_project_root()

BOOK_ID = "B05"

GRAPH_PATH = PROJECT_ROOT / "results" / "graphs" / f"{BOOK_ID}_graph.json"
COMMUNITIES_PATH = PROJECT_ROOT / "results" / "communities" / f"{BOOK_ID}_louvain_communities.json"

OUTPUT_DIR = PROJECT_ROOT / "results" / "summary"
COMMUNITY_OUTPUT = OUTPUT_DIR / f"{BOOK_ID}_community_attribute_summary.csv"
RELATION_OUTPUT = OUTPUT_DIR / f"{BOOK_ID}_relation_attribute_summary.csv"
RELATION_TYPE_OUTPUT = OUTPUT_DIR / f"{BOOK_ID}_relation_type_summary.csv"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def counter_to_text(counter: Counter) -> str:
    if not counter:
        return ""
    return "; ".join(f"{key}: {value}" for key, value in counter.items())


def get_node_lookup(graph_data: dict) -> dict:
    return {
        node["id"]: node
        for node in graph_data.get("nodes", [])
    }


def get_display_name(node: dict) -> str:
    return (
        node.get("canonical_name")
        or node.get("label")
        or node.get("id")
    )


def generate_community_summary(graph_data: dict, community_data: dict) -> list[dict]:
    node_lookup = get_node_lookup(graph_data)
    rows = []

    for community in community_data.get("communities", []):
        community_id = community["community_id"]
        members = community.get("members", [])

        gender_counter = Counter()
        occupation_counter = Counter()
        age_counter = Counter()
        wealth_counter = Counter()

        member_names = []

        for member in members:
            char_id = member["char_id"]
            node = node_lookup.get(char_id, {})

            member_names.append(get_display_name(node))

            gender_counter[node.get("gender", "unknown")] += 1
            occupation_counter[node.get("occupation", "unknown")] += 1
            age_counter[node.get("age_group", "unknown")] += 1
            wealth_counter[node.get("wealth_status", "unknown")] += 1

        row = {
            "book_id": graph_data.get("book_id", BOOK_ID),
            "community_id": community_id,
            "community_size": len(members),
            "members": "; ".join(member_names),
            "female_count": gender_counter.get("female", 0),
            "male_count": gender_counter.get("male", 0),
            "unknown_gender_count": gender_counter.get("unknown", 0),
            "gender_distribution": counter_to_text(gender_counter),
            "occupation_distribution": counter_to_text(occupation_counter),
            "age_group_distribution": counter_to_text(age_counter),
            "wealth_status_distribution": counter_to_text(wealth_counter),
            "internal_edge_count": community.get("internal_edge_count", ""),
            "internal_edge_weight": community.get("internal_edge_weight", ""),
            "external_edge_weight": community.get("external_edge_weight", "")
        }

        rows.append(row)

    return rows


def generate_relation_summary(graph_data: dict) -> list[dict]:
    node_lookup = get_node_lookup(graph_data)
    rows = []

    for edge in graph_data.get("edges", []):
        relation_type = edge.get("relation_type", "not_extracted")

        if relation_type in ("not_extracted", "unknown", ""):
            continue

        source = edge["source"]
        target = edge["target"]

        source_node = node_lookup.get(source, {})
        target_node = node_lookup.get(target, {})

        rows.append({
            "book_id": graph_data.get("book_id", BOOK_ID),
            "source": source,
            "source_name": get_display_name(source_node),
            "source_gender": source_node.get("gender", "unknown"),
            "source_occupation": source_node.get("occupation", "unknown"),
            "source_age_group": source_node.get("age_group", "unknown"),
            "source_wealth_status": source_node.get("wealth_status", "unknown"),

            "target": target,
            "target_name": get_display_name(target_node),
            "target_gender": target_node.get("gender", "unknown"),
            "target_occupation": target_node.get("occupation", "unknown"),
            "target_age_group": target_node.get("age_group", "unknown"),
            "target_wealth_status": target_node.get("wealth_status", "unknown"),

            "relation_type": relation_type,
            "relation_direction": edge.get("relation_direction", "unknown"),
            "relation_confidence": edge.get("relation_confidence", 0),
            "cooccurrence_weight": edge.get("weight", 0),
            "relation_evidence_segments": "; ".join(edge.get("relation_evidence_segments", []))
        })

    return rows


def generate_relation_type_summary(relation_rows: list[dict]) -> list[dict]:
    grouped = {}

    for row in relation_rows:
        relation_type = row["relation_type"]

        if relation_type not in grouped:
            grouped[relation_type] = {
                "relation_type": relation_type,
                "count": 0,
                "total_weight": 0,
                "female_character_mentions": 0,
                "male_character_mentions": 0,
                "unknown_gender_mentions": 0,
                "mixed_gender_pair_count": 0,
                "female_female_pair_count": 0,
                "male_male_pair_count": 0,
                "occupations": Counter(),
                "age_groups": Counter(),
                "wealth_statuses": Counter()
            }

        g = grouped[relation_type]
        g["count"] += 1
        g["total_weight"] += int(row["cooccurrence_weight"])

        genders = [row["source_gender"], row["target_gender"]]

        for gender in genders:
            if gender == "female":
                g["female_character_mentions"] += 1
            elif gender == "male":
                g["male_character_mentions"] += 1
            else:
                g["unknown_gender_mentions"] += 1

        if genders[0] == "female" and genders[1] == "female":
            g["female_female_pair_count"] += 1
        elif genders[0] == "male" and genders[1] == "male":
            g["male_male_pair_count"] += 1
        elif "unknown" not in genders and genders[0] != genders[1]:
            g["mixed_gender_pair_count"] += 1

        g["occupations"][row["source_occupation"]] += 1
        g["occupations"][row["target_occupation"]] += 1

        g["age_groups"][row["source_age_group"]] += 1
        g["age_groups"][row["target_age_group"]] += 1

        g["wealth_statuses"][row["source_wealth_status"]] += 1
        g["wealth_statuses"][row["target_wealth_status"]] += 1

    output_rows = []

    for relation_type, g in grouped.items():
        count = g["count"]

        output_rows.append({
            "book_id": BOOK_ID,
            "relation_type": relation_type,
            "relation_count": count,
            "average_cooccurrence_weight": round(g["total_weight"] / count, 2) if count else 0,
            "female_character_mentions": g["female_character_mentions"],
            "male_character_mentions": g["male_character_mentions"],
            "unknown_gender_mentions": g["unknown_gender_mentions"],
            "female_female_pair_count": g["female_female_pair_count"],
            "male_male_pair_count": g["male_male_pair_count"],
            "mixed_gender_pair_count": g["mixed_gender_pair_count"],
            "occupation_distribution": counter_to_text(g["occupations"]),
            "age_group_distribution": counter_to_text(g["age_groups"]),
            "wealth_status_distribution": counter_to_text(g["wealth_statuses"])
        })

    return sorted(output_rows, key=lambda x: (-x["relation_count"], x["relation_type"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("GRAPH_PATH:", GRAPH_PATH)
    print("COMMUNITIES_PATH:", COMMUNITIES_PATH)

    graph_data = load_json(GRAPH_PATH)
    community_data = load_json(COMMUNITIES_PATH)

    community_rows = generate_community_summary(graph_data, community_data)
    relation_rows = generate_relation_summary(graph_data)
    relation_type_rows = generate_relation_type_summary(relation_rows)

    write_csv(COMMUNITY_OUTPUT, community_rows)
    write_csv(RELATION_OUTPUT, relation_rows)
    write_csv(RELATION_TYPE_OUTPUT, relation_type_rows)

    print("\nCompleted.")
    print(f"Community summary saved to: {COMMUNITY_OUTPUT}")
    print(f"Relation summary saved to: {RELATION_OUTPUT}")
    print(f"Relation type summary saved to: {RELATION_TYPE_OUTPUT}")

    print("\nQuick summary:")
    print(f"Communities: {len(community_rows)}")
    print(f"Extracted relation edges: {len(relation_rows)}")
    print(f"Relation types: {len(relation_type_rows)}")


if __name__ == "__main__":
    main()