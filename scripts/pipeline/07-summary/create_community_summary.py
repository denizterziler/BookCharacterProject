import json
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BOOK_IDS = ["B01", "B02", "B03", "B04", "B05"]

GRAPHS_DIR = PROJECT_ROOT / "results" / "graphs"
COMMUNITIES_DIR = PROJECT_ROOT / "results" / "communities"

OUTPUT_JSON_PATH = (
    PROJECT_ROOT
    / "results"
    / "summary"
    / "community_summary.json"
)

OUTPUT_CSV_PATH = (
    PROJECT_ROOT
    / "results"
    / "summary"
    / "community_summary.csv"
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


def calculate_density(node_count: int, edge_count: int) -> float:
    """
    Density for undirected graph:
    2E / N(N-1)
    """
    if node_count <= 1:
        return 0.0

    return (2 * edge_count) / (node_count * (node_count - 1))


def calculate_total_edge_weight(graph_data: dict) -> int:
    return sum(
        edge.get("weight", 0)
        for edge in graph_data.get("edges", [])
    )


def calculate_average_weighted_degree(
    node_count: int,
    total_edge_weight: int
) -> float:
    """
    For undirected graph:
    average weighted degree = 2 * total edge weight / number of nodes
    """
    if node_count == 0:
        return 0.0

    return (2 * total_edge_weight) / node_count


def get_largest_community(community_data: dict) -> dict:
    communities = community_data.get("communities", [])

    if not communities:
        return {
            "community_id": "",
            "size": 0,
            "internal_edge_weight": 0,
            "internal_edge_count": 0,
            "members": []
        }

    return max(
        communities,
        key=lambda community: (
            community.get("size", 0),
            community.get("internal_edge_weight", 0)
        )
    )


def get_largest_community_names(largest_community: dict) -> list[str]:
    names = []

    for member in largest_community.get("members", []):
        name = member.get("name", "")

        if name:
            names.append(name)

    return names


def summarize_book(book_id: str) -> dict:
    graph_path = GRAPHS_DIR / f"{book_id}_graph.json"
    community_path = COMMUNITIES_DIR / f"{book_id}_louvain_communities.json"

    graph_data = load_json(graph_path)
    community_data = load_json(community_path)

    node_count = graph_data.get("node_count", len(graph_data.get("nodes", [])))
    edge_count = graph_data.get("edge_count", len(graph_data.get("edges", [])))

    total_edge_weight = calculate_total_edge_weight(graph_data)
    density = calculate_density(node_count, edge_count)
    average_weighted_degree = calculate_average_weighted_degree(
        node_count=node_count,
        total_edge_weight=total_edge_weight
    )

    largest_community = get_largest_community(community_data)
    largest_community_names = get_largest_community_names(largest_community)

    community_sizes = [
        community.get("size", 0)
        for community in community_data.get("communities", [])
    ]

    internal_weights = [
        community.get("internal_edge_weight", 0)
        for community in community_data.get("communities", [])
    ]

    return {
        "book_id": book_id,
        "title": graph_data.get("title", ""),

        "node_count": node_count,
        "edge_count": edge_count,
        "total_edge_weight": total_edge_weight,
        "density": round(density, 4),
        "average_weighted_degree": round(average_weighted_degree, 4),

        "community_count": community_data.get("community_count", 0),
        "modularity": round(community_data.get("modularity", 0.0), 4),

        "largest_community_id": largest_community.get("community_id", ""),
        "largest_community_size": largest_community.get("size", 0),
        "largest_community_internal_edge_count": largest_community.get("internal_edge_count", 0),
        "largest_community_internal_edge_weight": largest_community.get("internal_edge_weight", 0),
        "largest_community_members": largest_community_names,

        "community_sizes": community_sizes,
        "community_internal_weights": internal_weights,

        "graph_file": str(graph_path),
        "community_file": str(community_path)
    }


def save_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "book_id",
        "title",
        "node_count",
        "edge_count",
        "total_edge_weight",
        "density",
        "average_weighted_degree",
        "community_count",
        "modularity",
        "largest_community_id",
        "largest_community_size",
        "largest_community_internal_edge_count",
        "largest_community_internal_edge_weight",
        "largest_community_members",
        "community_sizes",
        "community_internal_weights",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            csv_row = {}

            for field in fieldnames:
                value = row.get(field, "")

                if isinstance(value, list):
                    value = "; ".join(str(item) for item in value)

                csv_row[field] = value

            writer.writerow(csv_row)


def main():
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("BOOK_IDS:", BOOK_IDS)
    print("OUTPUT_JSON_PATH:", OUTPUT_JSON_PATH)
    print("OUTPUT_CSV_PATH:", OUTPUT_CSV_PATH)

    summaries = []

    for book_id in BOOK_IDS:
        print(f"\nProcessing {book_id}...")

        summary = summarize_book(book_id)
        summaries.append(summary)

        print(
            f"{book_id} | "
            f"nodes={summary['node_count']} | "
            f"edges={summary['edge_count']} | "
            f"communities={summary['community_count']} | "
            f"modularity={summary['modularity']} | "
            f"density={summary['density']}"
        )

    output = {
        "description": (
            "Cross-book summary of feature-enriched character co-occurrence graphs "
            "and Louvain community detection results."
        ),
        "algorithm": "Louvain",
        "edge_weight": "co-occurrence weight",
        "book_count": len(summaries),
        "books": summaries
    }

    save_json(output, OUTPUT_JSON_PATH)
    save_csv(summaries, OUTPUT_CSV_PATH)

    print("\nCompleted.")
    print(f"JSON saved to: {OUTPUT_JSON_PATH}")
    print(f"CSV saved to:  {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()