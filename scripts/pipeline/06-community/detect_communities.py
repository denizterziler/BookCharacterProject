import json
from pathlib import Path
import networkx as nx
from networkx.algorithms.community import louvain_communities
from networkx.algorithms.community.quality import modularity


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BOOK_ID = "B05"

RANDOM_SEED = 42

GRAPH_PATH = (
    PROJECT_ROOT
    / "results"
    / "graphs"
    / f"{BOOK_ID}_graph.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "communities"
    / f"{BOOK_ID}_louvain_communities.json"
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


def build_networkx_graph(graph_data: dict) -> nx.Graph:
    G = nx.Graph()

    for node in graph_data["nodes"]:
        G.add_node(
            node["id"],
            label=node.get("label", ""),
            canonical_name=node.get("canonical_name", ""),
            aliases=node.get("aliases", []),
            gender=node.get("gender", "unknown"),
            role=node.get("role", ""),
            notes=node.get("notes", ""),

            occupation=node.get("occupation", "unknown"),
            occupation_confidence=node.get("occupation_confidence", 0),

            wealth_status=node.get("wealth_status", "unknown"),
            wealth_status_confidence=node.get("wealth_status_confidence", 0),

            age_group=node.get("age_group", "unknown"),
            age_group_confidence=node.get("age_group_confidence", 0),
        )

    for edge in graph_data["edges"]:
        source = edge["source"]
        target = edge["target"]
        weight = edge.get("weight", 1)

        if weight <= 0:
            continue

        G.add_edge(
            source,
            target,
            weight=weight,
            relation_type=edge.get("relation_type", "not_extracted"),
            relation_direction=edge.get("relation_direction", "unknown"),
            relation_confidence=edge.get("relation_confidence", 0),
        )

    return G


def weighted_degree(G: nx.Graph, node: str) -> float:
    return sum(
        data.get("weight", 1)
        for _, _, data in G.edges(node, data=True)
    )


def summarize_attribute_counts(members: list[str], G: nx.Graph, attribute: str) -> dict:
    counts = {}

    for node in members:
        value = G.nodes[node].get(attribute, "unknown")

        if value is None or value == "":
            value = "unknown"

        counts[value] = counts.get(value, 0) + 1

    return dict(
        sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0])
        )
    )


def summarize_internal_relations(G: nx.Graph, members: list[str]) -> dict:
    subgraph = G.subgraph(members)

    relation_counts = {}

    for _, _, data in subgraph.edges(data=True):
        relation_type = data.get("relation_type", "not_extracted")

        if relation_type is None or relation_type == "":
            relation_type = "not_extracted"

        relation_counts[relation_type] = relation_counts.get(relation_type, 0) + 1

    return dict(
        sorted(
            relation_counts.items(),
            key=lambda item: (-item[1], item[0])
        )
    )


def summarize_community(G: nx.Graph, members: list[str]) -> dict:
    subgraph = G.subgraph(members)

    sorted_members = sorted(
        members,
        key=lambda node: (-weighted_degree(G, node), G.nodes[node].get("label", node))
    )

    member_details = []

    for node in sorted_members:
        attrs = G.nodes[node]

        member_details.append({
            "char_id": node,
            "name": attrs.get("label", ""),
            "weighted_degree": weighted_degree(G, node),

            "gender": attrs.get("gender", "unknown"),
            "role": attrs.get("role", ""),

            "occupation": attrs.get("occupation", "unknown"),
            "occupation_confidence": attrs.get("occupation_confidence", 0),

            "wealth_status": attrs.get("wealth_status", "unknown"),
            "wealth_status_confidence": attrs.get("wealth_status_confidence", 0),

            "age_group": attrs.get("age_group", "unknown"),
            "age_group_confidence": attrs.get("age_group_confidence", 0),
        })

    internal_edge_weight = sum(
        data.get("weight", 1)
        for _, _, data in subgraph.edges(data=True)
    )

    external_edge_weight = 0

    member_set = set(members)

    for node in members:
        for neighbor, edge_data in G[node].items():
            if neighbor not in member_set:
                external_edge_weight += edge_data.get("weight", 1)

    return {
        "size": len(members),
        "internal_edge_count": subgraph.number_of_edges(),
        "internal_edge_weight": internal_edge_weight,
        "external_edge_weight": external_edge_weight,

        "gender_distribution": summarize_attribute_counts(members, G, "gender"),
        "occupation_distribution": summarize_attribute_counts(members, G, "occupation"),
        "wealth_status_distribution": summarize_attribute_counts(members, G, "wealth_status"),
        "age_group_distribution": summarize_attribute_counts(members, G, "age_group"),
        "internal_relation_distribution": summarize_internal_relations(G, members),

        "members": member_details
    }


def detect_louvain_communities(graph_data: dict) -> dict:
    G = build_networkx_graph(graph_data)

    if G.number_of_edges() == 0:
        communities = [{node} for node in G.nodes()]
        modularity_score = 0.0
    else:
        communities = louvain_communities(
            G,
            weight="weight",
            seed=RANDOM_SEED
        )

        modularity_score = modularity(
            G,
            communities,
            weight="weight"
        )

    community_outputs = []

    for index, community in enumerate(communities, start=1):
        members = sorted(list(community))

        summary = summarize_community(G, members)

        community_outputs.append({
            "community_id": f"{BOOK_ID}_COM{index:02d}",
            **summary
        })

    community_outputs = sorted(
        community_outputs,
        key=lambda item: (-item["size"], -item["internal_edge_weight"])
    )

    # Rename after sorting so COM01 is always the largest/strongest community.
    for index, community in enumerate(community_outputs, start=1):
        community["community_id"] = f"{BOOK_ID}_COM{index:02d}"

    output = {
        "book_id": BOOK_ID,
        "title": graph_data.get("title", ""),
        "source_graph_file": str(GRAPH_PATH),
        "algorithm": "networkx_louvain_communities",
        "random_seed": RANDOM_SEED,
        "edge_weight": "cooccurrence weight",
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "community_count": len(community_outputs),
        "modularity": modularity_score,
        "communities": community_outputs
    }

    return output


def main():
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("BOOK_ID:", BOOK_ID)
    print("GRAPH_PATH:", GRAPH_PATH)
    print("OUTPUT_PATH:", OUTPUT_PATH)

    graph_data = load_json(GRAPH_PATH)

    community_data = detect_louvain_communities(graph_data)

    save_json(community_data, OUTPUT_PATH)

    print("\nCompleted.")
    print(f"Book:             {BOOK_ID}")
    print(f"Nodes:            {community_data['node_count']}")
    print(f"Edges:            {community_data['edge_count']}")
    print(f"Communities:      {community_data['community_count']}")
    print(f"Modularity:       {community_data['modularity']:.4f}")
    print(f"Saved to:         {OUTPUT_PATH}")

    print("\nCommunities:")
    for community in community_data["communities"]:
        names = [
            member["name"]
            for member in community["members"]
        ]

        print(
            f"{community['community_id']} | "
            f"size={community['size']} | "
            f"internal_weight={community['internal_edge_weight']} | "
            f"external_weight={community['external_edge_weight']} | "
            f"{', '.join(names)}"
        )


if __name__ == "__main__":
    main()