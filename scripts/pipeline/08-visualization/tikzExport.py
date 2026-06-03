import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BOOK_ID = "B04"

GRAPH_PATH = (
    PROJECT_ROOT / "results" / "graphs" / f"{BOOK_ID}_graph.json"
)

COMMUNITIES_PATH = (
    PROJECT_ROOT / "results" / "communities" / f"{BOOK_ID}_louvain_communities.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "visualizations" / f"{BOOK_ID}_graph_tikz.tex"
)


COMMUNITY_COLORS = [
    "commblue",
    "commorange",
    "commgreen",
    "commyellow",
    "commpurple",
    "commpink",
    "commcyan",
    "commred",
]


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def latex_escape(text):
    if text is None:
        return ""

    text = str(text)

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def safe_tikz_id(char_id: str) -> str:
    """
    TikZ node/coordinate names should be simple.
    This keeps IDs safe by replacing problematic characters.
    """
    safe = str(char_id)
    safe = safe.replace("-", "_")
    safe = safe.replace(" ", "_")
    safe = safe.replace(".", "_")
    safe = safe.replace(":", "_")
    return safe


def get_display_name(node: dict) -> str:
    canonical_name = str(node.get("canonical_name", "")).strip()
    label = str(node.get("label", "")).strip()
    aliases = node.get("aliases", [])

    if canonical_name:
        return canonical_name

    if label:
        return label

    if isinstance(aliases, list) and aliases:
        first_alias = str(aliases[0]).strip()
        if first_alias:
            return first_alias

    return node["id"]


def build_community_lookup(community_data: dict) -> dict:
    lookup = {}

    for community in community_data.get("communities", []):
        community_id = community["community_id"]

        for member in community.get("members", []):
            lookup[member["char_id"]] = community_id

    return lookup


def calculate_node_weights(graph_data: dict) -> dict:
    weights = {}

    for node in graph_data.get("nodes", []):
        weights[node["id"]] = 0

    for edge in graph_data.get("edges", []):
        source = edge["source"]
        target = edge["target"]
        weight = edge.get("weight", 1)

        weights[source] = weights.get(source, 0) + weight
        weights[target] = weights.get(target, 0) + weight

    return weights


def make_circular_layout(graph_data: dict, community_lookup: dict) -> dict:
    """
    Deterministic layout:
    - communities are placed around a large circle
    - nodes inside each community are placed around smaller circles
    """
    communities = {}

    for node in graph_data.get("nodes", []):
        char_id = node["id"]
        community_id = community_lookup.get(char_id, "unknown")
        communities.setdefault(community_id, []).append(char_id)

    community_ids = sorted(communities.keys(), key=str)

    positions = {}

    outer_radius = 5.0
    inner_radius_base = 1.2

    for i, community_id in enumerate(community_ids):
        angle = 2 * math.pi * i / max(len(community_ids), 1)

        center_x = outer_radius * math.cos(angle)
        center_y = outer_radius * math.sin(angle)

        members = sorted(communities[community_id])
        inner_radius = inner_radius_base + 0.12 * len(members)

        for j, char_id in enumerate(members):
            node_angle = 2 * math.pi * j / max(len(members), 1)

            x = center_x + inner_radius * math.cos(node_angle)
            y = center_y + inner_radius * math.sin(node_angle)

            positions[char_id] = (x, y)

    return positions


def export_tikz(graph_data: dict, community_lookup: dict, output_path: Path):
    node_weights = calculate_node_weights(graph_data)
    positions = make_circular_layout(graph_data, community_lookup)

    max_node_weight = max(node_weights.values(), default=1)

    edges = graph_data.get("edges", [])
    max_edge_weight = max([edge.get("weight", 1) for edge in edges], default=1)

    node_lookup = {
        node["id"]: node
        for node in graph_data.get("nodes", [])
    }

    lines = []

    lines.append(r"\begin{tikzpicture}[")
    lines.append(r"    every node/.style={circle, draw=black!60, align=center, font=\scriptsize},")
    lines.append(r"    every path/.style={draw=gray!55}")
    lines.append(r"]")
    lines.append("")

    # ------------------------------------------------------------
    # Color definitions
    # ------------------------------------------------------------
    lines.append(r"\definecolor{commblue}{HTML}{4E9AF1}")
    lines.append(r"\definecolor{commorange}{HTML}{F4845F}")
    lines.append(r"\definecolor{commgreen}{HTML}{56C596}")
    lines.append(r"\definecolor{commyellow}{HTML}{F7C948}")
    lines.append(r"\definecolor{commpurple}{HTML}{C47BE5}")
    lines.append(r"\definecolor{commpink}{HTML}{F06292}")
    lines.append(r"\definecolor{commcyan}{HTML}{4DD0E1}")
    lines.append(r"\definecolor{commred}{HTML}{FF8A65}")
    lines.append("")

    community_ids = sorted(set(community_lookup.values()), key=str)
    community_to_color = {
        community_id: COMMUNITY_COLORS[i % len(COMMUNITY_COLORS)]
        for i, community_id in enumerate(community_ids)
    }

    # ------------------------------------------------------------
    # 1. Define coordinates first.
    # This prevents: "No shape named `B05_C001' is known."
    # ------------------------------------------------------------
    lines.append("% Coordinates")

    for char_id in node_lookup.keys():
        if char_id not in positions:
            continue

        x, y = positions[char_id]
        safe_id = safe_tikz_id(char_id)

        lines.append(
            rf"\coordinate (pos_{safe_id}) at ({x:.2f},{y:.2f});"
        )

    lines.append("")

    # ------------------------------------------------------------
    # 2. Draw edges using coordinates.
    # Edges are drawn before nodes so they stay behind nodes.
    # ------------------------------------------------------------
    lines.append("% Edges")

    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        weight = edge.get("weight", 1)
        relation_type = edge.get("relation_type", "")

        if source not in positions or target not in positions:
            continue

        safe_source = safe_tikz_id(source)
        safe_target = safe_tikz_id(target)

        line_width = 0.2 + 2.2 * (weight / max_edge_weight)

        if relation_type not in ("", "unknown", "not_extracted"):
            color = "orange!70!black"
        else:
            color = "gray!45"

        lines.append(
            rf"\draw[{color}, line width={line_width:.2f}pt] "
            rf"(pos_{safe_source}) -- (pos_{safe_target});"
        )

    lines.append("")

    # ------------------------------------------------------------
    # 3. Draw nodes using the same coordinates.
    # ------------------------------------------------------------
    lines.append("% Nodes")

    for char_id, node in node_lookup.items():
        if char_id not in positions:
            continue

        safe_id = safe_tikz_id(char_id)

        community_id = community_lookup.get(char_id, "unknown")
        color = community_to_color.get(community_id, "gray")

        weighted_degree = node_weights.get(char_id, 0)

        if max_node_weight > 0:
            size = 0.45 + 0.9 * math.sqrt(weighted_degree / max_node_weight)
        else:
            size = 0.45

        label = latex_escape(get_display_name(node))

        lines.append(
            rf"\node[fill={color}!75, minimum size={size:.2f}cm] "
            rf"({safe_id}) at (pos_{safe_id}) {{{label}}};"
        )

    lines.append("")
    lines.append(r"\end{tikzpicture}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    print("GRAPH_PATH:", GRAPH_PATH)
    print("COMMUNITIES_PATH:", COMMUNITIES_PATH)
    print("OUTPUT_PATH:", OUTPUT_PATH)

    graph_data = load_json(GRAPH_PATH)
    community_data = load_json(COMMUNITIES_PATH)

    community_lookup = build_community_lookup(community_data)

    export_tikz(
        graph_data=graph_data,
        community_lookup=community_lookup,
        output_path=OUTPUT_PATH
    )

    print("\nCompleted.")
    print(f"TikZ graph saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()