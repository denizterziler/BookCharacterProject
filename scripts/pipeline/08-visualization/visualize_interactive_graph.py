import json
from pathlib import Path
from pyvis.network import Network


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BOOK_ID = "B01"

GRAPH_PATH = (
    PROJECT_ROOT / "results" / "graphs" / f"{BOOK_ID}_graph.json"
)

COMMUNITIES_PATH = (
    PROJECT_ROOT / "results" / "communities" / f"{BOOK_ID}_louvain_communities.json"
)

INTERPRETATIONS_PATH = (
    PROJECT_ROOT / "results" / "community_interpretations" / f"{BOOK_ID}_community_interpretations.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "visualizations" / f"{BOOK_ID}_interactive_graph.html"
)


COMMUNITY_COLORS = [
    "#4E9AF1",
    "#F4845F",
    "#56C596",
    "#F7C948",
    "#C47BE5",
    "#F06292",
    "#4DD0E1",
    "#FF8A65",
    "#A5D6A7",
    "#CE93D8",
]


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def get_display_name(node: dict) -> str:
    """
    Safely chooses the best display name for a character.
    Priority:
    1. canonical_name
    2. label
    3. first alias
    4. character ID
    """
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


def build_interpretation_lookup(interpretation_data: dict) -> dict:
    lookup = {}

    for item in interpretation_data.get("interpretations", []):
        lookup[item["community_id"]] = {
            "community_label": item.get("community_label", ""),
            "dominant_social_logic": item.get("dominant_social_logic", ""),
            "confidence": item.get("confidence", 0)
        }

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


def get_community_color(community_id: str, color_map: dict) -> str:
    if community_id not in color_map:
        idx = len(color_map) % len(COMMUNITY_COLORS)
        color_map[community_id] = COMMUNITY_COLORS[idx]

    return color_map[community_id]


def build_legend_html(color_map: dict, interpretation_lookup: dict) -> str:
    items = ""

    for comm_id, color in color_map.items():
        if str(comm_id).startswith("ANCHOR_"):
            continue

        info = interpretation_lookup.get(comm_id, {})
        label = info.get("community_label") or str(comm_id)
        logic = info.get("dominant_social_logic", "")
        tooltip = f'title="{logic}"' if logic else ""

        items += f"""
        <div class="legend-item" {tooltip}>
          <span class="legend-dot" style="background:{color}"></span>
          <span class="legend-label">{label}</span>
        </div>
        """

    return items


def main():
    print("GRAPH_PATH:", GRAPH_PATH)
    print("COMMUNITIES_PATH:", COMMUNITIES_PATH)
    print("INTERPRETATIONS_PATH:", INTERPRETATIONS_PATH)
    print("OUTPUT_PATH:", OUTPUT_PATH)

    graph_data = load_json(GRAPH_PATH)
    community_data = load_json(COMMUNITIES_PATH)
    interpretation_data = load_json(INTERPRETATIONS_PATH)

    community_lookup = build_community_lookup(community_data)
    interpretation_lookup = build_interpretation_lookup(interpretation_data)
    node_weights = calculate_node_weights(graph_data)

    node_name_lookup = {
        node["id"]: get_display_name(node)
        for node in graph_data.get("nodes", [])
    }

    net = Network(
        height="100vh",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="#ffffff",
        notebook=False
    )

    # Slightly less spread than before, because community anchors now help clustering.
    net.force_atlas_2based(
        gravity=-80,
        central_gravity=0.006,
        spring_length=220,
        spring_strength=0.04,
        damping=0.55,
        overlap=1
    )

    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "stabilization": {
          "enabled": true,
          "iterations": 1200,
          "updateInterval": 25,
          "fit": true
        },
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "centralGravity": 0.006,
          "springLength": 220,
          "springConstant": 0.04,
          "damping": 0.55,
          "avoidOverlap": 1
        }
      },
      "nodes": {
        "shape": "dot"
      },
      "edges": {
        "smooth": {
          "enabled": true,
          "type": "dynamic",
          "roundness": 0.35
        },
        "color": {
          "inherit": false
        },
        "scaling": {
          "min": 1,
          "max": 7
        },
        "selectionWidth": 2,
        "hoverWidth": 2
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "hideEdgesOnDrag": false,
        "navigationButtons": true
      }
    }
    """)

    color_map = {}

    max_weighted_degree = max(node_weights.values()) if node_weights else 1

    # ------------------------------------------------------------
    # Add community anchor nodes first.
    # These are visual-only helper nodes that pull same-community
    # characters closer together. They do not affect your analysis.
    # ------------------------------------------------------------
    community_ids = sorted(set(community_lookup.values()))

    for community_id in community_ids:
        color = get_community_color(community_id, color_map)
        community_info = interpretation_lookup.get(community_id, {})
        community_label = community_info.get("community_label", community_id)

        net.add_node(
            f"ANCHOR_{community_id}",
            label="",
            title=f"Community anchor: {community_label}",
            size=2,
            color={
                "background": color,
                "border": color
            },
            physics=True,
            hidden=False
        )

    # Add character nodes
    for node in graph_data.get("nodes", []):
        char_id = node["id"]
        label = get_display_name(node)

        community_id = community_lookup.get(char_id, "unknown")
        community_info = interpretation_lookup.get(community_id, {})
        color = get_community_color(community_id, color_map)

        weighted_degree = node_weights.get(char_id, 0)

        if max_weighted_degree > 0:
            normalized = weighted_degree / max_weighted_degree
        else:
            normalized = 0

        size = 8 + (normalized * 20)

        # Show canonical names for all nodes
        show_label = label

        details_html = (
            f"<div style='font-family:sans-serif;font-size:13px;min-width:180px;padding:4px'>"
            f"<b style='font-size:15px'>{label}</b><br><br>"
            f"<b>Character ID:</b> {char_id}<br>"
            f"<b>Community:</b> {community_info.get('community_label', community_id)}<br>"
            f"<b>Social logic:</b> {community_info.get('dominant_social_logic', '—')}<br>"
            f"<b>Gender:</b> {node.get('gender', '—')}<br>"
            f"<b>Role:</b> {node.get('role', '—')}<br>"
            f"<b>Occupation:</b> {node.get('occupation', '—')} "
            f"({node.get('occupation_confidence', 0)}%)<br>"
            f"<b>Wealth:</b> {node.get('wealth_status', '—')} "
            f"({node.get('wealth_status_confidence', 0)}%)<br>"
            f"<b>Age group:</b> {node.get('age_group', '—')} "
            f"({node.get('age_group_confidence', 0)}%)<br>"
            f"<b>Weighted degree:</b> {weighted_degree}"
            f"</div>"
        )

        net.add_node(
            char_id,
            label=show_label,
            title=label,
            details=details_html,
            color={
                "background": color,
                "border": "#ffffff",
                "highlight": {
                    "background": "#ffffff",
                    "border": color
                },
                "hover": {
                    "background": "#ffffff",
                    "border": color
                },
            },
            size=size,
            font={
                "size": 11,
                "color": "#ffffff",
                "strokeWidth": 3,
                "strokeColor": "#1a1a2e",
                "bold": True,
            },
            borderWidth=1,
            borderWidthSelected=2,
        )

        # Visual-only anchor edge.
        # This pulls the character toward its community center.
        # It is almost invisible but still participates in physics.
        net.add_edge(
            f"ANCHOR_{community_id}",
            char_id,
            value=1,
            width=0.1,
            title="",
            color="rgba(255,255,255,0.01)",
            label="",
            physics=True
        )

    max_edge_weight = max(
        [edge.get("weight", 1) for edge in graph_data.get("edges", [])],
        default=1
    )

    # Add real graph edges
    for edge in graph_data.get("edges", []):
        source = edge["source"]
        target = edge["target"]
        weight = edge.get("weight", 1)

        source_name = edge.get("source_name") or node_name_lookup.get(source, source)
        target_name = edge.get("target_name") or node_name_lookup.get(target, target)

        relation_type = edge.get("relation_type", "not_extracted")
        relation_confidence = edge.get("relation_confidence", 0)

        if max_edge_weight > 0:
            normalized_weight = weight / max_edge_weight
        else:
            normalized_weight = 0

        # Thinner edges reduce overlap.
        edge_width = 0.6 + (normalized_weight * 3.2)

        alpha = 0.15 + (normalized_weight * 0.55)

        # Extracted-relation edges are slightly warmer.
        if relation_type not in ("not_extracted", "unknown", ""):
            edge_color = f"rgba(255,220,120,{min(alpha + 0.20, 0.95)})"
        else:
            edge_color = f"rgba(255,255,255,{alpha})"

        edge_hover_text = (
            f"{source_name} — {target_name} | "
            f"weight={weight} | relation={relation_type} | "
            f"confidence={relation_confidence}%"
        )
        show_relation_label = (
            relation_type
            if relation_type not in ("not_extracted", "unknown", "")
               and relation_confidence >= 85
               and weight >= 15
            else ""
        )
        net.add_edge(
            source,
            target,
            value=edge_width,
            width=edge_width,
            title=edge_hover_text,
            color=edge_color,
            label=show_relation_label,
            font={
                "size": 8,
                "color": "#dddddd",
                "strokeWidth": 2,
                "strokeColor": "#1a1a2e"
            },
            physics=True
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(OUTPUT_PATH), notebook=False)

    legend_items = build_legend_html(color_map, interpretation_lookup)

    injection = f"""
    <style>
      body {{
        margin: 0;
        overflow: hidden;
        background: #1a1a2e;
      }}

      #legend {{
        position: fixed;
        top: 16px;
        left: 16px;
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 10px;
        padding: 14px 16px;
        font-family: sans-serif;
        color: #ffffff;
        font-size: 13px;
        z-index: 999;
        backdrop-filter: blur(6px);
        max-width: 260px;
      }}

      #legend h3 {{
        margin: 0 0 10px 0;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #aaaaaa;
      }}

      .legend-item {{
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
        cursor: default;
      }}

      .legend-dot {{
        width: 12px;
        height: 12px;
        border-radius: 50%;
        flex-shrink: 0;
        border: 1.5px solid rgba(255,255,255,0.3);
      }}

      .legend-label {{
        line-height: 1.3;
        font-size: 12px;
      }}

      #info-panel {{
        position: fixed;
        bottom: 20px;
        right: 16px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 10px;
        padding: 16px 18px;
        font-family: sans-serif;
        color: #ffffff;
        font-size: 13px;
        z-index: 999;
        backdrop-filter: blur(6px);
        width: 280px;
        display: none;
        line-height: 1.7;
      }}

      #info-panel b {{
        color: #aaaaaa;
        font-weight: normal;
      }}

      #info-panel .name {{
        font-size: 16px;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 8px;
      }}

      #info-panel .close-btn {{
        position: absolute;
        top: 10px;
        right: 12px;
        cursor: pointer;
        color: #aaaaaa;
        font-size: 16px;
        line-height: 1;
      }}

      #hint {{
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 20px;
        padding: 6px 16px;
        font-family: sans-serif;
        font-size: 12px;
        color: #bbbbbb;
        pointer-events: none;
        z-index: 998;
      }}

      #export-png-btn {{
        position: fixed;
        top: 16px;
        right: 16px;
        z-index: 1000;
        background: rgba(255,255,255,0.12);
        color: white;
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 8px;
        padding: 8px 14px;
        font-family: sans-serif;
        font-size: 13px;
        cursor: pointer;
        backdrop-filter: blur(6px);
      }}

      #export-png-btn:hover {{
        background: rgba(255,255,255,0.22);
      }}
    </style>

    <div id="legend">
      <h3>Communities</h3>
      {legend_items}
    </div>

    <div id="info-panel">
      <span class="close-btn" onclick="document.getElementById('info-panel').style.display='none'">✕</span>
      <div class="name" id="ip-name"></div>
      <div id="ip-body"></div>
    </div>

    <div id="hint">Click a node for details · Scroll to zoom · Drag to pan</div>

    <button id="export-png-btn">Export PNG</button>

    <script>
      document.addEventListener("DOMContentLoaded", function () {{
        var checkNetwork = setInterval(function () {{
          if (typeof network !== "undefined") {{
            clearInterval(checkNetwork);

            network.once("stabilizationIterationsDone", function () {{
              network.setOptions({{ physics: {{ enabled: false }} }});
            }});

            var exportButton = document.getElementById("export-png-btn");

            exportButton.addEventListener("click", function () {{
              network.fit({{
                animation: false
              }});

              setTimeout(function () {{
                var canvas = document.querySelector("#mynetwork canvas");

                if (!canvas) {{
                  alert("Canvas not found.");
                  return;
                }}

                var image = canvas.toDataURL("image/png");

                var link = document.createElement("a");
                link.href = image;
                link.download = "{BOOK_ID}_character_cooccurrence_graph.png";
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
              }}, 500);
            }});

            network.on("click", function (params) {{
              if (params.nodes.length > 0) {{
                var nodeId = params.nodes[0];

                if (String(nodeId).startsWith("ANCHOR_")) {{
                  return;
                }}

                var nodeData = network.body.data.nodes.get(nodeId);
                var panel = document.getElementById("info-panel");

                var rawDetails = nodeData.details || "";
                var tmp = document.createElement("div");
                tmp.innerHTML = rawDetails;

                var inner = tmp.querySelector("div")
                  ? tmp.querySelector("div").innerHTML
                  : rawDetails;

                var nameMatch = inner.match(/<b[^>]*>(.*?)<\\/b>/);
                var name = nameMatch ? nameMatch[1] : (nodeData.title || nodeId);

                var body = inner.replace(/<b[^>]*>.*?<\\/b><br><br>/, "");

                document.getElementById("ip-name").textContent = name;
                document.getElementById("ip-body").innerHTML = body;

                panel.style.display = "block";
              }} else {{
                document.getElementById("info-panel").style.display = "none";
              }}
            }});
          }}
        }}, 100);
      }});
    </script>
    """

    html = OUTPUT_PATH.read_text(encoding="utf-8")
    html = html.replace("</body>", injection + "\n</body>")
    OUTPUT_PATH.write_text(html, encoding="utf-8")

    print("\nCompleted.")
    print(f"Interactive graph saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()