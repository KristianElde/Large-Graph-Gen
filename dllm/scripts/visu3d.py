from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import plotly.graph_objects as go


def load_graph(path: Path) -> nx.Graph:
    data = json.loads(path.read_text())
    G = nx.Graph()
    G.add_nodes_from(range(int(data["num_nodes"])))
    G.add_edges_from((int(u), int(v)) for u, v in data["edges"])
    return G


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_json", type=Path, required=True)
    parser.add_argument("--out_html", type=Path, required=True)
    parser.add_argument("--mode", choices=["largest", "full"], default="largest")
    parser.add_argument("--max_nodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    G = load_graph(args.graph_json)

    if args.mode == "largest":
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        nodes = list(components[0])
        if len(nodes) > args.max_nodes:
            # Keep highest-degree nodes from largest component.
            nodes = sorted(nodes, key=lambda n: G.degree[n], reverse=True)[: args.max_nodes]
        H = G.subgraph(nodes).copy()
    else:
        nodes = list(G.nodes())
        if len(nodes) > args.max_nodes:
            # Keep highest-degree nodes for readability.
            nodes = sorted(nodes, key=lambda n: G.degree[n], reverse=True)[: args.max_nodes]
        H = G.subgraph(nodes).copy()

    print("Full graph:")
    print("  nodes:", G.number_of_nodes())
    print("  edges:", G.number_of_edges())
    print("  components:", nx.number_connected_components(G))
    print("  isolates:", len(list(nx.isolates(G))))

    print("\nPlotted graph:")
    print("  nodes:", H.number_of_nodes())
    print("  edges:", H.number_of_edges())
    print("  components:", nx.number_connected_components(H))

    pos = nx.spring_layout(H, dim=3, seed=args.seed, iterations=80)

    edge_x, edge_y, edge_z = [], [], []
    for u, v in H.edges():
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    node_x, node_y, node_z = [], [], []
    degrees = []

    for n in H.nodes():
        x, y, z = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        degrees.append(H.degree[n])

    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(width=1),
        hoverinfo="none",
    )

    node_trace = go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode="markers",
        marker=dict(
            size=4,
            color=degrees,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Degree"),
        ),
        text=[f"node={n}, degree={H.degree[n]}" for n in H.nodes()],
        hoverinfo="text",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f"3D graph visualization — {args.mode}, {H.number_of_nodes()} nodes",
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, visible=False),
            yaxis=dict(showbackground=False, showticklabels=False, visible=False),
            zaxis=dict(showbackground=False, showticklabels=False, visible=False),
        ),
    )

    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.out_html)
    print(f"\nSaved 3D HTML to: {args.out_html}")


if __name__ == "__main__":
    main()