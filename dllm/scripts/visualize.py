from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def load_graph(path: Path) -> nx.Graph:
    data = json.loads(path.read_text())

    n = int(data["num_nodes"])
    edges = [(int(u), int(v)) for u, v in data["edges"]]

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)

    return G


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_json", type=Path, required=True)
    parser.add_argument("--out_png", type=Path, required=True)
    parser.add_argument("--mode", choices=["largest", "sample"], default="largest")
    parser.add_argument("--max_nodes", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    G = load_graph(args.graph_json)

    if args.mode == "largest":
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        nodes = list(components[0])
    else:
        nodes = list(G.nodes())

    if len(nodes) > args.max_nodes:
        nodes = random.sample(nodes, args.max_nodes)

    H = G.subgraph(nodes).copy()

    print("Original graph:")
    print("  nodes:", G.number_of_nodes())
    print("  edges:", G.number_of_edges())
    print("  components:", nx.number_connected_components(G))
    print("  isolates:", len(list(nx.isolates(G))))

    print("\nPlotted graph:")
    print("  nodes:", H.number_of_nodes())
    print("  edges:", H.number_of_edges())
    print("  components:", nx.number_connected_components(H))

    pos = nx.spring_layout(H, seed=args.seed, k=None)

    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(H, pos, node_size=20, alpha=0.8)
    nx.draw_networkx_edges(H, pos, width=0.4, alpha=0.4)
    plt.axis("off")
    plt.tight_layout()

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_png, dpi=200)
    print(f"\nSaved plot to: {args.out_png}")


if __name__ == "__main__":
    main()