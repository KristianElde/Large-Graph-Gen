from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_repaired_chunks(path: Path, min_nodes: int = 2) -> list[dict[int, list[int]]]:
    data = json.loads(path.read_text())
    chunks = []

    for sample in data.get("samples", []):
        if not sample.get("repaired_valid", False):
            continue

        raw_adj = sample.get("repaired_adjacency", {})
        if not raw_adj:
            continue

        adj = {int(k): sorted({int(v) for v in vals}) for k, vals in raw_adj.items()}
        n = len(adj)
        m = sum(len(v) for v in adj.values()) // 2
        avg_degree = (2 * m / n) if n > 0 else 0.0

        if n < min_nodes:
            continue

        if m < 50:
            continue

        if avg_degree < 0.5:
            continue

        # Ensure contiguous nodes 0..n-1.
        if set(adj.keys()) != set(range(n)):
            continue

        chunks.append(adj)

    if not chunks:
        raise ValueError(
            f"No usable repaired chunks found in {path}. "
            f"Try lowering --min_chunk_nodes or generating more samples."
        )

    return chunks


def adjacency_to_edges(adj: dict[int, list[int]]) -> set[tuple[int, int]]:
    edges = set()

    for u, nbrs in adj.items():
        for v in nbrs:
            if u == v:
                continue
            a, b = sorted((int(u), int(v)))
            edges.add((a, b))

    return edges


def assemble_large_graph(
    chunks: list[dict[int, list[int]]],
    *,
    target_nodes: int,
    seed: int,
    connect_chunks: bool = True,
) -> tuple[int, set[tuple[int, int]], list[dict[str, int]]]:
    rng = random.Random(seed)

    edges: set[tuple[int, int]] = set()
    placements: list[dict[str, int]] = []

    offset = 0
    prev_anchor: int | None = None

    while offset < target_nodes:
        chunk = rng.choice(chunks)
        chunk_n = len(chunk)
        remaining = target_nodes - offset
        take_n = min(chunk_n, remaining)

        chunk_edges = adjacency_to_edges(chunk)

        # Keep only edges inside the truncated chunk.
        kept_edges = [
            (u, v)
            for u, v in chunk_edges
            if u < take_n and v < take_n
        ]

        for u, v in kept_edges:
            edges.add((offset + u, offset + v))

        # Optional bridge edge to reduce disconnected components.
        if connect_chunks and prev_anchor is not None and take_n > 0:
            current_nodes = list(range(offset, offset + take_n))
            previous_nodes = list(range(max(0, offset - 50), offset))

            for _ in range(5):
                if previous_nodes and current_nodes:
                    a = rng.choice(previous_nodes)
                    b = rng.choice(current_nodes)
                    if a != b:
                        edges.add(tuple(sorted((a, b))))

        if take_n > 0:
            prev_anchor = offset + take_n - 1

        placements.append(
            {
                "offset": offset,
                "source_chunk_nodes": chunk_n,
                "used_nodes": take_n,
                "used_edges": len(kept_edges),
            }
        )

        offset += take_n

    return target_nodes, edges, placements


def edges_to_adjacency(num_nodes: int, edges: set[tuple[int, int]]) -> dict[int, list[int]]:
    adj = {i: set() for i in range(num_nodes)}

    for u, v in edges:
        if u == v:
            continue
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            continue
        adj[u].add(v)
        adj[v].add(u)

    return {u: sorted(vs) for u, vs in adj.items()}


class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def compute_metrics(num_nodes: int, edges: set[tuple[int, int]]) -> dict[str, Any]:
    degrees = [0] * num_nodes
    dsu = DSU(num_nodes)

    self_loops = 0
    out_of_range = 0

    for u, v in edges:
        if u == v:
            self_loops += 1
            continue
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            out_of_range += 1
            continue

        degrees[u] += 1
        degrees[v] += 1
        dsu.union(u, v)

    component_sizes: dict[int, int] = {}
    for i in range(num_nodes):
        r = dsu.find(i)
        component_sizes[r] = component_sizes.get(r, 0) + 1

    num_edges = len(edges)
    possible_edges = num_nodes * (num_nodes - 1) / 2

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": round(sum(degrees) / num_nodes, 4) if num_nodes else 0,
        "min_degree": min(degrees) if degrees else 0,
        "max_degree": max(degrees) if degrees else 0,
        "num_isolates": sum(1 for d in degrees if d == 0),
        "density": round(num_edges / possible_edges, 8) if possible_edges else 0,
        "num_components": len(component_sizes),
        "largest_component_size": max(component_sizes.values()) if component_sizes else 0,
        "self_loops": self_loops,
        "out_of_range_edges": out_of_range,
    }

def add_preferential_edges(num_nodes, edges, target_avg_degree, seed):
    rng = random.Random(seed)
    target_edges = int(num_nodes * target_avg_degree / 2)

    degrees = [0] * num_nodes
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1

    while len(edges) < target_edges:
        # Nodes with higher degree are more likely to be selected,
        # but zero-degree nodes still have chance because of +1.
        weights = [d + 1 for d in degrees]

        u = rng.choices(range(num_nodes), weights=weights, k=1)[0]
        v = rng.choices(range(num_nodes), weights=weights, k=1)[0]

        if u == v:
            continue

        a, b = sorted((u, v))
        if (a, b) in edges:
            continue

        edges.add((a, b))
        degrees[a] += 1
        degrees[b] += 1

    return edges


def save_edge_list_text(path: Path, num_nodes: int, edges: set[tuple[int, int]]) -> None:
    edge_text = ",".join(f"({u},{v})" for u, v in sorted(edges))
    path.write_text(f"N={num_nodes}; E={edge_text}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble a 5000+ node graph from repaired generated graph chunks."
    )
    parser.add_argument(
        "--eval_json",
        type=Path,
        required=True,
        help="Path to graph_generation_eval.json from an edge-list run.",
    )
    parser.add_argument(
        "--target_nodes",
        type=int,
        default=5000,
        help="Number of nodes in the assembled large graph.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--min_chunk_nodes",
        type=int,
        default=2,
        help="Ignore repaired chunks with fewer than this many nodes.",
    )
    parser.add_argument(
        "--no_connect_chunks",
        action="store_true",
        help="Do not add bridge edges between chunks.",
    )

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_repaired_chunks(args.eval_json, min_nodes=args.min_chunk_nodes)

    num_nodes, edges, placements = assemble_large_graph(
        chunks,
        target_nodes=args.target_nodes,
        seed=args.seed,
        connect_chunks=not args.no_connect_chunks,
    )

    edges = add_preferential_edges(
        num_nodes,
        edges,
        target_avg_degree=2.0,
        seed=args.seed + 999,
    )

    adjacency = edges_to_adjacency(num_nodes, edges)
    metrics = compute_metrics(num_nodes, edges)

    graph_json = {
        "num_nodes": num_nodes,
        "num_edges": len(edges),
        "edges": [[u, v] for u, v in sorted(edges)],
        "adjacency": {str(k): v for k, v in adjacency.items()},
        "placements": placements,
        "source_eval_json": str(args.eval_json),
        "seed": args.seed,
        "connect_chunks": not args.no_connect_chunks,
    }

    (args.out_dir / "assembled_graph.json").write_text(
        json.dumps(graph_json, indent=2) + "\n"
    )

    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )

    save_edge_list_text(args.out_dir / "assembled_graph.edgelist.txt", num_nodes, edges)

    print("Saved:")
    print(f"  {args.out_dir / 'assembled_graph.json'}")
    print(f"  {args.out_dir / 'assembled_graph.edgelist.txt'}")
    print(f"  {args.out_dir / 'metrics.json'}")
    print()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()