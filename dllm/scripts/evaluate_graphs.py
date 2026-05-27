from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------
# Loading generated graphs
# ---------------------------------------------------------------------

def load_assembled_graph(path: Path) -> nx.Graph:
    data = json.loads(path.read_text())

    n = int(data["num_nodes"])
    edges = [(int(u), int(v)) for u, v in data["edges"]]

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)

    return G


def load_eval_json_graphs(path: Path, use_repaired: bool = True) -> list[nx.Graph]:
    data = json.loads(path.read_text())
    graphs = []

    for sample in data.get("samples", []):
        if use_repaired:
            adj = sample.get("repaired_adjacency", {})
            valid = sample.get("repaired_valid", False)
        else:
            adj = sample.get("adjacency", {})
            valid = sample.get("raw_valid", False)

        if not valid or not adj:
            continue

        nodes = sorted(int(k) for k in adj.keys())
        if not nodes:
            continue

        G = nx.Graph()
        G.add_nodes_from(nodes)

        for u_str, nbrs in adj.items():
            u = int(u_str)
            for v in nbrs:
                v = int(v)
                if u != v:
                    G.add_edge(u, v)

        graphs.append(G)

    return graphs


# ---------------------------------------------------------------------
# Loading reference MalNetTiny graphs
# ---------------------------------------------------------------------

def pyg_to_nx(data) -> nx.Graph:
    n = int(data.num_nodes)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    edge_index = getattr(data, "edge_index", None)
    if edge_index is not None:
        ei = edge_index.cpu().numpy()
        for u, v in zip(ei[0], ei[1]):
            u = int(u)
            v = int(v)
            if u != v:
                G.add_edge(u, v)

    return G


def load_reference_graphs(
    dataset_name: str,
    data_root: str,
    max_graphs: int,
    max_nodes_per_graph: int,
    malnet_num_hops: int,
    seed: int,
) -> list[nx.Graph]:
    from dllm.data.load_graph_data import load_graph_samples

    data_args = SimpleNamespace(
        pyg_dataset=dataset_name,
        data_root=data_root,
        max_graphs=max_graphs,
        test_size=0.25,
        elliptic_num_hops=2,
        malnet_num_hops=malnet_num_hops,
        max_nodes_per_graph=max_nodes_per_graph,
    )

    graphs, _ = load_graph_samples(data_args, seed)
    return [pyg_to_nx(g) for g in graphs]


# ---------------------------------------------------------------------
# Structural metrics
# ---------------------------------------------------------------------

def graph_hash(G: nx.Graph) -> str:
    # Stable exact hash for relabeled graph with same node ids.
    edges = sorted((min(u, v), max(u, v)) for u, v in G.edges())
    s = f"N={G.number_of_nodes()};E={edges}"
    return hashlib.sha256(s.encode()).hexdigest()


def wl_hash(G: nx.Graph) -> str:
    # Isomorphism-ish hash. Better for uniqueness/novelty when node ids differ.
    try:
        return nx.weisfeiler_lehman_graph_hash(G)
    except Exception:
        return graph_hash(G)


def structural_metrics(G: nx.Graph) -> dict[str, Any]:
    n = G.number_of_nodes()

    self_loops = list(nx.selfloop_edges(G))
    out_of_range_edges = [
        (u, v)
        for u, v in G.edges()
        if not (0 <= int(u) < n and 0 <= int(v) < n)
    ]

    # NetworkX Graph deduplicates edges, so duplicate count is only meaningful
    # if raw edge list was preserved. Here it should be zero after assembly.
    duplicate_edges = 0

    degrees = [d for _, d in G.degree()]
    components = list(nx.connected_components(G))

    valid = (
        len(self_loops) == 0
        and len(out_of_range_edges) == 0
        and duplicate_edges == 0
    )

    return {
        "num_nodes": n,
        "num_edges": G.number_of_edges(),
        "valid": valid,
        "duplicate_edges": duplicate_edges,
        "self_loops": len(self_loops),
        "out_of_range_edges": len(out_of_range_edges),
        "avg_degree": float(np.mean(degrees)) if degrees else 0.0,
        "min_degree": int(np.min(degrees)) if degrees else 0,
        "max_degree": int(np.max(degrees)) if degrees else 0,
        "num_isolates": len(list(nx.isolates(G))),
        "num_components": len(components),
        "largest_component_size": max((len(c) for c in components), default=0),
        "density": nx.density(G),
        "avg_clustering": nx.average_clustering(G) if n > 0 else 0.0,
    }


def uniqueness_novelty(
    generated: list[nx.Graph],
    reference: list[nx.Graph],
) -> dict[str, Any]:
    gen_hashes = [wl_hash(G) for G in generated]
    ref_hashes = {wl_hash(G) for G in reference}

    unique = len(set(gen_hashes))
    total = len(gen_hashes)

    novel = sum(1 for h in gen_hashes if h not in ref_hashes)

    return {
        "num_generated": total,
        "num_unique": unique,
        "uniqueness": unique / total if total else 0.0,
        "num_novel": novel,
        "novelty": novel / total if total else 0.0,
    }


# ---------------------------------------------------------------------
# Distribution features
# ---------------------------------------------------------------------

def degree_hist(G: nx.Graph, max_degree: int = 100) -> np.ndarray:
    degs = np.array([d for _, d in G.degree()], dtype=int)
    degs = np.clip(degs, 0, max_degree)
    hist = np.bincount(degs, minlength=max_degree + 1).astype(float)
    return hist / hist.sum() if hist.sum() > 0 else hist


def clustering_hist(G: nx.Graph, bins: int = 50) -> np.ndarray:
    vals = np.array(list(nx.clustering(G).values()), dtype=float)
    hist, _ = np.histogram(vals, bins=bins, range=(0.0, 1.0), density=False)
    hist = hist.astype(float)
    return hist / hist.sum() if hist.sum() > 0 else hist


def spectral_hist(G: nx.Graph, bins: int = 80) -> np.ndarray:
    # SPECTRE-style/spectral metric: distribution of normalized Laplacian eigenvalues.
    # For huge graphs, use largest connected component if full eigendecomposition is too large.
    H = G

    if H.number_of_nodes() > 1000:
        comps = sorted(nx.connected_components(H), key=len, reverse=True)
        H = H.subgraph(comps[0]).copy()
        if H.number_of_nodes() > 1000:
            # sample high-degree nodes for tractability
            nodes = sorted(H.nodes(), key=lambda x: H.degree[x], reverse=True)[:1000]
            H = H.subgraph(nodes).copy()

    if H.number_of_nodes() <= 1:
        return np.zeros(bins)

    try:
        L = nx.normalized_laplacian_matrix(H).astype(float).toarray()
        eigs = np.linalg.eigvalsh(L)
        eigs = np.clip(eigs, 0.0, 2.0)
        hist, _ = np.histogram(eigs, bins=bins, range=(0.0, 2.0), density=False)
    except Exception:
        hist = np.zeros(bins)

    hist = hist.astype(float)
    return hist / hist.sum() if hist.sum() > 0 else hist


def orbit_features(G: nx.Graph) -> np.ndarray:
    # Lightweight orbit/graphlet-like summary.
    # Not full ORCA or official graphlet orbit counts, but useful and fast.
    n = G.number_of_nodes()
    m = G.number_of_edges()

    degrees = np.array([d for _, d in G.degree()], dtype=float)
    triangles = sum(nx.triangles(G).values()) / 3 if n > 0 else 0

    # wedges = connected triples centered at a node
    wedges = float(sum(d * (d - 1) / 2 for d in degrees))

    # approximate 4-cycles can be expensive; skip exact count for large graphs
    # and use transitivity/clustering proxies.
    transitivity = nx.transitivity(G) if n > 2 else 0.0
    avg_clust = nx.average_clustering(G) if n > 0 else 0.0

    return np.array(
        [
            n,
            m,
            float(np.mean(degrees)) if len(degrees) else 0.0,
            float(np.std(degrees)) if len(degrees) else 0.0,
            float(np.max(degrees)) if len(degrees) else 0.0,
            triangles,
            wedges,
            transitivity,
            avg_clust,
            len(list(nx.isolates(G))),
            nx.number_connected_components(G),
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------
# MMD
# ---------------------------------------------------------------------

def rbf_kernel_matrix(X: np.ndarray, Y: np.ndarray, sigma: float | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    XX = np.sum(X * X, axis=1, keepdims=True)
    YY = np.sum(Y * Y, axis=1, keepdims=True).T
    D = XX + YY - 2 * X @ Y.T
    D = np.maximum(D, 0.0)

    if sigma is None:
        vals = D[D > 0]
        sigma = math.sqrt(float(np.median(vals))) if vals.size else 1.0
        if sigma <= 0:
            sigma = 1.0

    return np.exp(-D / (2 * sigma * sigma))


def mmd_rbf(X: np.ndarray, Y: np.ndarray) -> float:
    if len(X) == 0 or len(Y) == 0:
        return float("nan")

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    Z = np.vstack([X, Y])
    ZZ = np.sum(Z * Z, axis=1, keepdims=True)
    D = ZZ + ZZ.T - 2 * Z @ Z.T
    D = np.maximum(D, 0.0)

    vals = D[D > 0]
    sigma = math.sqrt(float(np.median(vals))) if vals.size else 1.0
    if sigma <= 0:
        sigma = 1.0

    Kxx = rbf_kernel_matrix(X, X, sigma=sigma)
    Kyy = rbf_kernel_matrix(Y, Y, sigma=sigma)
    Kxy = rbf_kernel_matrix(X, Y, sigma=sigma)

    mmd = float(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean())
    return max(0.0, mmd)

def compute_mmd_metrics(generated: list[nx.Graph], reference: list[nx.Graph]) -> dict[str, float]:
    gen_degree = np.stack([degree_hist(G) for G in generated])
    ref_degree = np.stack([degree_hist(G) for G in reference])

    gen_clust = np.stack([clustering_hist(G) for G in generated])
    ref_clust = np.stack([clustering_hist(G) for G in reference])

    gen_spec = np.stack([spectral_hist(G) for G in generated])
    ref_spec = np.stack([spectral_hist(G) for G in reference])

    gen_orbit = np.stack([orbit_features(G) for G in generated])
    ref_orbit = np.stack([orbit_features(G) for G in reference])

    # Normalize orbit features before MMD because counts can be large.
    mean = ref_orbit.mean(axis=0)
    std = ref_orbit.std(axis=0) + 1e-8
    gen_orbit_z = (gen_orbit - mean) / std
    ref_orbit_z = (ref_orbit - mean) / std

    return {
        "mmd_degree": mmd_rbf(gen_degree, ref_degree),
        "mmd_clustering": mmd_rbf(gen_clust, ref_clust),
        "mmd_spectral_spectre": mmd_rbf(gen_spec, ref_spec),
        "mmd_orbit_graphlet_proxy": mmd_rbf(gen_orbit_z, ref_orbit_z),
    }

def load_assembled_graph_dir(path: Path) -> list[nx.Graph]:
    graphs = []

    for graph_path in sorted(path.glob("graph_seed_*/assembled_graph.json")):
        graphs.append(load_assembled_graph(graph_path))

    if not graphs:
        raise ValueError(f"No assembled_graph.json files found under {path}")

    return graphs

def summarize_values(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_graph_json", type=Path, default=None)
    parser.add_argument("--generated_eval_json", type=Path, default=None)
    parser.add_argument("--out_json", type=Path, required=True)

    parser.add_argument("--reference_dataset", default="MalNetTiny")
    parser.add_argument("--data_root", default="./data/pyg")
    parser.add_argument("--reference_max_graphs", type=int, default=512)
    parser.add_argument("--reference_max_nodes_per_graph", type=int, default=6000)
    parser.add_argument("--malnet_num_hops", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generated_graph_dir", type=Path, default=None)

    args = parser.parse_args()

    generated: list[nx.Graph] = []

    if args.generated_graph_json is not None:
        generated.append(load_assembled_graph(args.generated_graph_json))

    if args.generated_graph_dir is not None:
        generated.extend(load_assembled_graph_dir(args.generated_graph_dir))

    if args.generated_eval_json is not None:
        generated.extend(load_eval_json_graphs(args.generated_eval_json, use_repaired=True))

    if not generated:
        raise SystemExit("No generated graphs loaded.")

    reference = load_reference_graphs(
        dataset_name=args.reference_dataset,
        data_root=args.data_root,
        max_graphs=args.reference_max_graphs,
        max_nodes_per_graph=args.reference_max_nodes_per_graph,
        malnet_num_hops=args.malnet_num_hops,
        seed=args.seed,
    )

    structural = [structural_metrics(G) for G in generated]
    validity_rate = sum(m["valid"] for m in structural) / len(structural)

    result = {
        "num_generated_graphs": len(generated),
        "num_reference_graphs": len(reference),
        "validity_rate": validity_rate,
        "structural_summary": {
            "num_nodes": summarize_values([m["num_nodes"] for m in structural]),
            "num_edges": summarize_values([m["num_edges"] for m in structural]),
            "avg_degree": summarize_values([m["avg_degree"] for m in structural]),
            "self_loops": summarize_values([m["self_loops"] for m in structural]),
            "duplicate_edges": summarize_values([m["duplicate_edges"] for m in structural]),
            "out_of_range_edges": summarize_values([m["out_of_range_edges"] for m in structural]),
            "num_isolates": summarize_values([m["num_isolates"] for m in structural]),
            "num_components": summarize_values([m["num_components"] for m in structural]),
            "largest_component_size": summarize_values([m["largest_component_size"] for m in structural]),
            "max_degree": summarize_values([m["max_degree"] for m in structural]),
            "avg_clustering": summarize_values([m["avg_clustering"] for m in structural]),
        },
        "uniqueness_novelty": uniqueness_novelty(generated, reference),
        "mmd": compute_mmd_metrics(generated, reference),
        # "per_graph_structural": structural,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()