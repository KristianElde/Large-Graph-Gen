from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def _ensure_rng(rng):
    if rng is None:
        return np.random.mtrand._rand
    if isinstance(rng, int):
        return np.random.RandomState(rng)
    return rng


def _to_numpy_1d(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return x.reshape(-1).astype(np.int64, copy=False)


def _to_numpy_edge_index(edge_index) -> np.ndarray:
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.detach().cpu().numpy()
    edge_index = np.asarray(edge_index)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")
    return edge_index.astype(np.int64, copy=False)


def _build_out_neighbors(
    edge_index: np.ndarray, num_nodes: int
) -> Tuple[List[List[int]], List[List[int]]]:
    neighbors: List[List[int]] = [[] for _ in range(num_nodes)]
    edge_ids: List[List[int]] = [[] for _ in range(num_nodes)]
    for idx in range(edge_index.shape[1]):
        src = int(edge_index[0, idx])
        dst = int(edge_index[1, idx])
        if 0 <= src < num_nodes and 0 <= dst < num_nodes:
            neighbors[src].append(dst)
            edge_ids[src].append(idx)
    return neighbors, edge_ids


def sample_sent_from_graph(
    edge_index,
    num_nodes: int,
    max_length: int = -1,
    idx_offset: int = 0,
    reset: int = -1,
    ladj: int = -2,
    radj: int = -3,
    rng=None,
):
    rng = _ensure_rng(rng)
    edge_index_np = _to_numpy_edge_index(edge_index)

    if num_nodes <= 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)

    num_edges = edge_index_np.shape[1]
    if max_length < 0:
        max_length = 20 if num_nodes <= 1 else (num_edges + num_nodes) * 2

    out_neighbors, _ = _build_out_neighbors(edge_index_np, num_nodes)
    unvisited = np.ones(num_nodes, dtype=bool)
    node_index_map = np.full(num_nodes, -1, dtype=np.int64)
    sent_seq: List[int] = []

    current_node = int(rng.randint(0, num_nodes))
    node_index = 0
    node_index_map[current_node] = node_index
    sent_seq.append(node_index + idx_offset)
    node_index += 1
    unvisited[current_node] = False
    num_unvisited = num_nodes - 1

    while num_unvisited > 0 and len(sent_seq) < max_length:
        prev_node = current_node

        unvisited_neighbors = [n for n in out_neighbors[current_node] if unvisited[n]]
        if not unvisited_neighbors:
            sent_seq.append(reset)
            unvisited_nodes = np.flatnonzero(unvisited)
            sample_idx = int(rng.randint(0, len(unvisited_nodes)))
            current_node = int(unvisited_nodes[sample_idx])
        else:
            sample_idx = int(rng.randint(0, len(unvisited_neighbors)))
            current_node = int(unvisited_neighbors[sample_idx])

        node_index_map[current_node] = node_index
        sent_seq.append(node_index + idx_offset)
        node_index += 1
        unvisited[current_node] = False
        num_unvisited -= 1

        neighborhood = sorted(
            int(node_index_map[n])
            for n in out_neighbors[current_node]
            if (not unvisited[n]) and (n != prev_node)
        )
        if neighborhood:
            sent_seq.append(ladj)
            sent_seq.extend(n + idx_offset for n in neighborhood)
            sent_seq.append(radj)

    return np.asarray(sent_seq, dtype=np.int64), node_index_map


def sample_labeled_sent_from_graph(
    edge_index,
    node_labels,
    edge_labels,
    node_idx_offset: int,
    edge_idx_offset: int,
    num_nodes: int,
    max_length: int = -1,
    idx_offset: int = 0,
    reset: int = -1,
    ladj: int = -2,
    radj: int = -3,
    rng=None,
):
    rng = _ensure_rng(rng)
    edge_index_np = _to_numpy_edge_index(edge_index)
    node_labels_np = _to_numpy_1d(node_labels)
    edge_labels_np = _to_numpy_1d(edge_labels)

    if num_nodes <= 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    if len(node_labels_np) < num_nodes:
        raise ValueError("node_labels must have at least num_nodes elements.")
    if len(edge_labels_np) < edge_index_np.shape[1]:
        raise ValueError("edge_labels must have at least num_edges elements.")

    num_edges = edge_index_np.shape[1]
    if max_length < 0:
        max_length = 20 if num_nodes <= 1 else 2 * (num_nodes * 2 + num_edges * 2)

    out_neighbors, out_edge_ids = _build_out_neighbors(edge_index_np, num_nodes)
    unvisited = np.ones(num_nodes, dtype=bool)
    node_index_map = np.full(num_nodes, -1, dtype=np.int64)
    sent_seq: List[int] = []

    current_node = int(rng.randint(0, num_nodes))
    node_index = 0
    node_index_map[current_node] = node_index
    sent_seq.append(node_index + idx_offset)
    node_index += 1
    unvisited[current_node] = False
    num_unvisited = num_nodes - 1
    sent_seq.append(int(node_labels_np[current_node]) + node_idx_offset)

    while num_unvisited > 0 and len(sent_seq) < max_length:
        prev_node = current_node
        candidates: List[Tuple[int, int]] = []
        for n, eidx in zip(out_neighbors[current_node], out_edge_ids[current_node]):
            if unvisited[n]:
                candidates.append((n, eidx))

        if not candidates:
            sent_seq.append(reset)
            unvisited_nodes = np.flatnonzero(unvisited)
            sample_idx = int(rng.randint(0, len(unvisited_nodes)))
            current_node = int(unvisited_nodes[sample_idx])
        else:
            sample_idx = int(rng.randint(0, len(candidates)))
            current_node, edge_idx = candidates[sample_idx]
            sent_seq.append(int(edge_labels_np[edge_idx]) + edge_idx_offset)

        node_index_map[current_node] = node_index
        sent_seq.append(node_index + idx_offset)
        node_index += 1
        unvisited[current_node] = False
        num_unvisited -= 1
        sent_seq.append(int(node_labels_np[current_node]) + node_idx_offset)

        neighborhood: List[int] = []
        edge_for_neighborhood: Dict[int, int] = {}
        for n, eidx in zip(out_neighbors[current_node], out_edge_ids[current_node]):
            if (not unvisited[n]) and (n != prev_node):
                mapped = int(node_index_map[n])
                neighborhood.append(mapped)
                edge_for_neighborhood[mapped] = eidx
        neighborhood.sort()
        if neighborhood:
            sent_seq.append(ladj)
            for mapped in neighborhood:
                sent_seq.append(int(edge_labels_np[edge_for_neighborhood[mapped]]) + edge_idx_offset)
                sent_seq.append(mapped + idx_offset)
            sent_seq.append(radj)

    return np.asarray(sent_seq, dtype=np.int64), node_index_map


def reconstruct_graph_from_sent(
    sent_seq: np.ndarray,
    reset: int,
    ladj: int,
    radj: int,
) -> np.ndarray:
    edges: List[Tuple[int, int]] = []
    start_bracket = False
    bracket_idx = 0
    walk_length = len(sent_seq)

    for i in range(walk_length - 1):
        a = int(sent_seq[i])
        b = int(sent_seq[i + 1])
        if a == reset or b == reset or b == ladj:
            start_bracket = False
            continue
        if a == ladj:
            start_bracket = True
            bracket_idx = int(sent_seq[i - 1])
        elif a == radj and start_bracket:
            edges.append((bracket_idx, b))
            start_bracket = False
        elif start_bracket:
            edges.append((bracket_idx, a))
        else:
            edges.append((a, b))

    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64).T


def reconstruct_graph_from_labeled_sent(
    sent_seq: np.ndarray,
    reset: int,
    ladj: int,
    radj: int,
    idx_offset: int = 0,
):
    edge_index: List[Tuple[int, int]] = []
    edge_labels: List[int] = []
    node_labels: Dict[int, int] = {}

    i = 0
    walk_length = len(sent_seq)
    start_bracket = False
    bracket_idx = 0

    while i < walk_length - 1:
        current = int(sent_seq[i])
        nxt = int(sent_seq[i + 1])

        if current == reset or nxt == reset:
            start_bracket = False
            i += 1
            continue

        if current == ladj:
            start_bracket = True
            bracket_idx = int(sent_seq[i - 2])
            i += 1
            continue

        if current == radj and start_bracket:
            if i + 2 < walk_length:
                dst = int(sent_seq[i + 2])
                edge_index.append((bracket_idx, dst))
                edge_labels.append(nxt)
            start_bracket = False
            i += 2
            continue

        if start_bracket:
            if i + 1 < walk_length:
                edge_label = current
                dst = int(sent_seq[i + 1])
                edge_index.append((bracket_idx, dst))
                edge_labels.append(edge_label)
            i += 2
            continue

        if i + 2 >= walk_length:
            break

        node_idx = current
        node_label = nxt
        if node_idx not in node_labels:
            node_labels[node_idx] = node_label

        third = int(sent_seq[i + 2])
        if third == reset or third == ladj:
            i += 2
            continue

        if i + 3 <= walk_length - 1:
            dst = int(sent_seq[i + 3])
            edge_index.append((node_idx, dst))
            edge_labels.append(third)
        i += 3

    if edge_index:
        edge_index_np = np.asarray(edge_index, dtype=np.int64).T
        edge_labels_np = np.asarray(edge_labels, dtype=np.int64)
    else:
        edge_index_np = np.zeros((2, 0), dtype=np.int64)
        edge_labels_np = np.zeros((0,), dtype=np.int64)

    if node_labels:
        max_node = max(node_labels.keys())
    else:
        max_node = idx_offset - 1
    node_labels_np = np.full(max(max_node + 1, idx_offset), -1, dtype=np.int64)
    for node_idx, label in node_labels.items():
        if 0 <= node_idx < len(node_labels_np):
            node_labels_np[node_idx] = label

    return edge_index_np, node_labels_np, edge_labels_np


def remove_self_loops(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None):
    if edge_index.numel() == 0:
        return edge_index, edge_attr
    keep = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, keep]
    if edge_attr is not None:
        edge_attr = edge_attr[keep]
    return edge_index, edge_attr


def relabel_and_coalesce(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None):
    if edge_index.numel() == 0:
        empty_e = torch.zeros((2, 0), dtype=torch.long)
        if edge_attr is None:
            return empty_e, None
        empty_attr = torch.zeros((0,), dtype=edge_attr.dtype)
        return empty_e, empty_attr

    unique_nodes = torch.unique(edge_index)
    edge_index = torch.searchsorted(unique_nodes, edge_index)

    pairs = edge_index.t()
    if edge_attr is None:
        unique_pairs = torch.unique(pairs, dim=0, sorted=True)
        return unique_pairs.t().contiguous(), None

    unique_pairs, inverse = torch.unique(pairs, dim=0, sorted=True, return_inverse=True)
    reduced_attr = torch.full(
        (unique_pairs.shape[0],), torch.iinfo(edge_attr.dtype).max, dtype=edge_attr.dtype
    )
    for i in range(edge_attr.shape[0]):
        idx = int(inverse[i])
        reduced_attr[idx] = torch.minimum(reduced_attr[idx], edge_attr[i])
    return unique_pairs.t().contiguous(), reduced_attr


def get_graph_from_sent(walk_index, idx_offset, reset, ladj, radj, undirected=True):
    walk_np = _to_numpy_1d(walk_index)
    edge_np = reconstruct_graph_from_sent(walk_np, reset, ladj, radj)
    edge_index = torch.from_numpy(edge_np).long()

    if undirected and edge_index.numel() > 0:
        edge_sym = torch.stack((edge_index[1], edge_index[0]), dim=0)
        edge_index = torch.cat((edge_index, edge_sym), dim=1)

    edge_index, _ = remove_self_loops(edge_index, None)
    edge_index, _ = relabel_and_coalesce(edge_index, None)
    return edge_index


def get_graph_from_labeled_sent(
    walk_index,
    idx_offset,
    node_idx_offset,
    edge_idx_offset,
    num_node_types,
    num_edge_types,
    reset,
    ladj,
    radj,
    undirected=True,
):
    walk_np = _to_numpy_1d(walk_index)
    edge_np, node_labels_np, edge_labels_np = reconstruct_graph_from_labeled_sent(
        walk_np, reset, ladj, radj, idx_offset
    )

    edge_index = torch.from_numpy(edge_np).long()
    edge_labels = torch.from_numpy(edge_labels_np).long()
    node_labels = torch.from_numpy(node_labels_np).long()

    if edge_index.numel() > 0:
        max_node_idx = int(edge_index.max().item()) + 1
        node_labels = node_labels[idx_offset:max_node_idx]
    else:
        node_labels = torch.zeros((0,), dtype=torch.long)

    edge_index = edge_index - idx_offset
    node_labels = node_labels - node_idx_offset
    edge_labels = edge_labels - edge_idx_offset

    node_labels[(node_labels < 0) | (node_labels >= num_node_types)] = 0
    edge_labels[(edge_labels < 0) | (edge_labels >= num_edge_types)] = 0

    if undirected and edge_index.numel() > 0:
        edge_sym = torch.stack((edge_index[1], edge_index[0]), dim=0)
        edge_index = torch.cat((edge_index, edge_sym), dim=1)
        edge_labels = torch.cat((edge_labels, edge_labels), dim=0)

    edge_index, edge_labels = remove_self_loops(edge_index, edge_labels)
    edge_index, edge_labels = relabel_and_coalesce(edge_index, edge_labels)
    return edge_index, node_labels, edge_labels
