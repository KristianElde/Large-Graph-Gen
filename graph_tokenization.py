"""Standalone graph tokenization utilities extracted from AutoGraph.

This file is self-contained (only depends on torch and numpy), so you can copy
it into another project without importing anything from the AutoGraph package.

Quick start (unlabeled graphs):
    import torch
    from graph_tokenization import GraphSequenceTokenizer, graph_to_tokens

    edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long)
    tokenizer = GraphSequenceTokenizer(max_length=-1, undirected=True)
    tokenizer.set_num_nodes(3)

    tokens = graph_to_tokens(edge_index=edge_index, num_nodes=3, tokenizer=tokenizer)
    reconstructed_graph = tokenizer.decode(tokens)

Quick start (labeled graphs):
    import torch
    from graph_tokenization import GraphSequenceTokenizer, SimpleGraphData

    data = SimpleGraphData(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        num_nodes=3,
        x=torch.tensor([2, 0, 1], dtype=torch.long),         # node labels
        edge_attr=torch.tensor([1, 3], dtype=torch.long),    # edge labels
    )

    tokenizer = GraphSequenceTokenizer(labeled_graph=True, undirected=True)
    tokenizer.set_num_nodes(3)
    tokenizer.set_num_node_and_edge_types(num_node_types=5, num_edge_types=4)
    tokens = tokenizer.tokenize(data)
    reconstructed_graph = tokenizer.decode(tokens)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class SimpleGraphData:
    """Minimal graph container used by this standalone module."""

    edge_index: torch.Tensor
    num_nodes: int
    x: Optional[torch.Tensor] = None
    edge_attr: Optional[torch.Tensor] = None
    dataset_name: Optional[str] = None


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


def _sample_sent_from_graph(
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


def _sample_labeled_sent_from_graph(
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


def _reconstruct_graph_from_sent(
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


def _reconstruct_graph_from_labeled_sent(
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


def _remove_self_loops(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None):
    if edge_index.numel() == 0:
        return edge_index, edge_attr
    keep = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, keep]
    if edge_attr is not None:
        edge_attr = edge_attr[keep]
    return edge_index, edge_attr


def _relabel_and_coalesce(
    edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None
):
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


def _get_graph_from_sent(walk_index, idx_offset, reset, ladj, radj, undirected=True):
    walk_np = _to_numpy_1d(walk_index)
    edge_np = _reconstruct_graph_from_sent(walk_np, reset, ladj, radj)
    edge_index = torch.from_numpy(edge_np).long()

    if undirected and edge_index.numel() > 0:
        edge_sym = torch.stack((edge_index[1], edge_index[0]), dim=0)
        edge_index = torch.cat((edge_index, edge_sym), dim=1)

    edge_index, _ = _remove_self_loops(edge_index, None)
    edge_index, _ = _relabel_and_coalesce(edge_index, None)
    return edge_index


def _get_graph_from_labeled_sent(
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
    edge_np, node_labels_np, edge_labels_np = _reconstruct_graph_from_labeled_sent(
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

    edge_index, edge_labels = _remove_self_loops(edge_index, edge_labels)
    edge_index, edge_labels = _relabel_and_coalesce(edge_index, edge_labels)
    return edge_index, node_labels, edge_labels


class GraphSequenceTokenizer:
    """Tokenize graphs into AutoGraph SENT token sequences and decode back."""

    sos: int = 0
    reset: int = 1
    ladj: int = 2
    radj: int = 3
    eos: int = 4
    pad: int = 5
    special_toks = ["sos", "reset", "ladj", "radj", "eos", "pad"]

    def __init__(
        self,
        dataset_names=None,
        max_length=-1,
        truncation_length=None,
        labeled_graph=False,
        undirected=True,
        append_eos=True,
        rng=None,
    ):
        self.dataset_names = dataset_names or []
        self.max_length = max_length
        self.truncation_length = truncation_length
        self.labeled_graph = labeled_graph
        self.undirected = undirected
        self.append_eos = append_eos
        self.rng = rng

        self.dataset_to_idx = {
            dataset_name: i + len(self.special_toks)
            for i, dataset_name in enumerate(self.dataset_names)
        }
        self.idx_offset = len(self.special_toks) + len(self.dataset_names)

        self.max_num_nodes = None
        self.num_node_types = 0
        self.num_edge_types = 0
        self.node_idx_offset = None
        self.edge_idx_offset = None

    def set_num_nodes(self, max_num_nodes):
        if (self.max_num_nodes is None) or (self.max_num_nodes < max_num_nodes):
            self.max_num_nodes = max_num_nodes

    def set_num_node_and_edge_types(self, num_node_types=0, num_edge_types=0):
        if self.labeled_graph:
            if self.max_num_nodes is None:
                raise ValueError("Call set_num_nodes before setting node/edge types.")
            self.num_node_types = num_node_types
            self.num_edge_types = num_edge_types
            self.node_idx_offset = self.idx_offset + self.max_num_nodes
            self.edge_idx_offset = self.node_idx_offset + self.num_node_types

    def __len__(self):
        if self.max_num_nodes is None:
            raise ValueError("Call set_num_nodes before querying vocabulary length.")
        if self.labeled_graph:
            return (
                self.idx_offset
                + self.max_num_nodes
                + self.num_node_types
                + self.num_edge_types
            )
        return self.idx_offset + self.max_num_nodes

    def _coalesce_if_available(self, data):
        if hasattr(data, "is_coalesced") and callable(data.is_coalesced):
            if not data.is_coalesced():
                return data.coalesce()
        return data

    def tokenize(self, data):
        data = self._coalesce_if_available(data)
        if not hasattr(data, "edge_index"):
            raise ValueError("data must provide edge_index.")

        num_nodes = getattr(data, "num_nodes", None)
        if num_nodes is None:
            edge_index = _to_numpy_edge_index(data.edge_index)
            num_nodes = int(edge_index.max()) + 1 if edge_index.shape[1] > 0 else 0

        if self.labeled_graph:
            if self.node_idx_offset is None or self.edge_idx_offset is None:
                raise ValueError(
                    "For labeled graphs, call set_num_nodes and "
                    "set_num_node_and_edge_types before tokenizing."
                )
            if not hasattr(data, "x") or data.x is None:
                raise ValueError("Labeled graph tokenization requires data.x (node labels).")
            if not hasattr(data, "edge_attr") or data.edge_attr is None:
                raise ValueError(
                    "Labeled graph tokenization requires data.edge_attr (edge labels)."
                )
            walk_index, _ = _sample_labeled_sent_from_graph(
                edge_index=data.edge_index,
                node_labels=data.x,
                edge_labels=data.edge_attr,
                node_idx_offset=self.node_idx_offset,
                edge_idx_offset=self.edge_idx_offset,
                num_nodes=int(num_nodes),
                max_length=self.max_length,
                idx_offset=self.idx_offset,
                reset=self.reset,
                ladj=self.ladj,
                radj=self.radj,
                rng=self.rng,
            )
        else:
            walk_index, _ = _sample_sent_from_graph(
                edge_index=data.edge_index,
                num_nodes=int(num_nodes),
                max_length=self.max_length,
                idx_offset=self.idx_offset,
                reset=self.reset,
                ladj=self.ladj,
                radj=self.radj,
                rng=self.rng,
            )

        start_offset = 1  # sos
        end_offset = 1 if self.append_eos else 0

        dataset_name_idx = None
        if self.dataset_names and hasattr(data, "dataset_name"):
            dataset_name_idx = self.dataset_to_idx.get(data.dataset_name, None)
            if dataset_name_idx is not None:
                start_offset += 1

        walk_index_t = torch.from_numpy(walk_index)
        tokens = torch.zeros(
            (walk_index_t.shape[0] + start_offset + end_offset,), dtype=walk_index_t.dtype
        )
        tokens[0] = self.sos
        if dataset_name_idx is not None:
            tokens[1] = dataset_name_idx

        if self.append_eos:
            tokens[-1] = self.eos
            tokens[start_offset:-1] = walk_index_t
        else:
            tokens[start_offset:] = walk_index_t

        return tokens

    def decode(self, tokens: torch.Tensor):
        tokens = tokens[(tokens != self.pad) & (tokens != self.sos) & (tokens != self.eos)]
        if tokens.numel() == 0:
            return SimpleGraphData(edge_index=torch.zeros((2, 0), dtype=torch.long), num_nodes=0)

        dataset_name = None
        if self.dataset_names:
            maybe_dataset_idx = int(tokens[0].item()) - len(self.special_toks)
            if 0 <= maybe_dataset_idx < len(self.dataset_names):
                dataset_name = self.dataset_names[maybe_dataset_idx]
                tokens = tokens[1:]
                if tokens.numel() == 0:
                    return SimpleGraphData(
                        edge_index=torch.zeros((2, 0), dtype=torch.long),
                        num_nodes=0,
                        dataset_name=dataset_name,
                    )

        if self.labeled_graph:
            edge_index, node_labels, edge_labels = _get_graph_from_labeled_sent(
                walk_index=tokens,
                idx_offset=self.idx_offset,
                node_idx_offset=self.node_idx_offset,
                edge_idx_offset=self.edge_idx_offset,
                num_node_types=self.num_node_types,
                num_edge_types=self.num_edge_types,
                reset=self.reset,
                ladj=self.ladj,
                radj=self.radj,
                undirected=self.undirected,
            )
            num_nodes = (
                int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else int(node_labels.numel())
            )
            return SimpleGraphData(
                x=node_labels,
                edge_index=edge_index,
                edge_attr=edge_labels,
                num_nodes=num_nodes,
                dataset_name=dataset_name,
            )

        edge_index = _get_graph_from_sent(
            walk_index=tokens,
            idx_offset=self.idx_offset,
            reset=self.reset,
            ladj=self.ladj,
            radj=self.radj,
            undirected=self.undirected,
        )
        num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else 0
        return SimpleGraphData(edge_index=edge_index, dataset_name=dataset_name, num_nodes=num_nodes)


def graph_to_tokens(edge_index, num_nodes, tokenizer: GraphSequenceTokenizer, **attrs):
    """Tokenize from raw graph tensors.

    Args:
        edge_index: Torch tensor with shape [2, num_edges].
        num_nodes: Number of nodes.
        tokenizer: Configured GraphSequenceTokenizer.
        **attrs: Extra graph fields (e.g. dataset_name, x, edge_attr).
    """
    data = SimpleGraphData(edge_index=edge_index, num_nodes=num_nodes, **attrs)
    return tokenizer.tokenize(data)
