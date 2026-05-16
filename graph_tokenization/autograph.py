from __future__ import annotations

import torch

from .autograph_ops import (
    _to_numpy_edge_index,
    get_graph_from_labeled_sent,
    get_graph_from_sent,
    sample_labeled_sent_from_graph,
    sample_sent_from_graph,
)
from .base import GraphTokenizer, TokenizerFactory
from .types import SimpleGraphData

@TokenizerFactory.register('autograph')
class AutoGraphTokenizer(GraphTokenizer):
    """Tokenize graphs into AutoGraph SENT token sequences and decode back."""

    method_name = "autograph"

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
                self.idx_offset + self.max_num_nodes + self.num_node_types + self.num_edge_types
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
            walk_index, _ = sample_labeled_sent_from_graph(
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
            walk_index, _ = sample_sent_from_graph(
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
            edge_index, node_labels, edge_labels = get_graph_from_labeled_sent(
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

        edge_index = get_graph_from_sent(
            walk_index=tokens,
            idx_offset=self.idx_offset,
            reset=self.reset,
            ladj=self.ladj,
            radj=self.radj,
            undirected=self.undirected,
        )
        num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else 0
        return SimpleGraphData(edge_index=edge_index, dataset_name=dataset_name, num_nodes=num_nodes)
