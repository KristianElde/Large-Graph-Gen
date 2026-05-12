from __future__ import annotations

from abc import ABC, abstractmethod

from .types import SimpleGraphData


class GraphTokenizer(ABC):
    """Shared API for graph tokenization methods."""

    @abstractmethod
    def tokenize(self, data: SimpleGraphData):
        raise NotImplementedError

    @abstractmethod
    def decode(self, tokens):
        raise NotImplementedError


def graph_to_tokens(edge_index, num_nodes, tokenizer: GraphTokenizer, **attrs):
    """Tokenize from raw graph tensors.

    Args:
        edge_index: Torch tensor with shape [2, num_edges].
        num_nodes: Number of nodes.
        tokenizer: Configured graph tokenizer.
        **attrs: Extra graph fields (e.g. dataset_name, x, edge_attr).
    """
    data = SimpleGraphData(edge_index=edge_index, num_nodes=num_nodes, **attrs)
    return tokenizer.tokenize(data)
