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

class TokenizerFactory:
    """Factory to fetch tokenizers without importing specific classes manually."""
    _tokenizers = {}

    @classmethod
    def register(cls, name):
        def inner(subclass):
            cls._tokenizers[name.lower()] = subclass
            return subclass
        return inner

    @classmethod
    def get_tokenizer(cls, name, **kwargs):
        if name.lower() not in cls._tokenizers:
            raise ValueError(f"Tokenizer {name} not found. Available: {list(cls._tokenizers.keys())}")
        return cls._tokenizers[name.lower()](**kwargs)


def graph_to_tokens(edge_index, num_nodes, tokenizer: GraphTokenizer):
    """Convert an unlabeled graph to tokens.
    
    Args:
        edge_index: Tensor of shape [2, num_edges] with edge indices
        num_nodes: Number of nodes in the graph
        tokenizer: GraphTokenizer instance
        
    Returns:
        Tokenized representation of the graph
    """
    data = SimpleGraphData(edge_index=edge_index, num_nodes=num_nodes)
    return tokenizer.tokenize(data)