"""Graph tokenization utilities with method-specific tokenizers.

Quick start (unlabeled graphs):
    import torch
    from graph_tokenization import AutoGraphTokenizer, graph_to_tokens

    edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long)
    tokenizer = AutoGraphTokenizer(max_length=-1, undirected=True)
    tokenizer.set_num_nodes(3)

    tokens = graph_to_tokens(edge_index=edge_index, num_nodes=3, tokenizer=tokenizer)
    reconstructed_graph = tokenizer.decode(tokens)

Quick start (labeled graphs):
    import torch
    from graph_tokenization import AutoGraphTokenizer, SimpleGraphData

    data = SimpleGraphData(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        num_nodes=3,
        x=torch.tensor([2, 0, 1], dtype=torch.long),         # node labels
        edge_attr=torch.tensor([1, 3], dtype=torch.long),    # edge labels
    )

    tokenizer = AutoGraphTokenizer(labeled_graph=True, undirected=True)
    tokenizer.set_num_nodes(3)
    tokenizer.set_num_node_and_edge_types(num_node_types=5, num_edge_types=4)
    tokens = tokenizer.tokenize(data)
    reconstructed_graph = tokenizer.decode(tokens)
"""

from .autograph import AutoGraphTokenizer
from .base import GraphTokenizer, graph_to_tokens
from .types import SimpleGraphData

__all__ = [
    "AutoGraphTokenizer",
    "GraphTokenizer",
    "SimpleGraphData",
    "graph_to_tokens",
]
