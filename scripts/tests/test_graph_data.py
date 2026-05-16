"""
Run with:
    pytest scripts/tests/test_graph_data.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from graph_tokenization import AutoGraphTokenizer, SimpleGraphData


_GRAPH_DATA_PATH = Path(__file__).resolve().parents[2] / "dllm" / "utils" / "graph_data.py"
_GRAPH_DATA_SPEC = importlib.util.spec_from_file_location(
    "graph_data_under_test", _GRAPH_DATA_PATH
)
assert _GRAPH_DATA_SPEC is not None and _GRAPH_DATA_SPEC.loader is not None
_GRAPH_DATA_MODULE = importlib.util.module_from_spec(_GRAPH_DATA_SPEC)
_GRAPH_DATA_SPEC.loader.exec_module(_GRAPH_DATA_MODULE)

infer_graph_tokenizer_stats = _GRAPH_DATA_MODULE.infer_graph_tokenizer_stats
tokenize_graphs = _GRAPH_DATA_MODULE.tokenize_graphs


def test_tokenize_graphs_shifts_tokens_into_reserved_range():
    graph = SimpleGraphData(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        num_nodes=3,
    )
    tokenizer = AutoGraphTokenizer(max_length=-1, undirected=True, append_eos=True)
    tokenizer.set_num_nodes(3)

    rows = tokenize_graphs([graph], tokenizer, token_offset=100)

    assert len(rows) == 1
    expected = (tokenizer.tokenize(graph) + 100).tolist()
    assert rows[0]["input_ids"] == expected
    assert rows[0]["labels"] == expected


def test_infer_graph_tokenizer_stats_uses_graph_size():
    graph = SimpleGraphData(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        num_nodes=3,
    )

    stats = infer_graph_tokenizer_stats([graph], labeled_graph=False)

    assert stats["max_num_nodes"] == 3
    assert stats["num_node_types"] == 0
    assert stats["num_edge_types"] == 0