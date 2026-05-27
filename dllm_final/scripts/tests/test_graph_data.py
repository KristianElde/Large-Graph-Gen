"""Graph data tests.

Run with:
    pytest scripts/tests/test_graph_data.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graph_tokenization import AutoGraphTokenizer, SimpleGraphData


_GRAPH_DATA_PATH = Path(__file__).resolve().parents[2] / "dllm" / "utils" / "graph_data.py"
_GRAPH_DATA_SPEC = importlib.util.spec_from_file_location(
    "graph_data_under_test", _GRAPH_DATA_PATH
)
assert _GRAPH_DATA_SPEC is not None and _GRAPH_DATA_SPEC.loader is not None
_GRAPH_DATA_MODULE = importlib.util.module_from_spec(_GRAPH_DATA_SPEC)
_GRAPH_DATA_SPEC.loader.exec_module(_GRAPH_DATA_MODULE)

infer_graph_tokenizer_stats = _GRAPH_DATA_MODULE.infer_graph_tokenizer_stats
sample_k_hop_subgraphs = _GRAPH_DATA_MODULE.sample_k_hop_subgraphs
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


def test_autograph_decode_strict_raises_on_out_of_range_token():
    graph = SimpleGraphData(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        num_nodes=3,
    )
    tokenizer = AutoGraphTokenizer(max_length=-1, undirected=True, append_eos=True)
    tokenizer.set_num_nodes(3)

    tokens = tokenizer.tokenize(graph).clone()
    tokens[1] = len(tokenizer) + 1

    with pytest.raises(ValueError, match="out of range"):
        tokenizer.decode(tokens, strict=True)


def test_autograph_decode_strict_raises_on_bad_grammar():
    graph = SimpleGraphData(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        num_nodes=3,
    )
    tokenizer = AutoGraphTokenizer(max_length=-1, undirected=True, append_eos=True)
    tokenizer.set_num_nodes(3)

    tokens = tokenizer.tokenize(graph).clone()
    tokens[1] = tokenizer.ladj

    with pytest.raises(ValueError, match="Expected node token"):
        tokenizer.decode(tokens, strict=True)


def test_infer_graph_tokenizer_stats_uses_graph_size():
    graph = SimpleGraphData(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        num_nodes=3,
    )

    stats = infer_graph_tokenizer_stats([graph], labeled_graph=False)

    assert stats["max_num_nodes"] == 3
    assert stats["num_node_types"] == 0
    assert stats["num_edge_types"] == 0


def test_sample_k_hop_subgraphs_extracts_centered_subgraphs():
    graph = Data(
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
            dtype=torch.long,
        ),
        num_nodes=4,
        x=torch.arange(4, dtype=torch.long).view(-1, 1),
    )

    samples = sample_k_hop_subgraphs(
        graph,
        centers=[1, 2],
        num_hops=1,
        max_samples=2,
        seed=0,
        dataset_name="EllipticBitcoinDataset",
    )

    assert len(samples) == 2
    assert all(sample.dataset_name == "EllipticBitcoinDataset" for sample in samples)
    assert all(sample.num_nodes >= 2 for sample in samples)
    assert all(sample.edge_index.size(0) == 2 for sample in samples)