"""
Import these helpers from your training or preprocessing script to convert
graph samples into AutoGraph token sequences.

Example:
    from dllm.utils.graph_data import tokenize_graphs
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import torch

from graph_tokenization import SimpleGraphData


def _coerce_label_tensor(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if value.numel() == 0:
        return value.reshape(-1).to(dtype=torch.long)
    if value.ndim > 1:
        if value.shape[-1] == 1:
            value = value.squeeze(-1)
        else:
            value = value.argmax(dim=-1)
    return value.reshape(-1).to(dtype=torch.long)


def pyg_graph_to_simple_graph_data(
    graph: Any,
    *,
    labeled_graph: bool = False,
    dataset_name: str | None = None,
) -> SimpleGraphData:
    edge_index = getattr(graph, "edge_index", None)
    if edge_index is None:
        raise ValueError("graph must provide edge_index")

    num_nodes = getattr(graph, "num_nodes", None)
    if num_nodes is None:
        edge_index_tensor = torch.as_tensor(edge_index)
        if edge_index_tensor.numel() == 0:
            num_nodes = 0
        else:
            num_nodes = int(edge_index_tensor.max().item()) + 1

    x = getattr(graph, "x", None)
    edge_attr = getattr(graph, "edge_attr", None)
    if labeled_graph:
        x = _coerce_label_tensor(x)
        edge_attr = _coerce_label_tensor(edge_attr)
        if x is None:
            raise ValueError("labeled graph tokenization requires node labels in x")
        if edge_attr is None:
            raise ValueError(
                "labeled graph tokenization requires edge labels in edge_attr"
            )
    else:
        x = None
        edge_attr = None

    return SimpleGraphData(
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
        num_nodes=int(num_nodes),
        x=x,
        edge_attr=edge_attr,
        dataset_name=dataset_name,
    )


def infer_graph_tokenizer_stats(
    graphs: Iterable[Any],
    *,
    labeled_graph: bool = False,
) -> dict[str, int]:
    max_num_nodes = 0
    num_node_types = 0
    num_edge_types = 0

    for graph in graphs:
        data = pyg_graph_to_simple_graph_data(graph, labeled_graph=labeled_graph)
        max_num_nodes = max(max_num_nodes, int(data.num_nodes))

        if labeled_graph:
            assert data.x is not None
            assert data.edge_attr is not None
            if data.x.numel() > 0:
                num_node_types = max(num_node_types, int(data.x.max().item()) + 1)
            if data.edge_attr.numel() > 0:
                num_edge_types = max(num_edge_types, int(data.edge_attr.max().item()) + 1)

    return {
        "max_num_nodes": max_num_nodes,
        "num_node_types": num_node_types,
        "num_edge_types": num_edge_types,
    }

def tokenize_graphs_with_strategy(
    graphs,
    graph_tokenizer,
    lm_strategy: GraphLMTokenizerStrategy,
    *,
    labeled_graph: bool = False,
    dataset_name: str | None = None,
) -> list[dict[str, list[int]]]:
    """
    Like the original but converts graph token ids through
    the chosen LM strategy instead of a raw integer offset shift.
 
    The returned rows have `input_ids` and `labels` as lists of LLM token ids
    ready to be fed into MDLMTrainer.
    """
    # Import here to avoid circular deps when used standalone
    from dllm.utils.graph_data import pyg_graph_to_simple_graph_data
 
    rows: list[dict[str, list[int]]] = []
    for graph in graphs:
        data = pyg_graph_to_simple_graph_data(
            graph,
            labeled_graph=labeled_graph,
            dataset_name=dataset_name,
        )
        graph_tokens = graph_tokenizer.tokenize(data)
        lm_ids = lm_strategy.encode(graph_tokens.tolist())
        if not lm_ids:
            continue
        rows.append({"input_ids": lm_ids, "labels": lm_ids.copy()})
    return rows

# def tokenize_graphs(
#     graphs: Iterable[Any],
#     graph_tokenizer,
#     *,
#     token_offset: int,
#     labeled_graph: bool = False,
#     dataset_name: str | None = None,
# ) -> list[dict[str, list[int]]]:
def tokenize_graphs(
    graphs: Iterable[Any],
    graph_tokenizer,
    *,
    token_offset: int,
    labeled_graph: bool = False,
    dataset_name: str | None = None,
) -> list[dict[str, list[int]]]:
    """
    Original approach: raw id shift into a reserved embedding block.

    Returns rows with `input_ids` and `labels` as lists of ints.
    """
    rows: list[dict[str, list[int]]] = []

    for graph in graphs:
        data = pyg_graph_to_simple_graph_data(
            graph,
            labeled_graph=labeled_graph,
            dataset_name=dataset_name,
        )
        tokens = graph_tokenizer.tokenize(data)
        shifted = torch.as_tensor(tokens, dtype=torch.long) + int(token_offset)
        if shifted.numel() == 0:
            continue
        sequence = shifted.tolist()
        rows.append({"input_ids": sequence, "labels": sequence.copy()})

    return rows


def sample_k_hop_subgraphs(
    graph: Any,
    centers: Iterable[int],
    *,
    num_hops: int = 2,
    max_samples: int = 256,
    seed: int = 0,
    dataset_name: str | None = None,
) -> list[Any]:
    """Sample graph-level training examples from a large PyG graph.

    This is useful for node-level datasets such as Elliptic Bitcoin, where we
    want to fine-tune a graph tokenizer on many smaller subgraphs instead of a
    single giant graph.
    """
    from torch_geometric.data import Data
    from torch_geometric.utils import k_hop_subgraph

    edge_index = torch.as_tensor(graph.edge_index, dtype=torch.long)
    num_nodes = getattr(graph, "num_nodes", None)
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else 0

    ordered_centers = [int(center) for center in centers]
    if not ordered_centers:
        return []

    random.Random(seed).shuffle(ordered_centers)

    x = getattr(graph, "x", None)
    edge_attr = getattr(graph, "edge_attr", None)

    samples: list[Any] = []
    for center in ordered_centers[:max_samples]:
        subset, sub_edge_index, _, edge_mask = k_hop_subgraph(
            center,
            num_hops,
            edge_index,
            relabel_nodes=True,
            num_nodes=int(num_nodes),
        )
        if sub_edge_index.numel() == 0 and subset.numel() == 0:
            continue

        sample = Data(edge_index=sub_edge_index, num_nodes=int(subset.numel()))
        if x is not None:
            sample.x = x[subset]
        if edge_attr is not None:
            sample.edge_attr = edge_attr[edge_mask]
        if dataset_name is not None:
            sample.dataset_name = dataset_name
        samples.append(sample)

    return samples


def save_graph_tokenizer_metadata(output_dir: str | Path, metadata: dict[str, Any]) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "graph_tokenizer_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )