"""
Import these helpers from your training or preprocessing script to convert
graph samples into AutoGraph token sequences.

Example:
    from dllm.utils.graph_data import tokenize_graphs
"""

from __future__ import annotations

import json
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
    lm_strategy,
    *,
    labeled_graph: bool = False,
    dataset_name: str | None = None,
) -> list[dict[str, list[int]]]:
    """
    Convert graph samples to LLM token id sequences via a strategy object.

    The graph tokenizer produces its own integer ids; lm_strategy.encode()
    converts these to LLM vocabulary ids, using graph_tokens_to_text internally
    so that node ids become plain digit strings and structural tokens become
    either text fragments (TextMapping) or atomic special tokens
    (SelectiveSpecialTokens).

    The returned rows have `input_ids` and `labels` as lists of true LLM
    token ids, ready to be fed into MDLMTrainer.
    """
    rows: list[dict[str, list[int]]] = []
    for graph in graphs:
        data = pyg_graph_to_simple_graph_data(
            graph,
            labeled_graph=labeled_graph,
            dataset_name=dataset_name,
        )
        graph_tokens = graph_tokenizer.tokenize(data)
        # encode() converts graph vocab ids → text → LLM vocab ids
        lm_ids = lm_strategy.encode(graph_tokens.tolist())
        if not lm_ids:
            continue
        rows.append({"input_ids": lm_ids, "labels": lm_ids.copy()})
    return rows


# Original strategy — unchanged: raw id shift into a reserved embedding block.
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

    The graph tokenizer's integer ids are shifted by token_offset so they
    fall into a reserved region of the LLM's extended embedding matrix.
    input_ids are therefore in the shifted graph-vocab space, not the LLM's
    natural vocabulary — this is intentional and consistent with how the
    embedding matrix is resized in train.py for the 'original' strategy.

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


def save_graph_tokenizer_metadata(output_dir: str | Path, metadata: dict[str, Any]) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "graph_tokenizer_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    
# """
# Import these helpers from your training or preprocessing script to convert
# graph samples into AutoGraph token sequences.

# Example:
#     from dllm.utils.graph_data import tokenize_graphs
# """

# from __future__ import annotations

# import json
# from pathlib import Path
# from typing import Any, Iterable

# import torch

# from graph_tokenization import SimpleGraphData


# def _coerce_label_tensor(value: Any) -> torch.Tensor | None:
#     if value is None:
#         return None
#     if not isinstance(value, torch.Tensor):
#         value = torch.as_tensor(value)
#     if value.numel() == 0:
#         return value.reshape(-1).to(dtype=torch.long)
#     if value.ndim > 1:
#         if value.shape[-1] == 1:
#             value = value.squeeze(-1)
#         else:
#             value = value.argmax(dim=-1)
#     return value.reshape(-1).to(dtype=torch.long)


# def pyg_graph_to_simple_graph_data(
#     graph: Any,
#     *,
#     labeled_graph: bool = False,
#     dataset_name: str | None = None,
# ) -> SimpleGraphData:
#     edge_index = getattr(graph, "edge_index", None)
#     if edge_index is None:
#         raise ValueError("graph must provide edge_index")

#     num_nodes = getattr(graph, "num_nodes", None)
#     if num_nodes is None:
#         edge_index_tensor = torch.as_tensor(edge_index)
#         if edge_index_tensor.numel() == 0:
#             num_nodes = 0
#         else:
#             num_nodes = int(edge_index_tensor.max().item()) + 1

#     x = getattr(graph, "x", None)
#     edge_attr = getattr(graph, "edge_attr", None)
#     if labeled_graph:
#         x = _coerce_label_tensor(x)
#         edge_attr = _coerce_label_tensor(edge_attr)
#         if x is None:
#             raise ValueError("labeled graph tokenization requires node labels in x")
#         if edge_attr is None:
#             raise ValueError(
#                 "labeled graph tokenization requires edge labels in edge_attr"
#             )
#     else:
#         x = None
#         edge_attr = None

#     return SimpleGraphData(
#         edge_index=torch.as_tensor(edge_index, dtype=torch.long),
#         num_nodes=int(num_nodes),
#         x=x,
#         edge_attr=edge_attr,
#         dataset_name=dataset_name,
#     )


# def infer_graph_tokenizer_stats(
#     graphs: Iterable[Any],
#     *,
#     labeled_graph: bool = False,
# ) -> dict[str, int]:
#     max_num_nodes = 0
#     num_node_types = 0
#     num_edge_types = 0

#     for graph in graphs:
#         data = pyg_graph_to_simple_graph_data(graph, labeled_graph=labeled_graph)
#         max_num_nodes = max(max_num_nodes, int(data.num_nodes))

#         if labeled_graph:
#             assert data.x is not None
#             assert data.edge_attr is not None
#             if data.x.numel() > 0:
#                 num_node_types = max(num_node_types, int(data.x.max().item()) + 1)
#             if data.edge_attr.numel() > 0:
#                 num_edge_types = max(num_edge_types, int(data.edge_attr.max().item()) + 1)

#     return {
#         "max_num_nodes": max_num_nodes,
#         "num_node_types": num_node_types,
#         "num_edge_types": num_edge_types,
#     }

# def tokenize_graphs_with_strategy(
#     graphs,
#     graph_tokenizer,
#     lm_strategy: GraphLMTokenizerStrategy,
#     *,
#     labeled_graph: bool = False,
#     dataset_name: str | None = None,
# ) -> list[dict[str, list[int]]]:
#     """
#     Like the original but converts graph token ids through
#     the chosen LM strategy instead of a raw integer offset shift.
 
#     The returned rows have `input_ids` and `labels` as lists of LLM token ids
#     ready to be fed into MDLMTrainer.
#     """
#     # Import here to avoid circular deps when used standalone
#     from dllm.utils.graph_data import pyg_graph_to_simple_graph_data
 
#     rows: list[dict[str, list[int]]] = []
#     for graph in graphs:
#         data = pyg_graph_to_simple_graph_data(
#             graph,
#             labeled_graph=labeled_graph,
#             dataset_name=dataset_name,
#         )
#         graph_tokens = graph_tokenizer.tokenize(data)
#         lm_ids = lm_strategy.encode(graph_tokens.tolist())
#         if not lm_ids:
#             continue
#         rows.append({"input_ids": lm_ids, "labels": lm_ids.copy()})
#     return rows

# # def tokenize_graphs(
# #     graphs: Iterable[Any],
# #     graph_tokenizer,
# #     *,
# #     token_offset: int,
# #     labeled_graph: bool = False,
# #     dataset_name: str | None = None,
# # ) -> list[dict[str, list[int]]]:

# def tokenize_graphs(
#     graphs: Iterable[Any],
#     graph_tokenizer,
#     *,
#     token_offset: int,
#     labeled_graph: bool = False,
#     dataset_name: str | None = None,
# ) -> list[dict[str, list[int]]]:
#     """
#     Original approach: raw id shift into a reserved embedding block.

#     Returns rows with `input_ids` and `labels` as lists of ints.
#     """
#     rows: list[dict[str, list[int]]] = []

#     for graph in graphs:
#         data = pyg_graph_to_simple_graph_data(
#             graph,
#             labeled_graph=labeled_graph,
#             dataset_name=dataset_name,
#         )
#         tokens = graph_tokenizer.tokenize(data)
#         shifted = torch.as_tensor(tokens, dtype=torch.long) + int(token_offset)
#         if shifted.numel() == 0:
#             continue
#         sequence = shifted.tolist()
#         rows.append({"input_ids": sequence, "labels": sequence.copy()})

#     return rows


# def save_graph_tokenizer_metadata(output_dir: str | Path, metadata: dict[str, Any]) -> None:
#     path = Path(output_dir)
#     path.mkdir(parents=True, exist_ok=True)
#     (path / "graph_tokenizer_metadata.json").write_text(
#         json.dumps(metadata, indent=2, sort_keys=True) + "\n",
#         encoding="utf-8",
#     )