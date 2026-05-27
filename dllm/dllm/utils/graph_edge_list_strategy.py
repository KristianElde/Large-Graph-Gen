from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch


EDGE_LIST_PROMPT = "Graph format: N=number; M=number; E=(source,target),(source,target). Generate:"

def _num_nodes_from_graph(graph: Any) -> int:
    edge_index = torch.as_tensor(graph.edge_index, dtype=torch.long)
    num_nodes = getattr(graph, "num_nodes", None)
    if num_nodes is None:
        return int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else 0
    return int(num_nodes)


def pyg_graph_to_edge_list_string(
    graph: Any,
    *,
    undirected: bool = True,
    labeled: bool = False,
) -> str:
    if labeled:
        raise NotImplementedError("edge_list currently supports unlabeled graphs only")

    edge_index = torch.as_tensor(graph.edge_index, dtype=torch.long)
    n = _num_nodes_from_graph(graph)

    edges: set[tuple[int, int]] = set()

    if edge_index.numel() > 0:
        for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            u, v = int(u), int(v)

            if u == v:
                continue
            if not (0 <= u < n and 0 <= v < n):
                continue

            if undirected:
                u, v = sorted((u, v))

            edges.add((u, v))

    edge_text = ",".join(f"({u},{v})" for u, v in sorted(edges))
    return f"N={n}; M={len(edges)}; E={edge_text}"


class GraphEdgeListStrategy:
    strategy: str = "edge_list"
    prompt: str = EDGE_LIST_PROMPT

    def __init__(self, tokenizer, *, undirected: bool = True, labeled: bool = False):
        if labeled:
            raise NotImplementedError("edge_list currently supports unlabeled graphs only")
        self.tokenizer = tokenizer
        self.undirected = undirected
        self.labeled = labeled

    def encode_to_text(self, graph: Any) -> str:
        return pyg_graph_to_edge_list_string(
            graph,
            undirected=self.undirected,
            labeled=self.labeled,
        )

    def encode(self, graph: Any) -> list[int]:
        return self.tokenizer.encode(self.encode_to_text(graph), add_special_tokens=False)

    def save_metadata(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        meta = {
            "strategy": self.strategy,
            "labeled": self.labeled,
            "undirected": self.undirected,
            "new_tokens_added": 0,
            "embedding_resize": False,
            "format": "N=5; E=(0,1),(1,2),(3,4)",
            "prompt": self.prompt,
        }

        (path / "graph_lm_strategy.json").write_text(
            json.dumps(meta, indent=2) + "\n",
            encoding="utf-8",
        )


def build_edge_list_strategy(
    tokenizer,
    *,
    undirected: bool = True,
    labeled: bool = False,
    model=None,
) -> GraphEdgeListStrategy:
    return GraphEdgeListStrategy(
        tokenizer,
        undirected=undirected,
        labeled=labeled,
    )