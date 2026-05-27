from __future__ import annotations

from typing import Any

import networkx as nx

from .types import SimpleGraphData


def simpleGraph_to_networkx(
    data: SimpleGraphData,
) -> nx.Graph:
    """Convert ``SimpleGraphData`` directly to a NetworkX graph."""
    graph = nx.Graph()

    for node_idx in range(data.num_nodes):
        attrs: dict[str, Any] = {}
        graph.add_node(node_idx, **attrs)

    edge_index = data.edge_index.detach().cpu()
    edge_attr = data.edge_attr.detach().cpu() if data.edge_attr is not None else None

    for edge_idx, (src, dst) in enumerate(edge_index.t().tolist()):
        attrs = {}
        graph.add_edge(int(src), int(dst), **attrs)

    return graph
