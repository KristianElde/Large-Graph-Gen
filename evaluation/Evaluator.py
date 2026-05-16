from dataclasses import dataclass
from typing import Sequence

import numpy as np

import networkx as nx
import torch


from graph_tokenization import (
    AutoGraphTokenizer,
    GraphTokenizer,
    SimpleGraphData,
    simpleGraph_to_networkx,
)


@dataclass
class Evaluator:
    tokenizer: GraphTokenizer = AutoGraphTokenizer()

    def __call__(
        self,
        tokenized_graphs: torch.Tensor,
        train_data: Sequence[SimpleGraphData],
    ):
        total_graphs = tokenized_graphs.shape[0]

        graphs = []
        i = 0
        for tokens in tokenized_graphs:
            try:
                graph = self.tokenizer.decode(tokens)
                graphs.append(graph)
            except:
                i += 1
                print(f"Unparseable graph number {i}")

        valid_graphs = len(graphs)

        validity = valid_graphs / total_graphs
        nx_graphs = self._to_networkx_graphs(graphs)
        train_nx_graphs = self._to_networkx_graphs(train_data)

        graph_stats = self.graph_stats(nx_graphs)
        ple_stats = self.power_law_exponent(nx_graphs)
        edge_overlap = self.edge_overlap(nx_graphs, train_nx_graphs)
        uniqueness, novelty = self.eval_fraction_unique_non_isomorphic(
            nx_graphs, train_nx_graphs, total_gen_graphs=total_graphs
        )

        return {
            "validity": validity,
            "uniqueness": uniqueness,
            "novelty": novelty,
            **graph_stats,
            **ple_stats,
            **edge_overlap,
        }

    def _to_networkx_graphs(self, graphs: Sequence[SimpleGraphData]) -> list[nx.Graph]:
        return [simpleGraph_to_networkx(graph) for graph in graphs]

    def graph_stats(self, graphs: Sequence[nx.Graph]) -> dict[str, float | int]:
        total_graphs = len(graphs)

        if total_graphs == 0:
            return {
                "max_degree": 0.0,
                "min_degree": 0.0,
                "avg_degree": 0.0,
                "avg_num_connected_components": 0.0,
                "avg_largest_connected_component_size": 0.0,
                "total_nodes": 0,
                "total_edges": 0,
            }

        graph_max_degrees = []
        graph_min_degrees = []
        graph_avg_degrees = []
        graph_component_counts = []
        graph_largest_component_sizes = []

        for graph in graphs:
            degrees = [degree for _, degree in graph.degree()]
            if degrees:
                graph_max_degrees.append(max(degrees))
                graph_min_degrees.append(min(degrees))
                graph_avg_degrees.append(sum(degrees) / len(degrees))
            else:
                graph_max_degrees.append(0)
                graph_min_degrees.append(0)
                graph_avg_degrees.append(0.0)

            components = list(nx.connected_components(graph))
            graph_component_counts.append(len(components))
            graph_largest_component_sizes.append(
                max((len(component) for component in components), default=0)
            )

        total_nodes = sum(graph.number_of_nodes() for graph in graphs)
        total_edges = sum(graph.number_of_edges() for graph in graphs)

        max_degree = sum(graph_max_degrees) / total_graphs
        min_degree = sum(graph_min_degrees) / total_graphs
        avg_degree = sum(graph_avg_degrees) / total_graphs
        avg_num_connected_components = sum(graph_component_counts) / total_graphs
        avg_largest_connected_component_size = sum(
            graph_largest_component_sizes
        ) / total_graphs

        return {
            "max_degree": max_degree,
            "min_degree": min_degree,
            "avg_degree": avg_degree,
            "avg_num_connected_components": avg_num_connected_components,
            "avg_largest_connected_component_size": avg_largest_connected_component_size,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
        }

    def eval_fraction_unique_non_isomorphic(
        self,
        gen_graphs: Sequence[nx.Graph],
        train_graphs: Sequence[nx.Graph],
        total_gen_graphs,
    ):
        count_isomorphic = 0
        count_non_unique = 0

        gen_evaluated = []
        for gen_g in gen_graphs:
            unique = True

            for gen_old in gen_evaluated:
                if nx.faster_could_be_isomorphic(gen_g, gen_old):
                    if nx.is_isomorphic(gen_g, gen_old):
                        count_non_unique += 1
                        unique = False
                        break

            if not unique:
                continue

            gen_evaluated.append(gen_g)

            for train_g in train_graphs:
                if nx.faster_could_be_isomorphic(gen_g, train_g):
                    if nx.is_isomorphic(gen_g, train_g):
                        count_isomorphic += 1
                        break

        frac_unique = (float(len(gen_graphs)) - count_non_unique) / float(
            total_gen_graphs
        )  # Fraction of distinct isomorphism classes in the gen graphs
        frac_unique_non_isomorphic = (
            float(len(gen_graphs)) - count_non_unique - count_isomorphic
        ) / float(
            total_gen_graphs
        )  # Fraction of distinct isomorphism classes in the gen graphs that are not in the training set
        return (
            frac_unique,
            frac_unique_non_isomorphic,
        )

    def power_law_exponent(
        self,
        graphs: Sequence[nx.Graph],
        k_min: int = 1,
        min_tail_size: int = 10,
    ) -> dict[str, float | int]:
        """
        Estimate the power-law exponent (PLE / gamma) of the degree distribution.

        Uses the common discrete-degree approximation:
            gamma = 1 + n / sum(log(k / (k_min - 0.5)))

        Returns mean/std over graphs.
        """
        gammas: list[float] = []

        for graph in graphs:
            degrees = [degree for _, degree in graph.degree()]
            tail_degrees = [d for d in degrees if d >= k_min]

            if len(tail_degrees) < min_tail_size:
                continue

            denominator = sum(np.log(d / (k_min - 0.5)) for d in tail_degrees if d > 0)

            if denominator <= 0:
                continue

            gamma = 1.0 + len(tail_degrees) / denominator
            gammas.append(gamma)

        if not gammas:
            return {
                "ple_mean": 0.0,
                "ple_std": 0.0,
            }

        mean_gamma = sum(gammas) / len(gammas)

        if len(gammas) > 1:
            variance = sum((g - mean_gamma) ** 2 for g in gammas) / (len(gammas) - 1)
            std_gamma = np.sqrt(variance)
        else:
            std_gamma = 0.0

        return {
            "ple_mean": mean_gamma,
            "ple_std": std_gamma,
        }

    def edge_overlap(
        self,
        gen_graphs: Sequence[nx.Graph],
        train_graphs: Sequence[nx.Graph],
    ) -> dict[str, float | int]:
        """
        EDGE-style edge overlap.

        Approximation to max edge overlap over node permutations:
        1. Sort nodes in each graph by ascending degree.
        2. Relabel nodes by this degree order: lowest-degree node -> 0, etc.
        3. Compute edge overlap after this alignment.

        EO(G_gen, G_train) = |E_gen ∩ E_train| / |E_gen|

        If multiple train graphs are given, each generated graph is compared to all
        train graphs and the maximum overlap is used.
        """

        import math

        def degree_sorted_edge_set(graph: nx.Graph) -> set[tuple[int, int]]:
            """
            Relabel graph by ascending node degree and return canonical edge set.
            Ties are broken deterministically by string form of node ID.
            """
            sorted_nodes = sorted(
                graph.nodes(),
                key=lambda node: (graph.degree[node], str(node)),
            )

            node_to_rank = {node: rank for rank, node in enumerate(sorted_nodes)}

            if graph.is_directed():
                return {
                    (node_to_rank[u], node_to_rank[v])
                    for u, v in graph.edges()
                    if u in node_to_rank and v in node_to_rank
                }

            return {
                tuple(sorted((node_to_rank[u], node_to_rank[v])))
                for u, v in graph.edges()
                if u in node_to_rank and v in node_to_rank and u != v
            }

        train_edge_sets = [
            degree_sorted_edge_set(train_graph) for train_graph in train_graphs
        ]

        overlaps: list[float] = []

        for gen_graph in gen_graphs:
            gen_edges = degree_sorted_edge_set(gen_graph)

            if len(gen_edges) == 0:
                overlaps.append(0.0)
                continue

            best_overlap = 0.0

            for train_edges in train_edge_sets:
                overlap = len(gen_edges & train_edges) / len(gen_edges)
                best_overlap = max(best_overlap, overlap)

            overlaps.append(best_overlap)

        if not overlaps:
            return {
                "edge_overlap_mean": 0.0,
                "edge_overlap_std": 0.0,
                "edge_overlap_num_graphs": 0,
            }

        mean_overlap = sum(overlaps) / len(overlaps)

        if len(overlaps) > 1:
            variance = sum((x - mean_overlap) ** 2 for x in overlaps) / (
                len(overlaps) - 1
            )
            std_overlap = math.sqrt(variance)
        else:
            std_overlap = 0.0

        return {
            "edge_overlap_mean": mean_overlap,
            "edge_overlap_std": std_overlap,
            "edge_overlap_num_graphs": len(overlaps),
        }
