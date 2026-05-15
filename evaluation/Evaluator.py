from dataclasses import dataclass
from typing import Sequence

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

    def __call__(self, tokenized_graphs: torch.Tensor, train_data: torch.Tensor):
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
        graph_stats = self.graph_stats(graphs)
        uniqueness, novelty = self.eval_fraction_unique_non_isomorphic(
            graphs, train_data, total_gen_graphs=total_graphs
        )

        return {
            "validity": validity,
            "uniqueness": uniqueness,
            "novelty": novelty,
            **graph_stats,
        }

    def graph_stats(self, graphs: Sequence[SimpleGraphData]) -> dict[str, float | int]:
        total_graphs = len(graphs)

        if total_graphs == 0:
            return {
                "max_degree": 0.0,
                "min_degree": 0.0,
                "avg_degree": 0.0,
                "total_nodes": 0,
                "total_edges": 0,
            }

        nx_graphs = [simpleGraph_to_networkx(g) for g in graphs]
        graph_max_degrees = []
        graph_min_degrees = []
        graph_avg_degrees = []

        for graph in nx_graphs:
            degrees = [degree for _, degree in graph.degree()]
            if degrees:
                graph_max_degrees.append(max(degrees))
                graph_min_degrees.append(min(degrees))
                graph_avg_degrees.append(sum(degrees) / len(degrees))
            else:
                graph_max_degrees.append(0)
                graph_min_degrees.append(0)
                graph_avg_degrees.append(0.0)

        total_nodes = sum(graph.number_of_nodes() for graph in nx_graphs)
        total_edges = sum(graph.number_of_edges() for graph in nx_graphs)

        max_degree = sum(graph_max_degrees) / total_graphs
        min_degree = sum(graph_min_degrees) / total_graphs
        avg_degree = sum(graph_avg_degrees) / total_graphs

        return {
            "max_degree": max_degree,
            "min_degree": min_degree,
            "avg_degree": avg_degree,
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

        gen_graphs = [simpleGraph_to_networkx(g) for g in gen_graphs]
        train_graphs = [simpleGraph_to_networkx(g) for g in train_graphs]

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
