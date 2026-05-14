from typing import Sequence

import networkx as nx
import torch

from graph_tokenization.autograph import AutoGraphTokenizer
from graph_tokenization.types import SimpleGraphData
from graph_tokenization.networkx_utils import simple_graph_to_networkx


def eval_fraction_unique_non_isomorphic(
    gen_graphs: SimpleGraphData,
    train_graphs: SimpleGraphData,
    total_gen_graphs,
):
    count_isomorphic = 0
    count_non_unique = 0

    gen_graphs = [simple_graph_to_networkx(g) for g in gen_graphs]
    train_graphs = [simple_graph_to_networkx(g) for g in train_graphs]

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
