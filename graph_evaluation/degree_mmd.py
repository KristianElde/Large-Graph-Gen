import networkx as nx
from typing import Sequence
import numpy as np


class DegreeMMD:

    def __call__(
        self,
        gen_graphs: Sequence[nx.Graph],
        train_graphs: Sequence[nx.Graph],
        max_degree: int | None = None,
        bandwidth: float | None = None,
    ) -> dict[str, float | int]:
        """
        MMD between generated and training graphs using degree histograms.

        Each graph is represented by a normalized degree histogram.
        Then MMD compares the distribution of generated histograms against
        the distribution of training histograms.

        Lower is better.
        """

        if len(gen_graphs) == 0 or len(train_graphs) == 0:
            return {
                "degree_mmd": 0.0,
                "degree_mmd_bandwidth": 0.0,
                "degree_mmd_num_gen_graphs": len(gen_graphs),
                "degree_mmd_num_train_graphs": len(train_graphs),
            }

        if max_degree is None:
            max_degree = self._max_degree_across_graphs([*gen_graphs, *train_graphs])

        gen_features = np.stack(
            [self._degree_histogram(graph, max_degree) for graph in gen_graphs],
            axis=0,
        )
        train_features = np.stack(
            [self._degree_histogram(graph, max_degree) for graph in train_graphs],
            axis=0,
        )

        if bandwidth is None:
            bandwidth = self._median_heuristic_bandwidth(
                np.concatenate([gen_features, train_features], axis=0)
            )

        mmd = self._rbf_mmd_biased(
            train_features,
            gen_features,
            bandwidth=bandwidth,
        )

        return {
            "degree_mmd": mmd,
            "degree_mmd_bandwidth": bandwidth,
            "degree_mmd_num_gen_graphs": len(gen_graphs),
            "degree_mmd_num_train_graphs": len(train_graphs),
        }

    def _degree_histogram(self, graph: nx.Graph, max_degree: int) -> np.ndarray:
        """
        Convert one graph into a normalized degree histogram.

        Histogram bins:
            index 0 = degree 0
            index 1 = degree 1
            ...
            index max_degree = degree >= max_degree
        """

        degrees = np.array([degree for _, degree in graph.degree()], dtype=np.int64)

        hist = np.zeros(max_degree + 1, dtype=np.float64)

        if len(degrees) == 0:
            hist[0] = 1.0
            return hist

        clipped_degrees = np.clip(degrees, 0, max_degree)

        for degree in clipped_degrees:
            hist[degree] += 1.0

        hist_sum = hist.sum()
        if hist_sum > 0:
            hist /= hist_sum

        return hist

    def _max_degree_across_graphs(self, graphs: Sequence[nx.Graph]) -> int:
        max_degree = 0

        for graph in graphs:
            degrees = [degree for _, degree in graph.degree()]
            if degrees:
                max_degree = max(max_degree, max(degrees))

        return max_degree

    def _rbf_mmd_biased(
        self,
        x: np.ndarray,
        y: np.ndarray,
        bandwidth: float,
    ) -> float:
        """
        Biased squared MMD with RBF kernel.

        MMD^2 = mean k(x, x) + mean k(y, y) - 2 mean k(x, y)
        """

        if bandwidth <= 0:
            bandwidth = 1.0

        k_xx = self._rbf_kernel(x, x, bandwidth)
        k_yy = self._rbf_kernel(y, y, bandwidth)
        k_xy = self._rbf_kernel(x, y, bandwidth)

        mmd_squared = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()

        return float(max(mmd_squared, 0.0))

    def _rbf_kernel(
        self,
        x: np.ndarray,
        y: np.ndarray,
        bandwidth: float,
    ) -> np.ndarray:
        x_norm = np.sum(x * x, axis=1, keepdims=True)
        y_norm = np.sum(y * y, axis=1, keepdims=True).T

        squared_distances = x_norm + y_norm - 2.0 * x @ y.T
        squared_distances = np.maximum(squared_distances, 0.0)

        return np.exp(-squared_distances / (2.0 * bandwidth**2))

    def _median_heuristic_bandwidth(self, features: np.ndarray) -> float:
        """
        Median heuristic for RBF bandwidth.

        Uses pairwise Euclidean distances between descriptor vectors.
        """

        if len(features) < 2:
            return 1.0

        x_norm = np.sum(features * features, axis=1, keepdims=True)
        squared_distances = x_norm + x_norm.T - 2.0 * features @ features.T
        squared_distances = np.maximum(squared_distances, 0.0)

        distances = np.sqrt(squared_distances)

        # Exclude diagonal zeros.
        nonzero_distances = distances[distances > 0]

        if len(nonzero_distances) == 0:
            return 1.0

        return float(np.median(nonzero_distances))
