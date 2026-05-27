
from dllm.utils.graph_data import sample_k_hop_subgraphs
import random
import torch

def load_pyg_dataset(dataset_name: str, root: str):
    try:
        from torch_geometric.datasets import TUDataset
    except ImportError as exc:
        raise ImportError(
            "torch-geometric is required for graph fine-tuning. "
            "Install it before running this script."
        ) from exc

    dataset = TUDataset(root=root, name=dataset_name)
    if len(dataset) == 0:
        raise ValueError(f"PyG dataset '{dataset_name}' is empty.")
    return dataset


def select_graphs(dataset, max_graphs: int, seed: int):
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    if max_graphs > 0:
        indices = indices[: min(max_graphs, len(indices))]
    return [dataset[i] for i in indices]

def normalize_dataset_name(dataset_name: str) -> str:
    return dataset_name.strip().lower().replace("-", "_")


def load_graph_samples(data_args, seed: int):
    """
    Unified loader for all supported PyG datasets.
    Returns (graphs, canonical_dataset_name).
    Replaces the separate load_pyg_dataset + select_graphs calls in train().
    """
    try:
        from torch_geometric.datasets import EllipticBitcoinDataset, MalNetTiny, TUDataset
    except ImportError as exc:
        raise ImportError(
            "torch-geometric is required for graph fine-tuning. "
            "Install it before running this script."
        ) from exc

    dataset_key = normalize_dataset_name(data_args.pyg_dataset)

    if dataset_key in {"malnet", "malnet_tiny", "malnettiny"}:
        dataset = MalNetTiny(root=data_args.data_root, split=None)
        if len(dataset) == 0:
            raise ValueError("PyG dataset 'MalNetTiny' is empty.")
            
        indices = list(range(len(dataset)))
        random.Random(seed).shuffle(indices)
        if data_args.max_graphs > 0:
            indices = indices[: min(data_args.max_graphs, len(indices))]
            
        from torch_geometric.utils import k_hop_subgraph, subgraph
        
        processed_graphs = []
        # Define a hard limit to protect your H100 (e.g., 512 or 1024 nodes)
        # You can add this to your DataArguments, defaulting to 512 here.
        max_nodes = getattr(data_args, "max_nodes_per_graph", 1024) 
        num_hops = getattr(data_args, "malnet_num_hops", 2)

        for i in indices:
            data = dataset[i]
            
            if data.num_nodes > max_nodes:
                # 1. Pick a random central node
                central_node = random.randint(0, int(data.num_nodes) - 1)
                
                # 2. Extract local neighborhood
                subset, edge_index, mapping, edge_mask = k_hop_subgraph(
                    node_idx=central_node,
                    num_hops=num_hops,
                    edge_index=data.edge_index,
                    relabel_nodes=True,
                    num_nodes=data.num_nodes
                )
                
                # 3. If the local neighborhood is STILL too big, apply a hard truncation
                if len(subset) > max_nodes:
                    # Keep the central node (now at index 'mapping'), and randomly sample the rest
                    subset_indices = torch.randperm(len(subset))[:max_nodes]
                    if mapping not in subset_indices:
                        subset_indices[0] = mapping # Ensure central node is kept
                    
                    final_subset = subset[subset_indices]
                    edge_index, _ = subgraph(
                        final_subset, 
                        data.edge_index, 
                        relabel_nodes=True, 
                        num_nodes=data.num_nodes
                    )
                    subset = final_subset
                
                # 4. Update the data object
                data.edge_index = edge_index
                data.num_nodes = len(subset)
                if hasattr(data, 'x') and data.x is not None:
                    data.x = data.x[subset]
                    
            processed_graphs.append(data)

        return processed_graphs, "MalNetTiny"

    if dataset_key in {"elliptic", "elliptic_bitcoin", "ellipticbitcoindataset"}:
        dataset = EllipticBitcoinDataset(root=data_args.data_root)
        if len(dataset) == 0:
            raise ValueError("PyG dataset 'EllipticBitcoinDataset' is empty.")

        data = dataset[0]
        node_centers = torch.arange(int(data.num_nodes))

        y = getattr(data, "y", None)
        if y is not None and y.numel() == int(data.num_nodes):
            known_mask = y.reshape(-1) != 2
            node_centers = node_centers[known_mask]

        train_mask = getattr(data, "train_mask", None)
        if train_mask is not None and train_mask.numel() == int(data.num_nodes):
            node_centers = node_centers[train_mask.reshape(-1)]

        test_mask = getattr(data, "test_mask", None)
        if not len(node_centers) and test_mask is not None and test_mask.numel() == int(data.num_nodes):
            node_centers = torch.arange(int(data.num_nodes))[test_mask.reshape(-1)]

        if not len(node_centers):
            node_centers = torch.arange(int(data.num_nodes))

        graphs = sample_k_hop_subgraphs(
            data,
            node_centers.tolist(),
            num_hops=data_args.elliptic_num_hops,  # see DataArguments change below
            max_samples=data_args.max_graphs,
            seed=seed,
            dataset_name="EllipticBitcoinDataset",
        )
        if not graphs:
            raise ValueError("No Elliptic Bitcoin subgraphs could be sampled.")
        return graphs, "EllipticBitcoinDataset"

    # Default: TUDataset
    dataset = TUDataset(root=data_args.data_root, name=data_args.pyg_dataset)
    if len(dataset) == 0:
        raise ValueError(f"PyG dataset '{data_args.pyg_dataset}' is empty.")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    if data_args.max_graphs > 0:
        indices = indices[: min(data_args.max_graphs, len(indices))]
    return [dataset[i] for i in indices], data_args.pyg_dataset
