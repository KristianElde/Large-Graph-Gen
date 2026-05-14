import torch
import numpy as np
from .autograph import AutoGraphTokenizer
from .base import TokenizerFactory

@TokenizerFactory.register("nauty")
class NautyTokenizer(AutoGraphTokenizer):
    """Standardizes node ordering and features using Nauty before tokenizing."""
    
    def tokenize(self, data):
        import pynauty as n
        
        # 1. Build adjacency for nauty
        adj = {i: [] for i in range(data.num_nodes)}
        for src, dst in data.edge_index.t().tolist():
            adj[src].append(dst)
            
        g = n.Graph(data.num_nodes, adjacency_dict=adj)
        
        # 2. Get the canonical permutation
        # 'perm' is a list where perm[i] is the old index of the new i-th node
        perm = n.canon_label(g) 
        perm_tensor = torch.as_tensor(perm, dtype=torch.long)

        # 3. Relabel node features (x)
        if hasattr(data, "x") and data.x is not None:
            # We move the old features to their new canonical positions
            data.x = data.x[perm_tensor]

        # 4. Relabel edge_index
        # Create an inverse map: old_index -> new_canonical_index
        relabel_map = torch.zeros(data.num_nodes, dtype=torch.long)
        relabel_map[perm_tensor] = torch.arange(data.num_nodes)
        
        data.edge_index = relabel_map[data.edge_index]

        # 5. Handle edge attributes (edge_attr)
        # edge_index is now updated, but edge_attr is still tied to the original 
        # sequence of edges. Since edge_index and edge_attr are usually 
        # changed together during 'coalesce', we ensure consistency here.
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            # The relationship between edge_index[:, i] and edge_attr[i] 
            # is preserved even if we change the values inside edge_index.
            pass

        # 6. Proceed with standard AutoGraph logic
        return super().tokenize(data)

@TokenizerFactory.register("kandinsky")
class KandinskyTokenizer(NautyTokenizer):
    """Canonical Nauty ordering with features + Deterministic Walk."""
    def __init__(self, **kwargs):
        kwargs['rng'] = 42 
        super().__init__(**kwargs)