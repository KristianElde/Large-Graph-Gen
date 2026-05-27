from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SimpleGraphData:
    """Minimal graph container used by this standalone module."""

    edge_index: torch.Tensor
    num_nodes: int
    x: Optional[torch.Tensor] = None
    edge_attr: Optional[torch.Tensor] = None
    dataset_name: Optional[str] = None
