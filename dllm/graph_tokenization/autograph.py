from __future__ import annotations

"""AutoGraph tokenization.

Run the graph tokenization tests with:
    pytest scripts/tests/test_graph_data.py
"""

import torch

from .autograph_ops import (
    _to_numpy_edge_index,
    get_graph_from_labeled_sent,
    get_graph_from_sent,
    sample_labeled_sent_from_graph,
    sample_sent_from_graph,
)
from .base import GraphTokenizer, TokenizerFactory
from .types import SimpleGraphData

@TokenizerFactory.register('autograph')
class AutoGraphTokenizer(GraphTokenizer):
    """Tokenize graphs into AutoGraph SENT token sequences and decode back."""

    method_name = "autograph"

    sos: int = 0
    reset: int = 1
    ladj: int = 2
    radj: int = 3
    eos: int = 4
    pad: int = 5
    special_toks = ["sos", "reset", "ladj", "radj", "eos", "pad"]

    def __init__(
        self,
        dataset_names=None,
        max_length=-1,
        truncation_length=None,
        labeled_graph=False,
        undirected=True,
        append_eos=True,
        rng=None,
    ):
        self.dataset_names = dataset_names or []
        self.max_length = max_length
        self.truncation_length = truncation_length
        self.labeled_graph = labeled_graph
        self.undirected = undirected
        self.append_eos = append_eos
        self.rng = rng

        self.dataset_to_idx = {
            dataset_name: i + len(self.special_toks)
            for i, dataset_name in enumerate(self.dataset_names)
        }
        self.idx_offset = len(self.special_toks) + len(self.dataset_names)

        self.max_num_nodes = None
        self.num_node_types = 0
        self.num_edge_types = 0
        self.node_idx_offset = None
        self.edge_idx_offset = None

    def set_num_nodes(self, max_num_nodes):
        if (self.max_num_nodes is None) or (self.max_num_nodes < max_num_nodes):
            self.max_num_nodes = max_num_nodes

    def set_num_node_and_edge_types(self, num_node_types=0, num_edge_types=0):
        if self.labeled_graph:
            if self.max_num_nodes is None:
                raise ValueError("Call set_num_nodes before setting node/edge types.")
            self.num_node_types = num_node_types
            self.num_edge_types = num_edge_types
            self.node_idx_offset = self.idx_offset + self.max_num_nodes
            self.edge_idx_offset = self.node_idx_offset + self.num_node_types

    def __len__(self):
        if self.max_num_nodes is None:
            raise ValueError("Call set_num_nodes before querying vocabulary length.")
        if self.labeled_graph:
            return (
                self.idx_offset + self.max_num_nodes + self.num_node_types + self.num_edge_types
            )
        return self.idx_offset + self.max_num_nodes

    def _coalesce_if_available(self, data):
        if hasattr(data, "is_coalesced") and callable(data.is_coalesced):
            if not data.is_coalesced():
                return data.coalesce()
        return data

    def tokenize(self, data):
        data = self._coalesce_if_available(data)
        if not hasattr(data, "edge_index"):
            raise ValueError("data must provide edge_index.")

        num_nodes = getattr(data, "num_nodes", None)
        if num_nodes is None:
            edge_index = _to_numpy_edge_index(data.edge_index)
            num_nodes = int(edge_index.max()) + 1 if edge_index.shape[1] > 0 else 0

        if self.labeled_graph:
            if self.node_idx_offset is None or self.edge_idx_offset is None:
                raise ValueError(
                    "For labeled graphs, call set_num_nodes and "
                    "set_num_node_and_edge_types before tokenizing."
                )
            if not hasattr(data, "x") or data.x is None:
                raise ValueError("Labeled graph tokenization requires data.x (node labels).")
            if not hasattr(data, "edge_attr") or data.edge_attr is None:
                raise ValueError(
                    "Labeled graph tokenization requires data.edge_attr (edge labels)."
                )
            walk_index, _ = sample_labeled_sent_from_graph(
                edge_index=data.edge_index,
                node_labels=data.x,
                edge_labels=data.edge_attr,
                node_idx_offset=self.node_idx_offset,
                edge_idx_offset=self.edge_idx_offset,
                num_nodes=int(num_nodes),
                max_length=self.max_length,
                idx_offset=self.idx_offset,
                reset=self.reset,
                ladj=self.ladj,
                radj=self.radj,
                rng=self.rng,
            )
        else:
            walk_index, _ = sample_sent_from_graph(
                edge_index=data.edge_index,
                num_nodes=int(num_nodes),
                max_length=self.max_length,
                idx_offset=self.idx_offset,
                reset=self.reset,
                ladj=self.ladj,
                radj=self.radj,
                rng=self.rng,
            )

        start_offset = 1  # sos
        end_offset = 1 if self.append_eos else 0

        dataset_name_idx = None
        if self.dataset_names and hasattr(data, "dataset_name"):
            dataset_name_idx = self.dataset_to_idx.get(data.dataset_name, None)
            if dataset_name_idx is not None:
                start_offset += 1

        walk_index_t = torch.from_numpy(walk_index)
        tokens = torch.zeros(
            (walk_index_t.shape[0] + start_offset + end_offset,), dtype=walk_index_t.dtype
        )
        tokens[0] = self.sos
        if dataset_name_idx is not None:
            tokens[1] = dataset_name_idx

        if self.append_eos:
            tokens[-1] = self.eos
            tokens[start_offset:-1] = walk_index_t
        else:
            tokens[start_offset:] = walk_index_t

        return tokens

    def _validate_strict_tokens(self, tokens: torch.Tensor) -> None:
        if tokens.numel() == 0:
            raise ValueError("Graph token sequence is empty.")

        raw_tokens = tokens.detach().cpu().tolist()

        # Padding is only valid as a trailing batch artifact.
        end = len(raw_tokens)
        while end > 0 and raw_tokens[end - 1] == self.pad:
            end -= 1
        if any(token_id == self.pad for token_id in raw_tokens[:end]):
            raise ValueError("PAD tokens are only allowed at the end of the sequence.")
        raw_tokens = raw_tokens[:end]
        if not raw_tokens:
            raise ValueError("Graph token sequence is empty after removing trailing PAD tokens.")

        # EOS is optional, but if present it must be the final non-PAD token.
        if raw_tokens[-1] == self.eos:
            raw_tokens = raw_tokens[:-1]
        elif any(token_id == self.eos for token_id in raw_tokens):
            raise ValueError("EOS is only allowed at the end of the sequence.")
        if not raw_tokens:
            raise ValueError("Graph token sequence is empty after removing EOS/PAD tokens.")

        if raw_tokens[0] != self.sos:
            raise ValueError("Graph token sequence must start with SOS.")

        body = raw_tokens[1:]
        if self.dataset_names:
            dataset_start = len(self.special_toks)
            dataset_end = dataset_start + len(self.dataset_names)
            if not body:
                raise ValueError("Dataset token, if present, must be followed by graph content.")
            if dataset_start <= body[0] < dataset_end:
                body = body[1:]
            elif any(dataset_start <= tok < dataset_end for tok in body[1:]):
                raise ValueError("Dataset token must appear immediately after SOS.")

        if not body:
            raise ValueError("Graph token sequence must contain at least one graph token.")

        if self.labeled_graph:
            if self.node_idx_offset is None or self.edge_idx_offset is None:
                raise ValueError(
                    "For labeled graphs, call set_num_nodes and set_num_node_and_edge_types before decoding."
                )
            self._validate_strict_labeled_body(body)
        else:
            self._validate_strict_unlabeled_body(body)

    def _validate_strict_unlabeled_body(self, body: list[int]) -> None:
        # Use a manual index so we can clearly reason about position without
        # the enumerate() loop re-visiting tokens that were already consumed
        # as part of a multi-token structure (e.g. adjacency pairs).
        i = 0
        n = len(body)

        def is_node(t: int) -> bool:
            return t >= self.idx_offset

        # The grammar requires at least one node before anything else.
        if n == 0 or not is_node(body[0]):
            raise ValueError("Graph body must begin with a node token.")

        while i < n:
            tok = body[i]

            # ── node token: starts or continues a trail ──────────────────────
            if is_node(tok):
                i += 1
                adjacency_seen = False

                # After a node we may see: LADJ block, RESET, another node
                # (trail continuation), or end-of-body.
                while i < n:
                    inner = body[i]

                    if inner == self.reset:
                        # Segment break — next token must be a node.
                        i += 1
                        if i < n and not is_node(body[i]):
                            raise ValueError(
                                f"RESET must be followed by a node token, got {body[i]} at position {i}."
                            )
                        break  # exit inner loop; outer loop handles the next node

                    elif inner == self.ladj:
                        # Adjacency block: < (node)* >
                        # Empty blocks < > are valid per Theorem 2.15 when
                        # N_G(v) ∩ visited = ∅ at a segment-start node.
                        if adjacency_seen:
                            raise ValueError(
                                "Each node may contain at most one adjacency block."
                            )
                        i += 1  # consume LADJ
                        while i < n and body[i] != self.radj:
                            if not is_node(body[i]):
                                raise ValueError(
                                    f"Only node tokens are allowed inside an adjacency block, "
                                    f"got {body[i]} at position {i}."
                                )
                            i += 1
                        if i >= n:
                            raise ValueError("Adjacency block was never closed with RADJ.")
                        i += 1  # consume RADJ
                        adjacency_seen = True

                    elif is_node(inner):
                        # Trail continuation to the next node — break out so
                        # the outer loop handles it as a fresh node.
                        break

                    else:
                        raise ValueError(
                            f"Unexpected token {inner} at position {i} after node token."
                        )

            else:
                raise ValueError(
                    f"Expected a node token at position {i}, got {tok}."
                )

        # The sequence must not end mid-structure.  The inner loop already
        # catches unclosed adjacency blocks; we only need to verify that the
        # very last consumed token was not a bare RESET (handled above by
        # checking that RESET is followed by a node — if it is the last token,
        # i will equal n after the increment and the outer loop exits cleanly,
        # but body[-1] == reset means the sequence ends on a segment break
        # with no following node).
        if body and body[-1] == self.reset:
            raise ValueError("Graph sequence cannot end with RESET.")

    def _validate_strict_labeled_body(self, body: list[int]) -> None:
        # Manual index traversal (not enumerate) so that multi-token structures
        # — (edge_label, node, node_label) trail steps and (edge_label, node)
        # adjacency pairs — can be consumed atomically without the loop
        # re-entering on already-processed tokens.
        i = 0
        n = len(body)

        def is_node(t: int) -> bool:
            return self.idx_offset <= t < self.node_idx_offset

        def is_node_label(t: int) -> bool:
            return self.node_idx_offset <= t < self.edge_idx_offset

        def is_edge_label(t: int) -> bool:
            return self.edge_idx_offset <= t < self.edge_idx_offset + self.num_edge_types

        if n == 0 or not is_node(body[0]):
            raise ValueError("Labeled graph body must begin with a node token.")

        while i < n:
            # ── node_id ──────────────────────────────────────────────────────
            if not is_node(body[i]):
                raise ValueError(
                    f"Expected node token at position {i}, got {body[i]}."
                )
            i += 1

            # ── node_label ───────────────────────────────────────────────────
            if i >= n or not is_node_label(body[i]):
                raise ValueError(
                    f"Expected node-label token after node at position {i - 1}, "
                    f"got {body[i] if i < n else 'end-of-sequence'}."
                )
            i += 1

            adjacency_seen = False

            # After (node, node_label) we may see:
            #   • RESET           — segment break
            #   • LADJ … RADJ     — neighborhood set (possibly empty)
            #   • edge_label      — trail continuation
            #   • end-of-body     — last node in sequence
            while i < n:
                inner = body[i]

                if inner == self.reset:
                    # Segment break — next token must be a node.
                    i += 1
                    if i < n and not is_node(body[i]):
                        raise ValueError(
                            f"RESET must be followed by a node token, got {body[i]} at position {i}."
                        )
                    break  # outer loop handles the next node

                elif inner == self.ladj:
                    # Adjacency block: < (edge_label node)* >
                    # An empty block < > is valid (N_G(v) ∩ visited = ∅).
                    if adjacency_seen:
                        raise ValueError(
                            "Each labeled node may contain at most one adjacency block."
                        )
                    i += 1  # consume LADJ
                    while i < n and body[i] != self.radj:
                        # Consume (edge_label, node) pair.
                        if not is_edge_label(body[i]):
                            raise ValueError(
                                f"Expected edge-label token inside adjacency block at position {i}, "
                                f"got {body[i]}."
                            )
                        i += 1  # consume edge_label
                        if i >= n or not is_node(body[i]):
                            raise ValueError(
                                f"Edge-label in adjacency block must be followed by a node token "
                                f"at position {i}."
                            )
                        i += 1  # consume node
                    if i >= n:
                        raise ValueError("Adjacency block was never closed with RADJ.")
                    i += 1  # consume RADJ
                    adjacency_seen = True

                elif is_edge_label(inner):
                    # Trail continuation: edge_label → node_id → node_label
                    # Consume all three tokens atomically, then break so the
                    # outer loop re-enters at the new node_id position.
                    i += 1  # consume edge_label
                    if i >= n or not is_node(body[i]):
                        raise ValueError(
                            f"Edge-label for trail continuation at position {i - 1} must be "
                            f"followed by a node token."
                        )
                    i += 1  # consume node_id
                    if i >= n or not is_node_label(body[i]):
                        raise ValueError(
                            f"Trail-continuation node at position {i - 1} must be followed "
                            f"by a node-label token."
                        )
                    i += 1  # consume node_label
                    # The outer loop now expects another node_id or end-of-body,
                    # but we have already consumed (node_id, node_label) here.
                    # Reset adjacency_seen and stay in the inner loop so the
                    # next inner token is processed in the correct state.
                    adjacency_seen = False

                elif is_node(inner):
                    # Next node starts directly — break to outer loop.
                    break

                else:
                    raise ValueError(
                        f"Unexpected token {inner} at position {i} after labeled node."
                    )

        if body and body[-1] == self.reset:
            raise ValueError("Labeled graph sequence cannot end with RESET.")

    def decode(self, tokens: torch.Tensor, strict: bool = False):
        if strict:
            self._validate_strict_tokens(tokens)

        tokens = tokens[(tokens != self.pad) & (tokens != self.sos) & (tokens != self.eos)]
        if tokens.numel() == 0:
            return SimpleGraphData(edge_index=torch.zeros((2, 0), dtype=torch.long), num_nodes=0)

        dataset_name = None
        if self.dataset_names:
            maybe_dataset_idx = int(tokens[0].item()) - len(self.special_toks)
            if 0 <= maybe_dataset_idx < len(self.dataset_names):
                dataset_name = self.dataset_names[maybe_dataset_idx]
                tokens = tokens[1:]
                if tokens.numel() == 0:
                    return SimpleGraphData(
                        edge_index=torch.zeros((2, 0), dtype=torch.long),
                        num_nodes=0,
                        dataset_name=dataset_name,
                    )

        if self.labeled_graph:
            edge_index, node_labels, edge_labels = get_graph_from_labeled_sent(
                walk_index=tokens,
                idx_offset=self.idx_offset,
                node_idx_offset=self.node_idx_offset,
                edge_idx_offset=self.edge_idx_offset,
                num_node_types=self.num_node_types,
                num_edge_types=self.num_edge_types,
                reset=self.reset,
                ladj=self.ladj,
                radj=self.radj,
                undirected=self.undirected,
            )
            num_nodes = (
                int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else int(node_labels.numel())
            )
            return SimpleGraphData(
                x=node_labels,
                edge_index=edge_index,
                edge_attr=edge_labels,
                num_nodes=num_nodes,
                dataset_name=dataset_name,
            )

        edge_index = get_graph_from_sent(
            walk_index=tokens,
            idx_offset=self.idx_offset,
            reset=self.reset,
            ladj=self.ladj,
            radj=self.radj,
            undirected=self.undirected,
        )
        num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else 0
        return SimpleGraphData(edge_index=edge_index, dataset_name=dataset_name, num_nodes=num_nodes)