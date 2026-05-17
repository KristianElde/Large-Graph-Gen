"""
Graph-to-LLM tokenization strategies.

Two alternatives to the current "shift-all-graph-ids" approach:

  Strategy A — TextMapping
    Converts every graph token (node ids, structural tokens) into a short
    text string, then lets the LLM tokenizer split and encode those strings
    normally.  No new tokens are added; no embedding resize needed.

  Strategy B — SelectiveSpecialTokens
    Adds ONLY the small structural control tokens (sos, reset, ladj, radj,
    eos, pad) as genuine special tokens in the LLM tokenizer.  Node ids are
    rendered as text strings and tokenized normally, just like Strategy A,
    so the structural tokens remain atomic while node content uses the
    model's existing vocabulary.

Usage
-----
See `build_lm_tokenizer_strategy` for the recommended entry point.
The returned object has:
    .encode(graph_token_seq)  -> list[int]   (LLM token ids)
    .decode(lm_token_ids)     -> list[int]   (graph token ids, for AutoGraphTokenizer.decode)
    .tokenizer               -> the (possibly modified) HF tokenizer
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _graph_token_to_text(tok_id: int, graph_tok, *, sep: str = " ") -> str:
    """
    Map a single graph-tokenizer integer id to a human-readable string.

    Special tokens use their name; node ids become 'node_N'.
    In labeled-graph mode, node-type and edge-type ids get their own prefix.
    """
    if tok_id == graph_tok.sos:
        return "<graph_sos>"
    if tok_id == graph_tok.eos:
        return "<graph_eos>"
    if tok_id == graph_tok.pad:
        return "<graph_pad>"
    if tok_id == graph_tok.reset:
        return "<graph_reset>"
    if tok_id == graph_tok.ladj:
        return "<graph_ladj>"
    if tok_id == graph_tok.radj:
        return "<graph_radj>"

    # Dataset-name tokens
    offset = len(graph_tok.special_toks)
    if graph_tok.dataset_names:
        if offset <= tok_id < offset + len(graph_tok.dataset_names):
            name = graph_tok.dataset_names[tok_id - offset]
            return f"<graph_dataset_{name}>"

    # Node id tokens
    idx_offset = graph_tok.idx_offset
    if graph_tok.labeled_graph and graph_tok.node_idx_offset is not None:
        node_idx_offset = graph_tok.node_idx_offset
        edge_idx_offset = graph_tok.edge_idx_offset
        if idx_offset <= tok_id < node_idx_offset:
            return f"node_{tok_id - idx_offset}"
        if node_idx_offset <= tok_id < edge_idx_offset:
            return f"ntype_{tok_id - node_idx_offset}"
        if tok_id >= edge_idx_offset:
            return f"etype_{tok_id - edge_idx_offset}"
    else:
        if tok_id >= idx_offset:
            return f"node_{tok_id - idx_offset}"

    return f"gtok_{tok_id}"   # fallback


_STRUCTURAL_NAMES = ["sos", "eos", "pad", "reset", "ladj", "radj"]
_STRUCTURAL_TEMPLATE = "<graph_{name}>"


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class GraphLMTokenizerStrategy(ABC):

    @abstractmethod
    def encode(self, graph_token_seq: list[int] | torch.Tensor) -> list[int]:
        """Convert a graph-tokenizer id sequence to LLM input_ids."""

    @abstractmethod
    def decode(self, lm_token_ids: list[int] | torch.Tensor) -> list[int]:
        """
        Convert LLM input_ids back to graph-tokenizer ids.

        Returns a flat list that AutoGraphTokenizer.decode() can consume after
        converting to a tensor.
        """

    @abstractmethod
    def save_metadata(self, path: str | Path) -> None:
        """Persist any information needed to reconstruct the mapping."""


# ---------------------------------------------------------------------------
# Strategy A — TextMapping
# ---------------------------------------------------------------------------

class TextMappingStrategy(GraphLMTokenizerStrategy):
    """
    Encodes every graph token as a short text string, then tokenizes that
    string with the LLM tokenizer.  No new tokens are ever added to the LLM
    tokenizer vocabulary.

    Pros
    ----
    - Zero embedding-matrix growth.
    - The LLM's pretrained representations for digit strings (e.g. "node_0",
      "node_1") carry some numerical/positional meaning, which may help.
    - Works with any frozen tokenizer.

    Cons
    ----
    - Each graph token may expand to multiple LLM sub-tokens, making sequences
      longer and alignment between graph structure and LLM ids less direct.
    - Structural tokens like "<graph_reset>" may be split into several pieces
      unless they happen to exist in the vocabulary, diluting their signal.
    - Decoding back to graph ids requires careful bookkeeping (we record the
      span each graph token occupies in LLM space).
    """

    strategy = "text_mapping"

    def __init__(self, tokenizer, graph_tok, add_bos: bool = False, add_eos: bool = False):
        self.tokenizer = tokenizer
        self.graph_tok = graph_tok
        self.add_bos = add_bos
        self.add_eos = add_eos
        # Build a cache: graph_tok_id -> list[lm_tok_id]
        self._cache: dict[int, list[int]] = {}
        # Reverse: lm_tok_id pattern -> graph_tok_id  (only works when spans
        # don't collide; stored as frozenset of span tuples for lookup)
        self._spans: list[tuple[tuple[int, ...], int]] = []

    # -- internal -----------------------------------------------------------

    def _text_for(self, gid: int) -> str:
        return _graph_token_to_text(gid, self.graph_tok)

    def _lm_ids_for(self, gid: int) -> list[int]:
        if gid not in self._cache:
            text = self._text_for(gid)
            ids = self.tokenizer.encode(
                text, add_special_tokens=False
            )
            self._cache[gid] = ids
        return self._cache[gid]

    # -- public -------------------------------------------------------------

    def encode(self, graph_token_seq: list[int] | torch.Tensor) -> list[int]:
        if isinstance(graph_token_seq, torch.Tensor):
            graph_token_seq = graph_token_seq.tolist()

        lm_ids: list[int] = []
        # spans: for each graph token, record which slice of lm_ids it occupies
        self._last_spans: list[tuple[int, int, int]] = []  # (gid, start, end)

        if self.add_bos and self.tokenizer.bos_token_id is not None:
            lm_ids.append(self.tokenizer.bos_token_id)

        for gid in graph_token_seq:
            sub_ids = self._lm_ids_for(gid)
            start = len(lm_ids)
            lm_ids.extend(sub_ids)
            self._last_spans.append((gid, start, len(lm_ids)))

        if self.add_eos and self.tokenizer.eos_token_id is not None:
            lm_ids.append(self.tokenizer.eos_token_id)

        return lm_ids

    def decode(self, lm_token_ids: list[int] | torch.Tensor) -> list[int]:
        """
        Re-construct graph token ids from LLM ids by greedy span matching.

        This works well when the same tokenizer is used for both encode and
        decode.  It will not be perfectly robust if the LLM tokenizer is
        updated or if sub-token boundaries shift.
        """
        if isinstance(lm_token_ids, torch.Tensor):
            lm_token_ids = lm_token_ids.tolist()

        # Build a reverse lookup: tuple(sub_ids) -> graph_id
        # We need to scan all possible graph ids that were seen
        reverse: dict[tuple[int, ...], int] = {}
        for gid, sub_ids in self._cache.items():
            key = tuple(sub_ids)
            if key not in reverse:
                reverse[key] = gid

        graph_ids: list[int] = []
        i = 0
        while i < len(lm_token_ids):
            # Try longest match first (up to 8 sub-tokens per graph token)
            matched = False
            for length in range(min(8, len(lm_token_ids) - i), 0, -1):
                span = tuple(lm_token_ids[i: i + length])
                if span in reverse:
                    graph_ids.append(reverse[span])
                    i += length
                    matched = True
                    break
            if not matched:
                i += 1  # skip unknown sub-token
        return graph_ids

    def save_metadata(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "strategy": self.strategy,
            "add_bos": self.add_bos,
            "add_eos": self.add_eos,
            "token_text_cache": {str(k): self._text_for(k) for k in range(len(self.graph_tok))},
        }
        (path / "graph_lm_strategy.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Strategy B — SelectiveSpecialTokens
# ---------------------------------------------------------------------------

class SelectiveSpecialTokensStrategy(GraphLMTokenizerStrategy):
    """
    Adds ONLY the structural control tokens from the graph tokenizer as new
    special tokens in the LLM tokenizer.  Node ids, node-type tokens, and
    edge-type tokens are still rendered as text strings and tokenized normally.

    Structural tokens added:
        <graph_sos>, <graph_eos>, <graph_pad>,
        <graph_reset>, <graph_ladj>, <graph_radj>

    The model's embedding matrix is resized by at most 6 positions (or fewer
    if some of these strings already existed in the vocabulary).

    Pros
    ----
    - Structural tokens are guaranteed to be atomic (single LLM token each),
      giving the model a clean, unambiguous signal for graph syntax.
    - Node ids still benefit from the LLM's existing sub-word representations.
    - Embedding resize is tiny (≤6 rows) vs. potentially thousands in the
      current approach.
    - Better interpretability: you can inspect the embeddings for <graph_reset>
      etc. as the model trains.

    Cons
    ----
    - Node id text strings still expand to multiple sub-tokens, so sequences
      can be long.
    - Newly added structural token embeddings start from random init; they
      need training signal to become meaningful.
    """

    strategy = "selective_special_tokens"

    STRUCTURAL_TOKENS = [
        "<graph_sos>",
        "<graph_eos>",
        "<graph_pad>",
        "<graph_reset>",
        "<graph_ladj>",
        "<graph_radj>",
    ]

    def __init__(self, tokenizer, graph_tok):
        self.graph_tok = graph_tok
        # Add structural special tokens to the tokenizer
        new_tokens = [t for t in self.STRUCTURAL_TOKENS if t not in tokenizer.get_vocab()]
        if new_tokens:
            tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        self.tokenizer = tokenizer

        # Build the graph_id -> lm_id mapping for structural tokens
        self._structural_map: dict[int, int] = {
            graph_tok.sos: tokenizer.convert_tokens_to_ids("<graph_sos>"),
            graph_tok.eos: tokenizer.convert_tokens_to_ids("<graph_eos>"),
            graph_tok.pad: tokenizer.convert_tokens_to_ids("<graph_pad>"),
            graph_tok.reset: tokenizer.convert_tokens_to_ids("<graph_reset>"),
            graph_tok.ladj: tokenizer.convert_tokens_to_ids("<graph_ladj>"),
            graph_tok.radj: tokenizer.convert_tokens_to_ids("<graph_radj>"),
        }
        # Reverse mapping for decode
        self._reverse_structural: dict[int, int] = {
            v: k for k, v in self._structural_map.items()
        }

        # Text cache for non-structural tokens (node ids, dataset names)
        self._text_cache: dict[int, list[int]] = {}

    # -- internal -----------------------------------------------------------

    def _is_structural(self, gid: int) -> bool:
        return gid in self._structural_map

    def _lm_ids_for_non_structural(self, gid: int) -> list[int]:
        if gid not in self._text_cache:
            text = _graph_token_to_text(gid, self.graph_tok)
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            self._text_cache[gid] = ids
        return self._text_cache[gid]

    # -- public -------------------------------------------------------------

    def encode(self, graph_token_seq: list[int] | torch.Tensor) -> list[int]:
        if isinstance(graph_token_seq, torch.Tensor):
            graph_token_seq = graph_token_seq.tolist()

        lm_ids: list[int] = []
        for gid in graph_token_seq:
            if self._is_structural(gid):
                lm_ids.append(self._structural_map[gid])
            else:
                lm_ids.extend(self._lm_ids_for_non_structural(gid))
        return lm_ids

    def decode(self, lm_token_ids: list[int] | torch.Tensor) -> list[int]:
        if isinstance(lm_token_ids, torch.Tensor):
            lm_token_ids = lm_token_ids.tolist()

        # Build reverse for non-structural tokens
        reverse_non_structural: dict[tuple[int, ...], int] = {}
        for gid, sub_ids in self._text_cache.items():
            key = tuple(sub_ids)
            if key not in reverse_non_structural:
                reverse_non_structural[key] = gid

        graph_ids: list[int] = []
        i = 0
        while i < len(lm_token_ids):
            tok = lm_token_ids[i]
            # Structural token — direct 1:1 reverse lookup
            if tok in self._reverse_structural:
                graph_ids.append(self._reverse_structural[tok])
                i += 1
                continue
            # Non-structural — try greedy span match
            matched = False
            for length in range(min(8, len(lm_token_ids) - i), 0, -1):
                span = tuple(lm_token_ids[i: i + length])
                if span in reverse_non_structural:
                    graph_ids.append(reverse_non_structural[span])
                    i += length
                    matched = True
                    break
            if not matched:
                i += 1  # skip
        return graph_ids

    def save_metadata(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "strategy": self.strategy,
            "structural_tokens": self.STRUCTURAL_TOKENS,
            "structural_graph_to_lm": {str(k): v for k, v in self._structural_map.items()},
        }
        (path / "graph_lm_strategy.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Factory / entry point
# ---------------------------------------------------------------------------

def build_lm_tokenizer_strategy(
    strategy: str,
    tokenizer,
    graph_tok,
    model=None,
    **kwargs,
) -> GraphLMTokenizerStrategy:
    
    vocab_before = len(tokenizer)

    if strategy == "text_mapping":
        obj = TextMappingStrategy(tokenizer, graph_tok, **kwargs)
    elif strategy == "selective_special_tokens":
        obj = SelectiveSpecialTokensStrategy(tokenizer, graph_tok, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. "
                         "Choose 'text_mapping' or 'selective_special_tokens'.")

    vocab_after = len(tokenizer)
    if model is not None and vocab_after != vocab_before:
        model.resize_token_embeddings(vocab_after)

    return obj


def tokenize_graphs_with_strategy(
    graphs,
    graph_tokenizer,
    lm_strategy: GraphLMTokenizerStrategy,
    *,
    labeled_graph: bool = False,
    dataset_name: str | None = None,
) -> list[dict[str, list[int]]]:
    """Compatibility wrapper that delegates to `dllm.utils.graph_data.tokenize_graphs_with_strategy`."""
    from dllm.utils.graph_data import tokenize_graphs_with_strategy as _impl

    return _impl(
        graphs,
        graph_tokenizer,
        lm_strategy,
        labeled_graph=labeled_graph,
        dataset_name=dataset_name,
    )