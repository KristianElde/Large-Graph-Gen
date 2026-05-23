from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch


# ---------------------------------------------------------------------------
# PyG graph → adjacency dict data structures
# ---------------------------------------------------------------------------

def _extract_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> dict[int, list[int]]:
    """Build {src: [dst, ...]} from a PyG edge_index tensor."""
    adj: dict[int, list[int]] = {i: [] for i in range(num_nodes)}
    if edge_index.numel() == 0:
        return adj
    src_nodes = edge_index[0].tolist()
    dst_nodes = edge_index[1].tolist()
    for s, d in zip(src_nodes, dst_nodes):
        adj[s].append(d)
    return adj


def _extract_node_types(x: torch.Tensor | None) -> dict[int, int]:
    """
    Coerce the node feature matrix to a flat per-node integer type label.
    Handles one-hot encoded features (argmax) and plain integer labels.
    """
    if x is None:
        return {}
    if x.ndim == 2:
        if x.shape[1] == 1:
            labels = x.squeeze(1).long()
        else:
            labels = x.argmax(dim=1)
    else:
        labels = x.long()
    return {i: int(v) for i, v in enumerate(labels.tolist())}


def _extract_edge_types(
    edge_attr: torch.Tensor | None,
    edge_index: torch.Tensor,
) -> dict[tuple[int, int], int]:
    """
    Coerce edge attributes to a per-edge integer type label.
    Returns {(src, dst): type_id}.
    """
    if edge_attr is None or edge_index.numel() == 0:
        return {}
    if edge_attr.ndim == 2:
        if edge_attr.shape[1] == 1:
            labels = edge_attr.squeeze(1).long()
        else:
            labels = edge_attr.argmax(dim=1)
    else:
        labels = edge_attr.long()
    src_nodes = edge_index[0].tolist()
    dst_nodes = edge_index[1].tolist()
    return {
        (int(s), int(d)): int(t)
        for s, d, t in zip(src_nodes, dst_nodes, labels.tolist())
    }


def _get_num_nodes(graph: Any) -> int:
    """Robustly extract node count from a PyG Data object."""
    num_nodes = getattr(graph, "num_nodes", None)
    if num_nodes is not None:
        return int(num_nodes)
    edge_index = getattr(graph, "edge_index", None)
    if edge_index is not None and torch.as_tensor(edge_index).numel() > 0:
        return int(torch.as_tensor(edge_index).max().item()) + 1
    x = getattr(graph, "x", None)
    if x is not None:
        return int(x.shape[0])
    return 0


# ---------------------------------------------------------------------------
# Per-graph prompt builder
# ---------------------------------------------------------------------------

def make_graph_prompt(num_nodes: int, *, labeled: bool = False) -> str:
    """
    Build a concise, graph-specific prompt that tells the model exactly how
    many nodes to expect and what the valid node-index range is.

    The prompt is designed to:
      1. State the graph format (adjacency dict) unambiguously.
      2. Give the exact node count so the model knows when the dict is complete.
      3. Specify the valid node-index range [0, num_nodes-1] to prevent
         out-of-range hallucinations.
      4. For labeled graphs, remind the model of the per-node schema.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph being generated.
    labeled : bool
        Whether node/edge type labels are included in the serialization.

    Returns
    -------
    str
        A prompt string that is prepended to the serialized graph tokens.
    """
    last_node = num_nodes - 1

    if labeled:
        return (
            f"Generate a labeled graph with exactly {num_nodes} node"
            f"{'s' if num_nodes != 1 else ''} "
            f"(indices 0 to {last_node}). "
            f"Format: adjacency dict where each key is a node index and its value "
            f'is {{"t": <node_type>, "e": [[<neighbor>, <edge_type>], ...]}}. '
            f"Include all nodes 0 to {last_node} as keys:"
        )
    else:
        return (
            f"Generate a graph with exactly {num_nodes} node"
            f"{'s' if num_nodes != 1 else ''} "
            f"(indices 0 to {last_node}). "
            f"Format: adjacency dict where each key is a node index and its value "
            f"is a list of neighboring node indices. "
            f"Include all nodes 0 to {last_node} as keys:"
        )


# ---------------------------------------------------------------------------
# PyG graph → plain-text adjacency dict string
# ---------------------------------------------------------------------------

def pyg_graph_to_dict_string(
    graph: Any,
    *,
    labeled: bool = False,
) -> str:
    """
    Convert a PyG Data object directly to a plain-text adjacency dict string.

    Parameters
    ----------
    graph :
        A PyG ``Data`` object (or any object with ``edge_index``,
        ``num_nodes``, and optionally ``x`` and ``edge_attr``).
    labeled : bool
        If True, include node-type and edge-type labels from ``graph.x``
        and ``graph.edge_attr``.

    Returns
    -------
    str
        Adjacency dict string, e.g.::

            Unlabeled: {0: [1, 2], 1: [0], 2: [0]}
            Labeled:   {0: {"t": 2, "e": [[1, 0]]}, 1: {"t": 0, "e": [[0, 0]]}}

    Examples
    --------
    >>> from torch_geometric.data import Data
    >>> import torch
    >>> g = Data(edge_index=torch.tensor([[0,1],[1,0]]), num_nodes=2)
    >>> pyg_graph_to_dict_string(g)
    '{0: [1], 1: [0]}'
    """
    edge_index = torch.as_tensor(graph.edge_index, dtype=torch.long)

    num_nodes = _get_num_nodes(graph)

    adjacency   = _extract_adjacency(edge_index, num_nodes)
    node_types  = _extract_node_types(getattr(graph, "x", None)) if labeled else {}
    edge_types  = _extract_edge_types(getattr(graph, "edge_attr", None), edge_index) if labeled else {}

    parts: list[str] = []
    for node_idx in range(num_nodes):
        nbrs = adjacency.get(node_idx, [])

        if labeled:
            nt = node_types.get(node_idx, 0)
            edge_entries = [
                f"[{dst}, {edge_types.get((node_idx, dst), 0)}]"
                for dst in nbrs
            ]
            edges_str = "[" + ", ".join(edge_entries) + "]"
            parts.append(f'{node_idx}: {{"t": {nt}, "e": {edges_str}}}')
        else:
            nbrs_str = "[" + ", ".join(str(d) for d in nbrs) + "]"
            parts.append(f"{node_idx}: {nbrs_str}")

    return "{" + ", ".join(parts) + "}"


# ---------------------------------------------------------------------------
# Plain-text adjacency dict string → adjacency data structures (for eval)
# ---------------------------------------------------------------------------

def dict_string_to_adjacency(
    text: str,
    *,
    labeled: bool = False,
) -> tuple[list[int], dict[int, list[int]], dict[int, int], dict[tuple[int, int], int]]:
    """
    Parse a dict-format graph string back into raw adjacency structures.

    Returns
    -------
    node_order  : list[int]         — node ids in the order they appear
    adjacency   : dict[int, list[int]]
    node_types  : dict[int, int]    — empty when labeled=False
    edge_types  : dict[(src,dst), int] — empty when labeled=False

    Used during generation / evaluation to reconstruct the graph from the
    text the diffusion model produced.
    """
    text = text.strip()
    # Strip outer braces
    if text.startswith("{"):
        text = text[1:]
    if text.endswith("}"):
        text = text[:-1]

    node_order:  list[int]                    = []
    adjacency:   dict[int, list[int]]         = {}
    node_types:  dict[int, int]               = {}
    edge_types:  dict[tuple[int, int], int]   = {}

    # Split on top-level commas (ignoring commas inside [...] or {...})
    entries: list[str] = []
    depth   = 0
    current: list[str] = []
    for ch in text:
        if ch in ("{", "["):
            depth += 1
            current.append(ch)
        elif ch in ("}", "]"):
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            entries.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        entries.append("".join(current).strip())

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        colon = entry.find(":")
        if colon == -1:
            continue
        key_str = entry[:colon].strip().strip('"').strip("'")
        val_str = entry[colon + 1:].strip()
        try:
            node_idx = int(key_str)
        except ValueError:
            continue

        node_order.append(node_idx)
        adjacency[node_idx] = []

        if labeled and val_str.startswith("{"):
            m_t = re.search(r'"t"\s*:\s*(\d+)', val_str)
            if m_t:
                node_types[node_idx] = int(m_t.group(1))
            m_e = re.search(r'"e"\s*:\s*(\[.*\])', val_str, re.DOTALL)
            if m_e:
                for pair in re.finditer(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]', m_e.group(1)):
                    dst, et = int(pair.group(1)), int(pair.group(2))
                    adjacency[node_idx].append(dst)
                    edge_types[(node_idx, dst)] = et
        else:
            adjacency[node_idx] = [int(n) for n in re.findall(r'\d+', val_str)]

    return node_order, adjacency, node_types, edge_types


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

class GraphTextDictStrategy:
    """
    Converts PyG Data objects directly to LM token id sequences using a
    plain Python/JSON adjacency dict format. No graph tokenizer needed.

    Parameters
    ----------
    tokenizer :
        HuggingFace tokenizer. Not modified — zero new tokens added.
    labeled : bool
        Whether to include node/edge type labels from graph.x / graph.edge_attr.
    """

    strategy: str = "text_dict"

    def __init__(self, tokenizer, *, labeled: bool = False):
        self.tokenizer = tokenizer
        self.labeled   = labeled

    def encode_to_text(self, graph: Any) -> str:
        """PyG Data → adjacency dict string (for inspection / debugging)."""
        return pyg_graph_to_dict_string(graph, labeled=self.labeled)

    def encode(self, graph: Any) -> list[int]:
        """
        PyG Data → LM input_ids.

        The graph is serialised to a dict string, then tokenised with the
        LM tokenizer. No special tokens are prepended or appended.
        """
        text = pyg_graph_to_dict_string(graph, labeled=self.labeled)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def make_prompt(self, graph: Any) -> str:
        """
        Build a graph-specific prompt for a single PyG Data object.

        The prompt includes the exact node count and valid index range,
        giving the model strong structural priors before it sees any tokens.
        """
        num_nodes = _get_num_nodes(graph)
        return make_graph_prompt(num_nodes, labeled=self.labeled)

    def decode_to_adjacency(
        self,
        lm_token_ids: list[int] | torch.Tensor,
        strict: bool = False,
    ) -> tuple[list[int], dict[int, list[int]], dict[int, int], dict[tuple[int, int], int]]:
        """
        LM input_ids → (node_order, adjacency, node_types, edge_types).

        Useful during evaluation to reconstruct the graph from generated ids.
        """
        if isinstance(lm_token_ids, torch.Tensor):
            lm_token_ids = lm_token_ids.tolist()
        text = self.tokenizer.decode(lm_token_ids, skip_special_tokens=True)
        try:
            return dict_string_to_adjacency(text, labeled=self.labeled)
        except Exception as exc:
            if strict:
                raise ValueError(f"Failed to parse: {text!r}") from exc
            return [], {}, {}, {}

    def save_metadata(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "strategy": self.strategy,
            "labeled": self.labeled,
            "new_tokens_added": 0,
            "embedding_resize": False,
            "prompt_style": "per_graph_node_count",
            "format_unlabeled": "{0: [1, 2], 1: [0], 2: [0]}",
            "format_labeled": '{0: {"t": 2, "e": [[1, 0], [2, 1]]}, 1: {"t": 0, "e": []}}',
        }
        (path / "graph_lm_strategy.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_text_dict_strategy(
    tokenizer,
    *,
    labeled: bool = False,
    model=None,   # accepted for API symmetry; ignored (no resize needed)
) -> GraphTextDictStrategy:
    """
    Construct a GraphTextDictStrategy. No embedding resize is performed.

    Parameters
    ----------
    tokenizer :
        HuggingFace tokenizer.
    labeled : bool
        Set True to include node/edge type information.
    model :
        Ignored. Present only for API symmetry with other factory functions.
    """
    return GraphTextDictStrategy(tokenizer, labeled=labeled)


# ---------------------------------------------------------------------------
# Truncation helpers
# ---------------------------------------------------------------------------

def truncate_graph_token_ids(
    lm_ids: list[int],
    tokenizer,
    max_tokens: int,
) -> list[int]:
    """
    Truncate a tokenized graph dict sequence to fit within ``max_tokens``,
    while keeping the result a syntactically valid graph string.

    The serialized format is::

        {0: [1, 2], 1: [0, 3], 2: [0], 3: [1]}

    Truncation rules
    ----------------
    1. If ``len(lm_ids) <= max_tokens``, return unchanged.
    2. Reserve 1 token slot for the closing ``}`` token.
    3. Decode ``lm_ids[:max_tokens - 1]`` back to text.
    4. Find the last *complete* node entry — i.e. the last ``N: [...]``
       pair whose closing ``]`` is present in the decoded text.
       Everything after that closing ``]`` is a partial entry and is dropped,
       along with the trailing ``, `` separator if present.
    5. Append ``}`` and re-tokenize.  The result is guaranteed to be ≤
       ``max_tokens`` long (the re-tokenized closing brace is always 1
       token for every BPE tokenizer we use).

    Why "last complete entry" instead of a simple slice
    ---------------------------------------------------
    A raw token slice mid-sequence produces strings like::

        {0: [1, 2], 1: [0, 3], 2: [0

    The closing ``]`` and ``}`` are missing, so the string is not a valid
    Python/JSON dict and the decoder in ``dict_string_to_adjacency`` would
    either raise or silently drop data.  By scanning for the last complete
    ``N: [...]`` block we guarantee:
      - The output parses cleanly.
      - No node appears with a partial neighbour list (which would imply
        edges that the original graph does not have).
      - The adjacency is still a valid (smaller) graph — the removed nodes
        simply don't appear as *sources*; any back-references to them from
        surviving entries are still valid node ids in an undirected setting.

    Parameters
    ----------
    lm_ids   : list[int]   Token ids from ``tokenizer.encode(graph_text)``.
    tokenizer:             HuggingFace tokenizer used for the encoding.
    max_tokens : int       Hard budget (inclusive).  Must be ≥ 2.

    Returns
    -------
    list[int]  — token ids of a valid, closed graph dict string, length ≤ max_tokens.
    """
    if len(lm_ids) <= max_tokens:
        return lm_ids

    if max_tokens < 2:
        raise ValueError(f"max_tokens must be >= 2, got {max_tokens}")

    # Step 1 — decode the prefix that fits (leave 1 slot for "}")
    prefix_text = tokenizer.decode(
        lm_ids[: max_tokens - 1],
        skip_special_tokens=True,
    )

    # Step 2 — find the last complete "N: [...]" entry.
    is_labeled = '"t":' in prefix_text

    if is_labeled:
        last_complete_end = _find_last_complete_labeled_entry(prefix_text)
    else:
        last_complete_end = _find_last_complete_unlabeled_entry(prefix_text)

    if last_complete_end == -1:
        # No complete entry fits at all — return a minimal valid empty graph.
        empty_ids = tokenizer.encode("{}", add_special_tokens=False)
        return empty_ids[:max_tokens]

    # Step 3 — slice to the last complete entry, strip trailing ", " if present,
    # then close the dict.
    clean = prefix_text[:last_complete_end].rstrip(", ")
    closed = clean + "}"

    # Step 4 — re-tokenize and verify the budget.
    result_ids = tokenizer.encode(closed, add_special_tokens=False)

    if len(result_ids) > max_tokens:
        result_ids = result_ids[:max_tokens]

    return result_ids


# ---------------------------------------------------------------------------
# Internal helpers for truncate_graph_token_ids
# ---------------------------------------------------------------------------

def _find_last_complete_unlabeled_entry(text: str) -> int:
    """
    Return the index *after* the closing ']' of the last complete top-level
    list value in an unlabeled graph dict string.

    Example::

        "{0: [1, 2], 1: [0"   → returns 10  (after the ']' of "0: [1, 2]")

    Returns -1 if no complete entry is found.
    """
    depth   = 0
    last_close = -1
    for i, ch in enumerate(text):
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                last_close = i + 1   # position *after* the ']'
    return last_close


def _find_last_complete_labeled_entry(text: str) -> int:
    """
    Return the index *after* the closing '}' of the last complete top-level
    node-value dict in a labeled graph dict string.

    Labeled entries look like::

        N: {"t": T, "e": [[d, et], ...]}

    The outer dict is depth 1 (after the opening '{').  A node's value dict
    is at depth 2.  We look for '}'  that brings us from depth 2 back to 1.

    Returns -1 if no complete entry is found.
    """
    depth      = 0
    last_close = -1
    in_string  = False
    prev_ch    = ''

    for i, ch in enumerate(text):
        if ch == '"' and prev_ch != '\\':
            in_string = not in_string
        if not in_string:
            if ch in ('{', '['):
                depth += 1
            elif ch in ('}', ']'):
                depth -= 1
                if ch == '}' and depth == 1:
                    last_close = i + 1
        prev_ch = ch

    return last_close


# ---------------------------------------------------------------------------
# Dataset builder — replaces tokenize_graphs / tokenize_graphs_with_strategy
# ---------------------------------------------------------------------------

def prepend_prompt_to_row(
    row: dict,
    graph: Any,
    strategy: GraphTextDictStrategy,
    max_tokens: int | None = None,
) -> dict | None:
    """
    Prepend a **per-graph** prompt to a single tokenized row.

    The prompt is generated from the actual graph object so it can include
    graph-specific information (node count, valid index range, etc.).

    Labels on the prompt span are set to -100 so the MDLM/BD3LM trainer
    never computes loss or applies masking on the conditioning prefix.

    Parameters
    ----------
    row : dict
        Must contain ``"input_ids"`` (list[int]) — the graph token ids.
    graph : Any
        The original PyG Data object used to build the per-graph prompt.
    strategy : GraphTextDictStrategy
        Strategy instance; used to call ``make_prompt``.
    max_tokens : int | None
        If set, the combined sequence is capped at this length by first
        truncating the graph portion (never the prompt).

    Returns
    -------
    dict | None
        Updated row with ``input_ids``, ``labels``, ``attention_mask``, and
        ``prompt_length`` (for downstream reference / debugging).
        Returns None if the prompt alone already exceeds max_tokens.
    """
    prompt_text = strategy.make_prompt(graph)
    prompt_ids  = strategy.tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_len  = len(prompt_ids)

    graph_ids = row["input_ids"]

    if max_tokens is not None:
        budget = max_tokens - prompt_len
        if budget <= 0:
            # Prompt alone exceeds budget — skip this sample
            return None
        if len(graph_ids) > budget:
            graph_ids = truncate_graph_token_ids(
                graph_ids, strategy.tokenizer, budget
            )

    combined_ids = prompt_ids + graph_ids
    combined_len = len(combined_ids)

    return {
        "input_ids":      combined_ids,
        # -100 on prompt tokens → excluded from loss / masking in trainer
        "labels":         [-100] * prompt_len + list(graph_ids),
        "attention_mask": [1] * combined_len,
        # Stash prompt length so callers can split prompt vs graph at eval time
        "prompt_length":  prompt_len,
    }


def build_dataset_from_pyg_graphs(
    graphs: list[Any],
    strategy: GraphTextDictStrategy,
    max_tokens: int | None = None,
    prompt: str | None = None,   # kept for back-compat; ignored (per-graph prompt used)
) -> list[dict[str, list[int]]]:
    """
    Convert a list of PyG Data objects into HuggingFace-ready rows.

    Each row receives a **per-graph** prompt (built by ``strategy.make_prompt``)
    that states the exact node count and valid index range for that specific
    graph.  The prompt tokens receive ``labels = -100`` so they are masked out
    of training loss / diffusion masking.

    Parameters
    ----------
    graphs : list
        PyG Data objects to serialise.
    strategy : GraphTextDictStrategy
        Encodes graphs to token ids and builds per-graph prompts.
    max_tokens : int | None
        Hard token budget per sequence (prompt + graph).  Graph tokens are
        truncated to fit; samples where the prompt alone exceeds the budget
        are dropped with a warning.
    prompt : str | None
        Deprecated.  Ignored — a per-graph prompt is always used instead.
        Kept in the signature for backwards compatibility.

    Returns
    -------
    list[dict]  — each dict has ``input_ids``, ``labels``, ``attention_mask``,
                  and ``prompt_length``.
    """
    if prompt is not None:
        import warnings
        warnings.warn(
            "The `prompt` argument to build_dataset_from_pyg_graphs is deprecated "
            "and will be removed in a future version. Per-graph prompts are now "
            "always used; the passed string is ignored.",
            DeprecationWarning,
            stacklevel=2,
        )

    rows: list[dict[str, list[int]]] = []
    skipped = 0

    for graph in graphs:
        # 1. Encode the graph body to token ids
        lm_ids = strategy.encode(graph)
        if not lm_ids:
            skipped += 1
            continue

        base_row = {"input_ids": lm_ids, "labels": lm_ids.copy()}

        # 2. Prepend per-graph prompt (handles truncation internally)
        processed = prepend_prompt_to_row(
            base_row,
            graph,
            strategy,
            max_tokens=max_tokens,
        )

        if processed is None:
            # Prompt alone exceeds max_tokens budget — skip
            skipped += 1
            continue

        rows.append(processed)

    if skipped:
        import logging
        logging.getLogger(__name__).warning(
            f"build_dataset_from_pyg_graphs: skipped {skipped} graph(s) "
            f"(empty encoding or prompt exceeded token budget)."
        )

    return rows