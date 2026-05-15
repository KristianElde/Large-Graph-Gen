from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

def _load_elliptic(root: str):
    """Load EllipticBitcoinDataset; raise a clear error if PyG is missing."""
    try:
        from torch_geometric.datasets import EllipticBitcoinDataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "torch-geometric is required.  Install it with:\n"
            "  pip install torch-geometric"
        ) from exc

    dataset = EllipticBitcoinDataset(root=root)
    return dataset


def _extract_time_step_subgraph(
    data,
    time_step: int,
) -> Data | None:  # noqa: F821  (lazy import)
    """
    Return the induced subgraph of ``data`` for a single ``time_step``.

    Returns ``None`` if the time-step has no nodes.
    """

    # node mask for this time-step (feature 0 is 1-indexed time)
    node_mask = (data.x[:, 0] == time_step)
    node_indices = node_mask.nonzero(as_tuple=True)[0]

    if node_indices.numel() == 0:
        return None

    # Induce subgraph (re-index edges)
    edge_index_sub, _ = subgraph(
        node_indices,
        data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
    )

    # Node labels (y): 0=licit, 1=illicit, 2=unknown
    y_sub = data.y[node_indices] if data.y is not None else None

    # Node features (drop the time-step column so downstream sees actual feats)
    x_sub = data.x[node_indices, 1:]   # drop column 0

    sub = Data(
        x=x_sub,
        edge_index=edge_index_sub,
        y=y_sub,
        num_nodes=int(node_indices.numel()),
    )
    sub.time_step = time_step
    sub.dataset_name = "EllipticBitcoin"
    return sub


@dataclass
class GraphTextBatch:
    input_ids: torch.Tensor       # (1, seq_len)
    prompt_lengths: torch.Tensor  # (1,)
    time_step: int
    labels: torch.Tensor | None


class EllipticTimeStepDataset(Dataset):
    """
    Each item is one temporal subgraph

    Parameters
    ----------
    data            : the single PyG Data object from EllipticBitcoinDataset[0]
    time_steps      : list of integer time-steps to include in this split
    graph_tokenizer : AutoGraphTokenizer (or compatible GraphTokenizer)
    llada_tokenizer : HuggingFace tokenizer from LLaDAModel
    prompt_prefix   : text prepended before the graph-token sequence
    max_length      : maximum total token count (prompt + answer); longer
                      sequences are truncated on the answer side
    skip_empty      : if True, time-steps with no edges are silently skipped;
                      otherwise they raise ValueError
    """

    def __init__(
        self,
        data,
        time_steps: Sequence[int],
        graph_tokenizer,
        llada_tokenizer,
        prompt_prefix: str = "Generate graph text:\n",
        max_length: int = 512,
        skip_empty: bool = True,
    ) -> None:
        self.data = data
        self.graph_tokenizer = graph_tokenizer
        self.llada_tokenizer = llada_tokenizer
        self.prompt_prefix = prompt_prefix
        self.max_length = max_length
        self.skip_empty = skip_empty

        self._samples: list[GraphTextBatch] = []
        self._build(time_steps)

    def _build(self, time_steps: Sequence[int]) -> None:
        for ts in time_steps:
            sub = _extract_time_step_subgraph(self.data, ts)
            if sub is None:
                if not self.skip_empty:
                    raise ValueError(
                        f"Time-step {ts} has no nodes in the dataset."
                    )
                warnings.warn(
                    f"[EllipticDataset] Time-step {ts} has no nodes; "
                    f"skipping.",
                    stacklevel=2,
                )
                continue

            if sub.edge_index.numel() == 0:
                if not self.skip_empty:
                    raise ValueError(
                        f"Time-step {ts} subgraph has no edges."
                    )
                warnings.warn(
                    f"[EllipticDataset] Time-step {ts} has no edges; "
                    f"skipping.",
                    stacklevel=2,
                )
                continue

            # Tokenise with the graph tokenizer
            try:
                graph_token_ids = self.graph_tokenizer.tokenize(sub)
            except Exception as exc:
                warnings.warn(
                    f"[EllipticDataset] Failed to tokenize time-step {ts}: "
                    f"{exc}; skipping.",
                    stacklevel=2,
                )
                continue

            # Convert graph token tensor to text string
            graph_text = self._graph_tokens_to_text(graph_token_ids)

            # Encode with the LLaDA HF tokenizer
            try:
                batch = self._build_training_sequence(
                    prompt=self.prompt_prefix,
                    answer=graph_text,
                    time_step=ts,
                    labels=sub.y,
                )
            except ValueError as exc:
                warnings.warn(
                    f"[EllipticDataset] Skipping time-step {ts}: {exc}",
                    stacklevel=2,
                )
                continue

            self._samples.append(batch)

        if len(self._samples) == 0:
            raise RuntimeError(
                "No valid samples were built from the provided time-steps.  "
                "Check that the dataset loaded correctly and that max_length "
                "is large enough to hold at least one graph token."
            )

    def _graph_tokens_to_text(self, tokens: torch.Tensor) -> str:
        """
        Map graph token ids to a human-readable string for the LLaDA vocab.

        Only <reset>, <ladj>, <radj> are used
        """
        gt = self.graph_tokenizer
        skip_ids = {gt.sos, gt.eos, gt.pad}
        render_as = {
            gt.reset: "<reset>",
            gt.ladj:  "<ladj>",
            gt.radj:  "<radj>",
        }
        parts: list[str] = []
        for tok in tokens.tolist():
            if tok in skip_ids:
                continue
            elif tok in render_as:
                parts.append(render_as[tok])
            else:
                node_id = tok - gt.idx_offset
                parts.append(f"n{node_id}")
        return " ".join(parts)

    def _build_training_sequence(
        self,
        prompt: str,
        answer: str,
        time_step: int,
        labels: torch.Tensor | None,
    ) -> GraphTextBatch:
        tok = self.llada_tokenizer

        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        answer_ids = tok.encode(answer, add_special_tokens=False)

        bos_id = getattr(tok, "bos_token_id", None) ##TODO: Check the tok id for bos, pad and eos properly
        eos_id = getattr(tok, "eos_token_id", None)

        sequence: list[int] = []
        if bos_id is not None:
            sequence.append(int(bos_id))
        sequence.extend(prompt_ids)
        prompt_length = len(sequence)
        sequence.extend(answer_ids)
        if eos_id is not None:
            sequence.append(int(eos_id))

        if prompt_length >= self.max_length:
            raise ValueError(
                f"Prompt alone is {prompt_length} tokens, which fills the "
                f"max_length={self.max_length} budget.  Increase --max-length "
                f"or shorten the prompt."
            )

        sequence = sequence[: self.max_length]

        if len(sequence) <= prompt_length:
            raise ValueError(
                "Truncation removed all answer tokens.  Increase max_length."
            )

        input_ids = torch.tensor(sequence, dtype=torch.long).unsqueeze(0)
        prompt_lengths = torch.tensor([prompt_length], dtype=torch.long)

        return GraphTextBatch(
            input_ids=input_ids,
            prompt_lengths=prompt_lengths,
            time_step=time_step,
            labels=labels,
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> GraphTextBatch:
        return self._samples[idx]


def _collate_graph_text(
    batch: list[GraphTextBatch],
) -> dict[str, torch.Tensor | list]:
    """
    Collate a list of GraphTextBatch objects into a dict of tensors.

    Sequences are padded on the right to the longest sequence in the batch
    using the convention from the original finetune script (no explicit
    attention-mask needed because LLaDA's bidirectional model sees all tokens).
    """
    max_len = max(b.input_ids.shape[1] for b in batch)

    padded_ids: list[torch.Tensor] = []
    prompt_lengths: list[int] = []
    time_steps: list[int] = []
    labels: list[torch.Tensor | None] = []

    # Use 0 as padding id; LLaDA's loss ignores padding via prompt_lengths /
    # the masking mechanism in compute_sft_loss.
    for b in batch:
        seq_len = b.input_ids.shape[1]
        pad_len = max_len - seq_len
        padded = torch.cat(
            [b.input_ids, torch.zeros(1, pad_len, dtype=torch.long)], dim=1
        )
        padded_ids.append(padded)
        prompt_lengths.append(int(b.prompt_lengths.item()))
        time_steps.append(b.time_step)
        labels.append(b.labels)

    return {
        "input_ids": torch.cat(padded_ids, dim=0),           # (B, max_len)
        "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),  # (B,)
        "time_steps": time_steps,
        "labels": labels,
    }


def build_elliptic_dataloaders(
    root: str,
    graph_tokenizer,
    llada_tokenizer,
    prompt_prefix: str = "Generate graph text:\n",
    max_length: int = 512,
    batch_size: int = 1,
    num_workers: int = 0,
    train_time_steps: Sequence[int] | None = None,
    test_time_steps: Sequence[int] | None = None,
    skip_empty: bool = True,
    shuffle_train: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Parameters
    ----------
    root                : directory where PyG stores / caches the dataset
    graph_tokenizer     : AutoGraphTokenizer (must have set_num_nodes called)
    llada_tokenizer     : HF tokenizer from LLaDAModel (after integrating graph
                          special tokens)
    prompt_prefix       : text prepended to every graph-text answer
    max_length          : total sequence budget (prompt + answer tokens)
    batch_size          : number of time-step subgraphs per batch
    num_workers         : DataLoader workers (0 = main process)
    train_time_steps    : explicit list of training time-steps; defaults to 1–34
    test_time_steps     : explicit list of test time-steps; defaults to 35–49
    skip_empty          : skip time-steps with no nodes/edges instead of raising
    shuffle_train       : whether to shuffle the train loader
    """
    # Default temporal splits (paper convention)
    if train_time_steps is None:
        train_time_steps = list(range(1, 35))
    if test_time_steps is None:
        test_time_steps = list(range(35, 50))

    # Load the dataset (returns a list-like of length 1)
    dataset = _load_elliptic(root)
    data = dataset[0]

    print(
        f"[Elliptic] Loaded dataset: {data.num_nodes} nodes, "
        f"{data.edge_index.shape[1]} edges."
    )

    # Update graph tokenizer with the max node count across all time-steps in
    # both splits so the vocabulary is consistent.
    all_steps = list(train_time_steps) + list(test_time_steps)
    max_nodes_per_step = 0
    for ts in all_steps:
        mask = data.x[:, 0] == ts
        n = int(mask.sum().item())
        if n > max_nodes_per_step:
            max_nodes_per_step = n
    graph_tokenizer.set_num_nodes(max_nodes_per_step)
    print(
        f"[Elliptic] Graph tokenizer vocab set for max {max_nodes_per_step} "
        f"nodes per time-step."
    )

    # Build PyTorch Datasets
    train_ds = EllipticTimeStepDataset(
        data=data,
        time_steps=train_time_steps,
        graph_tokenizer=graph_tokenizer,
        llada_tokenizer=llada_tokenizer,
        prompt_prefix=prompt_prefix,
        max_length=max_length,
        skip_empty=skip_empty,
    )
    test_ds = EllipticTimeStepDataset(
        data=data,
        time_steps=test_time_steps,
        graph_tokenizer=graph_tokenizer,
        llada_tokenizer=llada_tokenizer,
        prompt_prefix=prompt_prefix,
        max_length=max_length,
        skip_empty=skip_empty,
    )

    print(
        f"[Elliptic] Train samples: {len(train_ds)}, "
        f"Test samples: {len(test_ds)}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=_collate_graph_text,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_graph_text,
    )

    return train_loader, test_loader

"""
Usage
-----
    from elliptic_dataloader import build_elliptic_dataloaders

    train_loader, test_loader = build_elliptic_dataloaders(
        root="./data/elliptic",
        graph_tokenizer=my_autograph_tokenizer,
        prompt_prefix="Generate graph text:\n",
        batch_size=1,           # each "batch" is one time-step subgraph
        llada_tokenizer=llada.tokenizer,
        max_length=512,
    )

    for batch in train_loader:
        input_ids      = batch["input_ids"]        # (1, seq_len)
        prompt_lengths = batch["prompt_lengths"]   # (1,)
        time_step      = batch["time_step"]        # int
        labels         = batch["labels"]           # node labels for this step
"""