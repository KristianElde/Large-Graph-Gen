from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

SplitName = Literal["train", "val", "trainval", "test"]


def _load_malnet_tiny(
    root: str,
    split: SplitName | None = None,
    force_reload: bool = False,
):
    """Load PyG's MalNetTiny dataset with a clear dependency error."""
    try:
        from torch_geometric.datasets import MalNetTiny  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "torch-geometric is required. Install it with:\n"
            "  pip install torch-geometric"
        ) from exc

    return MalNetTiny(root=root, split=split, force_reload=force_reload)


@dataclass
class GraphTextBatch:
    input_ids: torch.Tensor
    prompt_lengths: torch.Tensor
    graph_index: int
    labels: torch.Tensor | None
    num_nodes: int
    num_edges: int


class MalNetTinyGraphTextDataset(Dataset):
    """
    Each item is one MalNetTiny graph tokenized as graph text.

    Parameters
    ----------
    dataset         : PyG MalNetTiny dataset split.
    graph_tokenizer : AutoGraphTokenizer or compatible GraphTokenizer.
    llada_tokenizer : HuggingFace tokenizer from LLaDAModel.
    prompt_prefix   : text prepended before the graph-token sequence.
    max_length      : maximum total token count.
    max_graphs      : optional cap for quick experiments and notebooks.
    skip_invalid    : skip graphs that cannot be tokenized/encoded.
    """

    dataset_name = "MalNetTiny"

    def __init__(
        self,
        dataset,
        graph_tokenizer,
        llada_tokenizer,
        prompt_prefix: str = "Generate graph text:\n",
        max_length: int = 512,
        max_graphs: int | None = None,
        skip_invalid: bool = True,
    ) -> None:
        self.dataset = dataset
        self.graph_tokenizer = graph_tokenizer
        self.llada_tokenizer = llada_tokenizer
        self.prompt_prefix = prompt_prefix
        self.max_length = max_length
        self.max_graphs = min(len(dataset), max_graphs) if max_graphs else len(dataset)
        self.skip_invalid = skip_invalid

        self._samples: list[GraphTextBatch] = []
        self._build()

    def _build(self) -> None:
        for idx in range(self.max_graphs):
            graph = self.dataset[idx]
            graph = graph.clone() if hasattr(graph, "clone") else graph
            graph.dataset_name = self.dataset_name

            if int(graph.num_nodes) == 0 or graph.edge_index.numel() == 0:
                self._handle_invalid(idx, "empty graph")
                continue

            try:
                labels = getattr(graph, "y", None)
                if labels is not None and not torch.is_tensor(labels):
                    labels = torch.tensor(labels, dtype=torch.long)

                graph_token_ids = self.graph_tokenizer.tokenize(graph)
                graph_text = self._graph_tokens_to_text(graph_token_ids)
                batch = self._build_training_sequence(
                    prompt=self.prompt_prefix,
                    answer=graph_text,
                    graph_index=idx,
                    labels=labels,
                    num_nodes=int(graph.num_nodes),
                    num_edges=int(graph.edge_index.shape[1]),
                )
            except Exception as exc:
                self._handle_invalid(idx, str(exc))
                continue

            self._samples.append(batch)

        if len(self._samples) == 0:
            raise RuntimeError(
                "No valid MalNetTiny samples were built. Check that the dataset "
                "loaded correctly and that max_length is large enough to keep "
                "at least one graph token."
            )

    def _handle_invalid(self, idx: int, reason: str) -> None:
        if not self.skip_invalid:
            raise ValueError(f"Graph {idx} is invalid: {reason}")
        warnings.warn(
            f"[MalNetTiny] Skipping graph {idx}: {reason}",
            stacklevel=2,
        )

    def _graph_tokens_to_text(self, tokens: torch.Tensor) -> str:
        gt = self.graph_tokenizer
        skip_ids = {gt.sos, gt.eos, gt.pad}
        render_as = {
            gt.reset: "<reset>",
            gt.ladj: "<ladj>",
            gt.radj: "<radj>",
        }

        parts: list[str] = []
        for tok in tokens.tolist():
            tok = int(tok)
            if tok in skip_ids:
                continue
            if tok in render_as:
                parts.append(render_as[tok])
            elif (
                getattr(gt, "dataset_names", None)
                and len(gt.special_toks) <= tok < gt.idx_offset
            ):
                dataset_name = gt.dataset_names[tok - len(gt.special_toks)]
                parts.append(f"<{dataset_name}>")
            else:
                node_id = tok - gt.idx_offset
                parts.append(f"n{node_id}")
        return " ".join(parts)

    def _build_training_sequence(
        self,
        prompt: str,
        answer: str,
        graph_index: int,
        labels: torch.Tensor | None,
        num_nodes: int,
        num_edges: int,
    ) -> GraphTextBatch:
        tok = self.llada_tokenizer
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        answer_ids = tok.encode(answer, add_special_tokens=False)

        bos_id = getattr(tok, "bos_token_id", None)
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
                f"Prompt alone is {prompt_length} tokens, which fills "
                f"max_length={self.max_length}."
            )

        sequence = sequence[: self.max_length]
        if len(sequence) <= prompt_length:
            raise ValueError("truncation removed all answer tokens")

        return GraphTextBatch(
            input_ids=torch.tensor(sequence, dtype=torch.long).unsqueeze(0),
            prompt_lengths=torch.tensor([prompt_length], dtype=torch.long),
            graph_index=graph_index,
            labels=labels,
            num_nodes=num_nodes,
            num_edges=num_edges,
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> GraphTextBatch:
        return self._samples[idx]


def _collate_graph_text(
    batch: list[GraphTextBatch],
) -> dict[str, torch.Tensor | list]:
    max_len = max(b.input_ids.shape[1] for b in batch)

    padded_ids: list[torch.Tensor] = []
    prompt_lengths: list[int] = []
    graph_indices: list[int] = []
    labels: list[torch.Tensor | None] = []
    num_nodes: list[int] = []
    num_edges: list[int] = []

    for b in batch:
        seq_len = b.input_ids.shape[1]
        pad_len = max_len - seq_len
        padded = torch.cat(
            [b.input_ids, torch.zeros(1, pad_len, dtype=torch.long)],
            dim=1,
        )
        padded_ids.append(padded)
        prompt_lengths.append(int(b.prompt_lengths.item()))
        graph_indices.append(b.graph_index)
        labels.append(b.labels)
        num_nodes.append(b.num_nodes)
        num_edges.append(b.num_edges)

    return {
        "input_ids": torch.cat(padded_ids, dim=0),
        "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
        "graph_indices": graph_indices,
        "labels": labels,
        "num_nodes": torch.tensor(num_nodes, dtype=torch.long),
        "num_edges": torch.tensor(num_edges, dtype=torch.long),
    }


def _max_num_nodes(datasets: Sequence) -> int:
    max_nodes = 0
    for dataset in datasets:
        for graph in dataset:
            max_nodes = max(max_nodes, int(graph.num_nodes))
    if max_nodes == 0:
        raise ValueError("MalNetTiny appears to contain no nodes.")
    return max_nodes


def build_malnettiny_dataloaders(
    root: str,
    graph_tokenizer,
    llada_tokenizer,
    prompt_prefix: str = "Generate graph text:\n",
    max_length: int = 512,
    batch_size: int = 1,
    num_workers: int = 0,
    max_train_graphs: int | None = None,
    max_test_graphs: int | None = None,
    skip_invalid: bool = True,
    shuffle_train: bool = True,
    force_reload: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/validation/test dataloaders for MalNetTiny graph-text samples.
    """
    train_raw = _load_malnet_tiny(root, split="trainval", force_reload=force_reload)
    test_raw = _load_malnet_tiny(root, split="test", force_reload=force_reload)

    max_nodes = _max_num_nodes([train_raw, test_raw])
    graph_tokenizer.set_num_nodes(max_nodes)
    print(f"[MalNetTiny] Loaded splits: train={len(train_raw)}, test={len(test_raw)}.")
    print(f"[MalNetTiny] Graph tokenizer vocab set for max {max_nodes} nodes.")

    train_ds = MalNetTinyGraphTextDataset(
        dataset=train_raw,
        graph_tokenizer=graph_tokenizer,
        llada_tokenizer=llada_tokenizer,
        prompt_prefix=prompt_prefix,
        max_length=max_length,
        max_graphs=max_train_graphs,
        skip_invalid=skip_invalid,
    )
    test_ds = MalNetTinyGraphTextDataset(
        dataset=test_raw,
        graph_tokenizer=graph_tokenizer,
        llada_tokenizer=llada_tokenizer,
        prompt_prefix=prompt_prefix,
        max_length=max_length,
        max_graphs=max_test_graphs,
        skip_invalid=skip_invalid,
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


def build_malnettiny_pyg_dataloaders(
    root: str,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = True,
    force_reload: bool = False,
):
    """Build standard PyG graph dataloaders for dataset exploration or GNNs."""
    try:
        from torch_geometric.loader import DataLoader as PyGDataLoader  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "torch-geometric is required. Install it with:\n"
            "  pip install torch-geometric"
        ) from exc

    train_ds = _load_malnet_tiny(root, split="trainval", force_reload=force_reload)
    test_ds = _load_malnet_tiny(root, split="test", force_reload=force_reload)

    return (
        PyGDataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
        ),
        PyGDataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    )
