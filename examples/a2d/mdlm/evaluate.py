"""
Evaluate generated graphs from an MDLM graph checkpoint.

Run:
    python -u /home/scur0503/dllm/examples/a2d/mdlm/evaluate.py \
        --model_path /home/scur0503/dllm/.models/Qwen3/mdlm/graph-pt/checkpoint-final \
        --generation_batch_size 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import transformers

REPO_ROOT = Path(__file__).resolve().parents[3]
GRAPH_EVAL_ROOT = REPO_ROOT / "graph_evaluation"
for path in (REPO_ROOT, GRAPH_EVAL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dllm
from dllm.data.load_graph_data import load_graph_samples
from dllm.utils.graph_data import (
    infer_graph_tokenizer_stats,
    pyg_graph_to_simple_graph_data,
)
from dllm.utils.graph_token_strategies import build_lm_tokenizer_strategy
from graph_evaluation.evaluator import Evaluator
from graph_tokenization import TokenizerFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--pyg_dataset", default=None)
    parser.add_argument("--data_root", default="/home/scur0503/dllm/data/pyg")
    parser.add_argument("--max_graphs", type=int, default=256)
    parser.add_argument("--generation_batch_size", type=int, default=16)
    parser.add_argument("--num_generated_graphs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def checkpoint_metadata(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path).resolve()
    for candidate in (
        path / "graph_tokenizer_metadata.json",
        path.parent / "graph_tokenizer_metadata.json",
    ):
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def build_graph_tokenizer(metadata: dict[str, Any], graphs, dataset_name: str):
    labeled_graph = bool(metadata.get("labeled_graph", False))
    stats = infer_graph_tokenizer_stats(graphs, labeled_graph=labeled_graph)
    tokenizer = TokenizerFactory.get_tokenizer(
        metadata.get("graph_tokenizer_type", "autograph"),
        dataset_names=[metadata.get("dataset_name", dataset_name)],
        max_length=-1,
        labeled_graph=labeled_graph,
        undirected=True,
        append_eos=True,
    )
    tokenizer.set_num_nodes(stats["max_num_nodes"])
    if labeled_graph:
        tokenizer.set_num_node_and_edge_types(
            num_node_types=stats["num_node_types"],
            num_edge_types=stats["num_edge_types"],
        )
    return tokenizer, stats


def make_lm_strategy(tokenizer, graph_tokenizer, metadata: dict[str, Any]):
    strategy = metadata.get("token_strategy", "selective_special")
    if strategy == "original":
        return None
    if strategy == "selective_special":
        strategy = "selective_special_tokens"

    lm_strategy = build_lm_tokenizer_strategy(
        strategy,
        tokenizer,
        graph_tokenizer,
        model=None,
    )

    # Populate reverse caches so generated LM token ids can be parsed as graph ids.
    for graph_token_id in range(len(graph_tokenizer)):
        lm_strategy.encode([graph_token_id])
    return lm_strategy


def graph_prompt(graph_tokenizer, lm_strategy, metadata: dict[str, Any]) -> list[int]:
    if metadata.get("token_strategy") == "original":
        offset = int(metadata.get("graph_token_offset", metadata["base_vocab_size"]))
        return [offset + graph_tokenizer.sos]
    return lm_strategy.encode([graph_tokenizer.sos])


def lm_tokens_to_graph_tokens(
    sequences: torch.Tensor,
    graph_tokenizer,
    lm_strategy,
    metadata: dict[str, Any],
) -> torch.Tensor:
    rows = []
    strategy = metadata.get("token_strategy", "selective_special")

    for sequence in sequences.detach().cpu().tolist():
        if strategy == "original":
            offset = int(metadata.get("graph_token_offset", metadata["base_vocab_size"]))
            vocab_size = int(metadata.get("graph_vocab_size", len(graph_tokenizer)))
            graph_ids = [
                token_id - offset
                for token_id in sequence
                if offset <= token_id < offset + vocab_size
            ]
        else:
            graph_ids = lm_strategy.decode(sequence)

        if not graph_ids or graph_ids[0] != graph_tokenizer.sos:
            graph_ids = [graph_tokenizer.sos] + graph_ids
        rows.append(torch.tensor(graph_ids, dtype=torch.long))

    width = max((row.numel() for row in rows), default=1)
    batch = torch.full(
        (len(rows), width),
        fill_value=graph_tokenizer.pad,
        dtype=torch.long,
    )
    for i, row in enumerate(rows):
        batch[i, : row.numel()] = row
    return batch


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def main() -> None:
    args = parse_args()
    transformers.set_seed(args.seed)

    # 1: Load checkpoint
    metadata = checkpoint_metadata(args.model_path)
    data_args = SimpleNamespace(
        pyg_dataset=args.pyg_dataset or metadata.get("pyg_dataset", "MUTAG"),
        data_root=args.data_root,
        max_graphs=args.max_graphs,
        elliptic_num_hops=2,
    )
    train_graphs, dataset_name = load_graph_samples(data_args, args.seed)
    graph_tokenizer, graph_stats = build_graph_tokenizer(
        metadata,
        train_graphs,
        dataset_name,
    )

    model_args = SimpleNamespace(
        model_name_or_path=args.model_path,
        dtype="bfloat16",
        load_in_4bit=False,
        attn_implementation=None,
        lora=False,
    )
    model = dllm.utils.get_model(model_args=model_args).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    lm_strategy = make_lm_strategy(tokenizer, graph_tokenizer, metadata)

    # 2: Generate samples
    sampler = dllm.core.samplers.MDLMSampler(model=model, tokenizer=tokenizer)
    sampler_config = dllm.core.samplers.MDLMSamplerConfig(
        steps=args.steps,
        max_new_tokens=args.max_new_tokens,
        block_size=args.block_size,
        temperature=args.temperature,
        return_dict=True,
    )

    prompt = graph_prompt(graph_tokenizer, lm_strategy, metadata)
    num_generated = args.num_generated_graphs or args.generation_batch_size
    generated = []
    while len(generated) < num_generated:
        batch_size = min(args.generation_batch_size, num_generated - len(generated))
        outputs = sampler.sample(
            [prompt.copy() for _ in range(batch_size)],
            sampler_config,
            return_dict=True,
        )
        generated.extend(outputs.sequences.detach().cpu())

    generated_lm_tokens = torch.stack(generated)
    generated_graph_tokens = lm_tokens_to_graph_tokens(
        generated_lm_tokens,
        graph_tokenizer,
        lm_strategy,
        metadata,
    )

    # 3: Run Evaluator
    evaluator_train_data = [
        pyg_graph_to_simple_graph_data(
            graph,
            labeled_graph=bool(metadata.get("labeled_graph", False)),
            dataset_name=metadata.get("dataset_name", dataset_name),
        )
        for graph in train_graphs
    ]
    metrics = Evaluator(tokenizer=graph_tokenizer)(
        tokenized_graphs=generated_graph_tokens,
        train_data=evaluator_train_data,
    )

    # 4: Print metrics
    print(
        json.dumps(
            json_ready(
                {
                    "metrics": metrics,
                    "model_path": str(Path(args.model_path).resolve()),
                    "pyg_dataset": dataset_name,
                    "token_strategy": metadata.get("token_strategy"),
                    "num_generated_graphs": len(generated),
                    "graph_tokenizer_stats": graph_stats,
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
