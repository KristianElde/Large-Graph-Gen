from __future__ import annotations

import argparse
from pathlib import Path

import datasets
import torch

import dllm
from dllm.utils.graph_edge_list_strategy import build_edge_list_strategy
from dllm.utils.eval_utils import run_graph_generation_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_graphs", type=int, default=32)
    parser.add_argument("--num_nodes", type=int, default=5000)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model from:", model_path)

    model_args = dllm.utils.ModelArguments(
        model_name_or_path=str(model_path),
        dtype="bfloat16" if torch.cuda.is_available() else "float32",
        lora=False,
    )

    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    model = dllm.utils.get_model(model_args=model_args)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    strategy = build_edge_list_strategy(
        tokenizer,
        undirected=True,
        labeled=False,
    )

    rows = []
    for _ in range(args.num_graphs):
        text = strategy.prompt + f"N={args.num_nodes}; E="
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        rows.append(
            {
                "input_ids": input_ids,
                "labels": input_ids.copy(),
            }
        )

    eval_dataset = datasets.Dataset.from_list(rows)

    run_graph_generation_eval(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        strategy=strategy,
        output_dir=str(output_dir),
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_graphs,
        temperature=args.temperature,
        batch_size=args.batch_size,
        labeled=False,
        undirected=True,
    )


if __name__ == "__main__":
    main()