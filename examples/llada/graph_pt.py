"""
Fine-tune LLaDA on a graph dataset using AutoGraph tokens.

Example:
    PYTHONPATH=. accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/llada/graph_pt.py \
        --pyg_dataset MUTAG \
        --data_root ./data/pyg \
        --model_name_or_path GSAI-ML/LLaDA-8B-Base

This script converts each graph into an AutoGraph token sequence, shifts those
token ids into a reserved embedding range, and trains with the repo's existing
MDLM trainer so the batch format stays `input_ids` + `labels`.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, replace

import accelerate
import datasets
import transformers

import dllm
from dllm.utils.graph_data import (
    infer_graph_tokenizer_stats,
    save_graph_tokenizer_metadata,
    tokenize_graphs,
)
from graph_tokenization import TokenizerFactory

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Base"


@dataclass
class DataArguments:
    pyg_dataset: str = "MUTAG"
    data_root: str = "./data/pyg"
    graph_tokenizer_type: str = "autograph"
    max_graphs: int = 256
    test_size: float = 0.1
    max_length: int = 512
    labeled_graph: bool = False
    undirected: bool = True
    append_eos: bool = True
    dataset_name: str | None = None
    disable_caching: bool = False


@dataclass
class TrainingArguments(dllm.core.trainers.MDLMConfig):
    output_dir: str = ".models/LLaDA-8B-Base/graph-pt"
    group_by_length: bool = False
    num_train_epochs: float = 3
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4


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


def build_graph_tokenizer(data_args: DataArguments, graphs):
    stats = infer_graph_tokenizer_stats(graphs, labeled_graph=data_args.labeled_graph)
    tokenizer = TokenizerFactory.get_tokenizer(
        data_args.graph_tokenizer_type,
        dataset_names=[data_args.dataset_name or data_args.pyg_dataset],
        max_length=data_args.max_length,
        labeled_graph=data_args.labeled_graph,
        undirected=data_args.undirected,
        append_eos=data_args.append_eos,
    )
    tokenizer.set_num_nodes(stats["max_num_nodes"])
    if data_args.labeled_graph:
        tokenizer.set_num_node_and_edge_types(
            num_node_types=stats["num_node_types"],
            num_edge_types=stats["num_edge_types"],
        )
    return tokenizer, stats


def build_token_dataset(
    graphs,
    graph_tokenizer,
    *,
    token_offset: int,
    labeled_graph: bool,
    dataset_name: str,
    test_size: float,
    seed: int,
):
    rows = tokenize_graphs(
        graphs,
        graph_tokenizer,
        token_offset=token_offset,
        labeled_graph=labeled_graph,
        dataset_name=dataset_name,
    )
    if not rows:
        raise ValueError("No graph samples were tokenized.")
    dataset = datasets.Dataset.from_list(rows)
    if test_size <= 0.0:
        return datasets.DatasetDict({"train": dataset})
    return dataset.train_test_split(test_size=test_size, seed=seed)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    with accelerate.PartialState().local_main_process_first():
        raw_dataset = load_pyg_dataset(data_args.pyg_dataset, data_args.data_root)
        graphs = select_graphs(raw_dataset, data_args.max_graphs, training_args.seed)
        graph_tokenizer, stats = build_graph_tokenizer(data_args, graphs)

        model_args_no_lora = replace(model_args, lora=False)
        model = dllm.utils.get_model(model_args=model_args_no_lora)
        tokenizer = dllm.utils.get_tokenizer(model_args=model_args_no_lora)

        # Ensure tokenizer exposes a mask token required by MDLM training.
        # Some pretrained tokenizers (e.g., GPT-2) do not include a mask token by default.
        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({"mask_token": "<|mdm_mask|>"})
        base_vocab_size = len(tokenizer)
        graph_vocab_size = len(graph_tokenizer)
        model.resize_token_embeddings(base_vocab_size + graph_vocab_size)

        if model_args.lora:
            model = dllm.utils.load_peft(model=model, model_args=model_args)

        tokenized_dataset = build_token_dataset(
            graphs,
            graph_tokenizer,
            token_offset=base_vocab_size,
            labeled_graph=data_args.labeled_graph,
            dataset_name=data_args.dataset_name or data_args.pyg_dataset,
            test_size=data_args.test_size,
            seed=training_args.seed,
        )

        save_graph_tokenizer_metadata(
            training_args.output_dir,
            {
                "pyg_dataset": data_args.pyg_dataset,
                "dataset_name": data_args.dataset_name or data_args.pyg_dataset,
                "graph_tokenizer_type": data_args.graph_tokenizer_type,
                "labeled_graph": data_args.labeled_graph,
                "max_num_nodes": stats["max_num_nodes"],
                "num_node_types": stats["num_node_types"],
                "num_edge_types": stats["num_edge_types"],
                "base_vocab_size": base_vocab_size,
                "graph_vocab_size": graph_vocab_size,
                "graph_token_offset": base_vocab_size,
            },
        )

    accelerate.PartialState().wait_for_everyone()
    logger.info("Start graph token fine-tuning...")

    trainer = dllm.core.trainers.MDLMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"] if "test" in tokenized_dataset else None,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            return_tensors="pt",
            padding=True,
        ),
    )

    trainer.train()
    checkpoint_dir = os.path.join(training_args.output_dir, "checkpoint-final")
    trainer.save_model(checkpoint_dir)
    trainer.processing_class.save_pretrained(checkpoint_dir)


if __name__ == "__main__":
    train()