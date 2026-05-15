from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

from graph_tokenization import TokenizerFactory, GraphTokenizer
from models import LLaDAModel

DEFAULT_PROMPT = "Generate graph text:\n"


@dataclass
class GraphTextSample:
    prompt: str
    answer: str


class GraphTextDataset:
    def __init__(self, samples: list[GraphTextSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> GraphTextSample:
        return self.samples[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune LLaDA on graph text using swappable tokenizers."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Hugging Face model id or local path for the LLaDA checkpoint.",
    )
    parser.add_argument(
        "--graph-tokenizer-type",
        type=str,
        default="autograph",
        choices=["autograph", "nauty", "kandinsky"],
        help="Graph tokenizer type. Supported: autograph, nauty, kandinsky.",
    )
    parser.add_argument("--pyg-dataset", type=str, default="MUTAG")
    parser.add_argument("--data-root", type=str, default="./data/pyg")
    parser.add_argument("--max-graphs", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--train-device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16",
                        choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str,
                        default="./checkpoints/finetune-llada-graphs")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def map_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    return mapping[dtype_name]


def graph_tokens_to_text(
    tokens: torch.Tensor,
    tokenizer: GraphTokenizer,  # Use Base class type
) -> str:
    special = {
        tokenizer.sos: "<sos>",
        tokenizer.reset: "<reset>",
        tokenizer.ladj: "<ladj>",
        tokenizer.radj: "<radj>",
        tokenizer.eos: "<eos>",
        tokenizer.pad: "<pad>",
    }

    parts: list[str] = []
    for token in tokens.tolist():
        if token in special:
            parts.append(special[token])
        else:
            node_id = token - tokenizer.idx_offset
            parts.append(f"n{node_id}")
    return " ".join(parts)


def load_small_pyg_dataset(dataset_name: str, root: str):
    try:
        from torch_geometric.datasets import TUDataset
    except ImportError as exc:
        raise ImportError(
            "torch-geometric is required. Install it with: "
            "pip install torch-geometric"
        ) from exc

    dataset = TUDataset(root=root, name=dataset_name)
    if len(dataset) == 0:
        raise ValueError(f"PyG dataset '{dataset_name}' is empty.")
    return dataset


def build_graph_tokenizer(tokenizer_type: str, dataset) -> GraphTokenizer:
    tokenizer = TokenizerFactory.get_tokenizer(
        tokenizer_type,
        dataset_names=[dataset.name],
        undirected=True,
        append_eos=True,
    )

    # Standard setup for vocab size
    max_num_nodes = max(int(graph.num_nodes) for graph in dataset)
    tokenizer.set_num_nodes(max_num_nodes)
    return tokenizer


def build_graph_text_samples(
    dataset,
    graph_tokenizer: GraphTokenizer,
    prompt_prefix: str,
    max_graphs: int,
) -> GraphTextDataset:
    num_graphs = min(len(dataset), max_graphs)
    samples: list[GraphTextSample] = []

    for i in range(num_graphs):
        graph = dataset[i]
        graph.dataset_name = dataset.name

        token_ids = graph_tokenizer.tokenize(graph)

        graph_text = graph_tokens_to_text(token_ids, graph_tokenizer)
        samples.append(
            GraphTextSample(prompt=prompt_prefix, answer=graph_text)
        )
    return GraphTextDataset(samples)


def build_training_sequence(
    tokenizer,
    sample: GraphTextSample,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(sample.answer, add_special_tokens=False)

    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    sequence_ids: list[int] = []
    if bos_token_id is not None:
        sequence_ids.append(int(bos_token_id))
    sequence_ids.extend(prompt_ids)
    prompt_length = len(sequence_ids)
    sequence_ids.extend(answer_ids)
    if eos_token_id is not None:
        sequence_ids.append(int(eos_token_id))

    if prompt_length >= max_length:
        raise ValueError(
            "Prompt consumes the full sequence length. Increase "
            "--max-length or shorten --prompt."
        )

    sequence_ids = sequence_ids[:max_length]
    if len(sequence_ids) <= prompt_length:
        raise ValueError(
            "Sequence truncation removed all answer tokens. "
            "Increase --max-length."
        )

    input_ids = torch.tensor(sequence_ids, dtype=torch.long).unsqueeze(0)
    prompt_lengths = torch.tensor([prompt_length], dtype=torch.long)
    return input_ids, prompt_lengths


def get_gpu_memory_usage() -> dict[str, float]:
    """Return GPU memory stats in MB."""
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "max": 0.0}
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1e6
    reserved = torch.cuda.memory_reserved() / 1e6
    max_allocated = torch.cuda.max_memory_allocated() / 1e6
    return {"allocated": allocated, "reserved": reserved, "max": max_allocated}


def print_summary(title: str, items: dict[str, str], width: int = 70) -> None:
    """Pretty-print a summary box."""
    print("\n" + "=" * width)
    print(f"  {title}".ljust(width - 1) + "=")
    print("=" * width)
    for key, value in items.items():
        line = f"  {key}: {value}"
        print(line.ljust(width - 1) + " ")
    print("=" * width + "\n")


def run_finetuning(args: argparse.Namespace) -> None:
    training_start_time = time.time()

    print("\n" + "="*70)
    print("  LLaDA Fine-tuning on Graphs".center(70))
    print("="*70 + "\n")

    set_seed(args.seed)

    print("[1/6] Loading dataset...")
    dataset = load_small_pyg_dataset(args.pyg_dataset, args.data_root)
    print(f"      ✓ Loaded {len(dataset)} graphs from {args.pyg_dataset}\n")

    print("[2/6] Setting up tokenizer...")
    graph_tokenizer = build_graph_tokenizer(args.graph_tokenizer_type, dataset)
    max_num_nodes = max(int(graph.num_nodes) for graph in dataset)
    print(
        f"      ✓ {args.graph_tokenizer_type} tokenizer ready (max {max_num_nodes} nodes)\n")

    print("[3/6] Building training samples...")
    graph_text_dataset = build_graph_text_samples(
        dataset=dataset,
        graph_tokenizer=graph_tokenizer,
        prompt_prefix=args.prompt,
        max_graphs=args.max_graphs,
    )
    print(f"      ✓ Created {len(graph_text_dataset)} (prompt, graph) pairs\n")

    print("[4/6] Loading LLaDA model...")
    llada = LLaDAModel(
        hf_model_path=args.model,
        tokenizer=None,
        device=args.train_device,
        torch_dtype=map_dtype(args.torch_dtype),
    )
    gpu_mem = get_gpu_memory_usage()
    print(f"      ✓ Model loaded on {args.train_device}")
    if gpu_mem["allocated"] > 0:
        print(
            f"      ✓ GPU memory: {gpu_mem['allocated']:.1f} MB (reserved: {gpu_mem['reserved']:.1f} MB)\n")
    else:
        print()

    if args.use_lora:
        print("[5/6] Preparing LoRA adapters...")
        llada.prepare_for_lora(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        print(
            f"      ✓ LoRA enabled (r={args.lora_r}, alpha={args.lora_alpha})\n")
    else:
        print("[5/6] Skipping LoRA (full model fine-tuning)\n")

    print("[6/6] Finalizing training setup...")
    llada.model.train()
    optimizer = AdamW(
        llada.model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    print(
        f"      ✓ Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})\n")

    total_steps = args.epochs * len(graph_text_dataset)
    config_items = {
        "Dataset": f"{args.pyg_dataset} ({len(graph_text_dataset)} samples)",
        "Tokenizer": args.graph_tokenizer_type,
        "Epochs": str(args.epochs),
        "Total Steps": str(total_steps),
        "Batch Size": "1 (per-sample training)",
        "Max Sequence Length": str(args.max_length),
        "Device": args.train_device,
        "Precision": args.torch_dtype,
    }
    print_summary("TRAINING CONFIG", config_items)

    global_step = 0
    all_losses = []
    epoch_losses = []

    epoch_pbar = tqdm(range(args.epochs), desc="Epochs",
                      position=0, leave=True)
    for epoch in epoch_pbar:
        epoch_start = time.time()
        indices = list(range(len(graph_text_dataset)))
        random.shuffle(indices)
        running_loss = 0.0

        step_pbar = tqdm(
            indices,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            position=1,
            leave=False,
            total=len(indices)
        )

        for idx in step_pbar:
            sample = graph_text_dataset[idx]
            input_ids, prompt_lengths = build_training_sequence(
                tokenizer=llada.tokenizer,
                sample=sample,
                max_length=args.max_length,
            )

            input_ids = input_ids.to(args.train_device)
            prompt_lengths = prompt_lengths.to(args.train_device)

            loss = llada.compute_sft_loss(
                input_ids=input_ids,
                prompt_lengths=prompt_lengths,
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            loss_val = float(loss.item())
            running_loss += loss_val
            all_losses.append(loss_val)
            global_step += 1

            avg_loss = running_loss / \
                (step_pbar.n + 1) if step_pbar.n >= 0 else 0
            step_pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "avg": f"{avg_loss:.4f}"
            })

        epoch_loss = running_loss / len(graph_text_dataset)
        epoch_losses.append(epoch_loss)
        epoch_time = time.time() - epoch_start

        epoch_pbar.set_postfix(
            {"loss": f"{epoch_loss:.4f}", "time": f"{epoch_time:.1f}s"})

    total_time = time.time() - training_start_time
    avg_loss_all = sum(all_losses) / len(all_losses) if all_losses else 0.0
    min_loss = min(epoch_losses) if epoch_losses else 0.0

    final_gpu_mem = get_gpu_memory_usage()

    summary_items = {
        "Total Time": f"{total_time:.1f}s ({total_time/60:.1f}m)",
        "Total Steps": str(global_step),
        "Overall Avg Loss": f"{avg_loss_all:.6f}",
        "Best Epoch Loss": f"{min_loss:.6f}",
        "Final Epoch Loss": f"{epoch_losses[-1]:.6f}" if epoch_losses else "N/A",
    }
    if final_gpu_mem["max"] > 0:
        summary_items["Peak GPU Memory"] = f"{final_gpu_mem['max']:.1f} MB"

    print_summary("TRAINING SUMMARY", summary_items)

    print("Saving fine-tuned model...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    llada.model.save_pretrained(output_dir)

    if hasattr(llada.tokenizer, "save_pretrained"):
        llada.tokenizer.save_pretrained(output_dir)

    print(f"✓ Saved to: {output_dir.resolve()}\n")


def main() -> None:
    args = parse_args()
    run_finetuning(args)


if __name__ == "__main__":
    main()
