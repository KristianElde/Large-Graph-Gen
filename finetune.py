from __future__ import annotations

import argparse
import random
import sys
import time
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# Change: Import the Factory and the Base class instead of just AutoGraph
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
        description="Fine-tune LLaDA on graph text using swappable tokenizers and dataloaders."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Hugging Face model id or local path for the LLaDA checkpoint.",
    )
    parser.add_argument(
        "--dataloader",
        type=str,
        default="pyg",
        help="Dataloader type. Options: 'pyg' (PyTorch Geometric), 'bitcoin', or custom. "
             "The system will look for dataset/{dataloader}.py for custom loaders.",
    )
    # Adjusted: Choices now reflect your new swappable options
    parser.add_argument(
        "--graph-tokenizer-type",
        type=str,
        default="autograph",
        choices=["autograph", "nauty", "kandinsky"],
        help="Graph tokenizer type. Supported: autograph, nauty, kandinsky.",
    )
    parser.add_argument("--pyg-dataset", type=str, default="MUTAG",
                        help="PyG dataset name (only used if --dataloader=pyg)")
    parser.add_argument("--data-root", type=str, default="./data",
                        help="Root directory for dataset storage.")
    parser.add_argument("--max-graphs", type=int, default=64,
                        help="Max graphs to use (only for PyG dataloader).")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for training.")
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
    # Note: All your tokenizers share these IDs because they inherit from AutoGraph
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
            # Shared logic for node re-indexing
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
    # Adjusted: Use the Factory to get the requested tokenizer
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
        # Important: set dataset name for the tokenizer's internal mapping
        graph.dataset_name = dataset.name

        # This call now uses nauty/kandinsky logic if selected
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


def load_custom_dataloader(dataloader_name: str) -> object:
    """
    Dynamically load a dataloader module from dataset/{dataloader_name}.py.

    Checks if the module exists before importing.
    Raises ValueError if not found.
    """
    module_path = Path(__file__).parent / "dataset" / f"{dataloader_name}.py"

    if not module_path.exists():
        raise ValueError(
            f"Dataloader '{dataloader_name}' not found. "
            f"Expected to find: {module_path}\n"
            f"Available dataloaders are in the dataset/ directory."
        )

    spec = importlib.util.spec_from_file_location(
        f"dataset_{dataloader_name}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dataloaders(
    dataloader_type: str,
    args: argparse.Namespace,
    graph_tokenizer: GraphTokenizer | None,
    llada_tokenizer,
) -> tuple[DataLoader | Iterator | GraphTextDataset, GraphTokenizer]:
    """
    Load training dataloader and graph tokenizer based on dataloader type.

    Returns:
        (train_loader, graph_tokenizer) where train_loader can be either:
        - For PyG: GraphTextDataset wrapped in a simple iterator
        - For custom: DataLoader from the custom module
    """
    if dataloader_type == "pyg":
        print("[1/6] Loading PyG dataset...")
        dataset = load_small_pyg_dataset(args.pyg_dataset, args.data_root)
        print(
            f"      ✓ Loaded {len(dataset)} graphs from {args.pyg_dataset}\n")

        print("[2/6] Setting up tokenizer...")
        graph_tokenizer = build_graph_tokenizer(
            args.graph_tokenizer_type, dataset)
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
        print(
            f"      ✓ Created {len(graph_text_dataset)} (prompt, graph) pairs\n")

        # Wrap in simple iterator (maintains original PyG behavior)
        return graph_text_dataset, graph_tokenizer

    else:
        # Load custom dataloader from dataset/{dataloader_type}.py
        print(f"[1/3] Loading custom dataloader: {dataloader_type}...")
        try:
            module = load_custom_dataloader(dataloader_type)
        except (ValueError, ImportError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

        # Look for the main builder function (convention: build_{type}_dataloaders)
        builder_fn_name = f"build_{dataloader_type}_dataloaders"
        if hasattr(module, builder_fn_name):
            builder_fn = getattr(module, builder_fn_name)
        else:
            # Fallback: accept any function starting with 'build_' (for backwards
            # compatibility with dataset modules that use a different name,
            # e.g. build_elliptic_dataloaders in dataset/bitcoin.py).
            builder_fn = None
            for name in dir(module):
                if name.startswith("build_") and callable(getattr(module, name)):
                    builder_fn = getattr(module, name)
                    print(f"      ⚠ Using fallback builder '{name}' from module.")
                    break
            if builder_fn is None:
                print(
                    f"ERROR: Dataloader module must provide '{builder_fn_name}()' or another 'build_' function.",
                    file=sys.stderr,
                )
                sys.exit(1)

        print(f"[2/3] Setting up {dataloader_type} dataloader...")
        try:
            # For custom loaders, we need to create the graph tokenizer first
            if graph_tokenizer is None:
                # Create a placeholder tokenizer - the custom loader may override it
                graph_tokenizer = TokenizerFactory.get_tokenizer(
                    args.graph_tokenizer_type,
                    dataset_names=["custom"],
                    undirected=True,
                    append_eos=True,
                )

            train_loader, test_loader = builder_fn(
                root=args.data_root,
                graph_tokenizer=graph_tokenizer,
                llada_tokenizer=llada_tokenizer,
                prompt_prefix=args.prompt,
                max_length=args.max_length,
                batch_size=args.batch_size,
            )
            print(f"      ✓ {dataloader_type} dataloader ready\n")
        except Exception as e:
            print(
                f"ERROR: Failed to build {dataloader_type} dataloaders: {e}", file=sys.stderr)
            sys.exit(1)

        return train_loader, graph_tokenizer


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


def is_batch_loader(data_source) -> bool:
    """Check if data source is a DataLoader (batch mode) vs GraphTextDataset (single mode)."""
    return isinstance(data_source, DataLoader)


def run_finetuning(args: argparse.Namespace) -> None:
    training_start_time = time.time()

    # Print header
    print("\n" + "="*70)
    print("  LLaDA Fine-tuning on Graphs".center(70))
    print("="*70 + "\n")

    set_seed(args.seed)

    # -------- Model Loading -------- #
    print(f"[MODEL] Loading LLaDA model from {args.model}...")
    llada = LLaDAModel(
        hf_model_path=args.model,
        tokenizer=None,
        device=args.train_device,
        torch_dtype=map_dtype(args.torch_dtype),
    )
    gpu_mem = get_gpu_memory_usage()
    print(f"        ✓ Model loaded on {args.train_device}")
    if gpu_mem["allocated"] > 0:
        print(
            f"        ✓ GPU memory: {gpu_mem['allocated']:.1f} MB (reserved: {gpu_mem['reserved']:.1f} MB)\n")
    else:
        print()

    # -------- Dataloader Loading -------- #
    print(f"[DATA] Loading {args.dataloader} dataloader...")
    train_loader, graph_tokenizer = load_dataloaders(
        dataloader_type=args.dataloader,
        args=args,
        graph_tokenizer=None,  # Will be created in load_dataloaders
        llada_tokenizer=llada.tokenizer,
    )
    is_batch_mode = is_batch_loader(train_loader)
    print(
        f"[DATA] Training mode: {'Batch (DataLoader)' if is_batch_mode else 'Single sample'}\n")

    # -------- LoRA Setup -------- #
    if args.use_lora:
        print("[LORA] Preparing LoRA adapters...")
        llada.prepare_for_lora(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        print(
            f"       ✓ LoRA enabled (r={args.lora_r}, alpha={args.lora_alpha})\n")
    else:
        print("[LORA] Skipping LoRA (full model fine-tuning)\n")

    # -------- Training Setup -------- #
    print("[SETUP] Finalizing training...")
    llada.model.train()
    optimizer = AdamW(
        llada.model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Calculate total steps
    if is_batch_mode:
        steps_per_epoch = len(train_loader)
    else:
        steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch

    config_items = {
        "Dataloader": args.dataloader,
        "Graph Tokenizer": args.graph_tokenizer_type,
        "Epochs": str(args.epochs),
        "Steps per Epoch": str(steps_per_epoch),
        "Total Steps": str(total_steps),
        "Batch Size": str(args.batch_size),
        "Max Sequence Length": str(args.max_length),
        "Device": args.train_device,
        "Precision": args.torch_dtype,
    }
    print_summary("TRAINING CONFIG", config_items)

    # -------- Training Loop -------- #
    global_step = 0
    all_losses = []
    epoch_losses = []

    epoch_pbar = tqdm(range(args.epochs), desc="Epochs",
                      position=0, leave=True)
    for epoch in epoch_pbar:
        epoch_start = time.time()
        running_loss = 0.0

        step_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            position=1,
            leave=False,
            total=steps_per_epoch if is_batch_mode else None
        )

        for batch_idx, batch_data in enumerate(step_pbar):
            # Handle both single-sample and batch modes
            if is_batch_mode:
                # Batch mode (e.g., bitcoin dataloader)
                input_ids = batch_data["input_ids"].to(args.train_device)
                prompt_lengths = batch_data["prompt_lengths"].to(
                    args.train_device)
            else:
                # Single sample mode (PyG)
                sample = batch_data
                input_ids, prompt_lengths = build_training_sequence(
                    tokenizer=llada.tokenizer,
                    sample=sample,
                    max_length=args.max_length,
                )
                input_ids = input_ids.to(args.train_device)
                prompt_lengths = prompt_lengths.to(args.train_device)

            # Forward + backward
            loss = llada.compute_sft_loss(
                input_ids=input_ids,
                prompt_lengths=prompt_lengths,
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Track losses
            loss_val = float(loss.item())
            running_loss += loss_val
            all_losses.append(loss_val)
            global_step += 1

            # Update progress bar
            avg_loss = running_loss / (batch_idx + 1)
            step_pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "avg": f"{avg_loss:.4f}"
            })

        epoch_loss = running_loss / len(step_pbar)
        epoch_losses.append(epoch_loss)
        epoch_time = time.time() - epoch_start

        epoch_pbar.set_postfix(
            {"loss": f"{epoch_loss:.4f}", "time": f"{epoch_time:.1f}s"})

    # -------- Training Summary -------- #
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

    # -------- Save Model -------- #
    print("[SAVE] Saving fine-tuned model...")
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
