from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW

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
        description="Fine-tune LLaDA on graph text using swappable tokenizers."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Hugging Face model id or local path for the LLaDA checkpoint.",
    )
    # Adjusted: Choices now reflect your new swappable options
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
    parser.add_argument("--train-device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./checkpoints/finetune-llada-graphs")
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
    tokenizer: GraphTokenizer, # Use Base class type
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


def run_finetuning(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    dataset = load_small_pyg_dataset(args.pyg_dataset, args.data_root)
    graph_tokenizer = build_graph_tokenizer(args.graph_tokenizer_type, dataset)
    graph_text_dataset = build_graph_text_samples(
        dataset=dataset,
        graph_tokenizer=graph_tokenizer,
        prompt_prefix=args.prompt,
        max_graphs=args.max_graphs,
    )

    llada = LLaDAModel(
        hf_model_path=args.model,
        tokenizer=None,
        device=args.train_device,
        torch_dtype=map_dtype(args.torch_dtype),
    )

    if args.use_lora:
        llada.prepare_for_lora(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )

    llada.model.train()

    optimizer = AdamW(
        llada.model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(graph_text_dataset)
    global_step = 0

    for epoch in range(args.epochs):
        indices = list(range(len(graph_text_dataset)))
        random.shuffle(indices)
        running_loss = 0.0

        for idx in indices:
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

            running_loss += float(loss.item())
            global_step += 1

            if global_step % 10 == 0 or global_step == total_steps:
                avg_loss = running_loss / max(global_step, 1)
                print(
                    f"[step {global_step}/{total_steps}] "
                    f"loss={loss.item():.4f} avg_loss={avg_loss:.4f}"
                )

        epoch_loss = running_loss / len(graph_text_dataset)
        print(f"[epoch {epoch + 1}/{args.epochs}] avg_loss={epoch_loss:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    llada.model.save_pretrained(output_dir)

    if hasattr(llada.tokenizer, "save_pretrained"):
        llada.tokenizer.save_pretrained(output_dir)

    print(f"Saved fine-tuned artifacts to: {output_dir.resolve()}")


def main() -> None:
    args = parse_args()
    run_finetuning(args)


if __name__ == "__main__":
    main()
