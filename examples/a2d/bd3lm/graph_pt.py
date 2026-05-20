from __future__ import annotations

import os
import random
from dataclasses import dataclass, field, replace

import accelerate
import datasets
import torch
import transformers

import dllm
from dllm.utils.graph_data import (
    infer_graph_tokenizer_stats,
    save_graph_tokenizer_metadata,
    tokenize_graphs,
    pyg_graph_to_simple_graph_data,
)
from dllm.utils.graph_token_strategies import (
    build_lm_tokenizer_strategy,
    tokenize_graphs_with_strategy,
)
from graph_tokenization import TokenizerFactory

logger = dllm.utils.get_default_logger(__name__)

VALID_STRATEGIES = ("original", "text_mapping", "selective_special")


# ---------------------------------------------------------------------------
# Argument dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = ".models/a2d/Qwen3-0.6B"


@dataclass
class DataArguments:
    pyg_dataset: str = "MUTAG"
    data_root: str = "./data/pyg"
    graph_tokenizer_type: str = "autograph"
    max_graphs: int = 256
    test_size: float = 0.1
    max_length: int = 1024
    malnet_num_hops: int = 2
    max_nodes_per_graph: int = 1024
    labeled_graph: bool = False
    undirected: bool = True
    append_eos: bool = True
    dataset_name: str | None = None
    disable_caching: bool = False
    token_strategy: str = field(
        default="selective_special",
        metadata={
            "help": (
                "How to map graph tokens to LLM token ids. "
                "Choices: original | text_mapping | selective_special. "
                "'original'          — shift all graph ids into a reserved embedding block. "
                "'text_mapping'      — render every graph token as text, no new tokens added. "
                "'selective_special' — add only 6 structural tokens as special tokens "
                "                     with inner-word mean embedding initialisation."
            )
        },
    )
    elliptic_num_hops: int = 2


@dataclass
class TrainingArguments(dllm.core.trainers.BD3LMConfig):
    output_dir: str = ".models/Qwen3/bd3lm/graph-pt"
    group_by_length: bool = False
    num_train_epochs: float = 100
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    block_size: int = 32


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------

from dllm.data.load_graph_data import load_pyg_dataset, select_graphs, normalize_dataset_name, load_graph_samples

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
    # Allow capping the tokenizer's num_nodes via CLI. A value <=0 means no cap.
    max_nodes = stats["max_num_nodes"]
    if getattr(data_args, "max_nodes_per_graph", 0) and data_args.max_nodes_per_graph > 0:
        max_nodes = min(max_nodes, int(data_args.max_nodes_per_graph))
    tokenizer.set_num_nodes(max_nodes)
    stats["max_num_nodes"] = int(max_nodes)
    if data_args.labeled_graph:
        tokenizer.set_num_node_and_edge_types(
            num_node_types=stats["num_node_types"],
            num_edge_types=stats["num_edge_types"],
        )
    return tokenizer, stats


# ---------------------------------------------------------------------------
# Special-token embedding initialisation
# ---------------------------------------------------------------------------

def _mean_embedding_for_word(
    word: str,
    lm_tokenizer,
    embedding_matrix: torch.Tensor,
) -> torch.Tensor | None:
    """
    Return the mean of the base-model embeddings for the subword tokens that
    make up *word* (the inner text of a structural token, e.g. "node" from
    "<node>").  Falls back to None when the word tokenises to zero pieces.

    Strategy rationale
    ------------------
    Structural special tokens have the form "<word>".  The angle-brackets carry
    almost no semantic content, so we initialise each new embedding from the
    mean of the LM's existing embeddings for the *inner word only*.  This
    places the new token near the LM's existing representation for that concept
    rather than at a random point in embedding space, which:
      - speeds convergence (the model already "knows" the word),
      - avoids the cold-start problem where a random init is far from any
        meaningful direction,
      - is strictly better than single-token approximation when the inner word
        is split across multiple sub-word pieces.

    Alternatives considered
    -----------------------
    • Full-token text "<node>" → tokenised pieces often include the brackets,
      which are punctuation with weak semantics; the brackets' embeddings add
      noise rather than signal.
    • Random Gaussian init → cheapest, but slowest convergence.
    • Copy the nearest existing token → reasonable, but picking "nearest" is
      ambiguous and fragile across tokenisers.
    """
    ids = lm_tokenizer(word, add_special_tokens=False)["input_ids"]
    if not ids:
        return None
    vecs = embedding_matrix[ids]            # (n_pieces, d_model)
    return vecs.mean(dim=0)                 # (d_model,)


@torch.no_grad()
def initialise_special_token_embeddings(
    model,
    lm_tokenizer,
    structural_tokens: list[str],
) -> None:
    
    embedding_matrix = model.get_input_embeddings().weight  # (vocab, d_model)

    # We need the *original* vocab embeddings as the source for mean-pooling.
    # At this point the matrix has already been resized, so we work with a
    # detached snapshot of the pre-existing rows only.
    original_vocab_size = embedding_matrix.shape[0] - len(structural_tokens)

    for token in structural_tokens:
        token_id = lm_tokenizer.convert_tokens_to_ids(token)

        # Skip tokens that already existed before resizing
        if token_id < original_vocab_size:
            logger.debug(
                f"[emb_init] '{token}' already in base vocab (id={token_id}), skipping."
            )
            continue

        # Extract inner word: "<node>" -> "node", "<bos_graph>" -> "bos_graph"
        inner = token.strip("<>")

        # Use the original rows only as source (slice the live matrix)
        source_rows = embedding_matrix[:original_vocab_size].detach()
        init_vec = _mean_embedding_for_word(inner, lm_tokenizer, source_rows)

        if init_vec is None:
            logger.warning(
                f"[emb_init] '{inner}' tokenises to zero pieces; "
                f"leaving '{token}' (id={token_id}) randomly initialised."
            )
            continue

        embedding_matrix[token_id] = init_vec.to(embedding_matrix.dtype)
        logger.info(
            f"[emb_init] '{token}' (id={token_id}) initialised from "
            f"mean of {lm_tokenizer(inner, add_special_tokens=False)['input_ids']} "
            f"(inner word: '{inner}')."
        )


# ---------------------------------------------------------------------------
# Dataset-building for each strategy
# ---------------------------------------------------------------------------

def build_token_dataset_original(
    graphs,
    graph_tokenizer,
    *,
    token_offset: int,
    labeled_graph: bool,
    dataset_name: str,
    test_size: float,
    seed: int,
):
    """Original approach: raw id shift into a reserved embedding block."""
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


def build_token_dataset_strategy(
    graphs,
    graph_tokenizer,
    lm_strategy,
    *,
    labeled_graph: bool,
    dataset_name: str,
    test_size: float,
    seed: int,
):
    """Strategy A or B: LM token ids produced by the strategy object."""
    rows = tokenize_graphs_with_strategy(
        graphs,
        graph_tokenizer,
        lm_strategy,
        labeled_graph=labeled_graph,
        dataset_name=dataset_name,
    )
    if not rows:
        raise ValueError("No graph samples were tokenized.")
    dataset = datasets.Dataset.from_list(rows)
    if test_size <= 0.0:
        return datasets.DatasetDict({"train": dataset})
    return dataset.train_test_split(test_size=test_size, seed=seed)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.token_strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"--token_strategy must be one of {VALID_STRATEGIES}, "
            f"got {data_args.token_strategy!r}"
        )

    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    with accelerate.PartialState().local_main_process_first():

        # --- Graph data -----------------------------------------------
        graphs, canonical_dataset_name = load_graph_samples(data_args, training_args.seed)

        # Log graph size statistics (nodes) so we can see what the network
        # will receive. This prints min/max/avg/median for quick inspection.
        try:
            from statistics import median

            node_counts = []
            for g in graphs:
                simple = pyg_graph_to_simple_graph_data(g)
                node_counts.append(int(simple.num_nodes))

            if node_counts:
                node_counts_sorted = sorted(node_counts)
                min_nodes = node_counts_sorted[0]
                max_nodes = node_counts_sorted[-1]
                avg_nodes = sum(node_counts_sorted) / len(node_counts_sorted)
                med_nodes = int(median(node_counts_sorted))
                logger.info(
                    f"Prepared {len(node_counts)} graphs — nodes: min={min_nodes},"
                    f" max={max_nodes}, avg={avg_nodes:.2f}, median={med_nodes}"
                )
            else:
                logger.info("No graphs prepared for training.")
        except Exception:
            logger.exception("Failed to compute graph size statistics")

        graph_tokenizer, stats = build_graph_tokenizer(data_args, graphs)

        # --- Model + LM tokenizer (no LoRA yet; resize before LoRA) ---
        model_args_no_lora = replace(model_args, lora=False)
        model = dllm.utils.get_model(model_args=model_args_no_lora)
        tokenizer = dllm.utils.get_tokenizer(model_args=model_args_no_lora)

        # --- Check model context size vs graph token lengths ----------------
        try:
            tmp_graph_tokenizer, tmp_stats = build_graph_tokenizer(data_args, graphs)
            graph_token_lengths = []
            for g in graphs:
                simple = pyg_graph_to_simple_graph_data(g)
                toks = tmp_graph_tokenizer.tokenize(simple)
                graph_token_lengths.append(len(toks))

            max_graph_tokens = max(graph_token_lengths) if graph_token_lengths else 0
            model_max_pos = getattr(getattr(model, "config", None), "max_position_embeddings", None)
            logger.info(f"Max graph-token length: {max_graph_tokens}; model max_position_embeddings: {model_max_pos}")

            if data_args.token_strategy != "original":
                lm_strategy_check = None
                try:
                    lm_strategy_check = build_lm_tokenizer_strategy(
                        data_args.token_strategy if data_args.token_strategy != "selective_special" else "selective_special",
                        tokenizer,
                        tmp_graph_tokenizer,
                        model=None,
                    )
                except Exception:
                    lm_strategy_check = None

                if lm_strategy_check is not None and graph_token_lengths:
                    largest = graphs[graph_token_lengths.index(max_graph_tokens)]
                    simple_l = pyg_graph_to_simple_graph_data(largest)
                    gtoks = tmp_graph_tokenizer.tokenize(simple_l).tolist()
                    lm_ids = lm_strategy_check.encode(gtoks)
                    lm_len = len(lm_ids)
                    logger.info(f"Estimated LM-token length for largest graph (strategy={data_args.token_strategy}): {lm_len}")
                    if model_max_pos is not None and lm_len > model_max_pos:
                        logger.warning(
                            "Estimated LM token length for graphs exceeds model's positional embedding size. "
                            "Consider increasing model context size or limiting graph token length (data_args.max_length)."
                        )
            else:
                if model_max_pos is not None and max_graph_tokens > model_max_pos:
                    logger.warning(
                        "Graph token length exceeds model's positional embedding size. "
                        "Consider limiting data_args.max_length or using a model with larger context."
                    )
        except Exception:
            logger.exception("Failed to estimate graph token lengths vs model context")

        # --- Strategy-specific setup ----------------------------------
        strategy = data_args.token_strategy
        lm_strategy = None   # only set for text_mapping / selective_special

        if strategy == "original":
            
            base_vocab_size = len(tokenizer)
            graph_vocab_size = len(graph_tokenizer)
            model.resize_token_embeddings(base_vocab_size + graph_vocab_size)
            logger.info(
                f"[original] Added {graph_vocab_size} graph token embeddings. "
                f"New vocab size: {base_vocab_size + graph_vocab_size}"
                f"Original vocab size: {base_vocab_size}"
            )
            tokenized_dataset = build_token_dataset_original(
                graphs,
                graph_tokenizer,
                token_offset=base_vocab_size,
                labeled_graph=data_args.labeled_graph,
                dataset_name=data_args.dataset_name or canonical_dataset_name,
                test_size=data_args.test_size,
                seed=training_args.seed,
            )

        elif strategy == "text_mapping":
            lm_strategy = build_lm_tokenizer_strategy(
                "text_mapping",
                tokenizer,
                graph_tokenizer,
                model=None,
            )
            logger.info(
                "[text_mapping] All graph tokens rendered as text. "
                f"Vocab size unchanged: {len(tokenizer)}"
            )
            tokenized_dataset = build_token_dataset_strategy(
                graphs,
                graph_tokenizer,
                lm_strategy,
                labeled_graph=data_args.labeled_graph,
                dataset_name=data_args.dataset_name or canonical_dataset_name,
                test_size=data_args.test_size,
                seed=training_args.seed,
            )

        else:
            lm_strategy = build_lm_tokenizer_strategy(
                "selective_special_tokens",
                tokenizer,
                graph_tokenizer,
                model=model,   # resize handled inside factory
            )
            structural_tokens = list(lm_strategy.STRUCTURAL_TOKENS)
            initialise_special_token_embeddings(model, tokenizer, structural_tokens)

            logger.info(
                f"[selective_special] Added up to {len(structural_tokens)} structural "
                f"special tokens with inner-word embedding init. "
                f"New vocab size: {len(tokenizer)}"
            )
            tokenized_dataset = build_token_dataset_strategy(
                graphs,
                graph_tokenizer,
                lm_strategy,
                labeled_graph=data_args.labeled_graph,
                dataset_name=data_args.dataset_name or data_args.pyg_dataset,
                test_size=data_args.test_size,
                seed=training_args.seed,
            )

        # --- LoRA (applied after all embedding resizes) ---------------
        if model_args.lora:
            model = dllm.utils.load_peft(model=model, model_args=model_args)

        # --- Save metadata -------------------------------------------
        base_vocab_size_for_meta = (
            len(tokenizer) - len(graph_tokenizer)
            if strategy == "original"
            else len(tokenizer)
        )
        save_graph_tokenizer_metadata(
            training_args.output_dir,
            {
                "token_strategy": strategy,
                "pyg_dataset": data_args.pyg_dataset,
                "dataset_name": data_args.dataset_name or canonical_dataset_name,
                "graph_tokenizer_type": data_args.graph_tokenizer_type,
                "labeled_graph": data_args.labeled_graph,
                "max_num_nodes": stats["max_num_nodes"],
                "num_node_types": stats["num_node_types"],
                "num_edge_types": stats["num_edge_types"],
                "base_vocab_size": base_vocab_size_for_meta,
                "graph_vocab_size": len(graph_tokenizer) if strategy == "original" else 0,
                "graph_token_offset": base_vocab_size_for_meta if strategy == "original" else 0,
            },
        )
        if lm_strategy is not None:
            lm_strategy.save_metadata(training_args.output_dir)

    # Wait for data prep to complete on all ranks before training starts
    accelerate.PartialState().wait_for_everyone()
    logger.info(f"Start graph MDLM pre-training (strategy={strategy})...")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    trainer = dllm.core.trainers.BD3LMTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset.get("test"),
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