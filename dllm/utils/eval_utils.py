from __future__ import annotations

"""
graph_generation_eval.py
========================
Post-training evaluation loop for the graph diffusion LM.

What it does
------------
1. Takes the eval split of ``tokenized_dataset`` (already built in train.py).
2. For each eval graph, encodes a *prefix* — the first ``prefix_frac`` tokens
   of the graph dict string (i.e. the opening ``{`` plus some complete node
   entries).  This simulates "given the start of a graph, continue it".
3. Runs the trained model's ``generate()`` method to complete the sequence.
4. Decodes and parses the completed text back into an adjacency structure via
   ``dict_string_to_adjacency``.
5. Prints a sample input/output to stdout for quick sanity-checking.
6. Writes a JSON file ``graph_generation_eval.json`` to ``output_dir``
   containing every prefix + completion + parsed adjacency + validity flag.

Integration in train.py
-----------------------
After ``trainer.train()`` and ``trainer.save_model(...)``, add::

    from graph_generation_eval import run_graph_generation_eval
    run_graph_generation_eval(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=tokenized_dataset.get("test"),
        strategy=lm_strategy,
        output_dir=training_args.output_dir,
        # optional overrides:
        max_new_tokens=training_args.graph_eval_max_new_tokens,
        num_samples=training_args.graph_eval_num_generated_graphs,
        prefix_frac=0.4,            # use first 40 % of each sequence as prefix
        temperature=training_args.graph_eval_temperature,
        batch_size=training_args.graph_eval_generation_batch_size,
        labeled=data_args.labeled_graph,
    )

Generation API assumptions
--------------------------
``MDLMTrainer`` wraps a masked-diffusion model.  The generation call used here
is::

    model.generate(
        input_ids=...,          # (B, prefix_len)   — the prefix token ids
        attention_mask=...,     # (B, prefix_len)   — all ones for the prefix
        max_new_tokens=N,
        temperature=T,
    )

which returns a tensor of shape ``(B, prefix_len + generated_len)``.  If your
``MDLMTrainer`` exposes a different generation signature (e.g. via a
``MDLMSampler`` object), swap out ``_generate_batch`` below accordingly.

Output file schema
------------------
``<output_dir>/graph_generation_eval.json`` — a JSON object::

    {
      "config": { "num_samples": N, "prefix_frac": 0.4, ... },
      "samples": [
        {
          "index":            int,
          "prefix_text":      str,          // the conditioning prefix
          "generated_text":   str,          // full completed string
          "valid":            bool,         // parses to a non-empty adjacency?
          "num_nodes_prefix": int,          // nodes present in the prefix
          "num_nodes_output": int,          // nodes present after generation
          "adjacency":        {str: [int]}, // adjacency list as plain dict
          "node_types":       {str: int},   // empty unless labeled=True
          "edge_types":       {str: int},   // empty unless labeled=True (str = "src,dst")
        },
        ...
      ],
      "summary": {
        "num_valid":   int,
        "validity_rate": float,
        "avg_nodes_prefix": float,
        "avg_nodes_output": float,
      }
    }
"""

import json
import logging
import os
import textwrap
import time
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal: build a prefix from a full token sequence
# ---------------------------------------------------------------------------

def _make_prefix(
    input_ids: list[int],
    tokenizer,
    prefix_frac: float,
    labeled: bool,
) -> tuple[list[int], str]:
    """
    Slice ``input_ids`` at ``prefix_frac`` of its length, then walk *backwards*
    to land on a clean node-entry boundary so the prefix is a valid partial
    graph dict.

    Returns
    -------
    prefix_ids : list[int]  — token ids of the prefix
    prefix_text : str       — decoded prefix string (for logging / output)

    Boundary detection
    ------------------
    We decode the raw slice and look for the last complete ``N: [...]`` or
    ``N: {...}`` entry.  We keep everything up to and including that entry,
    so the prefix is a valid half-open dict::

        {0: [1, 2], 1: [0

    becomes →  ``{0: [1, 2]``  (trailing ``, `` stripped).

    We do *not* close the prefix with ``}`` — the model is expected to
    generate the remainder including the closing brace.
    """
    # Slice at prefix_frac, but keep at least 1 token (the opening '{')
    target_len = max(1, int(len(input_ids) * prefix_frac))
    raw_slice   = input_ids[:target_len]
    prefix_text = tokenizer.decode(raw_slice, skip_special_tokens=True)

    # Walk back to the last complete entry
    if labeled:
        from dllm.utils.graph_dict_strategy import _find_last_complete_labeled_entry
        end = _find_last_complete_labeled_entry(prefix_text)
    else:
        from dllm.utils.graph_dict_strategy import _find_last_complete_unlabeled_entry
        end = _find_last_complete_unlabeled_entry(prefix_text)

    if end == -1:
        # No complete entry yet — keep just the opening brace
        clean_prefix = "{"
    else:
        # Strip trailing ", " to leave a clean partial dict
        clean_prefix = prefix_text[:end].rstrip(", ")

    # Re-tokenize the clean prefix to get exact ids
    prefix_ids = tokenizer.encode(clean_prefix, add_special_tokens=False)
    return prefix_ids, clean_prefix


# ---------------------------------------------------------------------------
# Internal: run one generation batch
# ---------------------------------------------------------------------------

def _generate_batch(
    model,
    tokenizer,
    prefix_ids_batch: list[list[int]],
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
    sampler_config=None,
    generation_mode: str = "sample",   # "sample" | "infill"
) -> list[str]:
    """
    Generate completions using MDLMSampler.

    generation_mode
    ---------------
    "infill" : Concatenates [prefix + mask * max_new_tokens] and calls
               infill(). The prefix is frozen; only the masked suffix is
               filled. The model attends to the full sequence (including
               the prefix) at every diffusion step.

    "sample" : Passes only the prefix as a prompt and calls sample(),
               which appends its own mask tail and runs the full
               masked-diffusion generation loop. The prompt is used
               as conditioning context.
    """
    from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig

    if generation_mode not in ("sample", "infill"):
        raise ValueError(f"generation_mode must be 'sample' or 'infill', got {generation_mode!r}")

    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError(
            "tokenizer.mask_token_id is None — the tokenizer must have a "
            "mask token for MDLM generation."
        )

    if sampler_config is None:
        sampler_config = MDLMSamplerConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            steps=max_new_tokens,
            block_size=max_new_tokens,
            remasking="low_confidence",
            return_dict=False,
        )

    sampler = MDLMSampler(model=model, tokenizer=tokenizer)

    # ------------------------------------------------------------------
    # Build inputs
    # ------------------------------------------------------------------
    if generation_mode == "infill":
        # [prefix (frozen)] + [mask × max_new_tokens (to be filled)]
        inputs = []
        for ids in prefix_ids_batch:
            clean_ids = [tok for tok in ids if tok != mask_id]
            inputs.append(clean_ids + [mask_id] * max_new_tokens)

        output_ids = sampler.infill(inputs=inputs, config=sampler_config)
        # output_ids: (B, max_seq_len) — right-padded with EOS

        completed_texts = []
        for i, ids in enumerate(prefix_ids_batch):
            prefix_len = len([tok for tok in ids if tok != mask_id])
            full_ids = output_ids[i, : prefix_len + max_new_tokens].tolist()
            completed_texts.append(
                tokenizer.decode(full_ids, skip_special_tokens=True)
            )

    else:  # "sample"
        # Pass prefix as prompt; sample() appends its own mask tail
        inputs = []
        for ids in prefix_ids_batch:
            clean_ids = [tok for tok in ids if tok != mask_id]
            inputs.append(clean_ids)

        output_ids = sampler.sample(
            inputs=inputs,
            config=sampler_config,
        )
        # output_ids: (B, max(prompt_lens) + max_new_tokens)
        # sample() builds: [prompt][mask...] and fills the mask region

        completed_texts = []
        for i, ids in enumerate(prefix_ids_batch):
            prefix_len = len([tok for tok in ids if tok != mask_id])
            # Full output already contains prefix + generated tokens
            full_ids = output_ids[i, : prefix_len + max_new_tokens].tolist()
            completed_texts.append(
                tokenizer.decode(full_ids, skip_special_tokens=True)
            )

    return completed_texts


# ---------------------------------------------------------------------------
# Internal: parse and validate a completed graph text
# ---------------------------------------------------------------------------

def _parse_completion(
    text: str,
    labeled: bool,
    strategy,
    *,
    undirected: bool = True,
) -> dict[str, Any]:
    """
    Parse a completed generation and run strict graph validity checks.

    The generated text may contain the natural-language prompt before the graph
    dictionary.  We first extract the substring beginning at the first ``{``;
    otherwise valid samples can be incorrectly scored as invalid.
    """
    from dllm.utils.graph_dict_strategy import (
        dict_string_to_adjacency,
        extract_graph_dict_text,
        graph_dict_text_is_complete,
        is_valid_adjacency,
    )

    graph_text = extract_graph_dict_text(text)
    complete_dict = graph_dict_text_is_complete(graph_text)

    try:
        node_order, adjacency, node_types, edge_types = dict_string_to_adjacency(
            graph_text, labeled=labeled
        )
        structure_valid, validity_error = is_valid_adjacency(
            adjacency,
            undirected=undirected,
            require_contiguous_nodes=True,
            allow_self_loops=False,
        )
        valid = complete_dict and structure_valid
        if not complete_dict:
            validity_error = "incomplete or unbalanced graph dict"
    except Exception as exc:
        logger.debug(f"Parse error: {exc!r}  text={text[:120]!r}")
        node_order, adjacency, node_types, edge_types = [], {}, {}, {}
        valid = False
        validity_error = f"parse error: {exc!r}"

    # Serialise to plain Python dicts (JSON-safe keys)
    return {
        "valid"         : valid,
        "validity_error": validity_error,
        "graph_text"    : graph_text,
        "num_nodes"     : len(adjacency),
        "adjacency"     : {str(k): v for k, v in adjacency.items()},
        "node_types"    : {str(k): v for k, v in node_types.items()},
        "edge_types"    : {f"{s},{d}": t for (s, d), t in edge_types.items()},
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_graph_generation_eval(
    model,
    tokenizer,
    eval_dataset,
    strategy,
    output_dir: str,
    *,
    max_new_tokens: int    = 256,
    num_samples: int       = 16,
    prefix_frac: float     = 0.4,
    temperature: float     = 1.0,
    batch_size: int        = 4,
    labeled: bool          = False,
    filename: str          = "graph_generation_eval.json",
    generation_mode: str   = "sample",          # ← new
    sampler_config         = None,              # ← new, optional MDLMSamplerConfig
    undirected: bool       = True,
) -> None:
    """
    Run the post-training graph generation evaluation and write results to disk.

    Parameters
    ----------
    model          : trained language model (must expose ``.generate()``).
    tokenizer      : HuggingFace tokenizer matching the model.
    eval_dataset   : HuggingFace ``Dataset`` with an ``"input_ids"`` column.
                     Pass ``tokenized_dataset.get("test")`` from train.py.
    strategy       : ``GraphTextDictStrategy`` — used only for ``decode_to_adjacency``.
    output_dir     : directory where ``filename`` will be written.
    max_new_tokens : budget for generated tokens beyond the prefix.
    num_samples    : how many eval graphs to generate from.
    prefix_frac    : fraction of each sequence to use as the conditioning prefix.
                     0.4 means "show the model the first 40 % of the graph".
    temperature    : sampling temperature passed to ``model.generate()``.
    batch_size     : number of graphs to generate in parallel.
    labeled        : whether graphs have node/edge type labels.
    filename       : name of the output JSON file inside ``output_dir``.
    """
    if eval_dataset is None or len(eval_dataset) == 0:
        logger.warning("run_graph_generation_eval: eval_dataset is empty — skipping.")
        return

    device = next(model.parameters()).device
    model.eval()

    # Clamp num_samples to available eval examples
    num_samples = min(num_samples, len(eval_dataset))

    logger.info(
        f"\n{'='*60}\n"
        f"Graph generation evaluation\n"
        f"  eval examples : {len(eval_dataset)}\n"
        f"  num_samples   : {num_samples}\n"
        f"  prefix_frac   : {prefix_frac:.0%}\n"
        f"  max_new_tokens: {max_new_tokens}\n"
        f"  temperature   : {temperature}\n"
        f"  batch_size    : {batch_size}\n"
        f"{'='*60}"
    )

    # ------------------------------------------------------------------ #
    # Build prefixes for the first ``num_samples`` eval examples
    # ------------------------------------------------------------------ #
    prefixes: list[tuple[int, list[int], str]] = []   # (original_index, prefix_ids, prefix_text)

    for idx in range(num_samples):
        input_ids   = eval_dataset[idx]["input_ids"]
        prefix_ids, prefix_text = _make_prefix(
            input_ids, tokenizer, prefix_frac, labeled
        )
        prefixes.append((idx, prefix_ids, prefix_text))

    # ------------------------------------------------------------------ #
    # Print a sample input so the user can sanity-check the format
    # ------------------------------------------------------------------ #
    sample_idx, sample_prefix_ids, sample_prefix_text = prefixes[0]
    sample_full_text = tokenizer.decode(
        eval_dataset[sample_idx]["input_ids"], skip_special_tokens=True
    )
    num_nodes_in_sample = sample_full_text.count(":") - sample_full_text.count('"t":')

    print(
        "\n"
        + "━" * 60 + "\n"
        + "  SAMPLE EVAL INPUT (graph #0)\n"
        + "━" * 60 + "\n"
        + f"  Full sequence ({len(eval_dataset[sample_idx]['input_ids'])} tokens, "
        + f"~{num_nodes_in_sample} nodes):\n"
        + textwrap.fill(
            sample_full_text[:300] + ("…" if len(sample_full_text) > 300 else ""),
            width=56,
            initial_indent="    ",
            subsequent_indent="    ",
        )
        + "\n\n"
        + f"  Prefix fed to model ({len(sample_prefix_ids)} tokens):\n"
        + textwrap.fill(
            sample_prefix_text[:200] + ("…" if len(sample_prefix_text) > 200 else ""),
            width=56,
            initial_indent="    ",
            subsequent_indent="    ",
        )
        + "\n"
        + "━" * 60
        + "\n"
    )

    # ------------------------------------------------------------------ #
    # Generate in batches
    # ------------------------------------------------------------------ #
    t0      = time.time()
    samples : list[dict[str, Any]] = []

    for batch_start in range(0, num_samples, batch_size):
        batch = prefixes[batch_start : batch_start + batch_size]
        batch_orig_indices = [b[0] for b in batch]
        batch_prefix_ids   = [b[1] for b in batch]
        batch_prefix_texts = [b[2] for b in batch]

        logger.info(
            f"Generating batch {batch_start // batch_size + 1}/"
            f"{(num_samples + batch_size - 1) // batch_size} "
            f"(items {batch_start}–{batch_start + len(batch) - 1})"
        )

        completed_texts = _generate_batch(
            model,
            tokenizer,
            batch_prefix_ids,
            max_new_tokens   = max_new_tokens,
            temperature      = temperature,
            device           = device,
            sampler_config   = sampler_config,
            generation_mode  = generation_mode,     # ← pass through
        )

        for orig_idx, prefix_ids, prefix_text, gen_text in zip(
            batch_orig_indices, batch_prefix_ids, batch_prefix_texts, completed_texts
        ):
            parsed = _parse_completion(
                gen_text, labeled, strategy, undirected=undirected
            )

            # Count nodes visible in the prefix (rough: count ': [' or ': {')
            sep = ': {"t":' if labeled else ': ['
            num_nodes_prefix = prefix_text.count(sep)

            record: dict[str, Any] = {
                "index"            : orig_idx,
                "prefix_text"      : prefix_text,
                "generated_text"   : gen_text,
                "graph_text"       : parsed["graph_text"],
                "valid"            : parsed["valid"],
                "validity_error"   : parsed["validity_error"],
                "num_nodes_prefix" : num_nodes_prefix,
                "num_nodes_output" : parsed["num_nodes"],
                "adjacency"        : parsed["adjacency"],
                "node_types"       : parsed["node_types"],
                "edge_types"       : parsed["edge_types"],
            }
            samples.append(record)

    elapsed = time.time() - t0

    # ------------------------------------------------------------------ #
    # Print the generation result for the first sample
    # ------------------------------------------------------------------ #
    if samples:
        s0 = samples[0]
        print(
            "\n"
            + "━" * 60 + "\n"
            + "  SAMPLE GENERATION OUTPUT (graph #0)\n"
            + "━" * 60 + "\n"
            + f"  Valid parse : {s0['valid']}\n"
            + f"  Nodes in prefix : {s0['num_nodes_prefix']}\n"
            + f"  Nodes after gen : {s0['num_nodes_output']}\n"
            + "\n"
            + "  Generated text:\n"
            + textwrap.fill(
                s0["generated_text"][:400]
                + ("…" if len(s0["generated_text"]) > 400 else ""),
                width=56,
                initial_indent="    ",
                subsequent_indent="    ",
            )
            + "\n"
            + "━" * 60
            + "\n"
        )

    # ------------------------------------------------------------------ #
    # Compute summary statistics
    # ------------------------------------------------------------------ #
    num_valid          = sum(1 for s in samples if s["valid"])
    validity_rate      = num_valid / len(samples) if samples else 0.0
    avg_nodes_prefix   = (
        sum(s["num_nodes_prefix"] for s in samples) / len(samples) if samples else 0.0
    )
    avg_nodes_output   = (
        sum(s["num_nodes_output"] for s in samples) / len(samples) if samples else 0.0
    )

    summary = {
        "num_samples"       : len(samples),
        "num_valid"         : num_valid,
        "validity_rate"     : round(validity_rate, 4),
        "avg_nodes_prefix"  : round(avg_nodes_prefix, 2),
        "avg_nodes_output"  : round(avg_nodes_output, 2),
        "elapsed_seconds"   : round(elapsed, 2),
        "seconds_per_sample": round(elapsed / max(len(samples), 1), 3),
    }

    print(
        "\n"
        + "━" * 60 + "\n"
        + "  GENERATION EVAL SUMMARY\n"
        + "━" * 60 + "\n"
        + f"  Samples generated : {summary['num_samples']}\n"
        + f"  Valid graphs      : {summary['num_valid']} "
        + f"({summary['validity_rate']:.1%})\n"
        + f"  Avg nodes (prefix): {summary['avg_nodes_prefix']:.1f}\n"
        + f"  Avg nodes (output): {summary['avg_nodes_output']:.1f}\n"
        + f"  Total time        : {summary['elapsed_seconds']:.1f}s "
        + f"({summary['seconds_per_sample']:.2f}s/sample)\n"
        + "━" * 60
        + "\n"
    )

    # ------------------------------------------------------------------ #
    # Write JSON output
    # ------------------------------------------------------------------ #
    output: dict[str, Any] = {
        "config": {
            "num_samples"   : num_samples,
            "prefix_frac"   : prefix_frac,
            "max_new_tokens": max_new_tokens,
            "temperature"   : temperature,
            "batch_size"    : batch_size,
            "labeled"       : labeled,
        },
        "summary": summary,
        "samples": samples,
    }

    out_path = Path(output_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    logger.info(f"Eval results written to {out_path}")
    print(f"  Results saved → {out_path}\n")