from __future__ import annotations

"""
Post-training graph generation evaluation utilities.

Supports two graph text strategies:

1. text_dict
   Example:
       Generate a graph ... keys:{0: [1], 1: [0]}

2. edge_list
   Example:
       Generate a graph ... N=5; E=(0,1),(1,2),(3,4)

For evaluation we keep two validity notions separate:

- raw_valid: the model output parses directly as a valid graph.
- repaired_valid: a deterministic postprocessor can recover a valid graph.

Do not report repaired_valid as raw model validity.
"""

import json
import logging
import re
import textwrap
import time
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _strategy_name(strategy: Any) -> str:
    return getattr(strategy, "strategy", getattr(strategy, "name", "text_dict"))


def _expected_num_nodes_from_text(text: str) -> int | None:
    if not text:
        return None

    m = re.search(r"\bN\s*=\s*(\d+)", text)
    if m:
        return int(m.group(1))

    m = re.search(r"exactly\s+(\d+)\s+nodes", text)
    if m:
        return int(m.group(1))

    m = re.search(r"indices\s+0\s+to\s+(\d+)", text)
    if m:
        return int(m.group(1)) + 1

    return None


def _is_valid_adjacency(
    adjacency: dict[int, list[int]],
    *,
    undirected: bool = True,
    require_contiguous_nodes: bool = True,
    allow_self_loops: bool = False,
) -> tuple[bool, str]:
    if not adjacency:
        return False, "empty adjacency"

    nodes = set(adjacency.keys())

    if not all(isinstance(n, int) for n in nodes):
        return False, "non-integer node id"

    if require_contiguous_nodes:
        expected = set(range(max(nodes) + 1))
        if nodes != expected:
            missing = sorted(expected - nodes)[:10]
            extra = sorted(nodes - expected)[:10]
            return False, f"node ids are not contiguous from 0; missing={missing}; extra={extra}"

    for u, nbrs in adjacency.items():
        if not isinstance(nbrs, list):
            return False, f"neighbors for node {u} are not a list"

        if not all(isinstance(v, int) for v in nbrs):
            return False, f"neighbors for node {u} contain non-integer ids"

        if len(nbrs) != len(set(nbrs)):
            return False, f"duplicate neighbors for node {u}"

        for v in nbrs:
            if v not in nodes:
                return False, f"dangling edge {u}->{v}; node {v} is not declared"
            if not allow_self_loops and v == u:
                return False, f"self-loop at node {u}"

    if undirected:
        for u, nbrs in adjacency.items():
            for v in nbrs:
                if u not in adjacency.get(v, []):
                    return False, f"asymmetric edge {u}->{v}; missing {v}->{u}"

    return True, "ok"


def _extract_graph_dict_text(text: str) -> str:
    if not text:
        return ""

    start = text.find("{")
    if start == -1:
        return ""

    depth = 0
    in_string = False
    prev = ""

    for i, ch in enumerate(text[start:], start=start):
        if ch == '"' and prev != "\\":
            in_string = not in_string

        if not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1].strip()

        prev = ch

    return text[start:].strip()


def _graph_dict_text_is_complete(text: str) -> bool:
    graph_text = _extract_graph_dict_text(text)
    return graph_text.startswith("{") and graph_text.endswith("}")


def _repair_adjacency_from_dict_text(
    text: str,
    *,
    undirected: bool = True,
) -> tuple[dict[int, list[int]], str]:
    n = _expected_num_nodes_from_text(text)
    if n is None:
        return {}, "missing node count"

    body = _extract_graph_dict_text(text)
    if not body:
        body = text

    pairs = re.findall(r"(\d+)\s*:\s*\[([^\]]*)\]", body)
    if not pairs:
        return {}, "no parseable node entries"

    adj: dict[int, set[int]] = {i: set() for i in range(n)}

    for u_s, nbr_s in pairs:
        u = int(u_s)
        if not (0 <= u < n):
            continue

        for v_s in re.findall(r"\d+", nbr_s):
            v = int(v_s)
            if not (0 <= v < n):
                continue
            if v == u:
                continue
            adj[u].add(v)

    if undirected:
        for u in list(adj):
            for v in list(adj[u]):
                adj[v].add(u)

    repaired = {u: sorted(vs) for u, vs in sorted(adj.items())}
    return repaired, "repaired"


def _extract_edge_list_text(text: str) -> str:
    if not text:
        return ""

    m = re.search(r"\bN\s*=\s*\d+\s*;\s*E\s*=", text)
    if m:
        return text[m.start() :].strip()

    m = re.search(r"\bE\s*=", text)
    if m:
        n = _expected_num_nodes_from_text(text)
        if n is not None:
            return f"N={n}; " + text[m.start() :].strip()

    return ""


def _edge_list_to_adjacency(
    text: str,
    *,
    undirected: bool = True,
    require_n: bool = True,
) -> tuple[list[int], dict[int, list[int]], dict[int, int], dict[tuple[int, int], int]]:
    n = _expected_num_nodes_from_text(text)
    if n is None:
        if require_n:
            raise ValueError("missing N=<num_nodes>")
        all_ids = [int(x) for x in re.findall(r"\d+", text)]
        n = max(all_ids) + 1 if all_ids else 0

    if n <= 0:
        raise ValueError(f"invalid node count N={n}")

    adj_sets: dict[int, set[int]] = {i: set() for i in range(n)}
    pairs = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)

    for u_s, v_s in pairs:
        u, v = int(u_s), int(v_s)
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge ({u},{v}) outside node range 0..{n - 1}")
        if u == v:
            adj_sets[u].add(v)
            continue
        adj_sets[u].add(v)
        if undirected:
            adj_sets[v].add(u)

    adjacency = {u: sorted(vs) for u, vs in sorted(adj_sets.items())}
    return list(range(n)), adjacency, {}, {}


def _repair_edge_list_from_text(
    text: str,
    *,
    undirected: bool = True,
) -> tuple[dict[int, list[int]], str]:
    n = _expected_num_nodes_from_text(text)
    if n is None:
        return {}, "missing node count"
    if n <= 0:
        return {}, f"invalid node count N={n}"

    adj_sets: dict[int, set[int]] = {i: set() for i in range(n)}
    pairs = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)

    if not pairs:
        return {i: [] for i in range(n)}, "repaired-empty-edge-list"

    for u_s, v_s in pairs:
        u, v = int(u_s), int(v_s)
        if u == v:
            continue
        if not (0 <= u < n and 0 <= v < n):
            continue
        adj_sets[u].add(v)
        if undirected:
            adj_sets[v].add(u)

    repaired = {u: sorted(vs) for u, vs in sorted(adj_sets.items())}
    return repaired, "repaired"


def _make_prefix(
    input_ids: list[int],
    tokenizer,
    prefix_frac: float,
    labeled: bool,
) -> tuple[list[int], str]:
    full_text = tokenizer.decode(input_ids, skip_special_tokens=True)

    edge_match = re.search(r"\bN\s*=\s*\d+\s*;\s*E\s*=", full_text)
    if edge_match:
        clean_prefix = full_text[: edge_match.end()]
    else:
        brace_idx = full_text.find("{")
        if brace_idx != -1:
            clean_prefix = full_text[: brace_idx + 1]
        else:
            target_len = max(1, int(len(input_ids) * prefix_frac))
            clean_prefix = tokenizer.decode(
                input_ids[:target_len],
                skip_special_tokens=True,
            )

    prefix_ids = tokenizer.encode(clean_prefix, add_special_tokens=False)
    return prefix_ids, clean_prefix


def _generate_batch(
    model,
    tokenizer,
    prefix_ids_batch: list[list[int]],
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
    sampler_config=None,
    generation_mode: str = "sample",
) -> list[str]:
    from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig

    if generation_mode not in ("sample", "infill"):
        raise ValueError(
            f"generation_mode must be 'sample' or 'infill', got {generation_mode!r}"
        )

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

    if generation_mode == "infill":
        inputs = []
        for ids in prefix_ids_batch:
            clean_ids = [tok for tok in ids if tok != mask_id]
            inputs.append(clean_ids + [mask_id] * max_new_tokens)

        output_ids = sampler.infill(inputs=inputs, config=sampler_config)

        completed_texts = []
        for i, ids in enumerate(prefix_ids_batch):
            prefix_len = len([tok for tok in ids if tok != mask_id])
            full_ids = output_ids[i, : prefix_len + max_new_tokens].tolist()
            completed_texts.append(tokenizer.decode(full_ids, skip_special_tokens=True))

    else:
        inputs = []
        for ids in prefix_ids_batch:
            clean_ids = [tok for tok in ids if tok != mask_id]
            inputs.append(clean_ids)

        output_ids = sampler.sample(inputs=inputs, config=sampler_config)

        completed_texts = []
        for i, ids in enumerate(prefix_ids_batch):
            prefix_len = len([tok for tok in ids if tok != mask_id])
            full_ids = output_ids[i, : prefix_len + max_new_tokens].tolist()
            completed_texts.append(tokenizer.decode(full_ids, skip_special_tokens=True))

    return completed_texts


def _parse_completion(
    text: str,
    labeled: bool,
    strategy,
    undirected: bool = True,
) -> dict[str, Any]:
    strategy_name = _strategy_name(strategy)

    if strategy_name == "edge_list":
        graph_text = _extract_edge_list_text(text)
        raw_valid = False
        raw_error = ""
        adjacency: dict[int, list[int]] = {}
        node_types: dict[int, int] = {}
        edge_types: dict[tuple[int, int], int] = {}

        try:
            node_order, adjacency, node_types, edge_types = _edge_list_to_adjacency(
                graph_text,
                undirected=undirected,
            )
            raw_valid, raw_error = _is_valid_adjacency(
                adjacency,
                undirected=undirected,
                require_contiguous_nodes=True,
                allow_self_loops=False,
            )
        except Exception as exc:
            raw_error = str(exc)
            adjacency, node_types, edge_types = {}, {}, {}

        repaired_adjacency, repair_error = _repair_edge_list_from_text(
            text,
            undirected=undirected,
        )
        repaired_valid = False
        if repaired_adjacency:
            repaired_valid, repair_error = _is_valid_adjacency(
                repaired_adjacency,
                undirected=undirected,
                require_contiguous_nodes=True,
                allow_self_loops=False,
            )

        return {
            "valid": raw_valid,
            "raw_valid": raw_valid,
            "repaired_valid": repaired_valid,
            "validity_error": raw_error if not raw_valid else "ok",
            "repair_error": repair_error,
            "graph_text": graph_text,
            "num_nodes": len(adjacency),
            "num_nodes_repaired": len(repaired_adjacency),
            "adjacency": {str(k): v for k, v in adjacency.items()},
            "repaired_adjacency": {str(k): v for k, v in repaired_adjacency.items()},
            "node_types": {str(k): v for k, v in node_types.items()},
            "edge_types": {f"{src},{dst}": t for (src, dst), t in edge_types.items()},
        }

    from dllm.utils.graph_dict_strategy import dict_string_to_adjacency

    graph_text = _extract_graph_dict_text(text)
    raw_valid = False
    raw_error = ""
    adjacency: dict[int, list[int]] = {}
    node_types: dict[int, int] = {}
    edge_types: dict[tuple[int, int], int] = {}

    try:
        if not _graph_dict_text_is_complete(graph_text):
            raise ValueError("incomplete or unbalanced graph dict")

        node_order, adjacency, node_types, edge_types = dict_string_to_adjacency(
            graph_text,
            labeled=labeled,
        )
        raw_valid, raw_error = _is_valid_adjacency(
            adjacency,
            undirected=undirected,
            require_contiguous_nodes=True,
            allow_self_loops=False,
        )
    except Exception as exc:
        raw_error = str(exc)
        adjacency, node_types, edge_types = {}, {}, {}

    repaired_adjacency, repair_error = _repair_adjacency_from_dict_text(
        text,
        undirected=undirected,
    )
    repaired_valid = False
    if repaired_adjacency:
        repaired_valid, repair_error = _is_valid_adjacency(
            repaired_adjacency,
            undirected=undirected,
            require_contiguous_nodes=True,
            allow_self_loops=False,
        )

    return {
        "valid": raw_valid,
        "raw_valid": raw_valid,
        "repaired_valid": repaired_valid,
        "validity_error": raw_error if not raw_valid else "ok",
        "repair_error": repair_error,
        "graph_text": graph_text,
        "num_nodes": len(adjacency),
        "num_nodes_repaired": len(repaired_adjacency),
        "adjacency": {str(k): v for k, v in adjacency.items()},
        "repaired_adjacency": {str(k): v for k, v in repaired_adjacency.items()},
        "node_types": {str(k): v for k, v in node_types.items()},
        "edge_types": {f"{src},{dst}": t for (src, dst), t in edge_types.items()},
    }


def run_graph_generation_eval(
    model,
    tokenizer,
    eval_dataset,
    strategy,
    output_dir: str,
    *,
    max_new_tokens: int = 256,
    num_samples: int = 16,
    prefix_frac: float = 0.4,
    temperature: float = 1.0,
    batch_size: int = 4,
    labeled: bool = False,
    undirected: bool = True,
    filename: str = "graph_generation_eval.json",
    generation_mode: str = "sample",
    sampler_config=None,
) -> None:
    if eval_dataset is None or len(eval_dataset) == 0:
        logger.warning("run_graph_generation_eval: eval_dataset is empty — skipping.")
        return

    device = next(model.parameters()).device
    model.eval()

    num_samples = min(num_samples, len(eval_dataset))

    logger.info(
        f"\n{'=' * 60}\n"
        f"Graph generation evaluation\n"
        f"  eval examples : {len(eval_dataset)}\n"
        f"  num_samples   : {num_samples}\n"
        f"  prefix_frac   : {prefix_frac:.0%}\n"
        f"  max_new_tokens: {max_new_tokens}\n"
        f"  temperature   : {temperature}\n"
        f"  batch_size    : {batch_size}\n"
        f"  strategy      : {_strategy_name(strategy)}\n"
        f"{'=' * 60}"
    )

    if num_samples <= 0:
        output = {
            "config": {
                "num_samples": 0,
                "prefix_frac": prefix_frac,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "batch_size": batch_size,
                "labeled": labeled,
                "undirected": undirected,
                "strategy": _strategy_name(strategy),
            },
            "summary": {
                "num_samples": 0,
                "num_valid": 0,
                "validity_rate": 0.0,
                "num_repaired_valid": 0,
                "repaired_validity_rate": 0.0,
                "avg_nodes_prefix": 0.0,
                "avg_nodes_output": 0.0,
                "avg_nodes_repaired": 0.0,
                "elapsed_seconds": 0.0,
                "seconds_per_sample": 0.0,
            },
            "samples": [],
        }
        out_path = Path(output_dir) / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(f"  Results saved → {out_path}\n")
        return

    prefixes: list[tuple[int, list[int], str]] = []

    for idx in range(num_samples):
        input_ids = eval_dataset[idx]["input_ids"]
        prefix_ids, prefix_text = _make_prefix(input_ids, tokenizer, prefix_frac, labeled)
        prefixes.append((idx, prefix_ids, prefix_text))

    sample_idx, sample_prefix_ids, sample_prefix_text = prefixes[0]
    sample_full_text = tokenizer.decode(
        eval_dataset[sample_idx]["input_ids"],
        skip_special_tokens=True,
    )

    if _strategy_name(strategy) == "edge_list":
        n = _expected_num_nodes_from_text(sample_full_text)
        num_nodes_in_sample = n if n is not None else 0
    else:
        num_nodes_in_sample = sample_full_text.count(": [")

    print(
        "\n"
        + "━" * 60
        + "\n"
        + "  SAMPLE EVAL INPUT (graph #0)\n"
        + "━" * 60
        + "\n"
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

    t0 = time.time()
    samples: list[dict[str, Any]] = []

    for batch_start in range(0, num_samples, batch_size):
        batch = prefixes[batch_start : batch_start + batch_size]
        batch_orig_indices = [b[0] for b in batch]
        batch_prefix_ids = [b[1] for b in batch]
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
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
            sampler_config=sampler_config,
            generation_mode=generation_mode,
        )

        for orig_idx, prefix_ids, prefix_text, gen_text in zip(
            batch_orig_indices,
            batch_prefix_ids,
            batch_prefix_texts,
            completed_texts,
        ):
            parsed = _parse_completion(
                gen_text,
                labeled,
                strategy,
                undirected=undirected,
            )

            if _strategy_name(strategy) == "edge_list":
                num_nodes_prefix = 0
            else:
                sep = ': {"t":' if labeled else ": ["
                num_nodes_prefix = prefix_text.count(sep)

            record: dict[str, Any] = {
                "index": orig_idx,
                "prefix_text": prefix_text,
                "generated_text": gen_text,
                "valid": parsed["valid"],
                "raw_valid": parsed.get("raw_valid", parsed["valid"]),
                "repaired_valid": parsed.get("repaired_valid", False),
                "validity_error": parsed.get("validity_error", ""),
                "repair_error": parsed.get("repair_error", ""),
                "graph_text": parsed.get("graph_text", ""),
                "num_nodes_prefix": num_nodes_prefix,
                "num_nodes_output": parsed["num_nodes"],
                "num_nodes_repaired": parsed.get("num_nodes_repaired", 0),
                "adjacency": parsed["adjacency"],
                "repaired_adjacency": parsed.get("repaired_adjacency", {}),
                "node_types": parsed["node_types"],
                "edge_types": parsed["edge_types"],
            }
            samples.append(record)

    elapsed = time.time() - t0

    if samples:
        s0 = samples[0]
        print(
            "\n"
            + "━" * 60
            + "\n"
            + "  SAMPLE GENERATION OUTPUT (graph #0)\n"
            + "━" * 60
            + "\n"
            + f"  Valid parse : {s0['valid']}\n"
            + f"  Repaired valid : {s0.get('repaired_valid', False)}\n"
            + f"  Nodes in prefix : {s0['num_nodes_prefix']}\n"
            + f"  Nodes after gen : {s0['num_nodes_output']}\n"
            + f"  Nodes repaired  : {s0.get('num_nodes_repaired', 0)}\n"
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

    num_valid = sum(1 for s in samples if s["valid"])
    num_repaired_valid = sum(1 for s in samples if s.get("repaired_valid", False))

    validity_rate = num_valid / len(samples) if samples else 0.0
    repaired_validity_rate = num_repaired_valid / len(samples) if samples else 0.0

    avg_nodes_prefix = (
        sum(s["num_nodes_prefix"] for s in samples) / len(samples) if samples else 0.0
    )
    avg_nodes_output = (
        sum(s["num_nodes_output"] for s in samples) / len(samples) if samples else 0.0
    )
    avg_nodes_repaired = (
        sum(s.get("num_nodes_repaired", 0) for s in samples) / len(samples)
        if samples
        else 0.0
    )

    summary = {
        "num_samples": len(samples),
        "num_valid": num_valid,
        "validity_rate": round(validity_rate, 4),
        "num_repaired_valid": num_repaired_valid,
        "repaired_validity_rate": round(repaired_validity_rate, 4),
        "avg_nodes_prefix": round(avg_nodes_prefix, 2),
        "avg_nodes_output": round(avg_nodes_output, 2),
        "avg_nodes_repaired": round(avg_nodes_repaired, 2),
        "elapsed_seconds": round(elapsed, 2),
        "seconds_per_sample": round(elapsed / max(len(samples), 1), 3),
    }

    print(
        "\n"
        + "━" * 60
        + "\n"
        + "  GENERATION EVAL SUMMARY\n"
        + "━" * 60
        + "\n"
        + f"  Samples generated : {summary['num_samples']}\n"
        + f"  Valid graphs      : {summary['num_valid']} "
        + f"({summary['validity_rate']:.1%})\n"
        + f"  Repaired valid    : {summary['num_repaired_valid']} "
        + f"({summary['repaired_validity_rate']:.1%})\n"
        + f"  Avg nodes (prefix): {summary['avg_nodes_prefix']:.1f}\n"
        + f"  Avg nodes (output): {summary['avg_nodes_output']:.1f}\n"
        + f"  Avg nodes (repair): {summary['avg_nodes_repaired']:.1f}\n"
        + f"  Total time        : {summary['elapsed_seconds']:.1f}s "
        + f"({summary['seconds_per_sample']:.2f}s/sample)\n"
        + "━" * 60
        + "\n"
    )

    output: dict[str, Any] = {
        "config": {
            "num_samples": num_samples,
            "prefix_frac": prefix_frac,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "batch_size": batch_size,
            "labeled": labeled,
            "undirected": undirected,
            "strategy": _strategy_name(strategy),
        },
        "summary": summary,
        "samples": samples,
    }

    out_path = Path(output_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    logger.info(f"Eval results written to {out_path}")
    print(f"  Results saved → {out_path}\n")
