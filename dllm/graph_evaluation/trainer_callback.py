"""Graph generation evaluation callback for Trainer validation.

Run:
    python -u /home/scur0503/dllm/examples/a2d/mdlm/graph_pt.py \
        --eval_strategy steps \
        --graph_eval_num_generated_graphs 64
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import transformers

import dllm
from dllm.utils.graph_token_strategies import graph_tokens_to_text

from .evaluator import Evaluator


class GraphEvaluatorCallback(transformers.TrainerCallback):
    """Generate graph samples during validation and log graph-level metrics."""

    def __init__(
        self,
        *,
        trainer: transformers.Trainer,
        graph_tokenizer,
        train_data: Sequence[Any],
        token_strategy: str,
        lm_strategy=None,
        graph_token_offset: int = 0,
        graph_vocab_size: int = 0,
        generation_batch_size: int = 16,
        num_generated_graphs: int = 64,
        steps: int = 128,
        max_new_tokens: int = 1024,
        block_size: int = 32,
        temperature: float = 0.0,
        strict_decode: bool = True,
        metric_prefix: str = "eval_graph",
    ) -> None:
        super().__init__()
        self.trainer = trainer
        self.graph_tokenizer = graph_tokenizer
        self.train_data = train_data
        self.token_strategy = token_strategy
        self.lm_strategy = lm_strategy
        self.graph_token_offset = graph_token_offset
        self.graph_vocab_size = graph_vocab_size
        self.generation_batch_size = generation_batch_size
        self.num_generated_graphs = num_generated_graphs
        self.strict_decode = strict_decode
        self.metric_prefix = metric_prefix

        if self.generation_batch_size <= 0:
            raise ValueError("generation_batch_size must be positive.")
        if self.token_strategy != "original" and self.lm_strategy is None:
            raise ValueError(
                "lm_strategy is required unless token_strategy='original'."
            )
        if self.token_strategy == "original" and self.graph_vocab_size <= 0:
            raise ValueError("graph_vocab_size must be positive for original strategy.")

        self.sampler_config = dllm.core.samplers.MDLMSamplerConfig(
            steps=steps,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            temperature=temperature,
            return_dict=True,
            right_shift_logits=getattr(trainer.args, "right_shift_logits", False),
        )

    def _graph_prompt(self) -> list[int]:
        if self.token_strategy == "original":
            return [self.graph_token_offset + self.graph_tokenizer.sos]
        return self.lm_strategy.encode([self.graph_tokenizer.sos])

    def _lm_tokens_to_graph_tokens(
        self, sequences: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        rows = []
        decode_errors = 0

        for sequence in sequences.detach().cpu().tolist():
            if self.token_strategy == "original":
                graph_ids = [
                    token_id - self.graph_token_offset
                    for token_id in sequence
                    if self.graph_token_offset
                    <= token_id
                    < self.graph_token_offset + self.graph_vocab_size
                ]
            else:
                try:
                    graph_ids = self.lm_strategy.decode(
                        sequence, strict=self.strict_decode
                    )
                except ValueError:
                    decode_errors += 1
                    try:
                        graph_ids = self.lm_strategy.decode(sequence, strict=False)
                        graph_text = graph_tokens_to_text(
                            graph_ids, self.graph_tokenizer
                        )
                    except Exception:
                        graph_text = self.trainer.processing_class.decode(
                            sequence, skip_special_tokens=False
                        )
                    print(f"Unparseable graph number {decode_errors}: {graph_text}")
                    continue

            if not graph_ids or graph_ids[0] != self.graph_tokenizer.sos:
                graph_ids = [self.graph_tokenizer.sos] + graph_ids
            rows.append(torch.tensor(graph_ids, dtype=torch.long))

        width = max((row.numel() for row in rows), default=1)
        batch = torch.full(
            (len(rows), width),
            fill_value=self.graph_tokenizer.pad,
            dtype=torch.long,
        )
        for i, row in enumerate(rows):
            batch[i, : row.numel()] = row
        return batch, decode_errors

    def _json_ready_number(self, value: Any) -> float | int:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Only scalar tensors can be logged as graph metrics.")
            return value.detach().cpu().item()
        if isinstance(value, (int, float)):
            return value
        return float(value)

    @torch.no_grad()
    def _evaluate_graph_generation(self) -> dict[str, float | int]:
        model = self.trainer.accelerator.unwrap_model(self.trainer.model)
        was_training = model.training
        model.eval()

        try:
            sampler = dllm.core.samplers.MDLMSampler(
                model=model,
                tokenizer=self.trainer.processing_class,
            )
            prompt = self._graph_prompt()
            generated = []

            while len(generated) < self.num_generated_graphs:
                batch_size = min(
                    self.generation_batch_size,
                    self.num_generated_graphs - len(generated),
                )
                outputs = sampler.sample(
                    [prompt.copy() for _ in range(batch_size)],
                    self.sampler_config,
                    return_dict=True,
                )
                generated.extend(outputs.sequences.detach().cpu())

            generated_lm_tokens = torch.stack(generated)
            generated_graph_tokens, decode_errors = self._lm_tokens_to_graph_tokens(
                generated_lm_tokens
            )
            metrics = Evaluator(tokenizer=self.graph_tokenizer)(
                tokenized_graphs=generated_graph_tokens,
                train_data=self.train_data,
                total_gen_graphs=generated_lm_tokens.shape[0],
            )
            metrics["lm_decode_errors"] = self._json_ready_number(decode_errors)
            return {
                f"{self.metric_prefix}_{key}": self._json_ready_number(value)
                for key, value in metrics.items()
            }
        finally:
            if was_training:
                model.train()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.num_generated_graphs <= 0:
            return control

        accelerator = self.trainer.accelerator
        logs = None

        if accelerator.is_main_process:
            logs = self._evaluate_graph_generation()
            if logs:
                self.trainer.log(logs)
                formatted = " ".join(
                    f"{key}={value:.6f}" for key, value in logs.items()
                )
                print(f"[step {state.global_step} epoch {state.epoch}] {formatted}")

        accelerator.wait_for_everyone()
        return control
