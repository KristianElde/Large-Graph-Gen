from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


MASK_TOKEN_ID: int = 126336   # Reserved [MASK] id used by LLaDA


def forward_process(
    input_ids: torch.Tensor,
    eps: float = 1e-3,
    mask_token_id: int = MASK_TOKEN_ID,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Adds random masking noise to a batch of token sequences.

    The masking ratio t is sampled uniformly from [eps, 1] for each sequence
    independently.  This mirrors the continuous-time diffusion schedule used
    during LLaDA pre-training.

    Args:
        input_ids : (batch, seq_len)  integer token ids
        eps       : lower bound on masking probability (avoids p_mask = 0)

    Returns:
        noisy_batch     : (batch, seq_len) ids with some positions → MASK_TOKEN_ID
        masked_indices  : (batch, seq_len) bool — True where a token was masked
        p_mask          : (batch, seq_len) the per-token masking probability
    """
    b, l = input_ids.shape
    # Sample a masking ratio t ~ Uniform(eps, 1) per sequence in the batch
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps                            # (b,)
    p_mask = p_mask[:, None].expand(b, l)                   # (b, l)

    # Decide independently which positions actually get masked
    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask

    # Replace masked positions with the special [MASK] token
    noisy_batch = torch.where(masked_indices, mask_token_id, input_ids)

    return noisy_batch, masked_indices, p_mask


class LLaDAModel:
    """

    Parameters
    ----------
    hf_model_path : str
        HuggingFace hub repo-id or local directory containing the checkpoint.
        e.g. ``"GSAI-ML/LLaDA-8B-Instruct"``
    tokenizer : Any | None
        Tokenizer object for input/output conversion. If omitted, the Hugging Face
        tokenizer from ``hf_model_path`` is used. Custom tokenizers should expose
        ``tokenize(...)`` and ``decode(...)`` to match the shared graph API.
    extra_special_tokens : list[str] | None
        Additional tokens to add to the vocabulary before loading.
        Useful when your SFT data uses domain-specific control tokens.
        The embedding matrix is resized automatically.
    mask_token_id : int | None
        Mask token id used by diffusion corruption. If omitted, the model tries
        to infer it from the tokenizer and falls back to ``MASK_TOKEN_ID``.
    device : str
        ``"cuda"``, ``"cpu"``, or ``"auto"`` (uses accelerate device mapping).
    torch_dtype : torch.dtype
        Precision of the loaded weights.  ``torch.bfloat16`` recommended for
        modern GPUs; ``torch.float32`` for CPU / debugging.
    """

    def __init__(
        self,
        hf_model_path: str,
        tokenizer: Any | None = None,
        extra_special_tokens: list[str] | None = None,
        mask_token_id: int | None = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.hf_model_path = hf_model_path
        self.device = device
        self.torch_dtype = torch_dtype

        # ------------------------------------------------------------------ #
        # Step 1 — Load the tokenizer                                         #
        # ------------------------------------------------------------------ #
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                hf_model_path,
                trust_remote_code=True,
            )
        else:
            self.tokenizer = tokenizer

        # ------------------------------------------------------------------ #
        # Step 2 — Register extra special tokens (optional)                   #
        # ------------------------------------------------------------------ #
        if extra_special_tokens and tokenizer is not None:
            raise ValueError(
                "extra_special_tokens is only supported when using the Hugging Face tokenizer."
            )
        if extra_special_tokens:
            num_added = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": extra_special_tokens}
            )
            print(f"[LLaDA] Added {num_added} extra special token(s): "
                  f"{extra_special_tokens}")

        # ------------------------------------------------------------------ #
        # Step 3 — Load the model weights                                     #
        # ------------------------------------------------------------------ #
        load_kwargs: dict = dict(
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        if device == "auto":
            load_kwargs["device_map"] = "auto"

        self.model = AutoModel.from_pretrained(
            hf_model_path,
            **load_kwargs,
        )

        tokenizer_vocab_size = self._get_tokenizer_vocab_size()
        if extra_special_tokens:
            self.model.resize_token_embeddings(len(self.tokenizer))
            print(f"[LLaDA] Embedding matrix resized to "
                  f"{len(self.tokenizer)} tokens.")
        elif tokenizer_vocab_size is not None:
            current_vocab_size = self.model.get_input_embeddings().num_embeddings
            if tokenizer_vocab_size != current_vocab_size:
                self.model.resize_token_embeddings(tokenizer_vocab_size)
                print(
                    f"[LLaDA] Embedding matrix resized to {tokenizer_vocab_size} tokens "
                    f"to match the supplied tokenizer."
                )

        # Move to device when not using device_map="auto"
        if device != "auto":
            self.model = self.model.to(device)

        self.model.eval()

        inferred_mask_token_id = self._get_mask_token_id_from_tokenizer()
        if mask_token_id is not None:
            self.mask_token_id = int(mask_token_id)
        elif inferred_mask_token_id is not None:
            self.mask_token_id = inferred_mask_token_id
        else:
            self.mask_token_id = MASK_TOKEN_ID

        if tokenizer_vocab_size is not None and self.mask_token_id >= tokenizer_vocab_size:
            raise ValueError(
                f"mask_token_id={self.mask_token_id} is out of tokenizer vocabulary range "
                f"[0, {tokenizer_vocab_size - 1}]."
            )

    @torch.inference_mode()
    def generate(
        self,
        prompt: Any,
        max_new_tokens: int = 128,
        num_steps: int = 10,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Any:
        """
        Generate a response to ``prompt`` using iterative masked-diffusion
        decoding — the inference algorithm described in the LLaDA paper.

        Algorithm overview
        ------------------
        1. Encode the prompt and build an answer block filled with [MASK].
        2. For T steps (coarse → fine denoising):
           a. Run a single forward pass of the bidirectional model.
              Because attention is *not* causal, every [MASK] position can
              attend to every prompt token AND every other answer token.
           b. For each still-masked position, compute the predicted token
              and its confidence (softmax probability).
           c. Un-mask the ``k`` highest-confidence positions, where k is
              chosen so that after T steps every position has been committed.
        3. Decode the committed answer tokens back to a string.

        Args:
            prompt        : input object consumed by ``tokenizer.tokenize``.
                            With the default tokenizer this is a ``str``.
            max_new_tokens: maximum answer length in tokens
            num_steps     : number of denoising steps (more = better quality,
                            slower; 10–50 is a practical range)
            temperature   : softmax temperature (0 = greedy argmax)
            top_p         : nucleus sampling threshold (ignored when temp=0)

        Returns:
            Decoded answer object from ``tokenizer.decode``.
        """
        device = next(self.model.parameters()).device

        # ------------------------------------------------------------------ #
        # 5a — Encode the prompt                                               #
        # ------------------------------------------------------------------ #
        prompt_ids = self._tokenize_prompt(
            prompt, device=device)  # (1, prompt_len)

        prompt_len = prompt_ids.shape[1]
        eos_id = self._get_eos_token_id()

        # ------------------------------------------------------------------ #
        # 5b — Build the initial noisy answer (fully masked)                  #
        # ------------------------------------------------------------------ #
        # During inference we do NOT know the answer length in advance, so we
        # allocate ``max_new_tokens`` mask tokens and let the model fill them.
        # (Shorter answers will produce EOS before the buffer is exhausted.)
        answer_ids = torch.full(
            (1, max_new_tokens),
            fill_value=self.mask_token_id,
            dtype=torch.long,
            device=device,
        )

        # Concatenate prompt + masked answer into one sequence
        # Shape: (1, prompt_len + max_new_tokens)
        input_ids = torch.cat([prompt_ids, answer_ids], dim=1)

        # Track which positions are still masked (only within the answer)
        # Shape: (max_new_tokens,)  boolean
        answer_mask = torch.ones(
            max_new_tokens, dtype=torch.bool, device=device)

        # How many tokens do we commit per step?
        # We spread the un-masking evenly so that by step T all are committed.
        tokens_per_step = max(1, max_new_tokens // num_steps)

        # ------------------------------------------------------------------ #
        # 5c — Iterative denoising loop                                        #
        # ------------------------------------------------------------------ #
        for step in range(num_steps):
            # ---- Forward pass (bidirectional, no causal mask) ------------- #
            # ``logits`` shape: (1, seq_len, vocab_size)
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits                         # (1, seq_len, V)

            # Isolate logits for the answer portion only
            # (max_new_tokens, V)
            answer_logits = logits[0, prompt_len:, :]

            # ---- Sample / select tokens ----------------------------------- #
            if temperature == 0.0:
                # Greedy: pick the argmax at every still-masked position
                predicted_ids = answer_logits.argmax(
                    dim=-1)   # (max_new_tokens,)
                # Confidence = softmax probability of the predicted token
                probs = F.softmax(answer_logits, dim=-1)
                confidence = probs.gather(
                    1, predicted_ids.unsqueeze(-1)
                ).squeeze(-1)                               # (max_new_tokens,)
            else:
                # Stochastic: temperature-scaled softmax + optional nucleus
                scaled_logits = answer_logits / temperature
                probs = F.softmax(scaled_logits, dim=-1)
                if top_p < 1.0:
                    probs = _top_p_filter(probs, top_p)
                predicted_ids = torch.multinomial(
                    probs, num_samples=1).squeeze(-1)
                confidence = probs.gather(
                    1, predicted_ids.unsqueeze(-1)
                ).squeeze(-1)

            # ---- Decide which positions to commit this step --------------- #
            # We only consider positions that are STILL masked
            still_masked_positions = answer_mask.nonzero(as_tuple=True)[0]
            num_still_masked = still_masked_positions.numel()

            if num_still_masked == 0:
                break   # Nothing left to unmask

            # On the final step commit everything; otherwise commit top-k
            if step == num_steps - 1:
                commit_positions = still_masked_positions
            else:
                # How many to commit this step?
                n_commit = min(tokens_per_step, num_still_masked)
                # Pick the n_commit positions with highest confidence
                masked_conf = confidence[still_masked_positions]
                topk_local = torch.topk(masked_conf, k=n_commit).indices
                commit_positions = still_masked_positions[topk_local]

            # Write the committed tokens into input_ids
            input_ids[0, prompt_len + commit_positions] = (
                predicted_ids[commit_positions]
            )
            # Mark those positions as no longer masked
            answer_mask[commit_positions] = False

            # Early-exit: if an EOS was committed, stop denoising
            if eos_id is not None:
                committed_answer = input_ids[0, prompt_len:]
                if (committed_answer == eos_id).any():
                    break

        # ------------------------------------------------------------------ #
        # 5d — Decode the answer                                               #
        # ------------------------------------------------------------------ #
        answer_token_ids = input_ids[0, prompt_len:].tolist()

        # Truncate at EOS if present
        if eos_id is not None and eos_id in answer_token_ids:
            answer_token_ids = answer_token_ids[: answer_token_ids.index(
                eos_id)]

        # Remove any residual [MASK] tokens (shouldn't happen after full
        # denoising, but is a safe guard)
        answer_token_ids = [
            t for t in answer_token_ids if t != self.mask_token_id
        ]

        return self._decode_answer_tokens(answer_token_ids)

    def prepare_for_lora(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        bias: str = "none",
    ) -> None:
        """
        Wrap the underlying model with PEFT LoRA adapters, ready for SFT.

        After calling this method:
          • Only LoRA parameters require gradients — the base model is frozen.
          • ``self.model`` is replaced with the PEFT-wrapped version.
          • You can access the trainable parameters via
            ``self.model.parameters()`` as usual in your training loop.
          • Call ``self.model.save_pretrained(path)`` to save the LoRA adapter
            weights only (much smaller than the full model).

        The SFT loss you should use in your training loop is computed following
        the paper's recipe (see ``compute_sft_loss`` below).

        Args:
            r                   : LoRA rank — higher rank = more capacity but
                                  more parameters.  16 is a good default.
            lora_alpha          : LoRA scaling factor.  ``alpha/r`` controls the
                                  effective learning rate of the adapter.
            lora_dropout        : Dropout applied inside the LoRA branch.
            lora_target_modules : List of module names to attach LoRA to.
                                  Defaults to ``["q_proj", "v_proj"]`` — the
                                  query and value projections in each attention
                                  layer.  If your checkpoint uses different
                                  names (e.g. ``"query_key_value"`` for Falcon),
                                  pass them explicitly.
            bias                : Whether to train bias terms.
                                  ``"none"`` (default), ``"all"``,
                                  or ``"lora_only"``.
        """
        if lora_target_modules is None:
            # Standard LLaMA / Mistral projection names.
            # LLaDA is built on LLaMA-3, so these should be correct.
            lora_target_modules = ["q_proj", "v_proj"]

        # ------------------------------------------------------------------ #
        # Step A — Build the LoRA configuration                               #
        # ------------------------------------------------------------------ #
        # TaskType.FEATURE_EXTRACTION is the right task type for an encoder
        # that does token-level prediction (no dedicated "causal LM" head).
        # PEFT still wraps the model correctly for masked-LM style outputs.
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias=bias,
        )

        # ------------------------------------------------------------------ #
        # Step B — Wrap the model                                             #
        # ------------------------------------------------------------------ #
        # ``get_peft_model`` freezes all base-model parameters and inserts
        # trainable LoRA matrices (A and B) in parallel with the target linear
        # layers.  Only these matrices will appear in the optimizer.
        self.model = get_peft_model(self.model, lora_config)

        # ------------------------------------------------------------------ #
        # Step C — Print a parameter summary                                  #
        # ------------------------------------------------------------------ #
        self.model.print_trainable_parameters()

    def compute_sft_loss(
        self,
        input_ids: torch.Tensor,
        prompt_lengths: torch.Tensor,
        eps: float = 1e-3,
    ) -> torch.Tensor:
        """
        Compute the SFT masked-diffusion loss from the paper.

        The key insight vs. pre-training:
          • The *prompt* is NEVER masked — it provides clean conditioning.
          • Only *answer* tokens are noised and must be reconstructed.
          • The loss is normalised by the answer length (including padding EOS)
            so that longer answers don't dominate the gradient.

        Args:
            input_ids      : (batch, seq_len)  padded token ids (prompt + answer)
            prompt_lengths : (batch,)           number of tokens in each prompt
            eps            : lower bound on masking probability

        Returns:
            Scalar cross-entropy loss, ready for ``.backward()``.
        """
        device = input_ids.device
        b, l = input_ids.shape

        # ---- Apply the forward diffusion process to the full sequence ----- #
        noisy_batch, _, p_mask = forward_process(
            input_ids,
            eps=eps,
            mask_token_id=self.mask_token_id,
        )

        # ---- Restore the prompt (never add noise to it) ------------------- #
        # Build a boolean mask: True for prompt positions, False for answer
        positions = torch.arange(l, device=device).unsqueeze(0).expand(b, l)
        prompt_mask = positions < prompt_lengths.unsqueeze(1)   # (b, l)

        # Overwrite the prompt region in the noisy batch with the clean ids
        noisy_batch[prompt_mask] = input_ids[prompt_mask]

        answer_lengths = (1 - prompt_mask.long()).sum(dim=1,
                                                      keepdim=True)  # (b, 1)
        answer_lengths = answer_lengths.expand(
            b, l)                         # (b, l)

        logits = self.model(input_ids=noisy_batch).logits   # (b, l, V)

        # Only compute loss on positions that were actually masked
        masked_indices = noisy_batch == self.mask_token_id  # (b, l)

        token_loss = F.cross_entropy(
            logits[masked_indices],
            input_ids[masked_indices],
            reduction="none",
        )
        # Re-weight by 1/p_mask (paper eq.) and normalise by answer length
        token_loss = token_loss / p_mask[masked_indices]
        loss = torch.sum(token_loss / answer_lengths[masked_indices]) / b

        return loss

    def _tokenize_prompt(self, prompt: Any, device: torch.device) -> torch.Tensor:
        if hasattr(self.tokenizer, "encode"):
            tokenized = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
            )
        elif hasattr(self.tokenizer, "tokenize"):
            tokenized = self.tokenizer.tokenize(prompt)
        else:
            raise ValueError(
                "Tokenizer must provide either `tokenize(...)` or `encode(...)`."
            )

        if isinstance(tokenized, torch.Tensor):
            prompt_ids = tokenized.long()
        else:
            prompt_ids = torch.as_tensor(tokenized, dtype=torch.long)

        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if prompt_ids.ndim != 2:
            raise ValueError(
                f"Expected tokenized prompt to have shape (seq_len,) or (1, seq_len), got {tuple(prompt_ids.shape)}."
            )
        return prompt_ids.to(device)

    def _decode_answer_tokens(self, answer_token_ids: list[int]) -> Any:
        decode_fn = getattr(self.tokenizer, "decode", None)
        if decode_fn is None:
            raise ValueError("Tokenizer must provide a `decode(...)` method.")

        answer_tensor = torch.tensor(answer_token_ids, dtype=torch.long)
        if hasattr(self.tokenizer, "eos_token_id"):
            try:
                return decode_fn(answer_token_ids, skip_special_tokens=True)
            except TypeError:
                return decode_fn(answer_token_ids)
        return decode_fn(answer_tensor)

    def _get_eos_token_id(self) -> int | None:
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            return int(eos_token_id)
        eos_token_id = getattr(self.tokenizer, "eos", None)
        if eos_token_id is not None:
            return int(eos_token_id)
        return None

    def _get_mask_token_id_from_tokenizer(self) -> int | None:
        mask_token_id = getattr(self.tokenizer, "mask_token_id", None)
        if mask_token_id is not None:
            return int(mask_token_id)
        mask_token_id = getattr(self.tokenizer, "mask", None)
        if mask_token_id is not None:
            return int(mask_token_id)
        return None

    def _get_tokenizer_vocab_size(self) -> int | None:
        if hasattr(self.tokenizer, "__len__"):
            return int(len(self.tokenizer))
        return None


def _top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Zero out probability mass below the nucleus threshold.

    Args:
        probs : (seq_len, vocab_size) probability distributions
        top_p : cumulative probability threshold in (0, 1]

    Returns:
        Renormalised probability tensor with the same shape.
    """
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)

    # Remove tokens that push cumulative probability over the threshold
    sorted_probs[cumulative - sorted_probs > top_p] = 0.0

    # Scatter back to original ordering and renormalise
    filtered = torch.zeros_like(
        probs).scatter_(-1, sorted_indices, sorted_probs)
    filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return filtered


######### SAMPLE USAGE ##########

# if __name__ == "__main__":
#     # ---- 1. Basic usage --------------------------------------------------- #
#     llada = LLaDAModel(
#         hf_model_path="GSAI-ML/LLaDA-8B-Instruct",
#         extra_special_tokens=None,    # no extra tokens needed here
#         device="cuda",
#         torch_dtype=torch.bfloat16,
#     )

#     response = llada.generate(
#         prompt="What is the capital of France?",
#         max_new_tokens=64,
#         num_steps=20,
#         temperature=0.0,
#     )
#     print("Response:", response)

#     # # ---- 2. With extra special tokens ------------------------------------- #
#     # llada_custom = LLaDAModel(
#     #     hf_model_path="GSAI-ML/LLaDA-8B-Instruct",
#     #     extra_special_tokens=["[DOMAIN_START]", "[DOMAIN_END]"],
#     #     device="cuda",
#     #     torch_dtype=torch.bfloat16,
#     # )

#     # ---- 3. Prepare for LoRA fine-tuning ---------------------------------- #
#     llada_custom.prepare_for_lora(
#         r=16,
#         lora_alpha=32,
#         lora_dropout=0.05,
#         lora_target_modules=["q_proj", "v_proj"],
#     )

#     # ---- 4. Example SFT training step ------------------------------------- #
#     # (Normally you'd get these from a DataLoader)
#     dummy_input_ids = torch.randint(0, 1000, (2, 64)).cuda()
#     dummy_prompt_lengths = torch.tensor([17, 17]).cuda()

#     loss = llada_custom.compute_sft_loss(dummy_input_ids, dummy_prompt_lengths)
#     print("SFT loss:", loss.item())
#     loss.backward()
