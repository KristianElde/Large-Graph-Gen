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
    independently.
    """
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].expand(b, l)

    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    noisy_batch = torch.where(masked_indices, mask_token_id, input_ids)

    return noisy_batch, masked_indices, p_mask

class LLaDAModel:
    """
    Thin wrapper around a LLaDA HuggingFace checkpoint.
    """

    _AUTOGRAPH_EXTRA_TOKEN_NAMES: tuple[str, ...] = ("<sos>", "<eos>", "<reset>", "<ladj>", "<radj>",)

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

        self._original_vocab_size: int | None = None
        # Maps autograph attribute name → new HF token id
        self.graph_special_token_ids: dict[str, int] = {}

        # Step 1 — Load the tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                hf_model_path,
                trust_remote_code=True,
            )
        else:
            self.tokenizer = tokenizer

        # Step 2 — Register extra special tokens (optional, original API)
        if extra_special_tokens and tokenizer is not None:
            raise ValueError(
                "extra_special_tokens is only supported when using the "
                "Hugging Face tokenizer."
            )
        if extra_special_tokens:
            num_added = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": extra_special_tokens}
            )
            print(f"[LLaDA] Added {num_added} extra special token(s): "
                  f"{extra_special_tokens}")

        # Step 3 — Load the model weights
        load_kwargs: dict = dict(
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        if device == "auto":
            load_kwargs["device_map"] = "auto"

        self.model = AutoModel.from_pretrained(hf_model_path, **load_kwargs)

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
                    f"[LLaDA] Embedding matrix resized to "
                    f"{tokenizer_vocab_size} tokens to match the supplied "
                    f"tokenizer."
                )

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

        if (
            tokenizer_vocab_size is not None
            and self.mask_token_id >= tokenizer_vocab_size
        ):
            raise ValueError(
                f"mask_token_id={self.mask_token_id} is out of tokenizer "
                f"vocabulary range [0, {tokenizer_vocab_size - 1}]."
            )

    def integrate_graph_tokenizer_special_tokens(
        self,
        graph_tokenizer,
    ) -> dict[str, int]:
        """
        Register AutoGraph's graph-specific special tokens inside the HF
        tokenizer and resize the model's embedding matrix to accommodate them.

        """
        # Record original vocab size so we can unfreeze only the new rows
        self._original_vocab_size = (
            self.model.get_input_embeddings().num_embeddings
        )

        # Build the string tokens we want to add.
        # graph_tokens_to_text renders these as "<reset>", "<ladj>", "<radj>",
        # so we register exactly those strings — no prefix needed.
        new_tokens: list[str] = []
        for attr in self._AUTOGRAPH_EXTRA_TOKEN_NAMES:
            if not hasattr(graph_tokenizer, attr):
                raise AttributeError(
                    f"graph_tokenizer has no attribute '{attr}'. "
                    f"Make sure you are passing an AutoGraphTokenizer."
                )
            new_tokens.append(f"<graph_{attr}>")

        if new_tokens:
            num_added = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": new_tokens}
            )
            print(
                f"[LLaDA] Registered {num_added} graph special token(s) in "
                f"HF tokenizer: {new_tokens}"
            )
        else:
            print(
                "[LLaDA] All graph special tokens are already in the HF "
                "tokenizer; skipping add_special_tokens."
            )

        # Resize the embedding matrix
        new_vocab_size = len(self.tokenizer)
        self.model.resize_token_embeddings(new_vocab_size)
        print(
            f"[LLaDA] Embedding matrix resized: "
            f"{self._original_vocab_size} → {new_vocab_size} rows."
        )

        # Build the attr → hf_token_id mapping
        self.graph_special_token_ids = {}
        for attr, tok_str in zip(self._AUTOGRAPH_EXTRA_TOKEN_NAMES, new_tokens):
            hf_id = self.tokenizer.convert_tokens_to_ids(tok_str)
            self.graph_special_token_ids[attr] = hf_id
            print(f"[LLaDA]   {attr:10s} → '{tok_str}'  (id={hf_id})")

        return self.graph_special_token_ids

    # ---------------------------------------------------------------------- #
    # generate  (unchanged)                                                   #
    # ---------------------------------------------------------------------- #

    @torch.inference_mode()
    def generate(
        self,
        prompt: Any,
        max_new_tokens: int = 128,
        num_steps: int = 10,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Any:
        device = next(self.model.parameters()).device

        prompt_ids = self._tokenize_prompt(prompt, device=device)
        prompt_len = prompt_ids.shape[1]
        eos_id = self._get_eos_token_id()

        answer_ids = torch.full(
            (1, max_new_tokens),
            fill_value=self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
        answer_mask = torch.ones(
            max_new_tokens, dtype=torch.bool, device=device
        )
        tokens_per_step = max(1, max_new_tokens // num_steps)

        for step in range(num_steps):
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits
            answer_logits = logits[0, prompt_len:, :]

            if temperature == 0.0:
                predicted_ids = answer_logits.argmax(dim=-1)
                probs = F.softmax(answer_logits, dim=-1)
                confidence = probs.gather(
                    1, predicted_ids.unsqueeze(-1)
                ).squeeze(-1)
            else:
                scaled_logits = answer_logits / temperature
                probs = F.softmax(scaled_logits, dim=-1)
                if top_p < 1.0:
                    probs = _top_p_filter(probs, top_p)
                predicted_ids = torch.multinomial(
                    probs, num_samples=1
                ).squeeze(-1)
                confidence = probs.gather(
                    1, predicted_ids.unsqueeze(-1)
                ).squeeze(-1)

            still_masked_positions = answer_mask.nonzero(as_tuple=True)[0]
            num_still_masked = still_masked_positions.numel()

            if num_still_masked == 0:
                break

            if step == num_steps - 1:
                commit_positions = still_masked_positions
            else:
                n_commit = min(tokens_per_step, num_still_masked)
                masked_conf = confidence[still_masked_positions]
                topk_local = torch.topk(masked_conf, k=n_commit).indices
                commit_positions = still_masked_positions[topk_local]

            input_ids[0, prompt_len + commit_positions] = predicted_ids[
                commit_positions
            ]
            answer_mask[commit_positions] = False

            if eos_id is not None:
                committed_answer = input_ids[0, prompt_len:]
                if (committed_answer == eos_id).any():
                    break

        answer_token_ids = input_ids[0, prompt_len:].tolist()
        if eos_id is not None and eos_id in answer_token_ids:
            answer_token_ids = answer_token_ids[
                : answer_token_ids.index(eos_id)
            ]
        answer_token_ids = [
            t for t in answer_token_ids if t != self.mask_token_id
        ]
        return self._decode_answer_tokens(answer_token_ids)

    # ---------------------------------------------------------------------- #
    # prepare_for_lora  (extended to unfreeze new embedding rows)             #
    # ---------------------------------------------------------------------- #

    def prepare_for_lora(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        bias: str = "none",
    ) -> None:
        """
        Wrap the model with PEFT LoRA adapters.

        Extended behaviour (vs. original)
        ----------------------------------
        If ``integrate_graph_tokenizer_special_tokens`` has been called before
        this method, the newly added embedding rows (rows
        ``[_original_vocab_size, new_vocab_size)``) are **explicitly unfrozen**
        after PEFT wrapping.

        PEFT freezes the entire embedding table because it is not a LoRA
        target module.  We restore gradient flow to only the new rows via:

        1. ``requires_grad_(True)`` on the embedding weight.
        2. A ``register_hook`` on the weight that zeroes gradients for the
           original rows, so those rows behave as if frozen.

        This means the optimizer will see the new row gradients and update only
        those rows, preserving the pre-trained embeddings.
        """
        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "v_proj"]

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias=bias,
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # ------------------------------------------------------------------ #
        # Unfreeze the new embedding rows (if graph tokens were registered)   #
        # ------------------------------------------------------------------ #
        if self._original_vocab_size is None:
            # No graph tokens were added; nothing extra to do.
            return

        original_vocab_size = self._original_vocab_size

        # Locate the input embedding module inside the PEFT-wrapped model
        try:
            embed_module = self.model.get_input_embeddings()
        except Exception:
            print(
                "[LLaDA] WARNING: Could not locate input embeddings inside "
                "PEFT model.  New embedding rows will NOT be trained."
            )
            return

        weight = embed_module.weight          # (new_vocab_size, hidden_dim)
        new_vocab_size = weight.shape[0]

        if new_vocab_size <= original_vocab_size:
            # Nothing was added; skip.
            return

        # Enable gradient on the whole embedding weight tensor
        weight.requires_grad_(True)

        # Register a hook that zeroes gradients for original rows so they
        # remain effectively frozen
        def _mask_original_rows(grad: torch.Tensor) -> torch.Tensor:
            """Zero gradient for rows 0..original_vocab_size-1."""
            grad = grad.clone()
            grad[:original_vocab_size] = 0.0
            return grad

        weight.register_hook(_mask_original_rows)

        num_new = new_vocab_size - original_vocab_size
        print(
            f"[LLaDA] Unfrozen {num_new} new embedding row(s) "
            f"(rows {original_vocab_size}–{new_vocab_size - 1}) for training."
        )

    # ---------------------------------------------------------------------- #
    # compute_sft_loss  (unchanged)                                           #
    # ---------------------------------------------------------------------- #

    def compute_sft_loss(
        self,
        input_ids: torch.Tensor,
        prompt_lengths: torch.Tensor,
        eps: float = 1e-3,
    ) -> torch.Tensor:
        device = input_ids.device
        b, l = input_ids.shape

        noisy_batch, _, p_mask = forward_process(
            input_ids, eps=eps, mask_token_id=self.mask_token_id
        )

        positions = torch.arange(l, device=device).unsqueeze(0).expand(b, l)
        prompt_mask = positions < prompt_lengths.unsqueeze(1)

        noisy_batch[prompt_mask] = input_ids[prompt_mask]

        answer_lengths = (1 - prompt_mask.long()).sum(dim=1, keepdim=True)
        answer_lengths = answer_lengths.expand(b, l)

        logits = self.model(input_ids=noisy_batch).logits

        masked_indices = noisy_batch == self.mask_token_id

        token_loss = F.cross_entropy(
            logits[masked_indices],
            input_ids[masked_indices],
            reduction="none",
        )
        token_loss = token_loss / p_mask[masked_indices]
        loss = torch.sum(token_loss / answer_lengths[masked_indices]) / b

        return loss

    # ---------------------------------------------------------------------- #
    # Private helpers  (unchanged)                                            #
    # ---------------------------------------------------------------------- #

    def _tokenize_prompt(self, prompt: Any, device: torch.device) -> torch.Tensor:
        if hasattr(self.tokenizer, "encode"):
            tokenized = self.tokenizer.encode(
                prompt, return_tensors="pt", add_special_tokens=True
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
                f"Expected tokenized prompt to have shape (seq_len,) or "
                f"(1, seq_len), got {tuple(prompt_ids.shape)}."
            )
        return prompt_ids.to(device)

    def _decode_answer_tokens(self, answer_token_ids: list[int]) -> Any:
        decode_fn = getattr(self.tokenizer, "decode", None)
        if decode_fn is None:
            raise ValueError("Tokenizer must provide a `decode(...)` method.")

        if hasattr(self.tokenizer, "eos_token_id"):
            try:
                return decode_fn(answer_token_ids, skip_special_tokens=True)
            except TypeError:
                return decode_fn(answer_token_ids)
        return decode_fn(
            torch.tensor(answer_token_ids, dtype=torch.long)
        )

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


# ---------------------------------------------------------------------------
# _top_p_filter  (unchanged)
# ---------------------------------------------------------------------------

def _top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    sorted_probs[cumulative - sorted_probs > top_p] = 0.0
    filtered = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
    filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return filtered
