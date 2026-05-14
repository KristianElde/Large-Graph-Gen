# Large-Graph-Gen

Run the example I/O script:

```bash
python example_autograph_io.py
```

## API reference

### `graph_tokenization` package

Import surface:

```python
from graph_tokenization import (
    AutoGraphTokenizer,
    GraphTokenizer,
    SimpleGraphData,
    graph_to_tokens,
)
```

#### `SimpleGraphData`

Minimal graph container used by tokenizers.

- `edge_index: torch.Tensor`
- `num_nodes: int`
- `x: Optional[torch.Tensor] = None` (node labels)
- `edge_attr: Optional[torch.Tensor] = None` (edge labels)
- `dataset_name: Optional[str] = None`

#### `GraphTokenizer` (base API for all tokenizers)

All tokenizer methods should implement this interface:

- `tokenize(data: SimpleGraphData) -> torch.Tensor`
- `decode(tokens) -> SimpleGraphData`

#### `graph_to_tokens(...)`

Utility helper to tokenize from raw graph tensors:

```python
graph_to_tokens(edge_index, num_nodes, tokenizer: GraphTokenizer, **attrs)
```

`**attrs` is forwarded to `SimpleGraphData` (for example `dataset_name`, `x`, `edge_attr`).

#### `AutoGraphTokenizer`

Current tokenizer implementation (`method_name = "autograph"`).

Main methods:

- `set_num_nodes(max_num_nodes)`
- `set_num_node_and_edge_types(num_node_types=0, num_edge_types=0)` (for labeled graphs)
- `tokenize(data)`
- `decode(tokens)`
- `__len__()` (vocabulary size after configuration)

Special token ids exposed as attributes:

- `sos`, `reset`, `ladj`, `radj`, `eos`, `pad`

### `models` package (LLaDA)

Import surface:

```python
from models import LLaDAModel, MASK_TOKEN_ID
```

#### `LLaDAModel(...)`

```python
LLaDAModel(
    hf_model_path: str,
    tokenizer: Any | None = None,
    extra_special_tokens: list[str] | None = None,
    mask_token_id: int | None = None,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
)
```

Public methods:

- `generate(prompt, max_new_tokens=128, num_steps=10, temperature=0.0, top_p=1.0)`
- `prepare_for_lora(r=16, lora_alpha=32, lora_dropout=0.05, lora_target_modules=None, bias="none")`
- `compute_sft_loss(input_ids, prompt_lengths, eps=1e-3)`

#### `forward_process(...)`

Standalone helper used by training loss:

```python
forward_process(input_ids, eps=1e-3, mask_token_id=MASK_TOKEN_ID)
```

## Tokenizer compatibility contract for future tokenizers

To plug a new tokenizer into `LLaDAModel(tokenizer=...)`, provide:

1. Required:
   - `tokenize(prompt_like) -> torch.Tensor | sequence[int]`
   - `decode(tokens) -> output_object`
2. Optional (recommended):
   - `__len__()` for vocabulary size (enables embedding resize check)
   - `eos` or `eos_token_id` for early stop in generation
   - `mask` or `mask_token_id` for diffusion masking

If optional ids are not provided, the model falls back to defaults where possible.
