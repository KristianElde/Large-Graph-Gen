from __future__ import annotations

import argparse
from pathlib import Path

import torch
import dllm
from dllm.utils.configs import ModelArguments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_args = ModelArguments(
        model_name_or_path=args.base_model,
        lora=False,
    )

    print("Loading architecture/tokenizer from:", args.base_model)
    model = dllm.utils.get_model(model_args=model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    print("Reinitializing model weights...")
    if hasattr(model, "_init_weights"):
        for module in model.modules():
            model._init_weights(module)
    else:
        def reset(module):
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        model.apply(reset)

    print("Saving random-initialized checkpoint to:", out_dir)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print("Done.")


if __name__ == "__main__":
    main()