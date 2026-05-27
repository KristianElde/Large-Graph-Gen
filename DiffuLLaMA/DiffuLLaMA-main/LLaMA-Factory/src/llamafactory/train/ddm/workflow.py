# Inspired by: https://github.com/huggingface/transformers/blob/v4.29.2/examples/pytorch/summarization/run_summarization.py

from typing import TYPE_CHECKING, Optional, List, Tuple, Union
from transformers import DataCollatorForLanguageModeling, Seq2SeqTrainingArguments

import math, os, json, tqdm, sys
import wandb
from typing import TYPE_CHECKING, List, Optional
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from ...data import get_dataset
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from ...extras.constants import IGNORE_INDEX


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

from .trainer import CustomDiffusionTrainer
from .trainer import generate_samples, generate_samples_v2
from .metric import gsm_eval

def run_ddm(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    dataset_module = get_dataset(model_args, data_args, training_args, stage=finetuning_args.stage, **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0) and training_args.do_train:
        # dist.init_process_group(backend='nccl')
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "adaptation-diffusion"),
            name=training_args.output_dir,
        )
        # wandb.config.update(training_args.__dict__, allow_val_change=True)

    # Initialize our Trainer
    trainer = CustomDiffusionTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
    )
    
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")

        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # import pdb; pdb.set_trace();

    # Predict
    if training_args.do_predict:
        if data_args.eval_dataset is not None and "gsm" in data_args.eval_dataset[0]: 
            sample_function = generate_samples
            kargs = {"gsm": True}
        else:
            sample_function = generate_samples_v2
            kargs = {}
 
        data_loader = DataLoader(
            dataset_module["eval_dataset"],
            batch_size=training_args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=data_args.preprocessing_num_workers,
        )
        pred_list = []
        label_list = []
        for batch in tqdm.tqdm(data_loader):
            predict_results = sample_function(model, finetuning_args, tokenizer, batch, eval=False, **kargs)
            labels = torch.transpose(torch.stack(batch['input_ids']), 0, 1)
            
            predict_results = predict_results.tolist()
            # import pdb; pdb.set_trace();
            preds = np.where(predict_results != IGNORE_INDEX, predict_results, tokenizer.pad_token_id)
            stripped_preds = []
            for seq in preds:
                seq = seq.tolist()
                while seq and (seq[-1] == tokenizer.pad_token_id):
                    seq.pop()
                stripped_preds.append(seq)
            decoded_preds = tokenizer.batch_decode(stripped_preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            stripped_labels = []
            labels = labels.tolist()
            for seq in labels:
                while seq and seq[-1] == tokenizer.pad_token_id:
                    seq.pop()
                stripped_labels.append(seq)
            decoded_labels = tokenizer.batch_decode(stripped_labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            save_predictions(training_args.output_dir, decoded_preds, decoded_labels)
            pred_list.extend(decoded_preds)
            label_list.extend(decoded_labels)

        # sys.exit(0)
        # import pdb; pdb.set_trace();
        if data_args.eval_dataset is not None and "gsm" in data_args.eval_dataset[0]:    
            print("metric: ", gsm_eval(pred_list, label_list))

    # Create model card
    # create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)


def save_predictions(
    output_dir,
    decoded_preds,
    decoded_labels,
) -> None:
    r"""
    Saves model predictions to `output_dir`.
    """
    output_prediction_file = os.path.join(output_dir, "generated_predictions.jsonl")
    print(f"Saving prediction results to {output_prediction_file}")

    with open(output_prediction_file, "a", encoding="utf-8") as writer:
        res: List[str] = []
        for pred, label in zip(decoded_preds, decoded_labels):
            res.append(json.dumps({"predict": pred, "label": label}, ensure_ascii=False))
        writer.write("\n".join(res))
        writer.write("\n")
