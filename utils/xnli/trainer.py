import logging
import random
import sys
import datasets
import numpy as np
import pandas as pd
import torch
from typing import Any, Dict, Optional, Union
from transformers import (DefaultDataCollator, Trainer)


HOME_PATH = "/mnt/nas_home/dmf45/dynamic_tokenization"
sys.path.insert(0, HOME_PATH)


logger = logging.getLogger(__name__)


def seed_worker(_) -> None:
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DynamicDataCollator(DefaultDataCollator):
    # def __init__(
    #     self, tokenizer, hypernet, device, lang_index, surface_form_maxlen, source_embeddings,
    #     embeddings_cache, exp_type, bpe_tokenizer_boundary, max_length=128, merges=0,
    # ):
    #     super().__init__(tokenizer)
    #     self.tokenizer = tokenizer
    #     self.max_length = max_length
    #     self.datasetEncoder = DatasetEncoder(
    #         hypernet=hypernet,
    #         tokenizer=tokenizer,
    #         device=device,
    #         lang_index=lang_index,
    #         surface_form_maxlen=surface_form_maxlen,
    #         source_embeddings=source_embeddings,
    #         embeddings_cache=embeddings_cache,
    #         exp_type=exp_type,
    #         bpe_tokenizer_boundary=bpe_tokenizer_boundary,
    #         collect_extra_data=False,
    #         merges=merges,
    #     )
    #     self.merges = merges

    def __call__(self, examples):
        """
        Converts list of dictionaries to dictionaries of list to ease post-processing implementation of Trainer
        """
        if "inputs_embeds" in examples[0] and "attention_mask" in examples[0]:
            examples = {
                key if key != "label" else "labels": torch.stack([item[key] for item in examples])
                for key in ["inputs_embeds", "attention_mask", "label"]
            }
        else:
            examples = {
                key if key != "label" else "labels": [dic[key] for dic in examples]
                for key in examples[0]
            }
        return examples


class CustomTrainer(Trainer):
    def set_dataset_encoder_and_tokeniser_sampling(self, datasetEncoder):
        self.datasetEncoder = datasetEncoder
        self.do_tokeniser_sampling_per_sample = False
        self.do_tokeniser_sampling_per_batch = False
        self.do_tokeniser_sampling_per_batch_gaussian = False
        self.do_tokeniser_sampling_per_sample_gaussian = False
        self.do_tokeniser_sampling_per_batch_cauchy = False
        self.do_tokeniser_sampling_per_batch_student_t = False

    def _prepare_inputs(
            self, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        if "inputs_embeds" in inputs and "attention_mask" in inputs:  # Used for evaluation which is already tokenised
            return inputs

        encoded_batch = self.datasetEncoder.encode_examples_unique_tokens_lru(examples=inputs)
        encoded_batch["labels"] = torch.tensor(inputs["labels"])
        return encoded_batch


def prepare_trainer(model, training_args, raw_datasets, tokenizer, compute_metrics, model_args, datasetEncoder, eval_dataset):
    """Prepares and returns the configured Trainer instance."""
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"] if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=None if model_args.tokenization_type == "dynamic" else tokenizer,
        data_collator=DynamicDataCollator(),
        compute_metrics=compute_metrics,
    )
    trainer.seed = training_args.seed
    trainer.args.remove_unused_columns = False
    trainer.set_dataset_encoder_and_tokeniser_sampling(datasetEncoder)

    return trainer
