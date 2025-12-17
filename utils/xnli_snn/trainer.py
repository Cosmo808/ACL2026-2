import logging
import torch
from typing import Any, Dict, Optional, Union
from transformers import (DefaultDataCollator, Trainer)


logger = logging.getLogger(__name__)


class DynamicDataCollator(DefaultDataCollator):
    def __call__(self, examples):
        """
        Converts list of dictionaries to dictionaries of list to ease post-processing implementation of Trainer
        """
        examples = {
            key if key != "label" else "labels": [dic[key] for dic in examples]
            for key in examples[0]
        }
        examples['labels'] = torch.tensor(examples['labels'])
        return examples


def prepare_trainer(model, training_args, raw_datasets, compute_metrics):
    """Prepares and returns the configured Trainer instance."""
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"] if training_args.do_train else None,
        eval_dataset=raw_datasets["validation"] if training_args.do_eval else None,
        tokenizer=None,
        data_collator=DynamicDataCollator(),
        compute_metrics=compute_metrics,
    )
    trainer.seed = training_args.seed
    trainer.args.remove_unused_columns = False

    return trainer
