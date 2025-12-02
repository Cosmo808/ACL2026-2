import logging
import random
from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_raw_datasets(data_args, training_args, model_args):
    """Loads raw datasets without preprocessing or collator setup."""
    # --- Load raw datasets ---
    raw_datasets = load_dataset(
        data_args.dataset_name,  # xnli
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # --- Determine labels ---
    is_regression = raw_datasets["train"].features["label"].dtype in [
        "float32",
        "float64",
    ]  # False
    if is_regression:
        num_labels = 1
    else:
        label_list = raw_datasets["train"].unique("label")
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)

    # --- Train/Test Datasets ---
    train_dataset, eval_dataset, predict_dataset = None, None, None
    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        eval_dataset = raw_datasets["validation"]

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        predict_dataset = raw_datasets["test"]

    return raw_datasets, is_regression, label_list, num_labels, train_dataset, eval_dataset, predict_dataset
