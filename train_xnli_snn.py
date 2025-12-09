import logging
import os
import random
import sys
import numpy as np
import datetime
import torch
import transformers.utils
import datasets
from transformers.trainer_utils import get_last_checkpoint

from utils.xnli_snn.arguments import ModelArguments, DataTrainingArguments, CustomTrainingArguments, AdapterArguments
from utils.xnli_snn.dataloader import load_raw_datasets
from utils.xnli_snn.model import prepare_model
from utils.xnli_snn.trainer import prepare_trainer


def setup_seed(seed: int) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(mode=True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


if __name__ == "__main__":
    ###########################################################################
    # Parameter
    ###########################################################################
    model_name_or_path = "xlm-roberta-base"
    out_prefix = "E:/ACL2026-2/output_lora_peft/dynamic_tk_snn"
    snn_tokenizer_path = "E:/ACL2026-2/utils/xnli_snn/snn_tokenizer.pt"
    method = "lora"
    batchsize = 100
    max_len = 256
    dropout = 0.3
    rank = 128
    seed = 42
    tokenization_type = "dynamic"
    lng = "en"
    alpha = 2 * rank
    entropy = False
    entropyornot = "entropy" if entropy else "noentropy"
    snn_frozen = True

    today = datetime.date.today().strftime("%d%m%Y")
    output_dir = f"{out_prefix}/{tokenization_type}_{lng}_{method}_dropout_{dropout}_rank_{rank}_seed_{seed}_{today}_{entropyornot}"

    model_args = ModelArguments(
        model_name_or_path=model_name_or_path,
        tokenization_type=tokenization_type,
        max_char_len=max_len,
        entropy=entropy,
        snn_tokenizer_path=snn_tokenizer_path,
        snn_frozen=snn_frozen,
    )

    data_args = DataTrainingArguments(
        dataset_name="xnli",
        dataset_config_name=lng,
        max_seq_length=max_len,
    )

    training_args = CustomTrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        # do_predict=True,
        per_device_train_batch_size=batchsize,
        per_device_eval_batch_size=batchsize,
        learning_rate=1e-4,
        num_train_epochs=15.0,
        save_steps=5000,
        eval_steps=5000,
        seed=seed,
        bf16=True,
        overwrite_output_dir=True,
        report_to="wandb",
        log_level="info",
        push_to_hub=False,
        save_safetensors=False,
    )

    adapter_args = AdapterArguments(
        lora=(method == "lora"),
        adalora=False,
        lora_alpha=alpha,
        lora_dropout=dropout,
        lora_rank=rank,
    )

    ###########################################################################
    # Init
    ###########################################################################
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Log arguments
    print(f"Adapter Args: {adapter_args}")
    print(f"Model Args: {model_args}")
    print(f"Training Args: {training_args}")
    print(f"Data Args: {data_args}")

    # Configure training arguments for dynamic tokenization
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    training_args.remove_unused_columns = False
    training_args.data_seed = training_args.seed

    log_level = training_args.get_process_log_level()
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.logging.set_verbosity(log_level)
    transformers.logging.enable_default_handler()
    transformers.logging.enable_explicit_format()

    # Set seed before initializing model.
    setup_seed(training_args.seed)

    # Log on each process the small summary:
    logger.warning(f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
                    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                        "the `--output_dir` or add `--overwrite_output_dir` to train from scratch.")

    ###########################################################################
    # Load data
    ###########################################################################
    raw_datasets, is_regression, label_list, num_labels, train_dataset, eval_dataset, predict_dataset = load_raw_datasets(
        data_args, training_args, model_args
    )

    ###########################################################################
    # Prepare the model
    ###########################################################################
    model, compute_metrics = prepare_model(
        model_args, adapter_args, num_labels, label_list, is_regression, data_args
    )

    ###########################################################################
    # Train and Evaluate
    ###########################################################################
    trainer = prepare_trainer(
        model, training_args, raw_datasets, compute_metrics, model_args, eval_dataset
    )

    # Training
    if training_args.do_train:
        logger.info("\n*** Train ***")

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(raw_datasets["train"])
        )
        metrics["train_samples"] = min(max_train_samples, len(raw_datasets["train"]))
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("\n*** Evaluate ***")
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)
            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("\n*** Predict ***")
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        # Prepare predict_dataset here in main.py, as it was in the original script
        predict_dataset = raw_datasets[
            "test_matched" if data_args.task_name == "mnli" else "test"
        ]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples
            )
            predict_dataset = predict_dataset.select(range(max_predict_samples))

        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(
                predict_dataset, metric_key_prefix="predict"
            ).predictions
            predictions = (
                np.squeeze(predictions)
                if is_regression
                else np.argmax(predictions, axis=1)
            )
            output_predict_file = os.path.join(
                training_args.output_dir, f"predict_results_{task}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    # Finalize (push to hub if applicable)
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "text-classification",
    }
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    logger.info("Training/Prediction finished.")