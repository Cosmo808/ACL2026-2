import logging
import evaluate
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from peft import LoraConfig, TaskType
from utils.xnli.zett.tokenizer_converters import convert_to_byte_level
from utils.xnli.tokenizations.tokenization_utils import DatasetEncoder
from utils.xnli.tokenizations.hypernet_cache import LRU_Cache

logger = logging.getLogger(__name__)


def prepare_model_and_tokenizer_and_collator_and_encoder(
        model_args, adapter_args, num_labels, label_list, is_regression, data_args, training_args):
    """Prepares the model, tokenizer, data_collator, DatasetEncoder, and compute_metrics function for dynamic tokenization."""
    # --- Load Config ---
    config = AutoConfig.from_pretrained(
        (model_args.config_name if model_args.config_name else model_args.model_name_or_path),
        num_labels=num_labels, finetuning_task=data_args.task_name, cache_dir=model_args.cache_dir,
        revision=model_args.model_revision, use_auth_token=True if model_args.use_auth_token else None,
    )

    # --- Load Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        (model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision, use_auth_token=True if model_args.use_auth_token else None,
    )

    # --- Load Model ---
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config, cache_dir=model_args.cache_dir, revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None, ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    # --- Setup Adapters (LoRA/AdaLoRA) ---
    peft_config = None
    if adapter_args.lora:
        logger.info(f"Using PEFT-LoRA")
        peft_config = LoraConfig(
            lora_alpha=adapter_args.lora_alpha, lora_dropout=adapter_args.lora_dropout, r=adapter_args.lora_rank,
            bias=adapter_args.lora_bias, task_type=TaskType.SEQ_CLS, inference_mode=False, modules_to_save=["classifier"],
        )

    if (adapter_args.lora or adapter_args.adalora) and model_args.further_training_adapter_path == "":
        logger.info("Loading PEFT-adapter")
        model.add_adapter(peft_config)

    # --- Update Model Config with Labels ---
    label_to_id = {v: i for i, v in enumerate(label_list)}

    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}

    # --- Prepare compute_metrics function ---
    # Get the metric function
    metric = evaluate.load("accuracy")

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}  # xnli

    # --- DYNAMIC TOKENIZATION BRANCH ---
    logger.info("Setting up for Dynamic Tokenization.")
    # 1. Convert the initially loaded tokenizer
    tokenizer = convert_to_byte_level(tokenizer)[0]

    # 2. Load other dynamic components using the converted tokenizer and model
    langs = [x.strip() for x in open("data/artifacts/26.txt")]
    lang_index = torch.tensor(langs.index(data_args.dataset_config_name), dtype=torch.int32).to(training_args.device)
    hypernet = AutoModel.from_pretrained("benjamin/zett-hypernetwork-xlm-roberta-base", trust_remote_code=True).to(training_args.device)
    source_embeddings = model.get_input_embeddings().weight.data.to(training_args.device)
    embeddings_cache = LRU_Cache(cache_size=training_args.cache_size, device=training_args.device)

    # 3. Create the DatasetEncoder using the converted tokenizer and other components
    datasetEncoder = DatasetEncoder(
        hypernet=hypernet,
        tokenizer=tokenizer,
        device=training_args.device,
        lang_index=lang_index,
        surface_form_maxlen=training_args.surface_form_maxlen,
        source_embeddings=source_embeddings,
        embeddings_cache=embeddings_cache,
        exp_type=training_args.exp_type,
        bpe_tokenizer_boundary=training_args.bpe_tokenizer_boundary,
        merges=model_args.dynamic_tokenization_merges,
        collect_extra_data=training_args.collect_extra_data,
    )

    # 4. Create the DynamicDataCollator using the same components (and converted tokenizer)
    # data_collator = DynamicDataCollator(
    #     hypernet=hypernet,
    #     tokenizer=tokenizer,
    #     device=training_args.device,
    #     lang_index=lang_index,
    #     surface_form_maxlen=training_args.surface_form_maxlen,
    #     source_embeddings=source_embeddings,
    #     embeddings_cache=embeddings_cache,
    #     exp_type=training_args.exp_type,
    #     bpe_tokenizer_boundary=training_args.bpe_tokenizer_boundary,
    # )

    return model, hypernet, tokenizer, compute_metrics, datasetEncoder
