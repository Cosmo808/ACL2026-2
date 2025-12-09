import logging
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM
from peft import AdaLoraConfig, LoraConfig, PeftConfig, PeftModel, TaskType
from spikingjelly.activation_based import functional

from utils.xnli_snn.snn_tokenizer import SNNTokenizer
from utils.xnli_snn.neurons import MembraneLoss


logger = logging.getLogger(__name__)


class TkLM(nn.Module):
    def __init__(self, snn_tokenizer, lm, lm_head, lm_tk, entropy: bool):
        super().__init__()
        self.snn_tokenizer = snn_tokenizer
        self.lm = lm
        self.lm_head = lm_head
        self.lm_tk = lm_tk
        self.lambda_aux = 1.
        self.entropy = entropy

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.snn_loss = MembraneLoss()

    def forward(self, premise=None, hypothesis=None, labels=None, **kwargs):
        # === 1. Tokenization and LM Inference ===
        assert premise is not None and hypothesis is not None
        self.snn_tokenizer.node.training = True
        input_texts = ["<s>" + p + "</s>" + h + "</s>" for p, h in zip(premise, hypothesis)]
        use_hard_boundaries = False if self.training else True
        inputs_embeds = self.snn_tokenizer(input_texts, use_hard_boundaries=use_hard_boundaries)   # (B, L)
        outputs = self.lm(inputs_embeds=inputs_embeds, labels=labels, **kwargs)
        lm_loss = outputs.loss  # XLM-R classification loss

        # === 2. Get token boundaries from XLM-R tokenizer ===
        # Tokenize with original XLM-R tokenizer to get GT boundaries
        gt_boundaries = []
        for text in input_texts:
            tokens = self.lm_tk.tokenize(text)
            char_len = len(text.encode('utf-8'))
            boundary = torch.zeros(char_len, dtype=torch.long)
            pos = 0
            for token in tokens:
                token_str = token[1:] if token.startswith("▁") else token
                try:
                    token_bytes = token_str.encode('utf-8')
                except UnicodeEncodeError:
                    token_bytes = b''
                if len(token_bytes) == 0:
                    continue
                if pos < char_len:
                    boundary[pos] = 1
                pos += len(token_bytes)

            if char_len < self.snn_tokenizer.max_char_len:
                boundary_pad = torch.zeros(self.snn_tokenizer.max_char_len - char_len, dtype=torch.bool)
                boundary = torch.cat([boundary, boundary_pad])
            else:
                boundary = boundary[:self.snn_tokenizer.max_char_len]
            gt_boundaries.append(boundary)
        gt_boundaries = torch.stack(gt_boundaries).to(inputs_embeds.device)  # (B, T)

        # === 3. Top 20% high-entropy tokens ===
        reset_target = None
        token_entropy = self.chunked_topk_token_entropy(inputs_embeds, chunk_size=32, k=100)
        if self.entropy:
            # B, L = token_entropy.shape
            # k = max(1, int(0.2 * L))
            # entropy_top_mask = torch.zeros_like(gt_boundaries, dtype=torch.bool)  # (B, T)
            # token_ids = self.snn_tokenizer.token_ids  # (B, T)
            # for b in range(B):
            #     _, top_indices = torch.topk(token_entropy[b], k, largest=True)  # [k]
            #     match = token_ids[b].unsqueeze(1) == top_indices.unsqueeze(0)  # [T, k]
            #     first_pos = torch.argmax(match.to(torch.float32), dim=0)  # [k]
            #     entropy_top_mask[b, first_pos] = True
            # reset_target = entropy_top_mask.float()  # (B, T)

            token_ids = self.snn_tokenizer.token_ids  # (B, T)
            B, L = token_entropy.shape
            T = gt_boundaries.shape[1]

            k_tokens = max(1, int(0.2 * L))
            _, top_token_indices = torch.topk(token_entropy, k_tokens, dim=1)  # (B, k_tokens)

            first_pos = torch.full((B, L), T, dtype=torch.long, device=token_ids.device)
            batch_idx = torch.arange(B, device=token_ids.device).unsqueeze(1)  # (B, 1)
            char_positions = torch.arange(T, device=token_ids.device).unsqueeze(0).expand(B, T)  # (B, T)
            first_pos.scatter_reduce_(1, token_ids, char_positions, reduce="min", include_self=True)
            top_first_positions = first_pos.gather(1, top_token_indices)  # (B, k_tokens)

            entropy_top_mask = torch.zeros_like(gt_boundaries, dtype=torch.bool)
            valid = top_first_positions < T
            entropy_top_mask[batch_idx.expand_as(top_first_positions)[valid], top_first_positions[valid]] = True

            gt_boundaries = gt_boundaries | entropy_top_mask
            reset_target = entropy_top_mask.float()

        # === 4. Compute auxiliary losses ===
        # Get reset_logits from SNN tokenizer
        reset_logits = self.snn_tokenizer.reset_logits  # (B, T)

        # Reset predictor loss
        reset_loss = 0.0
        if reset_target is not None and reset_logits.shape == reset_target.shape:
            reset_loss = self.bce_loss(reset_logits, reset_target)

        # SNN membrane loss
        gt_idx = [torch.where(gt_boundaries[b])[0] for b in range(gt_boundaries.shape[0])]
        gt_idx = [idx[1:] - 1 for idx in gt_idx]  # remove the first one, SNN cannot spike at first time point
        snn_loss = self.snn_loss(self.snn_tokenizer.node.past_v, self.snn_tokenizer.I, gt_idx)

        # Total loss
        aux_loss = snn_loss / 2. + reset_loss
        total_loss = lm_loss + self.lambda_aux * aux_loss

        # Clean up SNN state
        functional.reset_net(self.snn_tokenizer)
        self.snn_tokenizer.I = []

        # Return
        return SequenceClassifierOutput(
            loss=total_loss,
            logits=outputs.logits,
        )

    def chunked_topk_token_entropy(self, inputs_embeds, chunk_size=32, k=100, **kwargs):
        with torch.no_grad():
            roberta_out = self.lm.roberta(inputs_embeds=inputs_embeds, **kwargs)
            hidden = roberta_out.last_hidden_state
            B, L, _ = hidden.shape

            entropies = []
            for start in range(0, L, chunk_size):
                end = min(start + chunk_size, L)
                hidden_chunk = hidden[:, start:end, :]
                logits_chunk = self.lm_head(hidden_chunk)  # (B, chunk, vocab_size) — allocated only for the chunk

                # Efficient top-k entropy approximation
                topk_logits = torch.topk(logits_chunk, k, dim=-1).values  # (B, chunk, k)
                probs = F.softmax(topk_logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)  # (B, chunk)
                entropies.append(entropy)

                del logits_chunk, topk_logits, probs, entropy

        token_entropy = torch.cat(entropies, dim=1)  # (B, L)
        return token_entropy


def prepare_model(model_args, adapter_args, num_labels, label_list, is_regression, data_args):
    """Prepares the model, tokenizer, data_collator, DatasetEncoder, and compute_metrics function for dynamic tokenization."""
    # --- Load Config ---
    config = AutoConfig.from_pretrained(
        (model_args.config_name if model_args.config_name else model_args.model_name_or_path),
        num_labels=num_labels, finetuning_task=data_args.task_name, cache_dir=model_args.cache_dir,
        revision=model_args.model_revision, use_auth_token=True if model_args.use_auth_token else None,
    )

    # --- Load Tokenizer ---
    tokenizer = SNNTokenizer(
        model_args.char_embed_dim, model_args.ann_hidden_dim, model_args.output_embed_dim,
        model_args.max_char_len, model_args.entropy
    )
    if model_args.snn_tokenizer_path:
        snn_tokenizer_dict = torch.load(model_args.snn_tokenizer_path, map_location="cuda", weights_only=False)
        tokenizer.load_state_dict(snn_tokenizer_dict)
    if model_args.snn_frozen:
        for param in tokenizer.parameters():
            param.requires_grad = False

    lm_tk = AutoTokenizer.from_pretrained(
        (model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision, use_auth_token=True if model_args.use_auth_token else None,
    )

    # --- Load Language Model ---
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config, cache_dir=model_args.cache_dir, revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None, ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    for name, param in model.named_parameters():
        if "lora" in name or "classifier" in name or "adapter" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    mlm_model = AutoModelForMaskedLM.from_pretrained(
        model_args.model_name_or_path, from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config, cache_dir=model_args.cache_dir, revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None, ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    lm_head = mlm_model.lm_head
    for param in lm_head.parameters():
        param.requires_grad = False

    # --- Setup Adapters (LoRA/AdaLoRA) for Language Model ---
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

    # --- Update Language Model Config with Labels ---
    label_to_id = {v: i for i, v in enumerate(label_list)}
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}

    # --- Build Tokenization and Language Model Integration ---
    tklm = TkLM(tokenizer, model, lm_head, lm_tk, model_args.entropy)
    print("Trainable parameters:")
    for name, param in tklm.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.shape}")

    # --- Prepare compute_metrics function ---
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        labels = p.label_ids

        logits = torch.tensor(preds)
        labels = torch.tensor(labels)

        loss = F.cross_entropy(logits, labels, reduction="mean").item()

        pred_classes = logits.argmax(dim=-1)
        acc = (pred_classes == labels).float().mean().item()

        return {
            "accuracy": acc,
            "classification_loss": loss
        }

    return tklm, compute_metrics
