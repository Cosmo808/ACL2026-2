import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from spikingjelly.activation_based import functional

from utils.xnli_snn.neurons import MembraneLoss
from utils.xnli_snn.snn_tokenizer import SNNTokenizer
from utils.xnli_snn.arguments import ModelArguments, DataTrainingArguments


def setup_seed(seed: int) -> None:
    """Sets random seeds for reproducibility."""
    import random
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
    setup_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_args = ModelArguments(
        entropy=False
    )
    data_args = DataTrainingArguments(
        dataset_name="xnli",
        dataset_config_name="en",
        max_seq_length=256,
    )

    # Initialize SNN Tokenizer
    snn_tokenizer = SNNTokenizer(
        model_args.char_embed_dim, model_args.ann_hidden_dim, model_args.output_embed_dim,
        model_args.max_char_len, model_args.entropy
    )
    snn_tokenizer.to(device)
    snn_tokenizer_dict = torch.load(r"E:\ACL2026-2\utils\xnli_snn\snn_tokenizer_epoch6.pt", weights_only=False, map_location=device)
    snn_tokenizer.load_state_dict(snn_tokenizer_dict)

    # Initialize XLM-R Tokenizer (Ground Truth)
    lm_tokenizer = AutoTokenizer.from_pretrained(
        (model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision, use_auth_token=True if model_args.use_auth_token else None,
    )

    # Load dataset
    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    dataloader_dict = {}
    for split in ['train', 'validation', 'test']:
        if split in raw_datasets:
            dataloader_dict[split] = DataLoader(
                raw_datasets[split],
                batch_size=128,
            )

    # Optimizer
    optimizer = optim.Adam(snn_tokenizer.parameters(), lr=5e-4)
    snn_tokenizer.train()
    binary_loss = torch.nn.BCELoss()
    snn_loss = MembraneLoss()

    # Training Loop
    for epoch in range(10):
        total_losses = 0.0
        spike_acc, not_spike_acc = 0., 0.
        batch_count = 0
        for split in ['train', 'validation', 'test']:
            for batch in tqdm(dataloader_dict[split], desc=f"Epoch {epoch}-{split}"):
                optimizer.zero_grad()

                premise = batch['premise'] if isinstance(batch['premise'], list) else [batch['premise']]
                hypothesis = batch['hypothesis'] if isinstance(batch['hypothesis'], list) else [batch['hypothesis']]

                # Prepare input texts for SNN Tokenizer
                input_texts = ["<s>" + p + "</s>" + h + "</s>" for p, h in zip(premise, hypothesis)]

                # Run SNN Tokenizer
                inputs_embeds = snn_tokenizer(input_texts, use_hard_boundaries=False)

                # --- Calculate Ground Truth Boundaries using XLM-R Tokenizer ---
                gt_boundaries_list = []
                for text in input_texts:
                    tokens = lm_tokenizer.tokenize(text)
                    char_len = len(text.encode('utf-8'))
                    boundary = torch.zeros(char_len, dtype=torch.long)
                    pos = 0
                    for token in tokens:
                        token_str = token[1:] if token.startswith("▁") else token
                        try:
                            token_bytes = token_str.encode('utf-8')
                        except UnicodeEncodeError:
                            continue
                        if len(token_bytes) == 0:
                            continue
                        if pos < char_len:
                            boundary[pos] = 1
                        pos += len(token_bytes)

                    max_char_len = snn_tokenizer.max_char_len
                    if char_len < max_char_len:
                        boundary_pad = torch.zeros(max_char_len - char_len, dtype=torch.long)
                        boundary = torch.cat([boundary, boundary_pad])
                    else:
                        boundary = boundary[:max_char_len]
                    gt_boundaries_list.append(boundary)
                gt_boundaries = torch.stack(gt_boundaries_list).to(inputs_embeds.device)

                # --- Calculate SNN Loss ---
                gt_idx = [torch.where(gt_boundaries[b])[0] for b in range(gt_boundaries.shape[0])]
                gt_idx = [idx[1:] - 1 for idx in gt_idx]  # remove the first one; offset -1
                snn_loss_value, acc = snn_loss(snn_tokenizer.node.past_v, snn_tokenizer.I, gt_idx)

                # --- BP ---
                total_loss = snn_loss_value
                total_loss.backward()
                optimizer.step()

                total_losses += total_loss.item()
                spike_acc += acc['spike']
                not_spike_acc += acc['not_spike']
                batch_count += 1
                if batch_count % 100 == 0:
                    print(f"Acc: {acc['spike']:.4f} / {acc['not_spike']:.4f}")

                # Clean up SNN state
                functional.reset_net(snn_tokenizer)
                snn_tokenizer.I = []

        avg_loss = total_losses / batch_count
        avg_s_acc = spike_acc / batch_count
        avg_ns_acc = not_spike_acc / batch_count
        print(f"\nEpoch {epoch}, Average Loss: {avg_loss:.6f}, Spike Acc: {avg_s_acc:.4f}, Not Spike Acc: {avg_ns_acc:.4f}")

        # Save the trained SNN Tokenizer
        save_file = rf"E:\ACL2026-2\utils\xnli_snn\snn_tokenizer_epoch{epoch}.pt"
        torch.save(snn_tokenizer.state_dict(), save_file)
        print(f"Model saved to {save_file}")