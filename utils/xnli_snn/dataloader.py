import logging
import random
import os
import torch
from datasets import load_dataset
from collections import OrderedDict, Counter

logger = logging.getLogger(__name__)


class Vocab(object):
    def __init__(self, max_len):
        self.counter = Counter()
        self.idx2sym = []
        self.sym2idx = OrderedDict()
        self.max_len = max_len

    def build_vocab(self):
        for sym, cnt in self.counter.most_common():
            if sym not in self.sym2idx:
                self.idx2sym.append(sym)
                self.sym2idx[sym] = len(self.idx2sym) - 1

    def convert_to_tensor(self, symbols):
        indices = []
        for sym in symbols:
            assert sym in self.sym2idx
            idx = self.sym2idx[sym]
            indices.append(idx)
        output = self.pad_or_truncate(torch.LongTensor(indices))
        return output

    def pad_or_truncate(self, tensor):
        pad_id = self.sym2idx[' ']
        if tensor.size(0) >= self.max_len:
            return tensor[:self.max_len]
        else:
            pad_length = self.max_len - tensor.size(0)
            padding = torch.full((pad_length,), pad_id, dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, padding], dim=0)

    def __len__(self):
        return len(self.idx2sym)


def load_raw_datasets(data_args, model_args):
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

    # --- Vocab can convert to tensors ---
    vocab = Vocab(model_args.max_char_len)
    for split in ['train', 'validation', 'test']:
        data_split = raw_datasets[split]
        for data in data_split:
            input_texts = data['premise'] + ' ' + data['hypothesis']
            vocab.counter.update(input_texts)
    vocab.build_vocab()

    return raw_datasets, is_regression, label_list, num_labels, vocab
