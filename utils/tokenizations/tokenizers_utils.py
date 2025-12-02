import copy
import tokenizers
from transformers import AutoTokenizer


def to_longest_prefix_tokenizer(tokenizer):
    print("Converting tokenizer to LP tokenizer...")
    # assumes tokenizer is byte converted
    lp_tokenizer = copy.deepcopy(tokenizer)
    unk_token = (
        lp_tokenizer.unk_token
        if lp_tokenizer.unk_token is not None
        else lp_tokenizer.eos_token
    )

    # use WordPiece without prefix to achieve longest-prefix tokenization
    lp_tokenizer._tokenizer.model = tokenizers.models.WordPiece(
        lp_tokenizer.get_vocab(), unk_token=unk_token
    )
    lp_tokenizer._tokenizer.model.continuing_subword_prefix = ""

    return lp_tokenizer


def tokenize(
    text: str, tokenizer, max_length: int = None, truncation: bool = False
) -> list:
    if max_length is None:
        return tokenizer.tokenize(text)
    else:
        return tokenizer.tokenize(text, max_length=max_length, truncation=truncation)


def pretokenize(text: str, tokenizer) -> list:
    if tokenizer._tokenizer.normalizer is not None:
        text = tokenizer._tokenizer.normalizer.normalize_str(text)

    return [x[0] for x in tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)]
