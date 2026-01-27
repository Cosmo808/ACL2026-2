from collections import Counter
import sys

from homoglyphs import normalise_homoglyphs
from alphabet_numerals import spellout_digits, keep_whitelist
from utils import (
    wiki40b_markers,
    change_unknowns,
    collapse_whitespace,
    collapse_unknowns,
)


def text8_cleaner(text, lang):
    text = wiki40b_markers(text, mode='remove')

    if not lang.startswith('he'):
        text = normalise_homoglyphs(text)

    text = text.lower()

    text = spellout_digits(text, lang)
    text = keep_whitelist(text, lang)

    text = collapse_whitespace(text)
    text = text.strip()

    return text


def soft_cleaner(text, lang, threshold=5, valid_test_size=int(5e6)):
    text = wiki40b_markers(text, mode='keep')
    text = text.lower()

    # This cleaner has to be applied to concatenated train/valid/test
    # Apart from replacing least frequent symbols with \unk we also want
    # to change to unks all symbols that are not in train but in valid/test.
    trainset = text[:2 * valid_test_size]
    train_counter = Counter(trainset)
    allowed_symbols = set([k for k, v in train_counter.items() if v > threshold])
    all_symbols = set(text)
    disallowed_symbols = all_symbols - allowed_symbols
    text = change_unknowns(text, disallowed_symbols=disallowed_symbols)
    text = collapse_unknowns(text)

    text = collapse_whitespace(text)
    text = text.strip()
    return text


if __name__ == "__main__":
    import os

    langs = ['fi', 'he', 'vi']
    splits = ['train', 'validation', 'test']

    for lang in langs:
        for split in splits:
            input_path = f'E:/ACL2026/data/wiki40b/{lang}/{split}.txt'
            output_dir = f'E:/ACL2026/data/wiki40b/{lang}/text8'
            os.makedirs(output_dir, exist_ok=True)
            output_path = f'{output_dir}/{split}.txt'

            with open(input_path, encoding='utf-8') as file:
                text = file.read()

            text = text8_cleaner(text, lang)

            with open(output_path, 'w', encoding='utf-8') as out_file:
                out_file.write(text)

            print(f"Language {lang} split {split} finished...")

        val_src = f'./data/wiki40b/{lang}/text8/validation.txt'
        val_dst = f'./data/wiki40b/{lang}/text8/valid.txt'
        if os.path.exists(val_src):
            os.rename(val_src, val_dst)
