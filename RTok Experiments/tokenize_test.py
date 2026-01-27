import torch
from transformers import AutoTokenizer
from utils.xnli_snn.snn_tokenizer import SNNTokenizer


text = "I study at a German university. Since I love chemistry, I work on germanium."

print("Input text:")
print(text)
print("\n" + "="*60 + "\n")

# --------------------------------------------------
# 1. BPE Tokenizer (Mistral-7B)
# --------------------------------------------------
print("1. BPE Tokenizer (Mistral-7B):")
bpe_tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
)
bpe_tokens = bpe_tokenizer.tokenize(text)
print("Tokens:", bpe_tokens)
print("\n" + "-"*60 + "\n")

# --------------------------------------------------
# 2. Unigram Tokenizer (XLM-R)
# --------------------------------------------------
print("2. Unigram Tokenizer (XLM-R):")
unigram_tokenizer = AutoTokenizer.from_pretrained(
    "xlm-roberta-base",
)
unigram_tokens = unigram_tokenizer.tokenize(text)
print("Tokens:", unigram_tokens)
print("\n" + "-"*60 + "\n")

# --------------------------------------------------
# 3. Ours
# --------------------------------------------------
print("3. Our:")
tokenizer = SNNTokenizer(159, 128, 768, False)
snn_tokenizer_dict = torch.load("./utils/xnli_snn/a1.2_b0.6_houglass_128.pt", map_location="cuda", weights_only=False)
tokenizer.load_state_dict(snn_tokenizer_dict)
vocab = torch.load('./utils/xnli_snn/vocab.pt', weights_only=False)
input_embs = vocab.convert_to_tensor(text).unsqueeze(0).to('cuda')
_ = tokenizer(input_embs)
hard_boundaries = tokenizer.hard_boundaries[0]


chars = list(text)
T = len(chars)
boundaries = hard_boundaries[:T]

splits = [0]
for i in range(T):
    if boundaries[i].item() == 1:
        splits.append(i + 1)
splits.append(T)

tokens = []
for i in range(len(splits) - 1):
    start, end = splits[i], splits[i + 1]
    if start < end:
        tokens.append(''.join(chars[start:end]))

print("Tokens:", tokens)



