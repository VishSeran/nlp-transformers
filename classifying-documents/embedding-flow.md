# 🧠 Understanding Embeddings in PyTorch

A beginner-friendly guide to `nn.Embedding` and `nn.EmbeddingBag` for NLP tasks.

---

## 📌 Table of Contents
- [The Problem: Computers Can't Read Words](#the-problem-computers-cant-read-words)
- [Step 1: Tokenization — Words to Numbers](#step-1-tokenization--words-to-numbers)
- [Step 2: nn.Embedding — Each Word Gets a Vector](#step-2-nnembedding--each-word-gets-a-vector)
- [Step 3: Embedding a Full Sentence](#step-3-embedding-a-full-sentence)
- [Step 4: nn.EmbeddingBag — Embedding + Averaging](#step-4-nnembeddingbag--embedding--averaging)
- [Step 5: Why Offsets? Handling a Batch](#step-5-why-offsets-handling-a-batch)
- [Full Pipeline](#full-pipeline)
- [Key Differences](#key-differences-nnembedding-vs-nnembeddingbag)

---

## The Problem: Computers Can't Read Words

Neural networks work with numbers — not words. So we need to convert text into numbers first.

```python
# We can't feed raw text into a neural network
text = "cat sat on mat"  # ❌ neural network can't process this

# First we convert words to numbers (token indices)
text = [4, 12, 6, 9]    # ✅ each word mapped to an index in vocab
```

---

## Step 1: Tokenization — Words to Numbers

Each word in the vocabulary is assigned a unique integer index.

```
Vocabulary:
  "cat" → 4
  "sat" → 12
  "on"  → 6
  "mat" → 9

Sentence: "cat sat on mat" → [4, 12, 6, 9]
```

---

## Step 2: `nn.Embedding` — Each Word Gets a Vector

An embedding is a **lookup table** — each word index maps to a vector of numbers.

```python
import torch
import torch.nn as nn

# Vocab has 10 words, each represented as a 4-dimensional vector
embedding = nn.Embedding(num_embeddings=10, embedding_dim=4)

# Internally it's just a lookup TABLE:
#
# word index │  vector
# ───────────┼─────────────────────────────
#     0      │ [ 0.1,  0.2, -0.1,  0.4]
#     1      │ [ 0.5, -0.3,  0.2,  0.1]
#     2      │ [-0.2,  0.4,  0.1, -0.3]
#    ...     │  ...
#     9      │ [ 0.3,  0.1, -0.4,  0.2]

word = torch.tensor([4])       # word index 4
print(embedding(word))
# tensor([[ 0.1, -0.2,  0.3,  0.5]])  ← shape [1, 4]
```

> 💡 These vectors are **randomly initialized** and **learned during training**.
> Similar words (e.g., "king" and "queen") end up with similar vectors
> because they appear in similar contexts.

---

## Step 3: Embedding a Full Sentence

```python
sentence = torch.tensor([4, 12, 6, 9])   # "cat sat on mat"
print(embedding(sentence))

# tensor([[ 0.1, -0.2,  0.3,  0.5],     ← cat
#         [ 0.2,  0.1, -0.1,  0.4],     ← sat
#         [-0.3,  0.5,  0.2, -0.2],     ← on
#         [ 0.1,  0.3, -0.4,  0.1]])    ← mat
#
# shape: [4, 4] → 4 words, each with a 4-dimensional vector
```

---

## Step 4: `nn.EmbeddingBag` — Embedding + Averaging

For **text classification**, we don't need individual word vectors.
We just need **one vector** representing the **whole sentence**.

`EmbeddingBag` does embedding **+** averaging in a single step.

```python
embedding_bag = nn.EmbeddingBag(num_embeddings=10, embedding_dim=4, mode="mean")

sentence = torch.tensor([4, 12, 6, 9])  # "cat sat on mat"
offsets  = torch.tensor([0])            # one sentence starting at index 0

output = embedding_bag(sentence, offsets)

# Internally:
#   cat → [ 0.1, -0.2,  0.3,  0.5]
#   sat → [ 0.2,  0.1, -0.1,  0.4]
#   on  → [-0.3,  0.5,  0.2, -0.2]
#   mat → [ 0.1,  0.3, -0.4,  0.1]
#           ↓      average all      ↓
# output → [ 0.025, 0.175, 0.0, 0.2]   ← ONE vector for whole sentence
# shape: [1, 4]
```

---

## Step 5: Why Offsets? Handling a Batch

When processing multiple sentences, they are **concatenated into one tensor**.
`offsets` tells `EmbeddingBag` **where each sentence starts**.

```python
# Multiple sentences concatenated into ONE tensor
texts   = torch.tensor([4, 12, 6, 9,    # sentence 1: "cat sat on mat"
                         3,  7,          # sentence 2: "dog ran"
                         1,  5,  8])     # sentence 3: "bird flew high"

offsets = torch.tensor([0, 4, 6])
#                        ↑  ↑  ↑
#                        │  │  └── sentence 3 starts at index 6
#                        │  └───── sentence 2 starts at index 4
#                        └──────── sentence 1 starts at index 0

output = embedding_bag(texts, offsets)
# shape: [3, 4] → one averaged vector per sentence
```

---

## Full Pipeline

```
"cat sat on mat"
       ↓  tokenize
  [4, 12, 6, 9]
       ↓  nn.Embedding (lookup table)
  [vector, vector, vector, vector]
       ↓  average (EmbeddingBag)
  [one vector representing whole sentence]
       ↓  nn.Linear
  [score1, score2, score3, score4]   ← 4 class scores
       ↓  argmax
  predicted class (e.g. "Sports")
```

---

## Full Model Example (AG_NEWS Classification)

```python
from torch import nn

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)  # [batch_size, embed_dim]
        return self.fc(embedded)                  # [batch_size, num_class]

# Initialize
vocab_size = len(vocab)   # e.g. 95811
embed_dim  = 64
num_class  = 4            # AG_NEWS: World, Sports, Business, Sci/Tech

model = TextClassificationModel(vocab_size, embed_dim, num_class)
```

---

## Key Differences: `nn.Embedding` vs `nn.EmbeddingBag`

| Feature | `nn.Embedding` | `nn.EmbeddingBag` |
|---|---|---|
| Output | One vector **per word** | One vector **per sentence** |
| Output shape | `[seq_len, embed_dim]` | `[batch_size, embed_dim]` |
| Needs offsets? | ❌ No | ✅ Yes |
| Use case | RNN, LSTM, Transformer | Text classification |
| Memory usage | Higher (stores all word vectors) | Lower (aggregates internally) |

---

## AG_NEWS Label Mapping

```
Raw label → After label_pipeline (label - 1)
    1      →  0  (World)
    2      →  1  (Sports)
    3      →  2  (Business)
    4      →  3  (Sci/Tech)
```

> PyTorch's `CrossEntropyLoss` expects labels starting from `0`, so the `-1` shift is necessary.

---

## References

- [PyTorch `nn.Embedding` docs](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- [PyTorch `nn.EmbeddingBag` docs](https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html)
- [torchtext AG_NEWS tutorial](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)
