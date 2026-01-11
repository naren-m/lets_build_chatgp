# =============================================================================
# GPT FROM SCRATCH - A Step-by-Step Implementation
# =============================================================================
# This file implements a character-level GPT (Generative Pre-trained Transformer)
# following Andrej Karpathy's "Let's build GPT" tutorial.
#
# The goal: Train a neural network to generate Shakespeare-like text by learning
# patterns in how characters follow each other.
#
# Architecture Overview:
# ┌─────────────────────────────────────────────────────────────────┐
# │  Input Text  →  Tokenizer  →  Transformer  →  Predictions      │
# │  "Hello"     →  [7,4,11,11,14] → Neural Net → Next char probs  │
# └─────────────────────────────────────────────────────────────────┘
# =============================================================================

import torch

# =============================================================================
# STEP 1: LOAD THE DATASET
# =============================================================================
# We're loading the Tiny Shakespeare dataset - all of Shakespeare's works
# concatenated into a single text file (~1MB, ~1 million characters).
#
# Why this dataset?
# - Small enough to train quickly on a laptop
# - Rich patterns: dialogue structure, character names, poetic meter
# - Fun to see the model "speak" like Shakespeare!

with open('data/tinyshakespeare/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Length of dataset in characters: {len(text):,}")
print(f"First 100 characters:\n{text[:100]}")

# =============================================================================
# STEP 2: BUILD THE VOCABULARY
# =============================================================================
# A vocabulary is the complete set of unique "tokens" our model can understand.
#
# KEY CONCEPT: Character-Level vs Subword-Level Tokenization
# ┌────────────────────────────────────────────────────────────────────────┐
# │ Approach        │ Vocabulary Size │ Sequence Length │ Example          │
# ├────────────────────────────────────────────────────────────────────────┤
# │ Character-level │ ~65 (a-z, A-Z..)│ Long            │ "cat" → [c,a,t]  │
# │ Subword (BPE)   │ ~50,000         │ Short           │ "cat" → [cat]    │
# │ Word-level      │ ~100,000+       │ Very short      │ "cat" → [cat]    │
# └────────────────────────────────────────────────────────────────────────┘
#
# We use character-level because:
# ✓ Simpler to implement (no complex tokenization algorithms)
# ✓ Tiny vocabulary (65 chars vs 50,000+ subwords)
# ✓ Can generate ANY text, including made-up words or typos
# ✗ Trade-off: Longer sequences make it harder to learn long-range patterns

# Extract all unique characters and sort them for reproducibility
chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f"\nAll unique characters ({vocab_size} total):")
print(''.join(chars))
print(f"\nThis includes: letters, punctuation, newlines, and spaces")

# =============================================================================
# STEP 3: CREATE THE TOKENIZER (Encoder + Decoder)
# =============================================================================
# Neural networks only understand numbers, not text. The tokenizer bridges this gap:
#
#   ENCODE: "Hi" → [20, 47]     (text to numbers, for model input)
#   DECODE: [20, 47] → "Hi"     (numbers to text, to read model output)
#
# How it works:
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  Character:  ' ' '!' '"' ... 'A' 'B' 'C' ... 'a' 'b' 'c' ... 'z'       │
# │  Index:       0   1   2  ...  13  14  15 ...  39  40  41 ...  64       │
# └─────────────────────────────────────────────────────────────────────────┘
#
# This is the SIMPLEST possible tokenizer. Production systems like GPT use
# Byte Pair Encoding (BPE) which groups common character sequences into tokens.
# Example: "the" might be a single token instead of ['t', 'h', 'e']

# String-to-Integer mapping: character → index
stoi = {ch: i for i, ch in enumerate(chars)}

# Integer-to-String mapping: index → character (the reverse lookup)
itos = {i: ch for i, ch in enumerate(chars)}

# Encoder: converts a string into a list of integers
# Example: encode("hi") → [46, 47] (assuming 'h'=46, 'i'=47)
encode = lambda s: [stoi[c] for c in s]

# Decoder: converts a list of integers back into a string
# Example: decode([46, 47]) → "hi"
decode = lambda l: ''.join([itos[i] for i in l])

# =============================================================================
# STEP 4: VERIFY THE TOKENIZER
# =============================================================================
# Golden rule: decode(encode(text)) should EXACTLY equal the original text.
# If not, you've lost information and your model can't learn properly!

test_string = "hii there"
encoded = encode(test_string)
decoded = decode(encoded)

print(f"\n--- Tokenizer Test ---")
print(f"Original:  '{test_string}'")
print(f"Encoded:   {encoded}")
print(f"Decoded:   '{decoded}'")
print(f"Roundtrip: {'PASS' if decoded == test_string else 'FAIL'}")

# =============================================================================
# STEP 5: CONVERT ENTIRE DATASET TO TENSOR
# =============================================================================
# Now we tokenize ALL of Shakespeare into one giant tensor of integers.
#
# Why a PyTorch tensor instead of a Python list?
# ┌──────────────────────────────────────────────────────────────────────┐
# │ Python List                    │ PyTorch Tensor                     │
# ├──────────────────────────────────────────────────────────────────────┤
# │ Scattered memory locations     │ Contiguous memory (cache-friendly) │
# │ CPU only                       │ GPU acceleration supported         │
# │ Slow element-by-element ops    │ Fast vectorized operations         │
# │ Can't do neural network ops    │ Built for deep learning            │
# └──────────────────────────────────────────────────────────────────────┘
#
# dtype=torch.long (int64) is the standard for token indices in PyTorch.
# This is required because embedding layers expect integer indices.

data = torch.tensor(encode(text), dtype=torch.long)

print(f"\n--- Dataset Tensor ---")
print(f"Shape: {data.shape}  (one long sequence of {len(data):,} tokens)")
print(f"Dtype: {data.dtype}")
print(f"First 20 tokens: {data[:20].tolist()}")
print(f"Decoded back:    '{decode(data[:20].tolist())}'")

# =============================================================================
# STEP 6: TRAIN/VALIDATION SPLIT
# =============================================================================
# We split the data so we can detect OVERFITTING:
# - Training set (90%): The model learns patterns from this
# - Validation set (10%): We test generalization on text the model hasn't seen
#
# If train_loss keeps dropping but val_loss stops improving (or gets worse),
# the model is MEMORIZING the training data instead of learning general patterns.
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ Shakespeare text:                                                       │
# │ [====================Training Data (90%)====================][Val 10%]  │ 
# └─────────────────────────────────────────────────────────────────────────┘

n = int(0.9 * len(data))  # Note: int(0.9 * len(data)), NOT int(0.9) * len(data)
train_data = data[:n]     # First 90% for training
val_data = data[n:]       # Last 10% for validation

print(f"\n--- Train/Val Split ---")
print(f"Training tokens:   {len(train_data):,} ({len(train_data)/len(data)*100:.1f}%)")
print(f"Validation tokens: {len(val_data):,} ({len(val_data)/len(data)*100:.1f}%)")

# =============================================================================
# STEP 7: UNDERSTANDING CONTEXT AND PREDICTION
# =============================================================================
# Language modeling = predicting the next token given previous tokens.
#
# BLOCK SIZE (context length) = how many previous tokens the model can "see"
# when making a prediction. GPT-3 uses 2048, GPT-4 uses 8192+, we start with 8.
#
# Example with block_size=8:
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ If we have the sequence: "First Cit" (9 chars)                          │
# │                                                                         │
# │ We get 8 training examples from this one chunk:                         │
# │   Context: "F"           → Target: "i"                                  │
# │   Context: "Fi"          → Target: "r"                                  │
# │   Context: "Fir"         → Target: "s"                                  │
# │   Context: "Firs"        → Target: "t"                                  │
# │   Context: "First"       → Target: " "                                  │
# │   Context: "First "      → Target: "C"                                  │
# │   Context: "First C"     → Target: "i"                                  │
# │   Context: "First Ci"    → Target: "t"                                  │
# └─────────────────────────────────────────────────────────────────────────┘
#
# WHY train on all context lengths (1 to block_size)?
# → At generation time, we might start with just 1 character
# → The model needs to handle any context length up to block_size

block_size = 8  # Maximum context length for predictions

print(f"\n--- Context Window Demo (block_size={block_size}) ---")

# TODO: Not clear on why we take block_size + 1 here
sample = train_data[:block_size + 1]  # +1 because we need targets too

print(f"Sample chunk: {sample.tolist()}")
print(f"As text: '{decode(sample.tolist())}'")
print(f"\nTraining examples extracted from this chunk:")

x = train_data[:block_size]     # Inputs (first 8 tokens)
y = train_data[1:block_size+1]  # Targets (offset by 1) from each position in the inputs

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"  Context: {str(context.tolist()):20} → Target: {target.item()} ('{itos[target.item()]}')")

# =============================================================================
# STEP 8: BATCHING FOR EFFICIENT TRAINING
# =============================================================================
# GPUs are massively parallel - they're inefficient processing one example at a time.
# BATCHING = processing multiple independent sequences simultaneously.
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ Without batching (slow):     │ With batching (fast):                    │
# │   Process sequence 1         │   Process sequences 1,2,3,4 in parallel  │
# │   Process sequence 2         │   (GPU does all 4 at once!)              │
# │   Process sequence 3         │                                          │
# │   Process sequence 4         │                                          │
# └─────────────────────────────────────────────────────────────────────────┘
#
# batch_size = how many independent sequences we process per forward pass
# Each sequence in a batch is COMPLETELY INDEPENDENT (no information sharing)

batch_size = 4  # Number of independent sequences per batch

torch.manual_seed(1337)  # For reproducibility

def get_batch(split):
    """
    Generate a batch of training examples.

    Returns:
        x: Input tensor of shape (batch_size, block_size)
        y: Target tensor of shape (batch_size, block_size)

    Each row in x is an independent sequence of tokens.
    Each row in y contains the "next token" targets for the corresponding x row.
    """
    data_split = train_data if split == 'train' else val_data

    # Pick random starting positions for each sequence in the batch
    # We subtract block_size to ensure we don't run off the end
    ix = torch.randint(len(data_split) - block_size, (batch_size,))

    # Stack sequences into a 2D tensor: (batch_size, block_size)
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])

    return x, y

# Demo the batching
xb, yb = get_batch('train')
print(f"\n--- Batch Demo ---")
print(f"Input shape:  {xb.shape}  (batch_size x block_size)")
print(f"Target shape: {yb.shape}")
print(f"\nBatch contains {batch_size} independent sequences of {block_size} tokens each")
print(f"Total training examples per batch: {batch_size * block_size}")

print(f"\n--- Detailed view of batch ---")
for b in range(batch_size):
    print(f"\nSequence {b}:")
    print(f"  Input:  {xb[b].tolist()}  ->  '{decode(xb[b].tolist())}'")
    print(f"  Target: {yb[b].tolist()}  ->  '{decode(yb[b].tolist())}'")



import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        logits = self.token_embedding_table(idx)  # (B, T, C)

        return logits
    

# Instantiate the model
model = BigramLanguageModel(vocab_size)
print(f"\n--- Model Summary ---")
print(model)
out = model(xb, yb)  # Forward pass to check everything works
print(f"Output shape from model: {out.shape} (should be (batch_size, block_size, vocab_size))")

# =============================================================================
# WHAT'S NEXT?
# =============================================================================
# With data preparation complete, the next steps are:
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ [ ] Step 9:  Build the Bigram model (simplest baseline)                 │
# │ [ ] Step 10: Add self-attention (tokens communicate with each other)    │
# │ [ ] Step 11: Add multi-head attention (parallel attention streams)      │
# │ [ ] Step 12: Add feed-forward layers (per-token computation)            │
# │ [ ] Step 13: Stack into Transformer blocks with residual connections    │
# │ [ ] Step 14: Add layer normalization                                    │
# │ [ ] Step 15: Scale up and train!                                        │
# │ [ ] Step 16: Generate Shakespeare!                                      │
# └─────────────────────────────────────────────────────────────────────────┘
#
# The full architecture we're building:
#
#   Input Tokens
#        |
#        v
#   Token Embeddings + Position Embeddings
#        |
#        v
#   ┌──────────────────────────────────┐
#   │  Transformer Block (xN layers)   │
#   │  ┌────────────────────────────┐  │
#   │  │ Multi-Head Self-Attention  │  │
#   │  └────────────────────────────┘  │
#   │           | (+ residual)         │
#   │           v                      │
#   │  ┌────────────────────────────┐  │
#   │  │ Feed-Forward Network       │  │
#   │  └────────────────────────────┘  │
#   │           | (+ residual)         │
#   └──────────────────────────────────┘
#        |
#        v
#   Linear Layer -> Vocabulary Logits
#        |
#        v
#   Softmax -> Next Token Probabilities
