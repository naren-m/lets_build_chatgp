## Tokenization

Tokenization is the process of **converting raw text strings into a sequence of integers** according to a specific vocabulary of possible elements. This allows a neural network to process the data, as it can only work with numerical values rather than raw characters or words.

According to the sources, there are several different strategies for tokenization, ranging from simple character-level mapping to complex subword units:

* **Character-Level Tokenization:** This is the simplest strategy where **each individual character is translated into a unique integer**. For example, in a model trained on Shakespeare, the vocabulary might consist of only 65 characters (including spaces, punctuation, and uppercase/lowercase letters). While this approach keeps the "codebook" or vocabulary size small and the encoding/decoding functions simple, it results in **very long sequences of integers** for the model to process.
* **Subword Tokenization:** This is the strategy typically used in production-grade systems like GPT. Instead of individual characters, the text is broken into **"chunks" of words or subword pieces**.
  * **Byte Pair Encoding (BPE):** This is the specific subword schema used by OpenAI via their library **tiktoken**.
  * **SentencePiece:** A similar subword-level tokenizer developed by Google.
* **Word-Level Tokenization:** In this approach, entire words (e.g., "Hello," "World") are mapped directly to integers.

### The Vocabulary-Sequence Tradeoff

The sources highlight a fundamental tradeoff between the **vocabulary size** and the **sequence length**.

* A **character-level tokenizer** has a tiny vocabulary (e.g., 65 elements) but generates very long sequences because every character needs its own integer.
* A **subword tokenizer** (like GPT-2’s BPE) has a much larger vocabulary (e.g., 50,257 elements) but generates much shorter sequences. For instance, the string "hi there" might be two or three subword tokens but eight individual characters.

### Implementation in the Sources

To implement a basic character-level tokenizer, the strategy follows these steps:

1. **Identify the Vocabulary:** Extract all unique characters from the training data and sort them.
2. **Create Mapping Tables:** Build two lookup tables: an **encoder** that maps characters to integers and a **decoder** that maps those integers back to characters.
3. **Data Transformation:** The entire text is then "stretched out" into a massive sequence of integers (represented as a tensor in PyTorch) to be fed into the Transformer.

**Analogy:**
Think of tokenization like a construction kit. A **character-level tokenizer** is like building a castle using only individual LEGO dots; it's simple to understand, but you need thousands of pieces to finish the wall. A **subword tokenizer** is like using pre-molded wall sections; the kit is more complex (a larger vocabulary), but you can build the same castle much faster with far fewer pieces.
