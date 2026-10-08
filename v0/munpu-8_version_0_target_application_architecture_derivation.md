# µNPU-8 — Version_0
## Target Application and Architecture Derivation

**Status:** Version 0 — Architecture Baseline  
**Project:** µNPU-8  
**Focus:** µROM-controlled INT8 GEMV accelerator for LLM autoregressive decode

**GEMM - GEneral Matrix-Matrix** multiplication ; **GEMV - GEneral Matrix Vector** multiplication
---

**Important Terminology** :
1. **Token** - A basic unit of text represented and processed by an LLM. The tokenizer converts text into tokens (and token IDs), while the LLM predicts the next token during generation.

Example

Text:
"What is a CPU?"
       ↓
["What", " is", " a", " CPU", "?"]
       ↓
[1254, 42, 17, 8392, 30]     ← token IDs
       ↓
[vector, vector, vector, ...] ← embeddings
       ↓
LLM computation

2. **Token ID** - An integer identifier assigned to a token. The embedding table consists of rows of vectors, with each row indexed/referenced by a specific token ID. Each vector is an array of learned numerical values (also called **model parameters**). In a typical LLM, these values are represented using floating-point formats during training and in the original/full-precision model.

Token:
"CPU"

↓ tokenizer

Token ID:
8392

↓ use ID as index

Embedding Table
┌───────┬──────────────────────────────┐
│ ID    │ Embedding Vector             │
├───────┼──────────────────────────────┤
│ 8390  │ [ ... ]                      │
│ 8391  │ [ ... ]                      │
│ 8392  │ [0.12, -0.31, 0.87, ...]    │ ← selected
│ 8393  │ [ ... ]                      │
└───────┴──────────────────────────────┘
                    ↓
             Embedding Vector

3. **Embedding Vector** - A array of learned numerical parameters associated with a token. These values are initialized and then progressively updated through training so that the resulting representation works effectively with the rest of the neural network.

For example, conceptually:
**Assume CPU is the token**
Initial training:
CPU → [0.12, -0.45, 0.31, ...]

        ↓ training update

CPU → [0.14, -0.42, 0.35, ...]

        ↓ another update

CPU → [0.21, -0.37, 0.41, ...]

        ↓ many training iterations

CPU → [0.72, -0.15, 0.63, ...]

**The embedding vector is updated during training and fixed for inference**.

