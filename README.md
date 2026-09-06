# lm-from-scratch

A language model built from scratch, following the arc of Stanford's
[CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/) and its
Assignment 1. The premise of the course is that you understand a system by building
every piece of it yourself, so nothing here is imported from a library beyond raw
tensors: the byte-level BPE tokenizer (trained on TinyStories and OpenWebText), every
layer of the decoder-only Transformer — linear, embedding, RMSNorm, SwiGLU, RoPE,
scaled dot-product attention, multi-head attention, the pre-norm block, the full LM,
the cross-entropy loss — and the training machinery around it: AdamW, the cosine
schedule with warmup, gradient clipping, the random-window data loader, and
checkpointing. Everything is written directly on `torch.nn.Module` and `torch.optim.Optimizer`,
with no `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.functional`, or `torch.optim.AdamW`
shortcuts, and each piece is checked against the assignment's reference tests.

<!-- TODO: lead with a generated TinyStories sample and the training loss curve
     once train.py exists. A reviewer should see results before code. -->

## Results

### Tokenizer

Byte-level BPE with GPT-2 pre-tokenization, parallel pre-tokenization across
document boundaries, and incremental pair-count updates during merging.

| corpus | tokenizer | bytes/token | tokens/pretoken |
|---|---|---:|---:|
| TinyStories | TinyStories 10K | 4.03 | 1.003 |
| OpenWebText | OpenWebText 32K | 4.25 | 1.093 |
| OpenWebText | TinyStories 10K | 3.17 | 1.467 |
| TinyStories | OpenWebText 32K | 3.91 | 1.033 |

Cross-tokenizing degrades compression asymmetrically: a broad vocabulary degrades
gracefully on narrow text (−3%), a narrow one degrades sharply on broad text (−25%).
Full write-up: [`experiments/TOKENIZER_EXPERIMENTS.md`](experiments/TOKENIZER_EXPERIMENTS.md).

### Model

<!-- TODO: parameter count, training config, loss curve, perplexity, sample text. -->

## Architecture

<!-- TODO: the module map — every component in order with tensor shapes and the
     governing equation. Source of truth for the from-memory rebuilds. -->

```
tokens (B, T)
  └─ Embedding                       (B, T, d_model)
  └─ × num_layers Transformer_block
       ├─ RMSNorm → MultiheadSelfAttention (+RoPE on Q, K) → residual
       └─ RMSNorm → SwiGLU FFN                              → residual
  └─ RMSNorm
  └─ Linear (LM head)                (B, T, vocab_size)
```

Conventions worth knowing: linear weights are stored `(in, out)` and applied as
`x @ W` (no transpose in forward); RoPE pairs adjacent components; the causal mask
is built per call from the sequence length.

## Layout

```
tokenizer/    bpe.py, bpe_multiprocessing.py — training;  tokenizer.py — encode/decode
model/        model.py — all modules + softmax, attention, cross-entropy
training/     optim.py — AdamW;  schedule.py — cosine LR schedule with warmup;  clip.py — gradient clipping;  data.py — random-window batch loader;  checkpoint.py — save/load
experiments/  tokenizer experiments, arXiv corpus builder, corpus → uint16 token-ID encoding, and the trained vocabularies
data/         corpora and encoded token IDs (gitignored): TinyStories valid .txt/.npy, sample_50MB.txt
train.py      training loop: memmap data, warmup+cosine LR, clipping, checkpoints, CSV logging
tests/        pytest suite for the training utilities (mirrors the CS336 checks)
```

## Running

```bash
pip install -r requirements.txt
python experiments/tokenizer_experiments.py       # reads data/ (TinyStories + OWT valid files)
python experiments/encode_datasets.py               # TinyStories valid, 50 MB sample, train → uint16 .npy (skips existing)
python experiments/fetch_arxiv.py                   # arXiv hep-ph abstracts → data/arxiv_hepph_{train,valid}.txt in TinyStories layout
python -m pytest tests/                            # schedule, clipping, data loader, checkpointing
```

```bash
python train.py Nrun=5000 Batch_size=32 Device=mps     # any setting at the top of train.py as key=value
python train.py LoadCkpt=True                            # resume from the checkpoint, append to the CSV log
```

Training writes `checkpoints/<name>.pt` (model + optimizer + step, resumable) and
`logs/<name>.csv` (step, train loss, val loss, lr, wall-clock) — both gitignored.

## Provenance

The structure and the problem sequence follow Stanford CS336 Assignment 1
(spring 2026 handout); the code is my own. The assignment's test harness (28 tests:
tokenizer, BPE training, every model component, AdamW, LR schedule, gradient
clipping, data loading, checkpointing) passes against this code. The assignment
repo itself is not included here — only the implementations.
