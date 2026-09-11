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

22.7M parameters: d_model 512, 4 layers, 16 heads, SwiGLU d_ff 1344, context 256,
vocabulary 10,000, RoPE θ = 10,000. Trained with AdamW (β = 0.9/0.999, weight decay
0.01), a cosine schedule with 100 steps of warmup decaying to a 1e-4 floor, and
gradient clipping at global norm 1.0.

Best validation loss **1.615** — perplexity 5.03, or 2.33 bits per token — after 5,000
steps at batch 32 (41M tokens) on TinyStories, about 24 minutes on one A100 MIG slice
(3g.20gb). Training and validation loss track each other throughout; no overfitting at
this budget.

The peak learning rate was chosen by a four-point sweep at the same budget:

| peak lr | best val loss |
|---|---:|
| 3e-4 | 1.755 |
| **1e-3** | **1.615** |
| 3e-3 | 2.100 |
| 1e-2 | 2.665 |

![learning curves and sweep curve](experiments/lr_sweep.png)

Two honest caveats. Nothing diverged — even 1e-2 descended monotonically, just to a
worse floor, so this sweep brackets the optimum without reaching the instability cliff.
And 3e-4 was still descending at step 5,000 while 1e-3 had flattened, so the ordering
between those two could change at a larger token budget.

<!-- TODO: a generated sample, once §6 decoding exists. -->

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
generate.py   sampling from a checkpoint: temperature, top-p (nucleus), <|endoftext|> stop
sweep.sh      learning-rate sweep launcher (sequential, one GPU)
sweep_batch.sh  batch-size sweep launcher (fixed step budget, one GPU)
ablate.sh     architecture ablations: no-norm, post-norm, NoPE, SiLU-FFN
tests/        pytest suite for the training utilities (mirrors the CS336 checks)
drills/       timed from-memory rebuilds of the whole model (practice notebooks, not library code)
```

## Running

```bash
pip install -r requirements.txt
python experiments/tokenizer_experiments.py       # reads data/ (TinyStories + OWT valid files)
python experiments/encode_datasets.py               # TinyStories valid, 50 MB sample, train → uint16 .npy (skips existing)
python experiments/fetch_arxiv.py                   # arXiv hep-ph abstracts → data/arxiv_hepph_{train,valid}.txt in TinyStories layout
python experiments/fetch_arxiv.py --year 2019       # one submission year (the API paginates to 10k results per query)
python experiments/train_arxiv_bpe.py               # BPE over the arXiv abstracts → experiments/arxiv_hepph_vocab10000.json
python -m pytest tests/                            # schedule, clipping, data loader, checkpointing
```

```bash
python train.py Nrun=5000 Batch_size=32 Device=mps     # any setting at the top of train.py as key=value
python train.py LoadCkpt=True                            # resume from the checkpoint, append to the CSV log
python generate.py Prompt="Once upon a time" Temp=0.7 Topp=0.9   # sample from checkpoints/baseline.pt
bash sweep.sh probe ; bash sweep.sh                      # LR sweep on one GPU (see the header for the SWAN recipe)
python experiments/plot_sweep.py                         # logs/lr*.csv → experiments/lr_sweep.png
LRS="2e-2 5e-2 1e-1 2e-1" LOGDIR=logs_edge bash sweep.sh   # edge of stability: push lr until a run diverges (CLIP=1e9 to see it undamped)
bash sweep_batch.sh ; python experiments/plot_batch.py    # batch 1..256 at fixed lr → batch_sweep.png (steps / tokens / wall-clock)
bash ablate.sh ; python experiments/plot_ablate.py         # the four §7.3 ablations vs the baseline
```

Training writes `checkpoints/<name>.pt` (model + optimizer + step, resumable) and
`logs/<name>.csv` (step, train loss, val loss, lr, wall-clock) — both gitignored.

## Provenance

The structure and the problem sequence follow Stanford CS336 Assignment 1
(spring 2026 handout); the code is my own. The assignment's test harness (28 tests:
tokenizer, BPE training, every model component, AdamW, LR schedule, gradient
clipping, data loading, checkpointing) passes against this code. The assignment
repo itself is not included here — only the implementations.
