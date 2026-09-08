# Timed from-memory rebuilds

The whole decoder-only Transformer written from memory, with no reference and no AI,
on a timer. The finished and tested versions of these modules live in `model/` and
`training/`; what is recorded here is what could be reproduced cold on a given day,
and what could not.

Each drill is committed as it ran, bugs and all — the fixes are described below rather
than applied, so successive drills can be diffed against each other.

| drill | date | time | outcome |
|---|---|---|---|
| 1 | 2026-09-07 | 3 h 45 min | all modules + training loop; ran, loss 9.26 → 5.37 in 10 steps |

## Drill 1 — what had to be looked up

Architecture came back in full. The six defects found on review, in the order they
mattered:

1. `masked_fill` returns a new tensor — the result was discarded, so attention saw
   the future. The tell was the loss starting at 7.3 instead of ln(10000) = 9.2 and
   falling suspiciously fast.
2. `RmsNorm` created `gamma` but never multiplied by it.
3. `MultiHeadAttention` created `Wo` but never applied it — 4M dead parameters.
4. RoPE: rotation sign flipped relative to the reference, and `[:seqlen+1]` where
   `[:seqlen]` was meant (harmless only because seqlen == max_length).
5. The model returned probabilities and `cross_entropy` took their log; the
   convention is logits out, log-sum-exp inside the loss.
6. Fixing (5) introduced a second bug: the max was subtracted in the log-sum-exp
   term but not in the picked-target term, making the loss go negative.

Stand-ins used, to be written from memory next time: AdamW (used `torch.optim.AdamW`),
the cosine schedule with warmup (constant lr instead), gradient clipping (omitted).

Checks worth keeping: the first loss must be ln(vocab_size); a negative loss means the
loss function, not the model; a model that learns *too* fast is a leak.
