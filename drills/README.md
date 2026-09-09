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
| 2 | 2026-09-09 | 2 h 00 min | all modules + training loop; ran, loss 9.24 → 7.90; two defects at review, both fixed the same day |

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

## Drill 2 — what had to be looked up

Five of the six drill-1 defects did not recur: the mask is assigned, `gamma` is applied,
`Mo` is applied, the model returns logits, and the cross-entropy subtracts the max in
both terms. Two new ones, found on review:

1. `TransformerBlock.forward` computed `ln1(xin)` and then passed `xin` to attention —
   the norm was discarded, leaving the attention sublayer with no pre-norm. Same class
   of error as drill 1's `masked_fill`: a value computed and dropped.
2. RoPE's `positions` branches were inverted, so the `None` path indexed the table with
   `None` and added an axis. It broadcast correctly only because T == Tmax in this run —
   the same accident as drill 1's `[:seqlen+1]`. RoPE's position path has now failed
   twice; it needs its own shape test with T < Tmax.

Still open, and still using stand-ins: AdamW (`torch.optim.AdamW`), the cosine schedule
with warmup, and gradient clipping. Those are the target for drill 3.

3. `RmsNorm` added eps outside the square root (`x/(sqrt(ms)+eps)`) where the reference
   has it inside (`x/sqrt(ms+eps)`). The two differ only when the mean square approaches
   zero — but that is the case eps exists to bound. Fixed.
