#!/bin/bash
# Architecture ablations (CS336 A1 §7.3).  One GPU, sequential, same config for every run.
#
#   nohup bash ablate.sh > logs_ablate.log 2>&1 &   ;   tail -f logs_ablate.log
#   RUNS="norm_none:Norm=none" bash ablate.sh       # just one of them
#   LR=3e-4 RUNS="norm_none:Norm=none" bash ablate.sh   # retry a diverged one lower

TRAIN=${TRAIN:-/tmp/tinystories_train.npy}
VAL=${VAL:-/tmp/tinystories_valid.npy}
CKPT=${CKPT:-/tmp}
LOGDIR=${LOGDIR:-logs_ablate}
DEV=${DEV:-cuda}
BATCH=${BATCH:-32}
LR=${LR:-1e-3}
STEPS=${STEPS:-5000}
RUNS=${RUNS:-"baseline: norm_none:Norm=none post_norm:Norm_pos=post nope:Pos=none silu:Ffn=silu"}

mkdir -p $LOGDIR
for R in $RUNS ; do
    NAME=${R%%:*} ; FLAG=${R#*:}
    echo "=== $NAME  $FLAG  lr=$LR  steps=$STEPS  $(date +%H:%M:%S) ==="
    python train.py Device=$DEV Batch_size=$BATCH Nrun=$STEPS Alpha_max=$LR $FLAG \
        Eval_every=100 File_train=$TRAIN File_val=$VAL \
        Log_path=$LOGDIR/$NAME.csv Ckpt_path=$CKPT/$NAME.pt || echo "  ($NAME failed)"
done
echo "=== done $(date +%H:%M:%S) ==="
