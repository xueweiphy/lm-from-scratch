#!/bin/bash
# Batch-size sweep (CS336 A1, batch_size_experiment).  Fixed step count, fixed lr;
# plot_batch.py then re-plots the same runs against steps, tokens and wall-clock.
#
#   bash sweep_batch.sh              # 1 4 16 64 128 256
#   BATCHES="64 128" bash sweep_batch.sh

TRAIN=${TRAIN:-/tmp/tinystories_train.npy}
VAL=${VAL:-/tmp/tinystories_valid.npy}
CKPT=${CKPT:-/tmp}
LOGDIR=${LOGDIR:-logs}
DEV=${DEV:-cuda}
LR=${LR:-1e-3}
STEPS=${STEPS:-2000}
BATCHES=${BATCHES:-"1 4 16 64 128 256"}

mkdir -p $LOGDIR
for B in $BATCHES ; do
    echo "=== batch=$B  lr=$LR  steps=$STEPS  $(date +%H:%M:%S) ==="
    python train.py Device=$DEV Batch_size=$B Nrun=$STEPS Alpha_max=$LR \
        Eval_every=$((STEPS/20)) File_train=$TRAIN File_val=$VAL \
        Log_path=$LOGDIR/b$B.csv Ckpt_path=$CKPT/b$B.pt || echo "  (batch $B failed — probably OOM)"
done
echo "=== done $(date +%H:%M:%S) ==="
