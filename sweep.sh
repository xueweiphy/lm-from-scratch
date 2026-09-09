#!/bin/bash
# Learning-rate sweep (CS336 A1, learning_rate).  One GPU, runs sequentially.
#
#   bash sweep.sh probe     # 200 steps at 1e-3 — read seconds/step before committing
#   DEV=mps STEPS=20 bash sweep.sh probe    # dry run of the whole path on the Mac mini
#   bash sweep.sh           # the four runs
#
# On SWAN (terminal via the >_ button, or a notebook cell prefixed with !):
#   cp -r /eos/user/<x>/<user>/lm-from-scratch ~/lm && cd ~/lm
#   cp <...>/tinystories_train.npy <...>/tinystories_valid.npy /tmp/   # never train off EOS
#   nohup bash sweep.sh > sweep.log 2>&1 &   ;   tail -f sweep.log
# Interrupted?  Rerun that one lr with LoadCkpt=True appended.

TRAIN=${TRAIN:-/tmp/tinystories_train.npy}
VAL=${VAL:-/tmp/tinystories_valid.npy}
CKPT=${CKPT:-/tmp}                       # 272 MB per run — keep these off EOS
LOGDIR=${LOGDIR:-logs}                   # point at EOS so the CSVs survive a killed session
BATCH=${BATCH:-32}
CLIP=${CLIP:-1.0}                       # CLIP=1e9 turns clipping off, to see the true edge
MIN=${MIN:-1e-4}                         # MIN=flat  ->  Alpha_min = Alpha_max, i.e. constant lr after warmup
DEV=${DEV:-cuda}                         # mps / cpu to rehearse without a GPU

if [ "$1" = probe ] ; then LRS="1e-3"; STEPS=${STEPS:-200}; EVERY=50
else                       LRS=${LRS:-"3e-4 1e-3 3e-3 1e-2"}; STEPS=${STEPS:-5000}; EVERY=100; fi

mkdir -p $LOGDIR
for LR in $LRS ; do
    M=$MIN ; [ "$MIN" = flat ] && M=$LR
    echo "=== Alpha_max=$LR  Alpha_min=$M  batch=$BATCH  steps=$STEPS  $(date +%H:%M:%S) ==="
    python train.py Device=$DEV Batch_size=$BATCH Nrun=$STEPS Alpha_max=$LR \
        Eval_every=$EVERY Max_norm=$CLIP Alpha_min=$M File_train=$TRAIN File_val=$VAL \
        Log_path=$LOGDIR/lr$LR.csv Ckpt_path=$CKPT/lr$LR.pt
done
echo "=== done $(date +%H:%M:%S) ==="
