"""Train the Transformer LM on a uint16 token stream (CS336 A1, training_together).

    python train.py                            # settings below
    python train.py Nrun=5000 Batch_size=32 Device=mps Alpha_max=3e-4
    python train.py LoadCkpt=True              # resume from Ckpt_path, append to the log
"""
import csv, os, sys, time
import numpy as np
import torch

from model import Transformer_lm, cross_entropy
from training import AdamW, learning_rate_schedule, gradient_clipping, data_loading, save_checkpoint, load_checkpoint

# ---- settings (override any of them as key=value on the command line) ----
Vocab_size, Context_length, Dmodel, Num_layers, Num_heads, Theta = 10000, 256, 512, 4, 16, 10000
Nrun, Batch_size, Alpha_max, Alpha_min, Tw, Max_norm = 1000, 8, 1e-3, 1e-4, 100, 1.0
File_train, File_val = "data/tinystories_sample50MB.npy", "data/tinystories_valid.npy"
Ckpt_path, Index_checkpoint, LoadCkpt = "checkpoints/tinystories.pt", 500, False
Log_path, Eval_every, Eval_batches = "logs/tinystories.csv", 100, 10
Norm, Norm_pos, Pos, Ffn = "rms", "pre", "rope", "swiglu"     # ablations: Norm=none, Norm_pos=post, Pos=none, Ffn=silu
Device = "cpu"

for arg in sys.argv[1:]:                       # e.g. Nrun=5000 -> Nrun = 5000 (same type as the default)
    key, value = arg.split("=", 1)
    default = globals()[key]
    globals()[key] = value == "True" if isinstance(default, bool) else type(default)(value)
Dff = round ( 8 * Dmodel / 3 / 64 ) * 64 if Ffn == "swiglu" else 4 * Dmodel
Tc = Nrun
# ---------------------------------------------------------------------------

data = np.load ( File_train, mmap_mode="r" )
data_val = np.load ( File_val, mmap_mode="r" )

transformer = Transformer_lm ( Vocab_size, Context_length, Num_layers, Dmodel, Num_heads, Dff, device = Device, theta = Theta if Pos == "rope" else None , norm = Norm , norm_pos = Norm_pos , ffn = Ffn )
opt = AdamW ( transformer.parameters(), lr = Alpha_max, betas = ( 0.9, 0.999 ), eps = 1.e-8, weight_decay = 0.01 )


@torch.no_grad()
def loss_estimate ( model, dataI, nbatches ) :
    losses = []
    for _ in range ( nbatches ):
        xin, xpred = data_loading ( dataI, Batch_size, Context_length, Device )
        losses.append ( cross_entropy ( model ( xin ), xpred ).item() )
    return sum ( losses ) / len ( losses )


Nini = 0
if LoadCkpt :
    Nini = load_checkpoint ( Ckpt_path, transformer, opt )
    Nini += 1

os.makedirs ( os.path.dirname ( Ckpt_path ), exist_ok = True ); os.makedirs ( os.path.dirname ( Log_path ), exist_ok = True )
filelog = open ( Log_path, "a" if LoadCkpt else "w", newline = "", encoding = "utf-8" )
writer = csv.writer ( filelog )
if not LoadCkpt :
    writer.writerow ( [ "step", "train_loss", "val_loss", "lr", "seconds" ] )
t0 = time.time()

for ii in range ( Nini, Nrun ) :
    opt.zero_grad()
    xin, xpred = data_loading ( data, Batch_size, Context_length, Device )

    logits = transformer ( xin )
    loss = cross_entropy ( logits, xpred )
    if not torch.isfinite ( loss ) or loss.item() > 20 :          # diverged — stop, don't burn the GPU
        print ( f"DIVERGED at step {ii}, loss = {loss.item()}" ) ; break
    loss.backward()
    gradient_clipping ( transformer.parameters(), Max_norm )
    for g in opt.param_groups :
        g["lr"] = learning_rate_schedule ( ii, Alpha_max, Alpha_min, Tw, Tc )
    opt.step()
    if ( ii % Index_checkpoint == 0 and ii > 0 ) or ii == Nrun - 1 :
        save_checkpoint ( transformer, opt, ii, Ckpt_path )
        print ( f"checkpoint, save to {Ckpt_path}" )
    if ii % Eval_every == 0 or ii == Nrun - 1 :
        val_loss = loss_estimate ( transformer, data_val, Eval_batches )
        train_loss = loss_estimate ( transformer, data, Eval_batches )
        writer.writerow ( [ ii, train_loss, val_loss, g["lr"], time.time() - t0 ] )
        filelog.flush()
        print ( f'step {ii}, train = {train_loss:.4f}, val = {val_loss:.4f}, lr = {g["lr"]:.2e}' )

filelog.close()
