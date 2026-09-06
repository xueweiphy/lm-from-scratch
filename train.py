"""Train the Transformer LM on a uint16 token stream (CS336 A1, training_together).

    python train.py                                    # defaults: TinyStories 50 MB sample, valid held out
    python train.py --nrun 5000 --batch_size 32 --device mps
    python train.py --load_ckpt                        # resume from --ckpt_path, append to the log

Every setting is a flag; data is memory-mapped; a checkpoint (model, optimizer,
step) is written every --ckpt_every steps; train/val loss, lr and wall-clock go
to --log_path as CSV and to the console every --eval_every steps.
"""

import argparse, csv, os, time

import numpy as np
import torch

from model import Transformer_lm, cross_entropy
from training import AdamW, learning_rate_schedule, gradient_clipping, data_loading, save_checkpoint, load_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def get_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    # data
    p.add_argument("--train_file", default=f"{DATA}/tinystories_sample50MB.npy")
    p.add_argument("--val_file",   default=f"{DATA}/tinystories_valid.npy")
    # model
    p.add_argument("--vocab_size",     type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model",        type=int, default=512)
    p.add_argument("--d_ff",           type=int, default=None, help="default round(8/3 d_model / 64) * 64")
    p.add_argument("--num_layers",     type=int, default=4)
    p.add_argument("--num_heads",      type=int, default=16)
    p.add_argument("--theta",          type=float, default=10000.)
    # optimizer and schedule
    p.add_argument("--nrun",       type=int,   default=1000, help="total training steps")
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--alpha_max",  type=float, default=1e-3)
    p.add_argument("--alpha_min",  type=float, default=1e-4)
    p.add_argument("--tw",         type=int,   default=100,  help="warmup steps")
    p.add_argument("--tc",         type=int,   default=None, help="cosine end step (default nrun)")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_norm",   type=float, default=1.0, help="gradient clipping norm")
    # checkpoints and logging
    p.add_argument("--ckpt_path",  default=f"{HERE}/checkpoints/tinystories.pt")
    p.add_argument("--ckpt_every", type=int, default=500)
    p.add_argument("--load_ckpt",  action="store_true", help="resume from --ckpt_path")
    p.add_argument("--log_path",   default=f"{HERE}/logs/tinystories.csv")
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_batches", type=int, default=10)
    p.add_argument("--device", default="cpu", help="cpu, mps, or cuda")
    args = p.parse_args()
    if args.d_ff is None:
        args.d_ff = round(8 * args.d_model / 3 / 64) * 64
    if args.tc is None:
        args.tc = args.nrun
    return args


@torch.no_grad()
def loss_estimate ( model, dataI, nbatches, batch_size, context_length, device ) :
    losses = []
    for _ in range ( nbatches ):
        xin, xpred = data_loading ( dataI, batch_size, context_length, device )
        losses.append ( cross_entropy ( model ( xin ), xpred ).item() )
    return sum ( losses ) / len ( losses )


def main():
    a = get_args()
    device = a.device

    data = np.load ( a.train_file, mmap_mode="r" )
    data_val = np.load ( a.val_file, mmap_mode="r" )

    transformer = Transformer_lm ( a.vocab_size, a.context_length, a.num_layers, a.d_model, a.num_heads, a.d_ff,
                                   device = device, theta = a.theta )
    opt = AdamW ( transformer.parameters(), lr = a.alpha_max, betas = ( 0.9, 0.999 ), eps = 1.e-8,
                  weight_decay = a.weight_decay )
    print ( f"{sum ( p.numel() for p in transformer.parameters() ):,} parameters on {device}" )

    Nini = 0
    if a.load_ckpt :
        Nini = load_checkpoint ( a.ckpt_path, transformer, opt )
        Nini += 1
        print ( f"resumed from {a.ckpt_path} at step {Nini}" )

    os.makedirs ( os.path.dirname ( a.ckpt_path ), exist_ok = True )
    os.makedirs ( os.path.dirname ( a.log_path ), exist_ok = True )
    filelog = open ( a.log_path, "a" if a.load_ckpt else "w", newline = "", encoding = "utf-8" )
    writer = csv.writer ( filelog )
    if not a.load_ckpt :
        writer.writerow ( [ "step", "train_loss", "val_loss", "lr", "seconds" ] )
    t0 = time.time()

    for ii in range ( Nini, a.nrun ) :
        opt.zero_grad()
        xin, xpred = data_loading ( data, a.batch_size, a.context_length, device )

        logits = transformer ( xin )
        loss = cross_entropy ( logits, xpred )
        loss.backward()
        gradient_clipping ( transformer.parameters(), a.max_norm )
        for g in opt.param_groups :
            g["lr"] = learning_rate_schedule ( ii, a.alpha_max, a.alpha_min, a.tw, a.tc )
        opt.step()

        if ( ii % a.ckpt_every == 0 and ii > 0 ) or ii == a.nrun - 1 :
            save_checkpoint ( transformer, opt, ii, a.ckpt_path )
            print ( f"checkpoint, save to {a.ckpt_path}" )
        if ii % a.eval_every == 0 or ii == a.nrun - 1 :
            val_loss = loss_estimate ( transformer, data_val, a.eval_batches, a.batch_size, a.context_length, device )
            train_loss = loss_estimate ( transformer, data, a.eval_batches, a.batch_size, a.context_length, device )
            writer.writerow ( [ ii, train_loss, val_loss, g["lr"], time.time() - t0 ] )
            filelog.flush()
            print ( f'step {ii}, train = {train_loss:.4f}, val = {val_loss:.4f}, lr = {g["lr"]:.2e}, {time.time() - t0:.0f} s' )

    filelog.close()


if __name__ == "__main__":
    main()
