"""Plot the LR sweep: learning curves, and best val loss vs learning rate.

    python experiments/plot_sweep.py          # reads logs/lr*.csv, writes lr_sweep.png
"""
import glob, os
import numpy as np, matplotlib.pyplot as plt

lr = lambda f: float(os.path.basename(f)[2:-4])          # logs/lr3e-4.csv -> 0.0003

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
res = []
for f in sorted(glob.glob("logs/lr*.csv"), key=lr):
    d = np.atleast_1d(np.genfromtxt(f, delimiter=",", names=True))
    a.semilogy(d["step"], d["val_loss"], label=f"{lr(f):g}")
    res.append((lr(f), d["val_loss"].min()))

a.set(xlabel="step", ylabel="val loss"); a.legend(title="lr")
b.semilogx(*zip(*res), "o-"); b.set(xlabel="lr", ylabel="best val loss")
fig.tight_layout(); fig.savefig("lr_sweep.png")
print(res)
