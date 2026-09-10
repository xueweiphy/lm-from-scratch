"""Plot the ablation runs: one val-loss curve per CSV in the directory.

    python experiments/plot_ablate.py [logdir]   # reads <logdir>/*.csv, writes <logdir>.png
"""
import glob, os, sys
import numpy as np, matplotlib.pyplot as plt

logdir = sys.argv[1] if len(sys.argv) > 1 else "logs_ablate"

for f in sorted(glob.glob(f"{logdir}/*.csv")):
    d = np.atleast_1d(np.genfromtxt(f, delimiter=",", names=True))
    plt.semilogy(d["step"], d["val_loss"], label=os.path.basename(f)[:-4])
    print(f"{os.path.basename(f)[:-4]:12s} {d['val_loss'].min():.4f}")

plt.ylim(1.5, 5)                                         # the interesting range; norm_none starts at 18
plt.xlabel("step"); plt.ylabel("val loss"); plt.legend()
plt.tight_layout(); plt.savefig(f"{logdir}.png")
