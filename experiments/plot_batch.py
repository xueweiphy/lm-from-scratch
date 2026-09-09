"""Plot the batch-size sweep: val loss vs steps, vs tokens, vs wall-clock.

    python experiments/plot_batch.py          # reads logs/b*.csv, writes batch_sweep.png
"""
import glob, os
import numpy as np, matplotlib.pyplot as plt

bs = lambda f: int(os.path.basename(f)[1:-4])            # logs/b64.csv -> 64
Context = 256

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for f in sorted(glob.glob("logs/b*.csv"), key=bs):
    d = np.atleast_1d(np.genfromtxt(f, delimiter=",", names=True))
    ax[0].plot(d["step"], d["val_loss"], label=f"{bs(f)}")
    ax[1].semilogx(d["step"] * bs(f) * Context, d["val_loss"])
    ax[2].plot(d["seconds"], d["val_loss"])

ax[0].set(xlabel="step", ylabel="val loss"); ax[0].legend(title="batch")
ax[1].set(xlabel="tokens seen", ylabel="val loss")
ax[2].set(xlabel="wall-clock (s)", ylabel="val loss")
fig.tight_layout(); fig.savefig("batch_sweep.png")
