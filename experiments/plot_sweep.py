"""Plot an LR sweep: learning curves per run, and final val loss vs learning rate.

    python experiments/plot_sweep.py                 # reads logs/lr*.csv
    python experiments/plot_sweep.py logs/other*.csv

Writes experiments/lr_sweep.png.  A run whose val loss ended above its own start
(or went NaN) is reported as diverged and marked on the right-hand panel.
"""
import glob, os, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]

paths = sorted(sys.argv[1:] or glob.glob("logs/lr*.csv"),
               key=lambda p: float(os.path.basename(p)[2:-4]))
if not paths:
    sys.exit("no logs/lr*.csv found — run sweep.sh first")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=160)
fig.patch.set_facecolor(SURF)
lrs, finals, diverged = [], [], []

for path, col in zip(paths, COLORS):
    lr = float(os.path.basename(path)[2:-4])
    d = np.genfromtxt(path, delimiter=",", names=True)
    step, val = np.atleast_1d(d["step"]), np.atleast_1d(d["val_loss"])
    bad = not np.isfinite(val).all() or val[-1] > 1.15 * np.nanmin(val)   # ended well above its own best
    axL.plot(step, val, color=col, lw=1.8, label=f"{lr:g}" + ("  diverged" if bad else ""))
    lrs.append(lr); finals.append(np.nanmin(val)); diverged.append(bad)
    print(f"{lr:>8g}  final val {val[-1]:.3f}   best {np.nanmin(val):.3f}"
          + ("   DIVERGED" if bad else ""))

axL.set_yscale("log")
axL.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
axL.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
axL.set_xlabel("step", color=INK2); axL.set_ylabel("validation loss  (log)", color=INK2)
axL.set_title("Learning curves", color=INK, fontsize=12, loc="left")
axL.legend(frameon=False, fontsize=9, labelcolor=INK2, title="learning rate",
           title_fontproperties={"size": 9})

ok = np.array([not b for b in diverged])
axR.semilogx(np.array(lrs)[ok], np.array(finals)[ok], "o-", color="#2a78d6", lw=1.8, ms=7)
for lr, f, b in zip(lrs, finals, diverged):
    if b:
        axR.semilogx([lr], [f], "x", color="#eb6834", ms=11, mew=2.5)
        axR.text(lr, f, "diverged  ", color="#eb6834", fontsize=9, va="center", ha="right")
axR.set_xlim(min(lrs)/1.7, max(lrs)*1.7)
axR.set_xlabel("learning rate  (peak of the cosine)", color=INK2)
axR.set_ylabel("best validation loss", color=INK2)
axR.set_title("Sweep curve", color=INK, fontsize=12, loc="left")

for ax in (axL, axR):
    ax.set_facecolor(SURF); ax.grid(True, color="#e6e5e0", lw=0.8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#d6d5d0")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.tick_params(which="minor", colors=INK2, labelsize=7.5)

fig.tight_layout()
fig.savefig("experiments/lr_sweep.png", facecolor=SURF)
print("wrote experiments/lr_sweep.png")
