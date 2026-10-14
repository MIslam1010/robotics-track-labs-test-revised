"""Plots for the C++ track of the SBEG 461 enrichment lab "CBF safety filter" (given; plotting stays in Python).

./build/lab_experiments [folder] writes one CSV log per run into the folder (default: the current folder):
  cbf_<scenario>_<k>.csv   scenario static / approaching; k = 1 nominal, 2 keep-out only, 3 filter with the
                           scenario v_max (0.20 m/s), 4 filter with your design speed
  columns  t, x, y, x_o, y_o, x_d, y_d, q1, q2, d, approach, F, active

Usage (from your Lab code folder):   python cpp/plot_results.py  [folder]
Saves cbf_paths_static.png (handle paths and the keep-out disc) and cbf_signals_approaching.png (distance to the
disc, approach speed, contact force) into the same folder.
"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = {1: "nominal (no filter)", 2: "keep-out only", 3: "filter, v_max 0.20", 4: "filter, design v_max"}


def load(folder, name):
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        print("missing", path, "(run ./build/lab_experiments first)")
        return None
    with open(path) as fh:
        header = fh.readline().strip().split(",")
    D = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return {h: D[:, k] for k, h in enumerate(header)}


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    with open(os.path.join(HERE, "..", "data", "scenario.json")) as fh:
        common = json.load(fh)["common"]
    r, F_max = common["r_keepout"], common["F_max"]

    runs = {k: load(folder, f"cbf_static_{k}.csv") for k in LABELS}
    if any(d is not None for d in runs.values()):
        fig, ax = plt.subplots(figsize=(7, 5))
        for k, d in runs.items():
            if d is not None:
                ax.plot(d["x"], d["y"], label=LABELS[k])
        d0 = next(d for d in runs.values() if d is not None)
        ax.add_patch(plt.Circle((d0["x_o"][0], d0["y_o"][0]), r, color="tab:red", alpha=0.25, label="keep-out disc"))
        ax.plot(d0["x_d"], d0["y_d"], "k:", lw=1, label="reference")
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Static scenario: handle paths")
        ax.grid(True)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = os.path.join(folder, "cbf_paths_static.png")
        fig.savefig(out, dpi=150)
        print("saved", out)

    runs = {k: load(folder, f"cbf_approaching_{k}.csv") for k in LABELS}
    if any(d is not None for d in runs.values()):
        fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
        for k, d in runs.items():
            if d is None:
                continue
            axes[0].plot(d["t"], 1000 * d["d"], label=LABELS[k])
            axes[1].plot(d["t"], d["approach"], label=LABELS[k])
            axes[2].plot(d["t"], d["F"], label=LABELS[k])
        axes[0].axhline(0.0, color="k", lw=0.8)
        axes[1].axhline(0.20, color="k", ls="--", lw=0.8)
        axes[2].axhline(F_max, color="k", ls="--", lw=0.8)
        for ax, lab in zip(axes, ("d (mm)", "approach speed (m/s)", "contact force if touched (N)")):
            ax.set_ylabel(lab)
            ax.grid(True)
        axes[2].set_xlabel("t (s)")
        axes[0].set_title("Approaching scenario")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        out = os.path.join(folder, "cbf_signals_approaching.png")
        fig.savefig(out, dpi=150)
        print("saved", out)


if __name__ == "__main__":
    main()
