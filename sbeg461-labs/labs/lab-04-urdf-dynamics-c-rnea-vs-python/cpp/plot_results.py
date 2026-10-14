"""Plots for the C++ track of SBEG 461 Lab 4, Step 6 (given; plotting stays in Python).

./build/lab_bench [folder] writes, with YOUR simulate_channel_2r (columns t, theta1, theta2, dtheta1, dtheta2, lam,
e, KE):
  step6_1_energy.csv       Example 8.5 state, tau = 0, 2 s, alpha = beta = 0              (Step 6.1)
  step6_2_plain.csv        0.57 deg initial violation, alpha = beta = 0                    (Step 6.2)
  step6_2_baumgarte.csv    same start, alpha = beta = 20 1/s                               (Step 6.2)
  step6_3_stiff.csv        same start, alpha = beta = 2000 1/s                             (Step 6.3)

Usage (from your Lab code folder):   python cpp/plot_results.py  [folder with the CSV files, default .]
Saves step6_energy_lambda.png and step6_drift.png into the same folder.
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(folder, name):
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        header = fh.readline().strip().split(",")
    D = np.genfromtxt(path, delimiter=",", skip_header=1, ndmin=2)
    return {h: D[:, k] for k, h in enumerate(header)}


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    s1 = load(folder, "step6_1_energy.csv")
    if s1 is None:
        sys.exit(f"no step6_1_energy.csv in {folder}: run ./build/lab_bench {folder} first")
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax[0].plot(s1["t"], s1["KE"])
    ax[0].set_ylabel("kinetic energy (J)")
    ax[0].set_title("Step 6.1: 2R arm in the channel x = 1, tau = 0")
    ax[1].plot(s1["t"], s1["e"])
    ax[1].set_ylabel("constraint error e (m)")
    ax[2].plot(s1["t"], s1["lam"])
    ax[2].set_ylabel("lambda (N)")
    ax[2].set_xlabel("t (s)")
    fig.tight_layout()
    out1 = os.path.join(folder, "step6_energy_lambda.png")
    fig.savefig(out1, dpi=150)
    print("saved", out1, "  KE range:", np.ptp(s1["KE"]), "J")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, label in (("step6_2_plain.csv", "alpha = beta = 0"), ("step6_2_baumgarte.csv", "alpha = beta = 20 1/s"),
                        ("step6_3_stiff.csv", "alpha = beta = 2000 1/s")):
        s = load(folder, name)
        if s is not None:
            ok = np.isfinite(s["e"]) & (np.abs(s["e"]) < 1.0)
            ax.plot(s["t"][ok], s["e"][ok], label=label)
            if not ok.all():
                print(f"{name}: the simulation diverged at t = {s['t'][~ok][0]:.3f} s")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("constraint error e (m)")
    ax.set_title("Step 6.2-6.3: constraint drift with and without Baumgarte stabilisation")
    ax.legend()
    fig.tight_layout()
    out2 = os.path.join(folder, "step6_drift.png")
    fig.savefig(out2, dpi=150)
    print("saved", out2)


if __name__ == "__main__":
    main()
