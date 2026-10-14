"""Plots for the C++ track of SBEG 46X Lab 6 (given; plotting stays in Python).

./build/lab_tests writes these CSV files into the folder you run it from (each one only when the functions it
uses work):
  lab6_cpp_segment_backbones.csv  theta_deg, s, x, z     (one 40 mm segment, theta = 0, 30, ..., 150 deg, phi = 0) -> E2
  lab6_cpp_goto.csv               step, err              (go_to towards the first ring point, dt = 10 ms, K = 5)  -> E7
  lab6_cpp_ring.csv               t, x_des, y_des, z_des, x, y, z, err, z_ins, th1x, th1y, th2x, th2y, dl1..dl6
                                                         (tracking of data/ablation_ring.csv, K = 20)             -> E8
  lab6_cpp_ring_backbones.csv     t, x, y, z             (catheter_backbone at t = 0, 2, 4, 6 s, 42 points each)  -> E8

Usage (from the folder that holds the CSV files):   python cpp/plot_results.py  [folder]
Saves lab6_e2_backbones.png, lab6_e7_goto.png, lab6_e8_ring.png and lab6_e8_tendons.png.
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
        print("missing", path, "(run ./build/lab_tests from this folder first)")
        return None
    with open(path) as fh:
        header = fh.readline().strip().split(",")
    D = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return {h: D[:, k] for k, h in enumerate(header)}


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "."

    d = load(folder, "lab6_cpp_segment_backbones.csv")
    if d is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        for th in np.unique(d["theta_deg"]):
            m = d["theta_deg"] == th
            ax.plot(1e3 * d["x"][m], 1e3 * d["z"][m], label=f"{th:.0f} deg")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z (mm)")
        ax.set_aspect("equal")
        ax.grid(True)
        ax.legend()
        ax.set_title("E2: one 40 mm segment, phi = 0")
        fig.savefig(os.path.join(folder, "lab6_e2_backbones.png"), dpi=150, bbox_inches="tight")
        print("saved lab6_e2_backbones.png")

    d = load(folder, "lab6_cpp_goto.csv")
    if d is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(d["step"], 1e3 * np.maximum(d["err"], 1e-12))
        ax.axhline(0.1, color="k", ls="--", lw=0.8)
        ax.set_xlabel("step (dt = 10 ms)")
        ax.set_ylabel("tip error (mm)")
        ax.grid(True, which="both")
        ax.set_title("E7: go_to the first ring point, K = 5 1/s")
        fig.savefig(os.path.join(folder, "lab6_e7_goto.png"), dpi=150, bbox_inches="tight")
        print("saved lab6_e7_goto.png")

    r = load(folder, "lab6_cpp_ring.csv")
    b = load(folder, "lab6_cpp_ring_backbones.csv")
    if r is not None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(projection="3d")
        ax.plot(1e3 * r["x_des"], 1e3 * r["y_des"], 1e3 * r["z_des"], "k--", lw=1, label="commanded")
        ax.plot(1e3 * r["x"], 1e3 * r["y"], 1e3 * r["z"], lw=1.5, label="achieved")
        if b is not None:
            for t in np.unique(b["t"]):
                m = b["t"] == t
                ax.plot(1e3 * b["x"][m], 1e3 * b["y"][m], 1e3 * b["z"][m], lw=2, label=f"catheter, t = {t:.0f} s")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_zlabel("z (mm)")
        ax.legend(fontsize=8)
        ax.set_title(f"E8: ablation ring, max error {1e3 * r['err'].max():.3f} mm")
        fig.savefig(os.path.join(folder, "lab6_e8_ring.png"), dpi=150, bbox_inches="tight")
        print("saved lab6_e8_ring.png")

        fig, ax = plt.subplots(figsize=(7, 4))
        for i in range(1, 7):
            ax.plot(r["t"], 1e3 * r[f"dl{i}"], label=f"tendon {i}")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("l - l(straight) (mm)")
        ax.grid(True)
        ax.legend(ncol=3, fontsize=8)
        ax.set_title("E8: tendon displacements")
        fig.savefig(os.path.join(folder, "lab6_e8_tendons.png"), dpi=150, bbox_inches="tight")
        print("saved lab6_e8_tendons.png")


if __name__ == "__main__":
    main()
