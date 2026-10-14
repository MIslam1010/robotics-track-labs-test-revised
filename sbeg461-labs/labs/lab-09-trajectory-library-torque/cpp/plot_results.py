"""Plots for the C++ track of SBEG 461 Lab 9 (given; plotting stays in Python).

./build/lab_tests writes these CSV files into the folder you run it from (each one only when the functions it
uses return results of the right size):
  lab9_cpp_scaling_cubic.csv, lab9_cpp_scaling_quintic.csv,
  lab9_cpp_scaling_trapezoid.csv     t, s, sd, sdd   (T = 1 s; trapezoid v = 1.5, a = 4.5)        -> Part A plot
  lab9_cpp_fig_9_7a.csv              t, x, y         (via-point cubic of MR Fig. 9.7a)            -> Part B plot
  lab9_cpp_bsplines.csv              r, x_open, y_open, x_clamped, y_clamped (same four points)   -> Part B plot
  lab9_cpp_tau_AB.csv                t, r1..r6       (tau_i / tau_max_i, A -> B quintic at T_min) -> Part C plot
  lab9_cpp_phase_2r.csv, lab9_cpp_phase_gantry.csv,
  lab9_cpp_phase_course_arm.csv      s, sdot_limit, sdot_topp (velocity limit curve and TOPP)     -> Part D plot

Usage (from the folder that holds the CSV files):   python cpp/plot_results.py  [folder]
Saves lab9_partA_scalings.png, lab9_partB_vias.png, lab9_partC_torques.png and lab9_partD_phase_plane.png.
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

    data = {k: load(folder, f"lab9_cpp_scaling_{k}.csv") for k in ("cubic", "quintic", "trapezoid")}
    if any(d is not None for d in data.values()):
        fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
        for k, d in data.items():
            if d is None:
                continue
            for ax, col in zip(axes, ("s", "sd", "sdd")):
                ax.plot(d["t"], d[col], label=k)
        for ax, lab in zip(axes, ("s", "ds/dt (1/s)", "d2s/dt2 (1/s^2)")):
            ax.set_ylabel(lab)
            ax.grid(True)
        axes[0].legend()
        axes[0].set_title("Part A: time scalings, T = 1 s")
        axes[-1].set_xlabel("t (s)")
        fig.tight_layout()
        fig.savefig(os.path.join(folder, "lab9_partA_scalings.png"), dpi=150)
        print("saved lab9_partA_scalings.png")

    via, bs = load(folder, "lab9_cpp_fig_9_7a.csv"), load(folder, "lab9_cpp_bsplines.csv")
    if via is not None or bs is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        if via is not None:
            ax1.plot(via["x"], via["y"], label="via-point cubic (MR Fig. 9.7a)")
            ax2.plot(via["t"], via["x"], label="x(t)")
            ax2.plot(via["t"], via["y"], label="y(t)")
        if bs is not None:
            ax1.plot(bs["x_open"], bs["y_open"], "--", label="B-spline, 4 control points")
            ax1.plot(bs["x_clamped"], bs["y_clamped"], ":", label="B-spline, clamped ends")
        ax1.plot([0, 0, 1, 1], [0, 1, 1, 0], "ko")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_aspect("equal")
        ax1.grid(True)
        ax1.legend(fontsize=8)
        ax1.set_title("Part B: path in the plane")
        ax2.set_xlabel("t (s)")
        ax2.set_ylabel("coordinate")
        ax2.grid(True)
        ax2.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(folder, "lab9_partB_vias.png"), dpi=150)
        print("saved lab9_partB_vias.png")

    tau = load(folder, "lab9_cpp_tau_AB.csv")
    if tau is not None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i in range(1, 7):
            ax.plot(tau["t"], tau[f"r{i}"], label=f"joint {i}")
        ax.axhline(1.0, color="k", lw=0.8)
        ax.axhline(-1.0, color="k", lw=0.8)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("tau_i / tau_max,i")
        ax.set_title("Part C: A -> B quintic at T = T_min")
        ax.grid(True)
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(folder, "lab9_partC_torques.png"), dpi=150)
        print("saved lab9_partC_torques.png")

    phases = [(k, load(folder, f"lab9_cpp_phase_{k}.csv")) for k in ("2r", "gantry", "course_arm")]
    phases = [(k, d) for k, d in phases if d is not None]
    if phases:
        fig, axes = plt.subplots(1, len(phases), figsize=(5 * len(phases), 4))
        axes = np.atleast_1d(axes)
        titles = {"2r": "MR 2R arm", "gantry": "demining gantry", "course_arm": "course arm (hw limits)"}
        for ax, (k, d) in zip(axes, phases):
            ax.plot(d["s"], d["sdot_limit"], "k--", label="velocity limit curve")
            ax.plot(d["s"], d["sdot_topp"], label="TOPP sdot(s)")
            ax.set_xlabel("s")
            ax.set_ylabel("sdot (1/s)")
            ax.set_title(f"Part D: {titles[k]}")
            ax.set_ylim(0, min(1.05 * np.max(d["sdot_limit"]), 2.5 * np.max(d["sdot_topp"])))  # VLC -> inf at s = 0
            ax.grid(True)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(folder, "lab9_partD_phase_plane.png"), dpi=150)
        print("saved lab9_partD_phase_plane.png")


if __name__ == "__main__":
    main()
