"""Plot the CSV files written by lab::export_results (C++ track of SBEG 461 Lab 11). Given code.

    python plot_results.py <prefix> <scene>        e.g.  python plot_results.py results/bench_seed0 bench_or

Reads <prefix>_tree.csv, _path.csv, _smooth.csv, _traj.csv, _tool.csv and ../data/scenes.json,
../data/course_arm.json, and saves <prefix>_fig1_tool_path.png (3-D tool-point path and obstacles),
<prefix>_fig2_q_qdot.png (q(t) and qdot(t) with the derated speed limits) and <prefix>_fig3_tree.png (tree and
paths projected on (theta1, theta2)). These are drafts of Figures 1-3 of the report; adapt them as you like.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"


def load(prefix, name):
    f = Path(f"{prefix}_{name}.csv")
    return np.loadtxt(f, delimiter=",", skiprows=1, ndmin=2) if f.exists() else None


def main(prefix, scene_name):
    scene = json.loads((DATA / "scenes.json").read_text())[scene_name]
    arm = json.loads((DATA / "course_arm.json").read_text())
    qdot_max = arm["derated_speed_fraction"] * np.array(arm["qdot_max"])
    tree, raw, smooth = load(prefix, "tree"), load(prefix, "path"), load(prefix, "smooth")
    traj, tool = load(prefix, "traj"), load(prefix, "tool")

    if tool is not None:  # Figure 1
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(projection="3d")
        u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
        for cx, cy, cz, R in scene["spheres"]:
            ax.plot_surface(cx + R * np.cos(u) * np.sin(v), cy + R * np.sin(u) * np.sin(v), cz + R * np.cos(v),
                            color="tab:red", alpha=0.3)
        ax.plot(tool[:, 1], tool[:, 2], tool[:, 3], "b-", label="tool point (smoothed, time-scaled)")
        ax.scatter(*tool[0, 1:], color="g", label="start")
        ax.scatter(*tool[-1, 1:], color="k", label="goal")
        ax.set_xlabel("x (m)"), ax.set_ylabel("y (m)"), ax.set_zlabel("z (m)")
        ax.legend(fontsize=7)
        fig.savefig(f"{prefix}_fig1_tool_path.png", dpi=150, bbox_inches="tight")

    if traj is not None:  # Figure 2
        n = (traj.shape[1] - 1) // 3
        t, Q, Qd = traj[:, 0], traj[:, 1:1 + n], traj[:, 1 + n:1 + 2 * n]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
        for i in range(n):
            a1.plot(t, Q[:, i], label=f"q{i + 1}")
            a2.plot(t, Qd[:, i], label=f"qd{i + 1}")
        a2.axhline(qdot_max.min(), color="k", ls="--", lw=0.8, label="speed limit")
        a2.axhline(-qdot_max.min(), color="k", ls="--", lw=0.8)
        a1.set_ylabel("q (rad)"), a2.set_ylabel("qdot (rad/s)"), a2.set_xlabel("t (s)")
        a1.legend(ncol=3, fontsize=7), a2.legend(ncol=4, fontsize=7)
        fig.savefig(f"{prefix}_fig2_q_qdot.png", dpi=150, bbox_inches="tight")

    if tree is not None:  # Figure 3
        fig, ax = plt.subplots(figsize=(6, 5))
        for row in tree[1:]:
            p = int(row[1])
            if p >= 0:
                ax.plot([tree[p, 2], row[2]], [tree[p, 3], row[3]], color="0.7", lw=0.6)
        if raw is not None and raw.size:
            ax.plot(raw[:, 0], raw[:, 1], "b.-", label="RRT path")
        if smooth is not None and smooth.size:
            ax.plot(smooth[:, 0], smooth[:, 1], "r.-", label="smoothed path")
        ax.set_xlabel(r"$\theta_1$ (rad)"), ax.set_ylabel(r"$\theta_2$ (rad)"), ax.legend()
        fig.savefig(f"{prefix}_fig3_tree.png", dpi=150, bbox_inches="tight")
    print("figures saved next to", prefix)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
