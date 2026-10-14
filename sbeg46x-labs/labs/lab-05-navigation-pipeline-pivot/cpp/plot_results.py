"""SBEG 46X Lab 5 -- C++ track: the three plots of the deliverables from the CSV files written by lab_tests.

When test_D2_navigation_pipeline passes, ./build/lab_tests writes into the folder you run it from:
    lab5_cpp_mc.csv        E7: 5000 Monte-Carlo trials (seed 0, FLE estimate of E6): fre, tre_T1_central, ...
    lab5_cpp_tre_map.csv   E8: predicted TRE on the grid z_I = 10 mm (x_I -60..180, y_I -60..140, step 2 mm)
    lab5_cpp_points.csv    E10: kind (0 digitised fiducial, 1 registered CT fiducial, 2 target), x_s, y_s, z_s

Usage (from your Lab code folder, after running lab_tests there):
    python cpp/plot_results.py            # or: python cpp/plot_results.py <folder with the CSV files>
Saves lab5_E7_fre_tre_scatter.png, lab5_E8_tre_map.png and lab5_E10_pipeline_3d.png next to the CSV files.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FOLDER = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()


def load(name):
    path = FOLDER / name
    if not path.exists():
        sys.exit(f"{path} not found: run ./build/lab_tests in this folder first (test_D2 must pass).")
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


# ---- E7: FRE-TRE scatter at T2_lateral
mc = load("lab5_cpp_mc.csv")
f, e2 = mc[:, 0], mc[:, 2]
r = np.corrcoef(f, e2)[0, 1]
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(f, e2, s=3, alpha=0.4)
ax.set_xlabel("FRE (mm)")
ax.set_ylabel("TRE at T2_lateral (mm)")
ax.set_title(f"E7: FRE vs TRE, {len(f)} trials, corr = {r:.3f}")
fig.tight_layout()
fig.savefig(FOLDER / "lab5_E7_fre_tre_scatter.png", dpi=150)
for j, name in enumerate(("T1_central", "T2_lateral", "T3_deep")):
    e = mc[:, 1 + j]
    print(f"E7 {name}: RMS TRE {np.sqrt(np.mean(e ** 2)):.3f} mm, 95th percentile {np.percentile(e, 95):.3f} mm")
print(f"E7 corr(FRE, TRE at T2_lateral) = {r:.3f}")

# ---- E8: TRE map with the fiducials
g = load("lab5_cpp_tre_map.csv")
xs, ys = np.unique(g[:, 0]), np.unique(g[:, 1])
Z = g[:, 2].reshape(len(ys), len(xs))          # written row by row: y outer, x inner
ph = json.loads((HERE.parent / "data" / "phantom.json").read_text())
X = np.array(ph["fiducials_image_mm"])
fig, ax = plt.subplots(figsize=(6, 4.5))
cs = ax.contourf(xs, ys, Z, levels=20, cmap="viridis")
fig.colorbar(cs, ax=ax, label="predicted RMS TRE (mm)")
ax.plot(X[:, 0], X[:, 1], "w^", ms=8, mec="k", label="fiducials")
for i, (name, t) in enumerate(ph["targets_image_mm"].items()):   # T1 and T3 share (x, y): stagger the labels
    ax.plot(t[0], t[1], "r*", ms=10)
    ax.annotate(f"{name} (z = {t[2]:g})", (t[0], t[1]), color="w", fontsize=8, xytext=(4, 4 - 10 * i),
                textcoords="offset points")
ax.set_xlabel("$x_I$ (mm)")
ax.set_ylabel("$y_I$ (mm)")
ax.set_title("E8: predicted TRE in the plane $z_I$ = 10 mm")
ax.set_aspect("equal")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(FOLDER / "lab5_E8_tre_map.png", dpi=150)

# ---- E10: digitised fiducials, registered CT fiducials and targets in {s}
p = load("lab5_cpp_points.csv")
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(projection="3d")
for kind, style, label in ((0, "bo", "digitised fiducials"), (1, "rx", "registered CT fiducials"), (2, "g*", "targets")):
    q = p[p[:, 0] == kind]
    ax.plot(q[:, 1], q[:, 2], q[:, 3], style, label=label)
ax.set_xlabel("$x_s$ (mm)")
ax.set_ylabel("$y_s$ (mm)")
ax.set_zlabel("$z_s$ (mm)")
ax.set_title("E10: navigation pipeline in the base frame {s}")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FOLDER / "lab5_E10_pipeline_3d.png", dpi=150)
print("saved lab5_E7_fre_tre_scatter.png, lab5_E8_tre_map.png, lab5_E10_pipeline_3d.png in", FOLDER)
