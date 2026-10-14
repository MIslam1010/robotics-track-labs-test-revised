"""Plots for the C++ track of SBEG 361 Lab 5 (given; plotting stays in Python).

./build/lab_tests writes two CSV files into the folder you run it from:
  lab5_cpp_log_errors.csv   t, e_1..e_6 (meas - cmd, degrees), task_err_mm   (from test_compare_log, Step 4b-c)
  lab5_cpp_tip_fits.csv     joint, delta_deg, rms_after_mm                     (from test_tip_checks_..., Step 5b)
They are written only when your compare_log / fit_joint_offset return results of the right size.

Usage (from the folder that holds the CSV files):   python cpp/plot_results.py  [folder]
Saves lab5_step4_errors.png and prints the table of Step 5b.
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    log_csv = os.path.join(folder, "lab5_cpp_log_errors.csv")
    fit_csv = os.path.join(folder, "lab5_cpp_tip_fits.csv")
    if os.path.exists(log_csv):
        with open(log_csv) as fh:
            header = fh.readline().strip().split(",")
        D = np.loadtxt(log_csv, delimiter=",", skiprows=1, ndmin=2)
        t = D[:, header.index("t")]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        for name in header:
            if name.startswith("e_"):
                ax1.plot(t, D[:, header.index(name)], label=f"joint {name[2:]}")
        ax1.set_ylabel("meas - cmd (deg)")
        ax1.set_title("Step 4b: joint errors")
        ax1.legend(ncol=3, fontsize=8)
        err = D[:, header.index("task_err_mm")]
        ax2.plot(t, err)
        ax2.set_xlabel("t (s)")
        ax2.set_ylabel("tool-point error (mm)")
        ax2.set_title(f"Step 4c: task-space error (RMS {np.sqrt(np.mean(err ** 2)):.2f} mm, max {err.max():.2f} mm)")
        fig.tight_layout()
        fig.savefig(os.path.join(folder, "lab5_step4_errors.png"), dpi=150)
        print("saved", os.path.join(folder, "lab5_step4_errors.png"))
    else:
        print("not found:", log_csv, "(run ./build/lab_tests from this folder first)")
    if os.path.exists(fit_csv):
        F = np.loadtxt(fit_csv, delimiter=",", skiprows=1, ndmin=2)
        print("Step 5b   joint   delta (deg)   RMS after (mm)")
        for j, d, r in F:
            print(f"          {int(j):5d}   {d:11.3f}   {r:14.3f}")
    else:
        print("not found:", fit_csv, "(run ./build/lab_tests from this folder first)")


if __name__ == "__main__":
    main()
