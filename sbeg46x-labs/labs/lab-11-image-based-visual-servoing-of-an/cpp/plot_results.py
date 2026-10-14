"""Plot the CSV files written by the C++ auto-checks of SBEG 46X Lab 11 (C++ track; plotting stays in Python).

Run from the folder where you ran ./build/lab_tests (the CSV files are written there):
    python cpp/plot_results.py

    lab11_cpp_ibvs.csv, lab11_cpp_pbvs.csv  -> lab11_fig1_image_paths.png (feature paths in the image, Corke start)
                                               lab11_fig2_error.png       (rms image error per step, log scale)
    lab11_cpp_endoscope.csv                 -> lab11_fig4_endoscope.png   (image error, insertion depth and port error)
Figures 3 and 5 of the report (depth estimates, roll about the optical axis, field of view, calibration error,
gain/latency sweeps) come from your own runs: write CSV files with sbeg::write_csv and adapt this script.
"""
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

P_STAR = np.array([[312, 312], [312, 712], [712, 712], [712, 312]], float)   # desired view (data/expected.json)


def load(name):
    if not os.path.exists(name):
        print(f"{name} not found (run ./build/lab_tests first; the test that writes it must pass)")
        return None
    return np.genfromtxt(name, delimiter=",", names=True)


def main():
    ib, pb = load("lab11_cpp_ibvs.csv"), load("lab11_cpp_pbvs.csv")
    if ib is not None and pb is not None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        for ax, d, title in ((axs[0], ib, "IBVS (true depth)"), (axs[1], pb, "PBVS")):
            for i in range(1, 5):
                ax.plot(d[f"u{i}"], d[f"v{i}"], "-", lw=1.5)
                ax.plot(d[f"u{i}"][0], d[f"v{i}"][0], "o", color="k", ms=5)
            ax.plot(P_STAR[:, 0], P_STAR[:, 1], "*", color="crimson", ms=12, label="desired p*")
            ax.set(xlim=(0, 1024), ylim=(1024, 0), xlabel="u (px)", title=title, aspect="equal")
            ax.grid(alpha=0.3)
        axs[0].set_ylabel("v (px)")
        axs[0].legend(loc="lower right")
        fig.tight_layout()
        fig.savefig("lab11_fig1_image_paths.png", dpi=200)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(ib["k"], ib["err"], label=f"IBVS ({int(ib['k'][-1])} steps)")
        ax.semilogy(pb["k"], pb["err"], "--", label=f"PBVS ({int(pb['k'][-1])} steps)")
        ax.set(xlabel="step k (h = 0.1 s)", ylabel="rms image error (px)")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
        fig.tight_layout()
        fig.savefig("lab11_fig2_error.png", dpi=200)
        print("saved lab11_fig1_image_paths.png, lab11_fig2_error.png")
    en = load("lab11_cpp_endoscope.csv")
    if en is not None:
        fig, axs = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
        axs[0].plot(en["t"], en["err"])
        axs[0].set_ylabel("tip distance from\nimage centre (px)")
        axs[1].plot(en["t"], 1e3 * en["ell"])
        axs[1].set_ylabel("insertion depth\nell (mm)")
        axs[2].plot(en["t"], 1e3 * en["e_lat"])
        axs[2].set(ylabel="port offset from\nscope axis (mm)", xlabel="t (s)")
        for ax in axs:
            ax.grid(alpha=0.3)
            ax.axvline(3.0, color="0.5", lw=0.8, ls=":")
        fig.tight_layout()
        fig.savefig("lab11_fig4_endoscope.png", dpi=200)
        sel = en["t"] >= 5.0
        print(f"saved lab11_fig4_endoscope.png; rms error (t >= 5 s) = {np.sqrt(np.mean(en['err'][sel] ** 2)):.2f} px, "
              f"max port offset = {1e3 * en['e_lat'].max():.3f} mm")


if __name__ == "__main__":
    main()
