"""Plot runs saved by the C++ track of SBEG 46X Lab 8 (Figures 1 and 2 of the report).

In your C++ experiments program, save each run with   lab::write_sim_csv("run_pd.csv", sim);
(columns t, theta, theta_dot, theta_d, u, tau_cmd, tau -- the same as data/ref_elbow_*.csv), then:

    python cpp/plot_results.py run_pd.csv run_lqr.csv run_mpc.csv            # elbow limits
    python cpp/plot_results.py --joint knee knee_pd.csv knee_mpc.csv          # knee limits (Experiment E2)
    python cpp/plot_results.py data/ref_elbow_pd.csv                          # the reference runs work too

The range-of-motion, torque and speed limits are read from data/exo_joint.json.  Figures are saved as
fig1_theta.png and fig2_tau_speed.png in the current folder (and shown unless --no-show).
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="CSV files written by lab::write_sim_csv")
    ap.add_argument("--joint", default="elbow", choices=["elbow", "knee"])
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()
    j = json.loads((HERE.parent / "data" / "exo_joint.json").read_text())[a.joint]

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for i, name in enumerate(a.csv):
        d = np.genfromtxt(name, delimiter=",", names=True)
        lab = Path(name).stem
        ax1.plot(d["t"], d["theta"], label=lab)
        if i == 0:
            ax1.step(d["t"], d["theta_d"], "k--", where="post", lw=1, label=r"$\theta_d$")
        ax2.plot(d["t"], d["tau"], label=lab)
        ax3.plot(d["t"], d["theta_dot"], label=lab)
    for v in j["rom"]:
        ax1.axhline(v, color="r", ls=":", lw=1)
    for s in (-1, 1):
        ax2.axhline(s * j["tau_max"], color="r", ls=":", lw=1)
        ax3.axhline(s * j["v_max"], color="r", ls=":", lw=1)
    ax1.set(xlabel="t (s)", ylabel=r"$\theta$ (rad)", title=f"{a.joint}: joint angle (dotted: range of motion)")
    ax2.set(ylabel=r"$\tau$ (N m)", title=r"applied torque (dotted: $\pm\tau_{max}$)")
    ax3.set(xlabel="t (s)", ylabel=r"$\dot\theta$ (rad/s)", title=r"speed (dotted: $\pm v_{max}$)")
    for ax in (ax1, ax2, ax3):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig("fig1_theta.png", dpi=150)
    fig2.savefig("fig2_tau_speed.png", dpi=150)
    print("saved fig1_theta.png, fig2_tau_speed.png")
    if not a.no_show:
        plt.show()


if __name__ == "__main__":
    main()
