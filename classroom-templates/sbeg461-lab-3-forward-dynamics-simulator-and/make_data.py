"""SBEG 461 Lab 3 -- regenerate the data files in data/ (instructor tool; students do not need to run it).

    ~/Documents/Edu/Lectures/Robotics/MR-Studio/.venv/bin/python make_data.py

Writes
  data/robots.json             Mlist / Glist / Slist / gravity of the three lab robots (modern_robotics format)
  data/ur5_fall_reference.csv  high-accuracy reference of MR Exercise 8.15(b): UR5 falling from the zero
                               configuration under gravity, zero torque, 0 <= t <= 3 s (SciPy DOP853, rtol = atol = 1e-11)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import modern_robotics as mr
from scipy.integrate import solve_ivp

DATA = Path(__file__).resolve().parent / "data"
G0 = 9.81


def planar_chain(L, m, r, I):
    """Planar n-R chain in the vertical x-y plane (gravity -y), joint axes along z, zero pose = links along +x.
    CoM frames {i} at distance r_i from joint i, aligned with the link (MR Sec. 8.3 conventions)."""
    n = len(L)
    jx = np.concatenate([[0.0], np.cumsum(L)])
    fx = [0.0] + [jx[i] + r[i] for i in range(n)] + [jx[n]]
    Mlist = [mr.RpToTrans(np.eye(3), [fx[i + 1] - fx[i], 0, 0]) for i in range(n + 1)]
    Glist = [np.diag([I[i]] * 3 + [m[i]] * 3) for i in range(n)]
    Slist = np.array([[0, 0, 1, 0, -jx[i], 0] for i in range(n)], float).T
    return Mlist, Glist, Slist


def ur5():
    """UR5 exactly as in MR Example 4.5 / Exercise 8.15 and the MR Ch. 8 software (UR5 parameter set)."""
    W1, L1, L2, H1, H2 = 0.10915, 0.425, 0.39225, 0.089159, 0.09465
    Slist = np.array([[0, 0, 0, 0, 0, 0],
                      [0, 1, 1, 1, 0, 1],
                      [1, 0, 0, 0, -1, 0],
                      [0, -H1, -H1, -H1, -W1, H2 - H1],
                      [0, 0, 0, 0, L1 + L2, 0],
                      [0, 0, L1, L1 + L2, 0, L1 + L2]], float)
    Mlist = [np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]]),
             np.array([[0, 0, 1, 0.28], [0, 1, 0, 0.13585], [-1, 0, 0, 0], [0, 0, 0, 1]]),
             np.array([[1, 0, 0, 0], [0, 1, 0, -0.1197], [0, 0, 1, 0.395], [0, 0, 0, 1]]),
             np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0.14225], [0, 0, 0, 1]]),
             np.array([[1, 0, 0, 0], [0, 1, 0, 0.093], [0, 0, 1, 0], [0, 0, 0, 1]]),
             np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.09465], [0, 0, 0, 1]]),
             np.array([[1, 0, 0, 0], [0, 0, 1, 0.0823], [0, -1, 0, 0], [0, 0, 0, 1]])]
    Glist = [np.diag([0.010267495893, 0.010267495893, 0.00666, 3.7, 3.7, 3.7]),
             np.diag([0.22689067591, 0.22689067591, 0.0151074, 8.393, 8.393, 8.393]),
             np.diag([0.049443313556, 0.049443313556, 0.004095, 2.275, 2.275, 2.275]),
             np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219]),
             np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219]),
             np.diag([0.0171364731454, 0.0171364731454, 0.033822, 0.1879, 0.1879, 0.1879])]
    return Mlist, Glist, Slist


def pack(Mlist, Glist, Slist, g, description):
    return {"description": description,
            "Mlist": [np.asarray(Mi).tolist() for Mi in Mlist],
            "Glist": [np.asarray(Gi).tolist() for Gi in Glist],
            "Slist": np.asarray(Slist).tolist(),
            "g": list(map(float, g))}


def main():
    DATA.mkdir(exist_ok=True)
    robots = {
        "_comment": "SBEG 461 Lab 3 robots in modern_robotics format: Mlist = [M01, M12, ..., M_n,n+1] (CoM frames, "
                    "MR Sec. 8.3.2), Glist = 6x6 spatial inertias in the CoM frames (angular block first, MR Eq. 8.32), "
                    "Slist = 6 x n space-frame screw axes (omega, v), g = gravity vector in {s} [m/s^2].",
        "mr_2r_point": pack(*planar_chain([1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]), [0, -G0, 0],
                            "MR Fig. 8.1 / Eq. 8.10 2R arm, point masses at the link ends, L1 = L2 = m1 = m2 = 1"),
        "leg_2r": pack(*planar_chain([0.45, 0.48], [7.0, 2.4], [0.20, 0.21], [0.11, 0.045]), [0, -G0, 0],
                       "Sagittal-plane thigh (residual limb + socket) and transfemoral prosthesis (shank + foot); hip = "
                       "joint 1, knee = joint 2; theta1 measured from +x (forward), hanging leg = (-90 deg, 0). "
                       "Illustrative values: L = (0.45, 0.48) m, m = (7.0, 2.4) kg, r = (0.20, 0.21) m, "
                       "I = (0.11, 0.045) kg m^2"),
        "ur5": pack(*ur5(), [0, 0, -G0],
                    "UR5 6R arm, MR Example 4.5 / Exercise 8.15 parameters; used in this lab as a stand-in for a "
                    "6-DOF arm on a tracked EOD / CBRN-inspection robot"),
    }
    with open(DATA / "robots.json", "w") as fh:
        json.dump(robots, fh, indent=1)

    Mlist, Glist, Slist = ur5()
    g = np.array([0, 0, -G0])

    def rhs(t, x):
        th, dth = x[:6], x[6:]
        return np.r_[dth, mr.ForwardDynamics(th, dth, np.zeros(6), g, np.zeros(6), Mlist, Glist, Slist)]

    ts = np.round(np.arange(0, 3.0 + 1e-9, 0.05), 10)
    sol = solve_ivp(rhs, [0, 3.0], np.zeros(12), method="DOP853", rtol=1e-11, atol=1e-11, t_eval=ts)
    assert sol.success
    header = ("# MR Exercise 8.15(b) reference: UR5 falling from theta = 0 with zero torque, g = 9.81 m/s^2 along -z_s. "
              "SciPy DOP853, rtol = atol = 1e-11. Columns: t [s], theta1..6 [rad], dtheta1..6 [rad/s]\n")
    with open(DATA / "ur5_fall_reference.csv", "w") as fh:
        fh.write(header)
        fh.write("t," + ",".join(f"theta{i}" for i in range(1, 7)) + "," + ",".join(f"dtheta{i}" for i in range(1, 7)) + "\n")
        for k, t in enumerate(sol.t):
            fh.write(f"{t:.2f}," + ",".join(f"{v:.10f}" for v in sol.y[:, k]) + "\n")
    print("wrote", DATA / "robots.json", "and", DATA / "ur5_fall_reference.csv", f"({len(sol.t)} rows)")


if __name__ == "__main__":
    main()
