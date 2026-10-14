"""Regenerate robots.json, states.csv and the two trajectory files for SBEG 461 Lab 2 (run once; outputs are committed).

    ~/Documents/Edu/Lectures/Robotics/MR-Studio/.venv/bin/python make_data.py

Robots (all in Modern Robotics list format: Mlist = [M_{0,1}, ..., M_{n,n+1}], Glist = [G_1..G_n] in centre-of-mass
frames, Slist = space screw axes as columns, g = gravity vector in {0}):
  mr_2r_point : MR Section 8.1 / Fig. 8.3 point-mass 2R arm, L1 = L2 = m1 = m2 = 1, gravity along -y.
  arm_exo_2r  : Lab 1 sagittal-plane arm + exoskeleton (same parameters as Week 1 data/arms.json), illustrative values.
  eod_3r      : Lab 1 vertical-plane arm of a tracked EOD robot (link 3 includes a 2 kg suspect package), illustrative values.
  ur5         : nominal UR5 model used with the Modern Robotics course software (links 1-3 as in the
                modern_robotics InverseDynamics docstring example), gravity along -z.
Planar links (arm_exo_2r, eod_3r) are slender bodies along x: I = diag(0.2 I_zz, I_zz, I_zz) about the CoM.
"""
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def trans(p, R=np.eye(3)):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def planar(L, m, r, Izz, g=9.81):
    """Planar n-R chain in the x-y plane, joint axes along z, zero configuration along +x, CoM r_i from joint i."""
    n = len(L)
    joints = np.r_[0, np.cumsum(L)[:-1]]
    com = joints + np.asarray(r)
    Mabs = [trans([x, 0, 0]) for x in com] + [trans([sum(L), 0, 0])]
    Mlist = [Mabs[0]] + [np.linalg.inv(Mabs[i - 1]) @ Mabs[i] for i in range(1, n + 1)]
    Glist = [np.diag([0.2 * I, I, I, mi, mi, mi]) for I, mi in zip(Izz, m)]
    Slist = np.array([[0, 0, 1, 0, -x, 0] for x in joints], float).T
    return Mlist, Glist, Slist, [0, -g, 0]


def point_2r(L1, L2, m1, m2, g=9.81):
    Mlist = [trans([L1, 0, 0]), trans([L2, 0, 0]), np.eye(4)]
    Glist = [np.diag([0, 0, 0, m1, m1, m1]), np.diag([0, 0, 0, m2, m2, m2])]
    Slist = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, -L1, 0]], float).T
    return Mlist, Glist, Slist, [0, -g, 0]


def ur5():
    Mlist = [trans([0, 0, 0.089159]),
             trans([0.28, 0.13585, 0], [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
             trans([0, -0.1197, 0.395]),
             trans([0, 0, 0.14225], [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
             trans([0, 0.093, 0]),
             trans([0, 0, 0.09465]),
             trans([0, 0.0823, 0], [[1, 0, 0], [0, 0, 1], [0, -1, 0]])]
    Glist = [np.diag([0.010267495893, 0.010267495893, 0.00666, 3.7, 3.7, 3.7]),
             np.diag([0.22689067591, 0.22689067591, 0.0151074, 8.393, 8.393, 8.393]),
             np.diag([0.049443313556, 0.049443313556, 0.004095, 2.275, 2.275, 2.275]),
             np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219]),
             np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219]),
             np.diag([0.0171364731454, 0.0171364731454, 0.033822, 0.1879, 0.1879, 0.1879])]
    Slist = np.array([[0, 0, 1, 0, 0, 0],
                      [0, 1, 0, -0.089159, 0, 0],
                      [0, 1, 0, -0.089159, 0, 0.425],
                      [0, 1, 0, -0.089159, 0, 0.81725],
                      [0, 0, -1, -0.10915, 0.81725, 0],
                      [0, 1, 0, 0.005491, 0, 0.81725]], float).T
    return Mlist, Glist, Slist, [0, 0, -9.81]


ROBOTS = {
    "mr_2r_point": ("MR Section 8.1 / Fig. 8.3 point-mass 2R arm: L1 = L2 = m1 = m2 = 1, gravity -y", point_2r(1, 1, 1, 1)),
    "arm_exo_2r": ("Sagittal-plane shoulder/elbow arm + exoskeleton (Week 1 values), illustrative",
                   planar([0.30, 0.35], [2.6, 2.0], [0.13, 0.17], [0.025, 0.020])),
    "eod_3r": ("Vertical-plane shoulder/elbow/wrist arm of a tracked EOD robot, link 3 incl. 2 kg package (Week 1 values), illustrative",
               planar([0.60, 0.50, 0.15], [8.0, 5.0, 3.5], [0.30, 0.25, 0.10], [0.24, 0.10, 0.008])),
    "ur5": ("Nominal UR5 model used with the Modern Robotics course software, gravity -z", ur5()),
}


def main():
    out = {"_comment": __doc__.strip().splitlines()[0] + " See make_data.py for the meaning of each field."}
    for name, (desc, (Ml, Gl, Sl, g)) in ROBOTS.items():
        out[name] = {"description": desc,
                     "Mlist": [np.round(M, 12).tolist() for M in Ml],
                     "Glist": [np.round(G, 12).tolist() for G in Gl],
                     "Slist": np.round(Sl, 12).tolist(),
                     "g": [float(x) for x in g]}
    (HERE / "robots.json").write_text(json.dumps(out, indent=1))

    rng = np.random.default_rng(2461)
    with open(HERE / "states.csv", "w", newline="") as fh:
        fh.write("# robot,theta(n),dtheta(n),ddtheta(n),Ftip(6)   rad, rad/s, rad/s^2, N m / N (Ftip in {n+1}, MR convention)\n")
        w = csv.writer(fh)
        for name, (_, (Ml, Gl, Sl, g)) in ROBOTS.items():
            n = Sl.shape[1]
            for k in range(4):
                th = rng.uniform(-np.pi, np.pi, n)
                dth = rng.uniform(-2, 2, n) if k else np.zeros(n)
                ddth = rng.uniform(-3, 3, n) if k else np.zeros(n)
                Ft = np.round(rng.normal(0, 2, 6), 3) if k >= 2 else np.zeros(6)
                w.writerow([name] + [f"{x:.6f}" for x in np.r_[th, dth, ddth, Ft]])

    def quintic(name, fname, th0, th1, Tf, header):
        t = np.linspace(0, Tf, 201)
        s = t / Tf
        S = 10 * s**3 - 15 * s**4 + 6 * s**5            # quintic time scaling, MR Section 9.2.1
        dS = (30 * s**2 - 60 * s**3 + 30 * s**4) / Tf
        ddS = (60 * s - 180 * s**2 + 120 * s**3) / Tf**2
        d = np.asarray(th1) - np.asarray(th0)
        with open(HERE / fname, "w", newline="") as fh:
            fh.write(header)
            n = len(th0)
            fh.write("t," + ",".join([f"theta{i+1}" for i in range(n)] + [f"dtheta{i+1}" for i in range(n)]
                                     + [f"ddtheta{i+1}" for i in range(n)]) + "\n")
            for k in range(len(t)):
                row = np.r_[t[k], th0 + S[k] * d, dS[k] * d, ddS[k] * d]
                fh.write(",".join(f"{x:.9f}" for x in row) + "\n")

    quintic("eod_3r", "eod_lift.csv", np.array([-0.9, 1.6, -0.7]), np.array([0.8, -0.9, 0.1]), 2.0,
            "# eod_3r: lift a suspect package from the ground to a carry pose, quintic time scaling, Tf = 2 s (illustrative)\n")
    quintic("arm_exo_2r", "exo_reach.csv", np.array([-1.5, 0.2]), np.array([0.3, 1.4]), 1.2,
            "# arm_exo_2r: forward reach (shoulder flexion + elbow flexion), quintic time scaling, Tf = 1.2 s (illustrative)\n")


if __name__ == "__main__":
    main()
