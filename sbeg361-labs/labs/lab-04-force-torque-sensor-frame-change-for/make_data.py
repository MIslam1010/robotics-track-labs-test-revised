"""Generate the synthetic data of SBEG 361 Lab 4 (read it AFTER the lab).

  lap_calib.csv      12 static readings of a 6-axis F/T sensor at the UR5 flange carrying a
                     0.30 m, 0.6 kg laparoscope (no contact): payload identification.
  lap_palpation.csv  10 s at 50 Hz: three palpation presses of the tool tip on tissue; during the
                     third press the abdominal-wall port (trocar) also pushes on the shaft.
  eod_lift.json      EOD 6R arm on a UGV: 6 static readings with the empty gripper, 6 with a
                     suspect package held, and one lift pose with the arm's twist.

The robots and tool are copied from the course robot models (see the local robots.py).  The F/T
convention is that of MR Example 3.28: the sensor reports the wrench of all loads acting on the
bodies distal to it (gravity, contacts), expressed in the sensor frame {f}, plus a constant bias.
All values are ILLUSTRATIVE.  Deterministic (fixed random seeds).
"""
import csv
import json
from pathlib import Path

import modern_robotics as mr
import numpy as np

from robots import DEG, G0, LAPAROSCOPE, eod_6r_ugv, trans, ur5_flange

HERE = Path(__file__).resolve().parent
G_S = np.array([0.0, 0.0, -G0])

# ---------------------------------------------------------------- laparoscope (UR5 flange sensor)
LAP_MASS = LAPAROSCOPE["m_tool"]                                  # 0.6 kg
LAP_COM = np.array([0.0, 0.0, LAPAROSCOPE["com_from_flange"]])    # (0, 0, 0.10) m in {f}
LAP_TIP = np.array([0.0, 0.0, LAPAROSCOPE["L_tool"]])             # (0, 0, 0.30) m in {f}
LAP_BIAS = np.array([0.012, -0.008, 0.004, 0.35, -0.22, 0.60])    # (m [N m], f [N])
LAP_NOISE = np.array([1e-3, 1e-3, 1e-3, 0.02, 0.02, 0.02])
TROCAR_POINT = np.array([0.0, 0.0, 0.18])                         # port on the shaft, in {f}
Q_LAP_HOME = np.array([2.6, -2.5, 2.3, -1.4, -1.57, 0.0])          # ur5_laparoscope q_home

# ---------------------------------------------------------------- EOD gripper (wrist sensor)
EOD_LG = 0.25                                                      # gripper frame 0.25 m ahead of {f}
GRIPPER_MASS, GRIPPER_COM = 3.0, np.array([0.12, 0.0, 0.0])
PACKAGE_MASS, PACKAGE_COM = 2.0, np.array([0.31, 0.015, -0.04])
EOD_BIAS = np.array([0.8, -0.5, 0.3, 4.0, -2.5, 6.0])
EOD_NOISE = np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])
Q_EOD = [[0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 90 * DEG, 0],
         [0, 0, 0, 90 * DEG, 0, 0],
         [0, -0.3, 0.6, 0, -0.5, 90 * DEG],
         [0.5, 0.2, -0.4, 1.0, 0.7, -0.8],
         [0, 0, 0, -90 * DEG, -60 * DEG, 0]]
Q_LIFT = [0.0, -0.35, 0.6, 0.0, -0.25, 0.0]
LIFT_JOINT, LIFT_RATE = 1, -0.2                                    # joint 2 (index 1) at -0.2 rad/s (raising)
FRONT_EDGE = [0.55, 0.0, -0.45]                                    # front track edge in {s}
UGV = dict(mass=220.0, com_behind_edge=0.35)                       # restoring moment m g d


def wrench_at(f, r):
    return np.r_[np.cross(r, f), f]


def gravity(mass, com, R):
    return wrench_at(mass * R.T @ G_S, com)


def press_profile(t):
    """Axial tissue force magnitude (N): three sin^2 presses of 2 s starting at 1, 4, 7 s; peak 2 N."""
    F = np.zeros_like(t)
    for t0 in (1.0, 4.0, 7.0):
        k = (t >= t0) & (t <= t0 + 2.0)
        F[k] = 2.0 * np.sin(np.pi * (t[k] - t0) / 2.0) ** 2
    return F


def palpation_truth():
    """(t, T_sf list, tip force f_t (K x 3) in {t}, trocar force in {f} (K x 3)) -- noise free."""
    S, M = ur5_flange()
    t = np.round(np.arange(0, 10.0 + 1e-9, 0.02), 4)
    Ts = []
    for tk in t:
        q = Q_LAP_HOME + np.array([0, 0, 0, 0.10 * np.sin(0.4 * np.pi * tk), 0.15 * np.sin(0.2 * np.pi * tk), 0])
        Ts.append(mr.FKinSpace(M, S, q))
    F = press_profile(t)
    f_tip = np.c_[0.15 * F, -0.05 * F, -F]               # tissue on the tool: friction + axial push-back
    f_troc = np.zeros((len(t), 3))
    k3 = t >= 7.0
    f_troc[k3, 1] = 0.25 * F[k3]                         # 0.5 N peak lateral port force, third press
    return t, Ts, f_tip, f_troc


def lap_reading(T, f_tip, f_troc, rng):
    R = T[:3, :3]
    F = LAP_BIAS + gravity(LAP_MASS, LAP_COM, R) + wrench_at(f_tip, LAP_TIP) + wrench_at(f_troc, TROCAR_POINT)
    return F + rng.normal(0, LAP_NOISE)


def write_csv(path, Ts, Fs, t=None):
    head = ([] if t is None else ["t"]) + [f"R{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)] + \
        ["px", "py", "pz", "mx", "my", "mz", "fx", "fy", "fz"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        for k, (T, F) in enumerate(zip(Ts, Fs)):
            row = ([] if t is None else [f"{t[k]:.2f}"]) + [f"{x:.9f}" for x in T[:3, :3].ravel()] + \
                [f"{x:.6f}" for x in T[:3, 3]] + [f"{x:.6f}" for x in F]
            w.writerow(row)


def eod_sensor_pose(q):
    S, M, _ = eod_6r_ugv()
    return mr.FKinSpace(M, S, q) @ trans([-EOD_LG, 0, 0])


def eod_reading(T, loaded, rng):
    R = T[:3, :3]
    F = EOD_BIAS + gravity(GRIPPER_MASS, GRIPPER_COM, R)
    if loaded:
        F = F + gravity(PACKAGE_MASS, PACKAGE_COM, R)
    return F + rng.normal(0, EOD_NOISE)


def main():
    S, M = ur5_flange()
    rng = np.random.default_rng(4)
    # -------- calibration poses: wrist joints spread over their range
    q4 = [-1.4, -0.4, 0.6, -2.4, -1.4, -1.4, 0.2, -0.8, -2.0, -1.0, 0.9, -1.9]
    q5 = [-1.57, -1.57, -1.57, -1.57, -0.3, -2.8, 0.8, -0.9, -2.2, 1.6, -1.2, 0.0]
    q6 = [0.0, 1.0, -1.0, 2.0, 0.5, -0.5, 1.5, -2.0, 2.5, -1.5, 0.3, 3.0]
    Tc = [mr.FKinSpace(M, S, np.r_[Q_LAP_HOME[:3], a, b, c]) for a, b, c in zip(q4, q5, q6)]
    Fc = [lap_reading(T, np.zeros(3), np.zeros(3), rng) for T in Tc]
    write_csv(HERE / "lap_calib.csv", Tc, Fc)
    # -------- palpation
    t, Tp, f_tip, f_troc = palpation_truth()
    Fp = [lap_reading(T, a, b, rng) for T, a, b in zip(Tp, f_tip, f_troc)]
    write_csv(HERE / "lap_palpation.csv", Tp, Fp, t)
    # -------- EOD
    rng = np.random.default_rng(5)
    Te = [eod_sensor_pose(q) for q in Q_EOD]
    empty = [dict(q=list(map(float, q)), T_sf=T.round(9).tolist(), F=eod_reading(T, False, rng).round(5).tolist())
             for q, T in zip(Q_EOD, Te)]
    loaded = [dict(q=list(map(float, q)), T_sf=T.round(9).tolist(), F=eod_reading(T, True, rng).round(5).tolist())
              for q, T in zip(Q_EOD, Te)]
    T_lift = eod_sensor_pose(Q_LIFT)
    S_e, _, _ = eod_6r_ugv()
    V_s = S_e[:, LIFT_JOINT] * LIFT_RATE
    V_f = mr.Adjoint(mr.TransInv(T_lift)) @ V_s
    data = dict(
        description="EOD 6R arm on a tracked UGV (SBEG 361 project S4 model, illustrative values). "
                    "{s} = turret base, {f} = wrist F/T sensor (axes of the gripper frame), "
                    "gripper frame {b} = 0.25 m ahead of {f} along x_f. Wrenches are (m [N m], f [N]).",
        g_s=G_S.tolist(), T_fb=trans([EOD_LG, 0, 0]).tolist(),
        empty=empty, loaded=loaded,
        lift=dict(q=Q_LIFT, T_sf=T_lift.round(9).tolist(), F=eod_reading(T_lift, True, rng).round(5).tolist(),
                  V_f=V_f.round(9).tolist(), note="joint 2 rotating at -0.2 rad/s (raising the gripper); V_f = body twist of {f}"),
        front_edge=dict(T_se=trans(FRONT_EDGE).tolist(), ugv_mass=UGV["mass"],
                        ugv_com_behind_edge=UGV["com_behind_edge"]),
    )
    with open(HERE / "eod_lift.json", "w") as fh:
        json.dump(data, fh, indent=1)
    print("wrote lap_calib.csv, lap_palpation.csv, eod_lift.json")


if __name__ == "__main__":
    main()
