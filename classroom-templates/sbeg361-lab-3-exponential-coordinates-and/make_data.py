"""Generate the synthetic data sets for SBEG 361 Lab 3 (illustrative values, deterministic).

1. imu_knee.csv (seed 361)
   Scenario: two IMUs (thigh, shank) strapped to a subject walking on a treadmill, as in a gait
   laboratory, or to the thigh and shank links of a load-carriage exoskeleton during a field trial.
   Each IMU's y-axis is aligned with the knee flexion axis, so  R_shank = R_thigh Rot(y, kappa).

   Columns (time in s, angles in rad, angular velocity in rad/s, quaternions scalar-first):
     t,
     thigh_q0..thigh_q3, shank_q0..shank_q3          ground-truth orientations in the lab frame {s}
     thigh_wx..thigh_wz, shank_wx..shank_wz          ideal body-frame gyro (exact for zero-order hold)
     thigh_nwx..thigh_nwz, shank_nwx..shank_nwz      the same + constant bias + white noise
     kappa                                           ground-truth knee flexion angle

   The ideal gyro sample k is  omega_b,k = so3ToVec(log(R_k^T R_{k+1})) / dt,  so the exponential
   update R_{k+1} = R_k exp([omega_b,k] dt) reproduces the truth to machine precision.

2. screw_insertion.csv (new; seed 3610)
   Scenario: navigated pedicle-screw insertion (the set-up of the Week 2-3 labs).  The optical tracker
   {s} reports the pose T_k of the calibrated screwdriver TIP frame {d} (origin at the tip, z along the
   shaft towards the bone) as a scalar-first quaternion and a position, at 50 Hz for 8 s.  The surgeon
   drives the screw 5 full turns with a smooth, non-constant speed; the screw advances by its lead
   (2.5 mm per turn, the same illustrative screw as before), so the motion is a screw motion about a
   FIXED axis with pitch h = lead / (2 pi):   T_k = exp([S_s] theta_k) T_0.

   Columns (s, m, quaternions scalar-first, q0 >= 0):
     t, q0..q3, px, py, pz              ideal tracker output
     nq0..nq3, npx, npy, npz            the same with tracker noise (illustrative: 0.05 deg rms per
                                        rotation axis, 0.10 mm rms per position axis)

   Ground truth (students should NOT read this before the lab): axis direction S_HAT, a point Q_AXIS on
   the axis (the entry point), pitch H, total angle 10 pi, total advance 12.5 mm.

Run:  python make_data.py   (deterministic; re-running reproduces both files byte for byte)
"""
from pathlib import Path

import numpy as np
import modern_robotics as mr

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------- 1. IMU data set
FS = 100.0            # Hz
T_END = 5.0           # s
STRIDE = 1.1          # s per gait cycle (illustrative)


def rot(axis, ang):
    return mr.MatrixExp3(mr.VecToso3(np.asarray(axis, float) * ang))


def quat(R):
    """Scalar-first unit quaternion with q0 >= 0 (via scipy, independent of the lab code)."""
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    q = np.array([w, x, y, z])
    return q if q[0] >= 0 else -q


def thigh_R(t):
    ph = 2 * np.pi * t / STRIDE
    yaw = 0.30 + 0.08 * np.sin(ph)                     # heading + pelvic rotation
    pitch = 0.10 + 0.40 * np.sin(ph)                   # hip flexion/extension about y
    roll = 0.05 * np.sin(2 * ph)                       # small ab/adduction
    return rot([0, 0, 1], yaw) @ rot([0, 1, 0], pitch) @ rot([1, 0, 0], roll)


def kappa(t):
    ph = 2 * np.pi * t / STRIDE
    # two-hump knee flexion profile, 0.09 .. ~1.0 rad (illustrative)
    return 0.09 + 0.90 * (1 - np.cos(ph)) / 2 + 0.20 * np.exp(-((ph % (2 * np.pi)) - 1.0) ** 2 / 0.15)


def make_imu(out=HERE / "imu_knee.csv"):
    dt = 1.0 / FS
    t = np.arange(0.0, T_END + dt / 2, dt)
    Rth = np.array([thigh_R(ti) for ti in t])
    Rsh = np.array([Rth[i] @ rot([0, 1, 0], kappa(ti)) for i, ti in enumerate(t)])
    rng = np.random.default_rng(361)
    rows = []
    bias = {"th": np.array([0.010, -0.006, 0.004]), "sh": np.array([-0.005, 0.008, 0.006])}
    for i, ti in enumerate(t):
        if i < len(t) - 1:
            wth = mr.so3ToVec(mr.MatrixLog3(Rth[i].T @ Rth[i + 1])) / dt
            wsh = mr.so3ToVec(mr.MatrixLog3(Rsh[i].T @ Rsh[i + 1])) / dt
        else:
            wth = np.zeros(3)
            wsh = np.zeros(3)
        nth = wth + bias["th"] + rng.normal(0, 0.005, 3)
        nsh = wsh + bias["sh"] + rng.normal(0, 0.005, 3)
        rows.append(np.concatenate([[ti], quat(Rth[i]), quat(Rsh[i]), wth, wsh, nth, nsh, [kappa(ti)]]))
    hdr = ("t,thigh_q0,thigh_q1,thigh_q2,thigh_q3,shank_q0,shank_q1,shank_q2,shank_q3,"
           "thigh_wx,thigh_wy,thigh_wz,shank_wx,shank_wy,shank_wz,"
           "thigh_nwx,thigh_nwy,thigh_nwz,shank_nwx,shank_nwy,shank_nwz,kappa")
    np.savetxt(out, np.array(rows), delimiter=",", header=hdr, comments="", fmt="%.12f")
    print(f"wrote {out} ({len(rows)} samples)")


# ----------------------------------------------------------------------------- 2. screw insertion
LEAD = 0.0025                                   # m per turn (illustrative pedicle-screw lead)
H = LEAD / (2 * np.pi)                          # pitch, m/rad
TURNS = 5
THETA_TOT = 2 * np.pi * TURNS                   # rad
DT_TRK, T_TRK = 0.02, 8.0                       # 50 Hz tracker, 8 s
S_HAT = np.array([0.30, -0.20, -0.9327379053])  # axis direction in the tracker frame (into the bone)
S_HAT = S_HAT / np.linalg.norm(S_HAT)
Q_AXIS = np.array([0.120, -0.060, 1.350])       # entry point on the axis, tracker frame (m)
SIG_ROT = np.deg2rad(0.05)                      # rad rms per axis (illustrative)
SIG_POS = 0.10e-3                               # m rms per axis (illustrative)


def theta_profile(t):
    """Smooth 0 -> 10 pi rad in T_TRK seconds, speed zero at both ends (peak 2 * mean speed)."""
    s = t / T_TRK
    return THETA_TOT * (s - np.sin(2 * np.pi * s) / (2 * np.pi))


def tip_frame0():
    """Initial tip frame: origin at the entry point, z along S_HAT, x roughly along the tracker x."""
    z = S_HAT
    x = np.array([1.0, 0.0, 0.0]) - z[0] * z
    x /= np.linalg.norm(x)
    R = np.column_stack([x, np.cross(z, x), z])
    return mr.RpToTrans(R, Q_AXIS)


def make_screw(out=HERE / "screw_insertion.csv"):
    t = np.arange(0.0, T_TRK + DT_TRK / 2, DT_TRK)
    S_s = np.r_[S_HAT, -np.cross(S_HAT, Q_AXIS) + H * S_HAT]       # MR Sec. 3.3.2.2
    T0 = tip_frame0()
    rng = np.random.default_rng(3610)
    rows = []
    for ti in t:
        T = mr.MatrixExp6(mr.VecTose3(S_s * theta_profile(ti))) @ T0
        R, p = mr.TransToRp(T)
        Rn = R @ mr.MatrixExp3(mr.VecToso3(rng.normal(0, SIG_ROT, 3)))
        pn = p + rng.normal(0, SIG_POS, 3)
        rows.append(np.concatenate([[ti], quat(R), p, quat(Rn), pn]))
    hdr = "t,q0,q1,q2,q3,px,py,pz,nq0,nq1,nq2,nq3,npx,npy,npz"
    np.savetxt(out, np.array(rows), delimiter=",", header=hdr, comments="", fmt="%.12f")
    print(f"wrote {out} ({len(rows)} samples)")


def main():
    make_imu()
    make_screw()


if __name__ == "__main__":
    main()
