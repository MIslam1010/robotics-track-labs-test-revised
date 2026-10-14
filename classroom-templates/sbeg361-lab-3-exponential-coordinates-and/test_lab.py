"""Auto-checks for SBEG 361 Lab 3 -- exponential coordinates and logarithms on SO(3) and SE(3); IMU orientation.

Usage:   LAB_IMPL=starter  python -m pytest -q test_lab.py     (your work, file starter.py)
         LAB_IMPL=solution python -m pytest -q test_lab.py     (reference; default)
Ground truth: the official `modern_robotics` library (Lynch & Park), scipy.spatial.transform,
MR Example 3.12 (p. 82) and Example 3.26 (pp. 104-106), and the noise-free data models in make_data.py.
"""
import importlib
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import modern_robotics as mr
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
lab = importlib.import_module(os.environ.get("LAB_IMPL", "starter"))
import make_data as md  # noqa: E402  (noise-free truth of the synthetic data)

DATA = HERE / "imu_knee.csv"
TRACK = HERE / "screw_insertion.csv"
RNG = np.random.default_rng(2026)
RANDOM_R = [Rotation.random(random_state=s).as_matrix() for s in range(25)]
S2 = 1 / math.sqrt(2)


def rot(axis, ang):
    return mr.MatrixExp3(mr.VecToso3(np.asarray(axis, float) * ang))


def random_T(seed):
    rng = np.random.default_rng(seed)
    return mr.RpToTrans(Rotation.random(random_state=seed).as_matrix(), rng.normal(size=3))


def random_twist(seed):
    return np.random.default_rng(1000 + seed).normal(size=6)


def same_quat(q, p, tol=1e-9):
    """q and -q represent the same rotation (MR App. B.3)."""
    return np.allclose(q, p, atol=tol) or np.allclose(q, -np.asarray(p), atol=tol)


def scipy_quat(R):
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


@pytest.fixture(scope="module")
def imu():
    return np.genfromtxt(DATA, delimiter=",", names=True)


def quats(d, prefix):
    return np.column_stack([d[f"{prefix}_q{i}"] for i in range(4)])


def gyro(d, prefix, noisy=False):
    n = "n" if noisy else ""
    return np.column_stack([d[f"{prefix}_{n}w{c}"] for c in "xyz"])[:-1]


def insertion_screw():
    """True space-frame screw axis of the insertion (make_data.py)."""
    return np.r_[md.S_HAT, -np.cross(md.S_HAT, md.Q_AXIS) + md.H * md.S_HAT]


# ------------------------------------------------------------- Checkpoint 1: skew matrices
def test_vec_to_so3_matches_mr():
    for _ in range(10):
        w = RNG.normal(size=3)
        np.testing.assert_allclose(lab.vec_to_so3(w), mr.VecToso3(w), atol=1e-12)


def test_skew_properties():
    a, b = RNG.normal(size=3), RNG.normal(size=3)
    A = lab.vec_to_so3(a)
    np.testing.assert_allclose(A, -A.T, atol=1e-12)                       # [a] = -[a]^T
    np.testing.assert_allclose(A @ b, np.cross(a, b), atol=1e-12)        # [a]b = a x b
    R = RANDOM_R[0]
    np.testing.assert_allclose(R @ A @ R.T, lab.vec_to_so3(R @ a), atol=1e-12)  # MR Eq. 3.31
    np.testing.assert_allclose(lab.so3_to_vec(A), a, atol=1e-12)


def test_axis_ang3():
    w, th = lab.axis_ang3([1.0, 2.0, 2.0])
    ref_w, ref_th = mr.AxisAng3(np.array([1.0, 2.0, 2.0]))
    np.testing.assert_allclose(w, ref_w, atol=1e-12)
    assert th == pytest.approx(ref_th)
    with pytest.raises(ValueError):
        lab.axis_ang3([0.0, 0.0, 0.0])


# ------------------------------------------------------------- Checkpoint 2: exponentials, SO(3) and SE(3)
def test_matrix_exp3_mr_example_3_12():
    """MR Example 3.12: w1 = (0, 0.866, 0.5), theta1 = 30 deg."""
    w = np.array([0.0, np.sqrt(3) / 2, 0.5])
    R = lab.matrix_exp3(lab.vec_to_so3(w * np.pi / 6))
    expected = np.array([[0.866, -0.250, 0.433], [0.250, 0.967, 0.058], [-0.433, 0.058, 0.899]])
    np.testing.assert_allclose(R, expected, atol=1.5e-3)


def test_matrix_exp3_matches_mr_and_is_rotation():
    for _ in range(20):
        so3 = mr.VecToso3(RNG.normal(size=3) * 2.0)
        R = lab.matrix_exp3(so3)
        np.testing.assert_allclose(R, mr.MatrixExp3(so3), atol=1e-12)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)
    np.testing.assert_allclose(lab.matrix_exp3(np.zeros((3, 3))), np.eye(3))


def test_g_matrix_and_inverse():
    """G(theta) of MR Eq. 3.87 = its power series (Eq. 3.86); G^-1 of Eq. 3.92 is its inverse;
    G(theta) v is the translation of e^{[S]theta} (Eq. 3.88)."""
    rng = np.random.default_rng(304)
    for _ in range(8):
        w = rng.normal(size=3)
        w /= np.linalg.norm(w)
        th = rng.uniform(0.1, 3.1)
        v = rng.normal(size=3)
        G = lab.g_matrix(w, th)
        W = mr.VecToso3(w)
        series = sum(np.linalg.matrix_power(W, k) * th ** (k + 1) / math.factorial(k + 1) for k in range(40))
        np.testing.assert_allclose(G, series, atol=1e-10)
        np.testing.assert_allclose(G @ lab.g_inv_matrix(w, th), np.eye(3), atol=1e-10)
        np.testing.assert_allclose(G @ v, mr.MatrixExp6(mr.VecTose3(np.r_[w, v]) * th)[:3, 3], atol=1e-10)


def test_matrix_exp6_matches_mr():
    for k in range(20):
        V = random_twist(k)
        se3 = mr.VecTose3(V)
        np.testing.assert_allclose(lab.matrix_exp6(se3), mr.MatrixExp6(se3), atol=1e-9)
    # pure translation, MR Eq. 3.89
    se3 = mr.VecTose3([0, 0, 0, 0.03, 0.04, 0])
    np.testing.assert_allclose(lab.matrix_exp6(se3), mr.RpToTrans(np.eye(3), [0.03, 0.04, 0]), atol=1e-12)


def test_exp6_screw_through_a_point():
    """Axis z through q = (1, 0, 0), theta = 90 deg, pitch 0 and 0.1 m/rad."""
    for h, pz in ((0.0, 0.0), (0.1, 0.1 * math.pi / 2)):
        S = np.r_[0, 0, 1, -np.cross([0, 0, 1], [1, 0, 0]) + h * np.array([0, 0, 1])]
        T = lab.matrix_exp6(mr.VecTose3(S) * math.pi / 2)
        np.testing.assert_allclose(T, [[0, -1, 0, 1], [1, 0, 0, -1], [0, 0, 1, pz], [0, 0, 0, 1]], atol=1e-12)


# ------------------------------------------------------------- Checkpoint 3: logarithms, SO(3) and SE(3)
def test_matrix_log3_matches_mr_random():
    for R in RANDOM_R:
        np.testing.assert_allclose(lab.matrix_log3(R), mr.MatrixLog3(R), atol=1e-9)


def test_matrix_log3_identity_and_pi_cases():
    np.testing.assert_allclose(lab.matrix_log3(np.eye(3)), np.zeros((3, 3)))
    for w in ([0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]):
        w = np.array(w, float) / np.linalg.norm(w)
        R = rot(w, np.pi)
        L = lab.matrix_log3(R)
        v = lab.so3_to_vec(L)
        assert np.linalg.norm(v) == pytest.approx(np.pi)
        np.testing.assert_allclose(mr.MatrixExp3(L), R, atol=1e-9)        # either +w or -w is fine


def test_exp_log_round_trip_small_and_large_angles():
    for th in (1e-5, 1e-3, 0.5, 2.0, 3.0, np.pi - 1e-4):
        w = RNG.normal(size=3)
        w /= np.linalg.norm(w)
        R = mr.MatrixExp3(mr.VecToso3(w * th))
        v = lab.so3_to_vec(lab.matrix_log3(R))
        np.testing.assert_allclose(v, w * th, atol=1e-6)


def test_matrix_log6_matches_mr_and_inverts_exp():
    for k in range(20):
        T = random_T(k)
        L = lab.matrix_log6(T)
        assert L.shape == (4, 4) and np.allclose(L[3], 0)
        np.testing.assert_allclose(L, mr.MatrixLog6(T), atol=1e-8)
        np.testing.assert_allclose(mr.MatrixExp6(L), T, atol=1e-9)


def test_matrix_log6_special_cases():
    np.testing.assert_allclose(lab.matrix_log6(np.eye(4)), np.zeros((4, 4)), atol=1e-12)
    T = mr.RpToTrans(np.eye(3), [0.4, 0, 0.3])                       # case (a): R = I
    np.testing.assert_allclose(lab.se3_to_vec(lab.matrix_log6(T)), [0, 0, 0, 0.4, 0, 0.3], atol=1e-12)
    for R in (np.diag([1.0, -1, -1]), np.diag([-1.0, 1, -1]), np.diag([-1.0, -1, 1])):   # theta = pi
        T = mr.RpToTrans(R, [0.1, 0.2, -0.3])
        L = lab.matrix_log6(T)
        assert abs(np.linalg.norm(lab.so3_to_vec(L[:3, :3])) - math.pi) < 1e-9
        np.testing.assert_allclose(mr.MatrixExp6(L), T, atol=1e-9)


def test_log6_mr_example_3_26():
    """MR Example 3.26 (pp. 104-106): S = (0, 0, 1, 3.37, -3.37, 0), theta = pi/6."""
    c30, s30, c60, s60 = math.cos(math.pi / 6), 0.5, 0.5, math.sin(math.pi / 3)
    Tsb = np.array([[c30, -s30, 0, 1], [s30, c30, 0, 2], [0, 0, 1, 0], [0, 0, 0, 1]])
    Tsc = np.array([[c60, -s60, 0, 2], [s60, c60, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
    L = lab.matrix_log6(Tsc @ mr.TransInv(Tsb))
    np.testing.assert_allclose(L, mr.MatrixLog6(Tsc @ mr.TransInv(Tsb)), atol=1e-9)
    St = lab.se3_to_vec(L)
    assert abs(np.linalg.norm(St[:3]) - math.pi / 6) < 1e-9
    np.testing.assert_allclose(St / (math.pi / 6), [0, 0, 1, 3.3660, -3.3660, 0], atol=1e-4)


# ------------------------------------------------------------- Checkpoint 4: screw motions
def test_screw_between_space_and_body():
    for k in range(10):
        T1, T2 = random_T(k), random_T(k + 50)
        Ss, ts = lab.screw_between(T1, T2, "space")
        Sb, tb = lab.screw_between(T1, T2, "body")
        assert abs(np.linalg.norm(Ss[:3]) - 1) < 1e-9 and abs(ts - tb) < 1e-9
        np.testing.assert_allclose(mr.MatrixExp6(mr.VecTose3(Ss) * ts) @ T1, T2, atol=1e-9)
        np.testing.assert_allclose(T1 @ mr.MatrixExp6(mr.VecTose3(Sb) * tb), T2, atol=1e-9)
        np.testing.assert_allclose(Sb, mr.Adjoint(mr.TransInv(T1)) @ Ss, atol=1e-9)   # S_b = [Ad_{T1^-1}] S_s
    S, th = lab.screw_between(np.eye(4), mr.RpToTrans(np.eye(3), [0, 0.3, 0.4]))
    np.testing.assert_allclose(S, [0, 0, 0, 0, 0.6, 0.8], atol=1e-12)
    assert abs(th - 0.5) < 1e-12
    S, th = lab.screw_between(random_T(3), random_T(3))
    np.testing.assert_allclose(S, np.zeros(6), atol=1e-12)
    assert th == 0.0
    with pytest.raises(ValueError):
        lab.screw_between(np.eye(4), np.eye(4), "world")


def test_screw_path_needle_guide():
    """Needle guide: half-way along the screw from T1 to T2."""
    T1 = mr.RpToTrans(np.eye(3), [0.3, 0, 0.1])
    T2 = mr.RpToTrans(mr.MatrixExp3(mr.VecToso3([0, 0, math.pi / 2])), [0.2, 0.3, 0.15])
    np.testing.assert_allclose(lab.screw_path(T1, T2, 0.0), T1, atol=1e-12)
    np.testing.assert_allclose(lab.screw_path(T1, T2, 1.0), T2, atol=1e-12)
    Tm = lab.screw_path(T1, T2, 0.5)
    np.testing.assert_allclose(Tm[:3, 3], [0.1 + 0.3 * S2, 0.1 + 0.1 * S2, 0.125], atol=1e-9)


# ------------------------------------------------------------- Checkpoint 5: quaternions
def test_rot_to_quat_matches_scipy():
    for R in RANDOM_R + [rot([1, 1, 1], np.pi), rot([0, 1, 0], np.pi - 1e-7), np.eye(3)]:
        q = lab.rot_to_quat(R)
        assert np.linalg.norm(q) == pytest.approx(1.0)
        assert q[0] >= -1e-12
        assert same_quat(q, scipy_quat(R), tol=1e-7)


def test_quat_to_rot_round_trip():
    for R in RANDOM_R:
        np.testing.assert_allclose(lab.quat_to_rot(lab.rot_to_quat(R)), R, atol=1e-12)
        np.testing.assert_allclose(lab.quat_to_rot(-lab.rot_to_quat(R)), R, atol=1e-12)


def test_quat_mul_is_matrix_product():
    for Ra, Rb in zip(RANDOM_R[:12], RANDOM_R[12:24]):
        q = lab.quat_mul(lab.rot_to_quat(Ra), lab.rot_to_quat(Rb))
        assert same_quat(q, scipy_quat(Ra @ Rb), tol=1e-9)
        qc = lab.quat_conj(lab.rot_to_quat(Ra))
        np.testing.assert_allclose(lab.quat_to_rot(qc), Ra.T, atol=1e-12)


def test_workshop_quaternion_example():
    """Rot(x, 90 deg) Rot(y, 90 deg) -> q = (1, 1, 1, 1)/2, a 120 deg rotation about (1,1,1)/sqrt 3."""
    qx = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])
    qy = np.array([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0])
    np.testing.assert_allclose(lab.quat_mul(qx, qy), 0.5 * np.ones(4), atol=1e-12)


# ------------------------------------------------------------- Checkpoint 6: IMU data
def test_gyro_exponential_integration_reproduces_truth(imu):
    for seg in ("thigh", "shank"):
        q = quats(imu, seg)
        R0 = lab.quat_to_rot(q[0])
        Rs = lab.integrate_gyro_exp(R0, gyro(imu, seg), 0.01)
        assert Rs.shape == (len(q), 3, 3)
        err = max(lab.rotation_angle(Rs[k], Rotation.from_quat(np.r_[q[k, 1:], q[k, 0]]).as_matrix())
                  for k in range(len(q)))
        assert err < 1e-8


def test_euler_integration_drifts_off_SO3(imu):
    q = quats(imu, "thigh")
    Rs = lab.integrate_gyro_euler(lab.quat_to_rot(q[0]), gyro(imu, "thigh"), 0.01)
    assert Rs.shape == (len(q), 3, 3)
    # forward Euler does not preserve orthonormality: det grows above 1
    assert np.linalg.det(Rs[-1]) > 1.01
    Re = lab.integrate_gyro_exp(lab.quat_to_rot(q[0]), gyro(imu, "thigh"), 0.01)
    assert np.linalg.det(Re[-1]) == pytest.approx(1.0, abs=1e-12)


def test_knee_flexion_from_quaternions(imu):
    Rth = [lab.quat_to_rot(q) for q in quats(imu, "thigh")]
    Rsh = [lab.quat_to_rot(q) for q in quats(imu, "shank")]
    kap = np.array([lab.knee_flexion(a, b) for a, b in zip(Rth, Rsh)])
    np.testing.assert_allclose(kap, imu["kappa"], atol=1e-8)


def test_noisy_gyro_drift_is_bounded_but_nonzero(imu):
    """Bias makes integrated orientation drift (reflection question); check the order of magnitude."""
    q = quats(imu, "shank")
    Rs = lab.integrate_gyro_exp(lab.quat_to_rot(q[0]), gyro(imu, "shank", noisy=True), 0.01)
    drift = lab.rotation_angle(Rs[-1], lab.quat_to_rot(q[-1]))
    assert 0.01 < drift < 0.2


# ------------------------------------------------------------- Checkpoint 7: tracked screw insertion
def test_insertion_axis_from_two_poses():
    """Biomedical: the log of T_210 T_200^-1 (0.2 s at full speed) gives the screw axis (direction, a point, pitch) of the insertion."""
    t, Ts = lab.load_pose_track(TRACK)
    assert Ts.shape == (401, 4, 4)
    S, th = lab.screw_between(Ts[200], Ts[210], "space")
    assert abs(th - (md.theta_profile(t[210]) - md.theta_profile(t[200]))) < 1e-9
    sc = lab.twist_to_screw(S)
    np.testing.assert_allclose(sc["s_hat"], md.S_HAT, atol=1e-9)
    assert abs(sc["h"] - md.H) < 1e-9
    dq = sc["q"] - md.Q_AXIS
    assert np.linalg.norm(dq - (dq @ md.S_HAT) * md.S_HAT) < 1e-9       # q lies on the true axis
    Sb, tb = lab.screw_between(Ts[200], Ts[210], "body")                 # tip frame: z along the shaft, on the axis
    np.testing.assert_allclose(Sb, [0, 0, 1, 0, 0, md.H], atol=1e-9)
    assert abs(tb - th) < 1e-9


def test_screw_motion_profile_ideal_data():
    t, Ts = lab.load_pose_track(TRACK)
    theta, d = lab.screw_motion_profile(Ts)
    assert theta.shape == (len(Ts),) and d.shape == (len(Ts),)
    np.testing.assert_allclose(theta, md.theta_profile(t), atol=1e-8)
    np.testing.assert_allclose(d, md.H * theta, atol=1e-10)
    assert abs(theta[-1] - 10 * np.pi) < 1e-8 and abs(d[-1] - 0.0125) < 1e-10
    # one log of the whole motion cannot see the 5 full turns: R_N = R_0, so it is a pure translation
    S, th = lab.screw_between(Ts[0], Ts[-1], "space")
    np.testing.assert_allclose(S, np.r_[0, 0, 0, md.S_HAT], atol=1e-6)
    assert abs(th - 0.0125) < 1e-9


def test_screw_motion_profile_handles_reversal():
    """The surgeon backs the screw out: the angle (and the advance) must decrease."""
    S, T0 = insertion_screw(), md.tip_frame0()
    ths = np.array([0.0, 0.5, 1.5, 1.5, 1.2, 1.4, 0.9])   # largest step (1.0 rad) is forward
    Ts = np.array([mr.MatrixExp6(mr.VecTose3(S * th)) @ T0 for th in ths])
    theta, d = lab.screw_motion_profile(Ts)
    np.testing.assert_allclose(theta, ths, atol=1e-9)
    np.testing.assert_allclose(d, md.H * ths, atol=1e-12)
    with pytest.raises(ValueError):
        lab.screw_motion_profile([np.eye(4), mr.RpToTrans(np.eye(3), [0, 0, 0.01])])


def test_screw_motion_profile_noisy_data():
    """Tracker noise: the fitted lead is still within 5 % of 2.5 mm per turn."""
    _, Ts = lab.load_pose_track(TRACK, noisy=True)
    theta, d = lab.screw_motion_profile(Ts)
    h_fit = np.polyfit(theta, d, 1)[0]
    assert abs(2 * np.pi * h_fit - md.LEAD) < 0.05 * md.LEAD
    assert abs(theta[-1] - 10 * np.pi) < 0.01 * 10 * np.pi


# ------------------------------------------------------------- Checkpoint 8: EOD gripper errors
def test_orientation_error_eod_gripper():
    """Defence/EOD: gripper wrist must turn from R_cur to R_des; error in the gripper frame."""
    R_cur = rot([0, 0, 1], np.deg2rad(40)) @ rot([0, 1, 0], np.deg2rad(-20))
    R_des = R_cur @ rot([1, 0, 0], np.deg2rad(35))
    e = lab.orientation_error(R_cur, R_des)
    np.testing.assert_allclose(e, [np.deg2rad(35), 0, 0], atol=1e-12)
    for Ra, Rb in zip(RANDOM_R[:10], RANDOM_R[10:20]):
        np.testing.assert_allclose(lab.orientation_error(Ra, Rb),
                                   mr.so3ToVec(mr.MatrixLog3(Ra.T @ Rb)), atol=1e-9)
        assert 0 <= lab.rotation_angle(Ra, Rb) <= np.pi + 1e-12


def test_eod_gripper_screw_motion():
    """Defence/EOD: full pose error of the gripper as a screw motion in the gripper frame.
    Desired pose = 35 deg about an axis parallel to the gripper x-axis through (0, 0, 0.05) m (gripper frame),
    with 0.02 m/rad advance along it."""
    R_cur = rot([0, 0, 1], np.deg2rad(40)) @ rot([0, 1, 0], np.deg2rad(-20))
    T_cur = mr.RpToTrans(R_cur, [0.60, 0.10, 0.40])
    Sb_true = np.r_[1, 0, 0, -np.cross([1, 0, 0], [0, 0, 0.05]) + 0.02 * np.array([1, 0, 0])]
    T_des = T_cur @ mr.MatrixExp6(mr.VecTose3(Sb_true) * np.deg2rad(35))
    Sb, th = lab.screw_between(T_cur, T_des, "body")
    np.testing.assert_allclose(Sb, [1, 0, 0, 0.02, 0.05, 0], atol=1e-9)
    assert abs(th - np.deg2rad(35)) < 1e-9
    sc = lab.twist_to_screw(Sb)
    np.testing.assert_allclose(sc["q"], [0, 0, 0.05], atol=1e-9)
    assert abs(sc["h"] - 0.02) < 1e-9
    Ss, ts = lab.screw_between(T_cur, T_des, "space")
    np.testing.assert_allclose(Ss, mr.Adjoint(T_cur) @ Sb, atol=1e-9)
    np.testing.assert_allclose(lab.screw_path(T_cur, T_des, 0.5),
                               T_cur @ mr.MatrixExp6(mr.VecTose3(Sb_true) * np.deg2rad(17.5)), atol=1e-9)
