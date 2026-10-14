"""Auto-checks for SBEG 46X Lab 5 (navigation pipeline: pivot calibration, paired-point registration, FLE/FRE/TRE).

    LAB_IMPL=starter  pytest -q test_lab.py      # untouched starter: every test that calls your code fails with
                                                 # NotImplementedError; the 3 data/infrastructure tests pass

LAB_IMPL is the name of the module to test (set it to starter, or to the name of your own copy of starter.py); it
is imported from this folder.
Ground truth: modern_robotics (TransInv, MatrixExp3), an independent standard-DH forward kinematics of the course arm
(DH table of the course robot models), Horn's unit-quaternion registration (a different algorithm from the SVD),
the eigenvalue form of the Fitzpatrick-West-Maurer TRE formula, brute-force searches written in this file and the
hand values of the Week 5 workshop.
"""
import importlib
import itertools
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import modern_robotics as mr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("LAB_IMPL", "starter")
lab = importlib.import_module(os.environ["LAB_IMPL"])
import course_models as cm  # noqa: E402

RNG = np.random.default_rng(5)


# ============================================================================ independent helpers
def dh_fk_mm(q):
    """Standard DH FK of the course arm flange (table in Course resources), base = I, tool = Rot(z, pi); mm."""
    d = [cm.H1, 0, 0, cm.W1, cm.H2, cm.W2]
    a = [0, cm.L1, cm.L2, 0, 0, 0]
    al = [-np.pi / 2, 0, 0, -np.pi / 2, np.pi / 2, 0]
    T = np.eye(4)
    for i in range(6):
        ct, st, ca, sa = np.cos(q[i]), np.sin(q[i]), np.cos(al[i]), np.sin(al[i])
        T = T @ np.array([[ct, -st * ca, st * sa, a[i] * ct], [st, ct * ca, -ct * sa, a[i] * st],
                          [0, sa, ca, d[i]], [0, 0, 0, 1]])
    T = T @ np.diag([-1.0, -1.0, 1.0, 1.0])
    T[:3, 3] *= 1000.0
    return T


def horn(X, Y):
    """Horn (1987) closed-form registration with unit quaternions: R, t with Y ~ R X + t."""
    xb, yb = X.mean(0), Y.mean(0)
    S = (X - xb).T @ (Y - yb)
    Sxx, Sxy, Sxz, Syx, Syy, Syz, Szx, Szy, Szz = S.ravel()
    N = np.array([[Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx],
                  [Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz],
                  [Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy],
                  [Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz]])
    w, V = np.linalg.eigh(N)
    q0, qx, qy, qz = V[:, -1]
    R = np.array([[q0 ** 2 + qx ** 2 - qy ** 2 - qz ** 2, 2 * (qx * qy - q0 * qz), 2 * (qx * qz + q0 * qy)],
                  [2 * (qy * qx + q0 * qz), q0 ** 2 - qx ** 2 + qy ** 2 - qz ** 2, 2 * (qy * qz - q0 * qx)],
                  [2 * (qz * qx - q0 * qy), 2 * (qz * qy + q0 * qx), q0 ** 2 - qx ** 2 - qy ** 2 + qz ** 2]])
    return R, yb - R @ xb


def tre_eig(X, r, fle):
    """Fitzpatrick-West-Maurer with the eigenvalues L_k of sum x~ x~^T:  f_k^2 = (L_i + L_j)/N (i, j != k)."""
    X = np.asarray(X, float)
    N = len(X)
    Xc = X - X.mean(0)
    L, V = np.linalg.eigh(Xc.T @ Xc)
    rr = V.T @ (np.asarray(r, float) - X.mean(0))
    tot = 0.0
    for k in range(3):
        i, j = [m for m in range(3) if m != k]
        tot += (rr[i] ** 2 + rr[j] ** 2) / ((L[i] + L[j]) / N)
    return math.sqrt(fle ** 2 / N * (1 + tot / 3))


def rand_R():
    return mr.MatrixExp3(mr.VecToso3(RNG.normal(size=3)))


PH = lab.load_phantom()
X_FID = PH["fiducials_image_mm"]


# ============================================================================ data and infrastructure
def test_data_files_load():
    assert X_FID.shape == (6, 3) and PH["check_points_image_mm"].shape == (2, 3)
    assert set(PH["targets_image_mm"]) == {"T1_central", "T2_lateral", "T3_deep"}
    for name, n in (("pivot_joints.csv", 10), ("fiducial_joints.csv", 6), ("check_joints.csv", 2)):
        labels, Q = lab.load_joints(name)
        assert Q.shape[1] == 6 and len(labels) == len(Q) >= n


def test_course_model_matches_dh_table():
    """course_models (PoE) and the DH table of the course robot models give the same flange pose."""
    k = cm.course_arm_kinematics()
    for q in [np.zeros(6), cm.Q_HOME, RNG.uniform(-2, 2, 6)]:
        T = mr.FKinSpace(k["M"], k["Slist"], q)
        T[:3, 3] *= 1000
        assert np.allclose(T, dh_fk_mm(q), atol=1e-9)


def test_logged_angles_are_encoder_counts_inside_soft_limits():
    k = cm.course_arm_kinematics()
    for name in ("pivot_joints.csv", "fiducial_joints.csv", "check_joints.csv"):
        _, Q = lab.load_joints(name)
        counts = Q / cm.ENC_RES
        assert np.allclose(counts, np.round(counts), atol=2e-3)
        assert np.all(np.abs(Q) <= k["limits"][:, 1] + 1e-9)


# ============================================================================ Part A
def test_A1_flange_pose_mm_matches_dh():
    for _ in range(5):
        q = RNG.uniform(-2.5, 2.5, 6)
        assert np.allclose(lab.flange_pose_mm(q), dh_fk_mm(q), atol=1e-8)
    T0 = lab.flange_pose_mm(np.zeros(6))
    assert np.allclose(T0[:3, 3], [206.0, 109.0, 56.1], atol=1e-9)


def test_A2_pivot_calibration_exact_and_workshop_values():
    p_tip, p_piv = np.array([2.0, -1.0, 95.0]), np.array([30.0, 40.0, 500.0])
    Rs = [rand_R() for _ in range(6)]
    ps = [p_piv - R @ p_tip for R in Rs]
    a, b, rms = lab.pivot_calibration(Rs, ps)
    assert np.allclose(a, p_tip, atol=1e-9) and np.allclose(b, p_piv, atol=1e-9) and rms < 1e-9
    # Workshop Problem 2 (three poses: I, Ry(90), Rx(-90))
    Ry = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)
    Rx = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
    pw = [np.array(v, float) for v in ([100, -52, 1050], [-50, -52, 1200], [100, -200, 1202])]
    a, b, rms = lab.pivot_calibration([np.eye(3), Ry, Rx], pw)
    assert np.allclose(a, [0, 2, 150], atol=1e-9) and np.allclose(b, [100, -50, 1200], atol=1e-9)


def test_A2_pivot_calibration_least_squares_with_noise():
    p_tip, p_piv = np.array([0.5, 0.3, 80.0]), np.array([10.0, 0.0, 200.0])
    Rs = [rand_R() for _ in range(15)]
    ps = [p_piv - R @ p_tip + RNG.normal(0, 0.3, 3) for R in Rs]
    a, b, rms = lab.pivot_calibration(Rs, ps)
    # independent: eliminate p_piv with differences to the mean pose, then solve for p_tip
    Rm, pm = np.mean(Rs, 0), np.mean(ps, 0)
    A = np.vstack([R - Rm for R in Rs])
    y = np.concatenate([pm - p for p in ps])
    tip, *_ = np.linalg.lstsq(A, y, rcond=None)
    assert np.allclose(a, tip, atol=1e-8)
    assert np.allclose(b, Rm @ tip + pm, atol=1e-8)
    res = [R @ tip + p - (Rm @ tip + pm) for R, p in zip(Rs, ps)]
    assert rms == pytest.approx(np.sqrt(np.mean(np.sum(np.square(res), 1))), abs=1e-9)


def test_A3_pivot_from_joint_log():
    _, Q = lab.load_joints("pivot_joints.csv")
    a, b, rms = lab.pivot_from_joint_log(Q)
    Ts = [dh_fk_mm(q) for q in Q]
    A = np.vstack([np.hstack([T[:3, :3], -np.eye(3)]) for T in Ts])
    x, *_ = np.linalg.lstsq(A, -np.concatenate([T[:3, 3] for T in Ts]), rcond=None)
    assert np.allclose(a, x[:3], atol=1e-8) and np.allclose(b, x[3:], atol=1e-8)
    assert abs(np.linalg.norm(a) - PH["pointer_nominal_length_mm"]) < 2.0
    assert 0.0 < rms < 0.5


def test_A4_tip_position():
    q = RNG.uniform(-2, 2, 6)
    p_tip = np.array([1.0, -2.0, 70.0])
    T = dh_fk_mm(q)
    assert np.allclose(lab.tip_position_mm(q, p_tip), T[:3, :3] @ p_tip + T[:3, 3], atol=1e-8)


# ============================================================================ Part B
def test_B1_register_points_recovers_exact_transform():
    R, t = rand_R(), RNG.normal(0, 50, 3)
    X = RNG.normal(0, 40, (5, 3))
    T = lab.register_points(X, X @ R.T + t)
    assert T.shape == (4, 4) and np.allclose(T[:3, :3], R, atol=1e-9) and np.allclose(T[:3, 3], t, atol=1e-8)
    assert np.allclose(T[3], [0, 0, 0, 1])


def test_B1_register_points_matches_horn_with_noise():
    for N in (3, 4, 8):
        R, t = rand_R(), RNG.normal(0, 50, 3)
        X = RNG.normal(0, 40, (N, 3))
        Y = X @ R.T + t + RNG.normal(0, 0.5, (N, 3))
        T = lab.register_points(X, Y)
        Rh, th = horn(X, Y)
        assert np.allclose(T[:3, :3], Rh, atol=1e-7) and np.allclose(T[:3, 3], th, atol=1e-6)


def test_B1_no_reflection_for_coplanar_points():
    """Coplanar fiducials: the plain SVD formula V U^T may be a reflection; the fix must return det R = +1."""
    X = np.array([[0, 0, 0], [60, 0, 0], [0, 40, 0], [60, 40, 0]], float)
    Y = X * np.array([1, 1, -1]) + RNG.normal(0, 0.01, X.shape)   # mirror image (plus noise)
    T = lab.register_points(X, Y)
    assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-9)


def test_B1_workshop_problem3_values():
    X = np.array([[0, 0, 0], [60, 0, 0], [0, 90, 0]], float)
    Y = np.array([[100, 50, 20], [136, 98, 20], [28, 104, 20]], float)
    T = lab.register_points(X, Y)
    assert np.allclose(T[:3, :3], [[0.6, -0.8, 0], [0.8, 0.6, 0], [0, 0, 1]], atol=1e-12)
    assert np.allclose(T[:3, 3], [100, 50, 20], atol=1e-9)


def test_B2_apply_T_shapes():
    T = mr.RpToTrans(rand_R(), [1.0, 2.0, 3.0])
    p = RNG.normal(size=3)
    P = RNG.normal(size=(4, 3))
    assert lab.apply_T(T, p).shape == (3,) and lab.apply_T(T, P).shape == (4, 3)
    assert np.allclose(lab.apply_T(T, P)[2], (T @ np.r_[P[2], 1])[:3])


def test_B3_fre():
    X = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], float)
    Y = X + np.array([[0.3, 0, 0], [0, 0.4, 0], [0, 0, 0]])
    assert lab.fre(X, Y, np.eye(4)) == pytest.approx(math.sqrt((0.09 + 0.16) / 3), abs=1e-12)


# ============================================================================ Part C
def test_C1_C2_fre_formulae():
    assert lab.fre_expected(4, 0.4) == pytest.approx(0.4 * math.sqrt(0.5), abs=1e-12)
    assert lab.fre_expected(3, 1.0) == pytest.approx(1 / math.sqrt(3), abs=1e-12)
    assert lab.fle_from_fre(lab.fre_expected(6, 0.7), 6) == pytest.approx(0.7, abs=1e-12)


def test_C3_tre_predict_matches_eigen_form_and_workshop():
    X = np.array([[60, 0, 0], [-60, 0, 0], [0, 40, 0], [0, -40, 0]], float)
    assert lab.tre_predict(X, [0, 0, -80], 0.4) == pytest.approx(math.sqrt(0.04 * (1 + (8 + 6400 / 1800) / 3)),
                                                                 abs=1e-12)
    assert lab.tre_predict(X, [0, 0, 0], 0.4) == pytest.approx(0.2, abs=1e-12)
    for _ in range(5):
        Xr = RNG.normal(0, 50, (5, 3))
        r = RNG.normal(0, 80, 3)
        assert lab.tre_predict(Xr, r, 0.6) == pytest.approx(tre_eig(Xr, r, 0.6), rel=1e-10)
    # invariant under a rigid motion of fiducials and target together
    R, t = rand_R(), RNG.normal(0, 20, 3)
    r = np.array([10.0, 20.0, -30.0])
    assert lab.tre_predict(X @ R.T + t, R @ r + t, 0.5) == pytest.approx(lab.tre_predict(X, r, 0.5), rel=1e-10)


def test_C4_monte_carlo_agrees_with_prediction():
    X = np.array([[70, 0, 0], [-70, 0, 0], [0, 60, 40], [0, -60, 40]], float)
    tg = np.array([[10, 20, -30], [0, 7.5, 15]], float)
    f, e = lab.monte_carlo(X, tg, 0.5, n=4000, seed=1)
    assert f.shape == (4000,) and e.shape == (4000, 2)
    assert np.sqrt(np.mean(f ** 2)) == pytest.approx(lab.fre_expected(4, 0.5), rel=0.04)
    for j in range(2):
        assert np.sqrt(np.mean(e[:, j] ** 2)) == pytest.approx(tre_eig(X, tg[j], 0.5), rel=0.05)
    f2, e2 = lab.monte_carlo(X, tg, 0.5, n=50, seed=1)
    assert np.allclose(f2, f[:50]) and np.allclose(e2, e[:50])        # same seed, same first samples
    assert abs(np.corrcoef(f, e[:, 0])[0, 1]) < 0.1                     # FRE says (almost) nothing about TRE


def test_C5_best_subset_brute_force():
    target = PH["targets_image_mm"]["T2_lateral"]
    for k in (3, 4, 6):
        idx, val = lab.best_subset(X_FID, target, k, 0.4)
        vals = {c: tre_eig(X_FID[list(c)], target, 0.4) for c in itertools.combinations(range(6), k)}
        best = min(vals, key=vals.get)
        assert tuple(idx) == best and val == pytest.approx(vals[best], rel=1e-10)


# ============================================================================ Part D
def test_D1_chain_robot_from_image():
    T_CR = mr.RpToTrans(rand_R(), RNG.normal(0, 500, 3))
    T_CP = mr.RpToTrans(rand_R(), RNG.normal(0, 500, 3))
    T_PI = mr.RpToTrans(rand_R(), RNG.normal(0, 50, 3))
    T = lab.chain_robot_from_image(T_CR, T_CP, T_PI)
    p_I = RNG.normal(0, 30, 3)
    p_C = (T_CP @ T_PI @ np.r_[p_I, 1])[:3]
    assert np.allclose((T @ np.r_[p_I, 1])[:3], (mr.TransInv(T_CR) @ np.r_[p_C, 1])[:3], atol=1e-8)


def test_D2_navigation_pipeline():
    out = lab.navigation_pipeline(n_mc=1500, seed=0)
    a, _, _ = lab.pivot_from_joint_log(lab.load_joints("pivot_joints.csv")[1])
    assert np.allclose(out["p_tip"], a, atol=1e-9)
    _, Qf = lab.load_joints("fiducial_joints.csv")
    Y = np.array([dh_fk_mm(q)[:3, :3] @ a + dh_fk_mm(q)[:3, 3] for q in Qf])
    assert np.allclose(out["Y_fid"], Y, atol=1e-8)
    Rh, th = horn(X_FID, Y)
    assert np.allclose(out["T_sI"][:3, :3], Rh, atol=1e-7) and np.allclose(out["T_sI"][:3, 3], th, atol=1e-5)
    assert out["fre"] == pytest.approx(np.sqrt(np.mean(np.sum((Y - (X_FID @ Rh.T + th)) ** 2, 1))), abs=1e-6)
    assert out["fle_est"] == pytest.approx(out["fre"] / math.sqrt(1 - 2 / 6), abs=1e-9)
    for k, r in PH["targets_image_mm"].items():
        assert out["tre_pred"][k] == pytest.approx(tre_eig(X_FID, r, out["fle_est"]), rel=1e-9)
        assert out["tre_mc"][k] == pytest.approx(out["tre_pred"][k], rel=0.08)
        assert np.allclose(out["targets_s"][k], Rh @ r + th, atol=1e-5)
    assert out["check_err"].shape == (2,) and np.all(out["check_err"] < 1.5)
