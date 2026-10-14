"""Auto-checks for SBEG 46X Lab 1 (software RCM resolved-rate control of the course arm with a laparoscope).

    LAB_IMPL=starter python -m pytest -q test_lab.py    # your work is in starter.py; always set LAB_IMPL

LAB_IMPL is the name of the module to test; it is imported from this folder (an untouched starter makes every test
fail with NotImplementedError).  Reference values come from modern_robotics (FKinSpace,
JacobianSpace, JacobianBody), finite differences of the forward kinematics and closed-form geometry.
Notation: Modern Robotics -- V = (w, v); point velocity pdot = v_s + w_s x p (MR Eq. 3.73).
"""
import importlib
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import modern_robotics as mr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
lab = importlib.import_module(os.environ.get("LAB_IMPL", "starter"))
from course_models import course_arm_kinematics  # noqa: E402

ROBOT = course_arm_kinematics()
TASK = lab.load_task()
RNG = np.random.default_rng(461)
THETAS = [TASK["theta0"]] + [TASK["theta0"] + RNG.uniform(-0.4, 0.4, 6) for _ in range(4)]


def fk_ref(theta):
    return mr.FKinSpace(ROBOT["M"], ROBOT["Slist"], theta)


@lru_cache(maxsize=None)
def sim(controller):
    return lab.simulate(ROBOT, TASK, controller=controller, dt=TASK["dt"], **TASK["gains"])


# ---------------------------------------------------------------- Checkpoint 0: your SBEG 361 library
@pytest.mark.local_only  # needs files outside a single-lab repository; not graded
@pytest.mark.skipif(lab.KIN_BACKEND != "sbeg361_kinematics", reason="sbeg361_kinematics not importable (fallback)")
def test_361_library_matches_modern_robotics():
    for th in THETAS:
        np.testing.assert_allclose(lab.tool_pose(ROBOT, th), fk_ref(th), atol=1e-9)
        np.testing.assert_allclose(lab.jacobian_space(ROBOT["Slist"], th), mr.JacobianSpace(ROBOT["Slist"], th),
                                   atol=1e-9)


# ---------------------------------------------------------------- Part A
@pytest.mark.parametrize("k", range(len(THETAS)))
def test_tool_pose_matches_modern_robotics(k):
    np.testing.assert_allclose(lab.tool_pose(ROBOT, THETAS[k]), fk_ref(THETAS[k]), atol=1e-12)


def test_start_pose_shaft_through_port():
    T = lab.tool_pose(ROBOT, TASK["theta0"])
    p_tip, z = lab.shaft(T)
    np.testing.assert_allclose(p_tip, T[:3, 3])
    np.testing.assert_allclose(z, T[:3, 2])
    np.testing.assert_allclose(z, [0, 0, -1], atol=1e-5)          # shaft straight down at the start
    geo = lab.rcm_geometry(T, TASK["p_port"])
    assert geo["depth"] == pytest.approx(0.03, abs=1e-6)            # 30 mm beyond the port
    assert geo["dist"] < 1e-6


def test_rcm_geometry_by_hand():
    # tip 0.064 m beyond the hole along z = (0.6, 0, -0.8): Practice Problem 4 numbers
    T = np.eye(4)
    z = np.array([0.6, 0.0, -0.8])
    x = np.array([0.0, 1.0, 0.0])
    T[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    T[:3, 3] = [0.62, 0.05, 0.28]
    geo = lab.rcm_geometry(T, [0.58, 0.06, 0.33])
    assert geo["depth"] == pytest.approx(0.064, abs=1e-12)
    np.testing.assert_allclose(geo["p_c"], [0.5816, 0.05, 0.3312], atol=1e-12)
    np.testing.assert_allclose(geo["e"], [-0.0016, 0.01, -0.0012], atol=1e-12)
    assert geo["dist"] == pytest.approx(np.sqrt(0.0016 ** 2 + 0.01 ** 2 + 0.0012 ** 2), abs=1e-12)
    assert abs(geo["e"] @ z) < 1e-12                               # error is perpendicular to the shaft


@pytest.mark.parametrize("z", [[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 1, 0], [0.6, 0, -0.8], [1, 2, 3]])
def test_perp_basis(z):
    z = np.asarray(z, float) / np.linalg.norm(z)
    N = lab.perp_basis(z)
    assert N.shape == (3, 2)
    np.testing.assert_allclose(N.T @ N, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(N.T @ z, np.zeros(2), atol=1e-12)
    assert np.linalg.det(np.column_stack([N, z])) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------- Part B
@pytest.mark.parametrize("k", range(len(THETAS)))
def test_point_jacobian_tip_vs_body_jacobian(k):
    th = THETAS[k]
    T = fk_ref(th)
    Js = mr.JacobianSpace(ROBOT["Slist"], th)
    Jb = mr.JacobianBody(mr.Adjoint(mr.TransInv(ROBOT["M"])) @ ROBOT["Slist"], th)
    # tip = origin of {b}: its velocity in {s} is R_sb v_b
    np.testing.assert_allclose(lab.point_jacobian(Js, T[:3, 3]), T[:3, :3] @ Jb[3:], atol=1e-12)


@pytest.mark.parametrize("k", range(len(THETAS)))
def test_point_jacobian_finite_difference(k):
    th = THETAS[k]
    T = fk_ref(th)
    p_body = np.array([0.0, 0.0, -0.05])                # a shaft point 5 cm behind the tip, in {b}
    p = T[:3, :3] @ p_body + T[:3, 3]
    Js = mr.JacobianSpace(ROBOT["Slist"], th)
    eps = 1e-6
    cols = []
    for i in range(6):
        d = np.zeros(6)
        d[i] = eps
        Tp, Tm = fk_ref(th + d), fk_ref(th - d)
        cols.append(((Tp[:3, :3] @ p_body + Tp[:3, 3]) - (Tm[:3, :3] @ p_body + Tm[:3, 3])) / (2 * eps))
    np.testing.assert_allclose(lab.point_jacobian(Js, p), np.column_stack(cols), atol=1e-8)


def test_rcm_jacobian_shape_rank_and_rows():
    th = TASK["theta0"]
    A, info = lab.rcm_jacobian(ROBOT, th, TASK["p_port"])
    assert A.shape == (5, 6)
    assert np.linalg.matrix_rank(A, tol=1e-6) == 5
    Js = mr.JacobianSpace(ROBOT["Slist"], th)
    T = fk_ref(th)
    np.testing.assert_allclose(A[:3], lab.point_jacobian(Js, T[:3, 3]), atol=1e-12)
    np.testing.assert_allclose(info["T"], T, atol=1e-12)
    N = info["N"]
    np.testing.assert_allclose(N.T @ T[:3, 2], np.zeros(2), atol=1e-12)
    # the null space of A: the tip and the port point do not move sideways
    v = np.linalg.svd(A)[2][-1]
    p_c = info["geometry"]["p_c"]
    np.testing.assert_allclose(lab.point_jacobian(Js, T[:3, 3]) @ v, np.zeros(3), atol=1e-10)
    lat = (np.eye(3) - np.outer(T[:3, 2], T[:3, 2])) @ lab.point_jacobian(Js, p_c) @ v
    np.testing.assert_allclose(lat, np.zeros(3), atol=1e-10)


def test_rcm_rows_predict_lateral_port_motion():
    """Rows 4-5 of A predict the first-order change of N^T e when the joints move (e = p_port - p_c)."""
    th = TASK["theta0"] + np.array([0.05, -0.03, 0.04, 0.02, -0.05, 0.1])
    A, info = lab.rcm_jacobian(ROBOT, th, TASK["p_port"])
    N = info["N"]
    qdot = np.array([0.2, -0.1, 0.3, 0.1, -0.2, 0.4])
    h = 1e-6
    e_p = lab.rcm_geometry(fk_ref(th + h * qdot), TASK["p_port"])["e"]
    e_m = lab.rcm_geometry(fk_ref(th - h * qdot), TASK["p_port"])["e"]
    de = (e_p - e_m) / (2 * h)
    # p_c moves with the lateral part of the shaft point velocity (sliding along z does not change e)
    np.testing.assert_allclose(N.T @ de, -(A[3:] @ qdot), atol=2e-6)


@pytest.mark.parametrize("lam", [0.0, 0.01, 0.1])
def test_dls_inverse(lam):
    A = RNG.normal(size=(5, 6))
    X = lab.dls_inverse(A, lam)
    if lam == 0.0:
        np.testing.assert_allclose(X, np.linalg.pinv(A), atol=1e-10)
    else:
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        np.testing.assert_allclose(X, Vt.T @ np.diag(s / (s ** 2 + lam ** 2)) @ U.T, atol=1e-10)


# ---------------------------------------------------------------- Part C
def test_rcm_rates_satisfy_both_tasks():
    th = TASK["theta0"]
    T = fk_ref(th)
    p_d, v_d = T[:3, 3] + [0.002, -0.001, 0.0], np.array([0.01, 0.0, -0.005])
    qdot = lab.rcm_rates(ROBOT, th, TASK["p_port"], p_d, v_d, kp=5.0, k_rcm=10.0)
    Js = mr.JacobianSpace(ROBOT["Slist"], th)
    np.testing.assert_allclose(lab.point_jacobian(Js, T[:3, 3]) @ qdot, v_d + 5.0 * (p_d - T[:3, 3]), atol=1e-10)
    A, info = lab.rcm_jacobian(ROBOT, th, TASK["p_port"])
    np.testing.assert_allclose(A[3:] @ qdot, 10.0 * info["N"].T @ info["geometry"]["e"], atol=1e-10)
    # minimum norm: no component in the null space of A
    v = np.linalg.svd(A)[2][-1]
    assert abs(v @ qdot) < 1e-10


def test_tip_only_rates():
    th = TASK["theta0"]
    T = fk_ref(th)
    p_d, v_d = T[:3, 3], np.array([0.0, 0.01, 0.0])
    qdot = lab.tip_only_rates(ROBOT, th, p_d, v_d, kp=5.0)
    Jt = lab.point_jacobian(mr.JacobianSpace(ROBOT["Slist"], th), T[:3, 3])
    np.testing.assert_allclose(Jt @ qdot, v_d, atol=1e-10)
    np.testing.assert_allclose(qdot, np.linalg.pinv(Jt) @ v_d, atol=1e-10)


def test_scale_rates():
    qmax = np.full(6, np.deg2rad(40.0))
    q = np.array([0.9, -0.3, 0.6, 0.2, 0.0, -0.1])
    out = lab.scale_rates(q, qmax)
    np.testing.assert_allclose(out, q * qmax[0] / 0.9, atol=1e-12)   # Practice Problem 5(d)
    assert np.max(np.abs(out) / qmax) == pytest.approx(1.0)
    small = 0.1 * q
    np.testing.assert_allclose(lab.scale_rates(small, qmax), small)


def test_simulation_rcm_tracks_and_keeps_the_port():
    log = sim("rcm")
    K = int(round(sum(s["T"] for s in TASK["path"]) / TASK["dt"]))
    assert log["theta"].shape == (K + 1, 6)
    assert log["p_tip"].shape == (K + 1, 3)
    acc = TASK["acceptance"]
    assert 1e3 * log["tip_err"].max() < acc["max_tip_error_mm"]
    assert 1e3 * log["rcm_err"].max() < acc["max_rcm_error_mm"]
    k_ins = int(round(TASK["path"][0]["T"] / TASK["dt"]))
    assert log["depth"][k_ins] == pytest.approx(0.05, abs=5e-4)      # inserted to 50 mm
    assert log["depth"].max() == pytest.approx(np.hypot(0.05, 0.01), abs=5e-4)   # circle: tip 10 mm off-axis
    assert log["depth"][-1] == pytest.approx(0.03, abs=5e-4)         # retracted to 30 mm
    lim = ROBOT["limits"]
    assert np.all(log["theta"] > lim[:, 0]) and np.all(log["theta"] < lim[:, 1])
    rates = np.abs(np.diff(log["theta"], axis=0)) / TASK["dt"]
    assert np.all(rates <= ROBOT["qdot_max"] + 1e-9)
    # logged tip positions agree with modern_robotics
    for k in (0, K // 3, K):
        np.testing.assert_allclose(log["p_tip"][k], fk_ref(log["theta"][k])[:3, 3], atol=1e-12)


def test_simulation_tip_only_violates_the_port():
    log_tip, log_rcm = sim("tip"), sim("rcm")
    assert 1e3 * log_tip["tip_err"].max() < TASK["acceptance"]["max_tip_error_mm"]   # the tip task is fine ...
    assert 1e3 * log_tip["rcm_err"].max() > 5.0                                          # ... but the port is torn
    assert log_tip["rcm_err"].max() > 20 * log_rcm["rcm_err"].max()


# ---------------------------------------------------------------- Part D
@pytest.mark.parametrize("L, d, r", [(0.36, 0.12, -0.5), (0.36, 0.24, -2.0), (0.36, 0.06, -0.2), (0.36, 0.18, -1.0)])
def test_fulcrum_ratio(L, d, r):
    assert lab.fulcrum_ratio(L, d) == pytest.approx(r, abs=1e-12)


def test_scaled_tip_command():
    R = mr.MatrixExp3(mr.VecToso3([0, 0, np.pi / 2]))
    np.testing.assert_allclose(lab.scaled_tip_command([15.0, 0, 0], 1 / 3, R), [0, 5.0, 0], atol=1e-12)
    np.testing.assert_allclose(lab.scaled_tip_command([3.0, -6.0, 9.0], 0.2), [0.6, -1.2, 1.8], atol=1e-12)
