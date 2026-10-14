"""Auto-checks for SBEG 361 Lab 7 -- Jacobian implementation and finite-difference check.

Usage:   LAB_IMPL=starter  python -m pytest -q test_lab.py     (your work, file starter.py)
         LAB_IMPL=solution python -m pytest -q test_lab.py     (reference; default)
Ground truth: the official `modern_robotics` library (JacobianSpace, JacobianBody, FKinSpace,
Adjoint), MR Example 5.2 (RRRP space Jacobian, printed p. 179), MR Example 5.5 / Eq. 5.25 and the
2R Jacobian of MR Eq. 5.1 (p. 170, numbers on p. 171).
"""
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import modern_robotics as mr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
lab = importlib.import_module(os.environ.get("LAB_IMPL", "starter"))

ROBOTS = ["course_arm", "desktop_6r", "ur5", "eod_6r_ugv", "rrp"]
d2r = np.deg2rad


def model(name):
    S, M, info = lab.load_robot(name)
    return S, M, mr.Adjoint(mr.TransInv(M)) @ S, info


def configs(n, k=6, seed=7):
    rng = np.random.default_rng(seed)
    return [rng.uniform(-np.pi, np.pi, n) for _ in range(k)]


def fk_of(S, M):
    return lambda q: mr.FKinSpace(M, S, q)


# ---------------------------------------------------------------- Checkpoint 1: geometric Jacobians
@pytest.mark.parametrize("name", ROBOTS)
def test_jacobian_space_matches_mr(name):
    S, M, B, _ = model(name)
    for q in configs(S.shape[1]):
        np.testing.assert_allclose(lab.jacobian_space(S, q), mr.JacobianSpace(S, q), atol=1e-10)


@pytest.mark.parametrize("name", ROBOTS)
def test_jacobian_body_matches_mr(name):
    S, M, B, _ = model(name)
    for q in configs(S.shape[1]):
        np.testing.assert_allclose(lab.jacobian_body(B, q), mr.JacobianBody(B, q), atol=1e-10)


def test_first_and_last_columns():
    """J_s1 = S_1 and J_bn = B_n for every configuration (MR Defs. 5.1, 5.4)."""
    S, M, B, _ = model("course_arm")
    for q in configs(6):
        np.testing.assert_allclose(lab.jacobian_space(S, q)[:, 0], S[:, 0], atol=1e-12)
        np.testing.assert_allclose(lab.jacobian_body(B, q)[:, -1], B[:, -1], atol=1e-12)


def test_mr_example_5_2_rrrp():
    """Space Jacobian of the spatial RRRP chain, MR Example 5.2 (Fig. 5.7, printed p. 179)."""
    L1, L2 = 0.4, 0.3
    S = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, -L1, 0], [0, 0, 1, 0, -(L1 + L2), 0],
                  [0, 0, 0, 0, 0, 1]], float).T
    for q in configs(4, k=4, seed=52):
        c1, s1 = np.cos(q[0]), np.sin(q[0])
        c12, s12 = np.cos(q[0] + q[1]), np.sin(q[0] + q[1])
        expected = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 0],
                             [0, L1 * s1, L1 * s1 + L2 * s12, 0],
                             [0, -L1 * c1, -L1 * c1 - L2 * c12, 0], [0, 0, 0, 1]])
        np.testing.assert_allclose(lab.jacobian_space(S, q), expected, atol=1e-12)


def test_dependence_on_joints():
    """J_s does not depend on theta_n; J_b does not depend on theta_1 (MR Sec. 5.1.1-5.1.2)."""
    S, M, B, _ = model("course_arm")
    q = configs(6, k=1, seed=3)[0]
    q2 = q.copy(); q2[-1] += 1.0
    q3 = q.copy(); q3[0] += 1.0
    np.testing.assert_allclose(lab.jacobian_space(S, q2), lab.jacobian_space(S, q), atol=1e-12)
    np.testing.assert_allclose(lab.jacobian_body(B, q3), lab.jacobian_body(B, q), atol=1e-12)


@pytest.mark.parametrize("name", ROBOTS)
def test_space_body_relation(name):
    """J_b = [Ad_Tbs] J_s and J_s = [Ad_Tsb] J_b (MR Eqs. 5.22-5.23)."""
    S, M, B, _ = model(name)
    for q in configs(S.shape[1], k=3):
        T = mr.FKinSpace(M, S, q)
        Js, Jb = mr.JacobianSpace(S, q), mr.JacobianBody(B, q)
        np.testing.assert_allclose(lab.body_from_space(Js, T), Jb, atol=1e-10)
        np.testing.assert_allclose(lab.space_from_body(Jb, T), Js, atol=1e-10)


def test_point_jacobian_2r_mr_eq_5_1():
    """For the planar 2R with L1 = L2 = 1 the point Jacobian equals MR Eq. 5.1; at (0, pi/4) it is
    [[-0.71, -0.71], [1.71, 0.71]] (MR p. 171)."""
    S = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, -1, 0]], float).T
    M = mr.RpToTrans(np.eye(3), [2, 0, 0])
    for q in [np.array([0, np.pi / 4]), np.array([0.3, -1.1]), np.array([2.0, 0.7])]:
        p = mr.FKinSpace(M, S, q)[:3, 3]
        Jp = lab.point_jacobian(mr.JacobianSpace(S, q), p)
        s1, c1, s12, c12 = np.sin(q[0]), np.cos(q[0]), np.sin(q.sum()), np.cos(q.sum())
        J51 = np.array([[-s1 - s12, -s12], [c1 + c12, c12]])
        np.testing.assert_allclose(Jp[:2], J51, atol=1e-12)
        np.testing.assert_allclose(Jp[2], 0, atol=1e-12)
    p = mr.FKinSpace(M, S, [0, np.pi / 4])[:3, 3]
    Jp = lab.point_jacobian(mr.JacobianSpace(S, [0, np.pi / 4]), p)
    np.testing.assert_allclose(Jp[:2], [[-0.7071, -0.7071], [1.7071, 0.7071]], atol=1e-4)


def test_point_jacobian_equals_R_Jbv():
    """pdot = v_s + w_s x p = R_sb v_b."""
    S, M, B, _ = model("desktop_6r")
    for q in configs(6, k=3):
        T = mr.FKinSpace(M, S, q)
        Jp = lab.point_jacobian(mr.JacobianSpace(S, q), T[:3, 3])
        np.testing.assert_allclose(Jp, T[:3, :3] @ mr.JacobianBody(B, q)[3:], atol=1e-12)


# ---------------------------------------------------------------- Checkpoint 2: finite differences
@pytest.mark.parametrize("name", ["course_arm", "ur5", "eod_6r_ugv"])
def test_fd_space_and_body_central(name):
    S, M, B, _ = model(name)
    fk = fk_of(S, M)
    for q in configs(S.shape[1], k=3):
        np.testing.assert_allclose(lab.fd_jacobian_space(fk, q, 1e-5), mr.JacobianSpace(S, q), atol=1e-8)
        np.testing.assert_allclose(lab.fd_jacobian_body(fk, q, 1e-5), mr.JacobianBody(B, q), atol=1e-8)


def test_fd_forward_is_first_order_and_central_second_order():
    """Halving h halves the forward-difference error and quarters the central-difference error."""
    S, M, B, _ = model("course_arm")
    fk, q = fk_of(S, M), np.array([0.3, -0.7, 1.2, -1.9, -1.2, 0.5])
    Js = mr.JacobianSpace(S, q)
    hs = np.array([4e-3, 2e-3, 1e-3])
    ef = lab.fd_error_study(fk, Js, q, hs, "forward")
    ec = lab.fd_error_study(fk, Js, q, hs, "central")
    assert ef.shape == (3,) and ec.shape == (3,)
    assert np.all(1.8 < ef[:-1] / ef[1:]) and np.all(ef[:-1] / ef[1:] < 2.2)
    assert np.all(3.6 < ec[:-1] / ec[1:]) and np.all(ec[:-1] / ec[1:] < 4.4)
    assert 0.9 < lab.fit_slope(hs, ef) < 1.1 and 1.9 < lab.fit_slope(hs, ec) < 2.1


def test_fd_bad_scheme_raises():
    S, M, B, _ = model("rrp")
    with pytest.raises(ValueError):
        lab.fd_jacobian_space(fk_of(S, M), [0.1, 0.2, 0.3], 1e-5, "backward")


@pytest.mark.parametrize("h", [0.5, 0.1, 1e-3])
def test_log_jacobian_is_exact(h):
    """One joint moving alone is a screw motion about J_si: log(T(q + h e_i) T(q)^-1) = [J_si] h."""
    S, M, B, _ = model("course_arm")
    for q in configs(6, k=3):
        np.testing.assert_allclose(lab.log_jacobian_space(fk_of(S, M), q, h), mr.JacobianSpace(S, q),
                                   atol=1e-9)


# ---------------------------------------------------------------- Checkpoint 3: analytic Jacobian
def test_a_matrix_properties():
    np.testing.assert_allclose(lab.a_matrix([0, 0, 0]), np.eye(3), atol=1e-12)
    r = np.array([0.3, -0.5, 0.9])
    np.testing.assert_allclose(lab.a_matrix(r) @ r, r, atol=1e-12)      # rotation about r itself: w_b = rdot
    A = lab.a_matrix([0, 0, np.pi / 2])                                    # lecture notes WE-B5 value
    np.testing.assert_allclose(A, [[0.6366, 0.6366, 0], [-0.6366, 0.6366, 0], [0, 0, 1]], atol=1e-4)


def test_a_matrix_against_finite_differences():
    """w_b = A(r) rdot with [w_b] = R^T Rdot (MR Example 5.5, Exercise 5.10)."""
    rng = np.random.default_rng(510)
    for _ in range(5):
        r = rng.normal(size=3); r *= 2.5 / np.linalg.norm(r) * rng.uniform(0.1, 1)
        rdot = rng.normal(size=3)
        h = 1e-6
        Rp, Rm = mr.MatrixExp3(mr.VecToso3(r + h * rdot)), mr.MatrixExp3(mr.VecToso3(r - h * rdot))
        R = mr.MatrixExp3(mr.VecToso3(r))
        wb = mr.so3ToVec(R.T @ (Rp - Rm) / (2 * h))
        np.testing.assert_allclose(lab.a_matrix(r) @ rdot, wb, atol=1e-7)


@pytest.mark.parametrize("name", ["course_arm", "eod_6r_ugv"])
def test_analytic_jacobian(name):
    """Eq. 5.25 against central differences of q = (r, x), away from |r| = pi."""
    S, M, B, _ = model(name)
    fk = fk_of(S, M)
    checked = 0
    for q in configs(6, k=12, seed=11):
        T = fk(q)
        if np.linalg.norm(mr.so3ToVec(mr.MatrixLog3(T[:3, :3]))) > 2.8:
            continue
        Ja = lab.analytic_jacobian(mr.JacobianBody(B, q), T)
        np.testing.assert_allclose(Ja, lab.fd_analytic_jacobian(fk, q, 1e-6), atol=1e-6)
        checked += 1
    assert checked >= 4


def test_analytic_position_rows_equal_point_jacobian():
    S, M, B, _ = model("course_arm")
    q = np.array([0.2, -0.5, 1.0, -2.0, -1.5, 0.3])
    T = mr.FKinSpace(M, S, q)
    Ja = lab.analytic_jacobian(mr.JacobianBody(B, q), T)
    np.testing.assert_allclose(Ja[3:], lab.point_jacobian(mr.JacobianSpace(S, q), T[:3, 3]), atol=1e-12)


# ---------------------------------------------------------------- Checkpoint 4: EOD singularity + teleop data
def test_eod_wrist_singularity_rank():
    """EOD arm at (0, -60, 60, 0, 0, 0) deg (shoulder at its -60 deg limit): axes 4 and 6 are
    collinear -> rank 5 for J_s and J_b."""
    S, M, B, _ = model("eod_6r_ugv")
    q = d2r([0, -60, 60, 0, 0, 0])
    Js, Jb = lab.jacobian_space(S, q), lab.jacobian_body(B, q)
    assert np.linalg.matrix_rank(Js, tol=1e-9) == 5
    assert np.linalg.matrix_rank(Jb, tol=1e-9) == 5
    np.testing.assert_allclose(Js[:, 3], Js[:, 5], atol=1e-12)
    np.testing.assert_allclose(Js[:, 3], [1, 0, 0, 0, 0.2 + 0.8 * np.sin(np.pi / 3), 0], atol=1e-12)


def test_teleop_tip_velocity():
    t, Q = lab.load_teleop_csv()
    S, M, B, _ = model("desktop_6r")
    Qd = np.gradient(Q, t, axis=0)
    V = lab.tip_velocity_series(S, M, Q, Qd)
    assert V.shape == (len(t), 3)
    P = np.array([mr.FKinSpace(M, S, q)[:3, 3] for q in Q])
    Pd = np.gradient(P, t, axis=0)
    assert np.max(np.abs(V - Pd)) < 1e-4                         # Jacobian vs differentiated FK
    speed = np.linalg.norm(V, axis=1)
    assert speed.max() == pytest.approx(0.0541, abs=5e-4)        # value quoted in the handout
    assert t[speed.argmax()] == pytest.approx(1.83, abs=0.02)
