"""Auto-checks for SBEG 361 Lab 10 -- resolved-rate control, damped least squares, redundancy, hexapod.

Usage:   LAB_IMPL=starter  python -m pytest -q test_lab.py     (your work, file starter.py)
         LAB_IMPL=solution python -m pytest -q test_lab.py     (reference; default)
Ground truth: NumPy (np.linalg.pinv, lstsq, svd), the official `modern_robotics` library
(FKinSpace, JacobianBody, IKinBody, MatrixLog6), finite differences of the hexapod leg lengths, and
the checkpoint values of lab_handout.md.
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

d2r = np.deg2rad
RNG = np.random.default_rng(2026)


def rand_J(m, n, k=5):
    return [RNG.normal(size=(m, n)) for _ in range(k)]


def eod_line():
    S, M, _ = lab.load_robot("eod_6r_ugv")
    q0 = d2r(np.array([0, -30, 60, 3, 5, 0.0]))
    T0 = mr.FKinSpace(M, S, q0)
    t, Xd, Vd = lab.line_trajectory(T0, [-0.10, 0, 0], 0.05, 0.02)
    return S, M, q0, t, Xd, Vd


# ---------------------------------------------------------------- Checkpoint 1: pseudo-inverse and DLS
@pytest.mark.parametrize("shape", [(2, 3), (6, 7), (6, 6), (3, 2), (6, 2)])
def test_pinv_svd_matches_numpy(shape):
    for J in rand_J(*shape):
        np.testing.assert_allclose(lab.pinv_svd(J), np.linalg.pinv(J), atol=1e-10)


def test_pinv_rank_deficient():
    """At a singularity J^+ still exists and gives the least-squares, minimum-norm solution."""
    J = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]])       # rank 1
    P = lab.pinv_svd(J)
    np.testing.assert_allclose(P, np.linalg.pinv(J), atol=1e-12)
    z = np.array([1.0, 0.0])
    np.testing.assert_allclose(P @ z, np.linalg.lstsq(J, z, rcond=None)[0], atol=1e-12)


def test_pinv_fat_is_right_inverse():
    """MR p. 227: J^+ = J^T (J J^T)^-1 for a fat full-rank J, and J J^+ = I."""
    for J in rand_J(2, 3):
        np.testing.assert_allclose(lab.pinv_svd(J), J.T @ np.linalg.inv(J @ J.T), atol=1e-10)
        np.testing.assert_allclose(J @ lab.pinv_svd(J), np.eye(2), atol=1e-10)


def test_dls_two_forms_and_limits():
    for J in rand_J(6, 6) + rand_J(2, 3):
        lam = 0.3
        n = J.shape[1]
        np.testing.assert_allclose(lab.dls_inverse(J, lam),
                                   np.linalg.solve(J.T @ J + lam ** 2 * np.eye(n), J.T), atol=1e-10)
        np.testing.assert_allclose(lab.dls_inverse(J, 0.0), np.linalg.pinv(J), atol=1e-10)
        np.testing.assert_allclose(lab.dls_inverse(J, 1e-9), np.linalg.pinv(J), atol=1e-6)


def test_dls_minimises_damped_cost():
    """J^# V minimises |J x - V|^2 + lam^2 |x|^2 (stacked least-squares problem)."""
    J, V, lam = RNG.normal(size=(6, 6)), RNG.normal(size=6), 0.2
    x = lab.dls_inverse(J, lam) @ V
    A = np.vstack([J, lam * np.eye(6)])
    b = np.r_[V, np.zeros(6)]
    np.testing.assert_allclose(x, np.linalg.lstsq(A, b, rcond=None)[0], atol=1e-10)


def test_dls_singular_gains():
    J = np.diag([0.5, 0.01])
    s, g = lab.dls_singular_gains(J, 0.05)
    np.testing.assert_allclose(s, [0.5, 0.01], atol=1e-12)
    np.testing.assert_allclose(g, [0.5 / 0.2525, 0.01 / 0.0026], atol=1e-10)
    s, g = lab.dls_singular_gains(J, 0.0)
    np.testing.assert_allclose(g, [2.0, 100.0], atol=1e-10)
    # the largest possible DLS gain is 1/(2 lam), at sigma = lam
    sig = np.linspace(1e-4, 1, 20001)
    gains = [lab.dls_singular_gains(np.array([[x]]), 0.05)[1][0] for x in sig[::500]]
    assert max(gains) <= 1 / (2 * 0.05) + 1e-9


def test_adaptive_lambda():
    assert lab.adaptive_lambda(0.06, 0.05, 0.04) == 0.0
    assert lab.adaptive_lambda(0.05, 0.05, 0.04) == 0.0
    assert lab.adaptive_lambda(0.0, 0.05, 0.04) == pytest.approx(0.04)
    assert lab.adaptive_lambda(0.03, 0.05, 0.04) == pytest.approx(0.032)


# ---------------------------------------------------------------- Checkpoint 2: redundancy
def test_null_projector_properties():
    for J in rand_J(2, 3) + rand_J(6, 7):
        P = lab.null_projector(J)
        np.testing.assert_allclose(J @ P, 0, atol=1e-10)                 # no task motion
        np.testing.assert_allclose(P @ P, P, atol=1e-10)                 # projector
        np.testing.assert_allclose(P, P.T, atol=1e-10)                   # orthogonal projector
        assert np.linalg.matrix_rank(P, tol=1e-8) == J.shape[1] - J.shape[0]


def test_null_projector_workshop_p1():
    """Workshop 12 P1: planar 3R at (0, 90, -90) deg, null space along (-1, 1, 2)."""
    J = np.array([[-0.4, -0.4, 0.0], [0.75, 0.25, 0.25]])
    n = np.array([-1.0, 1.0, 2.0]) / np.sqrt(6)
    np.testing.assert_allclose(lab.null_projector(J), np.outer(n, n), atol=1e-12)


def test_weighted_pinv():
    J = RNG.normal(size=(2, 3))
    W = np.diag([4.0, 2.0, 1.0])
    G = lab.weighted_pinv(J, W)
    np.testing.assert_allclose(J @ G, np.eye(2), atol=1e-10)
    np.testing.assert_allclose(lab.weighted_pinv(J, np.eye(3)), np.linalg.pinv(J), atol=1e-10)
    # among all solutions, G V has the smallest weighted norm
    V = np.array([0.1, -0.05])
    x = G @ V
    nvec = np.linalg.svd(J)[2][-1]
    for a in np.linspace(-1, 1, 11):
        y = x + a * nvec
        assert x @ W @ x <= y @ W @ y + 1e-12


def test_redundant_rates():
    for J in rand_J(6, 7):
        xd, q0 = RNG.normal(size=6), RNG.normal(size=7)
        qd = lab.redundant_rates(J, xd, q0)
        np.testing.assert_allclose(J @ qd, xd, atol=1e-10)                # task is met
        np.testing.assert_allclose(lab.redundant_rates(J, xd, np.zeros(7)), np.linalg.pinv(J) @ xd, atol=1e-10)


def test_joint_centering_gradient():
    lim = np.array([[-1.0, 1.0], [0.0, 2.0], [-2.0, 1.0]])
    th = np.array([0.5, 0.5, 0.0])
    h = lambda q: 0.5 * np.sum(((q - lim.mean(1)) / (lim[:, 1] - lim[:, 0])) ** 2)
    g = lab.joint_centering_gradient(th, lim)
    fd = np.array([(h(th + 1e-6 * e) - h(th - 1e-6 * e)) / 2e-6 for e in np.eye(3)])
    np.testing.assert_allclose(g, fd, atol=1e-8)
    np.testing.assert_allclose(lab.joint_centering_gradient(lim.mean(1), lim), 0, atol=1e-15)


def test_srs7r_self_motion():
    """Null-space motion of the 7R feeding arm moves the elbow but not the spoon."""
    S, M, _ = lab.load_robot("srs_7r_feeding")
    th = d2r(np.array([10, 30, 0, -70, 0, 40, 0.0]))
    T0 = mr.FKinSpace(M, S, th)
    B = mr.Adjoint(mr.TransInv(M)) @ S
    for _ in range(100):
        Jb = mr.JacobianBody(B, th)
        th = th + lab.redundant_rates(Jb, np.zeros(6), np.r_[0, 0, 0.5, 0, 0, 0, 0]) * 0.01
    err = np.linalg.norm(mr.FKinSpace(M, S, th) - T0)
    assert err < 5e-4                # Euler drift only (2.1e-4); no task feedback here
    assert abs(th[2]) > d2r(5)       # the shoulder roll ("elbow swivel") moved by several degrees


# ---------------------------------------------------------------- Checkpoint 3: resolved-rate control
def test_line_trajectory():
    T0 = mr.RpToTrans(mr.MatrixExp3(mr.VecToso3([0.3, -0.2, 0.5])), [0.2, 0.1, 0.3])
    t, Xd, Vd = lab.line_trajectory(T0, [0.03, 0.0, -0.04], 0.02, 0.01)
    assert t[0] == 0 and t[-1] == pytest.approx(2.5) and len(t) == len(Xd) == len(Vd) == 251
    np.testing.assert_allclose(Xd[-1][:3, 3], [0.23, 0.1, 0.26], atol=1e-12)
    np.testing.assert_allclose(Xd[-1][:3, :3], T0[:3, :3], atol=1e-12)
    # body twist: X_d^-1 dX_d/dt by finite difference
    Xdot = (Xd[11] - Xd[10]) / (t[11] - t[10])
    np.testing.assert_allclose(mr.se3ToVec(mr.TransInv(Xd[10]) @ Xdot), Vd[10], atol=1e-10)
    np.testing.assert_allclose(Vd[-1], 0, atol=1e-15)


def test_resolved_rate_course_arm_line():
    """Checkpoint 3/6: course arm, 0.06 m along +y_s at 0.02 m/s from q_home."""
    S, M, info = lab.load_robot("course_arm")
    q0 = np.array(info["q_home"])
    T0 = mr.FKinSpace(M, S, q0)
    t, Xd, Vd = lab.line_trajectory(T0, [0, 0.06, 0], 0.02, 0.02)
    o0 = lab.resolved_rate_sim(S, M, q0, Xd, Vd, 0.02, kp=0.0)
    o2 = lab.resolved_rate_sim(S, M, q0, Xd, Vd, 0.02, kp=2.0)
    assert o0["pos_err"][-1] == pytest.approx(7.56e-5, abs=3e-6)          # open-loop Euler drift
    assert o2["pos_err"][-1] == pytest.approx(1.50e-5, abs=1e-6)          # feedback removes most of it
    assert np.rad2deg(np.abs(o2["qdot"]).max()) == pytest.approx(11.17, abs=0.02)
    assert o2["sigma_min"].min() == pytest.approx(0.0389, abs=2e-4)
    np.testing.assert_allclose(mr.FKinSpace(M, S, o2["theta"][-1])[:3, 3], Xd[-1][:3, 3], atol=1e-4)


def test_resolved_rate_reach_boundary_detected():
    """+x_s line leaves the workspace: sigma_min collapses and the pseudo-inverse rates explode."""
    S, M, info = lab.load_robot("course_arm")
    q0 = np.array(info["q_home"])
    t, Xd, Vd = lab.line_trajectory(mr.FKinSpace(M, S, q0), [0.06, 0, 0], 0.02, 0.02)
    o = lab.resolved_rate_sim(S, M, q0, Xd, Vd, 0.02, kp=0.0)
    assert o["sigma_min"].min() < 1e-3
    assert np.rad2deg(np.abs(o["qdot"]).max()) > 1000


def test_resolved_rate_eod_wrist_singularity():
    S, M, q0, t, Xd, Vd = eod_line()
    pinv = lab.resolved_rate_sim(S, M, q0, Xd, Vd, 0.02, kp=2.0, lam=0.0)
    sat = lab.resolved_rate_sim(S, M, q0, Xd, Vd, 0.02, kp=2.0, lam=0.0, qdot_max=d2r(30))
    dls = lab.resolved_rate_sim(S, M, q0, Xd, Vd, 0.02, kp=2.0, lam=0.02)
    ada = lab.resolved_rate_sim(S, M, q0, Xd, Vd, 0.02, kp=2.0, eps=0.04, lam_max=0.04)
    assert np.rad2deg(np.abs(pinv["qdot"][:, 3]).max()) == pytest.approx(944.2, abs=1.0)
    assert np.rad2deg(np.abs(sat["qdot"]).max()) <= 30 + 1e-9
    assert sat["pos_err"][-1] * 1000 == pytest.approx(34.3, abs=0.2)
    assert np.rad2deg(np.abs(dls["qdot"]).max()) == pytest.approx(14.61, abs=0.05)
    assert dls["pos_err"].max() * 1000 == pytest.approx(2.72, abs=0.02)
    assert np.rad2deg(np.abs(ada["qdot"][:, [3, 5]]).max()) == pytest.approx(6.48, abs=0.05)
    assert ada["pos_err"].max() * 1000 == pytest.approx(2.75, abs=0.02)
    assert ada["lam"][0] == 0.0 and ada["lam"].max() > 0.039   # damping only near the singularity


# ---------------------------------------------------------------- Checkpoint 4: IK near singularities
def test_ik_newton_matches_mr_reachable():
    S, M, _ = lab.load_robot("course_arm")
    Td = M.copy()
    Td[0, 3] += 0.05
    q0 = np.array([0, 0.3, -0.6, 0.3, 0.0, 0])
    th, ok, it, err = lab.ik_newton(S, M, Td, q0, lam=0.0)
    B = mr.Adjoint(mr.TransInv(M)) @ S
    th_mr, ok_mr = mr.IKinBody(B, M, Td, q0, 1e-6, 1e-6)
    assert ok and ok_mr and it == 4 and len(err) == it + 1
    np.testing.assert_allclose(mr.FKinSpace(M, S, th), Td, atol=1e-6)
    np.testing.assert_allclose(th, th_mr, atol=1e-6)


def test_ik_newton_damping_slows_reachable():
    S, M, _ = lab.load_robot("course_arm")
    Td = M.copy()
    Td[0, 3] += 0.05
    q0 = np.array([0, 0.3, -0.6, 0.3, 0.0, 0])
    its = [lab.ik_newton(S, M, Td, q0, lam=lam)[2] for lam in (0.0, 0.01, 0.02, 0.05)]
    assert its == [4, 6, 10, 37]


def test_ik_newton_unreachable():
    """Target 0.08 m beyond home along x_s: pseudo-inverse steps jump about, DLS settles at the
    closest reachable pose (elbow straight) with a 14.4 mm residual."""
    S, M, _ = lab.load_robot("course_arm")
    Td = M.copy()
    Td[0, 3] += 0.08
    q0 = np.array([0, 0.3, -0.6, 0.3, 0.0, 0])
    _, ok0, _, e0 = lab.ik_newton(S, M, Td, q0, lam=0.0)
    th2, ok2, _, e2 = lab.ik_newton(S, M, Td, q0, lam=0.02)
    assert not ok0 and not ok2
    assert max(e[1] for e in e0) > 0.4
    assert e2[-1][1] == pytest.approx(0.0144, abs=2e-4)
    assert abs(e2[-1][1] - e2[-2][1]) < 1e-6
    assert abs(th2[2]) < 1e-3


# ---------------------------------------------------------------- Checkpoint 5: hexapod
def hexapod_pose():
    p = np.array([0.004, -0.003, 0.252])
    R = mr.MatrixExp3(mr.VecToso3(d2r(np.array([1.0, -0.5, 1.5]))))
    return p, R


def test_hexapod_leg_lengths_home():
    A, B, info = lab.load_hexapod()
    s = lab.hexapod_leg_lengths([0, 0, 0.25], np.eye(3), A, B)
    np.testing.assert_allclose(s, 0.297619 * np.ones(6), atol=1e-6)


def test_hexapod_inverse_jacobian_fd():
    """sdot = J^-1 V_s checked by differentiating s along a space-frame twist: T(e) = e^[V]e T."""
    A, B, _ = lab.load_hexapod()
    p, R = hexapod_pose()
    T = mr.RpToTrans(R, p)
    Ji = lab.hexapod_inverse_jacobian(p, R, A, B)
    h = 1e-6
    for V in np.eye(6):
        Tp = mr.MatrixExp6(mr.VecTose3(V * h)) @ T
        Tm = mr.MatrixExp6(mr.VecTose3(-V * h)) @ T
        fd = (lab.hexapod_leg_lengths(Tp[:3, 3], Tp[:3, :3], A, B)
              - lab.hexapod_leg_lengths(Tm[:3, 3], Tm[:3, :3], A, B)) / (2 * h)
        np.testing.assert_allclose(Ji @ V, fd, atol=1e-8)


def test_hexapod_jacobian_home_values():
    A, B, _ = lab.load_hexapod()
    Ji = lab.hexapod_inverse_jacobian([0, 0, 0.25], np.eye(3), A, B)
    np.testing.assert_allclose(Ji[0], [-0.04376, -0.24817, -0.10080, -0.47790, -0.25692, 0.84000], atol=1e-5)
    np.testing.assert_allclose(np.linalg.cond(Ji), 8.333, atol=1e-3)


def test_hexapod_forward_kinematics():
    A, B, _ = lab.load_hexapod()
    p, R = hexapod_pose()
    s = lab.hexapod_leg_lengths(p, R, A, B)
    p2, R2, it = lab.hexapod_forward_kinematics(s, A, B, [0, 0, 0.25], np.eye(3))
    np.testing.assert_allclose(p2, p, atol=1e-10)
    np.testing.assert_allclose(R2, R, atol=1e-10)
    assert 2 <= it <= 6


# ---------------------------------------------------------------- Checkpoint 6: hardware log
def test_tracking_report_example_log():
    S, M, _ = lab.load_robot("course_arm")
    t, cmd, meas = lab.read_joint_log(HERE / "data" / "hw_log_example.csv")
    r = lab.tracking_report(t, cmd, meas, S, M)
    assert set(r) == {"joint_rms_deg", "joint_max_deg", "task_rms_mm", "task_max_mm", "peak_speed_deg_s"}
    assert r["joint_rms_deg"].max() == pytest.approx(0.362, abs=0.002)
    assert r["joint_max_deg"].max() == pytest.approx(0.761, abs=0.002)
    assert r["task_rms_mm"] == pytest.approx(1.21, abs=0.01)
    assert r["task_max_mm"] == pytest.approx(2.02, abs=0.01)
    assert r["peak_speed_deg_s"].max() == pytest.approx(15.38, abs=0.01)


def test_tracking_report_zero_for_perfect_log():
    S, M, _ = lab.load_robot("course_arm")
    t = np.linspace(0, 1, 11)
    cmd = np.outer(t, np.ones(6)) * 10
    r = lab.tracking_report(t, cmd, cmd, S, M)
    assert r["task_max_mm"] == 0 and np.all(r["joint_max_deg"] == 0)
    np.testing.assert_allclose(r["peak_speed_deg_s"], 10.0, atol=1e-9)
