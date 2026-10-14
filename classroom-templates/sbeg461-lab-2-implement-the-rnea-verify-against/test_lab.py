"""Auto-checks for SBEG 461 Lab 2 (RNEA, verified against modern_robotics and Pinocchio).

    LAB_IMPL=solution pytest -q test_lab.py      # reference: all pass
    LAB_IMPL=starter  pytest -q test_lab.py      # untouched starter: every test fails (NotImplementedError)

LAB_IMPL is the name of the module to test (default "solution"); it is imported from this folder.
Ground truth: modern_robotics (InverseDynamics, MassMatrix, VelQuadraticForces, GravityForces, EndEffectorForces,
JacobianBody, FKinSpace, InverseDynamicsTrajectory), Pinocchio pin.rnea, the classical Newton/Euler equations
(MR Eqs. 8.22-8.23), the closed-form 2R dynamics (MR Eq. 8.10) and the Week 2 workshop hand calculation (W3).
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

ROBOTS = ["mr_2r_point", "arm_exo_2r", "eod_3r", "ur5"]
TOL = 1e-9


def robot(name):
    return lab.load_robot(name)       # provided (identical) in starter.py and the reference implementation


def states(name):
    return lab.load_states(name)


def args(r):
    return r["Mlist"], r["Glist"], r["Slist"]


def rand_state(rng, n):
    return rng.uniform(-np.pi, np.pi, n), rng.uniform(-2, 2, n), rng.uniform(-3, 3, n)


# ---------------------------------------------------------------- checkpoint 1: one rigid body
def test_spatial_inertia_blocks():
    I = np.array([[0.2, 0.01, 0], [0.01, 0.3, 0.02], [0, 0.02, 0.4]])
    G = lab.spatial_inertia(I, 2.5)
    assert G.shape == (6, 6)
    np.testing.assert_allclose(G[:3, :3], I)
    np.testing.assert_allclose(G[3:, 3:], 2.5 * np.eye(3))
    np.testing.assert_allclose(G[:3, 3:], 0)
    np.testing.assert_allclose(G[3:, :3], 0)


def test_transform_spatial_inertia_workshop_W1():
    """Prosthetic shank + foot about the knee (workshop W1): rotational block = Steiner, off-diagonal = m[q]^T."""
    Ic = np.array([[0.04412, 0, 0.00564], [0, 0.04895, 0], [0.00564, 0, 0.00741]])
    m, c = 2.0, np.array([0.02, 0, -0.294])
    Gk = lab.transform_spatial_inertia(lab.spatial_inertia(Ic, m), mr.RpToTrans(np.eye(3), -c))
    np.testing.assert_allclose(Gk[:3, :3], Ic + m * (c @ c * np.eye(3) - np.outer(c, c)), atol=1e-12)
    np.testing.assert_allclose(Gk[:3, 3:], [[0, 0.588, 0], [-0.588, 0, -0.04], [0, 0.04, 0]], atol=1e-12)
    np.testing.assert_allclose(Gk, Gk.T, atol=1e-14)


def test_transform_spatial_inertia_kinetic_energy_invariant():
    rng = np.random.default_rng(1)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    Gb = lab.spatial_inertia(Q @ np.diag([0.1, 0.2, 0.25]) @ Q.T, 3.0)
    Tba = mr.RpToTrans(mr.MatrixExp3(mr.VecToso3(rng.normal(size=3))), rng.normal(size=3))
    Ga = lab.transform_spatial_inertia(Gb, Tba)
    Va = rng.normal(size=6)
    Vb = mr.Adjoint(Tba) @ Va
    assert 0.5 * Va @ Ga @ Va == pytest.approx(0.5 * Vb @ Gb @ Vb, abs=1e-12)   # MR p. 288


def test_rigid_body_wrench_matches_newton_euler_mr_8_22_8_23():
    rng = np.random.default_rng(2)
    I = np.diag([0.8667, 0.7042, 0.3708])
    m = 20.0
    for _ in range(5):
        V, Vd = rng.normal(size=6), rng.normal(size=6)
        F = lab.rigid_body_wrench(lab.spatial_inertia(I, m), V, Vd)
        w, v, wd, vd = V[:3], V[3:], Vd[:3], Vd[3:]
        np.testing.assert_allclose(F[:3], I @ wd + np.cross(w, I @ w), atol=1e-12)
        np.testing.assert_allclose(F[3:], m * (vd + np.cross(w, v)), atol=1e-12)


def test_rigid_body_wrench_quiz_Q2_and_frame_invariance_mr_8_43():
    G = np.diag([2, 3, 4, 50, 50, 50.0])
    np.testing.assert_allclose(lab.rigid_body_wrench(G, [0, 0, 0.5, 2, 0, 0], np.zeros(6)), [0, 0, 0, 0, 50, 0], atol=1e-12)
    rng = np.random.default_rng(3)
    Gb = lab.spatial_inertia(np.diag([0.1, 0.2, 0.3]), 1.5)
    Tba = mr.RpToTrans(mr.MatrixExp3(mr.VecToso3(rng.normal(size=3))), rng.normal(size=3))
    Vb, Vdb = rng.normal(size=6), rng.normal(size=6)
    Ad = mr.Adjoint(Tba)
    Fa = lab.rigid_body_wrench(lab.transform_spatial_inertia(Gb, Tba), np.linalg.solve(Ad, Vb), np.linalg.solve(Ad, Vdb))
    np.testing.assert_allclose(Fa, Ad.T @ lab.rigid_body_wrench(Gb, Vb, Vdb), atol=1e-10)


# ---------------------------------------------------------------- checkpoint 2: forward iterations
@pytest.mark.parametrize("name", ROBOTS)
def test_link_screw_axes(name):
    r = robot(name)
    A = lab.link_screw_axes(r["Mlist"], r["Slist"])
    Mi = np.eye(4)
    for i in range(r["n"]):
        Mi = Mi @ r["Mlist"][i]
        np.testing.assert_allclose(mr.Adjoint(Mi) @ A[:, i], r["Slist"][:, i], atol=1e-12)


@pytest.mark.parametrize("name", ROBOTS)
def test_forward_transforms_and_twists_vs_fk_and_body_jacobian(name):
    """T_{i,i-1} = T_{0,i}^{-1} T_{0,i-1}  and  V_i = J_{b,i}(theta) dtheta  (MR p. 293, Eq. 8.56 notation)."""
    r = robot(name)
    n = r["n"]
    rng = np.random.default_rng(4)
    th, dth, ddth = rand_state(rng, n)
    out = lab.forward_iterations(th, dth, ddth, r["g"], r["Mlist"], r["Slist"])
    assert out["V"].shape == (6, n + 1) and out["Vdot"].shape == (6, n + 1)
    np.testing.assert_allclose(out["Vdot"][:, 0], np.r_[0, 0, 0, -r["g"]], atol=0)
    Mi, T_prev = np.eye(4), np.eye(4)
    for i in range(n):
        Mi = Mi @ r["Mlist"][i]
        T0i = mr.FKinSpace(Mi, r["Slist"][:, :i + 1], th[:i + 1])
        np.testing.assert_allclose(out["T"][i], mr.TransInv(T0i) @ T_prev, atol=1e-10)
        Jb = mr.JacobianBody(mr.Adjoint(mr.TransInv(Mi)) @ r["Slist"][:, :i + 1], th[:i + 1])
        np.testing.assert_allclose(out["V"][:, i + 1], Jb @ dth[:i + 1], atol=1e-10)
        T_prev = T0i
    np.testing.assert_allclose(out["T"][n], mr.TransInv(r["Mlist"][n]), atol=1e-12)


@pytest.mark.parametrize("name", ROBOTS)
def test_forward_accelerations_are_time_derivative_of_twists(name):
    """With g = 0, Vdot_i must equal dV_i/dt along theta(t) = theta0 + dtheta t + ddtheta t^2/2 (central difference)."""
    r = robot(name)
    n = r["n"]
    th, dth, ddth = rand_state(np.random.default_rng(5), n)
    z3 = np.zeros(3)
    out = lab.forward_iterations(th, dth, ddth, z3, r["Mlist"], r["Slist"])
    h = 1e-5

    def V_at(t):
        return lab.forward_iterations(th + dth * t + 0.5 * ddth * t**2, dth + ddth * t, ddth, z3,
                                      r["Mlist"], r["Slist"])["V"]

    np.testing.assert_allclose(out["Vdot"], (V_at(h) - V_at(-h)) / (2 * h), atol=1e-6)


def test_forward_workshop_W3_hand_values():
    Ml = [mr.RpToTrans(np.eye(3), [0.30, 0, 0]), mr.RpToTrans(np.eye(3), [0.35, 0, 0]), np.eye(4)]
    S = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, -0.30, 0]], float).T
    out = lab.forward_iterations([-np.pi / 2, np.pi / 2], [1, 2], [0, -1], [0, -9.81, 0], Ml, S)
    np.testing.assert_allclose(out["V"][:, 2], [0, 0, 3, 0.3, 1.05, 0], atol=1e-12)
    np.testing.assert_allclose(out["Vdot"][:, 1], [0, 0, 0, -9.81, 0, 0], atol=1e-12)
    np.testing.assert_allclose(out["Vdot"][:, 2], [0, 0, -1, 0, 8.86, 0], atol=1e-12)


# ---------------------------------------------------------------- checkpoint 3: backward iterations and full RNEA
def test_backward_workshop_W3_hand_values():
    Ml = [mr.RpToTrans(np.eye(3), [0.30, 0, 0]), mr.RpToTrans(np.eye(3), [0.35, 0, 0]), np.eye(4)]
    Gl = [np.diag([0, 0, 0, 2.0, 2.0, 2.0]), np.diag([0, 0, 0, 1.5, 1.5, 1.5])]
    S = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, -0.30, 0]], float).T
    fwd = lab.forward_iterations([-np.pi / 2, np.pi / 2], [1, 2], [0, -1], [0, -9.81, 0], Ml, S)
    tau, F = lab.backward_iterations(fwd, np.zeros(6), Gl)
    np.testing.assert_allclose(F[:, 1], [0, 0, 0, -4.725, 14.64, 0], atol=1e-12)
    np.testing.assert_allclose(F[:, 0], [0, 0, 5.124, -34.86, -4.725, 0], atol=1e-12)
    np.testing.assert_allclose(tau, [3.7065, 5.124], atol=1e-12)


def test_rnea_mr_2r_closed_form_mr_eq_8_10():
    r = robot("mr_2r_point")
    rng = np.random.default_rng(6)
    for _ in range(5):
        th, dth, ddth = rand_state(rng, 2)
        c2, s2 = np.cos(th[1]), np.sin(th[1])
        M = np.array([[3 + 2 * c2, 1 + c2], [1 + c2, 1]])
        c = np.array([-s2 * (2 * dth[0] * dth[1] + dth[1] ** 2), s2 * dth[0] ** 2])
        gv = 9.81 * np.array([2 * np.cos(th[0]) + np.cos(th.sum()), np.cos(th.sum())])
        tau = lab.rnea(th, dth, ddth, r["g"], np.zeros(6), *args(r))
        np.testing.assert_allclose(tau, M @ ddth + c + gv, atol=1e-10)


@pytest.mark.parametrize("name", ROBOTS)
def test_rnea_vs_modern_robotics_states_file(name):
    r = robot(name)
    for th, dth, ddth, Ft in states(name):
        np.testing.assert_allclose(lab.rnea(th, dth, ddth, r["g"], Ft, *args(r)),
                                   mr.InverseDynamics(th, dth, ddth, r["g"], Ft, *args(r)), atol=TOL)


def test_rnea_vs_modern_robotics_docstring_example():
    """The 3-link example printed in the modern_robotics InverseDynamics docstring."""
    M01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]])
    M12 = np.array([[0, 0, 1, 0.28], [0, 1, 0, 0.13585], [-1, 0, 0, 0], [0, 0, 0, 1]])
    M23 = np.array([[1, 0, 0, 0], [0, 1, 0, -0.1197], [0, 0, 1, 0.395], [0, 0, 0, 1]])
    M34 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.14225], [0, 0, 0, 1]])
    Gl = [np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7]), np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393]),
          np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])]
    S = np.array([[1, 0, 1, 0, 1, 0], [0, 1, 0, -0.089, 0, 0], [0, 1, 0, -0.089, 0, 0.425]]).T
    tau = lab.rnea([0.1, 0.1, 0.1], [0.1, 0.2, 0.3], [2, 1.5, 1], [0, 0, -9.8], [1, 1, 1, 1, 1, 1], [M01, M12, M23, M34], Gl, S)
    np.testing.assert_allclose(tau, [74.69616155, -33.06766016, -3.23057314], atol=1e-6)


# ---------------------------------------------------------------- checkpoint 4: M, c, g and J^T F by RNEA calls
@pytest.mark.parametrize("name", ROBOTS)
def test_dynamics_terms_vs_modern_robotics(name):
    r = robot(name)
    for th, dth, ddth, Ft in states(name):
        M, c, gv = lab.dynamics_terms(th, dth, r["g"], *args(r))
        np.testing.assert_allclose(M, mr.MassMatrix(th, *args(r)), atol=TOL)
        np.testing.assert_allclose(c, mr.VelQuadraticForces(th, dth, *args(r)), atol=TOL)
        np.testing.assert_allclose(gv, mr.GravityForces(th, r["g"], *args(r)), atol=TOL)
        np.testing.assert_allclose(M, M.T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(M) > 0)
        np.testing.assert_allclose(M @ ddth + c + gv + lab.tip_wrench_torques(th, Ft, *args(r)),
                                   lab.rnea(th, dth, ddth, r["g"], Ft, *args(r)), atol=TOL)


@pytest.mark.parametrize("name", ROBOTS)
def test_tip_wrench_torques_equal_JbT_F_mr_eq_5_26(name):
    r = robot(name)
    rng = np.random.default_rng(7)
    th, Ft = rng.uniform(-np.pi, np.pi, r["n"]), rng.normal(size=6)
    Mee = np.eye(4)
    for M in r["Mlist"]:
        Mee = Mee @ M
    Jb = mr.JacobianBody(mr.Adjoint(mr.TransInv(Mee)) @ r["Slist"], th)
    np.testing.assert_allclose(lab.tip_wrench_torques(th, Ft, *args(r)), Jb.T @ Ft, atol=TOL)


# ---------------------------------------------------------------- checkpoint 5: Pinocchio
@pytest.mark.parametrize("name", ROBOTS)
def test_pinocchio_matches_modern_robotics(name):
    pytest.importorskip("pinocchio")
    r = robot(name)
    model, info = lab.pinocchio_model(r)          # provided
    for th, dth, ddth, Ft in states(name):
        want = mr.InverseDynamics(th, dth, ddth, r["g"], Ft, *args(r))
        np.testing.assert_allclose(lab.pinocchio_inverse_dynamics(model, info, th, dth, ddth, Ft), want, atol=TOL)
    th, dth, ddth, _ = states(name)[1]
    np.testing.assert_allclose(lab.pinocchio_inverse_dynamics(model, info, th, dth, ddth),
                               mr.InverseDynamics(th, dth, ddth, r["g"], np.zeros(6), *args(r)), atol=TOL)


# ---------------------------------------------------------------- checkpoint 6: application -- torque profiles
@pytest.mark.parametrize("name, fname", [("eod_3r", "eod_lift.csv"), ("arm_exo_2r", "exo_reach.csv")])
def test_torque_profile_vs_modern_robotics_trajectory(name, fname):
    r = robot(name)
    t, th, dth, ddth = lab.load_trajectory(fname)
    N = len(t)
    want = mr.InverseDynamicsTrajectory(th, dth, ddth, r["g"], np.zeros((N, 6)), *args(r))
    tau = lab.torque_profile(r, th, dth, ddth)
    assert tau.shape == (N, r["n"])
    np.testing.assert_allclose(tau, want, atol=TOL)
    np.testing.assert_allclose(lab.peak_torques(tau), np.max(np.abs(want), axis=0), atol=TOL)


def test_peak_torques_simple():
    np.testing.assert_allclose(lab.peak_torques(np.array([[1.0, -5.0], [-3.0, 2.0]])), [3.0, 5.0])
