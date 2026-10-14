"""Auto-checks for SBEG 461 Lab 3 (forward-dynamics simulator, integrator comparison).

    LAB_IMPL=solution pytest -q test_lab.py      # reference: all pass
    LAB_IMPL=starter  pytest -q test_lab.py      # untouched starter: every test fails (NotImplementedError)

LAB_IMPL is the name of the module to test (default "solution"); it is imported from this folder.
Ground truth: modern_robotics MassMatrix, GravityForces, ForwardDynamics, ForwardDynamicsTrajectory (explicit Euler,
MR Sec. 8.5), MR Eq. 8.10 (2R point-mass mass matrix), SciPy DOP853 high-accuracy solutions, and the MR Exercise 8.15(b)
reference trajectory in data/ur5_fall_reference.csv.
"""
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import modern_robotics as mr
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
lab = importlib.import_module(os.environ.get("LAB_IMPL", "starter"))

ROBOTS = ["mr_2r_point", "leg_2r", "ur5"]
RNG = np.random.default_rng(461)


def robot(name):
    return lab.load_robot(name)


def random_states(n, k=4):
    return [(RNG.uniform(-np.pi, np.pi, n), RNG.uniform(-2, 2, n), RNG.uniform(-10, 10, n)) for _ in range(k)]


# ------------------------------------------------------------------------------------- checkpoint 1: M(theta)
@pytest.mark.parametrize("name", ROBOTS)
def test_mass_matrix_rnea_matches_mr(name):
    Ml, Gl, Sl, _ = robot(name)
    for th, _, _ in random_states(Sl.shape[1]):
        M = lab.mass_matrix_rnea(th, Ml, Gl, Sl)
        assert np.allclose(M, mr.MassMatrix(th, Ml, Gl, Sl), atol=1e-10)


def test_mass_matrix_rnea_uses_exactly_n_calls():
    Ml, Gl, Sl, _ = robot("ur5")
    calls = []

    def counting_rnea(th, dth, ddth, g, Ftip, Mlist, Glist, Slist):
        calls.append((np.array(dth), np.array(ddth), np.array(g)))
        return mr.InverseDynamics(th, dth, ddth, g, Ftip, Mlist, Glist, Slist)

    th = RNG.uniform(-np.pi, np.pi, 6)
    M = lab.mass_matrix_rnea(th, Ml, Gl, Sl, rnea=counting_rnea)
    assert len(calls) == 6, "one RNEA call per column"
    for dth, ddth, g in calls:
        assert np.allclose(dth, 0) and np.allclose(g, 0) and np.isclose(np.sum(ddth), 1.0)
    assert np.allclose(M, mr.MassMatrix(th, Ml, Gl, Sl), atol=1e-10)


@pytest.mark.parametrize("name", ROBOTS)
def test_mass_matrix_closed_form_matches_mr(name):
    Ml, Gl, Sl, _ = robot(name)
    for th, _, _ in random_states(Sl.shape[1]):
        M = lab.mass_matrix_closed_form(th, Ml, Gl, Sl)
        assert np.allclose(M, mr.MassMatrix(th, Ml, Gl, Sl), atol=1e-10)
        assert np.allclose(M, M.T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(M) > 0) or name == "mr_2r_point"


def test_mass_matrix_mr_eq_8_9():
    """MR Eq. 8.10 with L1 = L2 = m1 = m2 = 1: M = [[3 + 2 c2, 1 + c2], [1 + c2, 1]]."""
    Ml, Gl, Sl, _ = robot("mr_2r_point")
    for t2 in [0.0, np.pi / 2, 2.0]:
        c2 = np.cos(t2)
        expected = np.array([[3 + 2 * c2, 1 + c2], [1 + c2, 1.0]])
        assert np.allclose(lab.mass_matrix_closed_form([0.3, t2], Ml, Gl, Sl), expected, atol=1e-12)
        assert np.allclose(lab.mass_matrix_rnea([0.3, t2], Ml, Gl, Sl), expected, atol=1e-12)


# ------------------------------------------------------------------------------------- checkpoint 2: FD, energy
@pytest.mark.parametrize("name", ROBOTS)
def test_forward_dynamics_matches_mr(name):
    Ml, Gl, Sl, g = robot(name)
    for th, dth, tau in random_states(Sl.shape[1]):
        Ftip = RNG.uniform(-5, 5, 6)
        dd = lab.forward_dynamics(th, dth, tau, g, Ftip, Ml, Gl, Sl)
        assert np.allclose(dd, mr.ForwardDynamics(th, dth, tau, g, Ftip, Ml, Gl, Sl), atol=1e-8)
        # round trip through inverse dynamics (MR Eq. 8.77)
        assert np.allclose(mr.InverseDynamics(th, dth, dd, g, Ftip, Ml, Gl, Sl), tau, atol=1e-8)


@pytest.mark.parametrize("name", ROBOTS)
def test_energy_kinetic_and_gravity(name):
    """K = 1/2 dth^T M dth and dP/dtheta = g(theta) (gravity torques from modern_robotics)."""
    Ml, Gl, Sl, g = robot(name)
    n = Sl.shape[1]
    for th, dth, _ in random_states(n, 3):
        K, P = lab.energy(th, dth, g, Ml, Gl, Sl)
        assert np.isclose(K, 0.5 * dth @ mr.MassMatrix(th, Ml, Gl, Sl) @ dth, atol=1e-10)
        eps = 1e-6
        grad = np.array([(lab.energy(th + eps * e, dth, g, Ml, Gl, Sl)[1] - lab.energy(th - eps * e, dth, g, Ml, Gl, Sl)[1])
                         / (2 * eps) for e in np.eye(n)])
        assert np.allclose(grad, mr.GravityForces(th, g, Ml, Gl, Sl), atol=1e-6)


def test_energy_values_leg():
    """Leg at theta = (-70, -40) deg, dtheta = (2, -3): K = 1.47994 J, P = -27.50766 J (workshop Problem 1)."""
    Ml, Gl, Sl, g = robot("leg_2r")
    K, P = lab.energy(np.deg2rad([-70, -40]), [2.0, -3.0], g, Ml, Gl, Sl)
    assert np.isclose(K, 1.4799422, atol=1e-6) and np.isclose(P, -27.5076597, atol=1e-6)


# ------------------------------------------------------------------------------------- checkpoint 3: integrators
A_OSC = np.array([[0.0, 1.0], [-25.0, -0.4]])     # damped oscillator x = (theta, dtheta)


def f_osc(t, x):
    return A_OSC @ x


def test_euler_step_linear():
    x, h = np.array([0.3, -0.2]), 0.01
    assert np.allclose(lab.euler_step(f_osc, 0.0, x, h), (np.eye(2) + h * A_OSC) @ x, atol=1e-14)


def test_semi_implicit_step_linear():
    x, h = np.array([0.3, -0.2]), 0.01
    v = x[1] + h * (A_OSC @ x)[1]
    assert np.allclose(lab.semi_implicit_euler_step(f_osc, 0.0, x, h), [x[0] + h * v, v], atol=1e-14)


def test_rk4_step_linear_is_fourth_order_taylor():
    x, h = np.array([0.3, -0.2]), 0.05
    hA = h * A_OSC
    T4 = np.eye(2) + hA + hA @ hA / 2 + hA @ hA @ hA / 6 + hA @ hA @ hA @ hA / 24
    assert np.allclose(lab.rk4_step(f_osc, 0.0, x, h), T4 @ x, atol=1e-14)


def test_rk4_step_uses_time_argument():
    f = lambda t, x: np.array([t**3])          # x(t) = t^4 / 4 exactly integrated by RK4
    assert np.isclose(lab.rk4_step(f, 1.0, np.array([0.25]), 0.5)[0], 1.5**4 / 4, atol=1e-14)


def test_simulate_euler_matches_mr_forward_dynamics_trajectory():
    """Explicit Euler simulation = mr.ForwardDynamicsTrajectory (intRes = 1), with a hip torque profile and a tip force."""
    R = robot("leg_2r")
    Ml, Gl, Sl, g = R
    h, N = 0.005, 80
    tau_fn = lambda t, th, dth: np.array([6.0 * np.sin(2 * np.pi * t), -1.0])
    Ftip = np.array([0, 0, 0, 2.0, -5.0, 0])
    th0, dth0 = np.deg2rad([-100.0, -10.0]), np.array([0.5, 0.0])
    t, TH, DTH = lab.simulate(R, th0, dth0, N * h, h, "euler", tau_fn=tau_fn, Ftip_fn=lambda t, th, dth: Ftip)
    assert TH.shape == (N + 1, 2) and DTH.shape == (N + 1, 2) and np.isclose(t[-1], N * h)
    taumat = np.array([tau_fn(k * h, None, None) for k in range(N)])
    thm, dthm = mr.ForwardDynamicsTrajectory(th0, dth0, taumat, g, np.tile(Ftip, (N, 1)), Ml, Gl, Sl, h, 1)
    assert np.allclose(TH[:N], thm, atol=1e-10) and np.allclose(DTH[:N], dthm, atol=1e-10)


def leg_reference(th0, tf):
    Ml, Gl, Sl, g = robot("leg_2r")

    def rhs(t, x):
        return np.r_[x[2:], mr.ForwardDynamics(x[:2], x[2:], np.zeros(2), g, np.zeros(6), Ml, Gl, Sl)]

    return solve_ivp(rhs, [0, tf], np.r_[th0, 0, 0], method="DOP853", rtol=1e-12, atol=1e-12).y[:2, -1]


def test_observed_orders_leg_swing():
    """Free swing of the prosthetic leg from (-60, -30) deg: Euler and semi-implicit Euler ~ order 1, RK4 ~ order 4."""
    R = robot("leg_2r")
    th0, tf = np.deg2rad([-60.0, -30.0]), 0.4
    ref = leg_reference(th0, tf)
    hs = [0.02, 0.01, 0.005]
    p = {}
    for m in ["euler", "semi_implicit", "rk4"]:
        errs = [np.linalg.norm(lab.simulate(R, th0, [0, 0], tf, h, m)[1][-1] - ref) for h in hs]
        p[m] = lab.observed_order(hs, errs)
    assert 0.8 < p["euler"] < 1.3, p
    assert 0.8 < p["semi_implicit"] < 1.3, p
    assert 3.5 < p["rk4"] < 4.5, p


def test_observed_order_function():
    hs = np.array([0.1, 0.05, 0.025])
    assert np.isclose(lab.observed_order(hs, 3.0 * hs**2), 2.0, atol=1e-12)


# ------------------------------------------------------------------------------------- checkpoint 4: UR5, energy
def test_ur5_fall_rk4_vs_reference():
    """MR Exercise 8.15(b): UR5 falls from theta = 0 under gravity; RK4 with h = 0.01 s over the first second."""
    R = robot("ur5")
    ref = np.loadtxt(HERE / "data" / "ur5_fall_reference.csv", delimiter=",", skiprows=2)
    t, TH, DTH = lab.simulate(R, np.zeros(6), np.zeros(6), 1.0, 0.01, "rk4")
    for row in ref[ref[:, 0] <= 1.0 + 1e-9]:
        k = int(round(row[0] / 0.01))
        assert np.allclose(TH[k], row[1:7], atol=1e-4), row[0]
        assert np.allclose(DTH[k], row[7:13], atol=1e-3), row[0]


def test_ur5_energy_drift_ranking():
    """Energy check (zero torque => E constant): explicit Euler gains energy, RK4 conserves it to < 1e-4 J."""
    R = robot("ur5")
    Ml, Gl, Sl, g = R
    drift = {}
    for m in ["euler", "rk4"]:
        t, TH, DTH = lab.simulate(R, np.zeros(6), np.zeros(6), 0.5, 0.01, m)
        E = np.array([sum(lab.energy(TH[k], DTH[k], g, Ml, Gl, Sl)) for k in range(len(t))])
        drift[m] = E[-1] - E[0]
    assert drift["euler"] > 0.05, drift
    assert abs(drift["rk4"]) < 1e-4, drift


def test_power_balance_with_torque():
    """dE/dt = dtheta^T tau: E(T) - E(0) equals the work done by the hip torque (trapezoid rule)."""
    R = robot("leg_2r")
    Ml, Gl, Sl, g = R
    tau_fn = lambda t, th, dth: np.array([8.0 * np.sin(2 * np.pi * t), 0.0])
    h = 0.002
    t, TH, DTH = lab.simulate(R, np.deg2rad([-90.0, 0.0]), [0, 0], 0.5, h, "rk4", tau_fn=tau_fn)
    E = np.array([sum(lab.energy(TH[k], DTH[k], g, Ml, Gl, Sl)) for k in range(len(t))])
    power = np.array([DTH[k] @ tau_fn(t[k], TH[k], DTH[k]) for k in range(len(t))])
    work = np.sum(0.5 * (power[1:] + power[:-1]) * h)
    assert abs(work) > 0.1
    assert np.isclose(E[-1] - E[0], work, atol=2e-4)
