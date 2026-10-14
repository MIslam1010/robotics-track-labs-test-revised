"""Auto-checks for SBEG 46X Lab 8 (PD vs LQR vs MPC on an exoskeleton joint with torque and range limits).

    LAB_IMPL=starter  python -m pytest -q test_lab.py    # untouched starter: every test that calls your code
                                                         # fails with NotImplementedError; the 4 data tests pass
    LAB_IMPL=lab8_<ID> python -m pytest -q test_lab.py   # your renamed file

LAB_IMPL is the name of the module to test; it is imported from this folder.
Ground truth: closed-form results (scalar Riccati roots, the golden-ratio DARE of Workshop Problem 2, the LQR gains
of a damped double integrator, the ZOH discretisation of M theta'' + b theta' = u), SciPy's Riccati solvers
(solve_continuous_are, solve_discrete_are), direct simulation of the discrete model, the joint inertia and gravity
torque of the shared course models (modern_robotics MassMatrix / GravityForces via course_models), and reference runs
in data/ref_elbow_*.csv and data/expected.json.
"""
import csv
import importlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import solve_continuous_are, solve_discrete_are

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
lab = importlib.import_module(os.environ.get("LAB_IMPL", "starter"))
import course_models as cm  # noqa: E402

P = json.loads((HERE / "data" / "exo_joint.json").read_text())
E = P["elbow"]
EXP = json.loads((HERE / "data" / "expected.json").read_text())
RNG = np.random.default_rng(468)


def load_ref(kind):
    with open(HERE / "data" / f"ref_elbow_{kind}.csv") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def elbow_mats():
    M, b = E["M"], E["b"]
    A = np.array([[0.0, 1.0], [0.0, -b / M]])
    B = np.array([[0.0], [1.0 / M]])
    a = b / M
    h = E["h"]
    ea = math.exp(-a * h)
    Ad = np.array([[1.0, (1 - ea) / a], [0.0, ea]])
    Bd = np.array([[(h - (1 - ea) / a) / b], [(1 - ea) / b]])
    return A, B, Ad, Bd


# ------------------------------------------------------------------------------ data and provided code (pass)
def test_params_match_course_models():
    for name in ("elbow", "knee"):
        j = P[name]
        assert abs(j["M"] - cm.joint_inertia(name)) < 1e-9
        gs, gc = cm.gravity_coefficients(name)
        assert abs(j["g_s"] - gs) < 1e-8 and abs(j["g_c"] - gc) < 1e-8
        assert j["tau_max"] <= cm.joint_model(name)["tau_rated"]
        lo, hi = cm.joint_model(name)["limits"]
        assert lo <= j["rom"][0] < j["rom"][1] <= hi
        assert j["mpc"]["g_reserve"] >= max(abs(lab.gravity(t, j)) for t in np.linspace(*j["rom"], 201))
    assert abs(E["M"] - 0.0778) < 1e-12 and abs(E["g_s"] - 2.0 * 9.81 * 0.17) < 1e-9


def test_gravity_matches_modern_robotics():
    for name in ("elbow", "knee"):
        for th in RNG.uniform(*P[name]["rom"], 5):
            assert abs(lab.gravity(th, P[name]) - cm.joint_gravity(name, th)) < 1e-8


def test_solve_qp_known_problem():
    # min (z1-1)^2 + (z2-2)^2  s.t.  z1 + z2 <= 1   ->  z = (0, 1)
    z, st = lab.solve_qp(2 * np.eye(2), np.array([-2.0, -4.0]), np.array([[1.0, 1.0]]), np.array([1.0]))
    assert st == "optimal" and np.allclose(z, [0.0, 1.0], atol=1e-9)
    z, st = lab.solve_qp(2 * np.eye(2), np.array([-2.0, -4.0]), Aeq=np.array([[1.0, -1.0]]), beq=np.array([0.0]))
    assert st == "optimal" and np.allclose(z, [1.5, 1.5], atol=1e-9)
    z, st = lab.solve_qp(np.eye(1), np.zeros(1), np.array([[1.0], [-1.0]]), np.array([-1.0, -1.0]))
    assert st == "infeasible"


def test_setpoint_profile():
    assert lab.setpoint(0.0, E["task"]) == 1.5 and lab.setpoint(1.49, E["task"]) == 1.5
    assert lab.setpoint(1.5, E["task"]) == 0.4


# ------------------------------------------------------------------------------ models and PD
def test_joint_state_space():
    A, B = lab.joint_state_space(E["M"], E["b"])
    A0, B0, _, _ = elbow_mats()
    assert np.allclose(A, A0) and np.allclose(np.asarray(B).reshape(2, 1), B0)


def test_discretize_zoh_closed_form():
    A0, B0, Ad0, Bd0 = elbow_mats()
    Ad, Bd = lab.discretize_zoh(A0, B0, E["h"])
    assert np.allclose(Ad, Ad0, atol=1e-12) and np.allclose(np.asarray(Bd).reshape(2, 1), Bd0, atol=1e-12)
    # double integrator (b = 0): Bd = (h^2/2, h)/M
    Ad, Bd = lab.discretize_zoh(np.array([[0, 1.0], [0, 0]]), np.array([[0.0], [2.0]]), 0.1)
    assert np.allclose(Ad, [[1, 0.1], [0, 1]]) and np.allclose(np.ravel(Bd), [0.01, 0.2])


def test_pd_gains_give_wn_zeta():
    Kp, Kd = lab.pd_gains(E["M"], E["b"], 0.5, 15.0)
    r = np.roots([E["M"], E["b"] + Kd, Kp])
    wn = math.sqrt(abs(r[0] * r[1]))
    zeta = -(r[0] + r[1]).real / (2 * wn)
    assert abs(wn - 15.0) < 1e-9 and abs(zeta - 0.5) < 1e-9
    assert abs(Kp - EXP["gains"]["Kp_pd"]) < 1e-9 and abs(Kd - EXP["gains"]["Kd_pd"]) < 1e-9


# ------------------------------------------------------------------------------ Riccati equations
@pytest.mark.parametrize("a,b,q,r", [(-4.0, 2.0, 100.0, 1.0), (2.0, 1.0, 1.0, 1.0), (0.0, 1.0, 4.0, 0.25)])
def test_scalar_care(a, b, q, r):
    p, k = lab.scalar_care(a, b, q, r)
    assert p > 0 and abs(2 * a * p - p * p * b * b / r + q) < 1e-9 * max(1, q)
    assert abs(p - solve_continuous_are([[a]], [[b]], [[q]], [[r]])[0, 0]) < 1e-9 * max(1, p)
    assert abs((a - b * k) + math.sqrt(a * a + b * b * q / r)) < 1e-9


def test_scalar_dare_golden_ratio():
    s, k = lab.scalar_dare(1.0, 0.1, 1.0, 0.01)
    assert abs(s - (1 + math.sqrt(5)) / 2) < 1e-12 and abs(k - 6.180339887) < 1e-8
    for a, b, q, r in [(1.2, 0.5, 2.0, 1.0), (0.9, 0.1, 1.0, 0.1)]:
        s, k = lab.scalar_dare(a, b, q, r)
        assert abs(s - solve_discrete_are([[a]], [[b]], [[q]], [[r]])[0, 0]) < 1e-9 * max(1, s)


def test_lqr_matches_scipy_and_closed_form():
    A, B, _, _ = elbow_mats()
    Q, R = np.diag([100.0, 1 / 2.25]), np.array([[1 / 6.25]])
    K, S = lab.lqr(A, B, Q, R)
    S0 = solve_continuous_are(A, B, Q, R)
    assert np.allclose(S, S0, rtol=1e-9)
    # damped double integrator: Kp = sqrt(q1/r), Kd = -b + sqrt(b^2 + 2 M Kp + q2/r)
    Kp = math.sqrt(100.0 / (1 / 6.25))
    Kd = -E["b"] + math.sqrt(E["b"] ** 2 + 2 * E["M"] * Kp + (1 / 2.25) / (1 / 6.25))
    assert np.allclose(np.ravel(K), [Kp, Kd], rtol=1e-9)
    assert np.allclose(np.ravel(K), EXP["gains"]["K_clqr"], rtol=1e-9)


def test_riccati_recursion_fibonacci_and_convergence():
    K, S = lab.riccati_recursion([[1.0]], [[0.1]], [[1.0]], [[0.01]], [[0.0]], 4)
    vals = [float(np.ravel(S[n])[0]) for n in range(5)]
    assert np.allclose(vals[::-1], [0.0, 1.0, 1.5, 1.6, 21 / 13], atol=1e-12)
    assert abs(float(np.ravel(K[3])[0]) - 0.1 * 0.0 / 0.01) < 1e-12          # K[N-1] uses S[N] = 0
    assert abs(float(np.ravel(K[2])[0]) - 0.1 * 1.0 / (0.01 + 0.01)) < 1e-12  # 5.0
    _, _, Ad, Bd = elbow_mats()
    Q, R = lab.bryson_weights(E["lqr"]["x_max"], E["lqr"]["u_max"])
    K, S = lab.riccati_recursion(Ad, Bd, Q, R, np.zeros((2, 2)), 600)
    assert np.allclose(S[0], solve_discrete_are(Ad, Bd, Q, R), rtol=1e-8)
    assert len(K) == 600 and len(S) == 601


def test_dlqr_matches_scipy():
    _, _, Ad, Bd = elbow_mats()
    Q, R = np.diag([100.0, 1 / 2.25]), np.array([[0.16]])
    K, S = lab.dlqr(Ad, Bd, Q, R)
    S0 = solve_discrete_are(Ad, Bd, Q, R)
    K0 = np.linalg.solve(R + Bd.T @ S0 @ Bd, Bd.T @ S0 @ Ad)
    assert np.allclose(S, S0, rtol=1e-8) and np.allclose(K, K0, rtol=1e-8)
    assert np.allclose(np.ravel(K), EXP["gains"]["K_dlqr"], rtol=1e-8)


def test_bryson_weights():
    Q, R = lab.bryson_weights([0.1, 1.5], [2.5])
    assert np.allclose(Q, np.diag([100.0, 1 / 2.25])) and np.allclose(R, [[0.16]])
    Q, R = lab.bryson_weights(0.5, 2.0)
    assert np.allclose(Q, [[4.0]]) and np.allclose(R, [[0.25]])


# ------------------------------------------------------------------------------ trajectory optimisation and MPC
def test_prediction_matrices_match_simulation():
    _, _, Ad, Bd = elbow_mats()
    N = 7
    Sx, Su = lab.prediction_matrices(Ad, Bd, N)
    assert Sx.shape == (14, 2) and Su.shape == (14, 7)
    x0, U = RNG.normal(size=2), RNG.normal(size=N)
    x, X = x0.copy(), []
    for k in range(N):
        x = Ad @ x + Bd.ravel() * U[k]
        X.append(x)
    assert np.allclose(Sx @ x0 + Su @ U, np.concatenate(X), atol=1e-12)
    assert np.allclose(Su[:2, 1:], 0.0)                                       # causality


def test_mpc_qp_cost_and_constraints():
    _, _, Ad, Bd = elbow_mats()
    Q, R = np.diag([100.0, 0.4]), np.array([[0.16]])
    Pm = np.diag([300.0, 2.0])
    N, x0 = 5, np.array([-0.4, 0.5])
    lo, hi = np.array([-0.5, -3.0]), np.array([0.1, 3.0])
    H, f, G, w = lab.mpc_qp(Ad, Bd, Q, R, Pm, N, x0, (np.array([-2.5]), np.array([2.5])), (lo, hi))
    assert H.shape == (N, N) and G.shape == (2 * N + 4 * N, N)
    def cost(U):
        x, J, X = x0.copy(), 0.0, []
        for k in range(N):
            J += x @ Q @ x + U[k] ** 2 * R[0, 0]
            x = Ad @ x + Bd.ravel() * U[k]
            X.append(x.copy())
        return J + x @ Pm @ x, X

    J0, _ = cost(np.zeros(N))
    for _ in range(5):
        U = RNG.uniform(-3, 3, N)
        J, X = cost(U)
        assert abs(0.5 * U @ H @ U + f @ U + J0 - J) < 1e-8 * max(1, J)   # same cost up to the constant J(U = 0)
        feasible = np.all(np.abs(U) <= 2.5) and all(np.all(xx >= lo) and np.all(xx <= hi) for xx in X)
        assert feasible == bool(np.all(G @ U <= w + 1e-12))


def test_transcription_equals_shooting():
    _, _, Ad, Bd = elbow_mats()
    Q, R = lab.bryson_weights([0.02, 1.5], [2.5])
    _, S = lab.dlqr(Ad, Bd, Q, R)
    N, x0 = 12, np.array([-1.2, 0.0])
    lo, hi = np.array([0.2 - 1.5, -3.0]), np.array([1.6 - 1.5, 3.0])
    ul = (np.array([-2.5]), np.array([2.5]))
    H, f, G, w, Aeq, beq = lab.transcription_qp(Ad, Bd, Q, R, S, N, x0, ul, (lo, hi))
    assert H.shape == (3 * N, 3 * N) and Aeq.shape == (2 * N, 3 * N)
    U = RNG.normal(size=N)
    x, X = x0.copy(), []
    for k in range(N):
        x = Ad @ x + Bd.ravel() * U[k]
        X.append(x)
    assert np.allclose(Aeq @ np.r_[np.concatenate(X), U], beq, atol=1e-12)
    z, st = lab.solve_qp(H, f, G, w, Aeq, beq)
    H2, f2, G2, w2 = lab.mpc_qp(Ad, Bd, Q, R, S, N, x0, ul, (lo, hi))
    U2, st2 = lab.solve_qp(H2, f2, G2, w2)
    assert st == st2 == "optimal" and np.allclose(z[2 * N:], U2, atol=1e-7)


def test_mpc_without_active_constraints_is_lqr():
    _, _, Ad, Bd = elbow_mats()
    Q, R = lab.bryson_weights(E["lqr"]["x_max"], E["lqr"]["u_max"])
    K, S = lab.dlqr(Ad, Bd, Q, R)
    big = (np.array([-1e6]), np.array([1e6]))
    for x0 in (np.array([-0.05, 0.1]), np.array([0.02, -0.3])):
        u, st = lab.mpc_control(x0, Ad, Bd, Q, R, S, 10, big, None)       # terminal cost = DARE -> u = -K x
        assert st == "optimal" and abs(u + float((K @ x0)[0])) < 1e-8
        Kf, _ = lab.riccati_recursion(Ad, Bd, Q, R, Q, 10)                 # other terminal cost -> finite-horizon LQR
        u, _ = lab.mpc_control(x0, Ad, Bd, Q, R, Q, 10, big, None)
        assert abs(u + float(np.ravel(Kf[0] @ x0)[0])) < 1e-8


def test_mpc_control_respects_and_detects_limits():
    c = lab.make_controller("mpc", E)
    lo, hi = lab.error_limits(1.5, E)
    u, st = lab.mpc_control(np.array([-1.2, 0.0]), c["Ad"], c["Bd"], c["Q"], c["R"], c["S"], 30, c["u_lim"],
                            (lo, hi))
    assert st == "optimal" and abs(u - 2.5) < 1e-9                       # saturated at the tightened bound
    _, st = lab.mpc_control(np.array([0.05, 2.9]), c["Ad"], c["Bd"], c["Q"], c["R"], c["S"], 30, c["u_lim"],
                            (lo, hi))
    assert st == "infeasible"                                            # cannot stop before the range limit


def test_metrics_on_synthetic_run():
    j = dict(E)
    t = np.arange(0, 1.0, 0.01)
    thd = np.where(t < 0.5, 1.0, 0.5)
    th = np.where(t < 0.5, 1.0 + 0.1 * np.exp(-10 * t) * np.sin(20 * t), 0.5)
    th[:3] = 0.8
    sim = dict(t=t, theta=th, theta_dot=np.where(t < 0.2, 3.5, 0.0), theta_d=thd,
               tau=np.full_like(t, 2.0), tau_cmd=np.where(t < 0.1, 7.0, 2.0))
    m = lab.metrics(sim, j)
    assert set(m) == {"rom_violation", "v_violation", "overshoot", "settle_time", "peak_tau", "sat_fraction",
                      "rms_error", "effort"}
    assert m["rom_violation"] == 0.0 and abs(m["v_violation"] - 0.5) < 1e-12
    assert abs(m["peak_tau"] - 2.0) < 1e-12 and abs(m["sat_fraction"] - 0.1) < 1e-12
    assert abs(m["effort"] - 0.01 * 100 * 4.0) < 1e-9
    i = np.argmax(th[:50] - 1.0)
    assert abs(m["overshoot"] - (th[i] - 1.0)) < 1e-12
    ref_err = np.sqrt(np.mean((th - thd) ** 2))
    assert abs(m["rms_error"] - ref_err) < 1e-12
    out = np.nonzero(np.abs(th[:50] - 1.0) > 0.02)[0]
    assert abs(m["settle_time"] - (out[-1] + 1) * 0.01) < 1e-12


# ------------------------------------------------------------------------------ closed-loop runs
@pytest.mark.parametrize("kind", ["pd", "lqr", "mpc"])
def test_runs_match_reference(kind):
    sim = lab.simulate(kind, E)
    ref = load_ref(kind)
    assert np.allclose(sim["theta"], ref["theta"], atol=1e-6), "theta differs from the reference run"
    assert np.allclose(sim["tau"], ref["tau"], atol=1e-5), "applied torque differs from the reference run"
    m = lab.metrics(sim, E)
    for k, v in EXP[f"elbow_{kind}"].items():
        assert abs(m[k] - v) < 2e-5 * max(1.0, abs(v)), k


def test_mpc_run_is_safe_and_lqr_pd_are_not():
    m = {k: lab.metrics(lab.simulate(k, E), E) for k in ("pd", "lqr", "mpc")}
    assert m["mpc"]["rom_violation"] == 0.0 and m["mpc"]["v_violation"] < 1e-6 and m["mpc"]["sat_fraction"] == 0
    assert m["pd"]["rom_violation"] > 0.05 and m["lqr"]["v_violation"] > 1.0
