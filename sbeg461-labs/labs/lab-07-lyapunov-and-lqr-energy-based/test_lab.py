"""Auto-checks for SBEG 461 Lab 7 (Lyapunov and LQR: energy-based stability check of PD + gravity compensation; LQR for a linearised 2R arm and the course arm).

    LAB_IMPL=solution pytest -q test_lab.py      # reference: all pass
    LAB_IMPL=starter  pytest -q test_lab.py      # untouched starter: every test except test_starter_compiles fails
                                                 # with NotImplementedError (22 failed, 1 passed)

LAB_IMPL is the name of the module to test (default "solution"); it is imported from this folder.
Ground truth: modern_robotics (MassMatrix, VelQuadraticForces, GravityForces, ForwardDynamics), the closed-form 2R
model, finite differences, scipy.linalg (solve_continuous_lyapunov, solve_continuous_are), the identities of
MR Sec. 8.1.2 and Sec. 11.4.2.2 (C dq = c, M' - 2C skew-symmetric, V' = -dq^T (Kd + B) dq) and of LQR theory
(Riccati residual, J = x0^T P x0, no stabilising gain has a lower cost).
Each test that draws random numbers uses its own generator np.random.default_rng(<seed>), seeds 1-9.
"""
import importlib
import os
import py_compile
import sys
from pathlib import Path

import modern_robotics as mr
import numpy as np
import pytest
import scipy.linalg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("LAB_IMPL", "starter")
lab = importlib.import_module(os.environ["LAB_IMPL"])

CFG = lab.load_config()
C2 = CFG["planar_2r"]
Q_EQ = np.array(C2["q_eq"])
DT = CFG["dt"]


def p2r(friction=True):
    return lab.plant_2r(CFG, friction)


def pd_gains(robot, q):
    return lab.pd_gains_from_inertia(robot, q, C2["pd"]["zeta"], C2["pd"]["wn"])


def weights_2r():
    L = C2["lqr"]
    return lab.bryson_weights([L["q_max"]] * 2 + [L["dq_max"]] * 2, L["tau_max"])


def mr_lists(r):
    return r["Mlist"], r["Glist"], r["Slist"]


def test_starter_compiles():
    py_compile.compile(str(HERE / "starter.py"), doraise=True)


# ============================================================================ Part A: passivity and Lyapunov
def test_models_and_stored_energy():
    """The closed-form 2R equals modern_robotics; E = KE + P with P from the CoM heights."""
    rng = np.random.default_rng(1)
    r = p2r()
    for _ in range(3):
        q, dq = rng.uniform(-2, 2, 2), rng.uniform(-1, 1, 2)
        assert np.allclose(lab.mass_matrix(r, q), mr.MassMatrix(q, *mr_lists(r)), atol=1e-12)
        assert np.allclose(lab.coriolis_vector(r, q, dq), mr.VelQuadraticForces(q, dq, *mr_lists(r)), atol=1e-12)
        assert np.allclose(lab.gravity_torque(r, q), mr.GravityForces(q, r["g"], *mr_lists(r)), atol=1e-12)
        y1, y2 = 0.3 * np.sin(q[0]), 0.6 * np.sin(q[0]) + 0.2 * np.sin(q[0] + q[1])   # CoM heights (m)
        E = 0.5 * dq @ mr.MassMatrix(q, *mr_lists(r)) @ dq + 9.81 * (2.0 * y1 + 1.0 * y2)
        assert lab.stored_energy(r, q, dq) == pytest.approx(E, rel=1e-12)
    arm = lab.plant_course_arm(CFG)
    q, dq = arm["q_home"], np.full(6, 0.3)
    assert lab.stored_energy(arm, q, dq) == pytest.approx(
        0.5 * dq @ mr.MassMatrix(q, *mr_lists(arm)) @ dq + lab.potential_energy(arm, q), rel=1e-12)


def test_mass_matrix_partials():
    rng = np.random.default_rng(2)
    r = p2r()
    q = rng.uniform(-2, 2, 2)
    dM = lab.mass_matrix_partials(r, q)
    assert np.shape(dM) == (2, 2, 2)
    cf = r["closed_form"]
    a = -cf["m2"] * cf["L1"] * cf["lc2"] * np.sin(q[1])       # d m12 / d q2; d m11 / d q2 = 2a; M indep. of q1
    assert np.allclose(dM[0], 0.0, atol=1e-8)
    assert np.allclose(dM[1], [[2 * a, a], [a, 0.0]], atol=1e-8)
    arm = lab.plant_course_arm(CFG)
    q = arm["q_home"] + rng.uniform(-0.5, 0.5, 6)
    dMa = lab.mass_matrix_partials(arm, q)
    k, h = 2, 1e-4
    e = np.zeros(6)
    e[k] = h
    ref = (mr.MassMatrix(q + e, *mr_lists(arm)) - mr.MassMatrix(q - e, *mr_lists(arm))) / (2 * h)
    assert np.allclose(dMa[k], ref, atol=1e-8)
    assert np.allclose(dMa[0], 0.0, atol=1e-9)        # joint 1 (vertical axis) does not change M


def test_coriolis_matrix_reproduces_velocity_forces():
    """C(q, dq) dq = c(q, dq) = mr.VelQuadraticForces, for the 2R and the 6R course arm."""
    rng = np.random.default_rng(3)
    for r in (p2r(), lab.plant_course_arm(CFG)):
        for _ in range(2):
            q, dq = rng.uniform(-2, 2, r["n"]), rng.uniform(-1.5, 1.5, r["n"])
            C = lab.coriolis_matrix(r, q, dq)
            assert np.shape(C) == (r["n"], r["n"])
            assert np.allclose(C @ dq, mr.VelQuadraticForces(q, dq, *mr_lists(r)), atol=1e-7)
    r = p2r()
    q, dq = np.array([0.3, 1.1]), np.array([0.7, -0.4])
    cf = r["closed_form"]
    hh = -cf["m2"] * cf["L1"] * cf["lc2"] * np.sin(q[1])
    C_ref = np.array([[hh * dq[1], hh * (dq[0] + dq[1])], [-hh * dq[0], 0.0]])   # textbook 2R Coriolis matrix
    assert np.allclose(lab.coriolis_matrix(r, q, dq), C_ref, atol=1e-8)


def test_mdot_minus_2c_is_skew_symmetric():
    """MR Proposition 8.1, checked numerically on the 2R and the course arm."""
    rng = np.random.default_rng(4)
    for r in (p2r(), lab.plant_course_arm(CFG)):
        for _ in range(2):
            q, dq = rng.uniform(-2, 2, r["n"]), rng.uniform(-1.5, 1.5, r["n"])
            N = lab.mass_matrix_dot(r, q, dq) - 2 * lab.coriolis_matrix(r, q, dq)
            scale = max(1.0, np.abs(N).max())
            assert np.abs(N + N.T).max() < 1e-7 * scale
            x = rng.normal(size=r["n"])
            assert abs(x @ N @ x) < 1e-7 * scale * (x @ x)


def test_mass_matrix_dot_matches_time_derivative():
    rng = np.random.default_rng(5)
    for r in (p2r(), lab.plant_course_arm(CFG)):
        q, dq = rng.uniform(-2, 2, r["n"]), rng.uniform(-1.5, 1.5, r["n"])
        h = 1e-5
        ref = (lab.mass_matrix(r, q + h * dq) - lab.mass_matrix(r, q - h * dq)) / (2 * h)
        assert np.allclose(lab.mass_matrix_dot(r, q, dq), ref, atol=1e-8)


def test_manipulator_is_passive():
    """Power balance dE/dt = dq^T tau - dq^T B dq along an open-loop motion under a constant torque:
    without friction E(T) - E(0) = tau^T (q(T) - q(0)) exactly; with friction the difference is the dissipated energy."""
    tau = np.array([3.0, 0.5])
    for friction in (False, True):
        r = p2r(friction)
        sim = lab.simulate(r, lambda t, q, dq: tau, [0.2, 0.4], [0.0, 0.0], 1.5, 0.001)
        E = np.array([lab.stored_energy(r, q, dq) for q, dq in zip(sim["q"], sim["dq"])])
        supplied = tau @ (sim["q"][-1] - sim["q"][0])
        diss = np.einsum("ki,i,ki->k", sim["dq"], r["b"], sim["dq"])
        lost = float(np.sum(0.5 * (diss[1:] + diss[:-1]) * np.diff(sim["t"])))
        assert E[-1] - E[0] == pytest.approx(supplied - lost, abs=1e-5 if friction else 1e-8)
        if friction:
            assert lost > 0.1


def test_pd_gravity_setpoint_torque():
    r = p2r()
    Kp, Kd = np.diag([50.0, 4.0]), np.diag([10.0, 1.0])
    q, dq, qd = np.array([0.5, 1.0]), np.array([0.2, -0.3]), Q_EQ
    tau = lab.pd_gravity_setpoint_torque(q, dq, qd, Kp, Kd, r)
    assert np.allclose(tau, Kp @ (qd - q) - Kd @ dq + mr.GravityForces(q, r["g"], *mr_lists(r)), atol=1e-12)
    m08 = lab.scaled_model(r, 0.8)
    tau08 = lab.pd_gravity_setpoint_torque(q, dq, qd, Kp, Kd, m08)
    assert np.allclose(tau - tau08, 0.2 * lab.gravity_torque(r, q), atol=1e-12)   # g~(q) of the MODEL, at q


def test_lyapunov_energy_is_positive_definite():
    rng = np.random.default_rng(6)
    r = p2r()
    Kp, _ = pd_gains(r, Q_EQ)
    assert lab.lyapunov_energy(r, Q_EQ, [0.0, 0.0], Q_EQ, Kp) == pytest.approx(0.0, abs=1e-15)
    for _ in range(5):
        q, dq = Q_EQ + rng.uniform(-1, 1, 2), rng.uniform(-1, 1, 2)
        V = lab.lyapunov_energy(r, q, dq, Q_EQ, Kp)
        e = Q_EQ - q
        assert V == pytest.approx(0.5 * e @ Kp @ e + 0.5 * dq @ lab.mass_matrix(r, q) @ dq, rel=1e-12)
        assert V > 0
    q0 = np.array(C2["lyapunov"]["q0"])
    assert lab.lyapunov_energy(r, q0, [0.0, 0.0], Q_EQ, Kp) == pytest.approx(
        0.5 * (Q_EQ - q0) @ Kp @ (Q_EQ - q0), rel=1e-12)


def test_lyapunov_rate_is_minus_dq_kd_dq():
    """MR Eq. 11.43 with plant friction: for tau = PD + exact g, V' = -dq^T (Kd + B) dq; for any tau,
    V' = dq^T (tau - g(q) - B dq - Kp theta_e) (the skew-symmetric term vanishes)."""
    rng = np.random.default_rng(7)
    r = p2r()
    Kp, Kd = pd_gains(r, Q_EQ)
    B = np.diag(r["b"])
    for _ in range(4):
        q, dq = Q_EQ + rng.uniform(-1, 1, 2), rng.uniform(-2, 2, 2)
        tau = lab.pd_gravity_setpoint_torque(q, dq, Q_EQ, Kp, Kd, r)
        assert lab.lyapunov_rate(r, q, dq, Q_EQ, Kp, tau) == pytest.approx(-dq @ (Kd + B) @ dq, rel=1e-6, abs=1e-9)
        tau_any = rng.uniform(-5, 5, 2)
        ref = dq @ (tau_any - lab.gravity_torque(r, q) - B @ dq - Kp @ (Q_EQ - q))
        assert lab.lyapunov_rate(r, q, dq, Q_EQ, Kp, tau_any) == pytest.approx(ref, rel=1e-6, abs=1e-9)
    arm = lab.plant_course_arm(CFG)
    Kpa, Kda = lab.pd_gains_from_inertia(arm, arm["q_home"], 1.0, 8.0)
    q, dq = arm["q_home"] + rng.uniform(-0.3, 0.3, 6), rng.uniform(-1, 1, 6)
    tau = lab.pd_gravity_setpoint_torque(q, dq, arm["q_home"], Kpa, Kda, arm)
    ref = -dq @ (Kda + np.diag(arm["b"])) @ dq
    assert lab.lyapunov_rate(arm, q, dq, arm["q_home"], Kpa, tau) == pytest.approx(ref, rel=1e-6)


def test_energy_decreases_along_closed_loop():
    """E1: PD + exact gravity compensation from q0 = q_eq + (0.3, -0.4): V never increases and the arm converges."""
    r = p2r()
    Kp, Kd = pd_gains(r, Q_EQ)
    L = C2["lyapunov"]
    sim = lab.simulate(r, lambda t, q, dq: lab.pd_gravity_setpoint_torque(q, dq, Q_EQ, Kp, Kd, r),
                       L["q0"], L["dq0"], L["T"], DT)
    V = np.array([lab.lyapunov_energy(r, q, dq, Q_EQ, Kp) for q, dq in zip(sim["q"], sim["dq"])])
    assert V[0] == pytest.approx(2.586473, rel=1e-6)
    assert np.max(np.diff(V)) <= 1e-12
    assert V[-1] < 1e-12 and np.linalg.norm(sim["q"][-1] - Q_EQ) < 1e-6
    k = 100                                              # numerical dV/dt matches lyapunov_rate along the run
    dV = (V[k + 1] - V[k - 1]) / (2 * DT)
    rate = lab.lyapunov_rate(r, sim["q"][k], sim["dq"][k], Q_EQ, Kp, sim["tau"][k])
    assert dV == pytest.approx(rate, rel=2e-3)


def test_kd_zero_conserves_energy_lasalle_fails():
    """E2 (MR Exercise 11.12): Kd = 0 and no friction -> V' = 0 in continuous time; nothing dissipates energy and the
    arm never converges.  The sampled (zero-order-hold) controller even INJECTS energy, in proportion to dt."""
    r = p2r(friction=False)
    Kp, _ = pd_gains(r, Q_EQ)
    Kd = np.zeros((2, 2))
    L = C2["lyapunov"]
    drift = {}
    for dt in (DT, DT / 4):
        sim = lab.simulate(r, lambda t, q, dq: lab.pd_gravity_setpoint_torque(q, dq, Q_EQ, Kp, Kd, r),
                           L["q0"], L["dq0"], 2.0, dt)
        V = np.array([lab.lyapunov_energy(r, q, dq, Q_EQ, Kp) for q, dq in zip(sim["q"], sim["dq"])])
        assert np.min(V - V[0]) > -1e-9                   # never dissipated
        assert np.max(np.linalg.norm(sim["q"][-int(0.5 / dt):] - Q_EQ, axis=1)) > 0.5   # no convergence
        drift[dt] = V[-1] - V[0]
        k = len(sim["t"]) // 3
        assert lab.lyapunov_rate(r, sim["q"][k], sim["dq"][k], Q_EQ, Kp,
                                 lab.pd_gravity_setpoint_torque(sim["q"][k], sim["dq"][k], Q_EQ, Kp, Kd, r)) \
            == pytest.approx(0.0, abs=1e-8)
    assert drift[DT] == pytest.approx(0.997701, rel=1e-4)
    assert drift[DT / 4] < 0.3 * drift[DT]               # a sampling effect: it shrinks with dt


def test_wrong_gravity_model_offset():
    """E3: a gravity model 20 % too light; the arm comes to rest where Kp theta_e = g(q) - g~(q)."""
    r = p2r()
    model = lab.scaled_model(r, C2["wrong_model_mass_scale"])
    Kp, Kd = pd_gains(r, Q_EQ)
    q_ss = lab.equilibrium_offset(r, model, Q_EQ, Kp)
    F = Kp @ (Q_EQ - q_ss) + lab.gravity_torque(model, q_ss) - lab.gravity_torque(r, q_ss)
    assert np.linalg.norm(F) < 1e-10
    assert np.allclose(q_ss - Q_EQ, [-0.039760, -0.014271], atol=1e-6)
    L = C2["lyapunov"]
    sim = lab.simulate(r, lambda t, q, dq: lab.pd_gravity_setpoint_torque(q, dq, Q_EQ, Kp, Kd, model),
                       L["q0"], L["dq0"], L["T"], DT)
    assert np.allclose(sim["q"][-1], q_ss, atol=1e-5)
    with pytest.raises(ValueError):
        lab.equilibrium_offset(r, model, Q_EQ, Kp, iters=1)


# ============================================================================ Part B: LQR
def fd_jacobians(r, q_eq, u_eq, h=1e-6):
    """Independent check: Jacobians of f(x, u) = (dq, forward_dynamics) by central differences."""
    n = r["n"]
    f = lambda x, u: np.r_[x[n:], lab.forward_dynamics(r, x[:n], x[n:], u)]   # noqa: E731
    x0 = np.r_[q_eq, np.zeros(n)]
    A = np.column_stack([(f(x0 + h * e, u_eq) - f(x0 - h * e, u_eq)) / (2 * h) for e in np.eye(2 * n)])
    B = np.column_stack([(f(x0, u_eq + h * e) - f(x0, u_eq - h * e)) / (2 * h) for e in np.eye(n)])
    return A, B


def test_linearize_matches_finite_differences():
    for r, q_eq in ((p2r(), Q_EQ), (lab.plant_course_arm(CFG), lab.plant_course_arm(CFG)["q_home"])):
        A, B, u_eq = lab.linearize(r, q_eq)
        n = r["n"]
        assert np.shape(A) == (2 * n, 2 * n) and np.shape(B) == (2 * n, n)
        assert np.allclose(u_eq, mr.GravityForces(q_eq, r["g"], *mr_lists(r)), atol=1e-10)
        assert np.allclose(lab.forward_dynamics(r, q_eq, np.zeros(n), u_eq), 0.0, atol=1e-9)   # an equilibrium
        A_ref, B_ref = fd_jacobians(r, q_eq, u_eq)
        assert np.allclose(A, A_ref, atol=1e-5 * max(1.0, np.abs(A_ref).max()))
        assert np.allclose(B, B_ref, atol=1e-5 * max(1.0, np.abs(B_ref).max()))


def test_linearize_hanging_and_hold_pose():
    """Hanging down: a frictionless double pendulum (eigenvalues on the imaginary axis); the hold pose q_eq: unstable."""
    A, _, _ = lab.linearize(p2r(friction=False), C2["q_hang"])
    ev = np.linalg.eigvals(A)
    assert np.allclose(ev.real, 0.0, atol=1e-6)
    assert np.allclose(np.sort(np.abs(ev.imag)), [3.86986, 3.86986, 9.33178, 9.33178], atol=1e-3)
    A, _, _ = lab.linearize(p2r(), Q_EQ)
    assert np.max(np.linalg.eigvals(A).real) == pytest.approx(6.5706, abs=1e-3)
    assert not lab.is_hurwitz(A)


def test_bryson_weights():
    Q, R = lab.bryson_weights([0.05, 0.05, 0.5, 0.5], [5.0, 2.0])
    assert np.allclose(Q, np.diag([400.0, 400.0, 4.0, 4.0]))
    assert np.allclose(R, np.diag([0.04, 0.25]))


def test_lyapunov_solve_matches_scipy():
    rng = np.random.default_rng(8)
    for n in (2, 4, 7):
        A = rng.normal(size=(n, n)) - (n + 1.0) * np.eye(n)    # Hurwitz
        X = rng.normal(size=(n, n))
        Q = X @ X.T + np.eye(n)
        P = lab.lyapunov_solve(A, Q)
        assert np.allclose(P, scipy.linalg.solve_continuous_lyapunov(A.T, -Q), atol=1e-10)
        assert np.allclose(A.T @ P + P @ A + Q, 0.0, atol=1e-9)
        assert np.allclose(P, P.T) and np.min(np.linalg.eigvalsh(P)) > 0


def test_lqr_solves_riccati():
    r = p2r()
    A, B, _ = lab.linearize(r, Q_EQ)
    Q, R = weights_2r()
    K, P = lab.lqr(A, B, Q, R)
    res = A.T @ P + P @ A - P @ B @ np.linalg.solve(R, B.T @ P) + Q
    assert np.abs(res).max() < 1e-8 * np.abs(Q).max()
    assert np.allclose(P, P.T) and np.min(np.linalg.eigvalsh(P)) > 0
    assert np.allclose(K, np.linalg.solve(R, B.T @ P), atol=1e-10)
    assert lab.is_hurwitz(A - B @ K)
    assert np.allclose(K, [[106.933, -18.295, 16.263, -1.006], [10.246, 41.185, 1.878, 4.395]], atol=2e-3)
    Acl, Qcl = A - B @ K, Q + K.T @ R @ K          # P is also the Lyapunov function of the closed loop
    assert np.allclose(Acl.T @ P + P @ Acl + Qcl, 0.0, atol=1e-7 * np.abs(Qcl).max())


def test_pd_state_feedback_is_linearised_pd():
    r = p2r()
    Kp, Kd = pd_gains(r, Q_EQ)
    K0 = lab.pd_state_feedback(r, Q_EQ, Kp, Kd)
    assert np.shape(K0) == (2, 4)
    tau = lambda x: lab.pd_gravity_setpoint_torque(Q_EQ + x[:2], x[2:], Q_EQ, Kp, Kd, r)   # noqa: E731
    h = 1e-6
    J = np.column_stack([(tau(h * e) - tau(-h * e)) / (2 * h) for e in np.eye(4)])
    assert np.allclose(K0, -J, atol=1e-6)
    A, B, _ = lab.linearize(r, Q_EQ)
    assert lab.is_hurwitz(A - B @ K0)                # PD + g~ stabilises the unstable hold pose


def test_kleinman_converges_to_lqr():
    r = p2r()
    A, B, _ = lab.linearize(r, Q_EQ)
    Q, R = weights_2r()
    Kp, Kd = pd_gains(r, Q_EQ)
    K0 = lab.pd_state_feedback(r, Q_EQ, Kp, Kd)
    K, P, it = lab.kleinman_lqr(A, B, Q, R, K0)
    P_ref = scipy.linalg.solve_continuous_are(A, B, Q, R)
    assert np.allclose(P, P_ref, rtol=1e-8, atol=1e-10)
    assert np.allclose(K, np.linalg.solve(R, B.T @ P_ref), rtol=1e-8, atol=1e-8)
    assert 3 <= it <= 15
    with pytest.raises(ValueError):
        lab.kleinman_lqr(A, B, Q, R, np.zeros((2, 4)))    # K0 = 0 does not stabilise the hold pose


def test_lqr_cost_is_optimal():
    """J = x0^T P x0 for the LQR gain; the PD gain (and any other stabilising gain) costs more."""
    rng = np.random.default_rng(9)
    r = p2r()
    A, B, _ = lab.linearize(r, Q_EQ)
    Q, R = weights_2r()
    K, P = lab.lqr(A, B, Q, R)
    x0 = np.array(C2["lqr"]["x0"])
    J = lab.lqr_cost(A, B, Q, R, K, x0)
    assert J == pytest.approx(x0 @ P @ x0, rel=1e-8)
    assert J == pytest.approx(0.254473, rel=1e-5)
    Kp, Kd = pd_gains(r, Q_EQ)
    J_pd = lab.lqr_cost(A, B, Q, R, lab.pd_state_feedback(r, Q_EQ, Kp, Kd), x0)
    assert J_pd == pytest.approx(0.310228, rel=1e-5)
    for _ in range(5):
        Kpert = K * (1 + 0.2 * rng.uniform(-1, 1, K.shape))
        if lab.is_hurwitz(A - B @ Kpert):
            assert lab.lqr_cost(A, B, Q, R, Kpert, x0) >= J - 1e-12
    with pytest.raises(ValueError):
        lab.lqr_cost(A, B, Q, R, np.zeros((2, 4)), x0)


def test_state_feedback_torque_and_nonlinear_closed_loop():
    """LQR and linearised PD on the NONLINEAR 2R from x0: both converge; run costs close to the linear predictions."""
    r = p2r()
    A, B, u_eq = lab.linearize(r, Q_EQ)
    Q, R = weights_2r()
    K, P = lab.lqr(A, B, Q, R)
    tau = lab.state_feedback_torque([0.65, 0.85], [0.1, -0.2], Q_EQ, u_eq, K)
    assert np.allclose(tau, u_eq - K @ np.array([0.05, -0.05, 0.1, -0.2]), atol=1e-12)
    x0 = np.array(C2["lqr"]["x0"])
    Kp, Kd = pd_gains(r, Q_EQ)
    out = {}
    for name, KK in (("lqr", K), ("pd", lab.pd_state_feedback(r, Q_EQ, Kp, Kd))):
        sim = lab.simulate(r, lambda t, q, dq: lab.state_feedback_torque(q, dq, Q_EQ, u_eq, KK),
                           Q_EQ + x0[:2], x0[2:], C2["lqr"]["T"], DT)
        assert np.linalg.norm(sim["q"][-1] - Q_EQ) < 1e-5
        out[name] = lab.run_cost(sim, Q_EQ, u_eq, Q, R)
        assert out[name] == pytest.approx(lab.lqr_cost(A, B, Q, R, KK, x0), rel=0.02)
    assert out["lqr"] < out["pd"]
    assert out["lqr"] == pytest.approx(0.252415, rel=1e-4)


def test_course_arm_lqr():
    """12-state LQR of the course arm at q_home; Kleinman from the PD gains agrees with the Riccati solution."""
    arm = lab.plant_course_arm(CFG)
    q_eq = arm["q_home"]
    A, B, _ = lab.linearize(arm, q_eq)
    Lc = CFG["course_arm"]["lqr"]
    Q, R = lab.bryson_weights([Lc["q_max"]] * 6 + [Lc["dq_max"]] * 6, Lc["tau_fraction"] * arm["tau_max"])
    K, P = lab.lqr(A, B, Q, R)
    assert np.shape(K) == (6, 12)
    res = A.T @ P + P @ A - P @ B @ np.linalg.solve(R, B.T @ P) + Q
    assert np.abs(res).max() < 1e-7 * np.abs(Q).max()
    assert lab.is_hurwitz(A - B @ K)
    Kp, Kd = lab.pd_gains_from_inertia(arm, q_eq, CFG["course_arm"]["pd"]["zeta"], CFG["course_arm"]["pd"]["wn"])
    Kk, Pk, it = lab.kleinman_lqr(A, B, Q, R, lab.pd_state_feedback(arm, q_eq, Kp, Kd))
    assert np.allclose(Kk, K, rtol=1e-7, atol=1e-7 * np.abs(K).max())
    assert it <= 30
