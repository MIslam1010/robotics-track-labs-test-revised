"""Auto-checks for SBEG 461 Lab 6 (computed torque vs PID with model error on the course arm; task-space control).

    LAB_IMPL=solution pytest -q test_lab.py      # reference: all pass
    LAB_IMPL=starter  pytest -q test_lab.py      # untouched starter: every test that calls your code fails with
                                                 # NotImplementedError; the data/infrastructure tests pass
                                                 # (student copy: 19 failed, 4 passed, 1 skipped)

LAB_IMPL is the name of the module to test (default "solution"); it is imported from this folder.
Ground truth: modern_robotics (InverseDynamics, MassMatrix, GravityForces, ComputedTorque, ForwardDynamics,
JacobianBody), finite differences for the Jacobian derivative and the feedforward acceleration, the closed-loop
identities of MR Sec. 11.4 (computed torque gives q'' = qd'' + Kp e + Kd e'; PD + gravity compensation has
dV/dt = -q'^T (Kd + B) q', MR Eq. 11.43 with friction) and an independent Newton solution of the computed-torque
steady-state error under model error.
"""
import importlib
import os
import py_compile
import sys
from pathlib import Path

import modern_robotics as mr
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("LAB_IMPL", "starter")
lab = importlib.import_module(os.environ["LAB_IMPL"])
import course_arm as ca  # noqa: E402

CFG = lab.load_config()
RNG = np.random.default_rng(461)


def rand_state(n=6, s=1.0):
    return RNG.uniform(-2, 2, n), RNG.uniform(-s, s, n), RNG.uniform(-s, s, n)


def no_friction(r):
    r = dict(r)
    r["b"] = np.zeros(6)
    return r


# ============================================================================ data and infrastructure (provided code)
def test_config_and_cases():
    assert CFG["dt"] == 0.002 and CFG["payload"] == 0.25
    plant, model = lab.make_case("payload_unknown", CFG)
    assert plant["payload"] == 0.25 and model["payload"] == 0.0
    _, m120 = lab.make_case("masses_120", CFG)
    assert np.allclose(m120["Glist"][2], 1.2 * plant["Glist"][2])
    assert np.allclose(lab.tau_limits(plant, CFG), 0.6 * np.array([1.8, 3.0, 1.5, 0.6, 0.6, 0.4]))


def test_plant_matches_modern_robotics_forward_dynamics():
    plant, _ = lab.make_case("exact", CFG)
    P = lab.Plant(plant)
    for _ in range(4):
        q, dq, tau = rand_state()
        ref = mr.ForwardDynamics(q, dq, tau - plant["b"] * dq, plant["g"], np.zeros(6), plant["Mlist"],
                                 plant["Glist"], plant["Slist"])
        assert np.allclose(P.accel(q, dq, tau), ref, atol=1e-8)


def test_references_are_consistent():
    t = lab.time_grid(1.0, 0.001)
    q0, q1 = np.zeros(6), np.arange(6.0)
    qd, dqd, ddqd = lab.quintic_reference(q0, q1, 0.8, t)
    assert np.allclose(qd[0], q0) and np.allclose(qd[-1], q1) and np.allclose(dqd[[0, -1]], 0)
    assert np.allclose(np.gradient(qd, t, axis=0)[5:-5], dqd[5:-5], atol=1e-4)
    X0 = mr.FKinBody(ca.course_arm()["M"], mr.Adjoint(mr.TransInv(ca.course_arm()["M"])) @ ca.course_arm()["Slist"],
                     ca.course_arm()["q_home"])
    Xd, Vd, dVd = lab.cartesian_reference(X0, [0.01, -0.02, 0.03], 0.8, t)
    k, h = 300, t[1] - t[0]
    Vnum = mr.se3ToVec(mr.TransInv(Xd[k]) @ (Xd[k + 1] - Xd[k - 1]) / (2 * h))
    assert np.allclose(Vnum, Vd[k], atol=1e-6)
    assert np.allclose((Vd[k + 1] - Vd[k - 1]) / (2 * h), dVd[k], atol=1e-4)


def test_starter_compiles():
    py_compile.compile(str(HERE / "starter.py"), doraise=True)


# ============================================================================ Part A: gains
def test_ct_gains_and_ki_bound():
    kp, kd = lab.ct_gains(1.0, 20.0)
    assert (kp, kd) == pytest.approx((400.0, 40.0))
    kp, kd = lab.ct_gains(0.7, 10.0)
    assert (kp, kd) == pytest.approx((100.0, 14.0))
    kimax = lab.ct_ki_max(400.0, 40.0)
    assert kimax == pytest.approx(16000.0)
    for f, stable in ((0.9, True), (1.1, False)):
        r = np.roots([1, 40.0, 400.0, f * kimax])
        assert (np.max(r.real) < 0) == stable


def test_decentralized_gains():
    _, model = lab.make_case("exact", CFG)
    q = np.array(CFG["gains"]["gain_reference_q"])
    Kp, Ki, Kd = lab.decentralized_gains(model, q, 1.0, 20.0, 0.1)
    Md = np.diag(mr.MassMatrix(q, model["Mlist"], model["Glist"], model["Slist"]))
    assert np.allclose(Kp, 400 * Md) and np.allclose(Kd, 40 * Md - model["b"])
    assert np.allclose(Ki, 0.1 * (model["b"] + Kd) * Kp / Md)
    assert np.allclose((model["b"] + Kd) / (2 * np.sqrt(Kp * Md)), 1.0)      # zeta = 1 for every joint
    _, _, Kd0 = lab.decentralized_gains(model, q, 1.0, 20.0)
    assert np.allclose(Kd0, Kd)
    assert np.allclose(lab.decentralized_gains(model, q, 1.0, 20.0)[1], 0.0)


# ============================================================================ Part B: control laws
def test_pid_torque():
    q, dq, _ = rand_state()
    qd, dqd, eint = rand_state()
    Kp, Ki, Kd = np.arange(1, 7.0), 0.5, np.full(6, 0.2)
    ref = Kp * (qd - q) + 0.5 * eint + 0.2 * (dqd - dq)
    assert np.allclose(lab.pid_torque(q, dq, eint, qd, dqd, Kp, Ki, Kd), ref)


def test_pd_gravity_torque_uses_model_gravity_at_q():
    _, model = lab.make_case("payload_unknown", CFG)
    q, dq, eint = rand_state()
    qd, dqd, _ = rand_state()
    g = mr.GravityForces(q, model["g"], model["Mlist"], model["Glist"], model["Slist"])
    ref = 3.0 * (qd - q) + 0.1 * eint + 0.5 * (dqd - dq) + g
    assert np.allclose(lab.pd_gravity_torque(q, dq, eint, qd, dqd, 3.0, 0.1, 0.5, model), ref)


def test_feedforward_uses_reference_state():
    _, model = lab.make_case("exact", CFG)
    q, dq, eint = rand_state()
    qd, dqd, ddqd = rand_state()
    ff = mr.InverseDynamics(qd, dqd, ddqd, model["g"], np.zeros(6), model["Mlist"], model["Glist"], model["Slist"])
    ff = ff + model["b"] * dqd
    ref = ff + 2.0 * (qd - q) + 0.3 * eint + 0.4 * (dqd - dq)
    assert np.allclose(lab.feedforward_pd_torque(q, dq, eint, qd, dqd, ddqd, 2.0, 0.3, 0.4, model), ref)


def test_computed_torque_matches_modern_robotics():
    _, model = lab.make_case("masses_120", CFG)
    m0 = no_friction(model)
    for _ in range(3):
        q, dq, eint = rand_state()
        qd, dqd, ddqd = rand_state()
        # mr.ComputedTorque adds the current error to the integral it is given (Ki (eint + e)), so pass eint - e
        ref = mr.ComputedTorque(q, dq, eint - (qd - q), m0["g"], m0["Mlist"], m0["Glist"], m0["Slist"], qd, dqd,
                                ddqd, 400.0, 50.0, 40.0)
        assert np.allclose(lab.computed_torque(q, dq, eint, qd, dqd, ddqd, 400.0, 50.0, 40.0, m0), ref, atol=1e-9)
        # friction model enters h~ as b~ dq; vector gains act as diagonal matrices
        kv = np.array([400, 300, 200, 100, 50, 25.0])
        e, ed = qd - q, dqd - dq
        Mm = mr.MassMatrix(q, model["Mlist"], model["Glist"], model["Slist"])
        h = mr.InverseDynamics(q, dq, np.zeros(6), model["g"], np.zeros(6), model["Mlist"], model["Glist"],
                               model["Slist"]) + model["b"] * dq
        ref2 = Mm @ (ddqd + kv * e + 10 * eint + 0.1 * kv * ed) + h
        assert np.allclose(lab.computed_torque(q, dq, eint, qd, dqd, ddqd, kv, 10.0, 0.1 * kv, model), ref2,
                           atol=1e-9)


def test_computed_torque_linearizes_exact_plant():
    """Exact model: the plant acceleration under computed torque is qd'' + kp e + kd e' (MR Eq. 11.33, Ki = 0)."""
    plant, model = lab.make_case("exact", CFG)
    P = lab.Plant(plant)
    q, dq, _ = rand_state()
    qd, dqd, ddqd = rand_state()
    tau = lab.computed_torque(q, dq, np.zeros(6), qd, dqd, ddqd, 400.0, 0.0, 40.0, model)
    assert np.allclose(P.accel(q, dq, tau), ddqd + 400 * (qd - q) + 40 * (dqd - dq), atol=1e-7)


def test_computed_torque_with_scaled_model():
    """M~ = a M, h~ = a (c + g) + b dq:  q'' = a (qd'' + kp e + kd e') + (a - 1) M^-1 (c + g)  (model-error identity)."""
    plant = lab.arm_model(0.25, 1.0, CFG["friction_b"])
    model = lab.arm_model(0.25, 1.2, CFG["friction_b"])
    P = lab.Plant(plant)
    q, dq, _ = rand_state()
    qd, dqd, ddqd = rand_state()
    tau = lab.computed_torque(q, dq, np.zeros(6), qd, dqd, ddqd, 400.0, 0.0, 40.0, model)
    Mm = mr.MassMatrix(q, plant["Mlist"], plant["Glist"], plant["Slist"])
    cg = mr.InverseDynamics(q, dq, np.zeros(6), plant["g"], np.zeros(6), plant["Mlist"], plant["Glist"],
                            plant["Slist"])
    pred = 1.2 * (ddqd + 400 * (qd - q) + 40 * (dqd - dq)) + 0.2 * np.linalg.solve(Mm, cg)
    assert np.allclose(P.accel(q, dq, tau), pred, atol=1e-6)


def test_error_energy_and_its_derivative():
    """V of MR Eq. 11.41, and dV/dt = -dq^T (Kd + B) dq under PD + exact gravity compensation (MR Eq. 11.43 with
    viscous friction B), checked by differentiating V along the plant dynamics."""
    plant, model = lab.make_case("exact", CFG)
    P = lab.Plant(plant)
    q, dq, _ = rand_state(s=0.5)
    qd = q + RNG.uniform(-0.2, 0.2, 6)
    Kp, Kd = np.array([4, 4, 2, 1, 0.5, 0.1]), np.array([0.4, 0.4, 0.2, 0.1, 0.05, 0.01])
    Mm = mr.MassMatrix(q, model["Mlist"], model["Glist"], model["Slist"])
    e = qd - q
    assert lab.error_energy(q, dq, qd, Kp, model) == pytest.approx(0.5 * e @ (Kp * e) + 0.5 * dq @ Mm @ dq)
    tau = lab.pd_gravity_torque(q, dq, np.zeros(6), qd, np.zeros(6), Kp, 0.0, Kd, model)
    ddq = P.accel(q, dq, tau)
    h = 1e-6
    Vp = lab.error_energy(q + h * dq, dq + h * ddq, qd, Kp, model)
    Vm = lab.error_energy(q - h * dq, dq - h * ddq, qd, Kp, model)
    assert (Vp - Vm) / (2 * h) == pytest.approx(-dq @ ((Kd + plant["b"]) * dq), rel=1e-5, abs=1e-9)


def test_joint_controller_integral_and_dispatch():
    _, model = lab.make_case("exact", CFG)
    t = lab.time_grid(0.01, 0.002)
    qd, dqd, ddqd = (RNG.uniform(-1, 1, (len(t), 6)) for _ in range(3))
    q, dq = RNG.uniform(-1, 1, 6), RNG.uniform(-1, 1, 6)
    for law in ("pid", "pd_gravity", "feedforward", "computed_torque"):
        c = lab.JointController(law, t, qd, dqd, ddqd, 2.0, 5.0, 0.3, model)
        c.reset()
        eint = np.zeros(6)
        for k in range(3):
            eint = eint + (qd[k] - q) * 0.002
            tau = c(k, q, dq)
        assert np.allclose(c.eint, eint)
        if law == "pid":
            ref = 2.0 * (qd[2] - q) + 5.0 * eint + 0.3 * (dqd[2] - dq)
        elif law == "pd_gravity":
            ref = 2.0 * (qd[2] - q) + 5.0 * eint + 0.3 * (dqd[2] - dq) + mr.GravityForces(
                q, model["g"], model["Mlist"], model["Glist"], model["Slist"])
        elif law == "feedforward":
            ref = 2.0 * (qd[2] - q) + 5.0 * eint + 0.3 * (dqd[2] - dq) + mr.InverseDynamics(
                qd[2], dqd[2], ddqd[2], model["g"], np.zeros(6), model["Mlist"], model["Glist"], model["Slist"]) \
                + model["b"] * dqd[2]
        else:
            ref = mr.ComputedTorque(q, dq, eint - (qd[2] - q), model["g"], model["Mlist"], model["Glist"], model["Slist"],
                                    qd[2], dqd[2], ddqd[2], 2.0, 5.0, 0.3) + model["b"] * dq
        assert np.allclose(tau, ref, atol=1e-9), law
        c.reset()
        assert np.allclose(c.eint, 0)


def test_run_metrics():
    t = np.array([0.0, 1.0, 2.0])
    qd = np.zeros((3, 2))
    q = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 1.0]])
    tau = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, -2.0]])
    m = lab.run_metrics(t, qd, q, tau)
    assert m["max_err"] == pytest.approx(5.0) and m["final_err"] == pytest.approx(1.0)
    assert m["rms_err"] == pytest.approx(np.sqrt((0 + 25 + 1) / 3))
    assert m["effort"] == pytest.approx(0.5 * (1 + 2) + 0.5 * (2 + 4))
    assert np.allclose(m["peak_tau"], [1.0, 2.0])


# ============================================================================ closed-loop behaviour (short runs)
def _joint_run(case, law, T_end=0.8, ki=0.0):
    jm, g = CFG["joint_move"], CFG["gains"]
    t = lab.time_grid(T_end, CFG["dt"])
    q0 = np.array(jm["q0"])
    qd, dqd, ddqd = lab.quintic_reference(q0, q0 + np.array(jm["dq"]), jm["T"], t)
    plant, model = lab.make_case(case, CFG)
    if law == "computed_torque":
        kp, kd = lab.ct_gains(g["zeta"], g["wn"])
        c = lab.JointController(law, t, qd, dqd, ddqd, kp, ki, kd, model)
    else:
        Kp, Ki, Kd = lab.decentralized_gains(model, g["gain_reference_q"], g["zeta"], g["wn"], ki)
        c = lab.JointController(law, t, qd, dqd, ddqd, Kp, Ki, Kd, model)
    out = lab.simulate_arm(plant, c, t, q0, substeps=CFG["substeps"], tau_limit=lab.tau_limits(plant, CFG))
    return lab.run_metrics(t, qd, out["q"], out["tau"])


def test_closed_loop_ranking_exact_model():
    ct = _joint_run("exact", "computed_torque")
    pdg = _joint_run("exact", "pd_gravity")
    pid = _joint_run("exact", "pid", ki=0.1)
    assert ct["max_err"] < 2e-3
    assert ct["max_err"] < 0.1 * pdg["max_err"] and pdg["max_err"] < pid["max_err"]


def test_ct_steady_state_error_under_model_error():
    """Setpoint hold with M~ = 1.2 M: at rest  kp M~(q) e = g(q) - g~(q) = -0.2 g(q), q = qd - e (solved by Newton);
    the simulated computed-torque loop must settle there."""
    plant, model = lab.make_case("masses_120", CFG)
    qset = np.array(CFG["setpoint"]["q0"])
    t = lab.time_grid(1.2, CFG["dt"])
    N = len(t)
    Z = np.zeros((N, 6))
    c = lab.JointController("computed_torque", t, np.tile(qset, (N, 1)), Z, Z, 400.0, 0.0, 40.0, model)
    out = lab.simulate_arm(plant, c, t, qset, substeps=CFG["substeps"])
    e = np.zeros(6)
    for _ in range(30):
        def F(e_):
            q = qset - e_
            Mm = mr.MassMatrix(q, model["Mlist"], model["Glist"], model["Slist"])
            gp = mr.GravityForces(q, plant["g"], plant["Mlist"], plant["Glist"], plant["Slist"])
            gm = mr.GravityForces(q, model["g"], model["Mlist"], model["Glist"], model["Slist"])
            return 400.0 * Mm @ e_ - (gp - gm)
        Jn = np.column_stack([(F(e + 1e-7 * u) - F(e - 1e-7 * u)) / 2e-7 for u in np.eye(6)])
        e = e - np.linalg.solve(Jn, F(e))
    assert np.allclose(qset - out["q"][-1], e, atol=2e-4)
    assert np.linalg.norm(e) > 0.02


def test_setpoint_pd_gravity_energy_decreases():
    sp, g = CFG["setpoint"], CFG["gains"]
    plant, model = lab.make_case("exact", CFG)
    q0 = np.array(sp["q0"])
    qset = q0 + np.array(sp["dq0_offset"])
    t = lab.time_grid(0.8, CFG["dt"])
    N = len(t)
    Kp, _, Kd = lab.decentralized_gains(model, g["gain_reference_q"], g["zeta"], g["wn"])
    Z = np.zeros((N, 6))
    c = lab.JointController("pd_gravity", t, np.tile(qset, (N, 1)), Z, Z, Kp, 0.0, Kd, model)
    out = lab.simulate_arm(plant, c, t, q0, substeps=CFG["substeps"])
    V = np.array([lab.error_energy(out["q"][k], out["dq"][k], qset, Kp, plant) for k in range(0, N, 5)])
    assert np.all(np.diff(V) <= 1e-12) and V[-1] < 1e-3 * V[0]


# ============================================================================ Part C: task space
def test_body_jacobian_dot():
    r = lab.arm_model(0.25)
    for _ in range(3):
        q, dq, _ = rand_state()
        h = 1e-6
        num = (mr.JacobianBody(r["Blist"], q + h * dq) - mr.JacobianBody(r["Blist"], q - h * dq)) / (2 * h)
        assert np.allclose(lab.body_jacobian_dot(r["Blist"], q, dq), num, atol=1e-7)


def test_task_errors():
    r = lab.arm_model(0.25)
    q1, q2 = r["q_home"], r["q_home"] + RNG.uniform(-0.2, 0.2, 6)
    X, Xd = (mr.FKinBody(r["M"], r["Blist"], q) for q in (q1, q2))
    V, Vd = RNG.normal(size=6), RNG.normal(size=6)
    Xe, Ve = lab.task_errors(X, V, Xd, Vd)
    assert np.allclose(X @ mr.MatrixExp6(mr.VecTose3(Xe)), Xd, atol=1e-10)
    assert np.allclose(Ve, mr.Adjoint(mr.TransInv(X) @ Xd) @ Vd - V)
    Xe0, Ve0 = lab.task_errors(X, V, X, V)
    assert np.allclose(Xe0, 0, atol=1e-12) and np.allclose(Ve0, 0, atol=1e-12)


def test_task_feedforward_accel():
    r = lab.arm_model(0.25)
    X0 = mr.FKinBody(r["M"], r["Blist"], r["q_home"])
    W1, W2 = RNG.normal(size=6), RNG.normal(size=6)
    Xa = lambda s: X0 @ mr.MatrixExp6(mr.VecTose3(W1 * (s + 0.3 * s**2)))  # noqa: E731
    Xb = lambda s: X0 @ mr.MatrixExp6(mr.VecTose3(W2 * (s - 0.2 * s**3)))  # noqa: E731

    def twist(f, s, h=1e-6):
        return mr.se3ToVec(mr.TransInv(f(s)) @ (f(s + h) - f(s - h)) / (2 * h))

    def W(s):
        return mr.Adjoint(mr.TransInv(Xa(s)) @ Xb(s)) @ twist(Xb, s)

    s, h = 0.4, 1e-4
    num = (W(s + h) - W(s - h)) / (2 * h)
    dVd = (twist(Xb, s + h) - twist(Xb, s - h)) / (2 * h)
    got = lab.task_feedforward_accel(Xa(s), twist(Xa, s), Xb(s), twist(Xb, s), dVd)
    assert np.allclose(got, num, atol=1e-6)


def test_model_task_space_dynamics_and_ct_equal_eq_11_47():
    _, model = lab.make_case("masses_120", CFG)
    q = model["q_home"] + RNG.uniform(-0.3, 0.3, 6)
    dq, ddq = RNG.uniform(-1, 1, 6), RNG.uniform(-1, 1, 6)
    Lam, eta = lab.model_task_space_dynamics(model, q, dq)
    assert np.allclose(Lam, Lam.T, atol=1e-10) and np.all(np.linalg.eigvalsh(0.5 * (Lam + Lam.T)) > 0)
    J = mr.JacobianBody(model["Blist"], q)
    Jd = lab.body_jacobian_dot(model["Blist"], q, dq)
    tau_id = mr.InverseDynamics(q, dq, ddq, model["g"], np.zeros(6), model["Mlist"], model["Glist"], model["Slist"])
    tau_id = tau_id + model["b"] * dq
    assert np.allclose(J.T @ (Lam @ (J @ ddq + Jd @ dq) + eta), tau_id, atol=1e-8)        # MR Eq. 8.89
    X = mr.FKinBody(model["M"], model["Blist"], q)
    Xd = mr.FKinBody(model["M"], model["Blist"], q + RNG.uniform(-0.05, 0.05, 6))
    Vd, dVd, Xint = RNG.normal(size=6), RNG.normal(size=6), RNG.normal(size=6) * 0.01
    Xe, Ve = lab.task_errors(X, J @ dq, Xd, Vd)
    a = lab.task_feedforward_accel(X, J @ dq, Xd, Vd, dVd) + 400 * Xe + 50 * Xint + 40 * Ve
    ref = J.T @ (Lam @ a + eta)                                                                # MR Eq. 11.47
    assert np.allclose(lab.task_space_ct(q, dq, Xint, Xd, Vd, dVd, 400.0, 50.0, 40.0, model), ref, atol=1e-8)


def test_task_controller_tracks_line_with_exact_model():
    tm, g = CFG["task_move"], CFG["gains"]
    plant, model = lab.make_case("exact", CFG)
    t = lab.time_grid(0.6, CFG["dt"])
    q0 = np.array(tm["q0"])
    X0 = mr.FKinBody(model["M"], model["Blist"], q0)
    Xd, Vd, dVd = lab.cartesian_reference(X0, tm["dp_s"], tm["T"], t)
    kp, kd = lab.ct_gains(g["zeta"], g["wn"])
    c = lab.TaskController(t, Xd, Vd, dVd, kp, 0.0, kd, model)
    out = lab.simulate_arm(plant, c, t, q0, substeps=CFG["substeps"], tau_limit=lab.tau_limits(plant, CFG))
    err = [np.linalg.norm(mr.FKinBody(plant["M"], plant["Blist"], q)[:3, 3] - X[:3, 3]) for q, X in zip(out["q"], Xd)]
    assert max(err) < 1e-4                                   # < 0.1 mm along a 50 mm line
    assert np.linalg.norm(Xd[-1][:3, 3] - X0[:3, 3]) > 0.02  # the reference really moved
