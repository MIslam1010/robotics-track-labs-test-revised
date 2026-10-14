"""Auto-checks for SBEG 461 Lab 5 (error dynamics, joint PID in simulation, ROS 2 controller node logic).

    LAB_IMPL=solution pytest -q test_lab.py      # reference: all pass
    LAB_IMPL=starter  pytest -q test_lab.py      # untouched starter: every test that calls your code fails with
                                                 # NotImplementedError; the 3 data/infrastructure tests pass

LAB_IMPL is the name of the module to test (default "solution"); it is imported from this folder.
Ground truth: modern_robotics (MassMatrix, GravityForces of the course arm), the matrix exponential of the
companion matrix (MR Sec. 11.2.2, p. 410), scipy's ODE solver for the continuous closed loop, the textbook values
of MR pp. 415-427 (overshoot for zeta = 0.1, 0.5, 0.8; Kp = 20 root locus; Fig. 11.13 gains) and a loop
implementation of PID without anti-windup written in this file.
"""
import importlib
import json
import math
import os
import py_compile
import sys
from pathlib import Path

import numpy as np
import pytest
import modern_robotics as mr
from scipy.integrate import solve_ivp
from scipy.linalg import expm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("LAB_IMPL", "starter")
lab = importlib.import_module(os.environ["LAB_IMPL"])
import course_arm as ca  # noqa: E402

ENC = 2 * np.pi / 4096


def ref_unit_response(zeta, wn, t):
    A = np.array([[0.0, 1.0], [-wn**2, -2 * zeta * wn]])
    return np.array([(expm(A * ti) @ np.array([1.0, 0.0]))[0] for ti in np.atleast_1d(t)])


# ============================================================================ data and infrastructure
def test_plant_data_matches_modern_robotics():
    """Course-arm plants: M = MassMatrix[j, j] and g(q) = GravityForces[j] with the other joints at q_home."""
    arm = ca.course_arm(payload=0.25)
    rng = np.random.default_rng(0)
    for name, j in (("arm_j1", 0), ("arm_j2", 1)):
        p = lab.load_plant(name)
        assert p["locked_q"] == pytest.approx(list(arm["q_home"]))
        for qj in rng.uniform(-2.5, 2.5, 4):
            q = arm["q_home"].copy()
            q[j] = qj
            Mm = mr.MassMatrix(q, arm["Mlist"], arm["Glist"], arm["Slist"])
            g = mr.GravityForces(q, arm["g"], arm["Mlist"], arm["Glist"], arm["Slist"])
            assert p["M"] == pytest.approx(Mm[j, j], abs=1e-12)          # M_jj does not depend on q_j
            assert lab.plant_gravity(p, qj) == pytest.approx(g[j], abs=1e-12)
        assert p["tau_max"] == pytest.approx(0.6 * arm["tau_max"][j])
    knee, eod = lab.load_plant("knee"), lab.load_plant("eod_wrist")
    assert knee["M"] == pytest.approx(0.045 + 2.4 * 0.21**2) and knee["g_sin"] == pytest.approx(2.4 * 9.81 * 0.21)
    assert eod["M"] == pytest.approx(0.008 + 3.5 * 0.10**2) and eod["g_cos"] == pytest.approx(3.5 * 9.81 * 0.10)


def test_ros2_files_compile_and_import_without_ros():
    for f in ("joint_sim_node.py", "joint_pid_node.py"):
        py_compile.compile(str(HERE / "ros2" / f), doraise=True)
    sys.path.insert(0, str(HERE / "ros2"))
    importlib.import_module("joint_sim_node")
    importlib.import_module("joint_pid_node")


# ============================================================================ Part A: error dynamics
def test_second_order_params():
    assert lab.second_order_params(20.0, 100.0) == pytest.approx((1.0, 10.0))       # MR p. 418, Ki = 100
    assert lab.second_order_params(20.0, 400.0) == pytest.approx((0.5, 20.0))       # case III
    z, wn = lab.second_order_params(20.0, 44.4)
    assert z == pytest.approx(1.5, abs=1e-3) and wn == pytest.approx(math.sqrt(44.4))
    with pytest.raises(ValueError):
        lab.second_order_params(1.0, 0.0)


def test_char_roots_and_stability():
    r = np.sort_complex(lab.char_roots([1.0, 20.0, 100.0]))
    assert np.allclose(r, [-10, -10], atol=1e-6)
    assert lab.is_stable([1, 1, 1]) and not lab.is_stable([1, -1, 1]) and not lab.is_stable([1, 0, 4])
    rng = np.random.default_rng(1)
    for _ in range(50):                       # cubic: stable iff all coefficients > 0 and a2 a1 > a0 (MR p. 410)
        a2, a1, a0 = rng.uniform(0.1, 5, 3)
        assert lab.is_stable([1, a2, a1, a0]) == (a2 * a1 > a0)


@pytest.mark.parametrize("zeta", [0.0, 0.2, 0.5, 1.0, 1.5, 3.0])
def test_unit_error_response_matches_matrix_exponential(zeta):
    t = np.linspace(0, 2, 41)
    assert np.allclose(lab.unit_error_response(zeta, 5.0, t), ref_unit_response(zeta, 5.0, t), atol=1e-9)


def test_overshoot_formula_and_inverse():
    assert lab.overshoot_from_zeta(0.1) == pytest.approx(73, abs=0.5)               # MR p. 415
    assert lab.overshoot_from_zeta(0.5) == pytest.approx(16, abs=0.5)
    assert lab.overshoot_from_zeta(0.8) == pytest.approx(1.5, abs=0.05)
    assert lab.overshoot_from_zeta(1.0) == 0.0 and lab.overshoot_from_zeta(2.0) == 0.0
    for z in (0.05, 0.3, 0.7, 0.95):
        assert lab.zeta_from_overshoot(lab.overshoot_from_zeta(z)) == pytest.approx(z, abs=1e-12)


def test_step_metrics_first_and_second_order():
    t = np.linspace(0, 3, 300001)
    m = lab.step_metrics(t, np.exp(-t / 0.5))
    assert m["overshoot"] == 0.0 and m["ess"] == pytest.approx(0, abs=1e-2)
    assert lab.step_metrics(t, np.exp(-t / 0.5), ess=0.0)["settling_time"] == pytest.approx(0.5 * math.log(50), abs=1e-4)
    for z in (0.2, 0.5, 0.8):
        e = lab.unit_error_response(z, 10.0, t)
        m = lab.step_metrics(t, e, ess=0.0)
        assert m["overshoot"] == pytest.approx(lab.overshoot_from_zeta(z), abs=1e-3)
        bad = np.nonzero(np.abs(e) > 0.02)[0][-1]
        assert m["settling_time"] == pytest.approx(t[bad + 1])
    # offset response: e from 2 to ess = 1 with overshoot
    e = 1 + lab.unit_error_response(0.5, 10.0, t)
    assert lab.step_metrics(t, e)["overshoot"] == pytest.approx(16.3, abs=0.1)
    assert math.isnan(lab.step_metrics(t[:100], np.exp(-t[:100]), ess=0.0)["settling_time"])


# ============================================================================ Part B: velocity inputs
def test_pi_velocity_gains():
    assert lab.pi_velocity_gains(1.0, 10.0) == pytest.approx((20.0, 100.0))
    assert lab.pi_velocity_gains(0.5, 20.0) == pytest.approx((20.0, 400.0))


def test_p_control_setpoint_and_ramp():
    dt, kp = 1e-3, 20.0
    t = lab.time_grid(2.0, dt)
    r = lab.simulate_velocity_joint(t, np.zeros_like(t), np.zeros_like(t), kp, q0=-1.0)
    assert np.allclose(r["e"], (1 - kp * dt) ** np.arange(len(t)))                 # discrete first-order decay
    assert np.allclose(r["u"], kp * r["e"])
    qd, vd = lab.ramp_reference(t, 0.0, 0.5)
    r = lab.simulate_velocity_joint(t, qd, vd, kp, q0=0.0)
    assert r["e"][-1] == pytest.approx(0.5 / kp, abs=1e-9)                          # ess = c / Kp (MR p. 417)


def test_pi_and_feedforward_remove_ramp_error():
    t = lab.time_grid(3.0, 1e-3)
    qd, vd = lab.ramp_reference(t, 0.0, 0.5)
    r = lab.simulate_velocity_joint(t, qd, vd, 20.0, 100.0, q0=-0.2)
    assert abs(r["e"][-1]) < 1e-6
    r = lab.simulate_velocity_joint(t, qd, vd, 20.0, 0.0, q0=0.0, feedforward=True)
    assert np.max(np.abs(r["e"])) < 1e-12                                          # MR Eq. 11.15, no initial error
    r = lab.simulate_velocity_joint(t, qd, vd, 20.0, 100.0, q0=-0.2, feedforward=True)
    assert abs(r["e"][-1]) < 1e-6


def test_velocity_saturation_and_anti_windup():
    t = lab.time_grid(4.0, 1e-3)
    qd, vd = lab.step_reference(t, 0.0, 1.0)
    r = lab.simulate_velocity_joint(t, qd, vd, 20.0, 100.0, q0=0.0, qdot_max=1.4)
    assert np.max(np.abs(r["u"])) <= 1.4 + 1e-12
    # reference loop without anti-windup
    q, E, emin = 0.0, 0.0, 1.0
    for k in range(len(t)):
        e = qd[k] - q
        E += e * 1e-3
        q += float(np.clip(20 * e + 100 * E, -1.4, 1.4)) * 1e-3
        emin = min(emin, qd[k] - q)
    assert r["e"].min() > emin + 0.05                                               # far less overshoot
    assert abs(r["e"][-1]) < 1e-4


# ============================================================================ Part C: torque PID
def test_pd_gains_reproduce_mr_fig_11_13():
    kp, kd = lab.pd_gains(0.5, 0.1, 1.0, 2.1)
    assert (kp, kd) == pytest.approx((2.205, 2.0))                                  # MR Fig. 11.13 caption
    assert lab.ki_max(0.5, 0.1, kp, kd) == pytest.approx(2.1 * 2.205 / 0.5)
    c = lab.pid_char_poly(0.5, 0.1, kp, 0.0, kd)
    assert np.allclose(c, [1, 4.2, 4.41])
    assert np.allclose(lab.pid_char_poly(0.5, 0.1, kp, 1.0, kd), [1, 4.2, 4.41, 2.0])


@pytest.mark.parametrize("name", ["arm_j2", "knee", "eod_wrist", "mr_link"])
def test_ki_bound_is_the_stability_boundary(name):
    p = lab.load_plant(name)
    kp, kd = lab.pd_gains(p["M"], p["b"], 1.0, 15.0)
    km = lab.ki_max(p["M"], p["b"], kp, kd)
    assert lab.is_stable(lab.pid_char_poly(p["M"], p["b"], kp, 0.9 * km, kd))
    assert not lab.is_stable(lab.pid_char_poly(p["M"], p["b"], kp, 1.1 * km, kd))


def test_pd_gravity_error():
    for name, qd, kp in (("mr_link", 0.0, 2.205), ("arm_j2", -0.6, 9.62), ("eod_wrist", 0.3, 10.0), ("knee", 1.0, 50.0)):
        p = lab.load_plant(name)
        e = lab.pd_gravity_error(p, kp, qd)
        assert kp * e == pytest.approx(lab.plant_gravity(p, qd - e), abs=1e-10)     # MR Eq. 11.27 at rest
    assert lab.pd_gravity_error(lab.load_plant("arm_j1"), 5.0, 0.3) == 0.0


def test_joint_pid_terms_and_differencing():
    c = lab.JointPID(2.0, 3.0, 0.5, 0.01)
    c.reset(0.0)
    tau = c.update(1.0, 0.0, 0.0)                     # first call: qdot = (0 - 0)/dt = 0
    assert tau == pytest.approx(2.0 * 1 + 3.0 * 0.01 + 0.5 * 0)
    tau = c.update(1.0, 0.2, 0.1)                     # qdot = 10, e = 0.9, e_dot = -9.8, eint = 0.019
    assert tau == pytest.approx(2 * 0.9 + 3 * 0.019 + 0.5 * (0.2 - 10.0))
    assert c.eint == pytest.approx(0.019)
    assert c.terms["P"] == pytest.approx(1.8) and c.terms["D"] == pytest.approx(-4.9)
    tau = c.update(1.0, 0.0, 0.2, qdot=1.0, tau_ff=0.7)
    assert tau == pytest.approx(2 * 0.8 + 3 * (0.019 + 0.008) + 0.5 * (-1.0) + 0.7)
    c.reset()
    assert c.eint == 0.0 and c.update(0.0, 0.0, 5.0) == pytest.approx(-10 + 3 * (-0.05))


def test_joint_pid_saturation_and_anti_windup():
    c = lab.JointPID(10.0, 50.0, 0.0, 0.01, tau_max=1.0)
    c.reset(0.0)
    for _ in range(20):
        assert c.update(1.0, 0.0, 0.0, 0.0) == pytest.approx(1.0)
        assert c.saturated
    assert c.eint == pytest.approx(0.0)                                             # clamped
    c2 = lab.JointPID(10.0, 50.0, 0.0, 0.01, tau_max=1.0, anti_windup=False)
    c2.reset(0.0)
    for _ in range(20):
        c2.update(1.0, 0.0, 0.0, 0.0)
    assert c2.eint == pytest.approx(0.2)
    # leaving saturation: an error of the opposite sign integrates even while clipped
    c.eint = 0.5
    c.update(0.0, 0.0, 0.2, 0.0)
    assert c.eint == pytest.approx(0.5 - 0.002)


def test_simulate_joint_linear_matches_analytic():
    p = dict(lab.load_plant("arm_j1"), tau_max=None)
    zeta, wn = 0.6, 12.0
    kp, kd = lab.pd_gains(p["M"], p["b"], zeta, wn)
    t = lab.time_grid(1.5, 1e-4)
    qd, vd = lab.step_reference(t, 0.0, 0.0)
    r = lab.simulate_joint(p, lab.JointPID(kp, 0.0, kd, 1e-4), t, qd, vd, q0=-0.4)
    assert np.allclose(r["e"] / 0.4, lab.unit_error_response(zeta, wn, t), atol=2e-3)
    assert r["q"][0] == -0.4 and r["dq"][0] == 0.0
    assert np.allclose(r["tau"], kp * r["e"] - kd * r["dq"])


def test_simulate_joint_matches_continuous_ode_with_gravity():
    p = lab.load_plant("eod_wrist")
    kp, kd = lab.pd_gains(p["M"], p["b"], 1.0, 15.0)
    t = lab.time_grid(1.0, 2e-4)
    qd, vd = lab.step_reference(t, 0.5, 0.5)
    r = lab.simulate_joint(p, lab.JointPID(kp, 0.0, kd, 2e-4), t, qd, vd, q0=0.0)

    def f(_, x):
        tau = float(np.clip(kp * (0.5 - x[0]) - kd * x[1], -p["tau_max"], p["tau_max"]))
        return [x[1], (tau - p["b"] * x[1] - lab.plant_gravity(p, x[0])) / p["M"]]
    sol = solve_ivp(f, (0, 1.0), [0.0, 0.0], t_eval=t, rtol=1e-10, atol=1e-12, max_step=1e-3)
    assert np.max(np.abs(sol.y[0] - r["q"])) < 2e-3
    assert r["e"][-1] == pytest.approx(lab.pd_gravity_error(p, kp, 0.5), abs=1e-4)


def test_mr_fig_11_13_pd_and_pid():
    p = lab.load_plant("mr_link")
    t = lab.time_grid(10.0, 1e-3)
    qd, vd = lab.step_reference(t, 0.0, 0.0)
    pd = lab.simulate_joint(p, lab.JointPID(2.205, 0.0, 2.0, 1e-3), t, qd, vd, q0=-np.pi / 2)
    pid = lab.simulate_joint(p, lab.JointPID(2.205, 1.0, 2.0, 1e-3), t, qd, vd, q0=-np.pi / 2)
    assert pd["e"][-1] == pytest.approx(lab.pd_gravity_error(p, 2.205, 0.0), abs=1e-4)
    assert abs(pid["e"][-1]) < 0.01 and pid["e"].min() < -0.05                    # overshoot, then e -> 0


def test_arm_shoulder_power_on_sag_and_integral_action():
    p = lab.load_plant("arm_j2")
    kp, kd = lab.pd_gains(p["M"], p["b"], 1.0, 20.0)
    t = lab.time_grid(3.0, 1e-3)
    qd, vd = lab.step_reference(t, -0.6, -0.6)
    pd = lab.simulate_joint(p, lab.JointPID(kp, 0.0, kd, 1e-3, p["tau_max"]), t, qd, vd, q0=-0.6)
    assert pd["e"][-1] == pytest.approx(lab.pd_gravity_error(p, kp, -0.6), abs=1e-5)
    ki = 0.1 * lab.ki_max(p["M"], p["b"], kp, kd)
    pid = lab.simulate_joint(p, lab.JointPID(kp, ki, kd, 1e-3, p["tau_max"]), t, qd, vd, q0=-0.6)
    assert abs(pid["e"][-1]) < 1e-4 and pid["e"].min() > pd["e"][-1]
    assert np.max(np.abs(pid["tau"])) <= p["tau_max"] + 1e-12


def test_integral_gain_above_bound_diverges_in_simulation():
    p = lab.load_plant("arm_j2")
    kp, kd = lab.pd_gains(p["M"], p["b"], 1.0, 20.0)
    km = lab.ki_max(p["M"], p["b"], kp, kd)
    t = lab.time_grid(5.0, 1e-3)
    qd, vd = lab.step_reference(t, -0.6, -0.6)
    ok = lab.simulate_joint(p, lab.JointPID(kp, 0.9 * km, kd, 1e-3), t, qd, vd, q0=-0.6)
    bad = lab.simulate_joint(p, lab.JointPID(kp, 1.1 * km, kd, 1e-3), t, qd, vd, q0=-0.6)
    assert np.max(np.abs(ok["e"][-1000:])) < 0.02 < 1.0 < np.max(np.abs(bad["e"][-1000:]))


def test_encoder_differencing_makes_torque_chatter():
    t, qd, vd, _ = lab.load_knee_swing()
    p = lab.load_plant("knee")
    kp, kd = lab.pd_gains(p["M"], p["b"], 1.0, 30.0)
    a = lab.simulate_joint(p, lab.JointPID(kp, 0, kd, 1e-3, p["tau_max"]), t, qd, vd, qd[0])
    b = lab.simulate_joint(p, lab.JointPID(kp, 0, kd, 1e-3, p["tau_max"]), t, qd, vd, qd[0],
                           velocity="difference", enc_res=ENC)
    assert np.max(np.abs(a["e"])) < np.deg2rad(5.0)
    assert np.std(np.diff(b["tau"])) > 10 * np.std(np.diff(a["tau"]))


def test_ros_logic_matches_offline_simulation():
    sys.path.insert(0, str(HERE / "ros2"))
    sim_mod = importlib.import_module("joint_sim_node")
    pid_mod = importlib.import_module("joint_pid_node")
    dt, sub = 0.005, 10
    p = lab.load_plant("arm_j2")
    ctrl = pid_mod.default_controller("arm_j2", 1.0, 20.0, dt, ki=30.0)
    logic = pid_mod.PidNodeLogic(ctrl, "joint2", reference="step", q_target=-0.3, t_step=0.5)
    sim = sim_mod.JointSimLogic(p, dt, -0.6, 0.0, sub)
    q, dq, _ = sim.state()
    Q = []
    for _ in range(400):
        Q.append(q)
        tau, dbg = logic.on_joint_state(["joint2"], [q], [dq])
        sim.apply(tau)
        q, dq, _ = sim.step()
    t = lab.time_grid(399 * dt, dt)
    qd, vd = lab.step_reference(t, -0.6, -0.3, 0.5)
    kp, kd = lab.pd_gains(p["M"], p["b"], 1.0, 20.0)
    ref = lab.simulate_joint(p, lab.JointPID(kp, 30.0, kd, dt, p["tau_max"]), t, qd, vd, -0.6, substeps=sub)
    assert np.allclose(Q, ref["q"], atol=1e-12)
    assert len(dbg) == 9 and dbg[0] == pytest.approx(399 * dt)
