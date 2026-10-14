"""Auto-checks for SBEG 461 Lab 8 (impedance control on a haptic device and a simulated rehabilitation robot).

    LAB_IMPL=solution pytest -q test_lab.py      # reference: all pass
    LAB_IMPL=starter  pytest -q test_lab.py      # untouched starter: every test that calls your code fails with
                                                 # NotImplementedError; the 4 data/infrastructure tests pass

LAB_IMPL is the name of the module to test (default "solution"); it is imported from this folder.
Ground truth: modern_robotics (MassMatrix, GravityForces, JacobianSpace, ForwardDynamics), hand formulas
(parallel-axis theorem, reflected rotor inertia), the matrix exponential of the continuous plant, the exact
analytic damped response, independent loop code written in this file, and the apparent masses quoted in the
461 project scenario sheet S1 (0.70, 0.81, 0.99 kg).
"""
import importlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import modern_robotics as mr
from scipy.linalg import expm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("LAB_IMPL", "starter")
lab = importlib.import_module(os.environ["LAB_IMPL"])
import course_models as cm  # noqa: E402

HK = cm.hapkit()
RH = cm.rehab_2r()
DEV = json.loads((HERE / "data" / "device.json").read_text())


def lin_dev(**kw):
    d = dict(DEV)
    d.update(kw)
    return d


# ============================================================================ data and infrastructure
def test_rehab_dynamics_matches_modern_robotics():
    rng = np.random.default_rng(1)
    for _ in range(5):
        q, dq = rng.uniform([0.2, 0.3], [2.8, 2.6]), rng.normal(size=2)
        Mq, c = lab.rehab_dynamics(q, dq)
        assert np.allclose(Mq, mr.MassMatrix(q, RH["Mlist"], RH["Glist"], RH["Slist"]), atol=1e-12)
        assert np.allclose(c, mr.VelQuadraticForces(q, dq, RH["Mlist"], RH["Glist"], RH["Slist"]), atol=1e-12)
        assert np.allclose(lab.rehab_fk(q), mr.FKinSpace(RH["M"], RH["Slist"], q)[:2, 3])
        assert np.allclose(lab.rehab_fk(lab.rehab_ik(lab.rehab_fk(q))), lab.rehab_fk(q))
    assert np.allclose(lab.rehab_dynamics(RH["q_home"], [0, 0])[0], [[0.3545, 0.0628], [0.0628, 0.06569]], atol=1e-4)


def test_device_file_and_free_decay_data():
    assert DEV["T"] == pytest.approx(1.0 / HK["params"]["loop_rate_hz"])
    assert DEV["force_max"] == HK["params"]["force_max"]
    for K in (50, 100, 150):
        t, x = lab.load_free_decay(K)
        assert len(t) == 3001 and t[1] - t[0] == pytest.approx(DEV["T"])
        assert np.allclose(np.round(x / DEV["x_res"]) * DEV["x_res"], x, atol=1e-9)   # quantised
        assert x[0] == pytest.approx(0.012, abs=DEV["x_res"])


def test_rk4_and_jacobian_dot():
    q, dq, h = np.array([0.7, 1.4]), np.array([0.3, -0.5]), 1e-6
    Jd = (lab.rehab_jacobian(q + h * dq) - lab.rehab_jacobian(q - h * dq)) / (2 * h)
    assert np.allclose(lab.rehab_jacobian_dot(q, dq), Jd, atol=1e-6)
    x = lab.rk4_step(lambda s: -s, np.array([1.0]), 0.1)
    assert x[0] == pytest.approx(math.exp(-0.1), abs=1e-6)


# ============================================================================ Part A -- device model
def test_handle_mass():
    m_h = lab.handle_mass(HK)
    I_pivot = 1.5e-4 + 0.10 * 0.010**2                  # parallel-axis theorem
    J_refl = 1.0e-6 * (0.073 / 0.0042) ** 2             # apparent rotor inertia G^2 J (MR Sec. 8.9.2)
    assert m_h == pytest.approx((I_pivot + J_refl) / 0.090**2, rel=1e-9)
    assert m_h == pytest.approx(DEV["m_h"], abs=1e-5)


def test_gravity_stiffness():
    kg = lab.gravity_stiffness(HK)
    assert kg == pytest.approx(0.10 * 9.81 * 0.010 / 0.090**2, rel=1e-6)
    assert kg > 0                                        # sector below the pivot: restoring


def test_handle_force_to_current():
    I = lab.handle_force_to_current(np.array([2.0, -1.0]), HK, DEV["k_t"])
    assert np.allclose(I, np.array([2.0, -1.0]) * 0.090 / (0.073 / 0.0042) / 0.010)
    assert abs(I[0]) < DEV["i_max"]


def test_fit_free_decay_on_exact_signal():
    wn, ze = 31.0, 0.04
    t = np.arange(0, 3.0001, 0.001)
    wd = wn * math.sqrt(1 - ze**2)
    x = 0.01 * np.exp(-ze * wn * t) * (np.cos(wd * t) + ze / math.sqrt(1 - ze**2) * np.sin(wd * t))
    w_fit, z_fit = lab.fit_free_decay(t, x)
    assert w_fit == pytest.approx(wn, rel=2e-3)
    assert z_fit == pytest.approx(ze, rel=1e-2)


def test_fit_free_decay_on_logged_data():
    m, kg = lab.handle_mass(HK), lab.gravity_stiffness(HK)
    for K in (50, 100, 150):
        wn, ze = lab.fit_free_decay(*lab.load_free_decay(K))
        assert wn == pytest.approx(math.sqrt((K + kg) / m), rel=5e-3)
        b_app = 2 * ze * wn * m
        assert b_app == pytest.approx(DEV["b_h"] - K * DEV["T"] / 2, abs=0.005)


def test_identify_device():
    Ks = (50, 100, 150)
    fits = [lab.fit_free_decay(*lab.load_free_decay(K)) for K in Ks]
    ident = lab.identify_device(Ks, fits)
    assert ident["m_h"] == pytest.approx(lab.handle_mass(HK), rel=0.01)
    assert ident["b_h"] == pytest.approx(DEV["b_h"], rel=0.03)
    assert ident["tau_eff"] == pytest.approx(DEV["T"] / 2, rel=0.1)
    assert len(ident["b_app"]) == 3 and ident["b_app"][0] > ident["b_app"][2]


# ============================================================================ Part B -- rendering
def test_spring_damper_and_wall_force():
    assert lab.spring_damper_force(0.01, 0.1, 100.0, 2.0) == pytest.approx(-1.2)
    assert lab.spring_damper_force(0.01, 0.1, 100.0, 2.0, x0=0.02, v0=0.1) == pytest.approx(1.0)
    assert lab.wall_force(-0.001, 0.5, 1000.0, 1.0, 0.0) == 0.0                     # outside
    assert lab.wall_force(0.002, 0.01, 1000.0, 1.0, 0.0) == pytest.approx(-2.01)      # pushing in
    assert lab.wall_force(0.0005, -1.0, 1000.0, 1.0, 0.0) == 0.0                      # leaving fast: no pull
    assert lab.wall_force(-0.002, 0.0, 1000.0, 1.0, 0.0, side=-1) == pytest.approx(2.0)
    assert lab.wall_force(0.002, 0.0, 1000.0, 1.0, 0.0, side=-1) == 0.0


def test_simulate_device_free_motion_is_exact():
    dev = lin_dev()
    sim = lab.simulate_device(dev, lambda t, x, v: 0.0, 0.5, x0=0.005, v0=0.02, quantize=False)
    A = np.array([[0, 1], [-dev["k_g"] / dev["m_h"], -dev["b_h"] / dev["m_h"]]])
    for k in (100, 250, 500):
        s = expm(A * sim["t"][k]) @ np.array([0.005, 0.02])
        assert sim["x"][k] == pytest.approx(s[0], abs=1e-9)
        assert sim["v"][k] == pytest.approx(s[1], abs=1e-8)


def test_simulate_device_loop_details():
    dev = lin_dev()
    seen = []

    def law(t, x, v):
        seen.append((t, x, v))
        return 5.0 if t >= 0.0095 else 0.1                 # 5 N > force_max: must saturate

    sim = lab.simulate_device(dev, law, 0.02, x0=0.00123, delay_steps=3)
    xm = sim["x_meas"]
    assert np.allclose(np.round(xm / dev["x_res"]) * dev["x_res"], xm)
    assert xm[0] == pytest.approx(round(0.00123 / dev["x_res"]) * dev["x_res"])
    assert sim["v_est"][0] == 0.0
    assert np.allclose(sim["v_est"][1:], np.diff(xm) / dev["T"])
    assert np.allclose([s[1] for s in seen], xm) and np.allclose([s[2] for s in seen], sim["v_est"])
    assert np.max(np.abs(sim["F_cmd"])) <= dev["force_max"] + 1e-12
    assert sim["saturated"][10] and not sim["saturated"][9]
    assert np.allclose(sim["F_applied"][:3], 0.0)
    assert np.allclose(sim["F_applied"][3:], sim["F_cmd"][:-3])


def test_static_push_rendered_stiffness():
    kg = DEV["k_g"]
    for K, F in ((100.0, 0.49), (300.0, -0.98)):
        r = lab.static_push_test(DEV, K, 1.0, F)
        assert r["x_ss"] == pytest.approx(F / (K + kg), abs=DEV["x_res"])
        assert r["K_rendered"] == pytest.approx(K + kg, rel=0.02)
        assert abs(r["error_pct"]) < 2.0


def _own_loop(m, b, kg, K, B, T, d, x0, n):
    """Independent implementation of the linear sampled loop (augmented-matrix ZOH)."""
    Ac = np.array([[0, 1], [-kg / m, -b / m]])
    E = expm(np.block([[Ac, np.array([[0], [1 / m]])], [np.zeros((1, 3))]]) * T)
    Ad, Bd = E[:2, :2], E[:2, 2]
    s, xp, cmds, xs = np.array([x0, 0.0]), x0, [], []
    for k in range(n):
        c = -K * s[0] - B * (s[0] - xp) / T
        cmds.append(c)
        F = cmds[k - d] if k - d >= 0 else 0.0
        xp = s[0]
        xs.append(s[0])
        s = Ad @ s + Bd * F
    return np.array(xs)


def test_closed_loop_matrix_reproduces_the_loop():
    m, b, kg, T = DEV["m_h"], DEV["b_h"], DEV["k_g"], DEV["T"]
    for d, K, B in ((0, 200.0, 0.5), (2, 150.0, 1.0), (5, 20.0, 0.2)):
        A = lab.closed_loop_matrix(m, b, kg, K, B, T, d)
        assert A.shape == (3 + d, 3 + d)
        z = np.zeros(3 + d)
        z[0] = z[2] = 0.003
        xs = []
        for _ in range(60):
            xs.append(z[0])
            z = A @ z
        assert np.allclose(xs, _own_loop(m, b, kg, K, B, T, d, 0.003, 60), atol=1e-12)
        sim = lab.simulate_device(lin_dev(), lambda t, x, v, K=K, B=B: lab.spring_damper_force(x, v, K, B),
                                  0.059, x0=0.003, delay_steps=d, quantize=False)
        assert np.allclose(sim["x"], xs, atol=1e-9)


def test_max_stable_K_and_first_order_estimate():
    m, b, kg, T = DEV["m_h"], DEV["b_h"], DEV["k_g"], DEV["T"]
    rho = lambda K, B, d: np.max(np.abs(np.linalg.eigvals(lab.closed_loop_matrix(m, b, kg, K, B, T, d))))  # noqa
    for d in (0, 1, 2, 5):
        for B in (0.0, 0.5, 1.0):
            Km = lab.max_stable_K(m, b, kg, B, T, d)
            assert rho(Km - 0.05, B, d) < 1 < rho(Km + 0.05, B, d)
            est = lab.first_order_K_limit(b, B, T, d)
            assert est == pytest.approx(Km, rel=0.05)
    assert lab.max_stable_K(m, b, kg, 0.0, T, 0) == pytest.approx(300.0, rel=0.01)   # ~ b_h / (T/2)


def test_energy_and_simulated_stability_agree_with_eigenvalues():
    m, b, kg, T = DEV["m_h"], DEV["b_h"], DEV["k_g"], DEV["T"]
    Km = lab.max_stable_K(m, b, kg, 0.5, T, 2)
    out = {}
    for f in (0.8, 1.2):
        K = f * Km
        sim = lab.simulate_device(DEV, lambda t, x, v, K=K: lab.spring_damper_force(x, v, K, 0.5), 2.0,
                                  x0=0.004, delay_steps=2)
        E = lab.device_energy(DEV, sim, K)
        assert E[0] == pytest.approx(0.5 * (K + kg) * 0.004**2, rel=1e-9)
        out[f] = (lab.oscillation_growth(sim), E[-1] / E[0])
    assert out[0.8][0] < 0.5 and out[0.8][1] < 0.05
    assert out[1.2][0] > 1.2 and out[1.2][1] > 1.0


# ============================================================================ Part C -- rehab robot
def test_task_inertia_matches_modern_robotics():
    for q in ([0.7, 1.7], [1.2, 0.9], [0.3, 2.2]):
        q = np.array(q)
        Mq = mr.MassMatrix(q, RH["Mlist"], RH["Glist"], RH["Slist"])
        Js = mr.JacobianSpace(RH["Slist"], q)
        p = mr.FKinSpace(RH["M"], RH["Slist"], q)[:3, 3]
        # linear velocity of the point p: v = v_s + w_s x p  (MR Sec. 5.1.1)
        Jv = np.array([Js[3:, i] + np.cross(Js[:3, i], p) for i in range(2)]).T[:2]
        Lam = np.linalg.inv(Jv @ np.linalg.solve(Mq, Jv.T))
        assert np.allclose(lab.task_inertia(q), Lam, atol=1e-10)


def test_apparent_mass_along_the_therapy_path():
    expected_tangential = (0.70, 0.81, 0.99)             # 461 project scenario sheet S1
    for P, mt in zip(((0, 0.35), (0, 0.45), (0, 0.55)), expected_tangential):
        q = lab.rehab_ik(P)
        assert lab.apparent_mass(q, [0, 1]) == pytest.approx(mt, abs=0.006)
        ev = np.linalg.eigvalsh(lab.task_inertia(q))
        for u in ([1, 0], [1, 1], [0.3, -1]):
            assert ev[0] - 1e-9 <= lab.apparent_mass(q, u) <= ev[1] + 1e-9
    assert lab.apparent_mass(lab.rehab_ik((0, 0.45)), [0, 2]) == pytest.approx(
        lab.apparent_mass(lab.rehab_ik((0, 0.45)), [0, 1]))


def test_impedance_control_torque_and_rendered_stiffness():
    q, dq = np.array([0.7, 1.7]), np.array([0.2, -0.4])
    pd, vd = lab.rehab_fk(q) + [0.01, -0.02], np.array([0.05, 0.0])
    K = np.array([[500.0, 50.0], [50.0, 300.0]])
    J = lab.rehab_jacobian(q)
    F = K @ (pd - lab.rehab_fk(q)) + 15.0 * (vd - J @ dq)
    assert np.allclose(lab.impedance_control_torque(q, dq, pd, vd, K, 15.0), J.T @ F)
    pm = np.array([0.0, 0.45])
    sim = lab.simulate_rehab(lambda t, q, dq, f: lab.impedance_control_torque(q, dq, pm, [0, 0], 500.0, 15.0),
                             lab.rehab_ik(pm), 1.5, f_ext=lambda t, p, v: np.array([5.0, 0.0]))
    d = sim["p"][-1] - pm
    assert 5.0 / d[0] == pytest.approx(500.0, rel=0.01) and abs(d[1]) < 1e-4


def test_admittance_control_gives_the_commanded_hand_acceleration():
    rng = np.random.default_rng(3)
    p0 = np.array([0.0, 0.45])
    Md, Bd, Kd = np.diag([1.0, 2.0]), 15.0, np.diag([500.0, 300.0])
    for _ in range(4):
        q, dq, f = rng.uniform([0.4, 1.0], [1.2, 2.2]), rng.normal(size=2) * 0.5, rng.normal(size=2) * 5
        tau = lab.admittance_control_torque(q, dq, f, p0, Md, Bd, Kd)
        T = mr.FKinSpace(RH["M"], RH["Slist"], q)
        f_b = T[:3, :3].T @ np.r_[f, 0.0]                  # hand force in {b}; F_tip = -f (MR Eq. 11.48)
        qdd = mr.ForwardDynamics(q, dq, tau, RH["g"], np.r_[0, 0, 0, -f_b], RH["Mlist"], RH["Glist"], RH["Slist"])
        a = lab.rehab_jacobian(q) @ qdd + lab.rehab_jacobian_dot(q, dq) @ dq
        p, v = lab.rehab_fk(q), lab.rehab_jacobian(q) @ dq
        assert np.allclose(a, np.linalg.solve(Md, f - Bd * v - Kd @ (p - p0)), atol=1e-8)


def test_admittance_renders_the_one_dof_impedance_and_stiffness_control_does_not():
    pm = np.array([0.0, 0.45])
    q0 = lab.rehab_ik(pm)
    fp = lab.pulse_force([1, 0], 10.0, 0.1, 0.12)
    t_ref = None
    for Md in (2.0, 0.5):
        sim = lab.simulate_rehab(lambda t, q, dq, f, Md=Md: lab.admittance_control_torque(q, dq, f, pm, Md, 15, 500),
                                 q0, 0.6, f_ext=fp)
        t_ref = sim["t"]
        x1 = lab.one_dof_pulse_response(Md, 15.0, 500.0, 10.0, 0.1, 0.12, t_ref)
        dx = sim["p"] - pm
        assert dx[:, 0].max() == pytest.approx(x1.max(), rel=0.02)
        assert np.abs(dx[:, 1]).max() < 1e-5                 # decoupled
    sim = lab.simulate_rehab(lambda t, q, dq, f: lab.impedance_control_torque(q, dq, pm, [0, 0], 500, 15),
                             q0, 0.6, f_ext=fp)
    dx = sim["p"] - pm
    assert np.abs(dx[:, 1]).max() > 0.3 * dx[:, 0].max()      # the robot's own inertia couples x and y
