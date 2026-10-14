"""Auto-checks for SBEG 46X Lab 4 (haptic rendering: virtual wall, Z-width, virtual fixtures, needle insertion).

    LAB_IMPL=starter  pytest -q test_lab.py      # untouched starter: every test that calls your code fails with
                                                 # NotImplementedError; the 3 data/infrastructure tests pass

LAB_IMPL is the name of the module to test (set it to starter, or to the name of your own copy of starter.py); it
is imported from this folder.
Ground truth: modern_robotics (MassMatrix, GravityForces) for the device model, closed-form results (Colgate's
condition b > K T/2 + |B|, the exact energy K v T d of a sampled wall pushed at constant speed, the analytic damped
response of the device), independent code written in this file, and hand values from the Week 4 workshop.
"""
import importlib
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import modern_robotics as mr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("LAB_IMPL", "starter")
lab = importlib.import_module(os.environ["LAB_IMPL"])
import course_models as cm  # noqa: E402

DEV = lab.load_device()
TIS = lab.load_tissue()
mm = 1e-3


# ============================================================================ data and infrastructure
def test_device_json_matches_the_course_hapkit_model():
    """m_h and k_g in device.json are the handle mass and gravity stiffness of the shared hapkit model."""
    hk = cm.hapkit()
    p = hk["params"]
    I_piv = mr.MassMatrix([0.0], hk["Mlist"], hk["Glist"], hk["Slist"])[0, 0]
    m_h = (I_piv + p["reflected_rotor_inertia"]) / p["rh"] ** 2
    h = 1e-6
    dg = (mr.GravityForces([h], hk["g"], hk["Mlist"], hk["Glist"], hk["Slist"])[0]
          - mr.GravityForces([-h], hk["g"], hk["Mlist"], hk["Glist"], hk["Slist"])[0]) / (2 * h)
    assert m_h == pytest.approx(DEV["m_h"], abs=1e-4)
    assert dg / p["rh"] ** 2 == pytest.approx(DEV["k_g"], abs=1e-3)
    assert DEV["force_max"] == p["force_max"] and DEV["x_range"] == p["x_range"]
    # Coulomb friction at the handle for 0.2 mN m at the motor: G tau_f / r_h
    assert p["ratio"] * 0.2e-3 / p["rh"] == pytest.approx(DEV["f_c"], abs=1e-4)


def test_data_files_load():
    tr = lab.load_csv("zwidth_trials.csv")
    assert len(tr["K"]) == 15 and np.all(np.diff(tr["K"]) > 0)
    log = lab.load_csv("needle_log.csv")
    assert set(log) == {"t", "x_meas", "f_tissue"} and len(log["t"]) > 3000
    assert set(TIS) >= {"skin", "tissue", "pleura", "v_eps", "x_skin"}


def test_simulator_free_motion_matches_analytic_response():
    """Provided simulator, no rendering: m x'' + b x' + k_g x = 0 from x0 has the exact damped solution."""
    m, b, k = DEV["m_h"], DEV["b_h"], DEV["k_g"]
    sim = lab.simulate_device(DEV, lambda t, x, v: 0.0, 0.5, x0=0.01, quantize=False)
    wn, ze = math.sqrt(k / m), b / (2 * math.sqrt(k * m))
    wd = wn * math.sqrt(1 - ze**2)
    t = sim["t"]
    x = 0.01 * np.exp(-ze * wn * t) * (np.cos(wd * t) + ze * wn / wd * np.sin(wd * t))
    assert np.max(np.abs(sim["x"] - x)) < 1e-9


# ============================================================================ Part A -- passivity limits
def test_colgate_K_max():
    assert lab.colgate_K_max(0.15, 0.0, 1e-3) == pytest.approx(300.0)
    assert lab.colgate_K_max(0.15, 0.05, 1e-3) == pytest.approx(200.0)
    assert lab.colgate_K_max(0.15, -0.05, 1e-3) == pytest.approx(200.0)   # |B|: negative damping is no better
    assert lab.colgate_K_max(0.15, 0.0, 0.25e-3) == pytest.approx(1200.0)
    assert lab.colgate_K_max(0.15, 0.15, 1e-3) == 0.0
    assert lab.colgate_K_max(0.15, 1.0, 1e-3) == 0.0                      # B > b: no passive wall at all


def test_required_damping_frequency_form():
    th = np.linspace(1e-3, math.pi, 2001)
    for K, B, T in [(300, 0.0, 1e-3), (200, 0.05, 1e-3), (500, -0.03, 0.5e-3), (0.0, 0.1, 1e-3)]:
        got = np.asarray(lab.required_damping(K, B, T, th))
        # independent: evaluate Colgate's expression with explicit complex numbers
        ref = []
        for t in th:
            z = complex(math.cos(t), math.sin(t))
            H = K + B * (1 - 1 / z) / T
            ref.append((T / 2) * ((1 - 1 / z) * H).real / (1 - math.cos(t)))
        assert np.allclose(got, ref, atol=1e-12)
        assert got.max() == pytest.approx(K * T / 2 + abs(B), rel=1e-5)
    # the damper only hurts at high frequency when B > 0: at theta = pi/2 the requirement is K T / 2
    assert float(lab.required_damping(300, 0.05, 1e-3, math.pi / 2)) == pytest.approx(0.15, abs=1e-12)


def test_quantisation_K_max():
    assert lab.quantisation_K_max(DEV["f_c"], DEV["x_res"]) == pytest.approx(3860.0)
    assert lab.quantisation_K_max(0.0386, 1e-4) == pytest.approx(772.0)


def test_sampled_wall_work_is_KvTd():
    for K, T, v, d in [(500, 1e-3, 0.05, 5e-3), (300, 1e-3, 0.02, 4e-3), (1000, 0.5e-3, 0.1, 10e-3)]:
        assert lab.sampled_wall_work(K, T, v, d) == pytest.approx(K * v * T * d, rel=1e-9)
    # a faster loop creates proportionally less energy
    w1, w2 = lab.sampled_wall_work(500, 1e-3, 0.05, 5e-3), lab.sampled_wall_work(500, 0.1e-3, 0.05, 5e-3)
    assert w2 == pytest.approx(w1 / 10, rel=1e-9)


def test_energy_balance_gives_colgate_threshold():
    """Created energy K v T d against the energy 2 b v d dissipated by the device damping in the same push."""
    b, T, v, d = DEV["b_h"], DEV["T"], 0.05, 5e-3
    for K, active in [(250, False), (350, True), (500, True)]:
        net = lab.sampled_wall_work(K, T, v, d) - 2 * b * v * d
        assert (net > 0) == active
    assert lab.sampled_wall_work(2 * b / T, T, v, d) == pytest.approx(2 * b * v * d, rel=1e-9)


# ============================================================================ Part B -- stable stiffness limit
def test_wall_force_is_unilateral():
    assert lab.wall_force(-1e-3, 0.1, 300, 0.5) == 0.0
    assert lab.wall_force(2e-3, 0.0, 300, 0.0) == pytest.approx(-0.6)
    assert lab.wall_force(2e-3, -0.1, 300, 0.5) == pytest.approx(-0.55)
    assert lab.wall_force(1e-4, -0.5, 300, 0.5) == 0.0           # damper would pull the hand in: clipped
    assert lab.wall_force(-2e-3, 0.0, 300, 0.0, x_wall=0.0, side=-1) == pytest.approx(0.6)


def test_bounce_restitution_crosses_one_near_2b_over_T():
    assert lab.bounce_restitution(DEV, 200.0) < 0.99
    assert lab.bounce_restitution(DEV, 450.0) > 1.0
    assert lab.bounce_restitution(DEV, 450.0, B=0.1) < 1.0       # virtual damping helps in a slow bounce


def test_measure_K_limit():
    k0 = lab.measure_K_limit(DEV, 0.0)
    assert 300 <= k0 <= 325                                          # Colgate: 300 N/m (single-bounce test +4 %)
    k1 = lab.measure_K_limit(DEV, 0.05)
    assert 390 <= k1 <= 450                                          # low-frequency test: ~2 (b + B)/T,
    assert k1 > 2 * lab.colgate_K_max(DEV["b_h"], 0.05, DEV["T"])   # far above the passive limit 200 N/m


def test_threshold_from_trials():
    tr = lab.load_csv("zwidth_trials.csv")
    k = lab.threshold_from_trials(tr["K"], tr["e"])
    assert k == pytest.approx(349.1, abs=0.5)
    assert lab.threshold_from_trials([100, 200, 300], [0.9, 0.95, 1.05]) == pytest.approx(250.0)
    with pytest.raises(ValueError):
        lab.threshold_from_trials([100, 200], [0.9, 0.95])
    with pytest.raises(ValueError):
        lab.threshold_from_trials([100, 200], [1.01, 1.05])


# ============================================================================ Part C -- virtual fixtures
def test_penalty_slab_pops_through():
    a, w, K = 0.0, 6 * mm, 250.0
    assert lab.penalty_slab_force(-1 * mm, a, w, K) == 0.0
    assert lab.penalty_slab_force(1 * mm, a, w, K) == pytest.approx(-0.25)
    assert lab.penalty_slab_force(4 * mm, a, w, K) == pytest.approx(+0.5)    # past the mid-plane: pulls through


def test_proxy_slab_never_pops_through():
    a, w, K = 0.0, 6 * mm, 250.0
    pr = lab.ProxySlab(a, w, K, x0=-5 * mm)
    xs = np.r_[np.linspace(-5, 9, 141), np.linspace(9, -5, 141)] * mm   # through the slab and back
    F = [pr.force(x) for x in xs]
    inside = [(a < x < a + w) for x in xs]
    assert all(f <= 1e-12 for f, ins in zip(F, inside) if ins)          # always pushes back to the entry side
    i4 = int(np.argmin(np.abs(xs - 4 * mm)))
    assert F[i4] == pytest.approx(-K * xs[i4])
    i9 = int(np.argmin(np.abs(xs - 9 * mm)))
    assert F[i9] == pytest.approx(-K * 9 * mm)                           # beyond the slab: still held at a
    assert pr.force(-3 * mm) == 0.0                                      # free again once back outside
    far = lab.ProxySlab(a, w, K, x0=10 * mm)
    assert far.force(5 * mm) == pytest.approx(K * 1 * mm)               # entered from the far side


def test_forbidden_disc_force_workshop_values():
    F, q = lab.forbidden_disc_force([3 * mm, 2 * mm], [0.0, 0.0], 5 * mm, 500.0)
    assert np.allclose(q / mm, [4.1603, 2.7735], atol=1e-4)
    assert np.allclose(F, [0.5801, 0.3868], atol=1e-4)
    F0, q0 = lab.forbidden_disc_force([6 * mm, 0.0], [0.0, 0.0], 5 * mm, 500.0)
    assert np.allclose(F0, 0.0) and np.allclose(q0, [6 * mm, 0.0])
    F1, _ = lab.forbidden_disc_force([11 * mm, 13 * mm], [10 * mm, 10 * mm], 5 * mm, 200.0)
    r = math.hypot(1, 3) * mm
    assert np.linalg.norm(F1) == pytest.approx(200 * (5 * mm - r))


def test_guidance_velocity():
    f, d = [1.0, 0.5], [0.6, 0.8]
    assert np.allclose(lab.guidance_velocity(f, d, 0.01, 0.2), [0.0068, 0.0074], atol=1e-12)
    assert np.allclose(lab.guidance_velocity(f, d, 0.01, 1.0), [0.01, 0.005])
    v0 = lab.guidance_velocity(f, [3.0, 4.0], 0.01, 0.0)                 # d need not be unit length
    assert np.allclose(v0, [0.006, 0.008])


# ============================================================================ Part D -- tissue and needle models
def test_kelvin_voigt_and_hunt_crossley():
    assert lab.kelvin_voigt_force(5 * mm, 0.02, 200, 2.0) == pytest.approx(1.04)
    assert lab.kelvin_voigt_force(1e-9, 0.02, 200, 2.0) == pytest.approx(0.04, abs=1e-6)   # jump at touch
    assert lab.kelvin_voigt_force(0.1 * mm, -0.05, 200, 2.0) == pytest.approx(-0.08)     # pulls: non-physical
    assert lab.kelvin_voigt_force(-1 * mm, 0.02, 200, 2.0) == 0.0
    k, lam, n = 3000.0, 4500.0, 1.5
    d = 5 * mm
    assert lab.hunt_crossley_force(d, 0.02, k, lam, n) == pytest.approx(d**n * (k + lam * 0.02))
    assert lab.hunt_crossley_force(1e-12, 0.02, k, lam, n) == pytest.approx(0.0, abs=1e-12)
    assert lab.hunt_crossley_force(0.1 * mm, -0.05, k, lam, n) > 0
    assert lab.hunt_crossley_force(0.1 * mm, -1.0, k, lam, n) == 0.0                     # clipped, never pulls


def test_puncture_depth():
    s = TIS["skin"]
    dp = lab.puncture_depth(s["a1"], s["a2"], s["f_p"])
    assert dp == pytest.approx(4.3007e-3, abs=1e-7)
    assert s["a1"] * dp + s["a2"] * dp**2 == pytest.approx(s["f_p"])
    assert lab.puncture_depth(150.0, 0.0, 0.45) == pytest.approx(3e-3)


def test_needle_model_values():
    m = lab.NeedleTissue(TIS)
    assert m.force(-1 * mm, 0.01) == 0.0
    assert m.force(2 * mm, 0.01) == pytest.approx(0.28)
    assert m.force(1 * mm, -0.01) == pytest.approx(0.12)                 # elastic before puncture
    assert m.force(4.31 * mm, 0.01) == pytest.approx(12 * 4.31e-3 + 0.2)  # skin punctured
    assert m.events[0][0] == "skin"
    assert m.force(10 * mm, 0.01) == pytest.approx(0.32)
    assert m.force(10 * mm, 0.0) == pytest.approx(0.0)                   # at rest: no friction, no cutting
    assert m.force(10 * mm, -0.01) == pytest.approx(-0.12)               # withdrawal: friction pulls back in
    assert m.force(26 * mm, 0.01) == pytest.approx(0.65)                 # tissue + pleural membrane
    assert m.force(28.01 * mm, 0.01) == pytest.approx(0.30)              # pleura punctured: friction only
    assert [e[0] for e in m.events] == ["skin", "pleura"]
    assert m.events[1][1] == pytest.approx(28.01 * mm)
    assert m.force(20 * mm, -0.01) == pytest.approx(-0.24)


def test_profile_slope_is_renderable():
    z = np.linspace(0, 35 * mm, 3501)
    f = lab.needle_quasistatic_profile(TIS, z)
    assert f[200] == pytest.approx(0.28)
    k_skin = lab.max_profile_slope(z[z < 4.3 * mm], f[z < 4.3 * mm])
    assert k_skin == pytest.approx(271.8, abs=0.5)
    assert k_skin < lab.colgate_K_max(DEV["b_h"], 0.0, DEV["T"])


def test_detect_punctures_in_log():
    log = lab.load_csv("needle_log.csv")
    z = log["x_meas"] - TIS["x_skin"]
    idx = lab.detect_punctures(z, log["f_tissue"], 0.3)
    assert len(idx) == 2
    assert np.allclose(z[idx], [4.3 * mm, 28.0 * mm], atol=0.1 * mm)
    assert np.allclose(log["f_tissue"][idx], [0.80, 0.95], atol=0.01)
    assert lab.detect_punctures(z, log["f_tissue"], 0.8) == []


def test_render_needle_on_device():
    sim = lab.render_needle(DEV, TIS, TIS["x_skin"], 0.01, 4.5, -0.025)
    assert [e[0] for e in sim["events"]] == ["skin", "pleura"]
    assert not sim["saturated"].any() and np.max(np.abs(sim["x"])) < DEV["x_range"]
    zq = np.linspace(0, 35 * mm, 3501)
    prof = lab.needle_quasistatic_profile(TIS, zq)
    sel = (sim["z"] > 7 * mm) & (sim["z"] < 24 * mm)
    err = sim["f_tissue"][sel] - np.interp(sim["z"][sel], zq, prof)
    assert np.sqrt(np.mean(err**2)) < 0.03
