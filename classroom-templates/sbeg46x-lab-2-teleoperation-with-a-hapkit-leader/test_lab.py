"""Auto-checks for SBEG 46X Lab 2 (Week 2): teleoperation with a Hapkit leader -- motion scaling, clutching,
tremor filtering, hybrid matrices, transparency and the stability-transparency trade-off.

Usage:   LAB_IMPL=starter python -m pytest -q test_lab.py      (your work is in starter.py; always set LAB_IMPL)
Ground truth: closed forms (Butterworth magnitude, series stiffness, ideal hybrid matrix), scipy.signal
(lfilter, butter), direct solution of the leader/follower equations, independently built state matrices,
and modern_robotics (JacobianBody) for the EOD arm.
"""
import importlib
import math
import os
import sys
from pathlib import Path

import modern_robotics as mr
import numpy as np
import pytest
from scipy import signal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
lab = importlib.import_module(os.environ.get("LAB_IMPL", "starter"))
DT = 0.001


@pytest.fixture(scope="module")
def p():
    return lab.load_params()


@pytest.fixture(scope="module")
def log():
    return lab.load_log()


def wall(p, name="wall"):
    return p["raw"]["environments"][name]


# ------------------------------------------------------------------ Part A: mapping, clutch, device safety
def test_A1_effective_mass():
    h = lab.hapkit()
    pr = h["params"]
    expected = (h["I_com"][2, 2] + h["mass"] * 0.010**2 + 1.0e-6 * (pr["rs"] / pr["rp"]) ** 2) / pr["rh"] ** 2
    assert lab.hapkit_effective_mass(h) == pytest.approx(expected, rel=1e-9)
    assert lab.hapkit_effective_mass(h) == pytest.approx(0.05705, abs=5e-5)


def test_A2_A3_force_saturation_and_watchdog():
    f = np.array([-5.0, -2.0, -0.3, 0.0, 1.99, 2.5])
    assert np.allclose(lab.saturate_force(f, 2.0), [-2, -2, -0.3, 0, 1.99, 2])
    assert lab.saturate_force(3.0, 2.0) == pytest.approx(2.0)
    ok = lab.watchdog_ok(np.array([-0.041, -0.040, 0.0, 0.039, 0.0401]), 0.040)
    assert list(np.asarray(ok, bool)) == [False, True, True, True, False]


def test_A4_clutch_mapping_small_example():
    #            engaged ...      clutch (leader returns) ...  engaged again
    x_m = np.array([0.00, 0.01, 0.02, 0.02, 0.00, -0.01, -0.01, 0.00, 0.01])
    cl = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0], bool)
    x_sd = lab.map_with_clutch(x_m, cl, 0.5, x_s0=0.1)
    expected = [0.100, 0.105, 0.110, 0.110, 0.110, 0.110, 0.110, 0.115, 0.120]
    assert np.allclose(x_sd, expected, atol=1e-12)


def test_A4_clutch_mapping_log_no_jump(log):
    a = 0.25
    x_sd = lab.map_with_clutch(log["x_m"], log["clutch"], a)
    dx_m = np.abs(np.diff(log["x_m"]))
    assert np.all(np.abs(np.diff(x_sd)) <= a * dx_m + 1e-12)        # no jump at a clutch release
    assert np.all(np.diff(x_sd)[log["clutch"][1:]] == 0)             # follower holds while clutched
    # independent reference: sum of the scaled leader increments over the engaged steps
    eng = ~log["clutch"][1:] & ~log["clutch"][:-1]
    assert x_sd[-1] == pytest.approx(a * np.sum(np.diff(log["x_m"])[eng]), abs=1e-12)
    assert x_sd[-1] == pytest.approx(0.0378, abs=5e-4)


def test_A5_rate_limit():
    x = np.r_[np.zeros(5), np.ones(200) * 0.01]
    y = lab.rate_limit(x, 0.1, DT)                                    # 0.1 mm per sample
    assert np.max(np.abs(np.diff(y))) <= 0.1 * DT + 1e-15
    assert y[4] == 0 and y[5] == pytest.approx(1e-4) and y[104] == pytest.approx(0.01)
    assert np.allclose(lab.rate_limit(x, 100.0, DT), x)


def test_A6_clutch_strokes():
    assert lab.clutch_strokes(0.040, 0.25, 0.060) == 3
    assert lab.clutch_strokes(0.030, 0.25, 0.060) == 2                # exact multiple: no extra stroke
    assert lab.clutch_strokes(0.60, 4.0, 0.070) == 3


# ------------------------------------------------------------------ Part B: tremor filtering
def test_B1_lowpass1_matches_scipy():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(500)) * 1e-4 + 0.01
    fc = 2.0
    a = 1 - math.exp(-2 * math.pi * fc * DT)
    ref, _ = signal.lfilter([a], [1, a - 1], x, zi=[x[0] * (1 - a)])
    assert np.allclose(lab.lowpass1(x, fc, DT), ref, atol=1e-12)


def test_B2_butter2_matches_scipy():
    for fc in (1.0, 2.0, 2.857):
        b, a = lab.butter2(fc, DT)
        bs, as_ = signal.butter(2, fc, fs=1 / DT)
        assert np.allclose(b, bs, rtol=1e-9, atol=1e-15) and np.allclose(a, as_, rtol=1e-9)


def test_B3_iir_filter_matches_scipy():
    rng = np.random.default_rng(1)
    x = -0.03 + np.cumsum(rng.standard_normal(800)) * 1e-4
    b, a = signal.butter(2, 3.0, fs=1 / DT)
    ref, _ = signal.lfilter(b, a, x, zi=signal.lfilter_zi(b, a) * x[0])
    assert np.allclose(lab.iir_filter(b, a, x), ref, atol=1e-12)


def test_B4_band_rms_of_a_sine_on_a_ramp():
    t = np.arange(0, 4, DT)
    x = 0.02 * t + 0.0003 * np.sin(2 * np.pi * 9 * t)                 # ramp + 0.3 mm tremor
    assert lab.band_rms(x, DT, 8, 12) == pytest.approx(0.0003 / math.sqrt(2), rel=0.02)
    assert lab.band_rms(x, DT, 20, 40) < 2e-6


def test_B4_tremor_attenuation_on_log(log):
    x = log["x_m"]
    r0 = lab.band_rms(x, DT, 8, 12)
    r1 = lab.band_rms(lab.lowpass1(x, 2.0, DT), DT, 8, 12) / r0
    b, a = lab.butter2(2.0, DT)
    r2 = lab.band_rms(lab.iir_filter(b, a, x), DT, 8, 12) / r0
    assert r0 == pytest.approx(0.0003 / math.sqrt(2), rel=0.03)
    assert r1 == pytest.approx(1 / math.sqrt(1 + (9 / 2.0) ** 2), abs=0.005)   # 0.217
    assert r2 == pytest.approx(1 / math.sqrt(1 + (9 / 2.0) ** 4), abs=0.003)   # 0.049


def test_B5_lag_estimate():
    t = np.arange(0, 3, DT)
    x = np.sin(2 * np.pi * 0.5 * t) + 0.3 * np.sin(2 * np.pi * 0.2 * t)
    y = np.r_[np.full(37, x[0]), x[:-37]]
    assert lab.lag_estimate(x, y, DT, 0.2) == pytest.approx(0.037, abs=1.1e-3)


# ------------------------------------------------------------------ Part C: two-port models
def _solve_two_port(arch, s, p, v_h, f_e):
    """Solve the leader and follower equations directly for (f_h, v_e) given (v_h, f_e)."""
    Zm, Zs = p["m_m"] * s + p["b_m"], p["m_s"] * s + p["b_s"]
    C = p["bc"] + p["kc"] / s
    al, be = p["alpha"], p["beta"]
    v_e = (al * C * v_h - f_e) / (Zs + C)                 # follower: Zs v_e = C(al v_h - v_e) - f_e
    a_s = C * (al * v_h - v_e)
    a_m = -be * a_s if arch == "pp" else -be * f_e
    f_h = Zm * v_h - a_m                                   # leader: Zm v_h = f_h + a_m
    return f_h, v_e


@pytest.mark.parametrize("arch", ["pp", "pf"])
def test_C1_hybrid_matrix(arch, p):
    q = dict(p, alpha=0.25, beta=3.0)
    s = 1j * 2 * np.pi * 2.0
    H = np.asarray(lab.hybrid_matrix(arch, s, q))
    for v_h, f_e in [(1.0, 0.0), (0.0, 1.0), (0.3, -0.7)]:
        f_h, v_e = _solve_two_port(arch, s, q, v_h, f_e)
        assert H @ np.array([v_h, f_e]) == pytest.approx(np.array([f_h, -v_e]), rel=1e-10, abs=1e-12)
    Hs = np.asarray(lab.hybrid_matrix(arch, 1j * np.array([1.0, 10.0, 100.0]), q))
    assert Hs.shape == (3, 2, 2)


def test_C2_transmitted_impedance(p):
    H_ideal = np.array([[0, 2.0], [-0.25, 0]], complex)
    assert lab.transmitted_impedance(H_ideal, 300.0 + 5j) == pytest.approx(0.5 * (300.0 + 5j))
    # quasi-static contact with a stiff wall: both architectures feel alpha*beta*(series stiffness)
    s = 1j * 1e-3
    ke = wall(p)["k"]
    kser = p["kc"] * ke / (p["kc"] + ke)
    for arch in ("pp", "pf"):
        Zt = lab.transmitted_impedance(np.asarray(lab.hybrid_matrix(arch, s, p)), ke / s)
        assert (Zt * s).real == pytest.approx(p["alpha"] * p["beta"] * kser, rel=1e-3)
    # free space (Z_e = 0): P-F feels only the leader, Z_t = h11 = Z_m
    s = 1j * 20.0
    Zt = lab.transmitted_impedance(np.asarray(lab.hybrid_matrix("pf", s, p)), 0.0)
    assert Zt == pytest.approx(p["m_m"] * s + p["b_m"])


def test_C3_llewellyn(p):
    w = np.logspace(-1, 3, 400)
    eta_pp = lab.llewellyn(np.asarray(lab.hybrid_matrix("pp", 1j * w, dict(p, alpha=1.0, beta=1.0))))
    assert np.min(eta_pp) >= -1e-9                       # symmetric P-P is a passive network
    eta_pf = lab.llewellyn(np.asarray(lab.hybrid_matrix("pf", 1j * w, p)))
    assert np.min(eta_pf) < -0.1                         # P-F with b_m = 0.15 N s/m is not absolutely stable
    H = np.array([[2 + 1j, 0.5], [-0.5 + 0.2j, 1 - 1j]])
    g = H[0, 1] * H[1, 0]
    assert lab.llewellyn(H) == pytest.approx(2 * 2 * 1 - g.real - abs(g))


# ------------------------------------------------------------------ Part D: bilateral control in time
def test_D1_control_forces(p):
    q = dict(p, alpha=0.5, beta=2.0)
    a_m, a_s = lab.control_forces("pp", 0.01, 0.1, 0.004, 0.02, 0.3, q)
    a_s_ref = q["kc"] * (0.005 - 0.004) + q["bc"] * (0.05 - 0.02)
    assert a_s == pytest.approx(a_s_ref) and a_m == pytest.approx(-2.0 * a_s_ref)
    a_m, a_s = lab.control_forces("pf", 0.01, 0.1, 0.004, 0.02, 0.3, q)
    assert a_s == pytest.approx(a_s_ref) and a_m == pytest.approx(-0.6)


def _A_ref(arch, p, ke, be):
    al, bt, kc, bc = p["alpha"], p["beta"], p["kc"], p["bc"]
    mL, ms = p["m_m"] + p["m_h"], p["m_s"]
    kh, bh = p["k_h"], p["b_h"] + p["b_m"]
    if arch == "pp":
        r1 = [-(kh + bt * kc * al), -(bh + bt * bc * al), bt * kc, bt * bc]
    else:
        r1 = [-kh, -bh, -bt * ke, -bt * be]
    return np.array([[0, 1, 0, 0], np.array(r1) / mL, [0, 0, 0, 1],
                     [kc * al / ms, bc * al / ms, -(kc + ke) / ms, -(bc + p["b_s"] + be) / ms]])


@pytest.mark.parametrize("arch", ["pp", "pf"])
def test_D2_closed_loop_matrix(arch, p):
    q = dict(p, beta=1.7)
    e = wall(p)
    assert np.allclose(lab.closed_loop_matrix(arch, q, e["k"], e["b"]), _A_ref(arch, q, e["k"], e["b"]))


def test_D3_max_stable_beta(p):
    e = wall(p)
    bmax = lab.max_stable_beta("pf", p, e["k"], e["b"])
    lam = lambda b: np.max(np.linalg.eigvals(_A_ref("pf", dict(p, beta=b), e["k"], e["b"])).real)
    assert lam(0.995 * bmax) < 0 < lam(1.005 * bmax)
    assert bmax == pytest.approx(8.155, abs=0.01)
    assert lab.max_stable_beta("pp", p, e["k"], e["b"]) == math.inf


def test_D_steady_contact_force(p):
    """Static balance with the stiff wall: hand spring = beta f_e, f_e = series stiffness * (alpha x_m - x_w)."""
    rt, e = p["raw"]["reach_task"], wall(p)
    kser = p["kc"] * e["k"] / (p["kc"] + e["k"])
    x_m = (p["k_h"] * rt["x_goal"] + p["beta"] * kser * e["x_wall"]) / (p["k_h"] + p["beta"] * kser * p["alpha"])
    fe_ref = kser * (p["alpha"] * x_m - e["x_wall"])
    t = np.arange(0, rt["t_end"] + 1e-9, DT)
    for arch in ("pp", "pf"):
        r = lab.simulate_teleop(arch, p, e, lab.reach_intent(p, t), control=lab.control_forces)
        assert r["f_e"][-1] == pytest.approx(fe_ref, rel=1e-4)          # 0.649 N
        assert r["x_m"][-1] == pytest.approx(x_m, rel=1e-4)
        assert not r["tripped"]


def test_D_free_space_transparency(p):
    """In free space the P-F operator feels no motor force; P-P feels the follower's drag (small here)."""
    t = np.arange(0, 1.5, DT)
    xi = lab.reach_intent(p, t)
    rpf = lab.simulate_teleop("pf", p, wall(p, "free"), xi, control=lab.control_forces)
    rpp = lab.simulate_teleop("pp", p, wall(p, "free"), xi, control=lab.control_forces)
    assert np.max(np.abs(rpf["a_m"])) == 0.0
    assert 1e-3 < np.max(np.abs(rpp["a_m"])) < 0.05


def test_D_simulated_stability_trade_off(p):
    e = wall(p)
    bmax = lab.max_stable_beta("pf", p, e["k"], e["b"])
    t = np.arange(0, 3.0, DT)
    xi = lab.minjerk(t, 0.0, 2.0, 0.0, 0.020)
    ok = lab.simulate_teleop("pf", dict(p, beta=0.5 * bmax), e, xi, saturate=False, watchdog=False,
                             control=lab.control_forces)
    assert np.ptp(ok["f_e"][-500:]) < 1e-6                              # settles on the wall
    bad = lab.simulate_teleop("pf", dict(p, beta=1.2 * bmax), e, xi, saturate=False, watchdog=False,
                              control=lab.control_forces)
    assert np.ptp(bad["f_e"][-500:]) > 10 * max(np.ptp(bad["f_e"][-1000:-500]), 1e-9) or \
        np.ptp(bad["f_e"][-500:]) > 1e3                                 # grows without bound
    sat = lab.simulate_teleop("pf", dict(p, beta=1.2 * bmax), e, xi, control=lab.control_forces)
    assert np.max(np.abs(sat["a_m"])) <= p["f_max"] + 1e-12            # the 2 N limit holds
    assert np.ptp(sat["f_e"][-500:]) > 0.01                             # but the operator feels chatter


# ------------------------------------------------------------------ Part E: EOD follower speed limit
def test_E1_eod_approach_rates():
    m = lab.eod_6r_ugv()
    q = m["params"]["q_ready"]
    Jb = mr.JacobianBody(m["Blist"], q)
    qd = lab.eod_approach_rates(m, q)
    assert np.allclose(Jb @ qd, [0, 0, 0, 1, 0, 0], atol=1e-10)
    assert np.allclose(qd, [0, -1.3100, 2.0195, 0, -0.7095, 0], atol=1e-4)


def test_E2_max_approach_speed():
    m = lab.eod_6r_ugv()
    q = m["params"]["q_ready"]
    assert lab.max_approach_speed(m, q, 1.0) == pytest.approx(math.radians(30) / 2.019465, rel=1e-5)
    assert lab.max_approach_speed(m, q, 0.5) == pytest.approx(0.12964, abs=1e-5)
