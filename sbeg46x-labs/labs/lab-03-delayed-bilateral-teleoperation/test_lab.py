"""Auto-checks for SBEG 46X Lab 3 (delayed bilateral teleoperation: instability, wave variables, TDPA).

    LAB_IMPL=starter  python -m pytest -q test_lab.py    # untouched starter: every test that calls your code
                                                         # fails with NotImplementedError; the 3 data tests pass
    LAB_IMPL=lab3_<ID> python -m pytest -q test_lab.py   # your renamed file

LAB_IMPL is the name of the module to test; it is imported from this folder.
Ground truth: closed-form results (the integrator loop s + k e^{-s tau} has the delay margin pi/(2k), the series
spring, the wave power identity, the transmission-line impedances b coth(jwT) and b tanh(jwT)), an independent
frequency sweep for the delay margin written in this file, energy identities, the Hapkit handle mass from
course_models (modern_robotics MassMatrix), and reference runs in data/ref_*.csv (every 10th control sample).
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
lab = importlib.import_module(os.environ.get("LAB_IMPL", "starter"))
import course_models as cm  # noqa: E402

P = json.loads((HERE / "data" / "teleop.json").read_text())
EXP = json.loads((HERE / "data" / "expected.json").read_text())
RNG = np.random.default_rng(463)


def leader_model():
    return P["leader"]["m"] + P["hand"]["m"], P["leader"]["b"] + P["hand"]["b"], P["hand"]["k"]


def load_ref(name):
    with open(HERE / "data" / name) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


# ------------------------------------------------------------------------------ data and provided code (pass)
def test_params_match_course_hapkit():
    m = cm.handle_mass(cm.hapkit())
    assert abs(P["leader"]["m"] - m) < 1e-4 and abs(P["follower"]["m"] - m) < 1e-4
    assert P["force_max"] == 2.0 and P["x_range"] == 0.040


def test_delay_line_hold_and_out_of_order():
    d = np.array([3, 1, 1, -1, 1, 1, 1, 1])
    line = lab.DelayLine(d, 8, [0.0])
    got = []
    for n in range(8):
        line.send(n, [float(n)])
        got.append(line.receive(n)[0] if n > 0 else 0.0)
    # packet 0 arrives at 3 but packet 1 (newer) already arrived at 2 -> 0 is discarded; 3 is lost (held 2)
    assert got == [0.0, 0.0, 1.0, 2.0, 2.0, 4.0, 5.0, 6.0]


def test_reference_files_present():
    for f in ["ref_pp_tissue_T005.csv", "ref_pp_wall_T020.csv", "ref_pf_tissue_T020.csv",
              "ref_wave_tissue_T100.csv", "ref_tdpa_tissue_T020.csv"]:
        assert (HERE / "data" / f).exists()


# ------------------------------------------------------------------------------ Part A: delay margin
def test_contact_stiffness_series():
    assert math.isclose(lab.contact_stiffness(200.0, 300.0), 120.0)
    assert math.isclose(lab.contact_stiffness(200.0, 5000.0), 200 * 5000 / 5200)


def test_delay_margin_integrator_closed_form():
    # leader = pure damper (m = 0, b = 1, k = 0): s + kp e^{-s tau} -> w_c = kp, tau* = pi / (2 kp)
    for kp in [0.5, 2.0, 10.0]:
        tau, w = lab.delay_margin(0.0, 1.0, 0.0, kp)
        assert math.isclose(tau, math.pi / (2 * kp), rel_tol=1e-9) and math.isclose(w, kp, rel_tol=1e-9)


def _sweep_margin(m, b, k, kp):
    """Independent check: dense frequency sweep of |k - m w^2 + j b w| - kp, refined by bisection."""
    w = np.linspace(1e-3, 400, 400001)
    g = np.abs(k - m * w**2 + 1j * b * w) - kp
    best = math.inf
    for i in np.where(np.sign(g[:-1]) != np.sign(g[1:]))[0]:
        lo, hi = w[i], w[i + 1]
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            gm = abs(k - m * mid**2 + 1j * b * mid) - kp
            if np.sign(gm) == np.sign(abs(k - m * lo**2 + 1j * b * lo) - kp):
                lo = mid
            else:
                hi = mid
        wc = 0.5 * (lo + hi)
        phase = np.angle(-(k - m * wc**2 + 1j * b * wc) / kp)
        best = min(best, ((-phase) % (2 * np.pi)) / wc)
    return best


@pytest.mark.parametrize("env", ["tissue", "wall"])
def test_delay_margin_course_teleoperator(env):
    m, b, k = leader_model()
    kp = lab.contact_stiffness(P["controller"]["Kp"], P["environments"][env]["k"])
    tau, w = lab.delay_margin(m, b, k, kp)
    assert math.isclose(tau, _sweep_margin(m, b, k, kp), rel_tol=1e-6)
    assert math.isclose(tau, EXP[f"margin_{env}"]["tau_round_trip"], rel_tol=1e-9)
    s = 1j * w                                   # the characteristic equation is satisfied at s = j w_c
    assert abs(m * s**2 + b * s + k + kp * np.exp(-s * tau)) < 1e-8 * kp


def test_delay_margin_infinite_when_no_crossing():
    tau, w = lab.delay_margin(0.157, 50.0, 200.0, 120.0)        # heavy damping: |k - m w^2 + j b w| > kp
    assert math.isinf(tau) and math.isnan(w)


# ------------------------------------------------------------------------------ Part B: wave variables
def test_wave_encode_power_identity():
    for _ in range(20):
        F, xd, b = RNG.normal(), RNG.normal(), RNG.uniform(0.5, 50)
        u, v = lab.wave_encode(F, xd, b)
        assert math.isclose(0.5 * (u * u - v * v), F * xd, rel_tol=1e-9, abs_tol=1e-12)
        assert math.isclose(u, (b * xd + F) / math.sqrt(2 * b))


def test_wave_leader_and_follower_are_consistent():
    for _ in range(20):
        b = RNG.uniform(0.5, 50)
        xdot_m, v_m = RNG.normal(), RNG.normal()
        F_m, u_m = lab.wave_leader(xdot_m, v_m, b)
        assert np.allclose(lab.wave_encode(F_m, xdot_m, b), (u_m, v_m))
        u_s, F_c = RNG.normal(), RNG.normal()
        xdot_d, v_s = lab.wave_follower(u_s, F_c, b)
        assert np.allclose(lab.wave_encode(F_c, xdot_d, b), (u_s, v_s))


def test_wave_static_contact_transmits_force_exactly():
    b, F_c = 5.0, 0.8                     # resting leader and follower: u = F/sqrt(2b), v = -F/sqrt(2b)
    xdot_d, v_s = lab.wave_follower(F_c / math.sqrt(2 * b), F_c, b)
    F_m, _ = lab.wave_leader(0.0, v_s, b)
    assert abs(xdot_d) < 1e-12 and math.isclose(F_m, F_c)


def test_wave_apparent_impedance():
    b, T = 5.0, 0.1
    for w in [0.3, 3.0, 30.0]:
        x = 1j * w * T
        coth = (1 + np.exp(-2 * x)) / (1 - np.exp(-2 * x))
        assert np.isclose(lab.wave_apparent_impedance(b, T, w, "rigid"), b * coth)
        assert np.isclose(lab.wave_apparent_impedance(b, T, w, "free"), b / coth)
    w = 1e-3                                   # low frequency: spring b/T (rigid), mass b T (free)
    assert np.isclose(lab.wave_apparent_impedance(b, T, w, "rigid") * 1j * w, b / T, rtol=1e-5)
    assert np.isclose(lab.wave_apparent_impedance(b, T, w, "free") / (1j * w), b * T, rtol=1e-5)


# ------------------------------------------------------------------------------ Part C: energy and TDPA
def test_follower_pd():
    assert math.isclose(lab.follower_pd(0.012, 0.010, 0.05, 200.0, 2.0), 200 * 0.002 - 2 * 0.05)


def test_port_energies():
    h = 1e-3
    Pw = RNG.normal(size=500)
    E_in, E_out = lab.port_energies(Pw, h)
    assert np.all(np.diff(E_in) >= 0) and np.all(np.diff(E_out) >= 0)
    assert np.allclose(E_in - E_out, h * np.cumsum(Pw))
    assert np.isclose(E_in[-1], h * Pw[Pw > 0].sum())


def test_passivity_ratio():
    assert lab.passivity_ratio(-1.0, 0.0) == 1.0           # the port absorbs energy: nothing to do
    assert lab.passivity_ratio(0.2, 0.5) == 1.0            # within budget
    assert math.isclose(lab.passivity_ratio(0.5, 0.2), 0.4)
    assert lab.passivity_ratio(0.5, -0.1) == 0.0           # no budget left


def test_tdpa_leader_cases():
    h = 1e-3
    F, Ei, Eo, r = lab.tdpa_leader(1.0, 0.1, h, 0.0, 0.0, 0.0)          # input power: observer only
    assert (F, r) == (1.0, 1.0) and math.isclose(Ei, 1e-4) and Eo == 0.0
    F, Ei, Eo, r = lab.tdpa_leader(1.0, -0.1, h, 0.0, 0.0, 1.0)         # output within budget
    assert (F, r) == (1.0, 1.0) and math.isclose(Eo, 1e-4)
    F, Ei, Eo, r = lab.tdpa_leader(1.0, -0.1, h, 0.0, 2e-5, 6e-5)       # output beyond budget (4e-5)
    assert math.isclose(r, 0.4) and math.isclose(F, 0.4) and math.isclose(Eo, 6e-5)


def test_tdpa_follower_cases():
    F, dx, Ei, Eo = 0.5, 2e-4, 0.0, 0.0
    dxn, Ei2, Eo2, r = lab.tdpa_follower(F, -dx, Ei, Eo, 0.0)          # energy into the channel
    assert dxn == -dx and math.isclose(Ei2, 1e-4) and r == 1.0
    dxn, Ei2, Eo2, r = lab.tdpa_follower(F, dx, 0.0, 3e-5, 8e-5)       # wants 1e-4, budget 5e-5
    assert math.isclose(r, 0.5) and math.isclose(dxn, 1e-4) and math.isclose(Eo2, 8e-5)


# ------------------------------------------------------------------------------ Part D: simulations
def _compare(sim, ref, atol_x=2e-6, atol_F=2e-3, t_max=4.0):
    idx = np.arange(0, len(sim["t"]), 10)
    sel = ref["t"] <= t_max
    for key, tol in [("x_m", atol_x), ("x_s", atol_x), ("x_d", atol_x), ("F_m", atol_F), ("F_c", atol_F)]:
        err = np.max(np.abs(sim[key][idx][sel] - ref[key][sel]))
        assert err < tol, f"{key} differs from the reference run by {err:.3g}"


def test_pp_short_delay_is_stable_and_transparent():
    r = lab.simulate("pp", 0.005, env="tissue")
    _compare(r, load_ref("ref_pp_tissue_T005.csv"))
    m = lab.performance_metrics(r)
    assert m["x_std"] < 1e-5 and abs(m["force_ratio"] - 1) < 0.01 and abs(m["F_m_mean"] - 1.125) < 0.005


def test_pp_beyond_margin_chatters_and_channel_is_active():
    T = 0.020
    assert 2 * T > EXP["margin_wall"]["tau_round_trip"]
    r = lab.simulate("pp", T, env="wall")
    _compare(r, load_ref("ref_pp_wall_T020.csv"), t_max=1.5)
    m = lab.performance_metrics(r)
    assert m["x_std"] > 3e-3 and m["E_ch_min"] < -0.1


def test_pf_beyond_margin_chatters():
    r = lab.simulate("pf", 0.020, env="tissue")
    _compare(r, load_ref("ref_pf_tissue_T020.csv"), t_max=1.5)
    assert lab.performance_metrics(r)["x_std"] > 3e-3


def test_wave_long_delay_is_passive_and_stable():
    r = lab.simulate("wave", 0.100, env="tissue")
    _compare(r, load_ref("ref_wave_tissue_T100.csv"))
    assert np.min(r["E_ch"]) > -1e-12
    m = lab.performance_metrics(r)
    assert m["x_std"] < 5e-4 and abs(m["force_ratio"] - 1) < 0.02


def test_tdpa_is_passive_and_stable():
    r = lab.simulate("tdpa", 0.020, env="tissue")
    # the passivity controller switches on thresholds, so round-off can shift a switching instant and the trajectory
    # differs between computers: no trajectory comparison here — the passivity and stability properties below are
    # the real checks (tdpa_leader / tdpa_follower are checked exactly above)
    assert np.min(r["W_m"]) > -1e-12 and np.min(r["W_s"]) > -1e-12 and np.min(r["E_ch"]) > -1e-12
    m = lab.performance_metrics(r)
    assert m["x_std"] < 5e-4 and m["drift"] < -1e-3              # stable, at the price of position drift


def test_tdpa_with_jitter_and_loss_stays_passive():
    r = lab.simulate("tdpa", 0.100, env="wall", jitter=0.02, loss=0.1)
    assert np.min(r["W_m"]) > -1e-12 and np.min(r["W_s"]) > -1e-12 and np.min(r["E_ch"]) > -1e-12
    assert not r["tripped"]


def test_performance_metrics_synthetic():
    t = np.arange(0, 4, 1e-3)
    sim = dict(t=t, F_m=np.full_like(t, 0.9), f_e=np.full_like(t, 1.0), x_m=0.01 + 1e-3 * np.sin(20 * t),
               x_s=np.full_like(t, 0.01), x_d=np.full_like(t, 0.008), E_ch=-0.01 * np.sin(t))
    m = lab.performance_metrics(sim, t_from=2.0)
    sel = t >= 2.0
    assert math.isclose(m["force_ratio"], 0.9) and math.isclose(m["F_m_mean"], 0.9)
    assert math.isclose(m["drift"], float(np.mean(0.008 - sim["x_m"][sel])))
    assert math.isclose(m["x_std"], float(np.std(sim["x_m"][sel])))
    assert math.isclose(m["x_err_rms"], float(np.sqrt(np.mean((0.01 - sim["x_m"][sel]) ** 2))))
    assert math.isclose(m["E_ch_min"], float(np.min(sim["E_ch"])))


def test_latency_budget():
    total, margin = lab.latency_budget({"encode": 5.0, "radio": 60.0, "decode": 5.0}, 100.0)
    assert total == 70.0 and margin == 30.0
