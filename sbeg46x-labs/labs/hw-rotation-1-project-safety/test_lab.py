"""Auto-checks for SBEG 46X Hardware rotation 1 (Week 8): pre-flight record, dry run, contact equilibrium,
free-decay identification, watchdog check, trial metrics and channel energy, simulation versus hardware,
safe stop and the IMU at rest.

Usage:   LAB_IMPL=starter python -m pytest -q test_lab.py      (your work is in starter.py; always set LAB_IMPL)
Ground truth: closed forms (static equilibrium solved as a linear system, a sampled damped cosine, ramps with known
integrals), the course models in course_models.py, synthetic signals with known answers, and the synthetic station
logs in data/ with the course team's reference results in data/expected.json (quoted in the handout checkpoints).
"""
import copy
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

with open(HERE / "data" / "expected.json") as fh:
    EXP = json.load(fh)


@pytest.fixture(scope="module")
def task():
    return lab.load_task()


def good_record() -> dict:
    """A complete haptic-station record that satisfies every rule."""
    r = lab.load_record()
    r.update(date="2026-11-04", team="T07", scenario=1, units={"leader": "H07", "follower": "H08", "imu": "IMU-03"},
             members=[{"name": n, "induction_signed": True, "touches_hardware": True} for n in ("Salma", "Omar", "Mona")],
             supervisor="TA Hany", supervisor_present=True, estop_tested=True, stop_holder="Omar",
             leader_operator="Salma", hands_on_handle=["operator"])
    r["dry_run"] = {"done": True, "peak_force_N": 0.32, "peak_x_m": 0.0204}
    r["devices"] = {"force_max_N": 2.0, "watchdog_enabled": True, "watchdog_range_m": 0.040, "loop_overrun_watchdog": True}
    r["workspace"] = {"cables_tidy": True, "loose_items": False, "devices_fixed": True}
    r["imu"] = {"powered_by": "USB", "voltage_V": 5.0, "over_clothing": True}
    r["logging"] = {"csv": True, "session_log": True}
    return r


def good_arm() -> dict:
    return {"first_run": True, "speed_fraction": 0.25, "soft_limit_margin_deg": 10.0, "keepout_radius_m": 0.58,
            "keepout_taped": True, "keepout_clear": True, "payload_kg": 0.10, "current_fraction": 0.60,
            "estop_tested": True, "mat": True}


def labels(v):
    return sorted({s.split(":")[0] for s in v})


# ------------------------------------------------------------------ course models (provided, numbers identical)
def test_course_models_match_task(task):
    h = cm.handle_model()
    assert h["m_h"] == pytest.approx(task["device_nominal"]["m_h"], abs=5e-6)
    assert h["k_g"] == pytest.approx(task["device_nominal"]["k_g"], abs=5e-5)
    assert cm.HAPTIC_SAFETY == {"force_max": 2.0, "x_range": 0.040}
    S = cm.COURSE_ARM_SAFETY
    assert (S["soft_limit_margin_deg"], S["speed_fraction_first_run"], S["speed_fraction"], S["payload_max_kg"],
            S["current_fraction"]) == (10.0, 0.25, 0.50, 0.125, 0.60)
    assert S["reach_m"] + S["keepout_extra_m"] == pytest.approx(0.58)


# ------------------------------------------------------------------ Part A: pre-flight and dry run
def test_preflight_good_record_passes():
    assert lab.preflight(good_record()) == []
    r = good_record()
    r["arm"] = good_arm()
    assert lab.preflight(r) == []
    r["arm"].update(first_run=False, speed_fraction=0.50)      # a repeated motion may run at 50 %
    assert lab.preflight(r) == []
    r["arm"]["payload_kg"] = 0.125                             # exactly 0.125 kg is allowed (A4)
    assert lab.preflight(r) == []


def test_preflight_blank_template_fails():
    got = labels(lab.preflight(lab.load_record()))
    for rule in ("A5", "A7", "B1", "B2", "B3", "B4", "B5", "B6", "B7"):
        assert rule in got, rule


@pytest.mark.parametrize("path,value,rule", [
    (("members", 1, "induction_signed"), False, "A5"),
    (("supervisor_present",), False, "B1"),
    (("estop_tested",), False, "A7"),
    (("dry_run", "done"), False, "B2"),
    (("dry_run", "peak_force_N"), 2.4, "B2"),
    (("devices", "force_max_N"), 3.0, "B3"),
    (("devices", "watchdog_range_m"), 0.050, "B3"),
    (("hands_on_handle",), ["operator", "TA"], "B3"),
    (("channel", "delay_s"), 0.1, "B4"),
    (("stop_holder",), "Salma", "B4"),
    (("workspace", "loose_items"), True, "B5"),
    (("imu", "voltage_V"), 12.0, "B6"),
    (("imu", "over_clothing"), False, "B6"),
    (("logging", "csv"), False, "B7"),
])
def test_preflight_each_rule(path, value, rule):
    r = good_record()
    d = r
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value
    assert labels(lab.preflight(r)) == [rule]


def test_preflight_watchdog_disabled_is_b3_and_b4():
    r = good_record()
    r["devices"]["watchdog_enabled"] = False                   # "emulating" a delay by disabling the watchdog
    assert labels(lab.preflight(r)) == ["B3", "B4"]
    r = good_record()
    r["members"] = r["members"][:1]
    r["stop_holder"] = "Salma"
    assert "B1" in labels(lab.preflight(r))                    # never alone


@pytest.mark.parametrize("key,value,rule", [
    ("soft_limit_margin_deg", 5.0, "A1"), ("keepout_radius_m", 0.40, "A2"), ("keepout_taped", False, "A2"),
    ("speed_fraction", 0.30, "A3"), ("payload_kg", 0.20, "A4"), ("keepout_clear", False, "A6"),
    ("estop_tested", False, "A7"), ("mat", False, "B5"), ("current_fraction", 0.80, "I"),
])
def test_preflight_arm_rules(key, value, rule):
    r = good_record()
    r["arm"] = good_arm()
    r["arm"][key] = value
    assert labels(lab.preflight(r)) == [rule]


def test_contact_equilibrium_matches_linear_solve(task):
    pp, env, kg = task["pp"], task["tissue"], task["device_nominal"]["k_g"]
    Kp, a, b, ke, xw = pp["Kp"], pp["alpha"], pp["beta"], env["k"], env["x_wall"]
    for x_m in (0.004, 0.0205, 0.030):
        e = lab.contact_equilibrium(task, x_m)
        # unknowns x_s, F_c:  F_c = Kp (a x_m - x_s);  F_c = ke (x_s - xw)^+ + kg x_s
        for contact in (False, True):
            k_t = kg + (ke if contact else 0.0)
            A = np.array([[Kp, 1.0], [-k_t, 1.0]])
            rhs = np.array([Kp * a * x_m, -(ke * xw if contact else 0.0)])
            xs, Fc = np.linalg.solve(A, rhs)
            if (xs > xw) == contact:
                break
        assert e["x_s"] == pytest.approx(xs, abs=1e-12)
        assert e["F_c"] == pytest.approx(Fc, abs=1e-12) and e["F_m"] == pytest.approx(b * Fc, abs=1e-12)
        assert e["delta"] == pytest.approx(max(0.0, xs - xw), abs=1e-12)
    # felt stiffness = slope of F_m(x_m) in contact
    e1, e2 = lab.contact_equilibrium(task, 0.020), lab.contact_equilibrium(task, 0.021)
    assert e1["k_felt"] == pytest.approx((e2["F_m"] - e1["F_m"]) / 0.001, rel=1e-9)
    assert e1["k_felt"] == pytest.approx(30.048, abs=1e-3)


def test_dry_run(task):
    d = lab.dry_run(task)
    s = lab.simulate_pp(task)
    assert d["peak_F_m"] == pytest.approx(np.max(np.abs(s["F_m"])), rel=1e-9)
    assert d["peak_force"] == pytest.approx(EXP["dry_run"]["peak_force"], rel=1e-6)
    assert d["peak_x"] == pytest.approx(EXP["dry_run"]["peak_x"], rel=1e-6)
    assert d["ok"] is True and d["sat"] == 0
    hard = copy.deepcopy(task)
    hard["pp"].update(alpha=1.0, beta=4.0)                     # a force scale of 4 would push the leader too hard
    assert lab.dry_run(hard)["ok"] is False


# ------------------------------------------------------------------ Part B: identification and the watchdog
def _damped(t, A=0.012, zeta=0.03, wn=40.0, c=1e-4):
    wd = wn * math.sqrt(1 - zeta**2)
    return A * np.exp(-zeta * wn * t) * np.cos(wd * t) + c


def test_log_decrement_on_a_damped_cosine():
    t = np.arange(0, 1.5, 1e-3)
    g = lab.log_decrement(t, _damped(t, c=0.0), 0.0)
    assert g["zeta"] == pytest.approx(0.03, rel=0.02) and g["wn"] == pytest.approx(40.0, rel=0.01)
    assert g["n_peaks"] == 6


def test_fit_free_decay_on_the_nominal_device(task):
    """No friction: m_h within 1 %; b_h within 10 % with the K T / 2 correction, clearly too low without it."""
    nom = task["device_nominal"]
    r = lab.simulate_free_decay(lab.device_from(nom), 100.0, 0.015, 0.5, 3.0)
    f = lab.fit_free_decay(r["t"], r["x"], 100.0, nom["k_g"], 1e-3, 0.5, 1.5)
    assert f["m_h"] == pytest.approx(nom["m_h"], rel=0.01)
    assert f["b_h"] == pytest.approx(nom["b_h"], rel=0.10)
    assert f["b_h"] - 100.0 * 1e-3 / 2 < 0.85 * nom["b_h"]
    t = np.arange(0, 1.5, 1e-3)
    f2 = lab.fit_free_decay(t, _damped(t), 0.0, 40.0**2 * 0.05, 1e-3, 0.0)   # exact cosine, K = 0
    assert f2["m_h"] == pytest.approx(0.05, rel=1e-4) and f2["zeta"] == pytest.approx(0.03, rel=1e-4)


def test_fit_free_decay_station_units(task):
    dc = task["decay"]
    for unit in ("H07", "H08"):
        L = lab.load_log(f"decay_{unit}.csv")
        f = lab.fit_free_decay(L["t"], L["meas"], dc["K"], task["device_nominal"]["k_g"], task["dt"], dc["t_release"],
                               dc["t_fit_end"])
        e = EXP[f"decay_{unit}"]["fit"]
        assert f["m_h"] == pytest.approx(e["m_h"], rel=1e-3) and f["b_h"] == pytest.approx(e["b_h"], rel=1e-2)
        g = lab.log_decrement(L["t"], L["meas"], dc["t_release"], dc["t_fit_end"])
        assert g["wn"] == pytest.approx(f["wn"], rel=0.02)          # the hand estimate agrees with the fit


def test_watchdog_check():
    t = np.arange(0, 1.0, 1e-3)
    x = 0.045 * np.sin(np.pi * t)                               # out between 0.349 s and 0.651 s
    F = -50 * x
    F[t >= 0.353] = 0.0                                         # firmware cut 4 ms after the crossing, latched
    w = lab.watchdog_check(t, x, F, 0.040)
    assert w["t_cross"] == pytest.approx(0.349, abs=1e-9) and w["delay"] == pytest.approx(0.004, abs=1e-9)
    assert w["latched"] is True
    F2 = F.copy()
    F2[t > 0.8] = -50 * x[t > 0.8]                              # output came back: not latched
    assert lab.watchdog_check(t, x, F2, 0.040)["latched"] is False
    L = lab.load_log("watchdog_log.csv")
    w = lab.watchdog_check(L["t"], L["meas"], L["cur"], 0.040)
    assert w["delay"] <= 0.010 and w["latched"] is True
    assert w["t_cross"] == pytest.approx(EXP["watchdog"]["t_cross"], abs=1e-9)


# ------------------------------------------------------------------ Part C: the teleoperation trial
def test_resample():
    t = np.array([0.0, 0.0011, 0.0019, 0.0030])
    x = 2 * t + 1
    td = np.array([-0.001, 0.0, 0.001, 0.002, 0.003, 0.004])
    y = lab.resample(t, x, td)
    assert np.isnan(y[0]) and np.isnan(y[-1])
    assert np.allclose(y[1:-1], 2 * td[1:-1] + 1)


def test_channel_energy_sign_and_value():
    t = np.linspace(0, 1, 1001)
    x_m = 0.01 * t                                              # leader moves 10 mm in 1 s
    E = lab.channel_energy(t, np.full_like(t, 0.5), x_m, np.zeros_like(t), np.zeros_like(t))
    assert E[0] == 0.0 and E[-1] == pytest.approx(0.5 * 0.01, rel=1e-9)   # resisting force: energy INTO the channel
    E2 = lab.channel_energy(t, np.zeros_like(t), x_m, np.full_like(t, 0.5), x_m)
    assert E2[-1] == pytest.approx(-0.005, rel=1e-9)            # follower drives its port: energy OUT
    E3 = lab.channel_energy(t, np.full_like(t, 0.25), x_m, np.full_like(t, 0.5), 0.5 * x_m)
    assert np.max(np.abs(E3)) < 1e-12                           # beta = alpha = 0.5: power-conserving channel


def test_trial_metrics(task):
    L = lab.load_log("trial_log.csv")
    m = lab.trial_metrics(L, task)
    e = EXP["trial"]
    for k in ("rms_free", "x_m_hold", "delta_hold", "F_m_hold", "F_m_pred", "k_felt", "peak_F_m", "peak_x", "E_min"):
        assert m[k] == pytest.approx(e[k], rel=2e-3, abs=1e-9), k
    assert m["n_sat"] == 0 and m["ok"] is True
    assert abs(m["k_felt"] / lab.contact_equilibrium(task, m["x_m_hold"])["k_felt"] - 1) < 0.03
    assert m["E_min"] >= task["limits"]["E_min"]


def test_sim_vs_hw(task):
    L = lab.load_log("trial_log.csv")
    nom = lab.sim_vs_hw(L, task, lab.device_from(task["device_nominal"]))
    assert nom["rms"] == pytest.approx(EXP["sim_vs_hw"]["nominal"]["rms"], rel=0.02) and nom["ok"] is True
    f = EXP["decay_H08"]["fit"]
    ident = dict(m_h=f["m_h"], b_h=f["b_h"], k_g=task["device_nominal"]["k_g"])
    fr = lab.sim_vs_hw(L, task, lab.device_from(ident, f_c=0.008))
    assert fr["rms"] < 0.5 * nom["rms"]                        # most of the residual is unmodelled friction


# ------------------------------------------------------------------ Part D: safe stop and the IMU
def test_fade_time():
    t = np.arange(0, 1.0, 1e-3)
    F = np.where(t < 0.5, 0.6, np.clip(0.6 * (1 - (t - 0.5) / 0.05), 0, None))
    assert lab.fade_time(t, F, 0.5, 0.02) == pytest.approx(0.049, abs=1.1e-3)
    assert lab.fade_time(t, np.full_like(t, 0.01), 0.5, 0.02) == 0.0
    assert lab.fade_time(t, np.full_like(t, 0.5), 0.5, 0.02) == math.inf
    L = lab.load_log("stop_log.csv")
    ft = lab.fade_time(L["t"], np.c_[L["force"], L["cur"]], task_stop := 2.5, 0.02)
    assert ft == pytest.approx(EXP["stop"]["fade_time"], abs=1e-6) and ft <= 0.010 and task_stop == 2.5


def test_imu_rest_stats():
    rng = np.random.default_rng(1)
    t = np.arange(0, 20, 0.005)
    g = 0.03 + 0.008 * rng.standard_normal(len(t))
    th = math.radians(10)
    r = lab.imu_rest_stats(t, g, np.full_like(t, 9.81 * math.cos(th)), np.full_like(t, -9.81 * math.sin(th)), 10.0)
    assert r["bias"] == pytest.approx(np.mean(g[t < 10]), abs=1e-12)
    assert r["noise"] == pytest.approx(0.008, rel=0.05) and r["tilt_deg"] == pytest.approx(10.0, abs=1e-9)
    assert r["drift_deg"] == pytest.approx(math.degrees(abs(r["bias"]) * 10.0), rel=1e-12)
    L = lab.load_log("imu_rest.csv")
    r = lab.imu_rest_stats(L["t"], L["aux"]["gyro"], L["aux"]["ax"], L["aux"]["ay"], 10.0)
    for k in ("bias", "noise", "drift_deg", "tilt_deg", "residual_deg"):
        assert r[k] == pytest.approx(EXP["imu"][k], rel=1e-6, abs=1e-9), k
    assert abs(r["residual_deg"]) < 0.1 * r["drift_deg"]
