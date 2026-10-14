"""Auto-checks for SBEG 461 Hardware rotation 1 (Week 8): joint control on the course arm -- pre-flight and
dry run, the step response of the servo's PD loop, gravity check, gravity compensation and the station loop,
sim-vs-hardware comparison and the safe stop.

Usage:   LAB_IMPL=starter  python -m pytest -q test_lab.py     (your work, file starter.py; always set LAB_IMPL)
Ground truth: closed forms (second-order step response, Astrom-Murray Table 6.1, MR Eqs. 11.26-11.27),
modern_robotics (RNEA), the Lab 5 PID code, synthetic signals with known answers, and the synthetic station
logs in data/ (made by the course team; the reference values are quoted in the lab handout checkpoints).
"""
import importlib
import json
import math
import os
import sys
from pathlib import Path

import modern_robotics as mr
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
lab = importlib.import_module(os.environ.get("LAB_IMPL", "starter"))
ref = lab                                       # provided helpers (logs, simulators) are in starter.py too


@pytest.fixture(scope="module")
def task():
    return ref.load_task()


@pytest.fixture(scope="module")
def model(task):
    return ref.course_arm_model(task["payload_kg"])


@pytest.fixture(scope="module")
def pose(task):
    return np.array(task["tray_pose"], float)


@pytest.fixture(scope="module")
def nominal(model, pose, task):
    return ref.nominal_servo(model, pose, task)


@pytest.fixture(scope="module")
def steps(task):
    L = ref.load_log(ref.DATA / "step_log.csv")
    out = []
    for j, (a, b) in zip(task["step_test"]["joints"], task["step_test"]["windows_s"]):
        W = ref.log_window(L, a, b)
        out.append((j - 1, W["t"], W["cmd"][:, j - 1], W["meas"][:, j - 1], W["meas"][0]))
    return out


@pytest.fixture(scope="module")
def static(task):
    return ref.average_approaches(ref.load_log(ref.DATA / "static_log.csv"))


def _K(lab_, static_, model_, task_):
    gc = lab_.gravity_check(static_, model_, fit=task_["static_fit_poses"], check=task_["static_check_poses"])
    K = np.full(6, np.inf)
    K[1:4] = gc["K"]
    return K


def _identified(lab_, steps_, static_, model_, task_, nominal_):
    """The identified servo: fit + static stiffness per moving joint, nominal elsewhere."""
    K = _K(lab_, static_, model_, task_)
    fits = {}
    for j, t, c, y, q0 in steps_:
        r = lab_.identify_servo(t, c, y)
        s = lab_.servo_from_fit(r["wn"], r["zeta"], K[j])
        fits[j] = dict(kp=s["kp"], kd=s["kd"], M_eff=s["M_eff"], b=0.0, delay=r["delay"])
    return ref.make_servo(nominal_, fits), K


# ------------------------------------------------------------------ Part A: pre-flight, dry run
def test_servo_model_conversion(task, pose):
    off = task["servo_zero_offset_deg"]
    s = lab.model_to_servo(pose, off)
    assert np.allclose(s, [1.5, -36.3775, 80.7141, -137.5099, -90.9544, 0.0], atol=1e-4)
    assert np.allclose(lab.servo_to_model(s, off), pose, atol=1e-12)


def test_dry_run_step(model, pose, nominal):
    t, c = ref.step_command(pose, [2, 3], [0.5, 2.0], 3.0, 3.5)
    d = lab.dry_run(model, t, c, nominal, (1, 2), 0.25, 0.6)
    # peak speed of a second-order step response (zeta = 0.9, wn = 25): closed form 9.851 rad/s per rad
    z, wn = 0.9, 25.0
    wd = wn * math.sqrt(1 - z * z)
    ts = math.atan(math.sqrt(1 - z * z) / z) / wd
    vpk = wn / math.sqrt(1 - z * z) * math.exp(-z * wn * ts) * math.sin(wd * ts) * math.radians(3.0)
    assert abs(d["peak_qd"][1] - vpk) < 0.01 and abs(d["peak_qd"][2] - vpk) < 0.01
    assert np.allclose(d["speed_ratio"][1:3], [0.7419, 0.7402], atol=2e-3)
    assert np.allclose(d["torque_ratio"][1:3], [0.5375, 0.4584], atol=2e-3) and d["ok"]
    q0 = d["meas"][0]                                      # starts at rest on the sag: Kp (cmd - q) = g(q)
    g = mr.InverseDynamics(q0, np.zeros(6), np.zeros(6), model["g"], np.zeros(6), model["Mlist"],
                           model["Glist"], model["Slist"])
    assert np.allclose(nominal["kp"][1:3] * (pose - q0)[1:3], g[1:3], atol=1e-9)
    assert np.allclose(q0[[0, 3, 4, 5]], pose[[0, 3, 4, 5]]) and np.all(np.abs(pose - q0)[1:3] > 0.02)
    # the same step UPWARDS (against gravity) needs g + Kp A: 1.95 N m on joint 2 > 60 % of 3.0 N m
    t, c = ref.step_command(pose, [2, 3], [0.5, 2.0], -3.0, 3.5)
    d = lab.dry_run(model, t, c, nominal)
    assert not d["ok"] and np.allclose(d["torque_ratio"][1:3], [1.0817, 1.0771], atol=3e-3)


def _good_record():
    return dict(members_at_arm=["Salma", "Omar"], inducted=["Salma", "Omar", "Youssef"], supervisor="TA K.",
                estop_tested=True, estop_holder="Omar", keepout_clear=True, workspace_ready=True,
                first_run=True, limits_ok=True, margin_deg=10.0, dry_run_speed_fraction=0.25,
                dry_run_speed_ratio=0.742, dry_run_torque_ratio=0.538, current_limit_fraction=0.6,
                payload_kg=0.1, outer_ki=3.0, outer_clamp_deg=5.0)


def test_preflight():
    assert lab.preflight(_good_record()) == []
    with open(ref.DATA / "session_record_template.json") as fh:
        assert len(lab.preflight(json.load(fh))) == len(ref.PREFLIGHT_RULES)   # an empty record fails all
    cases = [("inducted", ["Salma"], 0), ("supervisor", " ", 1), ("estop_tested", False, 2),
             ("estop_holder", "Youssef", 3), ("keepout_clear", False, 4), ("workspace_ready", False, 5),
             ("limits_ok", False, 6), ("margin_deg", 5.0, 6), ("dry_run_speed_fraction", 0.5, 7),
             ("dry_run_speed_ratio", 1.05, 7), ("dry_run_torque_ratio", 1.08, 8),
             ("current_limit_fraction", 0.8, 9), ("payload_kg", 0.2, 10), ("payload_kg", None, 10),
             ("outer_ki", 8.0, 11), ("outer_clamp_deg", 10.0, 11), ("outer_ki", None, 11)]
    for key, val, rule in cases:
        r = _good_record()
        r[key] = val
        assert lab.preflight(r) == [ref.PREFLIGHT_RULES[rule]], key
    r = _good_record()
    r.update(first_run=False, dry_run_speed_fraction=0.5)   # a repeat run may be dry-run at 50 %
    assert lab.preflight(r) == []
    r = _good_record()
    r.update(payload_kg=0.125, outer_ki=0.0, outer_clamp_deg=0.0)   # A4 limit itself; no outer loop
    assert lab.preflight(r) == []


def test_joint_limit_margin_is_checklist_value(model):
    """Rule A1 (brief Sec. 5.2, shared checklist): software limits >= 10 deg inside the end stops (<= +-155 deg, J6 <= +-165 deg)."""
    import inspect
    m = ref._ca.COURSE_ARM_SAFETY["soft_limit_margin_deg"]
    assert m == 10.0
    assert inspect.signature(ref.check_command).parameters["margin_deg"].default == m
    assert np.allclose(np.rad2deg(model["limits"][:, 1]) - m, [155] * 5 + [165])
    t = np.arange(0, 1.02, 0.02)
    for deg, ok in ((150.0, True), (157.0, False)):      # 157 deg: inside a 5 deg margin, outside 10 deg
        q = np.zeros((len(t), 6))
        q[:, 0] = np.deg2rad(deg)
        assert ref.check_command(t, q, model, 0.25)["within_limits"] is ok, deg


# ------------------------------------------------------------------ Part B: the servo's step response
def test_zeta_wn_from_step_closed_form():
    zeta, wn = 0.6, 20.0
    Mp = math.exp(-math.pi * zeta / math.sqrt(1 - zeta**2))
    tp = math.pi / (wn * math.sqrt(1 - zeta**2))
    z, w = lab.zeta_wn_from_step(Mp, tp)
    assert math.isclose(z, 0.6, abs_tol=1e-12) and math.isclose(w, 20.0, abs_tol=1e-9)
    assert math.isclose(lab.zeta_wn_from_step(0.16, 0.2)[0], 0.5038, abs_tol=1e-4)   # A-M Table 6.1: 16 % <-> 0.5


def test_step_features_synthetic():
    t = np.arange(0, 2.0, 0.001)
    c = np.where(t >= 0.5, 0.05, 0.0)
    y = 0.05 * ref.step_response(t - 0.53, 20.0, 0.6)
    f = lab.step_features(t, c, y)
    assert math.isclose(f["t_step"], 0.5, abs_tol=1e-9) and math.isclose(f["A"], 0.05)
    assert math.isclose(f["overshoot"], math.exp(-0.6 * math.pi / 0.8), abs_tol=2e-4)
    assert math.isclose(f["t_peak_abs"] - 0.53, math.pi / 16, abs_tol=2e-3)
    assert 0.53 < f["t_move"] < 0.53 + 0.06


def test_identify_servo_noise_free():
    t = np.arange(0, 1.5, 0.02)
    c = np.where(t >= 0.4, 0.06, 0.0)
    y = 0.01 + 0.06 * ref.step_response(t - 0.4 - 0.04, 18.0, 0.5)
    r = lab.identify_servo(t, c, y)
    assert abs(r["wn"] - 18.0) < 0.05 and abs(r["zeta"] - 0.5) < 0.005 and abs(r["delay"] - 0.04) < 0.001
    assert abs(r["y0"] - 0.01) < 1e-6 and abs(r["A_fit"] - 0.06) < 1e-4 and r["rms"] < 1e-5


def test_identify_servo_station_log(steps):
    expected = {1: (21.36, 0.669, 0.0311), 2: (20.90, 0.658, 0.0283)}
    for j, t, c, y, _ in steps:
        r = lab.identify_servo(t, c, y)
        wn, z, d = expected[j]
        assert abs(r["wn"] - wn) < 0.3 and abs(r["zeta"] - z) < 0.01 and abs(r["delay"] - d) < 0.002, j
        assert r["rms"] < 1e-3


# ------------------------------------------------------------------ Part C: static poses and the PD gains
def test_torque_constant():
    i = np.array([0.2, 0.4, 0.6])
    assert math.isclose(lab.torque_constant(i, 1.25 * i), 1.25)
    assert math.isclose(lab.torque_constant([1.0, 2.0], [1.0, 3.0]), 7.0 / 5.0)


def test_loop_stiffness():
    tau = np.array([-0.9, -0.6, -0.3])
    assert math.isclose(lab.loop_stiffness(tau / 30.0, tau), 30.0)
    assert math.isclose(lab.loop_stiffness([0.01, 0.03], [0.1, 0.2]), 0.05 / 0.007)


def test_gravity_check_station_log(task, model, static):
    gc = lab.gravity_check(static, model, fit=task["static_fit_poses"], check=task["static_check_poses"])
    g = np.array([mr.InverseDynamics(q, np.zeros(6), np.zeros(6), model["g"], np.zeros(6), model["Mlist"],
                                     model["Glist"], model["Slist"]) for q in static["meas"]])
    assert np.allclose(gc["g"], g, atol=1e-10)
    assert np.allclose(gc["k"], [1.1823, 1.1750, 0.4443], atol=2e-3)
    assert np.allclose(gc["K"], [14.214, 7.460, 1.976], atol=0.03)
    assert np.all(gc["nrms"] < 0.05)


def test_servo_from_fit():
    M, b, zeta, wn = 0.035, 0.02, 0.6, 20.0
    kp, kd = ref.pd_gains(M, b, zeta, wn)                   # Lab 5 forwards ...
    s = lab.servo_from_fit(wn, zeta, kp, b, M_link=0.015)    # ... and backwards
    assert math.isclose(s["kp"], kp) and math.isclose(s["kd"], kd, abs_tol=1e-12)
    assert math.isclose(s["M_eff"], M, abs_tol=1e-12) and math.isclose(s["armature"], 0.020, abs_tol=1e-12)
    assert lab.servo_from_fit(wn, zeta, kp)["armature"] is None
    # station log, joint 2: K = 14.21 N m/rad, wn = 21.36 rad/s -> M_eff = 0.0312 kg m^2, about twice the link M22
    s = lab.servo_from_fit(21.357, 0.669, 14.214, 0.0, M_link=0.01546)
    assert abs(s["M_eff"] - 0.03116) < 2e-4 and abs(s["kd"] - 0.8905) < 3e-3 and abs(s["armature"] - 0.0157) < 3e-4


def test_predict_step(model, pose, steps, static, task, nominal):
    # (a) noise-free: the Lab 5 single-joint prediction equals the station-loop simulator (same PD, same delay)
    sv = dict(nominal)
    t, c = ref.step_command(pose, [2], [0.3], 3.0, 1.2)
    arm = ref.ServoArm(model, sv, joints=(1,), enc_res=0.0).run(t, c)
    pl = ref.joint_plant(model, pose, 1, sv["M_eff"][1], 0.0, None)
    p = lab.predict_step(pl, sv["kp"][1], sv["kd"][1], t, c[:, 1], sv["delay"])
    assert np.max(np.abs(p - arm["meas"][:, 1])) < 5e-4
    q0 = c[0, 1] - ref.pd_gravity_error(pl, sv["kp"][1], c[0, 1])
    assert math.isclose(p[0], q0, abs_tol=1e-12)
    # (b) station log: identified PD loop (other joints locked at their MEASURED angles) vs nominal
    K = _K(lab, static, model, task)
    for j, t, c, y, q_meas0 in steps:
        r = lab.identify_servo(t, c, y)
        s = lab.servo_from_fit(r["wn"], r["zeta"], K[j])
        lim = 0.6 * model["tau_max"][j]
        pi = lab.predict_step(ref.joint_plant(model, q_meas0, j, s["M_eff"], 0.0, lim), s["kp"], s["kd"], t, c,
                              r["delay"])
        pn = lab.predict_step(ref.joint_plant(model, q_meas0, j, nominal["M_eff"][j], 0.0, lim),
                              nominal["kp"][j], nominal["kd"][j], t, c, nominal["delay"])
        rms_i, rms_n = (float(np.sqrt(np.mean((x - y) ** 2))) for x in (pi, pn))
        assert rms_i < 1.2e-3 and rms_n > 0.01, (j, rms_i, rms_n)


# ------------------------------------------------------------------ Part D: compensation, station loop, logs
def test_gravity_comp_command(model, pose, nominal):
    K = np.array([1.0, 14.214, 7.460, 1.976, 1.0, 1.0])
    q = lab.gravity_comp_command(model, pose, K)
    g = ref.gravity_torques(model, pose)
    assert np.allclose(q - pose, [0, g[1] / K[1], g[2] / K[2], 0, 0, 0], atol=1e-12)
    assert np.allclose(np.rad2deg(q - pose)[1:3], [-3.2075, -2.7598], atol=1e-3)
    # with the exact stiffness, the feedforward removes the sag of a PD servo exactly (no encoder, no delay issue)
    arm = ref.ServoArm(model, nominal, enc_res=0.0)
    arm.reset(pose)
    r = lab.station_hold(arm, pose, 101, "ff", model, nominal["kp"])
    assert np.max(np.abs(r["meas"][-1] - pose)) < 1e-6
    arm.reset(pose)
    r0 = lab.station_hold(arm, pose, 101, "none", model, nominal["kp"])
    assert np.allclose(r0["cmd"], pose) and np.max(np.abs(r0["meas"][-1] - pose)) > 0.02


def test_station_hold_replays_log(task, model, pose, static):
    """Your station loop, fed the logged measurements, must send exactly the logged commands."""
    K = _K(lab, static, model, task)
    ht = task["hold_test"]
    H = ref.load_log(ref.DATA / "hold_log.csv")
    for mi, mode in enumerate(ht["modes"]):
        W = ref.log_window(H, 0.0, ht["duration_s"], mode=mi)
        r = lab.station_hold(ref.ReplayIO(W), pose, len(W["t"]), mode, model, K, ht["ki"],
                             np.deg2rad(ht["clamp_deg"]), ht["t_on_s"], ht["t_ramp_s"])
        assert np.allclose(r["t"], W["t"], atol=1e-9) and np.allclose(r["cmd"], W["cmd"], atol=1e-6), mode
        assert np.allclose(r["cur"], W["cur"])
    with pytest.raises(ValueError):
        lab.station_hold(ref.ReplayIO(W), pose, 5, "pid", model, K)


def test_station_hold_modes(task, model, pose, steps, static, nominal):
    ht = task["hold_test"]
    H = ref.load_log(ref.DATA / "hold_log.csv")
    idv, K = _identified(lab, steps, static, model, task, nominal)
    expected = {"none": (-3.1763, -2.4910), "ff": (-0.0122, -0.0301), "int": (0.0047, 0.0004),
                "ff+int": (0.0047, 0.0004)}
    for mi, mode in enumerate(ht["modes"]):
        W = ref.log_window(H, 0.0, ht["duration_s"], mode=mi)
        last = W["t"] >= ht["duration_s"] - 0.5 - 1e-9
        err = np.rad2deg(pose - W["meas"])[last].mean(axis=0)[1:3]       # logged steady error (deg)
        assert np.allclose(err, expected[mode], atol=0.01), mode
        arm = ref.ServoArm(model, idv)                                   # simulation fallback of the same test
        arm.reset(pose)
        r = lab.station_hold(arm, pose, len(W["t"]), mode, model, K, ht["ki"], np.deg2rad(ht["clamp_deg"]),
                             ht["t_on_s"], ht["t_ramp_s"])
        cr = lab.compare_runs(W["t"], W["cmd"], W["meas"], r["meas"])
        assert np.all(cr["rms_sim_meas"][1:3] < 2e-3), mode
        assert np.max(np.abs(r["cmd"] - pose)) < np.deg2rad(ht["clamp_deg"]) + 1e-9


def test_compare_runs_synthetic():
    t = np.linspace(0, 2, 101)
    c = np.column_stack([np.sin(t), np.cos(t)])
    out = lab.compare_runs(t, c, c + [0.01, -0.02], c + [0.01, 0.0])
    assert np.allclose(out["rms_cmd_meas"], [0.01, 0.02]) and np.allclose(out["max_cmd_meas"], [0.01, 0.02])
    assert np.allclose(out["rms_sim_meas"], [0.0, 0.02], atol=1e-12)
    lagged = np.column_stack([np.sin(t - 0.08), np.cos(t - 0.08)])
    assert math.isclose(lab.compare_runs(t, c, lagged, c)["lag"], 0.08, abs_tol=1e-9)


def test_move_log_identified_beats_nominal(task, model, steps, static, nominal):
    M = ref.load_log(ref.DATA / "move_log.csv")
    idv, _ = _identified(lab, steps, static, model, task, nominal)
    nom = lab.compare_runs(M["t"], M["cmd"], M["meas"], ref.ServoArm(model, nominal).run(M["t"], M["cmd"])["meas"])
    res = ref.ServoArm(model, idv).run(M["t"], M["cmd"])
    idm = lab.compare_runs(M["t"], M["cmd"], M["meas"], res["meas"])
    assert math.isclose(nom["lag"], 0.10, abs_tol=1e-9)
    assert np.allclose(nom["rms_cmd_meas"][1:3], [0.0551, 0.0486], atol=5e-4)
    assert np.allclose(nom["rms_sim_meas"][1:3], [0.0182, 0.0156], atol=5e-4)
    assert np.all(idm["rms_sim_meas"][1:3] < 1.5e-3) and np.all(idm["rms_sim_meas"][1:3] < nom["rms_sim_meas"][1:3] / 10)
    gc = lab.gravity_check(static, model, fit=task["static_fit_poses"], check=task["static_check_poses"])
    for col, j in ((0, 1), (1, 2)):                     # predicted servo torque vs k * current during the move
        assert ref.normalised_rms(res["tau"][:, j], gc["k"][col] * M["cur"][:, j]) < 0.05


def test_stop_time(task):
    t = np.arange(0, 2, 0.02)
    q = np.where(t < 1.0, 0.5 * t, 0.5)[:, None]          # moves at 0.5 rad/s, stops dead at t = 1
    # window 0.1 s -> +-2 samples: the last centre sample that still sees motion is t = 1.02 s
    assert math.isclose(lab.stop_time(t, q, 0.9), 0.14, abs_tol=1e-9)
    P = ref.load_log(ref.DATA / "stop_log.csv")
    ts = lab.stop_time(P["t"], P["meas"], task["stop_test"]["t_trigger_s"])
    assert math.isclose(ts, 0.18, abs_tol=1e-9) and ts < task["stop_test"]["pass_s"]
    assert lab.stop_time(t, (0.5 * t)[:, None], 0.5) == math.inf
