"""Scenario tasks: the test conditions, the runs and the measurements behind the simulation
acceptance tests AT k-1 ... AT k-5 of ``scenario_sheets.md``. Every team, the baselines and the
markers use these functions, so all designs are measured the same way.

Each scenario has
  sK_run(design, case, ...)  -> dict of logged signals of one test run
  sK_metrics(out)            -> dict of the numbers used in the acceptance tests
  sK_acceptance(design)      -> list of (AT id, passed, measured values) for the whole scenario
where ``design`` is the team's design object (see ``team_design.py`` for the interface of each
scenario) or the baseline of ``baselines.py``. All parameters are ILLUSTRATIVE VALUES.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from . import arm, devices, metrics, registration, teleop, tissue

# =============================================================================================
# Scenario 1 -- Haptic needle-insertion teleoperation (Hapkit pair)
# =============================================================================================
S1 = dict(
    x_skin=tissue.NEEDLE_S1["x_skin"],   # follower handle position of the skin surface (m)
    x0=-0.025,                           # start: needle 5 mm above the skin (leader and follower)
    z_lim=0.036,                         # depth limit (m): 1 mm before the dura at 37 mm
    target=(0.031, 0.036),               # target space: >= 1 mm past the ligament, before the limit
    zone=(0.025, 0.035),                 # ligament zone (planned from imaging)
    v_tissue=0.005,                      # insertion speed limit in tissue (m/s), set-point
    v_zone=0.002,                        # insertion speed limit in the ligament zone (m/s)
    F_lim=1.4,                           # tissue-force limit (N)
    delays=(0.05, 0.1, 0.2),             # one-way delays of the stability test (s)
    delays_task=(0.0, 0.1, 0.2),         # one-way delays of the insertion tests (s)
    overshoot_max=1.0,                   # advance after the loss of resistance (m * 1e-3 below)
)


def s1_tissue(variant: str = "normal") -> tissue.NeedleTissue:
    """Tissue variants: 'normal'; 'no_ligament' (the loss of resistance is not felt: a failed
    detection case); 'tough' (ligament puncture force 1.5 N, above the force limit)."""
    p = copy.deepcopy(tissue.NEEDLE_S1)
    if variant == "no_ligament":
        p["ligament"]["f_p"] = 0.0
    elif variant == "tough":
        p["ligament"]["f_p"] = 1.5
    elif variant != "normal":
        raise ValueError(variant)
    return tissue.NeedleTissue(p)


def s1_env(tis: tissue.NeedleTissue):
    """Firmware rendering of the tissue on the follower: depth from the measured handle position,
    speed from a 10 Hz low-pass velocity estimate (friction rendered with a noisy speed is active)."""
    est = devices.VelocityEstimator(fc=10.0)
    xs = S1["x_skin"]
    return lambda t, x, v: tis.force(x - xs, est(x), t)


def s1_operator(case: str):
    """Operator models of the S1 tests:
    'transparency' -- intended position ramps at 5 mm/s from the start to z = 20 mm, then holds;
    'release'      -- moves to z = -3 mm (no contact) at 5 mm/s, releases the handle at t = 4 s;
    'push'         -- pushes at 10 mm/s toward x = +35 mm and never stops (worst case);
    'lor'          -- like 'push' but stops after feeling the loss of resistance (reaction 0.25 s)."""
    xs, x0 = S1["x_skin"], S1["x0"]
    if case == "transparency":
        return devices.Operator(devices.ramp_to(x0, xs + 0.020, 0.005, 0.2))
    if case == "release":
        return devices.Operator(devices.ramp_to(x0, xs - 0.003, 0.005, 0.2), release_time=4.0)
    if case == "push":
        return devices.LossOfResistanceOperator(x0, 0.035, 0.010, f_min=np.inf)
    if case == "lor":
        return devices.LossOfResistanceOperator(x0, 0.035, 0.010, f_min=1.0, x_arm=xs + 0.025)
    raise ValueError(case)


def s1_run(arch, case="push", delay=0.0, variant="normal", T_end=None, seed=0):
    """One S1 run with the follower force sensor of the scenario sheet (5 ms lag, 0.01 N noise)."""
    tis = s1_tissue(variant)
    T_end = T_end or {"transparency": 6.0, "release": 8.0}.get(case, 14.0)
    out = teleop.simulate(arch, s1_operator(case), T_end, delay=delay, env=s1_env(tis),
                          x0_m=S1["x0"], sensor=tissue.ForceSensor(seed=seed))
    out["z"] = out["x_s"] - S1["x_skin"]
    out["z_set"] = out["r"] - S1["x_skin"]
    out["f_crit"] = np.array([tis.critical_force(z) for z in out["z"]])
    out["events"] = list(tis.events)
    out["case"], out["variant"] = case, variant
    return out


def s1_metrics(out) -> dict:
    t, z = out["t"], out["z"]
    m = {}
    tr = t >= 1.0
    m["rms_force_err"] = float(metrics.rms(out["F_m"][tr] - out["f_env"][tr]))
    m["rms_pos_err"] = float(metrics.rms(out["x_m"][tr] - out["x_s"][tr]))
    m["E_min"] = float(np.nanmin(out["E_ch"]))
    m["watchdog"] = bool(out["watchdog_t"] == out["watchdog_t"])
    m["sat"] = int(out["sat_m"] + out["sat_s"])
    m["x_abs_max"] = float(max(np.abs(out["x_m"]).max(), np.abs(out["x_s"]).max()))
    m["z_max"] = float(z.max())
    m["z_end"] = float(z[-1])
    m["f_max"] = float(np.max(out["f_env"]))
    m["f_crit_max"] = float(np.max(out["f_crit"]))
    lig = [e for e in out["events"] if e[0] == "ligament"]
    m["ligament_punctured"] = bool(lig)
    m["overshoot"] = float(z[t >= lig[0][2]].max() - lig[0][1]) if lig else float("nan")
    zs_all = out["z_set"]
    vset = np.diff(zs_all) / np.diff(t)              # backward difference, attributed to the start point
    zs = zs_all[:-1]
    tis_mask = zs > 0.0
    zone_mask = (zs >= S1["zone"][0]) & (zs <= S1["zone"][1])
    m["v_set_tissue"] = metrics.max_abs(np.maximum(vset, 0.0), tis_mask)
    m["v_set_zone"] = metrics.max_abs(np.maximum(vset, 0.0), zone_mask)
    return m


def s1_acceptance(make_arch) -> list:
    """All S1 acceptance tests for an architecture factory ``make_arch()`` (a fresh object per run)."""
    res = []
    m = s1_metrics(s1_run(make_arch(), "transparency", 0.0))
    res.append(("AT1-1", m["rms_force_err"] <= 0.10 and m["rms_pos_err"] <= 0.002,
                dict(rms_force_err=m["rms_force_err"], rms_pos_err=m["rms_pos_err"])))
    ok, vals = True, {}
    for T in S1["delays"]:
        m = s1_metrics(s1_run(make_arch(), "release", T))
        good = (not m["watchdog"]) and m["E_min"] >= -1e-3 and m["sat"] == 0 and m["x_abs_max"] <= 0.038
        ok &= good
        vals[T] = dict(watchdog=m["watchdog"], E_min=m["E_min"], sat=m["sat"], x_abs_max=m["x_abs_max"])
    res.append(("AT1-2", ok, vals))
    ok3, ok4, ok5, v3, v4, v5 = True, True, True, {}, {}, {}
    for T in S1["delays_task"]:
        mn = s1_metrics(s1_run(make_arch(), "push", T, "normal"))
        mt = s1_metrics(s1_run(make_arch(), "push", T, "tough"))
        mf = s1_metrics(s1_run(make_arch(), "push", T, "no_ligament"))
        g3 = (mn["v_set_tissue"] <= S1["v_tissue"] * 1.02 and mn["v_set_zone"] <= S1["v_zone"] * 1.02
              and mn["f_max"] <= S1["F_lim"] and mt["f_max"] <= S1["F_lim"] and mn["ligament_punctured"]
              and S1["target"][0] <= mn["z_end"] <= S1["target"][1])
        g4 = mf["z_max"] <= S1["z_lim"] + 2e-4 and max(mn["f_crit_max"], mf["f_crit_max"], mt["f_crit_max"]) == 0.0
        g5 = mn["ligament_punctured"] and mn["overshoot"] <= S1["overshoot_max"] * 1e-3
        ok3 &= g3; ok4 &= g4; ok5 &= g5
        v3[T] = dict(v_set_tissue=mn["v_set_tissue"], v_set_zone=mn["v_set_zone"], f_max=mn["f_max"],
                     f_max_tough=mt["f_max"], z_end=mn["z_end"])
        v4[T] = dict(z_max_no_ligament=mf["z_max"], f_crit_max=max(mn["f_crit_max"], mf["f_crit_max"]))
        v5[T] = dict(overshoot=mn["overshoot"])
    res += [("AT1-3", ok3, v3), ("AT1-4", ok4, v4), ("AT1-5", ok5, v5)]
    return res


# =============================================================================================
# Scenario 2 -- Image-guided needle targeting with the course arm
# =============================================================================================
def _T(p=(0, 0, 0), yaw_deg=0.0):
    return registration.pose(registration.rot_z(math.radians(yaw_deg)), p)


S2 = dict(
    T_sp_nominal=_T((0.200, 0.060, 0.040)),                  # phantom pose from the bench drawing
    T_sp_true=_T((0.200, 0.060, 0.040)) @ _T((0.0025, -0.0030, 0.0), 3.0),   # where it really is
    fiducials_p=np.array([[0.025, 0.025, 0.0], [-0.025, 0.025, -0.010],
                          [-0.025, -0.025, 0.0], [0.025, -0.025, -0.010]]),  # divots, phantom frame
    checks_p=np.array([[0.0, 0.025, -0.005], [0.0, -0.025, -0.005]]),    # check divots (hardware TRE)
    entry_p=np.array([0.0, 0.0, 0.0]),                       # entry point (port) on the top surface
    target_p=np.array([0.0, 0.0, -0.025]),                   # target 25 mm deep
    vessel_p=np.array([0.008, 0.0, -0.018]),                 # critical structure (centre)
    r_forbidden=0.005,                                       # 3 mm vessel + 2 mm margin
    nudge_p=np.array([0.006, 0.0, 0.0]),                     # target update toward the vessel (AT2-5)
    T_ft_nominal=_T((0.0, 0.0, 0.100)),                      # pointer / needle guide, drawing value
    T_ft_true=_T((0.0006, -0.0004, 0.1008)),                 # what the pivot calibration must find
    pivot_divot_s=np.array([0.170, 0.020, 0.020]),
    sigma_seat=0.0004,                                       # digitisation noise per axis (m)
    sigma_pivot=0.0001,                                      # seating noise in the pivot divot (m)
    pre_height=0.020,                                        # pre-insertion stand-off above the entry
    depth=0.025, T_insert=10.0, dt=0.01,                     # insertion depth, duration, controller period
    v_frac=0.5, a_max=np.radians(400.0),                     # approach limits: 50 % rated speed, 400 deg/s^2
    n_mc=200,
)


def s2_pivot_data(seed=0):
    """Flange orientations and positions (from the 12-bit encoders) for pivot poses about the divot."""
    rng = np.random.default_rng(seed)
    R_list, p_list, q = [], [], arm.Q_HOME.copy()
    for tilt, az in [(0, 0)] + [(tl, a) for tl in (10, 20, 25) for a in range(0, 360, 45)]:
        tl, a = math.radians(tilt), math.radians(az)
        ax = np.array([math.sin(tl) * math.cos(a), math.sin(tl) * math.sin(a), -math.cos(tl)])
        T_tip = arm.line_pose(S2["pivot_divot_s"] + S2["sigma_pivot"] * rng.standard_normal(3), ax, (0, -1, 0))
        found = False
        for s in range(40):
            q0 = q if s == 0 else arm.Q_HOME + rng.normal(0, 0.4, 6)
            qq, ok = arm.ik_tool(T_tip, q0, S2["T_ft_true"])
            lim = arm.soft_limits()
            if ok and np.all(qq > lim[:, 0]) and np.all(qq < lim[:, 1]):
                found = True
                break
        if not found:                      # pose not reachable inside the soft limits: skip it
            continue
        T_sf = arm.fk(arm.quantise(qq), 0.0)
        R_list.append(T_sf[:3, :3])
        p_list.append(T_sf[:3, 3])
    return R_list, p_list


def s2_digitise(T_ft_est, T_sp_true=None, seed=0, points_p=None):
    """Digitise points of the phantom (default: the four fiducials) with the arm: the true tool tip
    sits in each divot (pointer vertical, seating error sigma_seat), the joints are read with 12-bit
    encoders and the tip is computed with the ESTIMATED tool transform T_ft_est. Returns (N, 3) in {s}."""
    rng = np.random.default_rng(seed)
    T_sp_true = S2["T_sp_true"] if T_sp_true is None else T_sp_true
    P = registration.apply(T_sp_true, S2["fiducials_p"] if points_p is None else points_p)
    out, q = [], arm.Q_HOME.copy()
    for p in P:
        T_tip = arm.line_pose(p + S2["sigma_seat"] * rng.standard_normal(3), (0, 0, -1), (0, -1, 0))
        q, _ = arm.ik_tool(T_tip, q, S2["T_ft_true"])
        out.append(arm.fk_tool(arm.quantise(q), T_ft_est)[:3, 3])
    return np.array(out)


def s2_pre_insertion(T_ft, T_sp, q0=None):
    """Joint configuration with the tool tip pre_height above the (registered) entry, on the line
    entry -> target. Returns (q_pre, axis_s, entry_s, target_s)."""
    e = registration.apply(T_sp, S2["entry_p"])
    g = registration.apply(T_sp, S2["target_p"])
    ax = (g - e) / np.linalg.norm(g - e)
    q, ok = arm.ik_tool(arm.line_pose(e - S2["pre_height"] * ax, ax, (0, -1, 0)),
                        arm.Q_HOME if q0 is None else q0, T_ft)
    return q, ax, e, g


def s2_approach_limits():
    return S2["v_frac"] * arm.RATED_SPEED, S2["a_max"]


def s2_check_approach(t, Q, q_start, q_goal):
    """Check a sampled approach trajectory (t, Q): duration, limits (2 % tolerance for the finite
    differences) and end points. Returns dict."""
    t, Q = np.asarray(t, float), np.asarray(Q, float)
    vmax, amax = s2_approach_limits()
    V = np.diff(Q, axis=0) / np.diff(t)[:, None]
    A = np.diff(V, axis=0) / np.diff(t)[1:, None]
    T_q = arm.quintic_duration(q_start, q_goal, vmax, amax)
    return dict(duration=float(t[-1] - t[0]), quintic=T_q, ratio=float((t[-1] - t[0]) / T_q),
                v_peak=float(np.abs(V).max() / vmax), a_peak=float(np.abs(A).max() / amax),
                end_err=float(max(np.abs(Q[0] - q_start).max(), np.abs(Q[-1] - q_goal).max())))


def s2_insert(design, T_ft_est, T_sp_est, nudge=False, servo=True):
    """Insertion of S2: from the pre-insertion pose, the design's rate controller
    ``design.insertion_rate(t, q, ctx)`` (called every dt = 10 ms) moves the tool tip 25 mm along the
    registered line (pre_height + depth in T_insert s) through the entry point. ``ctx`` gives the
    estimated tool transform, the registered entry and target in {s} (the target moves toward the
    vessel at t = 3 s if nudge), the registered vessel centre and r_forbidden, and the planned tip
    path p_d(t), v_d(t). The joint-rate commands are integrated into joint commands; with servo=True
    the servo model of ``arm.ServoModel`` gives the actual joints. Measured on the TRUE geometry:
    RCM error (distance from the true entry point to the true shaft line, while the tip is below the
    entry), shaft-to-vessel clearance (shaft = tip to 0.1 m behind it), final tip-to-target error."""
    dt, Tn = S2["dt"], S2["T_insert"]
    q, ax, e, g = s2_pre_insertion(T_ft_est, T_sp_est)
    L = S2["pre_height"] + S2["depth"]
    p0 = e - S2["pre_height"] * ax
    vessel = registration.apply(T_sp_est, S2["vessel_p"])
    target_nudged = registration.apply(T_sp_est, S2["target_p"] + S2["nudge_p"])
    n = int(round(Tn / dt)) + 1
    Qc = np.zeros((n, 6))
    for k in range(n):
        t = k * dt
        s = min(t / Tn, 1.0)
        sm = 10 * s**3 - 15 * s**4 + 6 * s**5
        sd = (30 * s**2 - 60 * s**3 + 30 * s**4) / Tn if t < Tn else 0.0
        tgt = target_nudged if (nudge and t >= 3.0) else g
        axk = (tgt - e) / np.linalg.norm(tgt - e)
        ctx = dict(T_ft=T_ft_est, entry=e, target=tgt, axis=axk, vessel=vessel, r_forbidden=S2["r_forbidden"],
                   p_d=e + (sm * L - S2["pre_height"]) * axk, v_d=sd * L * axk, dt=dt)
        Qc[k] = q
        qd = np.asarray(design.insertion_rate(t, q, ctx), float)
        q = q + qd * dt
    t = np.arange(n) * dt
    if servo:
        tf = np.arange(0, t[-1] + 2.0, 1e-3)
        qcmd = np.array([Qc[min(int(tt / dt), n - 1)] for tt in tf])
        Qa = arm.ServoModel().simulate(tf, qcmd, q0=Qc[0])["q"][::10]
        t_eval = tf[::10]
    else:
        Qa, t_eval = Qc, t
    e_true = registration.apply(S2["T_sp_true"], S2["entry_p"])
    g_true = registration.apply(S2["T_sp_true"], S2["target_p"])
    v_true = registration.apply(S2["T_sp_true"], S2["vessel_p"])
    rcm, clr = [], []
    for qa in Qa:
        T = arm.fk_tool(qa, S2["T_ft_true"])
        tip, zb = T[:3, 3], T[:3, 2]
        if np.dot(tip - e_true, zb) > 0.002:             # tip at least 2 mm past the entry
            rcm.append(arm.shaft_distance(T, e_true))
        clr.append(arm.segment_point_distance(tip, tip - 0.1 * zb, v_true) - S2["r_forbidden"])
    T_end = arm.fk_tool(Qa[-1], S2["T_ft_true"])
    return dict(t=t_eval, q=Qa, q_cmd=Qc, rcm_max=float(max(rcm) if rcm else np.nan),
                clearance_min=float(min(clr)), tip_err=float(np.linalg.norm(T_end[:3, 3] - g_true)))


def s2_registration_mc(design, T_ft_est, n=None, seed=100):
    """Monte-Carlo registration: n repeated digitisations (new seating noise each time) with the tool
    transform T_ft_est (pass S2["T_ft_true"] to see the effect of the FLE alone). Returns dict with the
    RMS FRE, the RMS TRE at the target and the TRE samples (m)."""
    n = n or S2["n_mc"]
    fres, tres = [], []
    for i in range(n):
        P_s = s2_digitise(T_ft_est, seed=seed + i)
        T = design.register(S2["fiducials_p"], P_s)
        fres.append(registration.fre(T, S2["fiducials_p"], P_s))
        tres.append(registration.tre(T, S2["T_sp_true"], S2["target_p"]))
    return dict(fre_rms=float(metrics.rms(fres)), tre_rms=float(metrics.rms(tres)), tre=np.array(tres))


def s2_acceptance(design, n_mc=None) -> list:
    res = []
    R_list, p_list = s2_pivot_data()
    T_ft = design.calibrate_tool(R_list, p_list)
    err = float(np.linalg.norm(T_ft[:3, 3] - S2["T_ft_true"][:3, 3]))
    res.append(("AT2-1", err <= 0.5e-3, dict(tip_offset_error=err)))
    mc = s2_registration_mc(design, T_ft, n_mc)
    mc0 = s2_registration_mc(design, S2["T_ft_true"], n_mc)
    fle_rms = math.sqrt(3) * S2["sigma_seat"]
    pred = design.tre_prediction(S2["fiducials_p"], S2["target_p"], fle_rms)
    ok = (mc["tre_rms"] <= 1.0e-3 and mc["fre_rms"] <= 1.0e-3
          and abs(mc0["tre_rms"] / pred - 1) <= 0.15)
    res.append(("AT2-2", ok, dict(fre_rms=mc["fre_rms"], tre_rms=mc["tre_rms"], tre_rms_fle_only=mc0["tre_rms"],
                                  tre_predicted=pred)))
    T_sp = design.register(S2["fiducials_p"], s2_digitise(T_ft, seed=7))
    q_pre = s2_pre_insertion(T_ft, T_sp)[0]
    t, Q = design.plan_approach(arm.Q_HOME, q_pre, *s2_approach_limits())
    ap = s2_check_approach(t, Q, arm.Q_HOME, q_pre)
    ok = ap["ratio"] <= 0.8 and ap["v_peak"] <= 1.02 and ap["a_peak"] <= 1.02 and ap["end_err"] <= 1e-3
    res.append(("AT2-4", ok, ap))
    ins = s2_insert(design, T_ft, T_sp, nudge=False, servo=True)
    ok = ins["rcm_max"] <= 1.5e-3 and ins["tip_err"] <= 1.5e-3
    res.insert(2, ("AT2-3", ok, dict(rcm_max=ins["rcm_max"], tip_err=ins["tip_err"])))
    nud = s2_insert(design, T_ft, T_sp, nudge=True, servo=True)
    ok = nud["clearance_min"] >= -0.2e-3 and ins["clearance_min"] >= -0.2e-3
    res.append(("AT2-5", ok, dict(clearance_min_nudge=nud["clearance_min"], clearance_min=ins["clearance_min"])))
    return res


# =============================================================================================
# Scenario 3 -- Assist-as-needed wrist rehabilitation device (one Hapkit + IMU on the handle)
# =============================================================================================
S3 = dict(
    A=0.020, f=0.2,               # target: x_t(t) = A sin(2 pi f t) (1 mm of handle = 1 deg of wrist)
    x_rom=0.025,                  # range-of-motion limit of this patient: +-25 mm = +-25 deg
    r_h=0.090,                    # handle lever (m): gyro measures v / r_h
    T_end=25.0, t_skip=5.0,
    impaired=dict(g=0.2, kp=40.0), healthy=dict(g=1.0, kp=0.0),
    spasm=(11.0, 0.5, 1.2),       # outward spasm: start (s), duration (s), force (N)
    release=20.0,
)


def s3_target(t):
    return S3["A"] * math.sin(2 * math.pi * S3["f"] * t), S3["A"] * 2 * math.pi * S3["f"] * math.cos(2 * math.pi * S3["f"] * t)


class Patient:
    """Patient's hand on the handle (illustrative): drive g k (x_t(t - tau) - x) toward the delayed
    target (visuomotor delay tau = 0.15 s, k = 100 N/m, strength g), spastic stiffness kp toward 0,
    damping b = 2 N s/m, hand mass 0.10 kg; optional spasm pulse and release time."""

    def __init__(self, g=0.2, kp=40.0, k=100.0, b=2.0, m=0.10, tau=0.15, spasm=None, release=np.inf):
        self.g, self.kp, self.k, self.b, self.m, self.tau = g, kp, k, b, m, tau
        self.spasm, self.release = spasm, release

    def force(self, t, x, v):
        if t >= self.release:
            return 0.0
        F = self.g * self.k * (s3_target(t - self.tau)[0] - x) - self.kp * x - self.b * v
        if self.spasm is not None and self.spasm[0] <= t < self.spasm[0] + self.spasm[1]:
            F += self.spasm[2]
        return F

    def mass(self, t):
        return self.m if t < self.release else 0.0


def s3_patient(case):
    if case == "impaired":
        return Patient(**S3["impaired"])
    if case == "healthy":
        return Patient(**S3["healthy"])
    if case == "spasm":
        return Patient(**S3["impaired"], spasm=S3["spasm"])
    if case == "release":
        return Patient(**S3["impaired"], release=S3["release"])
    raise ValueError(case)


def s3_imu(seed=0, rate=200.0):
    """Gyro on the handle (Lab 9 IMU model): returns sensors(t, x, v) -> dict(gyro, new). The gyro
    reads v / r_h + bias + noise at 200 Hz (held between samples)."""
    rng = np.random.default_rng(seed)
    st = dict(last=-1.0, y=0.0)
    imu = devices.IMU

    def f(t, x, v):
        if t - st["last"] >= 1.0 / rate - 1e-9:
            st["last"] = t
            st["y"] = v / S3["r_h"] + imu["gyro_bias"] + imu["gyro_noise"] * rng.standard_normal()
            return dict(gyro=st["y"], new=True)
        return dict(gyro=st["y"], new=False)
    return f


def s3_run(design, case="impaired", seed=0):
    """One S3 run. ``design.controller(case)`` returns ctrl(t, x_meas, sens) -> F with a ``log`` dict
    of lists 't', 'v_hat' (m/s), 'F_u_hat' (N, estimated patient force)."""
    ctrl = design.controller(case)
    T_end = 24.0 if case == "release" else S3["T_end"]
    out = devices.simulate_device(ctrl, T_end, user=s3_patient(case), sensors=s3_imu(seed))
    out["log"] = {k: np.asarray(v, float) for k, v in getattr(ctrl, "log", {}).items()}
    out["ctrl"] = ctrl
    return out


def _moving_average(x, n):
    """Centred (zero-phase) moving average over n samples."""
    k = np.ones(n) / n
    return np.convolve(np.pad(x, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


def s3_metrics(out) -> dict:
    t = out["t"]
    m = t >= S3["t_skip"]
    xt = np.array([s3_target(tt)[0] for tt in t])
    r = dict(rmse=float(metrics.rms(xt[m] - out["x"][m])), F_mean=float(np.mean(np.abs(out["F"][m]))),
             x_abs_max=float(np.abs(out["x"]).max()), sat=int(out["sat"]),
             watchdog=bool(out["watchdog_t"] == out["watchdog_t"]))
    L = out["log"]
    if "v_hat" in L and len(L["v_hat"]) == len(t):
        r["v_rmse"] = float(metrics.rms(L["v_hat"][m] - out["v"][m]))
    if "F_u_hat" in L and len(L["F_u_hat"]) == len(t):
        r["F_u_rmse"] = float(metrics.rms(L["F_u_hat"][m] - out["F_h"][m]))
    tail = t >= t[-1] - 2.0
    x = out["x"]
    r["osc_tail"] = float(metrics.rms((x - _moving_average(x, 100))[tail]))   # content above ~5 Hz
    return r


def s3_acceptance(design) -> list:
    res = []
    mi = s3_metrics(s3_run(design, "impaired"))
    mh = s3_metrics(s3_run(design, "healthy"))
    ms = s3_metrics(s3_run(design, "spasm"))
    mr = s3_metrics(s3_run(design, "release"))
    from . import baselines
    mh0 = s3_metrics(s3_run(baselines.S3Baseline(), "healthy"))
    oi = s3_run(design, "spasm")
    msp = s3_metrics(oi)
    F_rms = float(metrics.rms(oi["F_h"][oi["t"] >= S3["t_skip"]]))
    res.append(("AT3-1", mi.get("v_rmse", 1.0) <= 1.0e-3 and msp.get("F_u_rmse", 1.0) <= 0.3 * F_rms,
                dict(v_rmse=mi.get("v_rmse"), F_u_rmse_spasm=msp.get("F_u_rmse"), F_h_rms_spasm=F_rms)))
    res.append(("AT3-2", mi["rmse"] <= 5.0e-3 and mi["sat"] == 0, dict(rmse=mi["rmse"], sat=mi["sat"])))
    res.append(("AT3-3", mi["F_mean"] >= 0.05 and mh["F_mean"] <= 0.25 * mi["F_mean"]
                and mh["rmse"] <= mh0["rmse"] + 0.5e-3,
                dict(F_mean_healthy=mh["F_mean"], F_mean_impaired=mi["F_mean"], rmse_healthy=mh["rmse"],
                     rmse_healthy_baseline=mh0["rmse"])))
    lim = S3["x_rom"] + 0.3e-3
    res.append(("AT3-4", ms["x_abs_max"] <= lim and mi["x_abs_max"] <= lim and ms["sat"] == 0,
                dict(x_abs_max_spasm=ms["x_abs_max"], x_abs_max=mi["x_abs_max"], sat=ms["sat"])))
    res.append(("AT3-5", (not mr["watchdog"]) and mr["x_abs_max"] <= lim and mr["osc_tail"] <= 0.1e-3,
                dict(watchdog=mr["watchdog"], x_abs_max=mr["x_abs_max"], osc_tail=mr["osc_tail"])))
    return res


# =============================================================================================
# Scenario 6 -- Load-carriage exoskeleton knee: Hapkit joint emulator + thigh IMU
# =============================================================================================
S6 = dict(
    x0_deg=35.0,                  # knee angle at x = 0; 1 mm of handle = 1 deg of knee
    knee_range=(0.0, 70.0),       # allowed knee range (deg) -> x in [-35, +35] mm
    f0=1.0, f1=1.25, t_switch=10.0, T_end=25.0,   # cadence (strides/s) and its change
    F_load=0.8,                   # load-carriage knee load in stance, scaled (N, toward flexion)
    sea=dict(m_m=0.05, b_m=0.5, k_s=300.0, u_max=1.5),   # virtual series-elastic actuator
    dF_max=40.0,                  # rate limit of the commanded assistance F_d (N/s)
    r_imu=0.15, imu_rate=200.0,
    push=(7.7, 0.3, 1.5),         # stumble: extra wearer force into flexion (s, s, N)
    dropout=(8.0, 0.3),           # IMU dropout (start, duration)
)
_D2R = math.pi / 180.0


def _bump(p, c, w):
    d = (np.asarray(p, float) - c + 0.5) % 1.0 - 0.5
    return np.exp(-0.5 * (d / w) ** 2)


def s6_knee_deg(phase):
    """Knee flexion (deg) over the gait cycle (phase 0 = heel strike): stance bump and swing peak."""
    return 5.0 + 12.0 * _bump(phase, 0.15, 0.07) + 55.0 * _bump(phase, 0.72, 0.10)


def s6_thigh_deg(phase):
    """Thigh (hip) angle (deg, flexion positive): 5 + 20 cos(2 pi phase)."""
    return 5.0 + 20.0 * np.cos(2 * np.pi * np.asarray(phase, float))


def s6_stance(phase):
    p = np.asarray(phase, float)
    return 0.5 * (1 - np.cos(2 * np.pi * np.clip(p / 0.6, 0, 1))) * (p < 0.6)


def s6_x(knee_deg):
    return (np.asarray(knee_deg, float) - S6["x0_deg"]) * 1e-3


class Gait:
    """True gait phase with the cadence change of the scenario."""

    def __init__(self, T_end=None, h=1e-3):
        T_end = T_end or S6["T_end"]
        self.h = h
        self.t = np.arange(0.0, T_end + 1.0, h)
        self.f = np.where(self.t < S6["t_switch"], S6["f0"], S6["f1"])
        self.phi = np.concatenate([[0.0], np.cumsum(self.f[:-1] * h)]) % 1.0

    def phase(self, t):
        return float(self.phi[min(int(round(t / self.h)), len(self.phi) - 1)])


class Wearer:
    """The wearer's leg (operator hand on the handle) follows the knee profile at the TRUE phase:
    F = k (x_knee(phase) - x) - b v, k = 200 N/m, b = 4 N s/m, 0.10 kg; optional stumble push."""

    def __init__(self, gait, k=200.0, b=4.0, m=0.10, push=None):
        self.g, self.k, self.b, self.m, self.push = gait, k, b, m, push

    def force(self, t, x, v):
        F = self.k * (float(s6_x(s6_knee_deg(self.g.phase(t)))) - x) - self.b * v
        if self.push and self.push[0] <= t < self.push[0] + self.push[1]:
            F += self.push[2]
        return F

    def mass(self, t):
        return self.m


def s6_imu(gait, T_end=None, seed=1, dropout=None):
    """Thigh IMU recording (200 Hz) from ``devices.imu_from_angle``; 'valid' marks dropouts."""
    T_end = T_end or S6["T_end"]
    t = np.arange(0.0, T_end, 1.0 / S6["imu_rate"])
    th = s6_thigh_deg(np.array([gait.phase(tt) for tt in t])) * _D2R
    d = devices.imu_from_angle(t, th, r_imu=S6["r_imu"], seed=seed)
    d["valid"] = np.ones(len(t), bool)
    if dropout:
        d["valid"][(t >= dropout[0]) & (t < dropout[0] + dropout[1])] = False
    return d


class VirtualSEA:
    """Series-elastic actuator rendered in the firmware: a virtual motor mass m_m (damping b_m)
    driven by the command u (|u| <= u_max) is connected to the handle by a spring k_s; the handle
    receives the spring force F_s = k_s (x_mot - x)."""

    def __init__(self, x0=0.0):
        p = S6["sea"]
        self.m, self.b, self.k, self.u_max = p["m_m"], p["b_m"], p["k_s"], p["u_max"]
        self.x_mot, self.v_mot = x0, 0.0

    def spring_force(self, x):
        return self.k * (self.x_mot - x)

    def step(self, u, x, h=1e-3):
        u = float(np.clip(u, -self.u_max, self.u_max))
        Fs = self.spring_force(x)
        self.v_mot += h * (u - self.b * self.v_mot - Fs) / self.m
        self.x_mot += h * self.v_mot
        return u


def s6_run(design, case="nominal", seed=1, assist=True):
    """One S6 run. ``design.controller(imu)`` returns ctrl(t, x_meas, sea) -> u (virtual motor command)
    with a ``log`` dict of lists 't', 'phase', 'theta' (rad, thigh estimate), 'F_d' (desired force).
    The handle receives F_s (SEA spring) + the load-carriage load in stance. assist=False: u = 0 and
    the SEA motor is locked (no assistance; reference run for the effort)."""
    g = Gait()
    imu = s6_imu(g, seed=seed, dropout=S6["dropout"] if case == "dropout" else None)
    ctrl = design.controller(imu) if design is not None else None
    x_start = float(s6_x(s6_knee_deg(0.0)))
    sea = VirtualSEA(x_start)
    rec = dict(Fs=[], u=[])

    def firmware(t, xm, sens):
        load = S6["F_load"] * float(s6_stance(g.phase(t)))
        if not assist or ctrl is None:
            if ctrl is not None:
                ctrl(t, xm, sea)
            rec["Fs"].append(0.0)
            rec["u"].append(0.0)
            return load
        u = sea.step(ctrl(t, xm, sea), xm)
        Fs = sea.spring_force(xm)
        rec["Fs"].append(Fs)
        rec["u"].append(u)
        return Fs + load
    push = S6["push"] if case == "push" else None
    out = devices.simulate_device(firmware, S6["T_end"], user=Wearer(g, push=push), x0=x_start)
    out["Fs"], out["u"] = np.array(rec["Fs"]), np.array(rec["u"])
    out["log"] = {k: np.asarray(v, float) for k, v in getattr(ctrl, "log", {}).items()} if ctrl else {}
    out["phase_true"] = np.array([g.phase(tt) for tt in out["t"]])
    out["imu"] = imu
    return out


def s6_metrics(out) -> dict:
    t = out["t"]
    L = out["log"]
    r = dict(x_max=float(out["x"].max()), x_min=float(out["x"].min()), sat=int(out["sat"]),
             effort=float(np.mean(np.abs(out["F_h"][t > 3.0]))),
             u_max=float(np.abs(out["u"]).max()),
             dFs_max=float(np.abs(np.diff(out["Fs"])).max() / 1e-3))
    if "phase" in L and len(L["phase"]) == len(t):
        e = metrics.wrap_phase(L["phase"] - out["phase_true"])
        m1 = (t > 3.0) & (t < S6["t_switch"])
        m2 = t > S6["t_switch"] + 3.0
        r["phase_rms_1"], r["phase_rms_2"] = float(metrics.rms(e[m1])), float(metrics.rms(e[m2]))
        r["phase_err"] = e
    if "theta" in L and len(L["theta"]) == len(t):
        th_true = s6_thigh_deg(out["phase_true"]) * _D2R
        r["theta_rms"] = float(metrics.rms((L["theta"] - th_true)[t > 3.0]))
    if "F_d" in L and len(L["F_d"]) == len(t):
        r["force_track_rms"] = float(metrics.rms((out["Fs"] - L["F_d"])[t > 3.0]))
        r["dFd_max"] = float(np.abs(np.diff(L["F_d"])).max() / 1e-3)
    return r


def s6_acceptance(design) -> list:
    res = []
    mn = s6_metrics(s6_run(design, "nominal"))
    m0 = s6_metrics(s6_run(design, "nominal", assist=False))
    mp = s6_metrics(s6_run(design, "push"))
    od = s6_run(design, "dropout")
    md = s6_metrics(od)
    x_lo, x_hi = (float(s6_x(a)) for a in S6["knee_range"])
    res.append(("AT6-1", mn.get("theta_rms", 1) <= 1.5 * _D2R and mn.get("phase_rms_1", 1) <= 0.03
                and mn.get("phase_rms_2", 1) <= 0.03,
                dict(theta_rms_deg=mn.get("theta_rms", np.nan) / _D2R, phase_rms_1=mn.get("phase_rms_1"),
                     phase_rms_2=mn.get("phase_rms_2"))))
    red = 1.0 - mn["effort"] / m0["effort"]
    res.append(("AT6-2", red >= 0.20 and mn.get("force_track_rms", 1) <= 0.08,
                dict(effort_reduction=red, force_track_rms=mn.get("force_track_rms"))))
    dFd = mn.get("dFd_max", np.inf)          # nominal walking (the range filter may override it in a stumble)
    res.append(("AT6-3", mn["u_max"] <= S6["sea"]["u_max"] + 1e-9 and dFd <= S6["dF_max"] and mn["sat"] == 0
                and mp["sat"] == 0, dict(u_max=mn["u_max"], dFd_max=dFd, dFs_max=mn["dFs_max"], sat=mn["sat"] + mp["sat"])))
    tol = 0.3e-3
    res.append(("AT6-4", mp["x_max"] <= x_hi + tol and min(mp["x_min"], mn["x_min"]) >= x_lo - tol
                and mn["x_max"] <= x_hi + tol,
                dict(x_max_push=mp["x_max"], x_max=mn["x_max"], x_min=min(mp["x_min"], mn["x_min"]))))
    t = od["t"]
    t0, dur = S6["dropout"]
    win = (t >= t0 + 0.1) & (t < t0 + dur)
    fade = float(np.abs(od["Fs"][win]).max()) if np.any(win) else np.nan
    rec = (t > t0 + dur + 2.0) & (t < t0 + dur + 4.0)
    ph_rec = float(metrics.rms(md["phase_err"][rec])) if "phase_err" in md else 1.0
    res.append(("AT6-5", fade <= 0.05 and md["x_max"] <= x_hi + tol and ph_rec <= 0.05,
                dict(F_s_during_dropout=fade, x_max=md["x_max"], phase_rms_after=ph_rec)))
    return res


# =============================================================================================
# Scenarios 4 and 5 -- teleoperation of the course arm over a delayed link (Hapkit leader)
# =============================================================================================
def _digitise_points(P_true_s, sigma, seed):
    return registration.localise(P_true_s, sigma, np.random.default_rng(seed))


def _registration_mc(design, P_model, T_true, targets, sigma, n, seed=200):
    """Monte-Carlo registration from points digitised with isotropic noise sigma. Returns the RMS and
    the maximum TRE (m) at each target (rows of ``targets``, model frame) and the RMS FRE."""
    tre = np.zeros((n, len(targets)))
    fre = np.zeros(n)
    for i in range(n):
        P_s = _digitise_points(registration.apply(T_true, P_model), sigma, seed + i)
        T = design.register(P_model, P_s)
        fre[i] = registration.fre(T, P_model, P_s)
        tre[i] = [registration.tre(T, T_true, p) for p in targets]
    return dict(tre_rms=metrics.rms(tre, axis=0), tre_max=tre.max(axis=0), fre_rms=float(metrics.rms(fre)))


def push_hold_back(x0, x_push, x_back, speed=0.010, t_hold_end=6.0, t_back=1.5):
    """Operator intent of the S4/S5 tests: ramp from x0 to x_push at ``speed``, hold until
    t_hold_end (pushing against the boundary), then move back to x_back in t_back seconds (free
    space); the handle is released later (``release``)."""
    ramp = devices.ramp_to(x0, x_push, speed, 0.2)

    def f(t):
        if t < t_hold_end:
            return ramp(t)
        s = min((t - t_hold_end) / t_back, 1.0)
        return x_push + s * (x_back - x_push)
    return f


def _servo_teleop_run(arch, intent, T_end, delay, alpha, release, x0_m, seed=3):
    op = devices.Operator(intent, release_time=release)
    servo = devices.ServoAxis(v_max=0.050)
    return teleop.simulate(arch, op, T_end, delay=delay, follower="servo", servo=servo, x0_m=x0_m,
                           x0_s=alpha * x0_m, seed=seed)


def _teleop_metrics(out, x_lim, x_wall, alpha, x0_m, t_hold=(6.0, 8.0)):
    """Common S4/S5 numbers. Follower coordinate u = x_s - alpha x0_m (0 at the start)."""
    t = out["t"]
    u = out["x_s"] - alpha * x0_m
    r = out["r"] - alpha * x0_m
    hold = (t >= t_hold[0]) & (t < t_hold[1])
    lead_lim = x0_m + (x_lim - 0.005) / alpha                  # leader position 5 mm (follower) before it
    near = np.nonzero((r >= x_lim - 0.005) | (out["x_m"] >= lead_lim))[0]
    free = t < (t[near[0]] if len(near) else np.inf)          # approach phase, before the boundary zone
    vset = np.diff(r) / np.diff(t)
    return dict(u_max=float(u.max()), wall_clear_min=float(x_wall - u.max()),
                watchdog=bool(out["watchdog_t"] == out["watchdog_t"]), E_min=float(np.nanmin(out["E_ch"])),
                sat=int(out["sat_m"]), x_m_abs_max=float(np.abs(out["x_m"]).max()),
                F_hold=float(np.mean(out["F_m"][hold])) if np.any(hold) else float("nan"),
                F_free_max=float(np.abs(out["F_m"][free & (t > 0.5)]).max()) if np.any(free & (t > 0.5)) else 0.0,
                reach=float(r.max()), u=u, r=r, t=t,
                v_set_near=float(np.maximum(vset, 0)[(r[:-1] > x_lim - 0.005)].max())
                if np.any(r[:-1] > x_lim - 0.005) else 0.0)


S4 = dict(
    alpha=2.0, x0_m=-0.030,                  # leader start; follower moves 2 mm per leader mm
    d_true=0.060,                            # true distance from the start to the object face (m)
    camera_error=0.008,                      # baseline: the camera estimate puts the face 8 mm further away
    standoff=0.020, allowance=0.002,
    marks_o=np.array([[0.0, 0.020, 0.020], [0.0, -0.020, 0.020], [0.0, -0.020, -0.020], [0.040, 0.020, -0.020]]),
    face_o=np.array([0.0, 0.0, 0.0]),        # face centre in the object frame (x_o = approach axis)
    T_so_true=registration.pose(registration.rot_z(math.radians(2.0)), (0.060, 0.0004, -0.0003)),
    sigma=0.0005, delays=(0.1, 0.3, 0.5), T_end=12.0, release=9.0, n_mc=200,
    x_push=0.0,                              # operator intent: 10 mm (leader) beyond the boundary
    x_back=-0.025,                           # then back into free space before the release
)


def s4_boundary(T_so):
    """Follower-axis boundary (m from the start) for an object pose T_so in the approach frame
    (x = approach axis, origin = start): face x-coordinate minus the stand-off."""
    return float(registration.apply(T_so, S4["face_o"])[0] - S4["standoff"])


def s4_run(design, delay, T_so_est=None, seed=3):
    T_so_est = S4["T_so_true"] if T_so_est is None else T_so_est
    x_lim = s4_boundary(T_so_est)
    arch = design.architecture(x_lim + S4["alpha"] * S4["x0_m"])     # boundary in follower coordinates
    intent = push_hold_back(S4["x0_m"], S4["x_push"], S4["x_back"])
    out = _servo_teleop_run(arch, intent, S4["T_end"], delay, S4["alpha"], S4["release"], S4["x0_m"], seed)
    m = _teleop_metrics(out, x_lim, S4["d_true"], S4["alpha"], S4["x0_m"], t_hold=(4.5, 6.0))
    m["x_lim"] = x_lim
    return out, m


def s4_acceptance(design) -> list:
    res = []
    mc = _registration_mc(design, S4["marks_o"], S4["T_so_true"], S4["face_o"][None, :], S4["sigma"], S4["n_mc"])
    res.append(("AT4-1", mc["tre_rms"][0] <= 1.0e-3 and mc["tre_max"][0] <= 2.0e-3,
                dict(tre_rms=float(mc["tre_rms"][0]), tre_max=float(mc["tre_max"][0]), fre_rms=mc["fre_rms"])))
    T_est = design.register(S4["marks_o"], _digitise_points(registration.apply(S4["T_so_true"], S4["marks_o"]),
                                                            S4["sigma"], 7))
    ok2 = ok3 = ok4 = ok5 = True
    v2, v3, v4, v5 = {}, {}, {}, {}
    for T in S4["delays"]:
        _, m = s4_run(design, T, T_est)
        ok2 &= (not m["watchdog"]) and m["E_min"] >= -1e-3 and m["sat"] == 0 and m["x_m_abs_max"] <= 0.038
        ok3 &= m["wall_clear_min"] >= S4["standoff"] - S4["allowance"]
        ok4 &= m["u_max"] >= m["x_lim"] - 0.002
        ok5 &= m["F_hold"] >= 0.5 and m["F_free_max"] <= 0.1
        v2[T] = dict(watchdog=m["watchdog"], E_min=m["E_min"], sat=m["sat"], x_m_abs_max=m["x_m_abs_max"])
        v3[T] = dict(clearance_min=m["wall_clear_min"])
        v4[T] = dict(u_max=m["u_max"], x_lim=m["x_lim"])
        v5[T] = dict(F_hold=m["F_hold"], F_free_max=m["F_free_max"])
    res += [("AT4-2", ok2, v2), ("AT4-3", ok3, v3), ("AT4-4", ok4, v4), ("AT4-5", ok5, v5)]
    return res


S5 = dict(
    alpha=0.75, x0_m=-0.035,                 # leader start; 0.75 mm of insertion per leader mm
    d0=-0.010,                               # start depth: laparoscope tip 10 mm above the port
    d_target=0.030, d_wall=0.035,            # target depth and back wall (critical) along the port axis
    d_limit=0.032,                           # depth limit of the safety filter (3 mm before the wall)
    allowance=0.001, fixture=(0.005, 0.002), # viscous fixture: <= 2 mm/s in the last 5 mm before the limit
    tool=0.12,                               # Lab 1 teaching laparoscope (m)
    fiducials_b=np.array([[0.030, 0.030, 0.0], [-0.030, 0.030, -0.015], [-0.030, -0.030, 0.0],
                          [0.030, -0.030, -0.015]]),          # box-trainer fiducials (box frame {k})
    port_b=np.array([0.0, 0.0, 0.0]), target_b=np.array([0.0, 0.0, -0.030]),
    T_sk_nominal=registration.pose(None, (0.200, 0.060, 0.050)),
    T_sk_true=registration.pose(registration.rot_z(math.radians(-2.5)), (0.2022, 0.0575, 0.050)),
    sigma=0.0005, delays=(0.3, 0.6), T_end=16.0, release=14.0, n_mc=200, dt=0.01,
    x_push=0.035, x_back=-0.030, t_hold_end=12.0,
)


def s5_depth_offset():
    """Follower coordinate u = x_s - alpha x0_m starts at 0; depth d = d0 + u."""
    return S5["d0"]


def s5_run(design, delay, seed=3):
    x_lim = S5["d_limit"] - S5["d0"]
    arch = design.architecture(x_lim + S5["alpha"] * S5["x0_m"], S5["fixture"])
    intent = push_hold_back(S5["x0_m"], S5["x_push"], S5["x_back"], t_hold_end=S5["t_hold_end"])
    out = _servo_teleop_run(arch, intent, S5["T_end"], delay, S5["alpha"], S5["release"], S5["x0_m"], seed)
    m = _teleop_metrics(out, x_lim, S5["d_wall"] - S5["d0"], S5["alpha"], S5["x0_m"], t_hold=(9.0, 11.0))
    m["x_lim"] = x_lim
    return out, m


def s5_line_insertion(design, T_sk_est, depth_t, depth, servo=True):
    """Arm insertion along the registered port axis following the teleoperated depth d(t) (sampled on
    depth_t). The design's ``insertion_rate(t, q, ctx)`` is called every 10 ms with ctx: T_ft (the
    laparoscope), entry (registered port), axis, target, p_d, v_d, dt (as in S2, no vessel). Returns the
    RCM error on the TRUE port while the tip is below it, and the lateral tip error at the end."""
    T_ft = registration.pose(None, (0.0, 0.0, S5["tool"]))
    e = registration.apply(T_sk_est, S5["port_b"])
    g = registration.apply(T_sk_est, S5["target_b"])
    ax = (g - e) / np.linalg.norm(g - e)
    dt = S5["dt"]
    tt = np.arange(0.0, depth_t[-1], dt)
    d = np.interp(tt, depth_t, depth)
    dd = np.gradient(d, tt)
    q, _ = arm.ik_tool(arm.line_pose(e + d[0] * ax, ax, (0, -1, 0)), arm.Q_HOME, T_ft)
    Qc = np.zeros((len(tt), 6))
    for k, t in enumerate(tt):
        ctx = dict(T_ft=T_ft, entry=e, target=g, axis=ax, p_d=e + d[k] * ax, v_d=dd[k] * ax, dt=dt,
                   vessel=None, r_forbidden=None)
        Qc[k] = q
        q = q + np.asarray(design.insertion_rate(t, q, ctx), float) * dt
    if servo:
        tf = np.arange(0.0, tt[-1], 1e-3)
        qcmd = Qc[np.minimum((tf / dt).astype(int), len(tt) - 1)]
        Qa = arm.ServoModel().simulate(tf, qcmd, q0=Qc[0])["q"][::10]
    else:
        Qa = Qc
    e_true = registration.apply(S5["T_sk_true"], S5["port_b"])
    rcm = []
    for qa in Qa:
        T = arm.fk_tool(qa, T_ft)
        if np.dot(T[:3, 3] - e_true, T[:3, 2]) > 0.002:
            rcm.append(arm.shaft_distance(T, e_true))
    return dict(rcm_max=float(max(rcm)) if rcm else float("nan"), q=Qa)


def s5_acceptance(design) -> list:
    res = []
    mc = _registration_mc(design, S5["fiducials_b"], S5["T_sk_true"], np.vstack([S5["port_b"], S5["target_b"]]),
                          S5["sigma"], S5["n_mc"])
    res.append(("AT5-1", bool(np.all(mc["tre_rms"] <= 1.0e-3)),
                dict(tre_rms_port=float(mc["tre_rms"][0]), tre_rms_target=float(mc["tre_rms"][1]), fre_rms=mc["fre_rms"])))
    T_est = design.register(S5["fiducials_b"], _digitise_points(registration.apply(S5["T_sk_true"], S5["fiducials_b"]),
                                                                S5["sigma"], 7))
    ok2 = ok3 = ok5 = True
    v2, v3, v5 = {}, {}, {}
    run_600 = None
    for T in S5["delays"]:
        out, m = s5_run(design, T)
        ok2 &= (not m["watchdog"]) and m["E_min"] >= -1e-3 and m["sat"] == 0 and m["x_m_abs_max"] <= 0.038
        # depth limit: the registration error along the axis is part of the allowance
        e_err = float(np.dot(registration.apply(T_est, S5["target_b"]) - registration.apply(S5["T_sk_true"], S5["target_b"]),
                             np.array([0, 0, -1.0])))
        clear = m["wall_clear_min"] - e_err
        ok3 &= clear >= (S5["d_wall"] - S5["d_limit"]) - S5["allowance"] and m["u_max"] + S5["d0"] >= S5["d_target"]
        ok5 &= m["v_set_near"] <= S5["fixture"][1] * 1.02 and m["F_free_max"] <= 0.1
        v2[T] = dict(watchdog=m["watchdog"], E_min=m["E_min"], sat=m["sat"], x_m_abs_max=m["x_m_abs_max"])
        v3[T] = dict(wall_clearance_min=clear, depth_max=m["u_max"] + S5["d0"])
        v5[T] = dict(v_set_near=m["v_set_near"], F_free_max=m["F_free_max"])
        run_600 = (out, m)
    res += [("AT5-2", ok2, v2), ("AT5-3", ok3, v3)]
    out, m = run_600
    ins = s5_line_insertion(design, T_est, m["t"], S5["d0"] + m["u"])
    res.append(("AT5-4", ins["rcm_max"] <= 1.5e-3, dict(rcm_max=ins["rcm_max"])))
    res.append(("AT5-5", ok5, v5))
    return res
