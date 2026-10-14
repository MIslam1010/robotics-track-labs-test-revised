"""The required hardware run of the SBEG 46X project: the provisional thresholds of the hardware
acceptance tests HWk-1 ... HWk-4 (``HW``), identification of the hardware models, "replay"
simulations of a logged command (the simulation half of every HWk-2 test) and reference signals.

Use it together with ``hardware`` (log format of the SBEG 461 project, plus aux_ columns):

    log = hardware.load_log("results/hardware/2026-11-25_session2_HW3-2.csv")
    ident = hwtasks.fit_free_decay(t, x, K=100.0)               # session 1 (spring K = 100 N/m): m_h, b_h
    sim = hwtasks.replay_device(ctrl, log["t"], dev=devices.Hapkit(m_h=ident["m_h"], b_h=ident["b_h"]))
    out = hardware.compare(log, sim, skip=1.0)                  # RMS(sim - meas) of HW3-2

All thresholds are ILLUSTRATIVE and PROVISIONAL: the instructor confirms them with a calibration run
on the course units in Week 9 and publishes any change before the first hardware session.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.signal import butter, filtfilt

from . import arm, devices
from .metrics import rms

# ------------------------------------------------------------------------------ thresholds (SI)
HW = {
    1: dict(device="Hapkit pair (leader + follower)", delays=(0.0, 0.1, 0.2), z_lim=0.036, z_lim_tol=0.3e-3,
            target=(0.031, 0.036), F_lim=1.4, v_tissue=0.005, v_zone=0.002, v_tol=0.05, trials=3,
            rms_sim_meas=0.5e-3, E_min=-1e-3, x_abs_max=0.038, release_ptp=1e-3, t_stop=0.010, hold_advance=0.1e-3,
            fade_jump=0.2,
            link_loss_fade=0.10),
    2: dict(device="course arm + pointer + 3-D-printed phantom", pivot_rms=1.0e-3, fre=1.5e-3, tre_check=2.0e-3,
            tip_target=2.5e-3, approach_time_tol=0.10,
            rms_sim_meas_deg=1.0, rcm=2.5e-3, clearance_tol=0.5e-3, speed_fraction=0.50, current_fraction=0.60,
            monitor_rcm=4.0e-3, inject=5.0e-3, t_stop=0.5, v_stop_deg=2.0, rcm_never=8.0e-3),
    3: dict(device="one Hapkit + IMU on the handle", F_u_tol=0.10, loads=(0.49, 0.98), v_rms=1.5e-3,
            rmse_assist=6.0e-3, effort_ratio=0.25, rms_sim_meas=1.0e-3, x_rom=0.025, x_rom_tol=0.5e-3,
            drop_mass=0.100, osc_tail=0.2e-3, t_stop=0.010, imu_fade=0.05, force_jump=0.5),
    4: dict(device="Hapkit leader + course-arm follower (approach axis)", delays=(0.1, 0.3, 0.5), fre=1.5e-3,
            tre_check=2.0e-3, reach_tol=2.0e-3, F_hold=0.5, rms_sim_meas=1.0e-3, E_min=-1e-3, standoff=0.020,
            allowance=2.0e-3, h_tol=0.5e-3, speed_fraction=0.50, current_fraction=0.60, hold_v=1.0e-3,
            hold_time=0.3, t_stop=0.010),
    5: dict(device="Hapkit leader + course arm with the teaching laparoscope through a ring port",
            delays=(0.3, 0.6), fre=1.5e-3, tre_check=2.0e-3, depth_target=0.030, depth_tol=2.0e-3,
            v_fixture=0.002, v_tol=0.05, rms_sim_meas_deg=1.0, E_min=-1e-3, rcm=3.0e-3, wall_clear=2.0e-3,
            speed_fraction=0.50, current_fraction=0.60, inject=6.0e-3, t_stop=0.5, v_stop_deg=2.0,
            rcm_never=8.0e-3),
    6: dict(device="one Hapkit as knee emulator + IMU module", cadences=(1.0, 1.25), phase_rms=0.06,
            cadence_tol=0.05, effort_reduction=0.20, force_track_rms=0.10, rms_sim_meas=1.5e-3,
            knee_x=(-0.035, 0.035), x_tol=0.3e-3, drop_mass=0.100, u_max=1.5, dF_max=40.0, imu_fade=0.05,
            fade_time=0.10, phase_recover=0.08, t_stop=0.010),
}
"""Provisional hardware thresholds (SI units) used in ``scenario_sheets.md``, HWk-1 ... HWk-4."""

# derated limits of the course hardware (Course resources: "Course robot models and hardware safety checklist")
HAPKIT_FORCE_MAX = devices.HAPKIT["force_max"]          # 2 N, saturated in the firmware
HAPKIT_RANGE = devices.HAPKIT["x_range"]                # +-40 mm, watchdog
ARM_SPEED_MAX = arm.SPEED_FRACTION * arm.RATED_SPEED   # 80 deg/s (40 deg/s on the first run of a motion)
ARM_SPEED_FIRST_RUN = arm.SPEED_FRACTION_FIRST_RUN * arm.RATED_SPEED


# ------------------------------------------------------------------------------ identification
def fit_free_decay(t, x, K: float = 0.0, k_g: float = devices.HAPKIT["k_g"], T: float = devices.HAPKIT["T"],
                   t0: float = 0.0, delay_eff: float | None = None) -> dict:
    """Identify the handle mass m_h and the physical damping b_h of a haptic device from a free-decay
    record x(t) (handle displaced and released at t0, no hand on it), while the firmware renders a
    spring K (N/m; K = 0 gives the device's own gravity stiffness k_g only).

    Fits x = A exp(-s t) cos(wd t + phi) + c by least squares; then wn^2 = s^2 + wd^2, zeta = s / wn,
    m_h = (K + k_g) / wn^2 and b_h = 2 zeta wn m_h + K * delay_eff. The last term corrects for the
    energy that a SAMPLED spring injects (it acts as a negative damper of about K T / 2, Week 4,
    Colgate and Brown 1994); delay_eff defaults to T / 2 (zero-order hold). Returns a dict with
    m_h, b_h, wn, zeta, fd (Hz) and the RMS fit residual."""
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    m = t >= t0
    tt, xx = t[m] - t[m][0], x[m]
    # initial guess from the zero crossings of the detrended signal
    xd = xx - np.mean(xx[len(xx) // 2:])
    zc = np.nonzero(np.diff(np.signbit(xd)))[0]
    wd0 = math.pi / np.mean(np.diff(tt[zc])) if len(zc) >= 3 else 2 * math.pi * 1.0
    p0 = [xd[0] if abs(xd[0]) > 0 else 1e-3, 1.0, wd0, 0.0, float(np.mean(xx[len(xx) // 2:]))]

    def model(tau, A, s, wd, phi, c):
        return A * np.exp(-s * tau) * np.cos(wd * tau + phi) + c
    p, _ = curve_fit(model, tt, xx, p0=p0, maxfev=20000)
    A, s, wd, phi, c = p
    wd = abs(wd)
    wn = math.hypot(s, wd)
    zeta = s / wn
    m_h = (K + k_g) / wn**2
    delay_eff = T / 2 if delay_eff is None else delay_eff
    b_h = 2 * zeta * wn * m_h + K * delay_eff
    return dict(m_h=float(m_h), b_h=float(b_h), wn=float(wn), zeta=float(zeta), fd=float(wd / (2 * math.pi)),
                rms_residual=float(rms(model(tt, *p) - xx)))


def fit_servo_step(t, q_cmd, q, delays=None, speed_fraction: float = 1.0) -> dict:
    """Identify the servo model of ONE joint of the course arm (``arm.ServoModel``: second-order loop
    wn, zeta behind a transport delay) from a logged step or staircase response. The command must be
    small enough that the servo's rate limit is not active (e.g. 5 deg steps). For each candidate
    delay (default 0 ... 60 ms in 2 ms steps) wn and zeta are fitted by least squares; the best delay
    is returned with wn, zeta and the RMS residual (rad). t must be uniform (resample first)."""
    t = np.asarray(t, float)
    qc = np.asarray(q_cmd, float).reshape(len(t), 1)
    qm = np.asarray(q, float).ravel()
    delays = np.arange(0.0, 0.061, 0.002) if delays is None else np.asarray(delays, float)
    best = None
    for d in delays:
        def res(p):
            sm = arm.ServoModel(wn=p[0], zeta=p[1], delay=float(d), speed_fraction=speed_fraction, n=1)
            sm.vmax = np.inf if speed_fraction >= 1.0 else sm.vmax
            return sm.simulate(t, qc, q0=[qm[0]])["q"][:, 0] - qm
        r = least_squares(res, x0=[25.0, 0.7], bounds=([1.0, 0.05], [200.0, 3.0]))
        cost = float(rms(r.fun))
        if best is None or cost < best["rms_residual"]:
            best = dict(wn=float(r.x[0]), zeta=float(r.x[1]), delay=float(d), rms_residual=cost)
    return best


def imu_bias_at_rest(t, gyro, t_end: float) -> float:
    """Gyro bias (rad/s): mean of the gyro signal while the IMU is at rest (t < t_end). Record at least
    10 s at rest at the start of every IMU session (HW3-1, HW6-1)."""
    t, g = np.asarray(t, float), np.asarray(gyro, float)
    return float(np.mean(g[t < t_end]))


# ------------------------------------------------------------------------------ replay simulations
def _uniform(t, h):
    t = np.asarray(t, float)
    return np.arange(t[0], t[-1] + 0.5 * h, h)


def replay_device(controller, t_log, dev=None, user=None, x0: float = 0.0, sensors=None) -> dict:
    """Simulate ONE haptic device (your identified ``devices.Hapkit``) with the SAME firmware controller
    that ran on the hardware, over the duration of the log (Scenarios 3 and 6, hand-off runs). Returns the
    ``devices.simulate_device`` result with ``q`` = x added, ready for ``hardware.compare``."""
    T_end = float(np.asarray(t_log, float)[-1])
    out = devices.simulate_device(controller, T_end, user=user, x0=x0, dev=dev, sensors=sensors)
    out["q"] = out["x"]
    return out


def replay_follower_hapkit(t_log, r_log, Kp: float, Kd: float, dev=None, f_env_log=None, x0=None,
                           h: float = devices.HAPKIT["T"]) -> dict:
    """Simulate the FOLLOWER haptic device driven by the logged set-point r(t) through the follower PD
    F = Kp (r - x) - Kd v_est (Labs 2-3), minus the logged rendered environment force f_env(t) if given
    (Scenario 1: aux_f_env). This is the "same command" simulation of HW1-2: it removes the operator and
    the channel and tests the follower model alone. Returns dict(t, q) on a uniform 1 kHz grid."""
    t = _uniform(t_log, h)
    r = np.interp(t, t_log, r_log)
    fe = np.zeros_like(t) if f_env_log is None else np.interp(t, t_log, f_env_log)
    D = dev or devices.Hapkit()
    D.x, D.v = (float(r[0]) if x0 is None else x0), 0.0
    est = devices.VelocityEstimator(h)
    q = np.zeros_like(t)
    for k in range(len(t)):
        xm = D.measure()
        F = Kp * (r[k] - xm) - Kd * est(xm) - fe[k]
        D.command(F)
        q[k] = D.x
        for _ in range(5):
            D.advance(h / 5)
    return dict(t=t, q=q, tripped=D.tripped, sat=D.sat_count)


def replay_servo_axis(t_log, r_log, servo=None, x0=None, h: float = 1e-3) -> dict:
    """Simulate the course-arm tip axis (``devices.ServoAxis`` with your identified wn, zeta, delay)
    driven by the logged follower set-point r(t) (Scenarios 4, 5). Returns dict(t, q)."""
    t = _uniform(t_log, h)
    r = np.interp(t, t_log, r_log)
    S = servo or devices.ServoAxis()
    S.x = float(r[0]) if x0 is None else x0
    S.v, S.x_cmd, S._xl, S._buf = 0.0, S.x, None, []
    q = np.zeros_like(t)
    for k in range(len(t)):
        S.command(r[k], h)
        q[k] = S.x
        for _ in range(5):
            S.advance(h / 5)
    return dict(t=t, q=q)


def replay_arm(t_log, q_cmd_log, servo_model=None, q0=None, h: float = 1e-3) -> dict:
    """Simulate the course arm's servos (``arm.ServoModel`` with your identified parameters) for the
    logged joint command (N x 6, rad). The log is resampled onto a uniform 1 kHz grid with a
    zero-order hold, as the servo bus applies each command until the next one. Returns dict(t, q, qd)."""
    t_log = np.asarray(t_log, float)
    Q = np.asarray(q_cmd_log, float).reshape(len(t_log), -1)
    t = _uniform(t_log, h)
    idx = np.clip(np.searchsorted(t_log, t, side="right") - 1, 0, len(t_log) - 1)
    sm = servo_model or arm.ServoModel(n=Q.shape[1])
    out = sm.simulate(t, Q[idx], q0=Q[0] if q0 is None else q0)
    return dict(t=t, q=out["q"], qd=out["qd"])


# ------------------------------------------------------------------------------ reference signals
def zero_phase_velocity(t, x, fc: float = 20.0, order: int = 4) -> np.ndarray:
    """Offline reference velocity of a logged position: central differences, then a zero-phase
    Butterworth low-pass (cut-off fc Hz, filtfilt). It is NOT causal, so it can only judge an online
    estimator after the run (HW3-1); it cannot replace one. t must be (nearly) uniform."""
    t, x = np.asarray(t, float), np.asarray(x, float)
    v = np.gradient(x, t)
    fs = 1.0 / float(np.median(np.diff(t)))
    b, a = butter(order, fc / (fs / 2))
    return filtfilt(b, a, v)


def phase_from_events(t, t_events) -> np.ndarray:
    """Reference gait phase (fraction of the cycle in [0, 1)) that rises linearly from 0 at each event
    (heel strike, or the metronome beat that the operator steps on) to 1 at the next one. NaN before
    the first and after the last event (HW6-1)."""
    t = np.asarray(t, float)
    e = np.sort(np.asarray(t_events, float))
    ph = np.full(len(t), np.nan)
    k = np.searchsorted(e, t, side="right") - 1
    ok = (k >= 0) & (k < len(e) - 1)
    ph[ok] = (t[ok] - e[k[ok]]) / (e[k[ok] + 1] - e[k[ok]])
    return ph


def cadence_from_events(t_events) -> float:
    """Mean cadence (strides/s) from consecutive events of the same leg."""
    return float(1.0 / np.mean(np.diff(np.sort(np.asarray(t_events, float)))))


def effort_reduction(F_net, F_load, mask=None) -> float:
    """1 - mean|F_net| / mean|F_load|: the fraction of the rendered load that the assistance removes
    from the operator's hand (HW6-1). F_net = SEA spring force + load, both logged by the firmware."""
    F_net, F_load = np.abs(np.asarray(F_net, float)), np.abs(np.asarray(F_load, float))
    if mask is not None:
        F_net, F_load = F_net[mask], F_load[mask]
    return float(1.0 - np.mean(F_net) / np.mean(F_load))


def rcm_errors(Q, T_ft, p_port, depth_min: float = 0.002) -> np.ndarray:
    """RCM error (m) from MEASURED joint angles: distance from the port point p_port (in {s}) to the
    tool-shaft line, for the samples where the tip is at least depth_min past the port; NaN elsewhere
    (HW2-3, HW5-3, HW2-4, HW5-4)."""
    out = []
    p_port = np.asarray(p_port, float)
    for q in np.asarray(Q, float):
        T = arm.fk_tool(q, T_ft)
        out.append(arm.shaft_distance(T, p_port) if np.dot(T[:3, 3] - p_port, T[:3, 2]) > depth_min else np.nan)
    return np.array(out)


def shaft_clearance(Q, T_ft, centre, radius: float, shaft: float = 0.10) -> np.ndarray:
    """Clearance (m) of the tool shaft (tip to ``shaft`` m behind it) from a forbidden sphere (centre,
    radius) for each row of measured joint angles; negative = inside the forbidden region (HW2-3)."""
    out = []
    for q in np.asarray(Q, float):
        T = arm.fk_tool(q, T_ft)
        tip, zb = T[:3, 3], T[:3, 2]
        out.append(arm.segment_point_distance(tip, tip - shaft * zb, centre) - radius)
    return np.array(out)


def tre_at_checks(T_reg, P_checks_model, P_checks_measured) -> np.ndarray:
    """Measured TRE (m) at check points that were NOT used in the registration: |T_reg p_model - p_meas|
    for each check point (HW2-1, HW4-1, HW5-1)."""
    P = np.asarray(P_checks_model, float)
    pred = P @ np.asarray(T_reg)[:3, :3].T + np.asarray(T_reg)[:3, 3]
    return np.linalg.norm(pred - np.asarray(P_checks_measured, float), axis=1)


def static_force_check(F_hat, F_known) -> float:
    """Largest error (N) of an online force estimate against known static loads (hanging masses
    m g on a string over a pulley; HW3-1). Use the mean of F_hat over each 2 s static hold."""
    return float(np.max(np.abs(np.asarray(F_hat, float) - np.asarray(F_known, float))))
