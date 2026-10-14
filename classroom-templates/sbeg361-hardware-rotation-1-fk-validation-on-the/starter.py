"""SBEG 361 Hardware rotation 1 (Week 8) -- FK validation on the course arm.  STARTER (edit the functions marked TODO).

Rules: do not import modern_robotics, scipy or course_arm_model here; the helpers above the TODO functions are
complete and may be used (you may replace fk_space / screw_exp by your own Lab 5 versions).

Module `validate.py` of your `sbeg361_kinematics` library: plan check, logged run, hardware acceptance tests
HW-1 ... HW-6 (project brief Sec. 4.1), accuracy and repeatability, backlash, joint-offset calibration and an
error budget.  Notation of Modern Robotics (MR): screw axes S = (omega, v), space-form PoE (MR Eq. 4.14).
Angles: SERVO degrees in logs and files, MODEL radians in FK; q_model = deg2rad(q_servo - offset).
"""
import csv
import json

import numpy as np

from safety_guard import SafetyViolation


# =========================================================== complete helpers (same in starter.py)
def vec_to_so3(omega):
    return np.array([[0, -omega[2], omega[1]], [omega[2], 0, -omega[0]], [-omega[1], omega[0], 0]], float)


def screw_exp(S, theta):
    """e^{[S] theta} for a screw axis S = (omega, v) (MR Eqs. 3.88-3.89); the Lab 5 helper screw_exp(S, theta).

    Same result as Lab 3's matrix_exp6(se3mat) called as matrix_exp6(vec_to_se3(S) * theta)."""
    S = np.asarray(S, float)
    w, v = S[:3], S[3:]
    T = np.eye(4)
    if np.linalg.norm(w) < 1e-12:
        T[:3, 3] = v * theta
        return T
    W = vec_to_so3(w)
    R = np.eye(3) + np.sin(theta) * W + (1 - np.cos(theta)) * W @ W
    G = np.eye(3) * theta + (1 - np.cos(theta)) * W + (theta - np.sin(theta)) * W @ W
    T[:3, :3], T[:3, 3] = R, G @ v
    return T


def fk_space(M, Slist, theta):
    """Space-form PoE T = e^{[S1]th1} ... e^{[Sn]thn} M (MR Eq. 4.14; Lab 5).  You may paste your own."""
    T = np.eye(4)
    for i, th in enumerate(np.asarray(theta, float)):
        T = T @ screw_exp(np.asarray(Slist)[:, i], th)
    return T @ M


def tool_points(M, Slist, Q, p_tool=(0.0, 0.0, 0.0)):
    """Positions in {s} (m) of the tool point p_tool (given in {b}) for each row of Q (model radians)."""
    pb = np.r_[np.asarray(p_tool, float), 1.0]
    return np.array([(fk_space(M, Slist, q) @ pb)[:3] for q in np.atleast_2d(Q)])


def servo_to_model(q_servo_deg, offsets_deg):
    """Model angles (radians) from servo angles (degrees): q_model = deg2rad(q_servo - offset)."""
    return np.deg2rad(np.asarray(q_servo_deg, float) - np.asarray(offsets_deg, float))


def model_to_servo(q_model_rad, offsets_deg):
    """Servo angles (degrees) from model angles (radians)."""
    return np.rad2deg(np.asarray(q_model_rad, float)) + np.asarray(offsets_deg, float)


def load_offsets(path):
    with open(path) as fh:
        return np.array(json.load(fh)["servo_zero_offset_deg"], float)


def load_plan(path):
    """Validation plan (JSON): poses, segment time, rate, tool point, limits, speeds (see data/validation_plan.json)."""
    with open(path) as fh:
        return json.load(fh)


def smooth_segments(q_via_deg, T_seg, rate_hz, t_hold=0.0):
    """Joint-space motion through via points (rows of q_via_deg, degrees) with the cubic time scaling
    s(tau) = 3 tau^2 - 2 tau^3 on each segment of duration T_seg (formula given here; time scaling is taught in SBEG 461 Week 10; MR Sec. 9.2.2.1, Eq. 9.11).
    Peak speed of a segment = 1.5 |dq| / T_seg.  Returns (t [N], q [N x n])."""
    Q = np.atleast_2d(np.asarray(q_via_deg, float))
    dt = 1.0 / rate_hz
    ts, qs, t0 = [], [], 0.0
    for a, b in zip(Q[:-1], Q[1:]):
        n = int(round(T_seg * rate_hz))
        tau = np.arange(n) / n
        s = 3 * tau ** 2 - 2 * tau ** 3
        ts.append(t0 + tau * T_seg)
        qs.append(a + s[:, None] * (b - a))
        t0 += T_seg
    n_hold = int(round(t_hold * rate_hz)) + 1
    ts.append(t0 + dt * np.arange(n_hold))
    qs.append(np.repeat(Q[-1:], n_hold, axis=0))
    return np.concatenate(ts), np.vstack(qs)


def write_log(path, log):
    """Write a log in the format of the project's hwcompare.py: t, cmd_1..n, meas_1..n, stop (degrees)."""
    n = log["cmd"].shape[1]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t"] + [f"cmd_{i + 1}" for i in range(n)] + [f"meas_{i + 1}" for i in range(n)] + ["stop"])
        for k in range(len(log["t"])):
            w.writerow([f"{log['t'][k]:.4f}"] + [f"{x:.4f}" for x in log["cmd"][k]]
                       + [f"{x:.4f}" for x in log["meas"][k]] + [int(log["stop"][k])])


def read_log(path):
    """Read a log written by write_log (or by hwcompare.py)."""
    with open(path, newline="") as fh:
        rows = [r for r in csv.reader(fh) if r]
    header = [h.strip() for h in rows[0]]
    data = np.array([[float(x) for x in r] for r in rows[1:]])
    col = {h: i for i, h in enumerate(header)}
    n = sum(h.startswith("cmd_") for h in header)
    return dict(t=data[:, col["t"]], cmd=data[:, [col[f"cmd_{i + 1}"] for i in range(n)]],
                meas=data[:, [col[f"meas_{i + 1}"] for i in range(n)]],
                stop=data[:, col["stop"]] if "stop" in col else np.zeros(len(data)))


def read_tips(path):
    """Tip-measurement sheet: pose, approach (+1/-1), servo_1..6 (deg), x_mm, y_mm, z_mm."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return dict(pose=np.array([r["pose"] for r in rows]),
                approach=np.array([int(r["approach"]) for r in rows]),
                Q=np.array([[float(r[f"servo_{i + 1}"]) for i in range(6)] for r in rows]),
                P=np.array([[float(r["x_mm"]), float(r["y_mm"]), float(r["z_mm"])] for r in rows]))


def _before_stop(log):
    stop = np.asarray(log["stop"]) > 0.5
    return int(np.argmax(stop)) if stop.any() else len(log["t"])


# =========================================================== Step 1 -- before the arm moves
def check_plan(t, q_servo_deg, offsets_deg, soft_limits_deg, rated_deg_s, speed_fraction=0.25,
               margin_deg=10.0):
    """Planning-level safety check of a joint trajectory (also used on MEASURED angles for HW-4 and HW-5).

    t [N] s; q_servo_deg [N x n]; soft_limits_deg [n x 2] in MODEL degrees (the arm's software limits).
    Returns dict with
      margin_deg     [n]  smallest distance of each joint (model degrees) from its software limits
      peak_speed     [n]  peak |dq/dt| (deg/s), by np.gradient
      speed_limit         speed_fraction * rated_deg_s
      ok_limits, ok_speed, ok   booleans (margin >= margin_deg on every joint; peak <= limit on every joint)"""
    # TODO: convert to model degrees (q - offsets); margin = min(min(q) - lower, upper - max(q)) per joint;
    #       peak = max |np.gradient(q, t, axis=0)| per joint; compare with margin_deg and speed_fraction * rated_deg_s
    raise NotImplementedError


# =========================================================== Step 2 -- logged run
def run_logged(arm, t, q_cmd_servo_deg, stop_at=None):
    """Send the planned commands and log commanded and measured angles (servo degrees).

    For each sample k: wait until t[k]; if stop_at is given and t[k] >= stop_at (first time), call arm.stop();
    while not stopped, send q_cmd[k] (a SafetyViolation also stops the run: call arm.stop() and keep logging);
    then read the measured angles.  After the stop the cmd column holds the LAST command actually sent.
    Returns dict t, cmd [N x n], meas [N x n], stop [N] (0/1), aborted (None or the violation message)."""
    # TODO: loop over k: arm.wait_until(t[k]); stop once when t[k] >= stop_at; while not stopped try arm.send(Q[k])
    #       (on SafetyViolation: arm.stop(), remember the message); store the last command sent, arm.read(), the stop flag
    raise NotImplementedError


# =========================================================== Step 3 -- joint space, task space, stop
def joint_tracking(log):
    """HW-2: statistics of e = meas - cmd (degrees) per joint, samples BEFORE the stop command.
    Returns dict rms [n], max [n], mean [n], n_samples."""
    # TODO: use only samples before the first stop = 1 (helper _before_stop); e = meas - cmd
    raise NotImplementedError


def task_tracking(log, M, Slist, offsets_deg, p_tool=(0.0, 0.0, 0.0)):
    """HW-3: tool-point distance |p(meas) - p(cmd)| (mm) via PoE FK, samples before the stop.
    Returns dict err_mm [N'], rms_mm, max_mm."""
    # TODO: FK of cmd and of meas (servo_to_model, tool_points), distance in mm, before the stop
    raise NotImplementedError


def stop_check(log, band_deg=0.3, tail_s=0.5):
    """HW-6 on a noisy, quantised log.  The rest pose is the per-joint MEDIAN of the measured angles over the last
    tail_s seconds of the log; t_halt = time from the stop command until every joint stays within band_deg of
    that rest pose; motion_deg = largest |meas - meas(at the stop command)| after the stop.
    Returns dict t_stop, t_halt, motion_deg (t_halt = inf if a joint is still outside the band at the end)."""
    # TODO: rest pose = median of meas over the last tail_s; find the last post-stop sample outside the band
    raise NotImplementedError


# =========================================================== Step 4 -- is the arm where the model says?
def hw_tip_errors_mm(Q_servo_deg, P_meas_mm, M, Slist, offsets_deg, p_tool=(0.0, 0.0, 0.0)):
    """HW-1: distance (mm) between the FK prediction of the tool point and the external measurement, per pose.
    Inputs are SERVO degrees [m x 6] plus the arm's offsets (q_model = q_servo - offset), not the model joint angles of
    Lab 5's tip_errors_mm(M, Slist, Q, ...) (hence the hw_ prefix; argument order differs too)."""
    # TODO: FK prediction (mm) of the tool point for each pose minus the measured point; norm per row
    raise NotImplementedError


def accuracy_repeatability(P_mm, p_ref_mm):
    """Pose accuracy and repeatability of repeated visits to ONE commanded pose (industrial convention).
    barycentre = mean of the attained points; AP = |barycentre - p_ref|; l_i = |P_i - barycentre|;
    RP = mean(l) + 3 std(l) with the sample standard deviation (ddof = 1).
    Returns dict barycentre [3], AP_mm, l_mm [m], RP_mm."""
    # TODO: barycentre, AP, distances l_i to the barycentre, RP = mean + 3 * std(ddof=1)
    raise NotImplementedError


def backlash_split(P_plus_mm, P_minus_mm):
    """Distance (mm) between the barycentres of the visits approached from the + side and from the - side."""
    # TODO: difference of the two barycentres
    raise NotImplementedError


def fit_offsets(Q_servo_deg, P_meas_mm, M, Slist, offsets_deg, p_tool=(0.0, 0.0, 0.0), joints=(1, 2),
                iters=10, h_deg=1e-3):
    """Zero-offset errors delta (degrees) of several joints (0-based indices) that best explain the measured
    tool points: minimise sum_k |p(q_k + delta) - p_meas_k|^2 by Gauss-Newton with a central-difference
    derivative.  The true link angle is q_servo - offset + delta, so the corrected calibration offset is
    offset - delta.  Returns dict delta_deg [len(joints)], rms_before_mm, rms_after_mm, new_offsets_deg [n]."""
    # TODO: residual(delta) = FK points with offsets[joints] -= delta, minus P (flattened, mm);
    #       Gauss-Newton: J by central differences (step h_deg), step = lstsq(J, -r); repeat
    raise NotImplementedError


def error_budget(M, Slist, theta, p_tool=(0.0, 0.0, 0.0), dtheta_deg=360.0 / 4096):
    """Tool-point error caused by small joint errors, joint by joint (revolute joints only).

    The axis of joint j at configuration theta is the home axis (omega_j, q_j) moved by
    T_{j-1} = e^{[S1]th1} ... e^{[S_{j-1}]th_{j-1}}; q_j = omega_j x v_j (point of the axis closest to the
    origin, |omega_j| = 1).  A rotation d theta_j moves the tool point p by |omega_j' x (p - q_j')| d theta_j.
    Returns dict lever_m [n] (m per rad), dp_mm [n] for dtheta_deg (scalar or [n]), worst_mm (sum), rss_mm."""
    # TODO: for each joint: home axis point q_j = omega_j x v_j; move omega_j and q_j with T_{j-1};
    #       lever = |omega_j' x (p - q_j')|; then update T = T @ screw_exp(S_j, theta_j)
    raise NotImplementedError
