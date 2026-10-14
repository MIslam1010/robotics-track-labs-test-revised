"""SBEG 46X Lab 10 -- CBF-QP safety filter versus saturation on an exoskeleton joint and on the Hapkit.
STARTER CODE -- you edit this file (rename it lab10_<ID>.py when you submit).  Write every function marked TODO
(keep the names and signatures); the "Provided" section is given (do not change it).  Its functions join the module `safety` of the library sbeg46x_medrob (and are ported to the course-project
package: cbf_filter_1d and cbf_qp keep the project signatures).

Exoskeleton joint (MR Sec. 11.4.1, Eq. 11.21, angle from the hanging position, as in Lab 8):
    M theta_ddot + b theta_dot + g(theta) = tau + tau_h,   g(theta) = g_s sin(theta) + g_c cos(theta),
    |tau| <= tau_max  (actuator),   tau_h = torque of the patient/wearer on the joint (cuff force sensor).
Hapkit handle (x = r_h theta):  m_h x_ddot + b_h x_dot + k_g x = F + F_h,  |F| <= 2 N,  watchdog at |x| > 40 mm.
Safe sets:  joint range  h_hi = theta_max - theta >= 0,  h_lo = theta - theta_min >= 0;
            forbidden region of the Hapkit  h = x_max - x >= 0.
Barrier condition (first order): h_dot >= -alpha h.  Second order (relative degree two, exponential CBF):
    psi1 = h_dot + alpha1 h >= 0,   psi2 = psi1_dot + alpha2 psi1 = h_ddot + (alpha1 + alpha2) h_dot + alpha1 alpha2 h >= 0.
One-input CBF-QP  min (u - u_nom)^2  s.t.  a u >= b   ->   u* = u_nom + max(0, b - a u_nom) / a  (a != 0).
References: Tedrake, Underactuated Robotics, Sec. 9.1.6 (barrier functions); MR Sec. 11.4.1 (Eq. 11.21),
Sec. 11.7 (impedance, Eqs. 11.62-11.65); Ames et al., ECC 2019 (not in the course library).
Units: SI.  All values are illustrative.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


# ==========================================================================================
# Provided (do not edit): parameters, gravity, references, patient and operator models, baselines,
# a small QP solver and the simulators
# ==========================================================================================
def load_params(path=DATA / "lab10.json") -> dict:
    """Lab parameters (data/lab10.json, illustrative values): 'elbow', 'knee' (optional) and 'hapkit'."""
    with open(path) as f:
        return json.load(f)


def gravity(theta: float, j: dict) -> float:
    """g(theta) = g_s sin(theta) + g_c cos(theta)  (N m)."""
    return j["g_s"] * math.sin(theta) + j["g_c"] * math.cos(theta)


def step_setpoint(t: float, task: dict) -> float:
    """Piecewise-constant set-point: task['steps'] = [[t_0, theta_0], [t_1, theta_1], ...]."""
    th = task["steps"][0][1]
    for t_k, th_k in task["steps"]:
        if t >= t_k - 1e-12:
            th = th_k
    return float(th)


def therapy_reference(t: float, ref: dict):
    """Slow flexion-extension therapy movement theta_d = mid - amp cos(2 pi t / period) and its rate."""
    w = 2 * math.pi / ref["period"]
    return ref["mid"] - ref["amp"] * math.cos(w * t), ref["amp"] * w * math.sin(w * t)


def smooth_pulse(t: float, amp: float, t_on: float, t_off: float, ramp: float) -> float:
    """Pulse of height amp on [t_on, t_off] with raised-cosine ramps of length ramp."""
    if t <= t_on or t >= t_off + ramp:
        return 0.0
    if t < t_on + ramp:
        return amp * 0.5 * (1 - math.cos(math.pi * (t - t_on) / ramp))
    if t <= t_off:
        return amp
    return amp * 0.5 * (1 + math.cos(math.pi * (t - t_off) / ramp))


def patient_torque(t, theta, omega, task) -> float:
    """Torque of the patient on the elbow: a fraction c of a voluntary tracking effort plus a flexor spasm."""
    p, sp = task["patient"], task["spasm"]
    th_d, om_d = therapy_reference(t, task["ref"])
    vol = p["c"] * (p["Kp"] * (th_d - theta) + p["Bp"] * (om_d - omega))
    return vol + smooth_pulse(t, sp["amp"], sp["t_on"], sp["t_off"], sp["ramp"])


def operator_target(t: float, op: dict, x_int: float) -> float:
    """Intended hand position: rests at x0, moves towards x_int at op['speed'] from op['t0'], then holds."""
    if t <= op["t0"]:
        return op["x0"]
    return min(x_int, op["x0"] + op["speed"] * (t - op["t0"]))


def saturate_toward_limit(x, u, x_min=-np.inf, x_max=np.inf):
    """BASELINE 'saturation' (as in the course-project package): a command u that would push x further outside
    [x_min, x_max] is set to zero once x has reached the limit.  It acts only AT the limit."""
    if (x >= x_max and u > 0) or (x <= x_min and u < 0):
        return 0.0
    return u


def virtual_wall_force(x, v, x_max, K=200.0, B=0.0):
    """BASELINE virtual wall (Week 4, Lab 4): a one-sided spring-damper that pushes back once x > x_max."""
    if x > x_max:
        return -K * (x - x_max) - B * max(v, 0.0)
    return 0.0


def solve_qp(H, f, A_ub=None, b_ub=None, lb=None, ub=None, x0=None):
    """min 1/2 z'Hz + f'z  s.t.  A_ub z <= b_ub,  lb <= z <= ub  (SciPy SLSQP; the same helper as the
    course-project optctrl.solve_qp).  Returns (z, info) with info = dict(success, fun, nit)."""
    H = np.atleast_2d(np.asarray(H, float))
    f = np.asarray(f, float).ravel()
    n = len(f)
    cons = []
    if A_ub is not None:
        A = np.atleast_2d(np.asarray(A_ub, float))
        b = np.asarray(b_ub, float).ravel()
        cons.append(dict(type="ineq", fun=lambda z: b - A @ z, jac=lambda z: -A))
    lb = -np.inf * np.ones(n) if lb is None else np.broadcast_to(np.asarray(lb, float), (n,))
    ub = np.inf * np.ones(n) if ub is None else np.broadcast_to(np.asarray(ub, float), (n,))
    bounds = [(None if not np.isfinite(l) else l, None if not np.isfinite(u) else u) for l, u in zip(lb, ub)]
    z0 = np.clip(np.zeros(n) if x0 is None else np.asarray(x0, float), np.where(np.isfinite(lb), lb, -1e9),
                 np.where(np.isfinite(ub), ub, 1e9))
    res = minimize(lambda z: 0.5 * z @ H @ z + f @ z, z0, jac=lambda z: H @ z + f, bounds=bounds,
                   constraints=cons, method="SLSQP", options=dict(maxiter=200, ftol=1e-12))
    return res.x, dict(success=bool(res.success), fun=float(res.fun), nit=int(res.nit))


def end_stop_torque(theta, omega, j) -> float:
    """Mechanical end stops of the joint at j['mech_limits'] (stiff one-sided spring-damper j['stop'])."""
    lo, hi = j["mech_limits"]
    K, B = j["stop"]["K"], j["stop"]["B"]
    if theta > hi:
        return -K * (theta - hi) - B * max(omega, 0.0)
    if theta < lo:
        return -K * (theta - lo) - B * min(omega, 0.0)
    return 0.0


def pd_torque(theta, omega, th_d, j, pd) -> float:
    """Nominal PD + gravity compensation (MR Eq. 11.38 with K_i = 0), gains from wn and zeta (Lab 8)."""
    Kp = j["M"] * pd["wn"] ** 2
    Kd = 2 * pd["zeta"] * pd["wn"] * j["M"] - j["b"]
    return gravity(theta, j) + Kp * (th_d - theta) - Kd * omega


def simulate_joint(j: dict, experiment: str, mode: str, tau_h_mode: str = "measured", alphas=None) -> dict:
    """Closed-loop run of one exoskeleton joint.  The controller runs every dt (zero-order hold); the plant
    M theta'' + b theta' + g = tau + tau_h + tau_stop (mechanical end stops) is integrated with n_sub semi-implicit Euler sub-steps.
      experiment  'step'  : nominal = PD + gravity compensation to the set-points of j['step'] (tau_h = 0)
                  'spasm' : nominal = gravity compensation + assist-as-needed (aan_torque) around the therapy
                            reference, with the patient torque of patient_torque (voluntary effort + spasm)
      mode        'none'  : nominal command, clipped at +-tau_max (and at the torque-rate limit if any)
                  'sat'   : as 'none' + set-point clipped to the ROM + saturate_toward_limit on u = tau - g
                  'cbf'   : nominal command through joint_safety_filter (ROM barriers + actuator limits)
      tau_h_mode  what the filter knows about tau_h: 'measured' (cuff sensor), 'zero' (ignored),
                  'robust' (worst case |tau_h| <= tau_h_max of the task).
    Returns a log dict of numpy arrays: t, theta, omega, tau, tau_nom, tau_h, th_d, active, feasible."""
    task = j[experiment]
    dt, n_sub = j["dt"], j["n_sub"]
    a1, a2 = alphas if alphas is not None else (j["cbf"]["alpha1"], j["cbf"]["alpha2"])
    lo, hi = j["rom"]
    th, om = task["theta0"], 0.0
    tau_prev = gravity(th, j)
    n = int(round(task["T"] / dt))
    log = {k: np.zeros(n) for k in ("t", "theta", "omega", "tau", "tau_nom", "tau_h", "th_d")}
    log["active"] = np.zeros(n, bool)
    log["feasible"] = np.ones(n, bool)
    for k in range(n):
        t = k * dt
        if experiment == "step":
            th_d, om_d = step_setpoint(t, task), 0.0
            if mode == "sat":
                th_d = min(max(th_d, lo), hi)
            tau_nom = pd_torque(th, om, th_d, j, task["pd"])
            tau_h_meas = 0.0
            tau_h_bound = 0.0
        else:
            th_d, om_d = therapy_reference(t, task["ref"])
            if mode == "sat":
                th_d = min(max(th_d, lo), hi)
            a = task["aan"]
            tau_nom = gravity(th, j) + aan_torque(th - th_d, om - om_d, a["K"], a["B"], a["delta"])
            tau_h_meas = patient_torque(t, th, om, task)
            tau_h_bound = task["tau_h_max"]
        act_lo, act_hi = -j["tau_max"], j["tau_max"]
        if j.get("tau_rate"):
            act_lo = max(act_lo, tau_prev - j["tau_rate"] * dt)
            act_hi = min(act_hi, tau_prev + j["tau_rate"] * dt)
        active, feasible = False, True
        if mode == "cbf":
            if tau_h_mode == "measured":
                th_lo_hi = rom_torque_bounds(th, om, j, a1, a2, tau_h=tau_h_meas)
            elif tau_h_mode == "zero":
                th_lo_hi = rom_torque_bounds(th, om, j, a1, a2, tau_h=0.0)
            else:   # worst case for each barrier: the upper one with +bound, the lower one with -bound
                up = rom_torque_bounds(th, om, j, a1, a2, tau_h=tau_h_bound)[1]
                dn = rom_torque_bounds(th, om, j, a1, a2, tau_h=-tau_h_bound)[0]
                th_lo_hi = (dn, up)
            tau, info = joint_safety_filter(tau_nom, th_lo_hi, (act_lo, act_hi))
            active, feasible = info["active"], info["feasible"]
        else:
            tau = tau_nom
            if mode == "sat":
                g_hat = gravity(th, j)
                tau = g_hat + saturate_toward_limit(th, tau - g_hat, lo, hi)
            tau = min(max(tau, act_lo), act_hi)
        for _ in range(n_sub):      # plant (true patient torque evaluated at every sub-step)
            ts = t
            tau_h = 0.0 if experiment == "step" else patient_torque(ts, th, om, task)
            acc = (tau + tau_h - j["b"] * om - gravity(th, j) + end_stop_torque(th, om, j)) / j["M"]
            om += acc * dt / n_sub
            th += om * dt / n_sub
        tau_prev = tau
        for key, val in (("t", t + dt), ("theta", th), ("omega", om), ("tau", tau), ("tau_nom", tau_nom),
                         ("tau_h", tau_h_meas), ("th_d", th_d)):
            log[key][k] = val
        log["active"][k], log["feasible"][k] = active, feasible
    return log


def simulate_hapkit(hk: dict, case: str, mode: str, alphas=None) -> dict:
    """Hapkit forbidden region (x <= x_max) with an operator pushing towards x_int (the handle is held).
    Firmware every T: quantised position, filtered backward-difference velocity, controller, saturation at
    +-force_max, watchdog (motor off, latched) outside +-x_range.  Plant: (m_h + m_op) x'' = F - b_h x' - k_g x
    + k_op (x_target - x) - b_op x'  (the hand is part of the moving mass).
      case  'push' or 'hard_push' (operator target inside the reach of the 2 N motor, or beyond it)
      mode  'none' (transparent, F = 0) | 'wall' (virtual_wall_force, K of hk['wall']) | 'cbf' (hapkit_safety_filter
            with the worst-case operator force F_h_max of the case).
    Returns log dict: t, x, v, F, F_h, active, feasible, tripped."""
    c = hk[case]
    op = hk["operator"]
    a1, a2 = alphas if alphas is not None else (hk["cbf"]["alpha1"], hk["cbf"]["alpha2"])
    T, n_sub = hk["T"], 4
    x = v = 0.0
    x_meas_prev = 0.0
    v_est = 0.0
    beta = math.exp(-2 * math.pi * hk["v_filter_hz"] * T)
    tripped = False
    n = int(round(c["T"] / T))
    log = {k: np.zeros(n) for k in ("t", "x", "v", "F", "F_h")}
    for k in ("active", "feasible", "tripped"):
        log[k] = np.zeros(n, bool)
    m_tot = hk["m_h"] + op["m"]
    for k in range(n):
        t = k * T
        x_meas = round(x / hk["x_res"]) * hk["x_res"]
        v_raw = (x_meas - x_meas_prev) / T if k > 0 else 0.0
        v_est = beta * v_est + (1 - beta) * v_raw
        x_meas_prev = x_meas
        active, feasible = False, True
        if mode == "none":
            F = 0.0
        elif mode == "wall":
            F = virtual_wall_force(x_meas, v_est, hk["x_max"], hk["wall"]["K"], hk["wall"]["B"])
        else:
            F, info = hapkit_safety_filter(0.0, x_meas, v_est, hk, a1, a2, c["F_h_max"])
            active, feasible = info["active"], info["feasible"]
        F = min(max(F, -hk["force_max"]), hk["force_max"])
        if abs(x_meas) > hk["x_range"]:
            tripped = True
        if tripped:
            F = 0.0
        x_tgt = operator_target(t, op, c["x_int"])
        for _ in range(n_sub):
            F_h = op["k"] * (x_tgt - x) - op["b"] * v
            acc = (F - hk["b_h"] * v - hk["k_g"] * x + F_h) / m_tot
            v += acc * T / n_sub
            x += v * T / n_sub
        for key, val in (("t", t + T), ("x", x), ("v", v), ("F", F), ("F_h", F_h)):
            log[key][k] = val
        log["active"][k], log["feasible"][k], log["tripped"][k] = active, feasible, tripped
    return log


# ==========================================================================================
# Part A -- one-input CBF-QP filters
# ==========================================================================================
def cbf_filter_1d(u_nom, h, Lf_h, Lg_h, alpha):
    """Closed-form CBF-QP with one input (course-project signature):
        min (u - u_nom)^2   s.t.   Lf_h + Lg_h u >= -alpha h.
    If Lg_h = 0 the input cannot change h_dot: return u_nom unchanged.  Otherwise return u_nom when it already
    satisfies the condition, else the input that makes the condition hold with equality."""
    # TODO: psi = Lf_h + Lg_h u_nom + alpha h; if psi >= 0 (or Lg_h ~ 0) keep u_nom, else move u along Lg_h
    #       until the condition holds with equality (u = u_nom - psi / Lg_h).
    raise NotImplementedError("cbf_filter_1d: TODO")


def filter_interval(u_nom, lo, hi):
    """1-D QP  min (u - u_nom)^2  s.t.  lo <= u <= hi  (lo, hi may be +-inf).
    Returns (u, active, feasible): u = clip(u_nom, lo, hi); active = (u != u_nom); feasible = (lo <= hi).
    When lo > hi the set is empty: return hi (the upper bound has priority) with feasible = False."""
    # TODO: clip u_nom to [lo, hi]; handle lo > hi as described in the docstring.
    raise NotImplementedError("filter_interval: TODO")


def cbf_qp(u_nom, A, b, W=None, u_min=None, u_max=None):
    """CBF-QP with several inputs and constraints (course-project signature):
        min (u - u_nom)' W (u - u_nom)   s.t.   A u >= b  (one row per barrier),  u_min <= u <= u_max.
    Use the provided solve_qp (H = 2W, f = -2 W u_nom, A_ub = -A, b_ub = -b).  Returns u as a 1-D array."""
    # TODO: build H = 2W, f = -2 W u_nom and call solve_qp(H, f, -A, -b, u_min, u_max, x0=u_nom).
    raise NotImplementedError("cbf_qp: TODO")


# ==========================================================================================
# Part B -- exoskeleton joint: range-of-motion barriers (relative degree two), assist-as-needed
# ==========================================================================================
def rom_torque_bounds(theta, omega, j, alpha1, alpha2, tau_h=0.0):
    """Second-order (exponential) CBF for the joint range [theta_min, theta_max] = j['rom'], written as bounds
    on the motor torque tau of  M theta'' + b theta' + g(theta) = tau + tau_h:
        upper barrier h = theta_max - theta:  theta'' <= alpha1 alpha2 (theta_max - theta) - (alpha1 + alpha2) omega
        lower barrier h = theta - theta_min:  theta'' >= -alpha1 alpha2 (theta - theta_min) - (alpha1 + alpha2) omega
    Returns (tau_lo, tau_hi)."""
    # TODO: write the two acceleration bounds of the docstring, then tau = M theta'' + b omega + g(theta) - tau_h.
    raise NotImplementedError("rom_torque_bounds: TODO")


def joint_safety_filter(tau_nom, cbf_bounds, act_bounds):
    """Combine the barrier bounds (tau_lo, tau_hi) with the actuator bounds (act_lo, act_hi) and filter tau_nom.
    Actuator bounds are physical and always hold.  If the combined set is empty, keep the actuator-feasible
    torque closest to the barrier interval (upper barrier first), i.e.
        u_c = min(max(tau_nom, tau_lo), tau_hi),   tau = clip(u_c, act_lo, act_hi).
    Returns (tau, info) with info = dict(active=..., feasible=...)."""
    # TODO: intersect the two intervals; if non-empty use filter_interval; else follow the docstring rule.
    raise NotImplementedError("joint_safety_filter: TODO")


def aan_torque(e, e_dot, K, B, delta):
    """Assist-as-needed torque with a dead-band ('virtual tunnel') of half-width delta around the reference:
        tau_a = -K dz(e) - B e_dot,   dz(e) = sign(e) max(|e| - delta, 0).
    No stiffness inside the tunnel (the patient moves freely), a spring outside it."""
    # TODO: dead-band dz(e), then -K dz(e) - B e_dot.
    raise NotImplementedError("aan_torque: TODO")


def violation_metrics(t, s, lo, hi):
    """Range violation of a signal s(t) (theta or x) for the limits [lo, hi] (use -inf/inf for one-sided):
    returns dict(max_violation = max(0, max(s - hi), max(lo - s)), time_outside = total time with s outside,
    s_max, s_min).  t is uniformly sampled."""
    # TODO: over = max(s - hi, lo - s) sample by sample; dt = t[1] - t[0].
    raise NotImplementedError("violation_metrics: TODO")


# ==========================================================================================
# Part C -- Hapkit forbidden region
# ==========================================================================================
def hapkit_force_bound(x, v, hk, alpha1, alpha2, F_h_max):
    """Upper bound on the motor force F from the second-order CBF of the forbidden region h = x_max - x >= 0
    for  m_h x'' + b_h x' + k_g x = F + F_h  with an unmeasured operator force |F_h| <= F_h_max (worst case
    F_h = +F_h_max pushes into the region):
        F <= m_h [alpha1 alpha2 (x_max - x) - (alpha1 + alpha2) v] + b_h v + k_g x - F_h_max."""
    # TODO: the same steps as rom_torque_bounds (upper barrier only) for the handle model.
    raise NotImplementedError("hapkit_force_bound: TODO")


def hapkit_safety_filter(F_nom, x, v, hk, alpha1, alpha2, F_h_max):
    """Filter the nominal handle force: F <= hapkit_force_bound(...), |F| <= force_max.
    Returns (F, info) like joint_safety_filter."""
    # TODO: call hapkit_force_bound, then joint_safety_filter with (-inf, F_hi) and (-force_max, force_max).
    raise NotImplementedError("hapkit_safety_filter: TODO")


if __name__ == "__main__":
    # Prints the runs of the procedure once your TODO functions work.
    # Check your functions with:  LAB_IMPL=starter python -m pytest -q test_lab.py
    P = load_params()
    E = P["elbow"]
    for exp, modes in (("step", [("none", "measured"), ("sat", "measured"), ("cbf", "measured")]),
                       ("spasm", [("sat", "measured"), ("cbf", "measured"), ("cbf", "zero"), ("cbf", "robust")])):
        for mode, thm in modes:
            log = simulate_joint(E, exp, mode, thm)
            m = violation_metrics(log["t"], log["theta"], *E["rom"])
            print(f"elbow {exp:6s} {mode:4s} {thm:8s} th_max={m['s_max']:.4f} viol={m['max_violation']:.4f} "
                  f"t_out={m['time_outside']:.3f} |tau|max={np.max(np.abs(log['tau'])):.3f} "
                  f"infeasible={int(np.sum(~log['feasible']))} |tau_h|max={np.max(np.abs(log['tau_h'])):.3f}")
    K = P["knee"]
    for mode in ("none", "sat", "cbf"):
        log = simulate_joint(K, "step", mode)
        m = violation_metrics(log["t"], log["theta"], *K["rom"])
        print(f"knee  step   {mode:4s} th_max={m['s_max']:.4f} viol={m['max_violation']:.4f} "
              f"|tau|max={np.max(np.abs(log['tau'])):.1f} infeasible={int(np.sum(~log['feasible']))}")
    H = P["hapkit"]
    for case in ("push", "hard_push"):
        for mode in ("none", "wall", "cbf"):
            log = simulate_hapkit(H, case, mode)
            m = violation_metrics(log["t"], log["x"], -np.inf, H["x_max"])
            print(f"hapkit {case:9s} {mode:4s} x_max={1e3 * m['s_max']:.3f} mm viol={1e3 * m['max_violation']:.3f} mm "
                  f"|F|max={np.max(np.abs(log['F'])):.3f} |F_h|max={np.max(np.abs(log['F_h'])):.3f} "
                  f"infeasible={int(np.sum(~log['feasible']))} tripped={bool(log['tripped'][-1])}")
