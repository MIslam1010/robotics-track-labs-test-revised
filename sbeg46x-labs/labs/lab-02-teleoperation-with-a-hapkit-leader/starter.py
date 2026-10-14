"""SBEG 46X Lab 2 -- Teleoperation with a Hapkit leader: scaling, clutch, tremor filter, two-port models.
STARTER CODE -- you edit this file.  It becomes the module `teleop` of your library sbeg46x_medrob.
Write every function marked TODO; the helpers above Part A are provided (do not change them).
Run the checks with:  LAB_IMPL=starter python -m pytest -q test_lab.py

Conventions (lecture notes, Week 2):
  leader  : Z_m v_m = f_h + a_m          (Z_m = m_m s + b_m; f_h = hand force on the handle; a_m = leader motor force)
  follower: Z_s v_s = a_s - f_e          (f_e = force of the follower on the environment, f_e = Z_e v_e)
  hybrid matrix  [f_h, -v_e]^T = H [v_h, f_e]^T,  ideal (scaled) transparency H = [[0, beta], [-alpha, 0]]
  controller C(s) = b_c + k_c / s (PD on position, written as an impedance)
  P-P: a_s = C(alpha v_m - v_s),   a_m = -beta * a_s
  P-F: a_s = C(alpha v_m - v_s),   a_m = -beta * f_e   (follower force sensor)
All parameter values are illustrative.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

from course_models import eod_6r_ugv, hapkit  # noqa: F401  (re-exported for the tests)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


# =====================================================================  provided helpers (same in starter.py)
def load_params(path=DATA / "teleop_params.json") -> dict:
    """Read the lab parameters and return a flat dict p used by every function below."""
    with open(path) as f:
        P = json.load(f)
    p = dict(m_m=P["leader"]["m"], b_m=P["leader"]["b"], m_s=P["follower"]["m"], b_s=P["follower"]["b"],
             m_h=P["hand"]["m"], b_h=P["hand"]["b"], k_h=P["hand"]["k"],
             kc=P["controller"]["kc"], bc=P["controller"]["bc"], alpha=P["alpha"], beta=P["beta"],
             f_max=P["force_max"], x_range=P["x_range"], dt=P["dt"], substeps=P["substeps"])
    p["raw"] = P
    return p


def load_log(path=DATA / "leader_log.csv") -> dict:
    """Read a leader log (columns t, x_m, clutch) into numpy arrays."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return dict(t=np.array([float(r["t"]) for r in rows]), x_m=np.array([float(r["x_m"]) for r in rows]),
                clutch=np.array([int(r["clutch"]) for r in rows], bool))


def minjerk(t, t0, t1, x0, x1):
    """Minimum-jerk (quintic) rest-to-rest profile from x0 at t0 to x1 at t1."""
    s = np.clip((np.asarray(t, float) - t0) / (t1 - t0), 0.0, 1.0)
    return x0 + (x1 - x0) * (10 * s**3 - 15 * s**4 + 6 * s**5)


def reach_intent(p, t):
    """Intended hand trajectory of the bilateral reach task (Part D)."""
    rt = p["raw"]["reach_task"]
    return minjerk(t, 0.0, rt["t_move"], rt["x_start"], rt["x_goal"])


def simulate_teleop(arch, p, env, x_i, dt=None, substeps=None, saturate=True, watchdog=True, control=None):
    """Sampled-data simulation of the hand + leader + follower + environment (provided).

    arch     : 'pp' or 'pf'
    env      : dict with k, b, x_wall (unilateral contact: force only when x_s > x_wall)
    x_i      : intended hand position at each control sample (m); the hand is a spring-damper k_h, b_h
               pulling the handle towards x_i; its mass m_h is lumped with the handle
    control  : function control_forces(arch, xm, vm, xs, vs, fe, p) -> (a_m, a_s); default: this module's
    The controller runs at 1/dt with zero-order hold; the mechanics are integrated with `substeps`
    semi-implicit Euler steps per sample.  The Hapkit limits are applied as on the device: motor forces are
    saturated at f_max and the watchdog sets both motor forces to zero for the rest of the run if
    |x_m| > x_range (tripped = True).
    Returns dict t, x_m, v_m, x_s, v_s, f_h, f_e, a_m, a_s, tripped.
    """
    control = control or control_forces
    dt = dt or p["dt"]
    substeps = substeps or p["substeps"]
    x_i = np.asarray(x_i, float)
    N = x_i.size
    v_i = np.gradient(x_i, dt)
    h = dt / substeps
    mL = p["m_m"] + p["m_h"]
    xm, vm, xs, vs = float(x_i[0]), 0.0, p["alpha"] * float(x_i[0]), 0.0
    out = {k: np.zeros(N) for k in ("x_m", "v_m", "x_s", "v_s", "f_h", "f_e", "a_m", "a_s")}
    tripped = False

    def fenv(xs_, vs_):
        pen = xs_ - env["x_wall"]
        if pen <= 0.0 or env["k"] == 0.0:
            return 0.0
        return max(0.0, env["k"] * pen + env["b"] * vs_)

    for k in range(N):
        fe = fenv(xs, vs)
        am, as_ = control(arch, xm, vm, xs, vs, fe, p)
        if saturate:
            am = float(np.clip(am, -p["f_max"], p["f_max"]))
            as_ = float(np.clip(as_, -p["f_max"], p["f_max"]))
        if watchdog and (tripped or abs(xm) > p["x_range"]):
            tripped = True
            am = as_ = 0.0
        for _ in range(substeps):
            fh = p["k_h"] * (x_i[k] - xm) + p["b_h"] * (v_i[k] - vm)
            fe_c = fenv(xs, vs)
            vm += h * (fh + am - p["b_m"] * vm) / mL
            xm += h * vm
            vs += h * (as_ - p["b_s"] * vs - fe_c) / p["m_s"]
            xs += h * vs
        for key, val in zip(out, (xm, vm, xs, vs, p["k_h"] * (x_i[k] - xm) + p["b_h"] * (v_i[k] - vm),
                                  fenv(xs, vs), am, as_)):
            out[key][k] = val
    out["t"] = np.arange(N) * dt
    out["tripped"] = tripped
    return out


def _skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]], float)


def _se3(V):
    X = np.zeros((4, 4))
    X[:3, :3] = _skew(V[:3])
    X[:3, 3] = V[3:]
    return X


def _adjoint(T):
    R, p = T[:3, :3], T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, 3:] = R
    Ad[3:, :3] = _skew(p) @ R
    return Ad


def jacobian_body(Blist, q):
    """Body Jacobian J_b(theta) (MR Eq. 5.18), provided; your 361 library has the same function."""
    Blist = np.asarray(Blist, float)
    n = Blist.shape[1]
    J = np.zeros((6, n))
    T = np.eye(4)
    J[:, n - 1] = Blist[:, n - 1]
    for i in range(n - 2, -1, -1):
        T = T @ expm(-_se3(Blist[:, i + 1]) * q[i + 1])
        J[:, i] = _adjoint(T) @ Blist[:, i]
    return J


# =====================================================================  Part A: mapping, clutch, device safety
def hapkit_effective_mass(model) -> float:
    """A1. Effective translational mass of the Hapkit at the handle grip (kg):
    m_eff = (I_zz,com + m d^2 + I_rotor (rs/rp)^2) / rh^2, d = distance of the CoM from the pivot axis."""
    # TODO: write this function (see the lab handout).
    raise NotImplementedError("hapkit_effective_mass")


def saturate_force(f, f_max=2.0):
    """A2. Saturate a commanded handle force at +-f_max (N); works on scalars and arrays."""
    # TODO: write this function (see the lab handout).
    raise NotImplementedError("saturate_force")


def watchdog_ok(x, x_range=0.040):
    """A3. True where the handle is inside the allowed range |x| <= x_range (m)."""
    # TODO: write this function (see the lab handout).
    raise NotImplementedError("watchdog_ok")


def map_with_clutch(x_m, clutch, alpha, x_s0=0.0):
    """A4. Follower set-point with motion scaling and clutching (indexing).
    While clutch[k] is True the follower set-point holds.  At every release (and at k = 0 if not clutched)
    the offsets are re-latched: x_sd = x_s,latch + alpha (x_m - x_m,latch), so the follower never jumps."""
    # TODO: write this function (see the lab handout).
    raise NotImplementedError("map_with_clutch")


def rate_limit(x_d, v_max, dt):
    """A5. Limit the rate of change of a set-point to |dx/dt| <= v_max (first sample unchanged)."""
    # TODO: write this function (see the lab handout).
    raise NotImplementedError("rate_limit")


def clutch_strokes(travel, alpha, stroke):
    """A6. Minimum number of leader strokes (forward motions between clutch actions) for a follower travel."""
    # TODO: write this function (see the lab handout).
    raise NotImplementedError("clutch_strokes")


# =====================================================================  Part B: tremor filtering
def lowpass1(x, fc, dt):
    """B1. First-order low-pass, exact discretisation of 1/(tau s + 1), tau = 1/(2 pi fc):
    y_k = y_{k-1} + a (x_k - y_{k-1}), a = 1 - exp(-2 pi fc dt), y_0 = x_0."""
    # TODO: write this function (see the lab handout).
    raise NotImplementedError("lowpass1")


def butter2(fc, dt):
    """B2. Second-order Butterworth low-pass (bilinear transform with pre-warping).  Returns (b, a), a[0] = 1."""
    # TODO: write this function (see the lab handout).
    raise NotImplementedError("butter2")


def iir_filter(b, a, x):
    """B3. Apply a causal IIR filter (difference equation, a[0] = 1), starting at rest at x[0]
    (past inputs and outputs equal to x[0]; valid for a low-pass with unit DC gain)."""
    # TODO: write this function (see the lab handout).
    raise NotImplementedError("iir_filter")


def band_rms(x, dt, f_lo, f_hi):
    """B4. RMS of the part of x with frequency in [f_lo, f_hi] Hz.  First remove the straight line joining the
    first and last samples and the mean (so the record has no jump when the FFT repeats it), then use the
    one-sided FFT and Parseval: rms^2 = 2 sum_band |X_k|^2 / N^2."""
    # TODO: write this function (see the lab handout).
    raise NotImplementedError("band_rms")


def lag_estimate(x, y, dt, max_lag=0.5):
    """B5. Time (s) by which y lags x: the shift k*dt (k = 0 .. max_lag/dt) that minimises the mean squared
    difference between y[k:] and x[:N-k]."""
    # TODO: write this function (see the lab handout).
    raise NotImplementedError("lag_estimate")


# =====================================================================  Part C: two-port models
def hybrid_matrix(arch, s, p):
    """C1. Hybrid matrix H(s) of the scaled P-P or P-F architecture, [f_h, -v_e] = H [v_h, f_e].
    s may be a complex scalar or array; returns an array of shape s.shape + (2, 2)."""
    # TODO: write this function (see the lab handout).
    raise NotImplementedError("hybrid_matrix")


def transmitted_impedance(H, Ze):
    """C2. Impedance felt by the operator, Z_t = f_h / v_h, when the follower touches Z_e:
    Z_t = (h11 + (h11 h22 - h12 h21) Z_e) / (1 + h22 Z_e)."""
    # TODO: write this function (see the lab handout).
    raise NotImplementedError("transmitted_impedance")


def llewellyn(H):
    """C3. Llewellyn's third condition 2 Re h11 Re h22 - Re(h12 h21) - |h12 h21| (>= 0 needed for absolute
    stability, together with Re h11 >= 0, Re h22 >= 0 and stable h11, h22)."""
    # TODO: write this function (see the lab handout).
    raise NotImplementedError("llewellyn")


# =====================================================================  Part D: bilateral control in time
def control_forces(arch, xm, vm, xs, vs, fe, p):
    """D1. Motor forces (a_m on the leader, a_s on the follower) before saturation.
    a_s = k_c (alpha x_m - x_s) + b_c (alpha v_m - v_s);  P-P: a_m = -beta a_s;  P-F: a_m = -beta f_e."""
    # TODO: write this function (see the lab handout).
    raise NotImplementedError("control_forces")


def closed_loop_matrix(arch, p, ke, be):
    """D2. A matrix of the linear in-contact model, state (x_m, v_m, x_s, v_s), wall at x = 0, hand intent 0:
    (m_m + m_h) dv_m = -k_h x_m - (b_h + b_m) v_m + a_m,   m_s dv_s = a_s - b_s v_s - (k_e x_s + b_e v_s)."""
    # TODO: write this function (see the lab handout).
    raise NotImplementedError("closed_loop_matrix")


def max_stable_beta(arch, p, ke, be, beta_hi=1000.0):
    """D3. Largest force scale beta for which the in-contact linear model is stable (max Re(eig) < 0).
    Returns inf if it is stable up to beta_hi."""
    # TODO: write this function (see the lab handout).
    raise NotImplementedError("max_stable_beta")


# =====================================================================  Part E: EOD follower speed limit
def eod_approach_rates(model, q):
    """E1. Joint rates (rad/s) that move the gripper at 1 m/s along its approach axis x_b with no rotation:
    theta_dot = J_b^{-1} (0, 0, 0, 1, 0, 0)."""
    # TODO: write this function (see the lab handout).
    raise NotImplementedError("eod_approach_rates")


def max_approach_speed(model, q, speed_fraction=1.0):
    """E2. Largest approach speed (m/s) for which every joint stays within speed_fraction * qdot_max."""
    # TODO: write this function (see the lab handout).
    raise NotImplementedError("max_approach_speed")
