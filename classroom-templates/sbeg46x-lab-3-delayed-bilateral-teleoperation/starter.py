"""SBEG 46X Lab 3 -- delayed bilateral teleoperation: instability, wave variables, time-domain passivity.
STARTER CODE -- you edit this file.  Its functions join the module `teleop` of your library sbeg46x_medrob.
Write every function marked TODO (keep the names and signatures); the helpers in the 'Provided' section are
given (do not change them).  Run the checks with:  LAB_IMPL=starter python -m pytest -q test_lab.py

Leader and follower are Hapkit-type 1-DOF devices (course_models.hapkit, numbers of the shared course robot
models).  Units: SI (s, m, kg, N).  Velocities are written xdot (xdot_m, xdot_s; vm, vs inside simulate());
the letters u and v are the forward and returning WAVES, as in the lecture notes.

Sign conventions used everywhere in this file
  * Leader (the operator's hand mass m_h is lumped with the handle mass m_m):
        (m_m + m_h) xddot_m = f_h - F_m - b_m xdot_m,   f_h = k_h (x_i - x_m) + b_h (xdot_i - xdot_m)
    F_m is the force of the teleoperator on the leader; it RESISTS the operator when F_m > 0 and xdot_m > 0.
    Power INTO the communication channel at the leader port:  P_m = F_m xdot_m.
  * Follower:  m_s xddot_s = F_c - f_e - b_s xdot_s,   F_c = Kp (x_d - x_s) - Kd xdot_s.
    x_d is the follower set-point delivered by the channel.  Power OUT of the channel at the follower port:
    P_s = F_c xdot_d (xdot_d = rate of change of x_d).
  * Channel energy  E_ch = E_in,m + E_in,s - E_out,m - E_out,s;  E_ch >= 0 at all times <=> passive channel.

  * Week 2 sign: Lab 2 writes the leader motor force a_m = -F_m and the follower motor force a_s = F_c.

Architectures (arch = "pp", "pf", "wave", "tdpa")
  pp    position-position (the follower controller force is reflected):
        x_d(t) = x_m(t - T),  F_m(t) = F_c(t - T)                         [Week 2 P-P with alpha = beta = 1]
  pf    position-force (the follower force sensor is reflected):
        x_d(t) = x_m(t - T),  F_m(t) = f_e(t - T)                         [Week 2 P-F; Anderson & Spong 1989]
  wave  wave variables with wave impedance b (Niemeyer & Slotine 1991):
        u = (b xdot + F)/sqrt(2b),  v = (b xdot - F)/sqrt(2b),  u_s(t) = u_m(t - T),  v_m(t) = v_s(t - T)
  tdpa  the pp channel plus a passivity observer and a passivity controller at each port
        (Hannaford & Ryu 2002, applied to the two energy flows of the delayed channel):
        E_out,s(n) <= E_in,m(n - N)   and   E_out,m(n) <= E_in,s(n - N)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


# ==========================================================================================
# Provided (do not edit): parameters, hand intent, network model, simulator
# ==========================================================================================
def load_params(path=DATA / "teleop.json") -> dict:
    """Read the lab parameters (data/teleop.json, illustrative values; the same devices, hand, gains,
    environments and reach task as Lab 2)."""
    with open(path) as f:
        return json.load(f)


def minjerk(t, t0, t1, x0, x1):
    """Minimum-jerk (quintic) rest-to-rest profile from x0 at t0 to x1 at t1 (as in Lab 2)."""
    s = np.clip((np.asarray(t, float) - t0) / (t1 - t0), 0.0, 1.0)
    return x0 + (x1 - x0) * (10 * s**3 - 15 * s**4 + 6 * s**5)


def reach_intent(p: dict, t):
    """Intended hand trajectory: reach from x_start to x_goal in t_move s, then hold (pushes into the wall)."""
    rt = p["reach_task"]
    return minjerk(t, 0.0, rt["t_move"], rt["x_start"], rt["x_goal"])


class DelayLine:
    """One direction of a packet network, sampled at the control period dt.

    A packet sent at sample n arrives at sample n + d[n] (d[n] >= 1; d[n] < 0 means the packet is lost).
    The receiver uses the most recently SENT packet that has arrived (older, out-of-order packets are
    discarded) and holds it until a newer one arrives: the usual zero-order hold of a teleoperation link.
    Before the first packet arrives, receive() returns `initial`.
    """

    def __init__(self, delays, n_steps: int, initial):
        d = np.full(n_steps, int(delays)) if np.isscalar(delays) else np.asarray(delays, int)
        if len(d) < n_steps:
            raise ValueError("need one delay per sample")
        if np.any((d >= 0) & (d < 1)):
            raise ValueError("delays must be >= 1 sample (or < 0 for a lost packet)")
        self.d = d
        self.initial = np.array(initial, float)
        self.values = {}
        self.arrivals = {}
        self.best = -1

    def send(self, n: int, value):
        self.values[n] = np.array(value, float)
        if self.d[n] >= 0:
            self.arrivals.setdefault(n + int(self.d[n]), []).append(n)

    def receive(self, n: int) -> np.ndarray:
        for s in self.arrivals.pop(n, []):
            self.best = max(self.best, s)
        return self.initial.copy() if self.best < 0 else self.values[self.best].copy()


def network_delays(T: float, dt: float, n_steps: int, jitter: float = 0.0, loss: float = 0.0, seed: int = 3):
    """Per-packet delays in samples: mean one-way delay T (s), uniform jitter +-jitter (s), loss probability.
    The minimum is one sample (a digital link always delays by at least one control period)."""
    rng = np.random.default_rng(seed)
    base = T / dt + (rng.uniform(-jitter / dt, jitter / dt, n_steps) if jitter > 0 else np.zeros(n_steps))
    d = np.maximum(1, np.rint(base)).astype(int)
    if loss > 0:
        d[rng.random(n_steps) < loss] = -1
    return d


def contact_force(x, xdot, env: dict) -> float:
    """Unilateral Kelvin-Voigt contact (never pulls): f_e = max(0, k (x - x_wall) + b xdot) for x > x_wall."""
    pen = x - env["x_wall"]
    if pen <= 0.0 or env["k"] == 0.0:
        return 0.0
    return max(0.0, env["k"] * pen + env["b"] * xdot)


ARCHS = ("pp", "pf", "wave", "tdpa")


def simulate(arch: str, T: float, p: dict | None = None, env: str = "tissue", t_end: float | None = None,
             b_wave: float | None = None, jitter: float = 0.0, loss: float = 0.0, seed: int = 3,
             saturate: bool = True, watchdog: bool = True) -> dict:
    """Sampled-data simulation of hand + leader + delayed channel + follower + environment (provided).

    arch   "pp"   position-position: x_d(t) = x_m(t - T), F_m(t) = F_c(t - T) (follower controller force)
           "pf"   position-force:    x_d(t) = x_m(t - T), F_m(t) = f_e(t - T) (follower force sensor)
           "wave" wave variables (impedance b_wave, default p["wave"]["b"]); the follower integrates the
                  decoded velocity command; the wave arriving at the leader is low-pass filtered (time constant
                  p["wave"]["tau_f"]): a filter with |H(jw)| <= 1 cannot add wave energy, and it removes the
                  undamped wave reflections at 1/(4T) that grow at very short delays
           "tdpa" the "pp" channel with a passivity observer and controller at both ports
    T      one-way delay (s), equal in both directions; jitter (s) and loss (probability) per packet.
    The controller runs at 1/dt with zero-order hold; the mechanics are integrated with `substeps`
    semi-implicit Euler steps per sample.  As on the course Hapkits, motor forces are saturated at force_max
    (saturate=True) and the watchdog switches both motor forces off for the rest of the run if a handle leaves
    +-x_range (watchdog=True; delay is emulated in software, never by disabling the watchdog).
    Returns a dict of arrays (one value per control sample): t, x_i (hand intent), x_m, xdot_m, x_s, xdot_s,
    x_d, F_m, F_c, f_e, P_m, P_s, E_ch, W_m, W_s, r_m, r_s, and the scalars tripped, N (delay in samples).
    """
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {ARCHS}")
    p = load_params() if p is None else p
    dt, nsub = p["dt"], p["substeps"]
    hsub = dt / nsub
    L, F, H, C = p["leader"], p["follower"], p["hand"], p["controller"]
    E = p["environments"][env] if isinstance(env, str) else env
    b = p["wave"]["b"] if b_wave is None else b_wave
    t_end = p["reach_task"]["t_end"] if t_end is None else t_end
    n = int(round(t_end / dt))
    t = np.arange(n) * dt
    x_i = reach_intent(p, t)
    v_i = np.gradient(x_i, dt)
    mL = L["m"] + H["m"]
    fmax = p["force_max"]
    sat = (lambda f: float(np.clip(f, -fmax, fmax))) if saturate else float
    fwd = DelayLine(network_delays(T, dt, n, jitter, loss, seed), n, [0.0, 0.0, 0.0])      # x_m, u_m, E_in,m
    bwd = DelayLine(network_delays(T, dt, n, jitter, loss, seed + 1), n, [0.0, 0.0, 0.0])  # F_back, v_s, E_in,s
    keys = "x_m xdot_m x_s xdot_s x_d F_m F_c f_e P_m P_s E_ch W_m W_s r_m r_s".split()
    out = {k: np.zeros(n) for k in keys}
    xm = vm = xs = vs = xd = 0.0
    x_recv_prev = 0.0
    E_in_m = E_out_m = E_in_s = E_out_s = 0.0
    v_f = 0.0
    a_f = dt / (p["wave"]["tau_f"] + dt)
    tripped = False
    for k in range(n):
        rf = fwd.receive(k)                      # arrives at the follower
        rb = bwd.receive(k)                      # arrives at the leader
        fe = contact_force(xs, vs, E)
        # ------------------------------------------------------------------ follower side
        F_c = sat(follower_pd(xd, xs, vs, C["Kp"], C["Kd"]))
        r_s, v_back = 1.0, 0.0
        if arch == "wave":
            xdot_d, v_back = wave_follower(rf[1], F_c, b)
            dx = xdot_d * dt
        else:
            dx = rf[0] - x_recv_prev
            x_recv_prev = rf[0]
            if arch == "tdpa":
                dx, E_in_s, E_out_s, r_s = tdpa_follower(F_c, dx, E_in_s, E_out_s, rf[2])
        P_s = F_c * dx / dt
        if arch != "tdpa":
            E_in_s += max(-P_s, 0.0) * dt
            E_out_s += max(P_s, 0.0) * dt
        # ------------------------------------------------------------------ leader side
        r_m, u_m = 1.0, 0.0
        if arch == "wave":
            v_f += a_f * (rb[1] - v_f)            # passive first-order low-pass on the arriving wave
            F_m, u_m = wave_leader(vm, v_f, b)
            F_m = sat(F_m)
        else:
            F_m = sat(rb[0])
            if arch == "tdpa":
                F_m, E_in_m, E_out_m, r_m = tdpa_leader(F_m, vm, dt, E_in_m, E_out_m, rb[2])
        P_m = F_m * vm
        if arch != "tdpa":
            E_in_m += max(P_m, 0.0) * dt
            E_out_m += max(-P_m, 0.0) * dt
        # ------------------------------------------------------------------ network
        fwd.send(k, [xm, u_m, E_in_m])
        bwd.send(k, [fe if arch == "pf" else F_c, v_back, E_in_s])
        # ------------------------------------------------------------------ watchdog and log
        if watchdog and (tripped or max(abs(xm), abs(xs)) > p["x_range"]):
            tripped = True
        for key, val in zip(keys, [xm, vm, xs, vs, xd, F_m, F_c, fe, P_m, P_s,
                                   E_in_m + E_in_s - E_out_m - E_out_s,
                                   rb[2] - E_out_m, rf[2] - E_out_s, r_m, r_s]):
            out[key][k] = val
        am_cmd, as_cmd = (0.0, 0.0) if tripped else (F_m, F_c)
        # ------------------------------------------------------------------ mechanics (zero-order hold)
        for _ in range(nsub):
            fh = H["k"] * (x_i[k] - xm) + H["b"] * (v_i[k] - vm)
            fe_c = contact_force(xs, vs, E)
            vm += hsub * (fh - am_cmd - L["b"] * vm) / mL
            xm += hsub * vm
            vs += hsub * (as_cmd - F["b"] * vs - fe_c) / F["m"]
            xs += hsub * vs
        xd += dx
    out.update(t=t, x_i=x_i, tripped=tripped, N=max(1, int(round(T / dt))), arch=arch, T=T)
    return out


# ==========================================================================================
# Part A -- why force reflection becomes unstable: the delay margin
# ==========================================================================================
def contact_stiffness(Kp: float, k_e: float) -> float:
    """Stiffness reflected to the leader when the follower presses on the wall: Kp in series with k_e."""
    # TODO A1: two springs in series.
    raise NotImplementedError("contact_stiffness: TODO")


def delay_margin(m: float, b: float, k: float, kp: float) -> tuple[float, float]:
    """Smallest round-trip delay tau* for which  m s^2 + b s + k + kp e^{-s tau} = 0  has a root on the
    imaginary axis (the delayed spring kp reflected to a leader of mass m, damping b, stiffness k).

    At s = j w the delay term has magnitude kp, so |k - m w^2 + j b w| = kp gives
        m^2 w^4 + (b^2 - 2 k m) w^2 + k^2 - kp^2 = 0;
    for each positive root w the phase condition e^{-j w tau} = -(k - m w^2 + j b w)/kp gives the smallest
    tau > 0.  Returns (tau*, w_c) with the crossing frequency w_c (rad/s), or (inf, nan) if no root exists
    (then the loop is stable for every delay).  m = 0 is allowed (first-order leader)."""
    # TODO A2: solve the quadratic in w^2 (np.roots), keep the real positive roots, get tau from the phase
    #       condition with np.angle and % (2 pi); return the smallest (tau, w).  Handle m = 0 (one root).
    raise NotImplementedError("delay_margin: TODO")


# ==========================================================================================
# Part B -- wave variables
# ==========================================================================================
def wave_encode(F, xdot, b: float):
    """Wave variables of a power pair (F, xdot) for wave impedance b (Niemeyer & Slotine):
        u = (b xdot + F)/sqrt(2b)   (forward wave),   v = (b xdot - F)/sqrt(2b)   (returning wave).
    Power: F xdot = (u^2 - v^2)/2."""
    # TODO B1: two lines.
    raise NotImplementedError("wave_encode: TODO")


def wave_leader(xdot_m: float, v_m: float, b: float) -> tuple[float, float]:
    """Leader side (input: leader velocity xdot_m and the returning wave v_m(t) = v_s(t - T)).
    Returns (F_m, u_m) with F_m = b xdot_m - sqrt(2b) v_m and u_m = sqrt(2b) xdot_m - v_m,
    so that wave_encode(F_m, xdot_m, b) == (u_m, v_m)."""
    # TODO B2: solve the wave definitions for F_m given xdot_m and v_m.
    raise NotImplementedError("wave_leader: TODO")


def wave_follower(u_s: float, F_c: float, b: float) -> tuple[float, float]:
    """Follower side (input: arriving wave u_s(t) = u_m(t - T) and the follower controller force F_c).
    Returns (xdot_d, v_s) with xdot_d = (sqrt(2b) u_s - F_c)/b and v_s = sqrt(2b) xdot_d - u_s,
    so that wave_encode(F_c, xdot_d, b) == (u_s, v_s)."""
    # TODO B3: solve the wave definitions for xdot_d given u_s and F_c.
    raise NotImplementedError("wave_follower: TODO")


def wave_apparent_impedance(b: float, T: float, omega: float, termination: str) -> complex:
    """Impedance Z(j w) = F_m / xdot_m felt at the leader through a wave channel of one-way delay T:
    termination "rigid" (xdot_d = 0):  Z = b coth(j w T);   "free" (F_c = 0):  Z = b tanh(j w T).
    Low frequency: rigid -> spring of stiffness b/T; free -> mass b T."""
    # TODO B4: use np.tanh with a complex argument.
    raise NotImplementedError("wave_apparent_impedance: TODO")


# ==========================================================================================
# Part C -- energy bookkeeping and time-domain passivity
# ==========================================================================================
def follower_pd(x_d: float, x_s: float, xdot_s: float, Kp: float, Kd: float) -> float:
    """Follower position controller F_c = Kp (x_d - x_s) - Kd xdot_s (damping on the follower velocity only,
    so that F_c does not depend on the set-point step of the current sample)."""
    # TODO C1: one line.
    raise NotImplementedError("follower_pd: TODO")


def port_energies(P, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Split the sampled power P[k] flowing INTO a port into cumulative input and output energies:
    E_in[k] = h sum_{i<=k} max(P_i, 0),  E_out[k] = h sum_{i<=k} max(-P_i, 0)."""
    # TODO C2: np.cumsum of the positive and negative parts.
    raise NotImplementedError("port_energies: TODO")


def passivity_ratio(out_trial: float, budget: float) -> float:
    """Passivity controller as a scaling ratio r in [0, 1]: if the energy a port is about to output in this
    sample (out_trial > 0) exceeds the available budget, scale the port's output variable by
    r = max(budget, 0) / out_trial so that exactly the budget is released; otherwise r = 1."""
    # TODO C3: three cases.
    raise NotImplementedError("passivity_ratio: TODO")


def tdpa_leader(F_m: float, xdot_m: float, h: float, E_in_m: float, E_out_m: float, E_in_s_recv: float):
    """Passivity observer + controller at the leader port (impedance causality: input xdot_m, output F_m).
    Power into the channel P = F_m xdot_m.  If P >= 0, add P h to E_in_m.  If P < 0 the channel would output
    -P h; allowed budget = E_in_s_recv - E_out_m (energy that entered at the follower, delayed, minus what the
    leader port has already released).  Scale F_m by passivity_ratio and add the released energy to E_out_m.
    Returns (F_m_new, E_in_m, E_out_m, r)."""
    # TODO C4: observer (update E_in_m or E_out_m) + controller (passivity_ratio).
    raise NotImplementedError("tdpa_leader: TODO")


def tdpa_follower(F_c: float, dx_d: float, E_in_s: float, E_out_s: float, E_in_m_recv: float):
    """Passivity observer + controller at the follower port (admittance causality: input F_c, output the
    set-point step dx_d).  Energy out of the channel in this sample = F_c dx_d.  If <= 0, add -F_c dx_d to
    E_in_s.  Otherwise budget = E_in_m_recv - E_out_s; scale dx_d by passivity_ratio and add the released
    energy to E_out_s.  Returns (dx_d_new, E_in_s, E_out_s, r)."""
    # TODO C5: the same at the follower port, acting on the set-point step dx_d.
    raise NotImplementedError("tdpa_follower: TODO")


# ==========================================================================================
# Part D -- performance metrics for the report
# ==========================================================================================
def performance_metrics(sim: dict, t_from: float = 2.5) -> dict:
    """Steady-state metrics after t_from (s):
      F_m_mean, f_e_mean      mean force felt by the operator and mean contact force (N)
      force_ratio             F_m_mean / f_e_mean (force transparency; 1 = ideal)
      drift                   mean (x_d - x_m) (m): set-point drift between leader and follower
      x_err_rms               rms of (x_s - x_m) (m)
      x_std                   standard deviation of x_m (m): the size of a sustained oscillation (chatter)
      E_ch_min                minimum of the channel energy over the whole run (J); < 0 => active channel."""
    # TODO D1: masks on sim['t'] and numpy means / std / min.
    raise NotImplementedError("performance_metrics: TODO")


def latency_budget(components_ms: dict, limit_ms: float) -> tuple[float, float]:
    """One-way latency budget: total = sum of the components (ms); margin = limit - total (ms)."""
    # TODO D2: two lines.
    raise NotImplementedError("latency_budget: TODO")
