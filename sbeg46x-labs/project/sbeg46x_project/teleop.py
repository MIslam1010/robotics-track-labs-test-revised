"""Delayed bilateral teleoperation: network model, architectures and the reference simulator
(Scenarios 1, 4 and 5). Sign conventions are those of Lab 3:

* Leader (operator's hand lumped in):  (m_h + m_op) dv_m/dt = f_h - F_m - b_h v_m - k_g x_m.
  F_m is the force of the teleoperator on the leader; it RESISTS the operator when F_m v_m > 0.
  Relation to Week 2 (two-port notation, a_m = leader motor force, Z_m v_m = f_h + a_m): F_m = -a_m.
  Power INTO the communication channel at the leader port: P_m = F_m v_m.
* Follower controller: F_c = Kp (r - x_s) - Kd v_s, where r is the follower set-point (x_d from the
  channel, possibly changed by a set-point filter such as a speed governor or a safety filter).
  Power OUT of the channel at the follower port: P_s = F_c v_d, v_d = rate of change of x_d.
* Channel energy E_ch(t) = h sum (P_m - P_s) >= 0 for a passive channel (Lab 3).

The follower is either a second Hapkit-type device (Scenario 1: it renders the tissue in its own
firmware, so its motor outputs F_c - f_tissue) or the course-arm tip axis (Scenarios 4, 5: a
position-controlled ``ServoAxis`` that receives r as its command; F_c is then a virtual coupling force
that is reflected to the operator).

Motion scaling: x_d = alpha * x_m (follower metres per leader metre); force scaling: the leader
receives beta * F_c. The scaled channel conserves power only if beta = alpha (P-F, TDPA); the wave
channel is written in leader units (see ``WaveVariables``). Delay T is one way, the same in both
directions (round trip 2T).

BASELINE: ``PositionPosition`` (Lab 3 "pp"): x_d(t) = alpha x_m(t - T), F_m(t) = beta F_c(t - T), with
the follower PD gains of Labs 2-3 (Kp = 200 N/m, Kd = 2 N s/m).
TEAM TODO: the wave-variable and time-domain-passivity functions (port your Lab 3 code); the
set-point filter hook ``PositionPosition.setpoint`` (speed governor / MPC, CBF safety filter).
"""
from __future__ import annotations

import math

import numpy as np

from .devices import Hapkit, ServoAxis, VelocityEstimator, HAPKIT


# ------------------------------------------------------------------------------------- network
class DelayLine:
    """One direction of a packet link sampled at the control period. A packet sent at step n arrives
    at step n + d[n] (d >= 1 sample; d < 0: lost). The receiver uses the most recently SENT packet
    that has arrived and holds it (zero-order hold); before the first arrival it returns ``initial``.
    Same behaviour as the Lab 3 network model."""

    def __init__(self, delays, initial):
        self.d = np.asarray(delays, int)
        self.initial = np.array(initial, float)
        self.values, self.arrivals, self.best = {}, {}, -1

    def send(self, n, value):
        self.values[n] = np.array(value, float)
        if self.d[n] >= 0:
            self.arrivals.setdefault(n + int(self.d[n]), []).append(n)

    def receive(self, n):
        for s in self.arrivals.pop(n, []):
            self.best = max(self.best, s)
        return self.initial.copy() if self.best < 0 else self.values[self.best].copy()


def network_delays(T, h, n, jitter=0.0, loss=0.0, seed=3):
    """Per-packet delays in samples: one-way delay T (s), uniform jitter +-jitter (s), loss rate."""
    rng = np.random.default_rng(seed)
    base = T / h + (rng.uniform(-jitter / h, jitter / h, n) if jitter > 0 else 0.0)
    d = np.maximum(1, np.rint(np.full(n, 0.0) + base)).astype(int)
    if loss > 0:
        d[rng.random(n) < loss] = -1
    return d


# ------------------------------------------------------------------------- wave variables (TODO)
def wave_leader(xdot_m: float, v_m: float, b: float):
    """TEAM TODO (port Lab 3 ``wave_leader``). Leader side of the wave channel (Niemeyer & Slotine
    1991), Lab 3 convention and names: here xdot is a VELOCITY and v is the RETURNING WAVE.
    Waves of a power pair (F, xdot):  u = (b xdot + F)/sqrt(2b) (forward),  v = (b xdot - F)/sqrt(2b)
    (returning), so F xdot = (u^2 - v^2)/2.
    Input the leader velocity xdot_m and the returning wave v_m(t) = v_s(t - T). Return (F_m, u_m) with
        F_m = b xdot_m - sqrt(2b) v_m,   u_m = sqrt(2b) xdot_m - v_m."""
    raise NotImplementedError("wave_leader: port your Lab 3 wave-variable code")


def wave_follower(u_s: float, F_c: float, b: float):
    """TEAM TODO (port Lab 3 ``wave_follower``). Follower side: input the arriving wave
    u_s(t) = u_m(t - T) and the follower controller force F_c. Return (xdot_d, v_s) with
        xdot_d = (sqrt(2b) u_s - F_c)/b,   v_s = sqrt(2b) xdot_d - u_s,
    the set-point velocity and the wave sent back (same convention as ``wave_leader``)."""
    raise NotImplementedError("wave_follower: port your Lab 3 wave-variable code")


def tdpa_leader(F_m, v_m, h, E_in_m, E_out_m, E_in_s_recv):
    """TEAM TODO (port Lab 3). Passivity observer + controller at the leader port (Hannaford & Ryu
    2002). Return (F_m_new, E_in_m, E_out_m, r)."""
    raise NotImplementedError("tdpa_leader: port your Lab 3 time-domain-passivity code")


def tdpa_follower(F_c, dx_d, E_in_s, E_out_s, E_in_m_recv):
    """TEAM TODO (port Lab 3). Passivity observer + controller at the follower port. Return
    (dx_d_new, E_in_s, E_out_s, r)."""
    raise NotImplementedError("tdpa_follower: port your Lab 3 time-domain-passivity code")


# ------------------------------------------------------------------------------- architectures
class PositionPosition:
    """BASELINE position-position architecture (Lab 3 "pp": the follower controller force F_c is
    reflected) with motion scaling alpha and force scaling
    beta. Packets: leader -> follower [x_m, u_m, E_in_m]; follower -> leader [F_c, returning wave v_s, E_in_s].

    Hook for the team: ``setpoint(k, t, x_d, x_s, v_s, f_meas)`` returns the follower set-point r
    (default r = x_d). A speed governor (MPC) or a CBF safety filter goes here. ``reflect`` returns
    the force sent back (default F_c); a governor may add a coupling term so the operator feels it."""
    name = "pp"

    def __init__(self, Kp=200.0, Kd=2.0, alpha=1.0, beta=1.0, b=5.0, tau_f=0.005):
        self.Kp, self.Kd, self.alpha, self.beta, self.b, self.tau_f = Kp, Kd, alpha, beta, b, tau_f
        self.reset()

    def reset(self):
        self.x_d = None
        self._x_recv_prev = None
        self.E_in_m = self.E_out_m = self.E_in_s = self.E_out_s = 0.0
        self.w_f = 0.0

    # -- hooks ----------------------------------------------------------------------------------
    def setpoint(self, k, t, x_d, x_s, v_s, f_meas):
        return x_d

    def reflect(self, F_c, x_d, r, x_s, v_s):
        return F_c

    # -- follower side --------------------------------------------------------------------------
    def follower(self, k, t, h, pkt, x_s, v_s, f_meas):
        x_new = self.alpha * pkt[0]
        dx = 0.0 if self.x_d is None else x_new - self.x_d
        self.x_d = x_new
        r = self.setpoint(k, t, self.x_d, x_s, v_s, f_meas)
        F_c = self.Kp * (r - x_s) - self.Kd * v_s
        back = [self.reflect(F_c, self.x_d, r, x_s, v_s), 0.0, self.E_in_s]
        return F_c, r, back, F_c * dx / h

    # -- leader side ----------------------------------------------------------------------------
    def leader(self, k, t, h, pkt, x_m, v_m):
        F_m = self.beta * pkt[0]
        return F_m, [x_m, 0.0, self.E_in_m]


class WaveVariables(PositionPosition):
    """Wave-variable channel (uses the TEAM TODO functions ``wave_leader``/``wave_follower``) with
    wave impedance b. The follower set-point integrates xdot_d; motion and force scaling are applied
    outside the wave channel: the channel works in leader units, the follower force enters it as
    alpha * F_c (same power), x_d = alpha * integral of v_d, and the leader displays beta * F. Keep
    beta = 1 for a passive channel. As in Lab 3, the wave arriving at the leader passes a first-order
    low-pass filter (time constant tau_f = 5 ms): a filter with |H| <= 1 cannot add wave energy, and
    it removes the undamped wave reflections that grow at very short delays."""
    name = "wave"

    def follower(self, k, t, h, pkt, x_s, v_s, f_meas):
        if self.x_d is None:
            self.x_d = x_s
        r = self.setpoint(k, t, self.x_d, x_s, v_s, f_meas)
        F_c = self.Kp * (r - x_s) - self.Kd * v_s
        F_wave = self.alpha * self.reflect(F_c, self.x_d, r, x_s, v_s)     # follower force in leader units
        xdot_d, v_s_wave = wave_follower(pkt[1], F_wave, self.b)   # (set-point velocity, returning wave)
        dx = self.alpha * xdot_d * h
        self.x_d += dx
        return F_c, r, [F_c, v_s_wave, 0.0], F_c * dx / h

    def leader(self, k, t, h, pkt, x_m, v_m):
        self.w_f += h / (self.tau_f + h) * (pkt[1] - self.w_f)      # filtered returning wave
        F, u_m = wave_leader(v_m, self.w_f, self.b)                   # v_m here = leader velocity
        return self.beta * F, [x_m, u_m, 0.0]


class TimeDomainPassivity(PositionPosition):
    """Position-force channel with a passivity observer/controller at each port (uses the TEAM TODO
    functions ``tdpa_leader``/``tdpa_follower``)."""
    name = "tdpa"

    def follower(self, k, t, h, pkt, x_s, v_s, f_meas):
        if self.x_d is None:
            self.x_d = self.alpha * pkt[0]
            self._x_recv_prev = pkt[0]
        r = self.setpoint(k, t, self.x_d, x_s, v_s, f_meas)
        F_c = self.Kp * (r - x_s) - self.Kd * v_s
        dx = self.alpha * (pkt[0] - self._x_recv_prev)
        self._x_recv_prev = pkt[0]
        dx, self.E_in_s, self.E_out_s, _ = tdpa_follower(F_c, dx, self.E_in_s, self.E_out_s, pkt[2])
        self.x_d += dx
        back = [self.reflect(F_c, self.x_d, r, x_s, v_s), 0.0, self.E_in_s]
        return F_c, r, back, F_c * dx / h

    def leader(self, k, t, h, pkt, x_m, v_m):
        F_m = self.beta * pkt[0]
        F_m, self.E_in_m, self.E_out_m, _ = tdpa_leader(F_m, v_m, h, self.E_in_m, self.E_out_m, pkt[2])
        return F_m, [x_m, 0.0, self.E_in_m]


# ----------------------------------------------------------------------------------- simulator
def simulate(arch, operator, T_end, delay=0.0, follower="hapkit", env=None, x0_m=0.0, x0_s=None,
             h=HAPKIT["T"], substeps=5, jitter=0.0, loss=0.0, seed=3, sensor=None, servo=None,
             leader_dev=None, follower_dev=None, true_velocity=False):
    """Reference simulator of a delayed bilateral teleoperator.

    arch      -- an architecture object (``PositionPosition``, ``WaveVariables``, ... or the team's)
    operator  -- ``devices.Operator`` (hand on the leader)
    delay     -- one-way delay T (s); ``jitter`` (s) and ``loss`` (rate) per packet
    follower  -- "hapkit" (second haptic device) or "servo" (course-arm tip axis, ``servo`` = ServoAxis)
    env       -- for a Hapkit follower: callable env(t, x_meas, v_est) -> resisting force (N, positive
                 opposes +x), rendered in the follower firmware (e.g. the needle model); it is also
                 evaluated on the true state for logging
    sensor    -- optional callable F -> measured F given to the follower controller (e.g. ForceSensor)
    Returns a dict of arrays (t, x_int, x_m, v_m, x_s, v_s, x_d, r, F_m, F_c, f_env, f_meas, F_h, P_m,
    P_s, E_ch) and scalars (sat_m, sat_s = saturation counts, tripped_m, tripped_s, watchdog_t)."""
    n = int(round(T_end / h))
    t = np.arange(n) * h
    L = leader_dev or Hapkit()
    L.x, L.v = x0_m, 0.0
    if follower == "hapkit":
        F = follower_dev or Hapkit()
    else:
        F = servo or ServoAxis()
    F.x = arch.alpha * x0_m if x0_s is None else x0_s
    F.v = 0.0
    if follower != "hapkit":
        F.x_cmd = F.x
    arch.reset()
    fwd = DelayLine(network_delays(delay, h, n, jitter, loss, seed), [x0_m, 0.0, 0.0])
    bwd = DelayLine(network_delays(delay, h, n, jitter, loss, seed + 1), [0.0, 0.0, 0.0])
    est_m, est_s = VelocityEstimator(h), VelocityEstimator(h)
    keys = "x_int x_m v_m x_s v_s x_d r F_m F_c f_env f_meas F_h P_m P_s E_ch".split()
    out = {k: np.zeros(n) for k in keys}
    E = 0.0
    watchdog_t = math.nan
    for k in range(n):
        tk = t[k]
        xm_meas, xs_meas = L.measure(), F.measure()
        vm_est, vs_est = est_m(xm_meas), est_s(xs_meas)
        vm_c = L.v if true_velocity else vm_est
        vs_c = F.v if true_velocity else vs_est
        f_env = env(tk, xs_meas, vs_est) if (env is not None and follower == "hapkit") else 0.0
        f_meas = sensor(f_env) if sensor is not None else f_env
        rf, rb = fwd.receive(k), bwd.receive(k)
        F_c, r, back, P_s = arch.follower(k, tk, h, rf, xs_meas, vs_c, f_meas)
        F_m, fwd_pkt = arch.leader(k, tk, h, rb, xm_meas, vm_c)
        fwd.send(k, fwd_pkt)
        bwd.send(k, back)
        # firmware output stages (saturation and watchdog)
        F_m_applied = -L.command(-F_m)           # the leader motor applies -F_m to the handle
        if follower == "hapkit":
            F.command(F_c - f_env)
        else:
            F.command(r, h)
        P_m = F_m_applied * L.v
        E += (P_m - P_s) * h
        if hasattr(operator, "observe"):
            operator.observe(tk, F_m_applied, L.x)
        F_h = operator.force(tk, L.x, L.v)
        for key, val in zip(keys, [operator.x_int(tk), L.x, L.v, F.x, F.v, arch.x_d, r, F_m_applied,
                                   F_c, f_env, f_meas, F_h, P_m, P_s, E]):
            out[key][k] = val
        if math.isnan(watchdog_t) and (getattr(L, "tripped", False) or getattr(F, "tripped", False)):
            watchdog_t = tk
        m_op = operator.mass(tk)
        for _ in range(substeps):
            L.advance(h / substeps, F_ext=operator.force(tk, L.x, L.v), m_extra=m_op)
            F.advance(h / substeps)
    out.update(t=t, sat_m=L.sat_count, sat_s=getattr(F, "sat_count", 0),
               tripped_m=L.tripped, tripped_s=getattr(F, "tripped", False), watchdog_t=watchdog_t,
               arch=arch.name, delay=delay)
    return out
