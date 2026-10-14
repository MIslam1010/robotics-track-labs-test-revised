"""SBEG 46X Lab 4 -- haptic rendering on a Hapkit-type device: virtual wall, passivity and the stable stiffness
limit (Z-width), virtual fixtures with a proxy, tool-tissue contact and a needle-insertion force profile.
STARTER CODE -- fill in every function marked TODO (in the order A1 ... D6).
Run the auto-checks with:  LAB_IMPL=starter python -m pytest -q test_lab.py   (or copy this file to your own module name)

These functions become the `haptics` module of your cumulative library `sbeg46x_medrob`.

Notation and units: SI (s, m, kg, N).  Impedance m x'' + b x' + k x = f (MR Eq. 11.62); impedance control
with M = 0 on a lightweight backdrivable device (MR Sec. 11.7.1, p. 446); current control of a geared motor
(MR Sec. 11.8, pp. 447-449).

Sign conventions used everywhere in this file
  * x is the handle position (m), positive away from the user; F_motor is the force the device applies TO the
    hand, f_hand the force the hand applies TO the device.  Device:  m_h x'' + b_h x' + k_g x + f_c sgn(x') =
    F_motor + f_hand.
  * Needle and tissue: z = x - x_skin is the needle-tip depth below the undeformed skin surface (m), v = z'.
    The tissue force f_tissue(z, v) >= 0 resists insertion; the device renders F_motor = -f_tissue during
    insertion (and +friction during withdrawal, see NeedleTissue).

Parts
  A. Passivity limits of a sampled virtual wall: Colgate's condition b > K T / 2 + |B|, its frequency-by-
     frequency form, the quantisation condition f_c > K dx / 2, and the energy that a zero-order hold
     creates in one push into a wall
  B. Measuring the stable stiffness limit: bounce (restitution) test in simulation, threshold from logged trials
  C. Virtual fixtures: unilateral forbidden-region wall, a proxy (god-object) for a thin keep-out slab,
     a 2-D forbidden disc around a vessel, an admittance-type guidance fixture
  D. Tool-tissue and needle-insertion force models: Kelvin-Voigt, Hunt-Crossley, a layered
     stiffness-friction-cutting needle model, puncture detection in a force log
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


# ==========================================================================================
# Provided (do not edit): data files, the device simulator
# ==========================================================================================
def load_device() -> dict:
    """Device and loop parameters from data/device.json (illustrative values): T (s), x_res (m), force_max (N),
    x_range (m), m_h (kg), b_h (N s/m), k_g (N/m), f_c (N), k_t (N m/A)."""
    return json.loads((DATA / "device.json").read_text())


def load_tissue() -> dict:
    """Layered needle-decompression trainer model from data/needle_tissue.json (illustrative values)."""
    return json.loads((DATA / "needle_tissue.json").read_text())


def load_csv(name: str) -> dict:
    """A CSV file of data/ as a dict of float arrays keyed by the header names."""
    with open(DATA / name) as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def rk4_step(f, s, h: float) -> np.ndarray:
    k1 = f(s)
    k2 = f(s + 0.5 * h * k1)
    k3 = f(s + 0.5 * h * k2)
    k4 = f(s + h * k3)
    return s + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_device(dev: dict, law, T_end: float, x0: float = 0.0, v0: float = 0.0, f_hand=None,
                    quantize: bool = True, friction: bool = False, substeps: int = 10, stop=None) -> dict:
    """Sampled-data simulation of the haptic device rendering `law` (same loop as SBEG 461 Lab 8).

    Plant (continuous):  m_h x'' + b_h x' + k_g x + f_c sgn(x') = F_applied + f_hand(t, x, v), with the Coulomb
    term only if friction=True (smoothed as f_c tanh(v / 1 mm/s)); RK4 with `substeps` steps per sample.
    Controller every T = dev["T"] at t_k = k T:
      1. x_meas = x rounded to the sensor resolution dev["x_res"] (if quantize), else x;
      2. v_est = (x_meas_k - x_meas_{k-1}) / T  (backward difference; v_est = v0 at k = 0);
      3. F_cmd = law(t_k, x_meas, v_est), saturated to +-dev["force_max"], held constant until t_{k+1}
         (zero-order hold).
    `stop(k, x, v)` may return True to end the run early (after sample k).
    Returns dict t, x, v (true state at the samples), x_meas, v_est, F (applied), saturated (bool array)."""
    f_hand = f_hand or (lambda t, x, v: 0.0)
    T, m, b, kg, Fmax = dev["T"], dev["m_h"], dev["b_h"], dev["k_g"], dev["force_max"]
    fc = dev.get("f_c", 0.0) if friction else 0.0
    n = int(round(T_end / T))
    t = np.arange(n + 1) * T
    X = np.zeros((n + 1, 2))
    X[0] = (x0, v0)
    xm, ve, Fa = np.zeros(n + 1), np.zeros(n + 1), np.zeros(n + 1)
    sat = np.zeros(n + 1, bool)
    last = n
    for k in range(n + 1):
        x, v = X[k]
        xm[k] = round(x / dev["x_res"]) * dev["x_res"] if quantize else x
        ve[k] = v0 if k == 0 else (xm[k] - xm[k - 1]) / T
        F = float(law(t[k], xm[k], ve[k]))
        sat[k] = abs(F) > Fmax
        Fa[k] = max(-Fmax, min(Fmax, F))
        if k == n or (stop is not None and stop(k, x, v)):
            last = k
            break
        Fk, tk = Fa[k], t[k]

        def rhs(s):
            fr = fc * math.tanh(s[1] / 1e-3) if fc else 0.0
            return np.array([s[1], (Fk + f_hand(tk, s[0], s[1]) - b * s[1] - kg * s[0] - fr) / m])

        s = X[k].copy()
        for _ in range(substeps):
            s = rk4_step(rhs, s, T / substeps)
        X[k + 1] = s
    sl = slice(0, last + 1)
    return dict(t=t[sl], x=X[sl, 0], v=X[sl, 1], x_meas=xm[sl], v_est=ve[sl], F=Fa[sl], saturated=sat[sl])


def hand_ramp(x_start: float, speed: float, k_hand: float = 400.0, b_hand: float = 4.0, t_start: float = 0.0):
    """A hand that drives the handle along x_ref(t) = x_start + speed (t - t_start) through a grasp
    spring-damper (k_hand, b_hand): f_hand = k_hand (x_ref - x) + b_hand (speed - v).  A passive model of a
    trainee pushing a needle in at a steady speed."""
    def f(t, x, v):
        tt = max(t - t_start, 0.0)
        return k_hand * (x_start + speed * tt - x) + b_hand * ((speed if t >= t_start else 0.0) - v)
    return f


# ==========================================================================================
# Part A -- passivity of a sampled virtual wall
# ==========================================================================================
def colgate_K_max(b: float, B: float, T: float) -> float:
    """Largest wall stiffness K (N/m) for which the sampled wall F = -K x - B v_est (v_est by backward
    difference, zero-order hold, period T) on a device with physical damping b is passive, from Colgate's
    condition  b > K T / 2 + |B|  (Colgate & Brown 1994):  K_max = 2 (b - |B|) / T, or 0 if |B| >= b."""
    # TODO (A1): implement as described in the docstring.
    raise NotImplementedError("colgate_K_max")


def required_damping(K: float, B: float, T: float, theta):
    """Physical damping needed for passivity at each normalised frequency theta = omega T in (0, pi]:

        b_req(theta) = (T/2) Re{ (1 - e^{-j theta}) H(e^{j theta}) } / (1 - cos theta),
        H(z) = K + B (1 - z^-1) / T   (spring + backward-difference damper),

    the frequency-by-frequency form of Colgate's passivity condition.  Returns an array like theta.
    (Its maximum over theta is K T/2 + |B|: at theta -> 0 if B < 0, at the Nyquist frequency theta = pi if B > 0.)"""
    # TODO (A2): implement as described in the docstring.
    raise NotImplementedError("required_damping")


def quantisation_K_max(f_c: float, x_res: float) -> float:
    """Largest K (N/m) for which the device's Coulomb friction f_c (N) can absorb the energy created by a
    position sensor of resolution x_res (m): f_c > K x_res / 2  ->  K_max = 2 f_c / x_res."""
    # TODO (A3): implement as described in the docstring.
    raise NotImplementedError("quantisation_K_max")


def sampled_wall_work(K: float, T: float, speed: float, depth: float) -> float:
    """Net work (J) done BY a sampled unilateral wall (surface at x = 0, wall at x > 0) ON a hand that pushes
    into it at constant `speed` to `depth` and comes straight back out at the same speed.

    The wall samples x at t_k = k T (starting at the moment of contact, t = 0) and holds F_k = -K max(x(t_k), 0)
    until t_{k+1}, so the work on interval k is exactly F_k (x(t_{k+1}) - x(t_k)).  Positive result = energy
    CREATED by the wall (a continuous spring would give exactly 0).  Assumes depth / (speed T) is close to an
    integer N; then the result is K speed T depth."""
    # TODO (A4): implement as described in the docstring.
    raise NotImplementedError("sampled_wall_work")


# ==========================================================================================
# Part B -- measuring the stable stiffness limit
# ==========================================================================================
def wall_force(x: float, v: float, K: float, B: float, x_wall: float = 0.0, side: int = +1) -> float:
    """Unilateral virtual wall occupying side*(x - x_wall) > 0 (a forbidden region).  Outside: F = 0.  Inside:
    F = -K (x - x_wall) - B v, clipped so that the wall only PUSHES the hand out (side*F <= 0)."""
    # TODO (B1): implement as described in the docstring.
    raise NotImplementedError("wall_force")


def bounce_restitution(dev: dict, K: float, B: float = 0.0, v_in: float = 0.05, quantize: bool = False,
                       friction: bool = False) -> float:
    """Coefficient of restitution e = |v_out| / |v_before| of the free handle bouncing once on a virtual wall
    at x = 0 (no hand).

    Start half a sample before the wall, x0 = -v_in T / 2, with speed v_in toward it; render
    wall_force(x, v, K, B) with simulate_device and stop at the first sample after contact at which the TRUE
    position is back outside (x < 0).  v_before is the true velocity at the last sample before contact (here the
    first sample), v_out the true velocity at the stop sample.  e > 1 means the wall CREATED energy during the
    contact (an active wall; on the real device a growing bounce or a buzz).  Max. run 1 s; return 0 if the
    handle never leaves the wall."""
    # TODO (B2): implement as described in the docstring.
    raise NotImplementedError("bounce_restitution")


def measure_K_limit(dev: dict, B: float = 0.0, v_in: float = 0.05, K_lo: float = 20.0, K_hi: float = 3000.0,
                    tol: float = 1.0, **kw) -> float:
    """Stiffness (N/m) at which the bounce restitution crosses 1, by bisection on [K_lo, K_hi] (to within tol),
    assuming e(K_lo) < 1 < e(K_hi).  This is the simulated version of the bench test 'raise K until the wall
    starts to buzz'."""
    # TODO (B3): implement as described in the docstring.
    raise NotImplementedError("measure_K_limit")


def threshold_from_trials(K, e) -> float:
    """Stiffness at which a logged restitution e(K) first crosses 1 (K sorted ascending): straight-line
    interpolation between the last trial with e < 1 and the first with e >= 1.  Raises ValueError if e never
    reaches 1 or already starts at or above 1."""
    # TODO (B4): implement as described in the docstring.
    raise NotImplementedError("threshold_from_trials")


# ==========================================================================================
# Part C -- virtual fixtures
# ==========================================================================================
def penalty_slab_force(x: float, a: float, w: float, K: float) -> float:
    """Penalty rendering of a thin forbidden slab a < x < a + w: push toward the NEAREST surface,
    F = K (a - x) if x < a + w/2 else K (a + w - x); F = 0 outside.  (Shows pop-through past the mid-plane.)"""
    # TODO (C1): implement as described in the docstring.
    raise NotImplementedError("penalty_slab_force")


class ProxySlab:
    """Proxy (god-object) rendering of the thin forbidden slab a < x < a + w on a 1-DOF device.

    The proxy p is a point that is never allowed inside the slab and follows the handle x when it can:
      * if x is outside the slab and the straight move from p to x does not cross the slab, p = x;
      * otherwise p stays on the surface it was last outside of (entry side remembered), i.e. p = a if the
        proxy was at or below a, p = a + w if it was at or above a + w.
    Force F = K (p - x) (a spring from the handle to the proxy; zero when p = x)."""

    def __init__(self, a: float, w: float, K: float, x0: float):
        if a < x0 < a + w:
            raise ValueError("start the proxy outside the slab")
        self.a, self.w, self.K, self.p = a, w, K, x0

    def update(self, x: float) -> float:
        """Move the proxy for the new handle position x by the two rules above; store it in self.p and return it."""
        # TODO (C2): implement as described in the docstring.
        raise NotImplementedError("ProxySlab.update")

    def force(self, x: float) -> float:
        return self.K * (self.update(x) - x)


def forbidden_disc_force(p, c, R: float, K: float):
    """Forbidden-region fixture around a vessel: the disc |p - c| < R (vessel radius + safety margin) in the
    plane.  Outside: F = 0 and the proxy is p.  Inside: the proxy is the closest boundary point
    q = c + R (p - c) / |p - c| and F = K (q - p).  Returns (F, q) as numpy arrays.  (At p = c exactly the
    direction is undefined: use n = (1, 0).)"""
    # TODO (C3): implement as described in the docstring.
    raise NotImplementedError("forbidden_disc_force")


def guidance_velocity(f, d, c: float, c_tau: float):
    """Admittance-type guidance virtual fixture (steady-hand cooperative robot): the commanded tool velocity is
    v = c (f_par + c_tau f_perp), with f_par = (f . d) d the component of the user's force along the preferred
    unit direction d and f_perp = f - f_par.  c_tau = 1: no fixture; c_tau = 0: hard fixture (motion only along
    d); 0 < c_tau < 1: soft fixture.  Returns v (numpy array)."""
    # TODO (C4): implement as described in the docstring.
    raise NotImplementedError("guidance_velocity")


# ==========================================================================================
# Part D -- tool-tissue and needle-insertion force models
# ==========================================================================================
def kelvin_voigt_force(delta: float, ddelta: float, k: float, b: float) -> float:
    """Kelvin-Voigt contact (spring k in parallel with damper b) for penetration delta (m) and rate ddelta:
    f = k delta + b ddelta while delta > 0, else 0.  The result is NOT clipped: it jumps to b ddelta at first
    touch and can be negative (a non-physical pull) when withdrawing fast."""
    # TODO (D1): implement as described in the docstring.
    raise NotImplementedError("kelvin_voigt_force")


def hunt_crossley_force(delta: float, ddelta: float, k: float, lam: float, n: float) -> float:
    """Hunt-Crossley contact f = k delta^n + lam delta^n ddelta for delta > 0, else 0, clipped at >= 0
    (contact can push, never pull).  Continuous at first touch (f = 0 at delta = 0)."""
    # TODO (D2): implement as described in the docstring.
    raise NotImplementedError("hunt_crossley_force")


def puncture_depth(a1: float, a2: float, f_p: float) -> float:
    """Membrane deflection delta_p (m) at which the stiffness force a1 delta + a2 delta^2 reaches the puncture
    force f_p (positive root of the quadratic; a2 = 0 allowed)."""
    # TODO (D3): implement as described in the docstring.
    raise NotImplementedError("puncture_depth")


class NeedleTissue:
    """Layered stiffness-friction-cutting needle model (after Okamura, Simone & O'Leary 2004), for the
    needle-decompression trainer (illustrative values in data/needle_tissue.json).

    Depth z (m) below the undeformed skin, speed v = dz/dt.  Parameters (dict `p`):
      skin:   a1 (N/m), a2 (N/m^2), f_p (N)  -- membrane stiffness f = a1 z + a2 z^2 until it reaches f_p
      tissue: f_cut (N), mu (N/m)           -- after skin puncture, while inserting (v > 0) a constant cutting
                                              force f_cut at the tip, and shaft friction mu * L with
                                              L = min(z, z_pl) the shaft length inside the tissue
      pleura: z_pl (m), k_pl (N/m), f_p (N)  -- at depth z_pl the tip meets the pleura: an extra membrane force
                                              k_pl (z - z_pl) until it reaches f_p (then the pleura is punctured:
                                              no cutting force any more, friction only)
      v_eps (m/s): friction is mu L sgn(v) for |v| >= v_eps and mu L v / v_eps below (Karnopp-type smoothing)

    Force on the needle (resisting insertion is positive):
      before skin puncture:  f = a1 z + a2 z^2 for z > 0 (elastic: the same curve on the way out), 0 for z <= 0;
      after skin puncture:   f = fric + [v > 0] * f_cut              (tip in tissue, z < z_pl, pleura intact)
                             f = fric + [v > 0] * f_cut + k_pl (z - z_pl)   (z >= z_pl, pleura intact; the
                                                                      membrane term is elastic, any v sign)
                             f = fric                                   (pleura punctured)
      with fric = mu L sgn_eps(v) (negative when withdrawing: the friction then pulls the needle back in).
    Punctures are permanent (state).  `events` records (kind, z) at each puncture."""

    def __init__(self, p: dict):
        self.p = p
        self.skin_punctured = False
        self.pleura_punctured = False
        self.events = []

    def force(self, z: float, v: float) -> float:
        s, tis, pl = self.p["skin"], self.p["tissue"], self.p["pleura"]
        # TODO (D4): implement as described in the docstring.
        raise NotImplementedError("NeedleTissue.force")


def needle_quasistatic_profile(p: dict, z) -> np.ndarray:
    """Insertion force (v > 0, slow) against depth for the NeedleTissue model, evaluated on the increasing depth
    grid z with a fresh model (punctures happen in order along the grid)."""
    m = NeedleTissue(p)
    return np.array([m.force(float(zz), 1.0) for zz in z])


def detect_punctures(z, F, drop: float, window: int = 5, hold: int = 20) -> list:
    """Puncture events in an insertion log (z depth, F resisting force, one row per sample).

    A puncture starts at sample i if the force falls by more than `drop` (N) within `window` samples AND stays
    down: F[i] - max(F[i + window : i + window + hold]) > drop, while the depth is increasing
    (z[i + window] > z[i]).  The event index is the sample of largest force in F[i : i + window + 1] (the peak
    just before the drop); after an event, skip to that index + window + hold.  Short dips (the friction
    changing sign for a few samples when the needle briefly moves back) are therefore not punctures.
    Returns a list of indices."""
    # TODO (D5): implement as described in the docstring.
    raise NotImplementedError("detect_punctures")


def max_profile_slope(z, f) -> float:
    """Largest positive slope df/dz (N/m) of a force-depth profile between consecutive samples (the stiffness the
    device must render there; compare with colgate_K_max)."""
    # TODO (D6): implement as described in the docstring.
    raise NotImplementedError("max_profile_slope")


def render_needle(dev: dict, p: dict, x_skin: float, speed: float, T_end: float, x0: float, n_vel: int = 20,
                  **kw) -> dict:
    """Render the needle model on the device while a hand (hand_ramp from x0 at `speed`) pushes the handle in.

    Law at each sample: z = x_meas - x_skin; velocity from the last n_vel + 1 measured positions,
    v_f = (x_meas_k - x_meas_{k-n_vel}) / (n_vel T) (a backward difference over n_vel samples: the one-sample
    difference of a 20 um sensor at 1 kHz only takes the values 0, 20, 40 mm/s ... and would switch the
    friction and cutting forces on and off); F_motor = -NeedleTissue.force(z, v_f).
    Returns the simulate_device dict plus z (true depth), f_tissue (= -F) and events (list of (kind, z))."""
    model = NeedleTissue(p)
    hist = []

    def law(t, x, v):
        hist.append(x)
        n = min(n_vel, len(hist) - 1)
        v_f = (hist[-1] - hist[-1 - n]) / (n * dev["T"]) if n > 0 else 0.0
        return -model.force(x - x_skin, v_f)

    sim = simulate_device(dev, law, T_end, x0=x0, f_hand=hand_ramp(x0, speed), **kw)
    sim["z"] = sim["x"] - x_skin
    sim["f_tissue"] = -sim["F"]
    sim["events"] = model.events
    return sim
