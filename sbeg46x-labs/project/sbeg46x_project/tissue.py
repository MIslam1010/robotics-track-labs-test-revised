"""Rendered tissue and contact models.

Needle model (Scenario 1). The same stiffness / friction / cutting structure as the needle model of
Lab 4 (after Okamura, Simone and O'Leary, IEEE Trans. Biomed. Eng. 51(10), 2004), with one more
membrane and a critical structure, and SCALED to the 2 N range of the Hapkit-type device. The
layers are those of an epidural-type insertion: skin, soft tissue, a tough ligament whose puncture
gives the "loss of resistance", a thin space (the target), then the dura, which must not be touched.
ILLUSTRATIVE VALUES, not tissue data.

Depth z (m) below the undeformed skin, speed v = dz/dt. The model returns the force of the tissue on
the needle that RESISTS insertion (positive while inserting), as in Lab 4:

* skin: f = a1 z + a2 z^2 for z > 0 until f reaches f_p (puncture; permanent);
* after the skin puncture: cutting force f_cut while v > 0 (only while the tip is in tissue, i.e.
  before the ligament puncture) and shaft friction mu L sgn_eps(v), L = min(z, z_lig) the shaft
  length in tissue (Karnopp-type smoothing below v_eps);
* ligament at z_lig: an extra membrane force k_lig (z - z_lig) until it reaches f_p_lig (puncture =
  loss of resistance; after it, friction only);
* critical structure (dura) at z_crit: a stiff wall k_crit (z - z_crit) that is never punctured.
  Touching it is a constraint violation in the acceptance tests.
"""
from __future__ import annotations

import copy
import math

import numpy as np

NEEDLE_S1 = {
    "skin": {"a1": 100.0, "a2": 20000.0, "f_p": 0.8},
    "tissue": {"f_cut": 0.20, "mu": 12.0},
    "ligament": {"z_lig": 0.030, "k_lig": 250.0, "f_p": 0.60},
    "critical": {"z_crit": 0.037, "k_crit": 2000.0},
    "v_eps": 0.002,
    "x_skin": -0.020,
    "note": "Scenario 1 epidural-type needle model (illustrative), scaled to the 2 N Hapkit range. "
            "Skin and tissue values as in the Lab 4 needle model; depth z = x - x_skin.",
}


class NeedleTissue:
    """Layered needle-insertion model described in the module docstring. Stateful (punctures are
    permanent): call ``force`` once per firmware tick, in time order. ``events`` lists
    (kind, z, t) for each puncture."""

    def __init__(self, p: dict | None = None):
        self.p = copy.deepcopy(NEEDLE_S1 if p is None else p)
        self.reset()

    def reset(self):
        self.skin_punctured = False
        self.lig_punctured = False
        self.events = []

    def puncture_depths(self) -> tuple:
        """Quasi-static depths (m) at which the skin and the ligament rupture."""
        s, lg = self.p["skin"], self.p["ligament"]
        z_skin = (-s["a1"] + math.sqrt(s["a1"] ** 2 + 4 * s["a2"] * s["f_p"])) / (2 * s["a2"])
        return z_skin, lg["z_lig"] + lg["f_p"] / lg["k_lig"]

    def force(self, z: float, v: float, t: float = math.nan) -> float:
        s, tis, lg = self.p["skin"], self.p["tissue"], self.p["ligament"]
        f_crit = self.critical_force(z)
        if not self.skin_punctured:
            if z <= 0:
                return 0.0
            f = s["a1"] * z + s["a2"] * z * z
            if f < s["f_p"]:
                return f + f_crit
            self.skin_punctured = True
            self.events.append(("skin", z, t))
        L = max(0.0, min(z, lg["z_lig"]))
        ve = self.p["v_eps"]
        sg = math.copysign(1.0, v) if abs(v) >= ve else v / ve
        fric = tis["mu"] * L * sg
        if self.lig_punctured:
            return fric + f_crit
        f = fric + (tis["f_cut"] if v > 0 else 0.0)
        if z >= lg["z_lig"]:
            f_m = lg["k_lig"] * (z - lg["z_lig"])
            if f_m >= lg["f_p"]:
                self.lig_punctured = True
                self.events.append(("ligament", z, t))
                return fric + f_crit
            f += f_m
        return f + f_crit

    def critical_force(self, z: float) -> float:
        cr = self.p["critical"]
        return cr["k_crit"] * (z - cr["z_crit"]) if z > cr["z_crit"] else 0.0


def needle_quasistatic_profile(p: dict, z) -> np.ndarray:
    """Insertion force (slow, v > 0) against depth on the increasing grid z, with a fresh model."""
    m = NeedleTissue(p)
    return np.array([m.force(float(zz), 1.0) for zz in z])


class ForceSensor:
    """Force measurement seen by a controller: first-order lag (time constant ``tau``) plus white
    noise ``sigma`` (N). On the Hapkit pair the tissue force is computed in the follower firmware;
    the lag and noise stand in for the force sensor of a real insertion robot."""

    def __init__(self, tau=0.005, sigma=0.01, T=1e-3, seed=0):
        self.a = 1.0 - math.exp(-T / tau)
        self.sigma = sigma
        self.y = 0.0
        self.rng = np.random.default_rng(seed)

    def __call__(self, F) -> float:
        self.y += self.a * (F - self.y)
        return float(self.y + self.sigma * self.rng.standard_normal())


def wall_force(x: float, v: float, K: float, B: float = 0.0, x_wall: float = 0.0, side: int = +1) -> float:
    """One-sided virtual wall (same convention as Lab 4): free space is side*(x - x_wall) <= 0. Inside
    the wall the force on the handle is -side*max(K pen + B side v, 0), pen = side*(x - x_wall);
    a wall pushes, never pulls."""
    pen = side * (x - x_wall)
    if pen <= 0:
        return 0.0
    return float(-side * max(K * pen + B * side * v, 0.0))
