"""SBEG 461 enrichment lab -- a control-barrier-function (CBF) safety filter for the rehabilitation robot.
STARTER CODE. Complete every function marked TODO (keep the names and signatures).
Run the auto-checks in your Lab code folder with:  LAB_IMPL=starter python -m pytest -q test_lab.py
The functions marked "provided" are complete; read them before you start.

The handle of the planar rehabilitation robot (shared model rehab_2r, horizontal plane) follows a reaching
movement with task-space computed torque (MR Sec. 11.4.3).  A therapist's hand rests on the table near the
path.  A CBF-QP safety filter changes the commanded handle acceleration as little as possible so that
  (1) the handle never enters a keep-out disc of radius r around the hand (distance barrier, relative degree 2,
      handled by a second-order / exponential CBF), and
  (2) the speed at which the handle approaches the hand never exceeds v_max, a power-and-force-limiting speed
      from the reduced-mass contact model of Week 13 (approach-speed barrier, relative degree 1).

Model and notation (MR Ch. 8 and 11; SI units)
  joint dynamics          M(q) q'' + c(q, q') = tau                       (horizontal plane: g(q) = 0)
  handle kinematics       p = f(q),  p' = J(q) q',  p'' = J q'' + J' q'
  computed torque         tau = M~ J^-1 (u - J' q') + c~   gives  p'' = u  with an exact model (M~ = M, c~ = c)
  nominal command         u_nom = p_d'' + Kd (p_d' - p') + Kp (p_d - p)
  person (hand) centre    p_o(t), velocity v_o, acceleration a_o (known, e.g. from a tracker)
  relative position       e = p - p_o,  rho = |e|,  n = e / rho,  w = p' - v_o
  distance barrier        d = rho - r,   d' = n.w,   d'' = n.(u - a_o) + kappa,   kappa = (|w|^2 - (n.w)^2) / rho
  CBF conditions (both are linear in u and have the same normal n):
      keep-out     psi1 = d' + a1 d ,       psi1' + a2 psi1 >= 0   <=>  n.u >= b1
      speed cap    h_v  = v_max + d' ,      h_v'  + g  h_v  >= 0   <=>  n.u >= b2
      b1 = n.a_o - kappa - (a1 + a2) d' - a1 a2 d,      b2 = n.a_o - kappa - g (v_max + d')
  CBF-QP                  u* = argmin |u - u_nom|^2  s.t.  n.u >= b = max(b1, b2)
                          closed form  u* = u_nom + max(0, b - n.u_nom) n            (|n| = 1)

Standard sources (not in the course library): A. D. Ames et al., "Control barrier function based quadratic
programs for safety critical systems", IEEE Trans. Automatic Control 62(8), 2017; Q. Nguyen and K. Sreenath,
"Exponential control barrier functions for enforcing high relative-degree safety-critical constraints", ACC 2016.

Parts: A. barrier and filter (barrier_terms, cbf_bound, safety_filter, safety_filter_qp)
       B. the controller (nominal_accel, computed_torque, simulate)
       C. force limits near a person (apparent_mass, pfl_speed_limit, design_speed_limit)
Run this file to print the four experiments of each scenario (nominal, keep-out only, filter with the Week 13
speed 0.20 m/s, filter with your conservative speed).  ILLUSTRATIVE VALUES throughout.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import course_models as cm

HERE = Path(__file__).resolve().parent
ROBOT = cm.rehab_2r()
L1, L2 = ROBOT["params"]["L"]
M1, M2 = ROBOT["params"]["m"]
R_ROD = 0.03                                    # rod radius used by rod_inertia_x
LC1, LC2 = L1 / 2, L2 / 2                       # CoM distances
IZ1 = M1 * (3 * R_ROD**2 + L1**2) / 12.0        # inertia about the CoM, z axis
IZ2 = M2 * (3 * R_ROD**2 + L2**2) / 12.0


# =============================================================================== provided: complete, read but do not change
def load_scenario(name: str = "static") -> dict:
    """Scenario parameters from data/scenario.json (common values merged with the named scenario)."""
    data = json.loads((HERE / "data" / "scenario.json").read_text())
    scn = dict(data["common"])
    scn.update(data["scenarios"][name])
    scn["name"] = name
    return scn


def fk(q) -> np.ndarray:
    """Handle position p = (x, y) in the space frame {s} (MR Eqs. 4.1-4.2 with two links)."""
    q1, q2 = q
    return np.array([L1 * np.cos(q1) + L2 * np.cos(q1 + q2), L1 * np.sin(q1) + L2 * np.sin(q1 + q2)])


def jacobian(q) -> np.ndarray:
    """2x2 Jacobian of the handle position, p' = J(q) q'."""
    q1, q2 = q
    s1, c1, s12, c12 = np.sin(q1), np.cos(q1), np.sin(q1 + q2), np.cos(q1 + q2)
    return np.array([[-L1 * s1 - L2 * s12, -L2 * s12], [L1 * c1 + L2 * c12, L2 * c12]])


def jdot_qdot(q, qd) -> np.ndarray:
    """The velocity-product term J'(q, q') q' of the handle acceleration p'' = J q'' + J' q'."""
    q1, q2 = q
    w1, w12 = qd[0], qd[0] + qd[1]
    return np.array([-L1 * np.cos(q1) * w1**2 - L2 * np.cos(q1 + q2) * w12**2,
                     -L1 * np.sin(q1) * w1**2 - L2 * np.sin(q1 + q2) * w12**2])


def dynamics(q, qd):
    """Closed-form M(q) and c(q, q') of rehab_2r (uniform rods); equal to modern_robotics.MassMatrix and
    VelQuadraticForces (checked by the tests).  Gravity is along -z, so g(q) = 0 in this horizontal plane."""
    c2, s2 = np.cos(q[1]), np.sin(q[1])
    m22 = IZ2 + M2 * LC2**2
    m12 = m22 + M2 * L1 * LC2 * c2
    m11 = IZ1 + M1 * LC1**2 + IZ2 + M2 * (L1**2 + LC2**2 + 2 * L1 * LC2 * c2)
    h = M2 * L1 * LC2 * s2
    c = np.array([-h * (2 * qd[0] * qd[1] + qd[1] ** 2), h * qd[0] ** 2])
    return np.array([[m11, m12], [m12, m22]]), c


def ik(p, elbow: float = 1.0) -> np.ndarray:
    """Inverse kinematics of the handle; elbow = +1 gives theta2 > 0 (inside the joint limits of rehab_2r)."""
    x, y = p
    c2 = (x * x + y * y - L1**2 - L2**2) / (2 * L1 * L2)
    q2 = elbow * np.arccos(np.clip(c2, -1.0, 1.0))
    q1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    return np.array([q1, q2])


def quintic(t: float, T: float):
    """Quintic time scaling s(t), s'(t), s''(t) on [0, T] (MR Sec. 9.2.2.1, Fig. 9.4), held at the ends."""
    if t <= 0.0:
        return 0.0, 0.0, 0.0
    if t >= T:
        return 1.0, 0.0, 0.0
    r = t / T
    return (10 * r**3 - 15 * r**4 + 6 * r**5, (30 * r**2 - 60 * r**3 + 30 * r**4) / T,
            (60 * r - 180 * r**2 + 120 * r**3) / T**2)


def person_state(t: float, scn: dict):
    """Centre p_o, velocity v_o and acceleration a_o of the person's hand: a straight quintic move from
    person_p0 to person_p1 during [person_t0, person_t1] (a resting hand if p0 = p1)."""
    p0, p1 = np.asarray(scn["person_p0"], float), np.asarray(scn["person_p1"], float)
    t0, t1 = scn["person_t0"], scn["person_t1"]
    s, sd, sdd = quintic(t - t0, t1 - t0)
    return p0 + s * (p1 - p0), sd * (p1 - p0), sdd * (p1 - p0)


def reference(t: float, scn: dict):
    """Nominal straight-line reaching movement p_d(t), p_d'(t), p_d''(t) from p_A to p_B in T_move seconds."""
    pA, pB = np.asarray(scn["p_A"], float), np.asarray(scn["p_B"], float)
    s, sd, sdd = quintic(t, scn["T_move"])
    return pA + s * (pB - pA), sd * (pB - pA), sdd * (pB - pA)


# =============================================================================== Part A -- barrier and filter
def barrier_terms(p, v, p_o, v_o, r: float):
    """Distance barrier of the keep-out disc.
    Returns (d, d_dot, n, kappa): d = |p - p_o| - r, d_dot = n.(v - v_o), n = (p - p_o)/|p - p_o| and the
    curvature term kappa = (|w|^2 - (n.w)^2)/|p - p_o| with w = v - v_o, so that d'' = n.(u - a_o) + kappa."""
    # TODO: e = p - p_o, w = v - v_o, rho = |e|, n = e/rho; d = rho - r, d_dot = n.w, kappa = (|w|^2 - d_dot^2)/rho
    raise NotImplementedError("TODO: barrier_terms")


def cbf_bound(p, v, p_o, v_o, a_o, scn: dict):
    """The CBF constraint n.u >= b on the commanded handle acceleration u.
    Returns (n, b, b1, b2): b1 from the keep-out barrier (second-order CBF with gains alpha1, alpha2),
    b2 from the approach-speed barrier h_v = v_max + d' (gain gamma), b = max(b1, b2)."""
    # TODO: call barrier_terms; base = n.a_o - kappa; b1 = base - (a1 + a2) d' - a1 a2 d; b2 = base - gamma (v_max + d')
    raise NotImplementedError("TODO: cbf_bound")


def safety_filter(u_nom, a, b: float):
    """Closed-form solution of the one-constraint CBF-QP  min |u - u_nom|^2  s.t.  a.u >= b.
    Returns (u, lam): lam >= 0 is the size of the correction, u = u_nom + lam a / |a|^2 (lam = 0: inactive)."""
    # TODO: lam = max(0, b - a.u_nom); u = u_nom + lam a / (a.a)
    raise NotImplementedError("TODO: safety_filter")


def safety_filter_qp(u_nom, A, b):
    """General small CBF-QP with scipy:  min |u - u_nom|^2  s.t.  A u >= b  (one row per constraint).
    Returns the optimal u (numpy array)."""
    # TODO: scipy.optimize.minimize with method='SLSQP' and an 'ineq' constraint A u - b >= 0; raise if not res.success
    raise NotImplementedError("TODO: safety_filter_qp")


# =============================================================================== Part B -- the controller
def nominal_accel(t: float, p, v, scn: dict) -> np.ndarray:
    """Nominal task-space command u_nom = p_d'' + Kd (p_d' - p') + Kp (p_d - p)  (MR Eq. 11.47 in
    minimal coordinates, with Ki = 0)."""
    # TODO: use reference(t, scn) and the gains scn['Kp'], scn['Kd']
    raise NotImplementedError("TODO: nominal_accel")


def computed_torque(q, qd, u, model_scale: float = 1.0) -> np.ndarray:
    """Task-space computed torque tau = M~ J^-1 (u - J' q') + c~ with the controller's model
    M~ = model_scale * M, c~ = model_scale * c (all link masses and inertias scaled by model_scale)."""
    # TODO: qdd_cmd = J^-1 (u - J' q') (np.linalg.solve); tau = model_scale (M qdd_cmd + c)
    raise NotImplementedError("TODO: computed_torque")


def simulate(scn: dict, filter_on: bool = True, model_scale: float = 1.0, use_qp: bool = False) -> dict:
    """Closed-loop simulation.  The controller runs every dt seconds (zero-order hold); the robot
    M(q) q'' + c = tau is integrated with one RK4 step per control period.
    Returns a log (dict of numpy arrays): t, q, qd, p, v, p_o, v_o, u_nom, u, tau, d, approach (= -d'),
    active (filter correction > 0)."""
    # TODO: start at q = ik(p_A), q' = 0; each step: p, v -> person_state -> u_nom -> cbf_bound -> filter (if filter_on) -> computed_torque -> log -> one RK4 step of q'' = M^-1 (tau - c) with tau held constant
    raise NotImplementedError("TODO: simulate")


# =============================================================================== Part C -- force limits near a person
def apparent_mass(q, n) -> float:
    """Apparent (effective) mass of the handle along the unit direction n: m_R = 1 / (n^T J M^-1 J^T n)
    (MR Eq. 8.90: Lambda^-1 = J M^-1 J^T for this 2x2 Jacobian)."""
    # TODO: m_R = 1 / (n^T J M^-1 J^T n)
    raise NotImplementedError("TODO: apparent_mass")


def pfl_speed_limit(m_R: float, m_H: float, k: float, F_max: float) -> float:
    """Largest relative contact speed whose transient peak force stays below F_max (Week 13, Sec. 4.3):
    mu = (1/m_R + 1/m_H)^-1,  F = v sqrt(mu k)  =>  v_max = F_max / sqrt(mu k)."""
    # TODO: mu = (1/m_R + 1/m_H)^-1; v_max = F_max / sqrt(mu k)
    raise NotImplementedError("TODO: pfl_speed_limit")


def design_speed_limit(scn: dict, n_samples: int = 201):
    """Conservative approach-speed limit for the whole reaching movement.
    For the poses q = ik(p_d(t)) of the nominal path (n_samples instants on [0, T_move]) take the largest
    apparent mass in ANY direction, m_R,max = max lambda_max(Lambda(q)) with Lambda = (J M^-1 J^T)^-1, and
    return (v_design, m_R_max, v_exact): v_exact = pfl_speed_limit(m_R_max, m_H, k_contact, F_max) and
    v_design = v_exact rounded DOWN to 0.01 m/s."""
    # TODO: loop over np.linspace(0, T_move, n_samples); Lambda = inv(J M^-1 J^T); np.linalg.eigvalsh(Lambda).max(); round down with np.floor(v*100 + 1e-9)/100
    raise NotImplementedError("TODO: design_speed_limit")


def peak_contact_force(log: dict, scn: dict) -> np.ndarray:
    """Transient contact force F(t) = v_app(t) sqrt(mu(t) k) that a contact at time t would cause, using the
    apparent mass of the handle along -n (towards the hand) at the logged pose; zero when moving away."""
    F = np.zeros(len(log["t"]))
    for k in range(len(F)):
        e = log["p"][k] - log["p_o"][k]
        n = e / np.linalg.norm(e)
        mR = apparent_mass(log["q"][k], n)
        mu = 1.0 / (1.0 / mR + 1.0 / scn["m_H"])
        F[k] = max(0.0, log["approach"][k]) * np.sqrt(mu * scn["k_contact"])
    return F


def metrics(log: dict, scn: dict) -> dict:
    """Summary of a run: min_d (m), max_approach (m/s), max_tau_ratio (max |tau_i| / tau_max_i),
    active_fraction, final_error (m, |p(t_end) - p_B|), max_dev (m, largest |p - p_d|)."""
    pd = np.array([reference(t, scn)[0] for t in log["t"]])
    return dict(min_d=float(log["d"].min()), max_approach=float(log["approach"].max()),
                max_tau_ratio=float(np.max(np.abs(log["tau"]) / ROBOT["tau_max"])),
                active_fraction=float(log["active"].mean()),
                final_error=float(np.linalg.norm(log["p"][-1] - np.asarray(scn["p_B"]))),
                max_dev=float(np.max(np.linalg.norm(log["p"] - pd, axis=1))))


if __name__ == "__main__":
    for name in ("static", "approaching"):
        scn = load_scenario(name)
        v_design = design_speed_limit(scn)[0]
        runs = [("nominal (no filter)", scn, False),
                ("keep-out only", dict(scn, alpha1=10.0, v_max=1e3), True),
                (f"filter, v_max {scn['v_max']:.2f}", scn, True),
                (f"filter, v_max {v_design:.2f}", dict(scn, v_max=v_design), True)]
        for label, run_scn, on in runs:
            log = simulate(run_scn, filter_on=on)
            m = metrics(log, run_scn)
            m["F_peak"] = float(peak_contact_force(log, run_scn).max())
            print(f"{name:12s} {label:20s}", {k: round(v, 4) for k, v in m.items()})
