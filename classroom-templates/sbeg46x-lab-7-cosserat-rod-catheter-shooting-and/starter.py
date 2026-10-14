"""SBEG 46X Lab 7 -- Cosserat-rod statics of a tendon-driven catheter: shooting and SciPy BVP solvers, comparison
with the constant-curvature (CC) and Euler-Bernoulli (EB) models under tip loads, and tip-force estimation.
STARTER CODE -- fill in every function marked TODO (in the order A1 ... D5).
Run the auto-checks with:  LAB_IMPL=starter python -m pytest -q test_lab.py   (or copy this file to your own module name)

These functions extend the `continuum` module of your cumulative library `sbeg46x_medrob` (started in Lab 6 with
the constant-curvature kinematics).

Model and sign conventions (the same as the Week 7 lectures and workshop)
  * Planar rod in the x-y plane, clamped at the origin with its base tangent along +x (theta0 = 0), arc length
    s in [0, L].  State z = [x, y, theta, nx, ny, m]:
        x' = cos(theta),  y' = sin(theta)                  (inextensible, unshearable: Kirchhoff rod)
        theta' = m / EI + kappa0                           (linear constitutive law, precurvature kappa0)
        n' = -f                                            (force balance, f = distributed force per length)
        m' = -(x' ny - y' nx)                              (moment balance, m' + (p' x n)_z = 0)
    n(s), m(s) = internal force and moment exerted by the distal part (s+) on the proximal part.
    Boundary conditions: x(0) = y(0) = 0, theta(0) = theta0;  n(L) = F, m(L) = M  (F, M = tip wrench).
  * A tendon routed parallel to the backbone at offset r on the +normal (+y at the base) side, tension tau and
    terminated at the tip, is exactly equivalent to an extra tip moment M_t = r * tau for an inextensible rod (its
    axial pull compresses the rod but does not bend it).  The tendon side becomes the inside of the bend.
  * F is the force applied TO the catheter tip (e.g. by tissue), in the base frame.
Units SI (m, N, N m, N m^2).  ILLUSTRATIVE VALUES (data/catheter.json): L = 0.10 m, EI = 1.0e-3 N m^2, r = 1.2 mm.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import modern_robotics as mr
import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import fsolve, least_squares

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RTOL, ATOL = 1e-10, 1e-13


# ==========================================================================================
# Provided (do not edit): data files and small helpers
# ==========================================================================================
def load_catheter() -> dict:
    """Catheter and sensor parameters from data/catheter.json (illustrative values): L (m), EI (N m^2),
    r (tendon offset, m), tau_nominal (N), sigma_p (EM-sensor position noise, m), sigma_theta (rad)."""
    return json.loads((DATA / "catheter.json").read_text())


def load_csv(name: str) -> dict:
    """Load a CSV file from data/ into a dict of float numpy arrays (one per column)."""
    with open(DATA / name, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def tip_pose(sol: dict) -> np.ndarray:
    """Tip pose [x, y, theta] of a rod solution returned by solve_rod_shooting / solve_rod_bvp."""
    return np.array([sol["x"][-1], sol["y"][-1], sol["theta"][-1]])


def contact_direction(theta_tip: float) -> np.ndarray:
    """Unit direction of a contact force that pushes the tip back against its bend: -(tip normal) =
    (sin theta, -cos theta).  A force F_n >= 0 along it 'unbends' the catheter (tissue resisting the tip)."""
    return np.array([np.sin(theta_tip), -np.cos(theta_tip)])


# ==========================================================================================
# Part A -- constant-curvature (CC) and piecewise-constant-curvature (PCC) models
# ==========================================================================================
def cc_tip(kappa: float, L: float) -> np.ndarray:
    """A1. Tip pose [x, y, theta] of a planar constant-curvature arc of curvature kappa and length L that leaves
    the origin along +x:  x = sin(kappa L)/kappa, y = (1 - cos(kappa L))/kappa, theta = kappa L.
    Must also work for kappa -> 0 (straight: [L, 0, 0]) without dividing by zero."""
    # TODO A1
    raise NotImplementedError("A1: cc_tip not implemented yet")


def tendon_curvature(tau: float, r: float, EI: float) -> float:
    """A2. Curvature of an unloaded tendon-driven segment: the tendon gives the constant moment r tau, so
    kappa = r tau / EI (exact for the Kirchhoff rod; this is why the CC model holds without external load)."""
    # TODO A2
    raise NotImplementedError("A2: tendon_curvature not implemented yet")


def cc_transform(kappa: float, L: float) -> np.ndarray:
    """A3. The same arc as an element of SE(3): T(L) = exp([V] L) with the constant body twist per unit arc
    length V = (omega, v) = (0, 0, kappa, 1, 0, 0) (MR Eq. 3.71 with s in place of t; MR Prop. 3.25, Eq. 3.88).
    Use modern_robotics (VecTose3, MatrixExp6)."""
    # TODO A3
    raise NotImplementedError("A3: cc_transform not implemented yet")


def pcc_tip(kappas, lengths) -> np.ndarray:
    """A4. Tip pose [x, y, theta] of a planar piecewise-constant-curvature (PCC) robot: segments i = 1..n with
    curvature kappas[i] and length lengths[i], each starting where (and tangent to where) the previous ended:
        p_i = p_{i-1} + R(theta_{i-1}) cc_xy(kappa_i, L_i),  theta_i = theta_{i-1} + kappa_i L_i."""
    # TODO A4
    raise NotImplementedError("A4: pcc_tip not implemented yet")


def pneumatic_curvature(p: float, A: float, d: float, EI: float) -> float:
    """A5. Linear bending model of a pneumatic soft segment: chamber pressure p (Pa, gauge) on area A (m^2) whose
    centroid is d (m) from the neutral axis gives the moment p A d, so kappa = p A d / EI (EI = effective bending
    stiffness of the elastomer body; calibrate it, real elastomers are nonlinear)."""
    # TODO A5
    raise NotImplementedError("A5: pneumatic_curvature not implemented yet")


# ==========================================================================================
# Part B -- Cosserat-rod statics: equations, shooting, SciPy BVP, Euler-Bernoulli check
# ==========================================================================================
def rod_rhs(s: float, z, EI: float, f=(0.0, 0.0), kappa0: float = 0.0) -> np.ndarray:
    """B1. Right-hand side dz/ds of the planar Cosserat (Kirchhoff) rod, z = [x, y, theta, nx, ny, m].
    f = (fx, fy) is a constant distributed force per unit length (e.g. weight), kappa0 the precurvature.
    Must also accept z of shape (6, k) (vectorised use by solve_bvp)."""
    # TODO B1
    raise NotImplementedError("B1: rod_rhs not implemented yet")


# Provided helper (do not edit): integrate the rod ODE from the base state z0 (uses your rod_rhs)
def _integrate(z0, L, EI, f, kappa0, s_eval=None):
    return solve_ivp(rod_rhs, (0.0, L), z0, args=(EI, f, kappa0), t_eval=s_eval, rtol=RTOL, atol=ATOL)


# Provided helper (do not edit)
def _rigid_guess(L, F, M, f):
    """Base wrench of the straight (undeformed) rod: a good first guess for shooting."""
    F, f = np.asarray(F, float), np.asarray(f, float)
    n0 = F + f * L
    m0 = M + L * F[1] + 0.5 * L**2 * f[1]
    return np.array([n0[0], n0[1], m0])


def shooting_residual(g, L: float, EI: float, F=(0.0, 0.0), M: float = 0.0, f=(0.0, 0.0),
                      kappa0: float = 0.0) -> np.ndarray:
    """B2. Shooting residual.  g = guessed base values [nx(0), ny(0), m(0)] (the unknown half of the initial state;
    x(0) = y(0) = theta(0) = 0 are known).  Integrate from s = 0 to L and return the tip-condition error
    [nx(L) - Fx, ny(L) - Fy, m(L) - M]."""
    # TODO B2
    raise NotImplementedError("B2: shooting_residual not implemented yet")


def solve_rod_shooting(L: float, EI: float, F=(0.0, 0.0), M: float = 0.0, f=(0.0, 0.0), kappa0: float = 0.0,
                       n_eval: int = 101, guess=None) -> dict:
    """B3. Solve the rod BVP by shooting: find g = [nx(0), ny(0), m(0)] with shooting_residual(g) = 0 (fsolve;
    start from the straight-rod statics unless `guess` is given), then integrate once more on
    s = linspace(0, L, n_eval).  Return dict(s, x, y, theta, nx, ny, m, g).
    Shortcut (allowed, used here): with f = 0 the internal force is constant, n(0) = F, and only m(0) is unknown
    (scalar shooting, secant iterations)."""
    # TODO B3
    # Hint: fsolve(shooting_residual, g0, args=(L, EI, F, M, f, kappa0)) from g0 = _rigid_guess(L, F, M, f);
    # then integrate once more with _integrate(..., s_eval=np.linspace(0, L, n_eval)).
    raise NotImplementedError("B3: solve_rod_shooting not implemented yet")


def solve_rod_bvp(L: float, EI: float, F=(0.0, 0.0), M: float = 0.0, f=(0.0, 0.0), kappa0: float = 0.0,
                  n_mesh: int = 41, n_eval: int = 101) -> dict:
    """B4. Solve the same BVP with scipy.integrate.solve_bvp (collocation): six boundary conditions
    x(0) = y(0) = theta(0) = 0, nx(L) = Fx, ny(L) = Fy, m(L) = M.  Initial guess on s = linspace(0, L, n_mesh):
    the straight rod with the rigid-body internal wrench.  Raise RuntimeError if the solver fails.
    Return dict(s, x, y, theta, nx, ny, m) evaluated on s = linspace(0, L, n_eval)."""
    # TODO B4
    # Hint: solve_bvp(fun, bc, s, Z0) with fun(s, Z) = rod_rhs(s, Z, EI, f, kappa0) and
    # bc(za, zb) returning the six boundary residuals; use res.sol(s_eval) for the output.
    raise NotImplementedError("B4: solve_rod_bvp not implemented yet")


def euler_bernoulli_tip(L: float, EI: float, P: float = 0.0, M: float = 0.0, w: float = 0.0) -> np.ndarray:
    """B5. Linear (small-deflection) Euler-Bernoulli cantilever: transverse tip force P (+y), tip moment M and
    uniform transverse load w per length (+y), superposed.  Return [x, y, theta] with x = L (no foreshortening):
        y = P L^3/(3EI) + M L^2/(2EI) + w L^4/(8EI),   theta = P L^2/(2EI) + M L/EI + w L^3/(6EI)."""
    # TODO B5
    raise NotImplementedError("B5: euler_bernoulli_tip not implemented yet")


# ==========================================================================================
# Part C -- catheter: Cosserat vs CC vs EB under tip loads
# ==========================================================================================
def catheter_tip(tau: float, F, cat: dict) -> np.ndarray:
    """C1. Cosserat tip pose [x, y, theta] of the tendon-driven catheter `cat` with tendon tension tau and tip
    force F (base frame): the rod of B3 with tip moment M = r tau."""
    # TODO C1
    raise NotImplementedError("C1: catheter_tip not implemented yet")


def compare_models(tau: float, Fn_values, cat: dict) -> dict:
    """C2. For contact forces of magnitude Fn (array) along contact_direction(theta_free) of the UNLOADED tip
    (theta_free = CC tip angle for tension tau), return
        dict(Fn, cosserat (k,3), cc (3,), eb (k,3), err_cc (k,), err_eb (k,))
    cc = CC tip pose (independent of the load), eb = euler_bernoulli_tip with P = Fy (the transverse component)
    and M = r tau, err_* = Euclidean tip-POSITION error (m) of that model w.r.t. Cosserat."""
    # TODO C2
    raise NotImplementedError("C2: compare_models not implemented yet")


# ==========================================================================================
# Part D -- tip-force estimation from the measured tip pose (EM sensor)
# ==========================================================================================
def compliance_matrix(tau: float, cat: dict, F0=(0.0, 0.0), h: float = 1e-3) -> np.ndarray:
    """D1. 3x2 tip compliance C = d[x, y, theta]/d[Fx, Fy] at tension tau and load F0, by central differences
    with step h (N) on the Cosserat model (catheter_tip)."""
    # TODO D1
    raise NotImplementedError("D1: compliance_matrix not implemented yet")


# Provided helpers (do not edit)
def _weights(cat):
    return np.diag([1 / cat["sigma_p"], 1 / cat["sigma_p"], 1 / cat["sigma_theta"]])


def free_tip_angle(tau: float, cat: dict) -> float:
    """Tip angle of the unloaded catheter (CC: kappa L with kappa = r tau / EI)."""
    return tendon_curvature(tau, cat["r"], cat["EI"]) * cat["L"]


def estimate_force_linear(pose_meas, tau: float, cat: dict, direction=None) -> np.ndarray:
    """D2. Linear (small-force) weighted least-squares estimate from dz = pose_meas - pose_free (pose_free = unloaded
    Cosserat tip pose), C = compliance_matrix(tau) at F0 = 0 and W = diag(1/sigma_p, 1/sigma_p, 1/sigma_theta):
      * direction=None:  both components,  F = (W C)^+ W dz;
      * direction = unit vector d (contact model: force along a known direction, e.g. the tissue normal):
        one unknown magnitude, c = C d, F = d (W c)^+ W dz.
    Returns F (2,)."""
    # TODO D2
    raise NotImplementedError("D2: estimate_force_linear not implemented yet")


def estimate_force_nl(pose_meas, tau: float, cat: dict, direction=None, F_init=None) -> np.ndarray:
    """D3. Nonlinear estimate: minimise || W (catheter_tip(tau, F) - pose_meas) ||^2 with
    scipy.optimize.least_squares over F (2 unknowns) or, if `direction` d is given, over the magnitude a in F = a d.
    Start from F_init (default: zero force).  Returns F (2,)."""
    # TODO D3
    # Hint: scipy.optimize.least_squares(lambda F: W @ (catheter_tip(tau, F, cat) - z), x0, x_scale=0.05)
    raise NotImplementedError("D3: estimate_force_nl not implemented yet")


def estimate_log(log: dict, cat: dict, method: str = "nl", normal_only: bool = True) -> np.ndarray:
    """D4. Estimate the tip force for every row of a log (columns tau, x_meas, y_meas, theta_meas): returns an
    (N, 2) array.  method = "linear" (D2) or "nl" (D3); normal_only=True uses the contact model
    direction = contact_direction(free_tip_angle(tau)) (force pushing the tip back against its bend)."""
    # TODO D4
    raise NotImplementedError("D4: estimate_log not implemented yet")


def force_errors(F_est, F_ref, theta_tip) -> dict:
    """D5. Error statistics of force estimates (N, 2) against reference forces (N, 2).  theta_tip (N,) is the tip
    angle used to split each error into the component along the tip normal (-sin, cos) and along the tip
    tangent (cos, sin).  Return dict(rms, rms_normal, rms_tangential) (N)."""
    # TODO D5
    raise NotImplementedError("D5: force_errors not implemented yet")
