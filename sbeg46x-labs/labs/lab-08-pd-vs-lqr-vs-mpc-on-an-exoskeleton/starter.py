"""SBEG 46X Lab 8 -- PD vs LQR vs MPC on an exoskeleton joint with torque and range limits (formal report 2).
STARTER CODE -- you edit this file.  Its functions join the module `optctrl` of your library sbeg46x_medrob.
Write every function marked TODO (keep the names and signatures); the helpers in the "Provided" section are
given (do not change them).  Run the checks with:  LAB_IMPL=starter python -m pytest -q test_lab.py
Checkpoint 1 (state space, ZOH, PD gains, Riccati equations, Bryson weights) is a warm-up: a recap of the LQR of
SBEG 461 Week 7.  The new work is Checkpoint 2 (MPC as a QP) and the closed-loop experiments.

Plant (MR Sec. 11.4.1, Eq. 11.21, angle measured from the hanging position):
    M theta_ddot + b theta_dot + g(theta) = sat(tau),   g(theta) = g_s sin(theta) + g_c cos(theta),
    sat(tau) = clip(tau, -tau_max, tau_max)   (the actuator torque limit),
with the elbow of the shared course model arm_exo_2r (upper arm hanging): M = 0.0778 kg m^2, g_s = 3.3354 N m.
Every controller adds the model gravity torque and designs the remaining torque u (PD + gravity compensation,
MR Eq. 11.38 with K_i = 0):
    tau = g(theta) + u,   so that   M theta_ddot + b theta_dot = u   (while tau is not saturated).
Error state  x = (theta - theta_d, theta_dot);  continuous model  x_dot = A x + B u,  A = [[0, 1], [0, -b/M]],
B = [0, 1/M]^T;  discrete (zero-order hold, period h)  x[n+1] = Ad x[n] + Bd u[n].
Units: SI (rad, rad/s, N m, s).  All values are illustrative.
References: Tedrake, Underactuated Robotics, Ch. 8 (Sec. 8.1 continuous LQR, Sec. 8.3.1 discrete Riccati
equations, Sec. 8.3.2 constraints -> MPC) and Ch. 10 (Sec. 10.2.1 direct transcription, Sec. 10.2.2 direct
shooting, Sec. 10.5.2 MPC); Astrom & Murray, Feedback Systems, Sec. 6.3 (Eq. 6.26-6.27) and Exercise 6.15
(Bryson's rule); MR Sec. 10.7 (Eqs. 10.6-10.12) and Sec. 11.4.1.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import expm, solve_continuous_are
from scipy.optimize import nnls

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


# ==========================================================================================
# Provided (do not edit): parameters, set-points, gravity, QP solver, simulator
# ==========================================================================================
def load_params(path=DATA / "exo_joint.json") -> dict:
    """Read the lab parameters (data/exo_joint.json, illustrative values).  Keys: 'elbow' (main lab) and
    'knee' (optional Experiment E), each with the joint model, limits, task and controller settings."""
    with open(path) as f:
        return json.load(f)


def setpoint(t: float, task: dict) -> float:
    """Piecewise-constant set-point theta_d(t): task['steps'] = [[t_0, theta_0], [t_1, theta_1], ...]."""
    th = task["steps"][0][1]
    for t_k, th_k in task["steps"]:
        if t >= t_k - 1e-12:
            th = th_k
    return float(th)


def gravity(theta: float, j: dict, scale: float = 1.0) -> float:
    """g(theta) = scale (g_s sin theta + g_c cos theta); scale != 1 models a heavier/lighter limb than the model."""
    return scale * (j["g_s"] * math.sin(theta) + j["g_c"] * math.cos(theta))


def solve_qp(H, f, G=None, w=None, Aeq=None, beq=None, tol=1e-9):
    """Solve   min 1/2 z^T H z + f^T z   s.t.   G z <= w,  Aeq z = beq   (H symmetric positive definite).

    Course fallback solver (only NumPy/SciPy needed; the handout names faster QP solvers you may use instead):
    equality constraints are eliminated with a null-space basis, the remaining problem is turned into a
    least-distance problem  min |y|  s.t.  M y <= d  (y = R z + R^{-T} f, H = R^T R) and solved exactly with
    non-negative least squares (Lawson & Hanson's LDP method; scipy.optimize.nnls, an active-set method).
    Returns (z, status) with status 'optimal' or 'infeasible'.
    """
    H = np.atleast_2d(np.asarray(H, float))
    f = np.asarray(f, float).ravel()
    n = f.size
    G = np.zeros((0, n)) if G is None else np.atleast_2d(np.asarray(G, float))
    w = np.zeros(0) if w is None else np.asarray(w, float).ravel()
    zp, N = np.zeros(n), np.eye(n)
    if Aeq is not None and np.size(Aeq) > 0:
        Aeq = np.atleast_2d(np.asarray(Aeq, float))
        beq = np.asarray(beq, float).ravel()
        zp = np.linalg.lstsq(Aeq, beq, rcond=None)[0]
        if np.linalg.norm(Aeq @ zp - beq) > 1e-8 * (1 + np.linalg.norm(beq)):
            return zp, "infeasible"
        _, s, Vt = np.linalg.svd(Aeq)
        rank = int(np.sum(s > 1e-10 * max(1.0, s[0])))
        N = Vt[rank:].T
    Hr = N.T @ H @ N
    fr = N.T @ (H @ zp + f)
    Gr = G @ N
    wr = w - G @ zp
    Hr = 0.5 * (Hr + Hr.T)
    v_free = -np.linalg.solve(Hr, fr)
    if Gr.shape[0] == 0 or np.all(Gr @ v_free <= wr + tol):
        return zp + N @ v_free, "optimal"
    R = np.linalg.cholesky(Hr).T                      # Hr = R^T R
    Rinv = np.linalg.inv(R)
    Mc = Gr @ Rinv
    d = wr - Gr @ v_free                              # constraints on y = R (v - v_free)
    nv = Hr.shape[0]
    E = np.vstack([-Mc.T, -d[None, :]])               # LDP:  (-Mc) y >= -d
    e = np.r_[np.zeros(nv), 1.0]
    lam, _ = nnls(E, e, maxiter=50 * E.shape[1])
    res = E @ lam - e
    if np.linalg.norm(res) < 1e-10:
        return zp + N @ v_free, "infeasible"
    y = -res[:nv] / res[nv]
    v = v_free + Rinv @ y
    z = zp + N @ v
    if np.any(G @ z > w + 1e-6 * (1 + np.abs(w))):
        return z, "infeasible"
    return z, "optimal"


def error_limits(theta_d: float, j: dict):
    """State limits in error coordinates x = (theta - theta_d, theta_dot) for the range of motion and speed."""
    lo = np.array([j["rom"][0] - theta_d, -j["v_max"]])
    hi = np.array([j["rom"][1] - theta_d, j["v_max"]])
    return lo, hi


def make_controller(kind: str, j: dict):
    """Build the controller data once (gains, Riccati solutions).  Uses YOUR functions."""
    A, B = joint_state_space(j["M"], j["b"])
    Ad, Bd = discretize_zoh(A, B, j["h"])
    c = dict(kind=kind, A=A, B=B, Ad=Ad, Bd=Bd)
    if kind == "pd":
        c["Kp"], c["Kd"] = pd_gains(j["M"], j["b"], j["pd"]["zeta"], j["pd"]["wn"])
    else:
        Q, R = bryson_weights(j["lqr"]["x_max"], j["lqr"]["u_max"])
        K, S = dlqr(Ad, Bd, Q, R)
        c.update(Q=Q, R=R, K=K, S=S)
        if kind == "mpc":
            c["N"] = int(j["mpc"]["N"])
            u_m = j["tau_max"] - j["mpc"]["g_reserve"]
            c["u_lim"] = (np.array([-u_m]), np.array([u_m]))
    return c


def simulate(kind: str, j: dict, T: float | None = None, g_scale: float = 1.0, task: dict | None = None,
             substeps: int = 10, ctrl: dict | None = None) -> dict:
    """Simulate the exoskeleton joint under controller kind = 'pd', 'lqr' or 'mpc'.

    The controller runs every h seconds (zero-order hold); the plant M theta_ddot = sat(tau) - b theta_dot -
    g_scale g(theta) is integrated with RK4 (`substeps` per control period).  The controller always uses the
    MODEL gravity g(theta) (g_scale = 1); g_scale != 1 tests a limb heavier or lighter than the model.
    Returns a dict of arrays sampled at the control instants (t, theta, theta_dot, theta_d, u, tau_cmd, tau)
    plus 'status' (the MPC QP status per sample) and the controller data 'ctrl'.
    """
    task = task or j["task"]
    T = task["T"] if T is None else T
    h = j["h"]
    n_steps = int(round(T / h))
    c = ctrl or make_controller(kind, j)
    th, om = task["theta0"], 0.0
    out = {k: np.zeros(n_steps) for k in ("t", "theta", "theta_dot", "theta_d", "u", "tau_cmd", "tau")}
    status = []
    for n in range(n_steps):
        t = n * h
        th_d = setpoint(t, task)
        x = np.array([th - th_d, om])
        if kind == "pd":
            u = c["Kp"] * (th_d - th) - c["Kd"] * om
            st = "-"
        elif kind == "lqr":
            u = float(-(c["K"] @ x)[0])
            st = "-"
        elif kind == "mpc":
            lo, hi = error_limits(th_d, j)
            u, st = mpc_control(x, c["Ad"], c["Bd"], c["Q"], c["R"], c["S"], c["N"], c["u_lim"], (lo, hi))
            if st != "optimal":                     # state constraints cannot be met: drop them for this sample
                u, st2 = mpc_control(x, c["Ad"], c["Bd"], c["Q"], c["R"], c["S"], c["N"], c["u_lim"], None)
                st = "state-constraints-dropped"
        else:
            raise ValueError(kind)
        tau_cmd = gravity(th, j) + u
        tau = float(np.clip(tau_cmd, -j["tau_max"], j["tau_max"]))
        for k, v in zip(out, (t, th, om, th_d, u, tau_cmd, tau)):
            out[k][n] = v
        status.append(st)

        def f(s):
            return np.array([s[1], (tau - j["b"] * s[1] - gravity(s[0], j, g_scale)) / j["M"]])

        s = np.array([th, om])
        dt = h / substeps
        for _ in range(substeps):
            k1 = f(s)
            k2 = f(s + 0.5 * dt * k1)
            k3 = f(s + 0.5 * dt * k2)
            k4 = f(s + dt * k3)
            s = s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        th, om = float(s[0]), float(s[1])
    out["status"] = status
    out["ctrl"] = c
    return out


# ==========================================================================================
# Your functions (module optctrl) -- write the bodies marked TODO
# ==========================================================================================
def joint_state_space(M: float, b: float):
    """Continuous error model of the gravity-compensated joint  M theta_ddot + b theta_dot = u:
    x = (theta - theta_d, theta_dot), x_dot = A x + B u.  Returns A (2x2), B (2x1)."""
    # TODO: write A and B from M theta_ddot + b theta_dot = u with x = (theta - theta_d, theta_dot).
    raise NotImplementedError("joint_state_space")


def discretize_zoh(A, B, h: float):
    """Exact zero-order-hold discretisation: Ad = e^{A h}, Bd = int_0^h e^{A s} ds B.
    Use the matrix exponential of the block matrix [[A, B], [0, 0]] h."""
    # TODO: build Z = [[A, B], [0, 0]] (size n+m), E = scipy.linalg.expm(Z h); Ad = E[:n, :n], Bd = E[:n, n:].
    raise NotImplementedError("discretize_zoh")


def pd_gains(M: float, b: float, zeta: float, wn: float):
    """PD gains (with gravity compensation) that give M e'' + (b + Kd) e' + Kp e = 0 the natural frequency wn
    and damping ratio zeta (MR Eqs. 11.25-11.26; SBEG 461 Week 5).  Returns (Kp, Kd)."""
    # TODO: match M s^2 + (b + Kd) s + Kp to M (s^2 + 2 zeta wn s + wn^2).
    raise NotImplementedError("pd_gains")


def scalar_care(a: float, b: float, q: float, r: float):
    """Scalar continuous LQR: x_dot = a x + b u, J = int (q x^2 + r u^2) dt.  The Riccati equation
    2 a p - p^2 b^2 / r + q = 0 has the positive root p; u = -k x with k = b p / r.  Returns (p, k)."""
    # TODO: solve the quadratic in p and keep the POSITIVE root (Workshop Problem 1).
    raise NotImplementedError("scalar_care")


def scalar_dare(a: float, b: float, q: float, r: float):
    """Scalar discrete LQR: x[n+1] = a x[n] + b u[n], J = sum (q x^2 + r u^2).  The fixed point of
    s = q + a^2 s - a^2 b^2 s^2 / (r + b^2 s) is the positive root of
    b^2 s^2 + (r (1 - a^2) - q b^2) s - q r = 0;  u = -k x with k = a b s / (r + b^2 s).  Returns (s, k)."""
    # TODO: solve the quadratic in s and keep the POSITIVE root (Workshop Problem 2).
    raise NotImplementedError("scalar_dare")


def lqr(A, B, Q, R):
    """Continuous infinite-horizon LQR (Tedrake Sec. 8.1; Astrom & Murray Eq. 6.27):
    S A + A^T S - S B R^{-1} B^T S + Q = 0,  K = R^{-1} B^T S.  Returns (K, S).
    (scipy.linalg.solve_continuous_are may be used.)"""
    # TODO: S from scipy.linalg.solve_continuous_are(A, B, Q, R); K = R^{-1} B^T S (np.linalg.solve, not inv).
    raise NotImplementedError("lqr")


def riccati_recursion(A, B, Q, R, Qf, N: int):
    """Finite-horizon discrete LQR by the Riccati difference equation (Tedrake Sec. 8.3.1), backwards from
    S[N] = Qf:   K[n] = (R + B^T S[n+1] B)^{-1} B^T S[n+1] A,
                 S[n] = Q + A^T S[n+1] A - A^T S[n+1] B K[n].
    Returns (K_list, S_list) with K_list[n] for n = 0..N-1 and S_list[n] for n = 0..N.
    (Index convention: the gain applied at step n uses S[n+1].)"""
    # TODO: loop n = N-1 down to 0; keep S symmetric (0.5 (S + S^T)).
    raise NotImplementedError("riccati_recursion")


def dlqr(A, B, Q, R, tol: float = 1e-12, max_iter: int = 100000):
    """Infinite-horizon discrete LQR: iterate the Riccati difference equation from S = Q until
    max|S_new - S| < tol * max(1, max|S|) (the DARE fixed point).  Returns (K, S), u = -K x."""
    # TODO: iterate the same update as riccati_recursion until it stops changing; raise RuntimeError if it does not.
    raise NotImplementedError("dlqr")


def bryson_weights(x_max, u_max):
    """Bryson's rule (Astrom & Murray Exercise 6.15): Q = diag(1/x_max_i^2), R = diag(1/u_max_j^2)."""
    # TODO: two diagonal matrices; accept lists or scalars.
    raise NotImplementedError("bryson_weights")


def prediction_matrices(A, B, N: int):
    """Direct-shooting (condensed) prediction X = Sx x0 + Su U with X = [x1; ...; xN] (N n x 1) and
    U = [u0; ...; u_{N-1}] (N m x 1):  block row k (k = 1..N) of Sx is A^k, block (k, i) of Su is A^{k-1-i} B
    for i < k (zero otherwise).  Returns (Sx, Su)."""
    # TODO: x_k = A^k x0 + sum_{i<k} A^{k-1-i} B u_i; fill the blocks with a double loop.
    raise NotImplementedError("prediction_matrices")


def _limit_rows(N, n, lim_lo, lim_hi):
    """PROVIDED HELPER (do not edit): rows of  [I; -I] X <= [hi; -lo]  for the finite entries of the per-step
    bounds lim_lo, lim_hi (length n each) repeated over N steps.  Returns (G, w)."""
    lo = np.tile(np.asarray(lim_lo, float).ravel(), N)
    hi = np.tile(np.asarray(lim_hi, float).ravel(), N)
    I = np.eye(N * n)
    keep_hi, keep_lo = np.isfinite(hi), np.isfinite(lo)
    return np.vstack([I[keep_hi], -I[keep_lo]]), np.r_[hi[keep_hi], -lo[keep_lo]]


def mpc_qp(A, B, Q, R, P, N: int, x0, u_lim, x_lim=None):
    """Condensed (direct-shooting) MPC QP:  min_U  sum_{k=0}^{N-1} (x_k^T Q x_k + u_k^T R u_k) + x_N^T P x_N
    written as  1/2 U^T H U + f^T U + const  with  H = 2 (Su^T Qb Su + Rb),  f = 2 Su^T Qb Sx x0,
    Qb = blkdiag(Q, ..., Q, P) over x1..xN, Rb = blkdiag(R, ..., R)  (the x0 term is constant).
    Constraints G U <= w:  u_lim = (u_lo, u_hi) on every u_k;  x_lim = (x_lo, x_hi) on x1..xN (None = no state
    limits; +-inf entries are skipped).  Returns (H, f, G, w)."""
    # TODO: use prediction_matrices and the provided _limit_rows; remember the terminal weight P on x_N.
    raise NotImplementedError("mpc_qp")


def transcription_qp(A, B, Q, R, P, N: int, x0, u_lim, x_lim=None):
    """Direct-transcription MPC QP (Tedrake Sec. 10.2.1): decision z = [x1; ...; xN; u0; ...; u_{N-1}],
    cost 1/2 z^T H z + f^T z with H = 2 blkdiag(Qb, Rb), f = 0 (x0^T Q x0 is constant),
    dynamics as equality constraints  x_{k+1} - A x_k - B u_k = 0  (k = 0..N-1, x_0 given): Aeq z = beq,
    bounds as G z <= w.  Returns (H, f, G, w, Aeq, beq)."""
    # TODO: one block row of Aeq per step k: x_{k+1} - A x_k - B u_k = 0, with A x0 moved to beq for k = 0.
    raise NotImplementedError("transcription_qp")


def mpc_control(x0, A, B, Q, R, P, N: int, u_lim, x_lim=None):
    """Receding-horizon control law (Tedrake Sec. 10.5.2): build the condensed QP with mpc_qp, solve it with
    solve_qp, and apply only the FIRST input.  Returns (u0 as float for one input / array otherwise, status)."""
    # TODO: mpc_qp -> solve_qp -> return the first input and the status.
    raise NotImplementedError("mpc_control")


def metrics(sim: dict, j: dict, band: float = 0.02) -> dict:
    """Performance and safety numbers of one run:
      rom_violation   max over time of max(theta - rom_hi, rom_lo - theta, 0)            [rad]
      v_violation     max over time of max(|theta_dot| - v_max, 0)                        [rad/s]
      overshoot       max over each set-point segment of the motion past the new set-point,
                      in the direction of the step (0 if none); the largest over all steps [rad]
      settle_time     for each step, the time from the step until |theta - theta_d| stays <= band to the end of
                      the segment (nan if never); the LARGEST over all steps                [s]
      peak_tau        max |tau| applied                                                   [N m]
      sat_fraction    fraction of samples with |tau_cmd| > tau_max                        [-]
      rms_error       sqrt(mean((theta - theta_d)^2)) over the whole run                  [rad]
      effort          h * sum(tau^2)                                                      [N^2 m^2 s]
    Returns a dict with these keys."""
    # TODO: follow the definitions above exactly (the tests compare with reference values).
    raise NotImplementedError("metrics")
