"""Optimal control helpers (Scenarios 1, 3 and 6): discretisation, LQR and a small QP solver.

Provided: ``c2d`` (zero-order hold), ``dlqr`` (discrete LQR from the discrete algebraic Riccati
equation; Tedrake Ch. 8, Astrom-Murray §6.3), ``solve_qp`` (a dense QP solved with SciPy SLSQP: the
"SciPy fallback" of the Lab 8 handout; use a faster solver if you have one).
TEAM TODO (port your Lab 8 code): ``LinearMPC.solve``.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm, solve_discrete_are
from scipy.optimize import minimize


def c2d(A, B, T):
    """Zero-order-hold discretisation x[k+1] = Ad x[k] + Bd u[k] of x' = A x + B u."""
    A, B = np.atleast_2d(np.asarray(A, float)), np.asarray(B, float).reshape(len(A), -1)
    n, m = A.shape[0], B.shape[1]
    E = expm(np.block([[A, B], [np.zeros((m, n + m))]]) * T)
    return E[:n, :n], E[:n, n:]


def dlqr(Ad, Bd, Q, R):
    """Discrete LQR: returns (K, P) with u = -K x minimising sum x'Qx + u'Ru (DARE solution P)."""
    Ad, Bd = np.atleast_2d(Ad), np.atleast_2d(Bd)
    Q, R = np.atleast_2d(Q), np.atleast_2d(R)
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K, P


def solve_qp(H, f, A_ub=None, b_ub=None, lb=None, ub=None, x0=None):
    """min 1/2 x'Hx + f'x  s.t.  A_ub x <= b_ub,  lb <= x <= ub  (SciPy SLSQP fallback).
    Returns (x, info) with info = dict(success, fun, nit)."""
    H = np.atleast_2d(np.asarray(H, float))
    f = np.asarray(f, float).ravel()
    n = len(f)
    cons = []
    if A_ub is not None:
        A = np.atleast_2d(np.asarray(A_ub, float))
        b = np.asarray(b_ub, float).ravel()
        cons.append(dict(type="ineq", fun=lambda x: b - A @ x, jac=lambda x: -A))
    lb = -np.inf * np.ones(n) if lb is None else np.broadcast_to(np.asarray(lb, float), (n,))
    ub = np.inf * np.ones(n) if ub is None else np.broadcast_to(np.asarray(ub, float), (n,))
    bounds = [(None if not np.isfinite(l) else l, None if not np.isfinite(u) else u) for l, u in zip(lb, ub)]
    x0 = np.clip(np.zeros(n) if x0 is None else np.asarray(x0, float), np.where(np.isfinite(lb), lb, -1e9),
                 np.where(np.isfinite(ub), ub, 1e9))
    res = minimize(lambda x: 0.5 * x @ H @ x + f @ x, x0, jac=lambda x: H @ x + f, bounds=bounds,
                   constraints=cons, method="SLSQP", options=dict(maxiter=200, ftol=1e-12))
    return res.x, dict(success=bool(res.success), fun=float(res.fun), nit=int(res.nit))


class LinearMPC:
    """Linear MPC with a horizon of N steps for x[k+1] = Ad x[k] + Bd u[k]:
        min sum_{i<N} (x_i - x_ref)' Q (x_i - x_ref) + u_i' R u_i + (x_N - x_ref)' P (x_N - x_ref)
        s.t. u_min <= u_i <= u_max,  G x_i <= g  (state constraints, optional).
    TEAM TODO (port Lab 8): build the condensed QP and call ``solve_qp``; return the first input."""

    def __init__(self, Ad, Bd, Q, R, N, u_min=None, u_max=None, G=None, g=None, P=None):
        self.Ad, self.Bd, self.Q, self.R, self.N = (np.atleast_2d(Ad), np.atleast_2d(Bd),
                                                    np.atleast_2d(Q), np.atleast_2d(R), int(N))
        self.u_min, self.u_max, self.G, self.g = u_min, u_max, G, g
        self.P = P if P is not None else dlqr(self.Ad, self.Bd, self.Q, self.R)[1]

    def solve(self, x0, x_ref=None):
        raise NotImplementedError("LinearMPC.solve: port your Lab 8 MPC")
