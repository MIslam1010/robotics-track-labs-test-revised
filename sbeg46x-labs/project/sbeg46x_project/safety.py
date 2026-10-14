"""Safety filters (all scenarios).

Provided BASELINES: ``saturate_toward_limit`` (clip the command at the limit: what a plain software
limit does) and ``virtual_wall_force`` (a stiff spring at the limit); ``barrier_report``.
TEAM TODO (port your Lab 10 code): ``cbf_filter_1d`` (closed-form CBF-QP for one input) and
``cbf_qp`` (several constraints, solved with ``optctrl.solve_qp``). Ames et al. (ECC 2019);
Tedrake §9.1.6.
"""
from __future__ import annotations

import numpy as np


def saturate_toward_limit(x, u, x_min=-np.inf, x_max=np.inf):
    """BASELINE kinematic limit: a velocity (or position-step) command u that would move x outside
    [x_min, x_max] is set to zero once x has reached the limit. It acts only AT the limit, so a
    second-order plant overshoots it."""
    if (x >= x_max and u > 0) or (x <= x_min and u < 0):
        return 0.0
    return u


def virtual_wall_force(x, v, x_min=-np.inf, x_max=np.inf, K=200.0, B=0.0):
    """BASELINE force limit: a one-sided spring-damper that pushes back once x is past a limit."""
    if x > x_max:
        return -K * (x - x_max) - B * max(v, 0.0)
    if x < x_min:
        return -K * (x - x_min) - B * min(v, 0.0)
    return 0.0


def barrier_report(h, tol=0.0) -> dict:
    h = np.asarray(h, float)
    return dict(h_min=float(np.min(h)), n_violations=int(np.sum(h < -tol)))


def cbf_filter_1d(u_nom, h, Lf_h, Lg_h, alpha):
    """TEAM TODO (port Lab 10). Closed-form CBF-QP with one input:
    min (u - u_nom)^2  s.t.  Lf_h + Lg_h u >= -alpha h. Return the filtered input."""
    raise NotImplementedError("cbf_filter_1d: port your Lab 10 code")


def cbf_qp(u_nom, A, b, W=None, u_min=None, u_max=None):
    """TEAM TODO (port Lab 10). CBF-QP with several inputs and constraints A u >= b (one row per
    barrier: A = Lg_h, b = -Lf_h - alpha(h)), weighted by W: min (u - u_nom)' W (u - u_nom)."""
    raise NotImplementedError("cbf_qp: port your Lab 10 code")
