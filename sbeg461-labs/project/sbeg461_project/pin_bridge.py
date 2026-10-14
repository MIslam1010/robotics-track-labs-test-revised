"""Optional Pinocchio bridge: builds a Pinocchio model from the PoE description of a Scenario.

Used (1) to cross-check `modern_robotics` dynamics with an independent implementation
(Featherstone-style RNEA/CRBA/ABA, Featherstone 2008 Ch. 5-7), and (2) as a fast simulation
engine (ABA) for 6-DOF scenarios. `modern_robotics` stays the reference ("ground truth").

Construction, for each joint i (all frames at the home configuration):
  A_i   = [Ad_{M_si^-1}] S_i              screw axis in CoM frame {i} (change of frame, MR Sec. 3.3.2)
  r_i   = w_i x v_i                        point of the axis closest to the {i} origin (unit w_i)
  J_i   = M_si Trans(r_i)                  joint frame: on the axis, axes parallel to {i}
  placement_i = J_{i-1}^-1 J_i             relative to the parent joint frame (J_0 = identity)
"""
from __future__ import annotations

import modern_robotics as mr
import numpy as np

from .models import Scenario, spatial_inertia_to_mci, trans

try:
    import pinocchio as pin
    HAVE_PINOCCHIO = True
except Exception:  # pragma: no cover
    pin = None
    HAVE_PINOCCHIO = False


def to_pinocchio(s: Scenario):
    if not HAVE_PINOCCHIO:
        raise ImportError("pinocchio is not installed")
    model = pin.Model()
    model.gravity.linear = np.asarray(s.g, float)
    M_si = np.eye(4)
    J_prev = np.eye(4)
    parent = 0
    for i in range(s.n):
        M_si = M_si @ s.Mlist[i]
        A = mr.Adjoint(mr.TransInv(M_si)) @ s.Slist[:, i]
        w, v = A[:3], A[3:]
        if np.linalg.norm(w) > 1e-9:
            w = w / np.linalg.norm(w)
            r = np.cross(w, v)
            jm = pin.JointModelRevoluteUnaligned(w)
        else:
            r = np.zeros(3)
            jm = pin.JointModelPrismaticUnaligned(v / np.linalg.norm(v))
        J_i = M_si @ trans(r)
        Tpl = mr.TransInv(J_prev) @ J_i
        jid = model.addJoint(parent, jm, pin.SE3(Tpl[:3, :3], Tpl[:3, 3]), f"joint{i + 1}")
        m, c, Ic = spatial_inertia_to_mci(s.Glist[i])
        model.appendBodyToJoint(jid, pin.Inertia(m, c - r, Ic), pin.SE3.Identity())
        J_prev, parent = J_i, jid
    return model


class PinEngine:
    """Fast forward/inverse dynamics with Pinocchio for the scenario `s` (same conventions as dyn)."""

    def __init__(self, s: Scenario):
        self.s = s
        self.model = to_pinocchio(s)
        self.data = self.model.createData()

    def inverse_dynamics(self, q, qd, qdd, Ftip=None, g=None):
        self.model.gravity.linear = np.asarray(self.s.g if g is None else g, float)
        tau = pin.rnea(self.model, self.data, np.asarray(q, float), np.asarray(qd, float),
                       np.asarray(qdd, float)).copy()
        if Ftip is not None:
            tau += self.s.jacobian_body(q).T @ Ftip
        return tau

    def forward_dynamics(self, q, qd, tau, Ftip=None, g=None):
        self.model.gravity.linear = np.asarray(self.s.g if g is None else g, float)
        tau = np.asarray(tau, float)
        if Ftip is not None:
            tau = tau - self.s.jacobian_body(q).T @ Ftip
        return pin.aba(self.model, self.data, np.asarray(q, float), np.asarray(qd, float), tau).copy()

    def mass_matrix(self, q):
        Mq = pin.crba(self.model, self.data, np.asarray(q, float))
        return np.triu(Mq) + np.triu(Mq, 1).T
