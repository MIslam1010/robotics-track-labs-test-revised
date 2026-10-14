"""The two robots of SBEG 461 Lab 7: the planar 2R arm (closed-form dynamics) and the course arm (MR lists).

# from the shared course robot models (Course resources): planar_2r (L = 0.6, 0.4 m, uniform rods m = 2, 1 kg,
# radius 0.025 m, vertical x-y plane, gravity -y) and course_arm (copied as course_arm.py). Numbers are IDENTICAL
# to the shared models; copied here so that the lab folder is self-contained.

Both models are dicts with the Modern Robotics lists (Slist, M, Mlist, Glist, g), so that modern_robotics can
check every number, plus `b` (viscous joint friction, N m s/rad; illustrative values) and `closed_form` (the
parameters of the closed-form 2R dynamics, None for the course arm, whose dynamics is evaluated with
modern_robotics).
"""
from __future__ import annotations

import modern_robotics as mr
import numpy as np

import course_arm as ca

G0 = 9.81


def _rod_inertia(m: float, L: float, r: float = 0.025) -> np.ndarray:
    """Spatial inertia of a uniform rod along x (radius r) in its CoM frame (MR Eq. 8.32): MR Studio rod_inertia."""
    Ixx = 0.5 * m * r**2
    Iyy = Izz = m * (3 * r**2 + L**2) / 12.0
    return np.diag([Ixx, Iyy, Izz, m, m, m])


def planar_2r(b=(0.0, 0.0)) -> dict:
    """Planar 2R in the vertical x-y plane (gravity along -y), joints about z, zero pose along +x.
    Uniform rods L = (0.6, 0.4) m, m = (2, 1) kg (shared model `planar_2r`, MR Sec. 8.1, MR Studio "Planar 2R arm").
    `b` = viscous friction per joint (N m s/rad)."""
    L1, L2, m1, m2, r = 0.6, 0.4, 2.0, 1.0, 0.025
    S = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, -L1, 0]], float).T
    M = mr.RpToTrans(np.eye(3), [L1 + L2, 0, 0])
    Mlist = [mr.RpToTrans(np.eye(3), p) for p in ([L1 / 2, 0, 0], [(L1 + L2) / 2, 0, 0], [L2 / 2, 0, 0])]
    Glist = [_rod_inertia(m1, L1, r), _rod_inertia(m2, L2, r)]
    cf = dict(L1=L1, L2=L2, m1=m1, m2=m2, lc1=L1 / 2, lc2=L2 / 2,
              I1=Glist[0][2, 2], I2=Glist[1][2, 2], g0=G0)
    return dict(key="planar_2r", name="Planar 2R arm", n=2, Slist=S, M=M, Mlist=Mlist, Glist=Glist,
                g=np.array([0.0, -G0, 0.0]), tau_max=np.array([40.0, 15.0]), q_home=np.array([0.6, 0.9]),
                b=np.asarray(b, float), closed_form=cf)


def course_arm_model(b=None) -> dict:
    """The course arm (course_arm.py, identical to the shared model), no payload, with friction `b`
    (zeros if None) and closed_form = None."""
    r = ca.course_arm()
    r["key"] = "course_arm"
    r["b"] = np.zeros(6) if b is None else np.asarray(b, float)
    r["closed_form"] = None
    return r
