"""SBEG 361 Lab 0 -- starter code (student copy).

Fill in every function marked TODO.  Run the checks with
    LAB_IMPL=starter pytest -q test_lab.py
(rename this file or set LAB_IMPL to your module name).  Read the lab handout first.

Topics: environment check, degrees of freedom (MR Sec. 2.1-2.2, Gruebler's formula MR Eq. 2.4),
C-space topology and representation (MR Sec. 2.3), first contact with the `modern_robotics` library.
"""
from __future__ import annotations

import json  # noqa: F401  (you will need it in dof_table)
import platform  # noqa: F401  (you will need it in env_report)
from pathlib import Path  # noqa: F401

import numpy as np  # noqa: F401

# Freedoms f of common joints, MR Table 2.1 (printed p. 17), plus three course extras:
#   RP = revolute-prismatic pair treated as one 2-dof joint (MR Ex 2.3 (ii), Ex 2.5 (ii))
#   E  = planar contact joint (a body sliding on a plane: x, y and rotation about the normal)
#   F  = rigid attachment ("weld", rigid grasp), f = 0
# TODO: complete the table (H, C, U, S, RP, E, F are missing).
JOINT_FREEDOMS = {"R": 1, "P": 1}
PLANAR_JOINTS = {"R", "P", "RP", "F"}


def env_report() -> dict:
    """Return the versions of Python, NumPy and modern_robotics as strings.

    Keys: "python", "numpy", "modern_robotics".
    Hint: platform.python_version(), np.__version__, importlib.metadata.version("modern_robotics").
    """
    # TODO
    raise NotImplementedError


def joint_freedoms(jtype: str) -> int:
    """Number of freedoms f provided by a joint of type `jtype` (MR Table 2.1).

    Raises ValueError for an unknown type.
    """
    # TODO
    raise NotImplementedError


def joint_constraints(jtype: str, m: int) -> int:
    """Number of constraints c = m - f provided by a joint (MR Table 2.1).

    m = 3 for planar, 6 for spatial mechanisms.  For planar mechanisms only R, P, RP and F are
    meaningful (H, C, U, S are "N/A" in MR Table 2.1): raise ValueError for the others, and for m
    not in {3, 6}.
    """
    # TODO
    raise NotImplementedError


def grubler(m: int, N: int, joints) -> int:
    """Gruebler's formula, MR Eq. 2.4:  dof = m (N - 1 - J) + sum_i f_i.

    m      : 3 (planar) or 6 (spatial); raise ValueError otherwise
    N      : number of links, ground INCLUDED
    joints : list of joint-type strings / integers f_i, or a dict {type: count}
    """
    # TODO
    raise NotImplementedError


def dof_table(path) -> dict:
    """Read a mechanisms JSON file (see data/mechanisms.json) and return {name: dof}."""
    # TODO
    raise NotImplementedError


def wrap_to_pi(theta):
    """Wrap angle(s) to the interval [-pi, pi).  Works on scalars and arrays."""
    # TODO
    raise NotImplementedError


def torus_distance(q1, q2) -> float:
    """Shortest distance between two configurations of an nR robot on T^n (no joint limits).

    Each joint difference is wrapped to [-pi, pi) before taking the Euclidean norm, because
    theta and theta + 2 pi are the same point of S^1 (MR Sec. 2.3.1).
    """
    # TODO
    raise NotImplementedError


def cspace_topology(joints) -> str:
    """Topology of the C-space of an open chain of R and P joints, ignoring joint limits.

    Each P joint contributes R^1, each R joint contributes S^1, and S^1 x ... x S^1 (n copies) = T^n
    (MR Sec. 2.3.1).  Format: "R^k x T^n", omitting a factor that does not appear; one R joint gives
    "S^1" (not "T^1").  Examples: ["P", "R"] -> "R^1 x S^1"; ["R", "R"] -> "T^2"; ["P"] -> "R^1".
    Raises ValueError for other joint types or an empty chain.
    """
    # TODO
    raise NotImplementedError


def latlong_to_xyz(lat, lon) -> np.ndarray:
    """Point on the unit sphere S^2 from latitude/longitude in radians (explicit parametrisation)."""
    # TODO
    raise NotImplementedError


def xyz_to_latlong(p) -> tuple:
    """Latitude in [-pi/2, pi/2] and longitude in [-pi, pi) of a (not necessarily unit) vector p."""
    # TODO
    raise NotImplementedError


def longitude_rate(lat, speed):
    """Longitude rate d(lon)/dt (rad/s) of a point moving due east on the unit sphere at angular
    speed `speed` (rad/s) at latitude `lat`.  Derive the formula in the pre-lab.
    """
    # TODO
    raise NotImplementedError


# ---- first contact with modern_robotics: planar 2R arm (same as MR Studio "Planar 2R arm") ----
def fk_2r_trig(theta, L1=0.6, L2=0.4) -> np.ndarray:
    """Tip position (x, y) of a planar 2R arm by trigonometry."""
    # TODO
    raise NotImplementedError


def fk_2r_mr(theta, L1=0.6, L2=0.4) -> np.ndarray:
    """4x4 tip pose T_sb of the planar 2R arm computed with modern_robotics.FKinSpace.

    Screw axes and home pose (MR Ch. 4, used here as a black box):
      S1 = (0,0,1, 0,0,0),  S2 = (0,0,1, 0,-L1,0),  M = [[I, (L1+L2, 0, 0)], [0, 1]].
    Put S1 and S2 as the COLUMNS of Slist (shape 6 x 2), then call mr.FKinSpace(M, Slist, theta).
    """
    # TODO
    raise NotImplementedError
