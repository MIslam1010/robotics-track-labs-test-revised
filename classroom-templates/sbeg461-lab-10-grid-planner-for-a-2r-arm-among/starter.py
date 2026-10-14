"""SBEG 461 Lab 10 -- Grid planner for a planar 2R arm among obstacles (MR Sec. 10.2-10.4).
STARTER -- you edit this file. Replace every `raise NotImplementedError` with your code.
Run the auto-checks with:  LAB_IMPL=starter python -m pytest -q test_lab.py
Rules: do not import modern_robotics, scipy or any planning library here (the tests use them as ground truth).

Notation follows Modern Robotics (MR): configuration q = (theta1, theta2), C-space C, free space C_free,
C-obstacles C_obs (MR Sec. 10.2.1), distance d(q, B) (MR Sec. 10.2.2), A* search (MR Algorithm 10.1),
Dijkstra = A* with zero heuristic (MR Sec. 10.2.4.2), grid methods (MR Sec. 10.4).

Parts
  A. Geometry and collision checking: FK of the 2R arm, capsule-circle and capsule-floor distances,
     clearance d(q), config_free                                                    MR Sec. 10.2.1-10.2.2
  B. C-space grid and edge checking: the k x k grid of free cells, a sampled edge check, and a
     certified (conservative, "grown robot") edge check                              MR Sec. 10.2.2, 10.4
  C. Graph search: a generic A* with optional node reopening (for inconsistent heuristics),
     grid neighbours (4/8-connected, with or without wrap-around), heuristics       MR Algorithm 10.1, Sec. 10.4
  D. The planner: snap start/goal to the grid, search with edge checks, return a continuous joint path
"""
from __future__ import annotations

import heapq
import itertools
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# ==========================================================================================
# Provided (do not edit): the robot, the scenes and small grid helpers
# ==========================================================================================

# from the shared course robot models (Course resources): planar_2r  (MR Sec. 4.1 / 8.1; MR Studio "Planar 2R arm")
L_PLANAR_2R = (0.6, 0.4)                      # link lengths (m)
LIMITS_PLANAR_2R = ((-math.pi, math.pi), (-math.pi, math.pi))
SLIST_PLANAR_2R = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, -0.6, 0]], float).T
M_PLANAR_2R = np.array([[1, 0, 0, 1.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)


def load_scene(name: str) -> dict:
    """Return the scene `name` from data/scenes.json as a dict with numpy arrays:
    L (2,), link_radius, circles (m x 3: cx, cy, r), floor (y of a floor line or None), wrap (bool),
    q_start, q_goal (rad), description."""
    with open(HERE / "data" / "scenes.json") as f:
        s = json.load(f)[name]
    out = dict(s)
    out["L"] = np.asarray(s["L"], float)
    out["circles"] = np.asarray(s["circles"], float).reshape(-1, 3)
    out["q_start"] = np.radians(s["q_start_deg"])
    out["q_goal"] = np.radians(s["q_goal_deg"])
    return out


def wrap_angle(a):
    """Map angles to [-pi, pi)."""
    return (np.asarray(a, float) + math.pi) % (2 * math.pi) - math.pi


# ==========================================================================================
# Part A -- geometry and collision checking (MR Sec. 10.2.1-10.2.2)
# ==========================================================================================

def fk_2r(q, L=L_PLANAR_2R):
    """Joint and tip positions of the planar 2R arm.
    Returns (p0, p1, p2): base (origin), elbow and tip, each a numpy array (2,).
    p1 = L1 (c1, s1), p2 = p1 + L2 (c12, s12)  (MR Sec. 4.1, Fig. 10.2)."""
    # TODO: p1 = L1 (cos t1, sin t1); p2 = p1 + L2 (cos(t1 + t2), sin(t1 + t2)); p0 = (0, 0).
    raise NotImplementedError


def point_segment_distance(p, a, b) -> float:
    """Euclidean distance from point p to the segment [a, b] (all 2-vectors)."""
    # TODO: Project p onto the line through a and b: t = (p - a).(b - a) / |b - a|^2,
    #       clip t to [0, 1], return |a + t (b - a) - p|. Guard against a == b.
    raise NotImplementedError


def capsule_circle_distance(a, b, r_link, c, r_obs) -> float:
    """Signed distance between a capsule (segment [a, b] grown by r_link) and a disc (centre c, radius r_obs):
    > 0 no contact, = 0 contact, < 0 penetration (MR Sec. 10.2.2, sphere form d = ||r - b|| - R - B)."""
    # TODO: distance(centre, segment) - r_link - r_obs.
    raise NotImplementedError


def clearance(q, scene) -> float:
    """d(q): the smallest signed distance between the two link capsules and every obstacle
    (discs in scene['circles'] and, if scene['floor'] is not None, the half-plane y <= floor).
    The base link (link 1) starts at the joint-1 axis; the robot base itself is not an obstacle."""
    # TODO: Loop over the two links (p0-p1, p1-p2) and every disc; also the floor if scene['floor'] is not None:
    #       the floor clearance of a capsule is min(a_y, b_y) - r_link - floor. Return the minimum.
    raise NotImplementedError


def config_free(q, scene, margin: float = 0.0) -> bool:
    """True if q is inside the joint limits [-pi, pi] and clearance(q) > margin (MR: C_free)."""
    # TODO: False outside [-pi, pi] on either joint; otherwise clearance(q) > margin.
    raise NotImplementedError


# ==========================================================================================
# Part B -- C-space grid and edge checking (MR Sec. 10.4)
# ==========================================================================================

def grid_angles(k: int) -> np.ndarray:
    """The k grid values of each joint: theta_i = -pi + i * (2 pi / k), i = 0..k-1."""
    return -math.pi + np.arange(k) * (2 * math.pi / k)


def cell_to_q(cell, k: int) -> np.ndarray:
    """Grid cell (i1, i2) -> configuration q (rad)."""
    th = grid_angles(k)
    return np.array([th[cell[0]], th[cell[1]]])


def q_to_cell(q, k: int) -> tuple:
    """Configuration q (rad) -> nearest grid cell (i1, i2) (with wrap-around)."""
    # TODO: Wrap q to [-pi, pi), divide by the grid step, round to the nearest integer, take it modulo k.
    raise NotImplementedError


def cspace_grid(scene, k: int) -> np.ndarray:
    """Boolean k x k array `free[i1, i2]` = config_free(cell_to_q((i1, i2))). Row index = joint 1."""
    # TODO: Evaluate config_free at every grid configuration; free[i1, i2] (row = joint 1).
    raise NotImplementedError


def joint_step(qa, qb, wrap: bool = True) -> np.ndarray:
    """Joint-space displacement from qa to qb; with wrap=True the shortest way round each joint circle."""
    # TODO: qb - qa, wrapped to [-pi, pi) if wrap (the short way round).
    raise NotImplementedError


def edge_free(qa, qb, scene, step: float = math.radians(1.0), wrap: bool = True) -> bool:
    """Sampled check of the straight joint-space motion qa -> qb: every sample (spacing <= step in the
    max-norm, both ends included) must be collision-free. Fast but NOT a guarantee between samples."""
    # TODO: d = joint_step(qa, qb); n = ceil(max|d| / step); test q = qa + s d for s in linspace(0, 1, n + 1)
    #       (wrap q to [-pi, pi) when wrap=True).
    raise NotImplementedError


def motion_bound(dq, L=L_PLANAR_2R) -> float:
    """Upper bound on how far ANY point of the 2R arm moves when the joints move straight by dq:
    D = (L1 + L2)|dq1| + L2|dq2|  (a point at distance rho from joint 1 moves <= rho |dq1| + ...)."""
    # TODO: (L1 + L2)|dq1| + L2|dq2|.
    raise NotImplementedError


def edge_free_certified(qa, qb, scene, step: float = math.radians(1.0), wrap: bool = True) -> bool:
    """Conservative ("grown robot", MR p. 365) check of the straight motion qa -> qb: sample with spacing
    <= step and require clearance >= D/2 at every sample, where D = motion_bound(sample spacing).
    Every configuration between two samples is within D/2 of one of them, so True GUARANTEES the whole
    motion is collision-free. False may be a false alarm (the check is conservative)."""
    # TODO: As edge_free, but each sample must have clearance >= motion_bound(d / n) / 2 (and be inside
    #       the joint limits). Explain in your report why this is a guarantee.
    raise NotImplementedError


# ==========================================================================================
# Part C -- graph search (MR Algorithm 10.1, Sec. 10.2.4, Sec. 10.4)
# ==========================================================================================

def astar(start, goal, neighbors, h, eta: float = 1.0, reopen: bool = False) -> dict:
    """A* search (MR Algorithm 10.1) on an implicit graph.

    start, goal : hashable nodes;  neighbors(node) -> iterable of (nbr, edge_cost > 0);
    h(node) -> heuristic cost-to-go >= 0;  est_total_cost = past_cost + eta * h  (eta > 1: MR's
    "suboptimal A*");  reopen=False is MR Algorithm 10.1 exactly (neighbours in CLOSED are skipped);
    reopen=True moves a CLOSED node back to OPEN when a cheaper path to it is found (needed for optimality
    when h is admissible but not consistent).
    The goal test is made when a node is removed from OPEN (MR line 6). Ties in OPEN are broken by the
    smaller h, then first-in first-out.

    Returns dict(success, path [start..goal] or None, cost, expanded = number of nodes removed from OPEN
    and expanded (the goal is counted when it is removed), parent, past_cost)."""
    # TODO: Follow MR Algorithm 10.1. Use heapq for OPEN with entries (past + eta*h, h, counter, node) and
    #       'lazy deletion': skip a popped node that is already CLOSED or whose entry is out of date.
    #       Goal test when a node is popped. Skip neighbours in CLOSED unless reopen=True; with reopen=True,
    #       remove an improved neighbour from CLOSED and push it again. Count every expanded node.
    raise NotImplementedError


def dijkstra(start, goal, neighbors) -> dict:
    """Dijkstra's algorithm = A* with a zero heuristic (MR Sec. 10.2.4.2)."""
    # TODO: One line: A* with a zero heuristic.
    raise NotImplementedError


def grid_neighbors(cell, k: int, connectivity: int = 8, wrap: bool = True, free=None):
    """Neighbours of `cell` on the k x k grid with their costs in rad (grid step d = 2 pi / k):
    4-connected: N, S, E, W with cost d;  8-connected: also the diagonals with cost sqrt(2) d (MR p. 372).
    wrap=True: the C-space is a torus (cell k-1 is next to cell 0, MR p. 360); wrap=False: joint stops.
    If `free` (k x k bool) is given, only free neighbours are returned. Returns a list of ((i, j), cost)."""
    # TODO: Steps (+-1, 0), (0, +-1) and, for 8-connectivity, the four diagonals (cost sqrt(2) d).
    #       wrap=True: indices modulo k; wrap=False: drop neighbours outside 0..k-1. Skip non-free ones.
    raise NotImplementedError


def grid_heuristic(cell, goal, k: int, kind: str = "euclidean", wrap: bool = True) -> float:
    """Heuristic cost-to-go (rad) on the grid. Per-axis cell differences n1, n2 (the shorter way round if wrap):
    'zero' 0;  'manhattan' d (n1 + n2);  'euclidean' d sqrt(n1^2 + n2^2);
    'octile' d (max + (sqrt 2 - 1) min)  (exact cost-to-go of an empty 8-connected grid)."""
    # TODO: n_i = |cell_i - goal_i| (with wrap: min(n_i, k - n_i)); then the formula of the docstring.
    raise NotImplementedError


def grid_astar(free, start, goal, connectivity: int = 8, heuristic: str = "euclidean", eta: float = 1.0,
               wrap: bool = True, edge_ok=None) -> dict:
    """A* on the C-space grid `free` (k x k bool) from cell `start` to cell `goal`.
    edge_ok(cell_a, cell_b) -> bool, optional: a step is used only if it is collision-free (MR p. 372).
    Returns the dict of `astar` (path = list of cells)."""
    # TODO: Return failure if start or goal is not free. Build neighbors(c) from grid_neighbors (and edge_ok)
    #       and h(c) from grid_heuristic, then call astar.
    raise NotImplementedError


# ==========================================================================================
# Part D -- the planner
# ==========================================================================================

def unwrap_path(qs) -> np.ndarray:
    """Make a joint path continuous: replace each jump of about 2 pi (crossing -pi/pi) by the short step."""
    # TODO: Start with qs[0]; each next point = previous + wrap_angle(q - previous).
    raise NotImplementedError


def path_length(qs) -> float:
    """Sum of the Euclidean joint-space step lengths of a (continuous) path (rad)."""
    # TODO: Sum of the norms of consecutive differences.
    raise NotImplementedError


def plan_2r(scene, k: int = 72, connectivity: int = 8, heuristic: str = "euclidean", eta: float = 1.0,
            certified: bool = True) -> dict:
    """Plan a collision-free joint path for the 2R arm of `scene` from scene['q_start'] to scene['q_goal'].
    1. build the grid (cspace_grid); 2. snap start and goal to the nearest cells; the snap motions must be
    collision-free; 3. grid_astar with an edge check (edge_free_certified if certified else edge_free,
    both with step = grid step / 4); 4. return the continuous path q_start -> cells -> q_goal.
    Returns dict(success, q_path (N x 2, continuous, starts at q_start and ends at q_goal or an
    equivalent angle), cells, cost (grid cost, rad), expanded, free)."""
    # TODO: Follow steps 1-4 of the docstring; reuse cspace_grid, q_to_cell, cell_to_q, grid_astar and
    #       edge_free_certified / edge_free with step = (2 pi / k) / 4. Unwrap the path if wrap.
    raise NotImplementedError
