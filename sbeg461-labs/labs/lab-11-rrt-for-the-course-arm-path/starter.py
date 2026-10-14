"""SBEG 461 Lab 11 -- RRT for the course arm; path smoothing + time scaling (formal report 3).
STARTER CODE. Complete every function marked TODO (keep the names and signatures).
Run the auto-checks with:  LAB_IMPL=starter python -m pytest -q test_lab.py
See lab_handout.md for the checkpoints.  The reference solution is released after the report deadline.

Notation follows Modern Robotics (MR): configuration q in C (6 joint angles), free space C_free, signed
distance d(q, B) to an obstacle (MR Sec. 10.2.2), RRT (MR Algorithm 10.3, Sec. 10.5.1), RRT* (MR p. 385),
smoothing by "subdivide and reconnect" (MR Sec. 10.8), straight-line trapezoidal time scaling (MR Sec. 9.2.2,
Lab 9).

Parts
  A. Collision model of the course arm: a 7-point skeleton moved by the PoE formula, capsules (segment +
     radius) against spheres and planes, clearance d(q), config_free, edge_free          MR Sec. 10.2
  B. Sampling-based planning: sampler with goal bias, nearest node, straight-line local planner (steer),
     RRT and RRT*                                                                         MR Sec. 10.5.1
  C. Smoothing and time scaling: path length, greedy and random shortcutting, rest-to-rest trapezoidal
     time scaling of the via-point path, torque profile and limit check                  MR Sec. 10.8, 9.2

These functions become the module planning_rrt.py of your `sbeg461_dynamics` library (not Lab 10's
planning.py, whose clearance/config_free/edge_free/motion_bound/path_length differ). The project
(scenarios 3 and 5: RRT plan + time scaling + control) uses the same algorithms but NOT the same names
or signatures: its starter module `sbeg461_project/planning.py` takes a Scenario `s` instead of
(arm, scene) and provides clearance, config_free, edge_free, rrt, rrt_bidirectional (no rrt_star),
shortcut (random subdivide-and-reconnect between existing vertices, closest to shortcut_random here;
there is no shortcut_greedy) and path_clearance; time scaling and torque checks live in
`sbeg461_project/traj.py` (torque_profile, check_limits, stop_and_go_time). Port your Lab 11 code to
those signatures, or wrap it, when you reuse it in the project.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# ==========================================================================================
# Provided (do not edit): robot, scenes, batch PoE, Lab 9 time-scaling helpers, RNEA
# ==========================================================================================


def load_course_arm(payload: bool = True, derated: bool = True) -> dict:
    """The course arm (numbers identical to the course model in Course resources: Course robot models and
    hardware safety checklist).

    Returns a dict with
      Slist (6x6), M (4x4), Mlist (7 4x4), Glist (6 6x6), g (3,)      kinematics and dynamics (MR list form)
      limits (6x2), qdot_max (6,), qddot_max (6,), tau_max (6,)        joint limits and ratings
      skeleton (7x3), skeleton_link (7,), link_radius (5,), rho (6,)   collision model (see Part A)
    payload=True uses a 0.125 kg payload at the tool point (the hardware-run limit: <= 50 % of the rated
    0.25 kg, safety checklist rule 4); derated=True applies the
    hardware-run derating (50 % of rated speed, 60 % of rated torque).  Inertial data, ratings and the
    acceleration limit (4 rad/s^2, as in Lab 9) are ILLUSTRATIVE values."""
    d = json.loads((DATA / "course_arm.json").read_text())
    kf = d["derated_speed_fraction"] if derated else 1.0
    tf = d["derated_current_fraction"] if derated else 1.0
    return dict(Slist=np.array(d["Slist"]), M=np.array(d["M"]),
                Mlist=[np.array(m) for m in d["Mlist"]],
                Glist=[np.array(G) for G in (d["Glist_payload"] if payload else d["Glist"])],
                g=np.array(d["g"]), limits=np.array(d["limits"]),
                qdot_max=kf * np.array(d["qdot_max"]), qddot_max=np.array(d["qddot_max"]),
                tau_max=tf * np.array(d["tau_max"]), q_home=np.array(d["q_home"]),
                skeleton=np.array(d["skeleton"]), skeleton_link=np.array(d["skeleton_link"], int),
                link_radius=np.array(d["link_radius"]), rho=np.array(d["rho"]))


def load_scene(name: str) -> dict:
    """Scene `name` from data/scenes.json: spheres (m x 4: cx, cy, cz, R), planes (p x 4: nx, ny, nz, d,
    free side n.x >= d), q_start, q_goal (rad, 6), sample_limits (6 x 2: the task sampling box used by the
    planners; inside the joint limits except joint 4, lower bound -3.2 < -2.88 rad, so the joint-limit check
    still rejects some samples), description."""
    s = json.loads((DATA / "scenes.json").read_text())[name]
    out = dict(s)
    out["spheres"] = np.asarray(s["spheres"], float).reshape(-1, 4)
    out["planes"] = np.asarray(s["planes"], float).reshape(-1, 4)
    out["q_start"] = np.asarray(s["q_start"], float)
    out["q_goal"] = np.asarray(s["q_goal"], float)
    out["sample_limits"] = np.asarray(s["sample_limits"], float)
    return out


def exp6_batch(S, theta) -> np.ndarray:
    """e^{[S] theta} for one screw axis S = (omega, v), ||omega|| = 1, and an array of angles theta (N,).
    Rodrigues' formula (MR Eq. 3.51) and MR Prop. 3.25 for the translation.  Returns (N, 4, 4)."""
    th = np.atleast_1d(np.asarray(theta, float))
    w, v = np.asarray(S[:3], float), np.asarray(S[3:], float)
    W = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    W2 = W @ W
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    I = np.eye(3)[None]
    R = I + s * W + (1 - c) * W2
    G = th[:, None, None] * I + (1 - c) * W + (th[:, None, None] - s) * W2
    T = np.zeros((th.size, 4, 4))
    T[:, :3, :3] = R
    T[:, :3, 3] = G @ v
    T[:, 3, 3] = 1.0
    return T


def poe_frames(arm: dict, Q) -> np.ndarray:
    """Partial products of exponentials T_i(q) = e^{[S1] q1} ... e^{[Si] qi}, i = 0..6 (T_0 = I), for a
    batch of configurations Q (N, 6).  A point fixed in link i at home, p_home, is at T_i(q) p_home
    (MR Eq. 4.14 applied to a truncated chain).  Returns (N, 7, 4, 4)."""
    Q = np.atleast_2d(np.asarray(Q, float))
    N, n = Q.shape
    out = np.empty((N, n + 1, 4, 4))
    out[:, 0] = np.eye(4)
    for i in range(n):
        out[:, i + 1] = out[:, i] @ exp6_batch(arm["Slist"][:, i], Q[:, i])
    return out


def trapezoid_params(v, a):
    """(Lab 9) Total time, acceleration time and peak s-speed of a trapezoidal time scaling with s-units
    v (1/s) and a (1/s^2): T = (a + v^2)/(v a) if v^2/a <= 1 (MR p. 334), else bang-bang T = 2/sqrt(a)."""
    if v * v / a <= 1.0:
        return (a + v * v) / (v * a), v / a, v
    ta = 1.0 / np.sqrt(a)
    return 2 * ta, ta, np.sqrt(a)


def trapezoid_time_scaling(v, a, t):
    """(Lab 9) Trapezoidal time scaling (MR Eqs. 9.16-9.24); returns (s, sdot, sddot) at times t."""
    T, ta, vp = trapezoid_params(v, a)
    t = np.clip(np.asarray(t, float), 0.0, T)
    s = np.where(t <= ta, 0.5 * a * t**2,
                 np.where(t <= T - ta, vp * t - vp**2 / (2 * a),
                          (2 * a * vp * T - 2 * vp**2 - a**2 * (t - T) ** 2) / (2 * a)))
    sd = np.where(t <= ta, a * t, np.where(t <= T - ta, vp, a * (T - t)))
    sdd = np.where(t < ta, a, np.where(t <= T - ta, 0.0, -a))
    return s, sd, sdd


def trapezoid_from_limits(theta_start, theta_end, qdot_lim, qddot_lim):
    """(Lab 9) Fastest trapezoidal straight-line motion: the largest s-units v and a with
    |D_i| v <= qdot_lim_i and |D_i| a <= qddot_lim_i for every moving joint (MR Eqs. 9.14-9.15).
    Returns (v, a, T)."""
    D = np.abs(np.asarray(theta_end, float) - np.asarray(theta_start, float))
    moving = D > 1e-12
    vl = np.broadcast_to(np.asarray(qdot_lim, float), D.shape)
    al = np.broadcast_to(np.asarray(qddot_lim, float), D.shape)
    v = np.min(vl[moving] / D[moving])
    a = np.min(al[moving] / D[moving])
    T, _, _ = trapezoid_params(v, a)
    return float(v), float(a), float(T)


def rnea(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, Slist):
    """Inverse dynamics; uses modern_robotics.InverseDynamics (MR Eqs. 8.50-8.54).  Replace it by your
    own Lab 2 RNEA if you like (same interface)."""
    import modern_robotics as mr
    return mr.InverseDynamics(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, Slist)


# ==========================================================================================
# Part A -- collision model and clearance (MR Sec. 10.2.1-10.2.2)
# ==========================================================================================

def link_points(arm: dict, q) -> np.ndarray:
    """Skeleton points of the course arm at configuration q.

    arm['skeleton'][k] is point k at home (in {s}); it is fixed in link arm['skeleton_link'][k], so it is at
    T_link(q) p_k (use poe_frames).  Accepts q of shape (6,) -> returns (7, 3), or Q (N, 6) -> (N, 7, 3).
    Point 6 is the tool point: it equals the position part of FKinSpace(M, Slist, q)."""
    # TODO: implement link_points (see lab_handout.md)
    # Hint: for each skeleton point k use T = poe_frames(arm, Q)[:, arm['skeleton_link'][k]] and p = R p_k + t
    raise NotImplementedError("link_points")


def segment_point_distance(a, b, c) -> np.ndarray:
    """Distance from points c to segments [a, b], broadcasting over leading axes (last axis = 3)."""
    ab = b - a
    den = np.maximum(np.sum(ab * ab, axis=-1), 1e-15)
    t = np.clip(np.sum((c - a) * ab, axis=-1) / den, 0.0, 1.0)
    return np.linalg.norm(a + t[..., None] * ab - c, axis=-1)


def clearance(arm: dict, scene: dict, q):
    """Signed clearance d(q) (m): the smallest signed distance between the 5 link capsules (segments k-k+1,
    k = 1..5, radius arm['link_radius'][k-1]) and every obstacle:
        sphere (c, R):     dist(c, segment) - r_link - R                      (MR Sec. 10.2.2)
        plane (n, d):      min(n.a, n.b) - d - r_link   (free side n.x >= d)
    > 0 free, < 0 penetration.  The base column (points 0-1) does not move and is not checked.
    q (6,) -> float;  Q (N, 6) -> (N,)."""
    # TODO: implement clearance (see lab_handout.md)
    # Hint: segments a = P[:, 1:6], b = P[:, 2:7]; use segment_point_distance for spheres; min(n.a, n.b) - d - r for planes
    raise NotImplementedError("clearance")


def within_limits(arm: dict, q) -> np.ndarray | bool:
    """True where every joint is inside arm['limits'] (q (6,) or Q (N, 6))."""
    Q = np.asarray(q, float)
    ok = np.all((Q >= arm["limits"][:, 0] - 1e-12) & (Q <= arm["limits"][:, 1] + 1e-12), axis=-1)
    return bool(ok) if Q.ndim == 1 else ok


def config_free(arm: dict, scene: dict, q, margin: float = 0.0) -> bool:
    """q in C_free: inside the joint limits and clearance(q) > margin."""
    # TODO: implement config_free (see lab_handout.md)
    # Hint: within_limits(...) and clearance(...) > margin
    raise NotImplementedError("config_free")


def motion_bound(arm: dict, dq) -> float:
    """(Provided) Upper bound on how far any skeleton point moves for a straight joint motion dq:
    D = sum_i rho_i |dq_i|, rho_i = length of the skeleton beyond joint axis i (Lab 10, motion_bound)."""
    return float(np.sum(arm["rho"] * np.abs(np.asarray(dq, float))))


def edge_free(arm: dict, scene: dict, qa, qb, step: float = 0.02, margin: float = 0.0,
              certified: bool = False) -> bool:
    """Straight joint-space motion qa -> qb (MR Sec. 10.5.1.3, straight-line local planner) is free.

    Sample n + 1 configurations (both ends included) with n = max(1, ceil(max_i |qb_i - qa_i| / step));
    every sample must be within limits and have clearance > margin.  This sampled check is fast but not a
    guarantee between samples (MR p. 397).  certified=True requires clearance > margin + D/2 with
    D = motion_bound(arm, (qb - qa)/n): then the whole motion is guaranteed free (grown robot, MR p. 365).
    Evaluate all samples in one batch call of clearance()."""
    # TODO: implement edge_free (see lab_handout.md)
    # Hint: n = max(1, ceil(max|qb - qa| / step)); Q = qa + linspace(0, 1, n + 1)[:, None] (qb - qa); one batch clearance call
    raise NotImplementedError("edge_free")


# ==========================================================================================
# Part B -- RRT and RRT* (MR Sec. 10.5.1, Algorithm 10.3)
# ==========================================================================================

def sample_config(rng: np.random.Generator, limits, q_goal, p_goal: float) -> np.ndarray:
    """MR Algorithm 10.3 line 3: with probability p_goal return q_goal (goal bias), otherwise a sample
    from the uniform distribution over the joint-limit box.  Draw u = rng.random() first; if u < p_goal
    return a copy of q_goal, else return rng.uniform(limits[:, 0], limits[:, 1])."""
    # TODO: implement sample_config (see lab_handout.md)
    # Hint: u = rng.random(); if u < p_goal: return q_goal; else rng.uniform(limits[:, 0], limits[:, 1])
    raise NotImplementedError("sample_config")


def nearest(nodes, q) -> int:
    """MR Algorithm 10.3 line 4: index of the row of nodes (N, n) closest to q in Euclidean joint distance
    (ties: smallest index)."""
    # TODO: implement nearest (see lab_handout.md)
    # Hint: np.argmin of squared distances (np.argmin returns the first minimum)
    raise NotImplementedError("nearest")


def steer(q_from, q_to, d: float) -> np.ndarray:
    """MR Algorithm 10.3 line 5 (straight-line local planner): the point a distance d from q_from on the
    straight line towards q_to, or q_to itself if it is closer than d."""
    # TODO: implement steer (see lab_handout.md)
    # Hint: if ||q_to - q_from|| <= d return q_to, else q_from + d (q_to - q_from)/||q_to - q_from||
    raise NotImplementedError("steer")


def tree_path(nodes, parent, i: int) -> np.ndarray:
    """Follow parent pointers from node i back to the root (node 0); return the path root -> i (k, n)."""
    # TODO: implement tree_path (see lab_handout.md)
    # Hint: walk parent[] back to -1, then reverse
    raise NotImplementedError("tree_path")


def path_length(path) -> float:
    """Sum of Euclidean joint-space lengths of the straight segments of a via-point path (k, n)."""
    # TODO: implement path_length (see lab_handout.md)
    # Hint: sum of norms of np.diff(path, axis=0)
    raise NotImplementedError("path_length")


def rrt(arm: dict, scene: dict, q_start, q_goal, rng: np.random.Generator, d: float = 0.3,
        p_goal: float = 0.1, max_nodes: int = 3000, step: float = 0.02, limits=None) -> dict:
    """Basic RRT, MR Algorithm 10.3, for a kinematic problem in joint space.

    Each iteration: q_samp = sample_config; i = nearest(tree, q_samp); q_new = steer(tree[i], q_samp, d);
    if the straight motion tree[i] -> q_new is free (edge_free with `step`), add q_new with parent i.
    Goal set: if the new node is within d of q_goal and the motion q_new -> q_goal is free, add q_goal
    itself as a node (parent = new node) and return SUCCESS.  Skip q_new if it coincides with its parent.
    Stop with FAILURE when the tree has max_nodes nodes.  `limits` (6 x 2) is the sampling box (default
    arm['limits']); the collision checks always use the true joint limits arm['limits'].

    Returns dict(success, path (k, 6) or None, nodes (N, 6), parent (N,) with parent[0] = -1,
    iterations = number of samples drawn)."""
    # TODO: implement rrt (see lab_handout.md)
    # Hint: MR Algorithm 10.3; preallocate nodes (max_nodes x n) and parent; see lab_handout.md Checkpoint 3
    raise NotImplementedError("rrt")


def rrt_star(arm: dict, scene: dict, q_start, q_goal, rng: np.random.Generator, d: float = 0.3,
             p_goal: float = 0.1, max_nodes: int = 1000, r_near: float = 0.6, step: float = 0.02,
             limits=None) -> dict:
    """RRT* (MR Sec. 10.5.1.4, p. 385) with path length as cost and a fixed neighbourhood radius r_near.

    Like rrt(), but (1) the new node q_new chooses as parent the node x in X_near = {x : ||x - q_new|| <=
    r_near} (plus the nearest node) with a free motion x -> q_new that minimises cost(x) + ||x - q_new||;
    (2) every x in X_near that can be reached more cheaply through q_new (free motion) is re-wired to
    q_new, and the costs of its descendants are updated; (3) the search does NOT stop at the first
    solution: the goal is added as a node once (when a new node is within d of it with a free motion) and
    the loop runs until the tree has max_nodes nodes; the goal's cost keeps improving by rewiring.

    Returns dict(success, path, cost (N,), nodes, parent, goal_index, first_solution_nodes,
    best_cost_history (list of (tree size, cost-to-goal) each time it improves)).  `limits` as in rrt()."""
    # TODO: implement rrt_star (see lab_handout.md)
    # Hint: start from your rrt(); add cost[], children lists, choose-parent and rewire steps (lab_handout.md Checkpoint 4)
    raise NotImplementedError("rrt_star")


# ==========================================================================================
# Part C -- smoothing (MR Sec. 10.8) and time scaling (MR Sec. 9.2, Lab 9)
# ==========================================================================================

def shortcut_greedy(arm: dict, scene: dict, path, step: float = 0.02) -> np.ndarray:
    """Deterministic "subdivide and reconnect" (MR Sec. 10.8): from via point i, connect straight to the
    FARTHEST later via point j whose straight motion is free (try j = last, last-1, ..., i+1); keep j and
    repeat from j.  Returns the shorter via-point path (same first and last points)."""
    # TODO: implement shortcut_greedy (see lab_handout.md)
    # Hint: for j in range(last, i, -1): if edge_free(path[i], path[j]): keep j
    raise NotImplementedError("shortcut_greedy")


def point_on_path(path, s: float) -> tuple[int, np.ndarray]:
    """Point at arc length s along a via-point path: returns (segment index k, point) with the point on
    segment path[k] -> path[k+1]."""
    # TODO: implement point_on_path (see lab_handout.md)
    # Hint: cumulative segment lengths cum; k = segment with cum[k] <= s < cum[k+1]; interpolate
    raise NotImplementedError("point_on_path")


def shortcut_random(arm: dict, scene: dict, path, rng: np.random.Generator, n_iter: int = 100,
                    step: float = 0.02) -> np.ndarray:
    """Randomised test-and-replace smoothing (MR Sec. 10.8): n_iter times, draw two arc-length values
    s1 < s2 uniformly on [0, length] (u = rng.uniform(0, length, 2), sorted); take the points p1, p2 at
    s1, s2 (point_on_path); if the straight motion p1 -> p2 is free, replace the path between them by
    that segment, i.e. new path = path[:k1+1] + [p1, p2] + path[k2+1:].  Skip draws with k1 == k2."""
    # TODO: implement shortcut_random (see lab_handout.md)
    # Hint: use point_on_path(path, s) for the two random points; splice with np.vstack
    raise NotImplementedError("shortcut_random")


def time_scale_path(path, qdot_max, qddot_max, dt: float = 0.01):
    """Rest-to-rest time scaling of a via-point path (MR Sec. 9.2 applied segment by segment): each
    straight segment q_k -> q_{k+1} gets the fastest trapezoidal time scaling allowed by the joint speed
    and acceleration limits (trapezoid_from_limits), and the robot stops at every via point.

    Sample every segment at t = 0, dt, 2 dt, ... plus its end time; concatenate the segments without
    repeating the junction points (time is cumulative).  Returns (t (K,), Q (K, n), Qd (K, n),
    Qdd (K, n), seg_T (number of segments,)), with Qd = sdot (q_{k+1} - q_k), Qdd = sddot (q_{k+1} - q_k)."""
    # TODO: implement time_scale_path (see lab_handout.md)
    # Hint: per segment: v, a, T = trapezoid_from_limits(...); tt = np.append(np.arange(dt, T, dt), T); trapezoid_time_scaling(v, a, tt)
    raise NotImplementedError("time_scale_path")


def torque_profile(arm: dict, Q, Qd, Qdd) -> np.ndarray:
    """tau_k = M(q_k) qdd_k + c(q_k, qd_k) + g(q_k) along a sampled trajectory (RNEA; Lab 9).  (K, n)."""
    # TODO: implement torque_profile (see lab_handout.md)
    # Hint: one rnea(q, qd, qdd, arm['g'], zeros(6), Mlist, Glist, Slist) call per sample
    raise NotImplementedError("torque_profile")


def check_limits(arm: dict, Qd, Tau, tol: float = 1e-9) -> dict:
    """(Lab 9) Worst |velocity|/limit and |torque|/limit per joint; ok = all ratios <= 1."""
    vr = np.max(np.abs(Qd) / arm["qdot_max"], axis=0)
    tr = np.max(np.abs(Tau) / arm["tau_max"], axis=0)
    return dict(vel_ratio=vr, tau_ratio=tr, ok=bool(np.all(vr <= 1 + tol) and np.all(tr <= 1 + tol)))


def plan_and_scale(arm: dict, scene: dict, seed: int = 0, planner: str = "rrt", smooth: str = "greedy",
                   **kw) -> dict:
    """Convenience pipeline for the experiments: plan (rrt or rrt_star), smooth (none, greedy, random,
    or greedy+random), time-scale, compute torques.  Returns a dict with every intermediate result."""
    rng = np.random.default_rng(seed)
    kw.setdefault("limits", scene.get("sample_limits"))
    res = (rrt if planner == "rrt" else rrt_star)(arm, scene, scene["q_start"], scene["q_goal"], rng, **kw)
    if not res["success"]:
        return dict(plan=res, success=False)
    path = res["path"]
    if smooth in ("greedy", "greedy+random"):
        path = shortcut_greedy(arm, scene, path)
    if smooth in ("random", "greedy+random"):
        path = shortcut_random(arm, scene, path, rng)
    t, Q, Qd, Qdd, segT = time_scale_path(path, arm["qdot_max"], arm["qddot_max"])
    tau = torque_profile(arm, Q, Qd, Qdd)
    return dict(plan=res, success=True, path=path, t=t, Q=Q, Qd=Qd, Qdd=Qdd, seg_T=segT, tau=tau,
                limits=check_limits(arm, Qd, tau))
