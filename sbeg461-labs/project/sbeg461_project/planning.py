"""Collision checking and a BASELINE joint-space RRT (MR Sec. 10.5.1, Algorithm 10.3).

The robot is approximated by capsules (segments with a radius) attached to its links; obstacles
are spheres, axis-aligned boxes and a ground plane z = ground_z. Self-collision is not checked
(state this limitation in your report or add it).
"""
from __future__ import annotations

import numpy as np

from .models import Scenario


def _seg_point_dist(a, b, p) -> float:
    ab = b - a
    t = np.clip((p - a) @ ab / max(ab @ ab, 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(a + t * ab - p))


def _points_box_dist(P, lo, hi) -> np.ndarray:
    """Signed distances of the points P (k x 3) to an axis-aligned box (negative inside)."""
    d = np.maximum(np.maximum(lo - P, 0.0), P - hi)
    outside = np.linalg.norm(d, axis=1)
    inside = -np.min(np.minimum(P - lo, hi - P), axis=1)
    return np.where(outside > 0, outside, inside)


def segments_world(s: Scenario, q) -> list:
    """Capsule end points in {s} for configuration q: [(a, b, radius), ...]."""
    Ts = s.link_transforms(q)
    out = []
    for seg in s.segments:
        T = Ts[seg.link]
        a = T[:3, :3] @ seg.p0 + T[:3, 3]
        b = T[:3, :3] @ seg.p1 + T[:3, 3]
        out.append((a, b, seg.radius))
    return out


def clearance(s: Scenario, q, spacing: float = 0.02) -> float:
    """Smallest signed distance between the robot capsules and the obstacles (m); < 0 = collision."""
    obs = s.obstacles
    best = np.inf
    for a, b, r in segments_world(s, q):
        if "ground_z" in obs:
            best = min(best, min(a[2], b[2]) - r - obs["ground_z"])
        for c, rc in obs.get("spheres", []):
            best = min(best, _seg_point_dist(a, b, c) - r - rc)
        if obs.get("boxes"):
            k = max(2, int(np.ceil(np.linalg.norm(b - a) / spacing)) + 1)
            pts = a + np.linspace(0, 1, k)[:, None] * (b - a)
            for lo, hi in obs["boxes"]:
                best = min(best, float(np.min(_points_box_dist(pts, lo, hi))) - r)
    return float(best)


def config_free(s: Scenario, q, margin: float = 0.0) -> bool:
    q = np.asarray(q, float)
    if np.any(q < s.q_limits[:, 0]) or np.any(q > s.q_limits[:, 1]):
        return False
    return clearance(s, q) > margin


def edge_free(s: Scenario, qa, qb, step: float = 0.05, margin: float = 0.0) -> bool:
    """Straight joint-space edge check at resolution `step` (rad, max-norm)."""
    qa, qb = np.asarray(qa, float), np.asarray(qb, float)
    k = max(2, int(np.ceil(np.max(np.abs(qb - qa)) / step)) + 1)
    return all(config_free(s, qa + u * (qb - qa), margin) for u in np.linspace(0, 1, k))


def rrt(s: Scenario, q_start, q_goal, *, rng=None, step: float = 0.25, goal_bias: float = 0.15,
        max_nodes: int = 4000, margin: float = 0.0, edge_step: float = 0.05):
    """BASELINE RRT in joint space (MR Algorithm 10.3) with a straight-line local planner and goal
    bias. Returns the waypoint list [q_start, ..., q_goal] or None."""
    rng = np.random.default_rng(0) if rng is None else rng
    q_start, q_goal = np.asarray(q_start, float), np.asarray(q_goal, float)
    if not (config_free(s, q_start, margin) and config_free(s, q_goal, margin)):
        raise ValueError("start or goal is in collision")
    nodes, parent = [q_start], [-1]
    lo, hi = s.q_limits[:, 0], s.q_limits[:, 1]
    for _ in range(max_nodes):
        q_samp = q_goal if rng.random() < goal_bias else rng.uniform(lo, hi)
        d = np.linalg.norm(np.array(nodes) - q_samp, axis=1)
        i = int(np.argmin(d))
        v = q_samp - nodes[i]
        q_new = q_samp if d[i] <= step else nodes[i] + step * v / d[i]
        if not edge_free(s, nodes[i], q_new, edge_step, margin):
            continue
        nodes.append(q_new)
        parent.append(i)
        if np.linalg.norm(q_new - q_goal) <= step and edge_free(s, q_new, q_goal, edge_step, margin):
            nodes.append(q_goal)
            parent.append(len(nodes) - 2)
            path, j = [], len(nodes) - 1
            while j >= 0:
                path.append(nodes[j])
                j = parent[j]
            return path[::-1]
    return None


def rrt_bidirectional(s: Scenario, q_start, q_goal, *, rng=None, step: float = 0.3,
                      max_iters: int = 3000, margin: float = 0.0, edge_step: float = 0.05):
    """BASELINE bidirectional RRT (MR Sec. 10.5.1, "Bidirectional RRT", printed p. 384): one tree
    grows from q_start and one from q_goal; after each extension the other tree tries to connect
    greedily to the new node. Returns [q_start, ..., q_goal] or None."""
    rng = np.random.default_rng(0) if rng is None else rng
    qa, qb = np.asarray(q_start, float), np.asarray(q_goal, float)
    if not (config_free(s, qa, margin) and config_free(s, qb, margin)):
        raise ValueError("start or goal is in collision")
    lo, hi = s.q_limits[:, 0], s.q_limits[:, 1]
    trees = [([qa], [-1]), ([qb], [-1])]

    def nearest(tree, q):
        return int(np.argmin(np.linalg.norm(np.array(tree[0]) - q, axis=1)))

    def extend(tree, q_target):
        i = nearest(tree, q_target)
        v = q_target - tree[0][i]
        d = np.linalg.norm(v)
        q_new = q_target if d <= step else tree[0][i] + step * v / d
        if edge_free(s, tree[0][i], q_new, edge_step, margin):
            tree[0].append(q_new)
            tree[1].append(i)
            return len(tree[0]) - 1, d <= step
        return None, False

    def branch(tree, j):
        out = []
        while j >= 0:
            out.append(tree[0][j])
            j = tree[1][j]
        return out

    for it in range(max_iters):
        A, B = (trees[0], trees[1]) if it % 2 == 0 else (trees[1], trees[0])
        j, _ = extend(A, rng.uniform(lo, hi))
        if j is None:
            continue
        target = A[0][j]
        while True:                                   # greedy connect of B toward the new node
            k, reached = extend(B, target)
            if k is None:
                break
            if reached:
                pa, pb = branch(A, j), branch(B, k)
                path = pa[::-1] + pb[1:]
                if A is trees[1]:
                    path = path[::-1]
                return path
    return None


def shortcut(s: Scenario, path, *, rng=None, iters: int = 100, margin: float = 0.0,
             edge_step: float = 0.05):
    """Subdivide-and-reconnect smoothing (MR Sec. 10.8): replace a random sub-path by a straight
    joint-space edge when that edge is collision-free."""
    rng = np.random.default_rng(1) if rng is None else rng
    path = [np.asarray(p, float) for p in path]
    for _ in range(iters):
        if len(path) <= 2:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i < 2:
            continue
        if edge_free(s, path[i], path[j], edge_step, margin):
            path = path[:i + 1] + path[j:]
    return path


def path_clearance(s: Scenario, path, step: float = 0.05) -> float:
    """Minimum clearance along a piecewise-linear joint-space path."""
    best = np.inf
    for a, b in zip(path[:-1], path[1:]):
        k = max(2, int(np.ceil(np.max(np.abs(b - a)) / step)) + 1)
        for u in np.linspace(0, 1, k):
            best = min(best, clearance(s, a + u * (b - a)))
    return float(best)
