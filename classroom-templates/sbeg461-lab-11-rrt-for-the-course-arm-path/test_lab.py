"""Auto-checks for SBEG 461 Lab 11 (RRT for the course arm; path smoothing + time scaling).

    LAB_IMPL=solution pytest -q test_lab.py      # reference: all pass
    LAB_IMPL=starter  pytest -q test_lab.py      # untouched starter: every test fails (NotImplementedError)

LAB_IMPL is the name of the module to test (default "solution"); it is imported from this folder.
Ground truth: modern_robotics FKinSpace (tool point and truncated chains), modern_robotics InverseDynamics,
an independent loop implementation of the capsule clearance written in this file, and the defining
properties of RRT / RRT* trees (MR Algorithm 10.3, p. 381; RRT* p. 385), shortcutting (MR Sec. 10.8) and trapezoidal time
scaling (MR Sec. 9.2.2).  Planning tests use fixed seeds but only check properties, so any correct
implementation passes whatever order it draws its random numbers in.
"""
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import modern_robotics as mr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
lab = importlib.import_module(os.environ.get("LAB_IMPL", "starter"))

ARM = lab.load_course_arm()
SCENES = ["bench_or", "eod_rig"]
RNG = np.random.default_rng(461)


def scene(name):
    return lab.load_scene(name)


def rand_q(k=5):
    lo, hi = ARM["limits"][:, 0], ARM["limits"][:, 1]
    return [RNG.uniform(lo, hi) for _ in range(k)]


# ------------------------------------------------------------------------------------------ independent model
def ref_points(q):
    """Skeleton points by modern_robotics: point k is at FKinSpace(I, S[:, :l], q[:l]) p_k, l = its link."""
    P = []
    for p, l in zip(ARM["skeleton"], ARM["skeleton_link"]):
        T = np.eye(4) if l == 0 else mr.FKinSpace(np.eye(4), ARM["Slist"][:, :l], q[:l])
        P.append(T[:3, :3] @ p + T[:3, 3])
    return np.array(P)


def ref_clearance(q, sc):
    P = ref_points(q)
    best = np.inf
    for k in range(1, 6):
        a, b, r = P[k], P[k + 1], ARM["link_radius"][k - 1]
        for cx, cy, cz, R in sc["spheres"]:
            c = np.array([cx, cy, cz])
            t = np.clip((c - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
            best = min(best, np.linalg.norm(a + t * (b - a) - c) - r - R)
        for nx, ny, nz, d in sc["planes"]:
            n = np.array([nx, ny, nz])
            best = min(best, min(n @ a, n @ b) - d - r)
    return best


def path_is_free(path, sc, step=0.005, tol=0.003):
    """Re-check a via-point path more finely than the planner (sampled checks may miss < tol)."""
    for qa, qb in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(np.max(np.abs(qb - qa)) / step)))
        for s in np.linspace(0, 1, n + 1):
            q = qa + s * (qb - qa)
            if np.any(q < ARM["limits"][:, 0] - 1e-9) or np.any(q > ARM["limits"][:, 1] + 1e-9):
                return False
            if ref_clearance(q, sc) < -tol:
                return False
    return True


# ------------------------------------------------------------------------------------------ Part A
def test_link_points_tool_point_is_fk():
    for q in rand_q():
        P = lab.link_points(ARM, q)
        assert P.shape == (7, 3)
        assert np.allclose(P[6], mr.FKinSpace(ARM["M"], ARM["Slist"], q)[:3, 3], atol=1e-12)


def test_link_points_match_truncated_chains_and_batch():
    Q = np.array(rand_q(6))
    PB = lab.link_points(ARM, Q)
    assert PB.shape == (6, 7, 3)
    for q, P in zip(Q, PB):
        assert np.allclose(P, ref_points(q), atol=1e-12)


def test_link_points_rigid_segments():
    L0 = np.linalg.norm(np.diff(ARM["skeleton"], axis=0), axis=1)
    for q in rand_q():
        L = np.linalg.norm(np.diff(lab.link_points(ARM, q), axis=0), axis=1)
        assert np.allclose(L, L0, atol=1e-12)


@pytest.mark.parametrize("name", SCENES)
def test_clearance_matches_independent_model(name):
    sc = scene(name)
    qs = rand_q(6) + [sc["q_start"], sc["q_goal"], 0.5 * (sc["q_start"] + sc["q_goal"])]
    for q in qs:
        assert np.isclose(lab.clearance(ARM, sc, q), ref_clearance(q, sc), atol=1e-10)
    got = lab.clearance(ARM, sc, np.array(qs))
    assert got.shape == (len(qs),)
    assert np.allclose(got, [ref_clearance(q, sc) for q in qs], atol=1e-10)


@pytest.mark.parametrize("name", SCENES)
def test_start_goal_free_and_straight_line_blocked(name):
    sc = scene(name)
    assert lab.config_free(ARM, sc, sc["q_start"]) and lab.config_free(ARM, sc, sc["q_goal"])
    assert not lab.config_free(ARM, sc, 0.5 * (sc["q_start"] + sc["q_goal"]))
    assert not lab.edge_free(ARM, sc, sc["q_start"], sc["q_goal"])


def test_config_free_respects_joint_limits():
    sc = scene("bench_or")
    q = sc["q_start"].copy()
    q[0] = ARM["limits"][0, 1] + 0.01
    assert not lab.config_free(ARM, sc, q)
    assert lab.config_free(ARM, sc, sc["q_start"], margin=0.0)
    assert not lab.config_free(ARM, sc, sc["q_start"], margin=1.0)


def test_edge_free_short_moves_and_certified():
    sc = scene("bench_or")
    qa = sc["q_start"]
    qb = qa + np.array([0.05, -0.05, 0.0, 0.0, 0.0, 0.05])
    assert lab.edge_free(ARM, sc, qa, qb)
    assert lab.edge_free(ARM, sc, qa, qb, certified=True)
    # an edge that leaves the joint limits is not free
    qc = qa.copy()
    qc[4] = ARM["limits"][4, 0] - 0.1
    assert not lab.edge_free(ARM, sc, qa, qc)
    # certified implies sampled (the certified check is conservative)
    for _ in range(20):
        qx = qa + RNG.normal(scale=0.15, size=6)
        if lab.edge_free(ARM, sc, qa, qx, certified=True):
            assert lab.edge_free(ARM, sc, qa, qx)


def test_edge_free_catches_a_thin_collision():
    """A motion whose endpoints are free but whose middle collides must be rejected."""
    sc = scene("eod_rig")
    qs, qg = sc["q_start"], sc["q_goal"]
    s = np.linspace(0, 1, 401)
    c = lab.clearance(ARM, sc, qs + s[:, None] * (qg - qs))
    k = int(np.argmin(c))
    assert c[k] < 0
    # free points on either side of the collision
    left = max(i for i in range(k) if c[i] > 0)
    right = min(i for i in range(k, len(s)) if c[i] > 0)
    qa, qb = qs + s[left] * (qg - qs), qs + s[right] * (qg - qs)
    assert not lab.edge_free(ARM, sc, qa, qb)


# ------------------------------------------------------------------------------------------ Part B
def test_sample_config_box_and_goal_bias():
    rng = np.random.default_rng(1)
    box = scene("bench_or")["sample_limits"]
    qg = np.full(6, 9.0)
    X = np.array([lab.sample_config(rng, box, qg, 0.2) for _ in range(3000)])
    is_goal = np.all(X == qg, axis=1)
    assert 0.17 < is_goal.mean() < 0.23
    Y = X[~is_goal]
    assert np.all(Y >= box[:, 0]) and np.all(Y <= box[:, 1])
    assert np.allclose(Y.mean(axis=0), box.mean(axis=1), atol=0.12)
    assert all(np.all(lab.sample_config(rng, box, qg, 1.0) == qg) for _ in range(10))
    assert not any(np.all(lab.sample_config(rng, box, qg, 0.0) == qg) for _ in range(50))


def test_nearest_and_steer():
    nodes = RNG.normal(size=(50, 6))
    for _ in range(10):
        q = RNG.normal(size=6)
        assert lab.nearest(nodes, q) == int(np.argmin(np.linalg.norm(nodes - q, axis=1)))
    nodes2 = np.vstack([np.zeros(6), np.ones(6), np.zeros(6)])
    assert lab.nearest(nodes2, 0.1 * np.ones(6)) == 0          # tie -> smallest index
    a, b = np.zeros(6), np.array([3.0, 4.0, 0, 0, 0, 0])
    assert np.allclose(lab.steer(a, b, 1.0), [0.6, 0.8, 0, 0, 0, 0])
    assert np.allclose(lab.steer(a, b, 6.0), b)
    assert np.allclose(lab.steer(a, b, 5.0), b)


def test_tree_path_and_path_length():
    nodes = np.array([[0.0] * 6, [1.0] + [0] * 5, [1.0, 1.0] + [0] * 4, [0, 0, 2.0, 0, 0, 0]])
    parent = np.array([-1, 0, 1, 0])
    p = lab.tree_path(nodes, parent, 2)
    assert np.allclose(p, nodes[[0, 1, 2]])
    assert np.isclose(lab.path_length(p), 2.0)
    assert np.isclose(lab.path_length(lab.tree_path(nodes, parent, 3)), 2.0)


@pytest.mark.parametrize("name", SCENES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rrt_finds_a_valid_path(name, seed):
    sc = scene(name)
    d = 0.3
    res = lab.rrt(ARM, sc, sc["q_start"], sc["q_goal"], np.random.default_rng(seed), d=d,
                  limits=sc["sample_limits"])
    assert res["success"]
    path, nodes, parent = res["path"], res["nodes"], res["parent"]
    assert np.allclose(path[0], sc["q_start"]) and np.allclose(path[-1], sc["q_goal"])
    assert parent[0] == -1 and np.all(parent[1:] >= 0) and np.all(parent[1:] < np.arange(1, len(parent)))
    edge = np.linalg.norm(nodes[1:] - nodes[parent[1:]], axis=1)
    assert np.all(edge <= d + 1e-9) and np.all(edge > 0)
    assert path_is_free(path, sc)


def test_rrt_reports_failure():
    sc = scene("bench_or")
    bad = 0.5 * (sc["q_start"] + sc["q_goal"])                  # in collision
    res = lab.rrt(ARM, sc, sc["q_start"], bad, np.random.default_rng(0), limits=sc["sample_limits"],
                  max_nodes=200)
    assert not res["success"] and res["path"] is None
    res = lab.rrt(ARM, sc, sc["q_start"], sc["q_goal"], np.random.default_rng(0), d=0.01, max_nodes=30,
                  limits=sc["sample_limits"])
    assert not res["success"] and len(res["nodes"]) <= 30


@pytest.mark.parametrize("name", SCENES)
def test_rrt_star_tree_is_consistent(name):
    sc = scene(name)
    res = lab.rrt_star(ARM, sc, sc["q_start"], sc["q_goal"], np.random.default_rng(3), max_nodes=400,
                       r_near=1.0, limits=sc["sample_limits"])
    assert res["success"]
    nodes, parent, cost = res["nodes"], res["parent"], res["cost"]
    assert len(nodes) == 400                                       # RRT* keeps going after the first solution
    assert cost[0] == 0 and parent[0] == -1
    for i in range(1, len(nodes)):
        assert np.isclose(cost[i], cost[parent[i]] + np.linalg.norm(nodes[i] - nodes[parent[i]]), atol=1e-9)
    path = res["path"]
    assert np.allclose(path[-1], sc["q_goal"])
    assert np.isclose(lab.path_length(path), cost[res["goal_index"]], atol=1e-9)
    costs = [c for _, c in res["best_cost_history"]]
    assert all(b <= a + 1e-12 for a, b in zip(costs, costs[1:]))
    assert path_is_free(path, sc)


def test_rrt_star_beats_rrt_on_average():
    sc = scene("bench_or")
    star, plain = [], []
    for seed in range(4):
        rs = lab.rrt_star(ARM, sc, sc["q_start"], sc["q_goal"], np.random.default_rng(seed), max_nodes=700,
                          r_near=1.0, limits=sc["sample_limits"])
        rp = lab.rrt(ARM, sc, sc["q_start"], sc["q_goal"], np.random.default_rng(seed),
                     limits=sc["sample_limits"])
        star.append(lab.path_length(rs["path"]))
        plain.append(lab.path_length(rp["path"]))
    assert np.median(star) < np.median(plain)


# ------------------------------------------------------------------------------------------ Part C
def test_shortcut_greedy_straight_when_free():
    sc = dict(scene("bench_or"))
    sc["spheres"] = np.zeros((0, 4))                                # remove the clutter
    qs, qg = sc["q_start"], sc["q_goal"]
    detour = np.array([qs, qs + np.array([0.3, -0.3, 0.2, 0, 0, 0]), qg + np.array([0, -0.3, 0, 0.2, 0, 0]), qg])
    out = lab.shortcut_greedy(ARM, sc, detour)
    assert out.shape == (2, 6) and np.allclose(out[0], qs) and np.allclose(out[1], qg)


@pytest.mark.parametrize("name", SCENES)
def test_shortcut_greedy_on_rrt_path(name):
    sc = scene(name)
    path = lab.rrt(ARM, sc, sc["q_start"], sc["q_goal"], np.random.default_rng(5),
                   limits=sc["sample_limits"])["path"]
    out = lab.shortcut_greedy(ARM, sc, path)
    assert np.allclose(out[0], path[0]) and np.allclose(out[-1], path[-1])
    assert lab.path_length(out) <= lab.path_length(path) + 1e-12
    assert len(out) >= 3                                            # straight line is blocked
    assert all(any(np.allclose(p, x) for x in path) for p in out)  # keeps original via points only
    # greedy property: from each kept point the next-but-one kept point is not reachable directly
    for a, c in zip(out[:-2], out[2:]):
        assert not lab.edge_free(ARM, sc, a, c)
    assert path_is_free(out, sc)


def test_point_on_path():
    path = np.array([[0.0] * 6, [1.0] + [0] * 5, [1.0, 2.0] + [0] * 4])
    k, p = lab.point_on_path(path, 1.5)
    assert k == 1 and np.allclose(p, [1, 0.5, 0, 0, 0, 0])


@pytest.mark.parametrize("name", SCENES)
def test_shortcut_random(name):
    sc = scene(name)
    path = lab.rrt(ARM, sc, sc["q_start"], sc["q_goal"], np.random.default_rng(7),
                   limits=sc["sample_limits"])["path"]
    out = lab.shortcut_random(ARM, sc, path, np.random.default_rng(7), n_iter=60)
    assert np.allclose(out[0], path[0]) and np.allclose(out[-1], path[-1])
    assert lab.path_length(out) <= lab.path_length(path) + 1e-9
    assert path_is_free(out, sc)


def test_time_scale_path():
    sc = scene("bench_or")
    path = np.array([sc["q_start"], sc["q_start"] + np.array([0.4, -0.4, 0.3, 0, 0, 0]), sc["q_goal"]])
    dt = 0.01
    t, Q, Qd, Qdd, segT = lab.time_scale_path(path, ARM["qdot_max"], ARM["qddot_max"], dt)
    T_ref = [lab.trapezoid_from_limits(a, b, ARM["qdot_max"], ARM["qddot_max"])[2]
             for a, b in zip(path[:-1], path[1:])]
    assert np.allclose(segT, T_ref) and np.isclose(t[-1], sum(T_ref))
    assert np.all(np.diff(t) > 0) and np.all(np.diff(t) <= dt + 1e-12) and t[0] == 0
    assert np.allclose(Q[0], path[0]) and np.allclose(Q[-1], path[-1])
    k = int(np.argmin(np.abs(t - T_ref[0])))
    assert np.isclose(t[k], T_ref[0]) and np.allclose(Q[k], path[1]) and np.allclose(Qd[k], 0)
    assert np.allclose(Qd[0], 0) and np.allclose(Qd[-1], 0)
    assert np.all(np.abs(Qd) <= ARM["qdot_max"] + 1e-9) and np.all(np.abs(Qdd) <= ARM["qddot_max"] + 1e-9)
    # velocity is the time derivative of position (central differences)
    v_fd = (Q[2:] - Q[:-2]) / (t[2:] - t[:-2])[:, None]
    assert np.max(np.abs(v_fd - Qd[1:-1])) < 0.05
    # every sample lies on the straight segments
    for q in Q[::7]:
        res = []
        for a, b in zip(path[:-1], path[1:]):
            s = np.clip((q - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
            res.append(np.linalg.norm(a + s * (b - a) - q))
        assert min(res) < 1e-9


def test_torque_profile_matches_mr():
    sc = scene("eod_rig")
    path = np.array([sc["q_start"], sc["q_start"] + np.array([-0.4, -0.3, 0.2, 0.1, 0, 0])])
    t, Q, Qd, Qdd, _ = lab.time_scale_path(path, ARM["qdot_max"], ARM["qddot_max"], 0.02)
    tau = lab.torque_profile(ARM, Q, Qd, Qdd)
    assert tau.shape == Q.shape
    for k in range(0, len(t), 5):
        ref = mr.InverseDynamics(Q[k], Qd[k], Qdd[k], ARM["g"], np.zeros(6), ARM["Mlist"], ARM["Glist"],
                                 ARM["Slist"])
        assert np.allclose(tau[k], ref, atol=1e-10)


@pytest.mark.parametrize("name", SCENES)
def test_pipeline_feasible_on_course_arm(name):
    sc = scene(name)
    out = lab.plan_and_scale(ARM, sc, seed=0, smooth="greedy")
    assert out["success"] and out["limits"]["ok"]
    assert out["t"][-1] < 3.0                                       # smoothed motion is short
