"""Auto-checks for SBEG 461 Lab 10 (grid planner for a planar 2R arm among obstacles).

    LAB_IMPL=solution python -m pytest -q test_lab.py      # reference: all pass
    LAB_IMPL=starter  python -m pytest -q test_lab.py      # untouched starter: every test fails (NotImplementedError)

LAB_IMPL is the name of the module to test (default "solution"); it is imported from this folder.
Ground truth (independent of your code): modern_robotics FKinSpace for the 2R arm, brute-force dense sampling
of the link capsules for distances and collisions, scipy.sparse.csgraph shortest paths on the same grid graph,
exact cost-to-go tables for the heuristic tests, and the hand-checked graphs of Workshop 11 (Problems 3 and 5).
"""
import importlib
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import modern_robotics as mr
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
lab = importlib.import_module(os.environ.get("LAB_IMPL", "starter"))

RNG = np.random.default_rng(461)
K_FAST = 36          # coarse grid for the heavier tests (10 deg)
K_LAB = 72           # lab grid (5 deg)


# ----------------------------------------------------------------------------- independent reference helpers
def dense_points(q, L, n=400):
    """Points along both link axes (brute force)."""
    t1, t2 = q
    e = L[0] * np.array([math.cos(t1), math.sin(t1)])
    tip = e + L[1] * np.array([math.cos(t1 + t2), math.sin(t1 + t2)])
    s = np.linspace(0, 1, n)[:, None]
    return np.vstack([s * e, e + s * (tip - e)])


def brute_clearance(q, scene):
    P = dense_points(q, scene["L"])
    r = scene["link_radius"]
    best = math.inf
    for cx, cy, rc in scene["circles"]:
        best = min(best, float(np.min(np.linalg.norm(P - [cx, cy], axis=1))) - r - rc)
    if scene.get("floor") is not None:
        best = min(best, float(np.min(P[:, 1])) - r - scene["floor"])
    return best


def scipy_grid(free, conn, wrap):
    """CSR adjacency of the grid graph (costs in rad), built independently of the lab code."""
    k = free.shape[0]
    steps = [(1, 0), (-1, 0), (0, 1), (0, -1)] + ([(1, 1), (1, -1), (-1, 1), (-1, -1)] if conn == 8 else [])
    rows, cols, w = [], [], []
    for i in range(k):
        for j in range(k):
            if not free[i, j]:
                continue
            for di, dj in steps:
                a, b = i + di, j + dj
                if wrap:
                    a, b = a % k, b % k
                elif not (0 <= a < k and 0 <= b < k):
                    continue
                if free[a, b]:
                    rows.append(i * k + j)
                    cols.append(a * k + b)
                    w.append(math.hypot(di, dj) * 2 * math.pi / k)
    return csr_matrix((w, (rows, cols)), shape=(k * k, k * k))


def graph_fns(edge_list):
    E = {}
    for a, b, c in edge_list:
        E.setdefault(a, []).append((b, c))
        E.setdefault(b, []).append((a, c))
    return (lambda n: E[n]), E


@pytest.fixture(scope="module")
def ward():
    return lab.load_scene("ward")


@pytest.fixture(scope="module")
def eod():
    return lab.load_scene("eod")


@pytest.fixture(scope="module")
def grids(ward, eod):
    return {"ward": lab.cspace_grid(ward, K_FAST), "eod": lab.cspace_grid(eod, K_FAST)}


# ============================================================================ Checkpoint 1: geometry & collision
def test_fk_matches_modern_robotics():
    for _ in range(10):
        q = RNG.uniform(-math.pi, math.pi, 2)
        p0, p1, p2 = lab.fk_2r(q, lab.L_PLANAR_2R)
        T = mr.FKinSpace(lab.M_PLANAR_2R, lab.SLIST_PLANAR_2R, q)
        assert np.allclose(p2, T[:2, 3], atol=1e-12)
        assert np.allclose(p0, 0) and np.isclose(np.linalg.norm(p1), 0.6)


def test_point_segment_distance():
    a, b = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    assert math.isclose(lab.point_segment_distance([0.5, 0.3], a, b), 0.3)
    assert math.isclose(lab.point_segment_distance([-0.3, 0.4], a, b), 0.5)     # beyond end a
    assert math.isclose(lab.point_segment_distance([1.6, -0.8], a, b), 1.0)     # beyond end b
    for _ in range(20):
        a, b, p = RNG.uniform(-1, 1, (3, 2))
        s = np.linspace(0, 1, 20001)[:, None]
        ref = np.min(np.linalg.norm(a + s * (b - a) - p, axis=1))
        assert abs(lab.point_segment_distance(p, a, b) - ref) < 1e-6


def test_capsule_circle_distance_sign():
    a, b = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    assert math.isclose(lab.capsule_circle_distance(a, b, 0.05, (0.5, 0.3), 0.1), 0.15)
    assert math.isclose(lab.capsule_circle_distance(a, b, 0.05, (0.5, 0.1), 0.1), -0.05)  # penetration


def test_clearance_hand_values(ward, eod):
    # q = 0: arm straight along +x. Ward: head (0.70, 0.30) r 0.12 -> 0.30 - 0.03 - 0.12 = 0.15.
    assert math.isclose(lab.clearance((0.0, 0.0), ward), 0.15, abs_tol=1e-9)
    # EOD: bumper (0.80, 0.10) r 0.08, link radius 0.04 -> 0.10 - 0.12 = -0.02 (collision)
    assert math.isclose(lab.clearance((0.0, 0.0), eod), -0.02, abs_tol=1e-9)
    assert not lab.config_free((0.0, 0.0), eod) and lab.config_free((0.0, 0.0), ward)
    # EOD floor y = -0.35: link 1 pointing straight down (theta1 = -90 deg) is in the ground
    assert lab.clearance((-math.pi / 2, 0.0), eod) < 0


def test_clearance_matches_brute_force(ward, eod):
    for sc in (ward, eod):
        for _ in range(25):
            q = RNG.uniform(-math.pi, math.pi, 2)
            assert abs(lab.clearance(q, sc) - brute_clearance(q, sc)) < 2e-3


def test_config_free_joint_limits(ward):
    q_ok = (0.0, 0.5)
    assert lab.config_free(q_ok, ward)
    assert not lab.config_free((3.3, 0.5), ward) and not lab.config_free((0.0, -3.3), ward)


def test_start_and_goal_are_free(ward, eod):
    for sc in (ward, eod):
        assert lab.config_free(sc["q_start"], sc) and lab.config_free(sc["q_goal"], sc)


# ============================================================================ Checkpoint 2: C-space grid & edges
def test_grid_angles_and_cells():
    k = 72
    th = lab.grid_angles(k)
    assert th.shape == (k,) and math.isclose(th[0], -math.pi) and math.isclose(th[1] - th[0], 2 * math.pi / k)
    for _ in range(20):
        c = tuple(RNG.integers(0, k, 2))
        assert lab.q_to_cell(lab.cell_to_q(c, k), k) == (int(c[0]), int(c[1]))
    assert lab.q_to_cell((math.pi - 1e-9, 0.0), k) == (0, 36)          # +pi wraps to the -pi cell


def test_cspace_grid_matches_brute_force(ward, eod, grids):
    for name, sc in (("ward", ward), ("eod", eod)):
        free = grids[name]
        assert free.shape == (K_FAST, K_FAST) and free.dtype == bool
        th = lab.grid_angles(K_FAST)
        for _ in range(60):
            i, j = RNG.integers(0, K_FAST, 2)
            c = brute_clearance((th[i], th[j]), sc)
            if abs(c) > 3e-3:                                           # skip grazing contacts
                assert free[i, j] == (c > 0)


def test_link1_obstacle_is_a_full_band(ward):
    """A C-obstacle of link 1 blocks every theta2 (a vertical band, MR Fig. 10.2): the IV pole."""
    k = K_LAB
    free = lab.cspace_grid(ward, k)
    i = lab.q_to_cell((math.atan2(0.55, -0.30), 0.0), k)[0]
    assert not free[i].any()
    assert 0.80 < free.mean() < 0.93


def test_motion_bound_is_an_upper_bound():
    L = lab.L_PLANAR_2R
    for _ in range(30):
        qa = RNG.uniform(-math.pi, math.pi, 2)
        dq = RNG.uniform(-0.3, 0.3, 2)
        Pa = dense_points(qa, L, 200)
        worst = max(np.max(np.linalg.norm(dense_points(qa + s * dq, L, 200) - Pa, axis=1))
                    for s in np.linspace(0, 1, 21))
        assert worst <= lab.motion_bound(dq, L) + 1e-9


def test_sampled_check_can_miss_but_certified_cannot():
    """Workshop/Practice 11 case: arm sweeps straight from 0 to 90 deg past a thin pole at (0.5, 0.5)."""
    sc = dict(L=np.array([0.6, 0.4]), link_radius=0.03, circles=np.array([[0.5, 0.5, 0.02]]), floor=None)
    qa, qb = (0.0, 0.0), (math.pi / 2, 0.0)
    assert lab.edge_free(qa, qb, sc, step=math.radians(18.0))            # samples 0, 18, ..., 90 all free
    assert not lab.edge_free(qa, qb, sc, step=math.radians(2.0))
    assert not lab.edge_free_certified(qa, qb, sc, step=math.radians(18.0))


def test_certified_true_implies_dense_free(ward, eod):
    n_true = 0
    for sc in (ward, eod):
        for _ in range(40):
            qa = RNG.uniform(-math.pi, math.pi, 2)
            qb = qa + RNG.uniform(-0.2, 0.2, 2)
            if np.any(np.abs(qb) > math.pi):
                continue
            if lab.edge_free_certified(qa, qb, sc, step=math.radians(2.0), wrap=False):
                n_true += 1
                assert all(brute_clearance(qa + s * (qb - qa), sc) > -1e-3 for s in np.linspace(0, 1, 201))
    assert n_true > 10


def test_edge_free_wraps_the_short_way(ward):
    qa, qb = (math.radians(170), math.radians(-60)), (math.radians(-170), math.radians(-60))
    assert lab.edge_free(qa, qb, ward, wrap=True)          # 20 deg through +-180: free
    assert not lab.edge_free(qa, qb, ward, wrap=False)     # 340 deg the long way: crosses the IV-pole band


# ============================================================================ Checkpoint 3: graph search
P3_EDGES = [("S", "A", 3.0), ("S", "B", 4.0), ("A", "C", 3.5), ("A", "E", 4.5), ("B", "C", 4.5),
            ("B", "D", 3.0), ("C", "G", 4.5), ("C", "E", 3.7), ("E", "G", 4.2), ("D", "G", 6.2)]
P3_H = dict(S=9.4, A=7.0, B=8.0, C=4.4, D=6.0, E=4.1, G=0.0)


def test_astar_workshop_problem_3():
    nb, _ = graph_fns(P3_EDGES)
    r = lab.astar("S", "G", nb, lambda n: P3_H[n])
    assert r["success"] and r["path"] == ["S", "A", "C", "G"] and math.isclose(r["cost"], 11.0)
    assert r["expanded"] == 4
    d = lab.dijkstra("S", "G", nb)
    assert d["path"] == ["S", "A", "C", "G"] and math.isclose(d["cost"], 11.0) and d["expanded"] == 7


def test_inconsistent_heuristic_needs_reopening():
    """Workshop 11 Problem 5: h admissible but not consistent (h(A) = 4 > c(A,B) + h(B) = 1)."""
    nb, _ = graph_fns([("S", "A", 1.0), ("A", "B", 1.0), ("S", "B", 3.0), ("B", "G", 3.0)])
    h = dict(S=2.0, A=4.0, B=0.0, G=0.0)
    r = lab.astar("S", "G", nb, lambda n: h[n], reopen=False)
    assert r["path"] == ["S", "B", "G"] and math.isclose(r["cost"], 6.0)       # MR Algorithm 10.1: suboptimal
    r = lab.astar("S", "G", nb, lambda n: h[n], reopen=True)
    assert r["path"] == ["S", "A", "B", "G"] and math.isclose(r["cost"], 5.0)


def test_astar_failure():
    nb, _ = graph_fns([("S", "A", 1.0), ("B", "G", 1.0)])
    r = lab.astar("S", "G", nb, lambda n: 0.0)
    assert not r["success"] and r["path"] is None and math.isinf(r["cost"])


def test_grid_neighbors():
    k, d = 10, 2 * math.pi / 10
    n4 = lab.grid_neighbors((0, 0), k, 4, wrap=True)
    n8 = lab.grid_neighbors((0, 0), k, 8, wrap=True)
    assert len(n4) == 4 and len(n8) == 8
    assert {c for c, _ in n4} == {(1, 0), (9, 0), (0, 1), (0, 9)}
    assert sorted(round(w / d, 6) for _, w in n8) == [1] * 4 + [round(math.sqrt(2), 6)] * 4
    assert len(lab.grid_neighbors((0, 0), k, 4, wrap=False)) == 2
    assert len(lab.grid_neighbors((0, 0), k, 8, wrap=False)) == 3
    assert len(lab.grid_neighbors((5, 5), k, 8, wrap=False)) == 8
    free = np.ones((k, k), bool)
    free[1, 0] = False
    assert (1, 0) not in {c for c, _ in lab.grid_neighbors((0, 0), k, 4, True, free)}


def test_heuristics_admissible_and_consistent(grids, eod):
    free = grids["eod"]
    k, wrap = K_FAST, False
    goal = lab.q_to_cell(eod["q_goal"], k)
    for conn, good, bad in ((4, ["zero", "manhattan", "euclidean", "octile"], []),
                            (8, ["zero", "euclidean", "octile"], ["manhattan"])):
        G = scipy_grid(free, conn, wrap)
        true = shortest_path(G, indices=goal[0] * k + goal[1])      # undirected: cost-to-go = cost-from-goal
        cells = [(i, j) for i in range(k) for j in range(k) if free[i, j] and np.isfinite(true[i * k + j])]
        for name in good:
            assert all(lab.grid_heuristic(c, goal, k, name, wrap) <= true[c[0] * k + c[1]] + 1e-9 for c in cells)
            for c in cells[::7]:                                     # consistency on every edge of the cell
                for n, w in lab.grid_neighbors(c, k, conn, wrap, free):
                    assert lab.grid_heuristic(c, goal, k, name, wrap) <= w + lab.grid_heuristic(n, goal, k, name, wrap) + 1e-9
        for name in bad:
            assert any(lab.grid_heuristic(c, goal, k, name, wrap) > true[c[0] * k + c[1]] + 1e-9 for c in cells)


@pytest.mark.parametrize("name", ["ward", "eod"])
@pytest.mark.parametrize("conn,heur", [(4, "manhattan"), (8, "euclidean"), (8, "octile")])
def test_grid_astar_optimal_vs_scipy(name, conn, heur, grids, ward, eod):
    sc = {"ward": ward, "eod": eod}[name]
    free, k, wrap = grids[name], K_FAST, bool(sc["wrap"])
    s, g = lab.q_to_cell(sc["q_start"], k), lab.q_to_cell(sc["q_goal"], k)
    ref = shortest_path(scipy_grid(free, conn, wrap), indices=s[0] * k + s[1])[g[0] * k + g[1]]
    r = lab.grid_astar(free, s, g, conn, heur, wrap=wrap)
    dj = lab.grid_astar(free, s, g, conn, "zero", wrap=wrap)
    assert r["success"] and math.isclose(r["cost"], ref, rel_tol=1e-9)
    assert math.isclose(dj["cost"], ref, rel_tol=1e-9)
    assert r["expanded"] < dj["expanded"]                           # the heuristic saves work
    # the path is a chain of free neighbouring cells whose costs add up
    path = r["path"]
    assert path[0] == s and path[-1] == g and all(free[c] for c in path)
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        nb = dict(lab.grid_neighbors(a, k, conn, wrap))
        assert b in nb
        total += nb[b]
    assert math.isclose(total, r["cost"], rel_tol=1e-9)


def test_inflated_astar_bound(grids, ward):
    free, k = grids["ward"], K_FAST
    s, g = lab.q_to_cell(ward["q_start"], k), lab.q_to_cell(ward["q_goal"], k)
    opt = lab.grid_astar(free, s, g, 8, "euclidean", wrap=True)
    for eta in (1.5, 3.0):
        r = lab.grid_astar(free, s, g, 8, "euclidean", eta=eta, wrap=True)
        assert r["success"] and opt["cost"] - 1e-9 <= r["cost"] <= eta * opt["cost"] + 1e-9


def test_topology_wrap_matters(grids, ward):
    """The ward IV-pole band cuts the square; only the torus (wrap) connects start and goal."""
    free, k = grids["ward"], K_FAST
    s, g = lab.q_to_cell(ward["q_start"], k), lab.q_to_cell(ward["q_goal"], k)
    assert lab.grid_astar(free, s, g, 8, "euclidean", wrap=True)["success"]
    assert not lab.grid_astar(free, s, g, 8, "euclidean", wrap=False)["success"]


def test_enclosed_goal_reports_failure():
    k = 12
    free = np.ones((k, k), bool)
    free[4:9, 4:9] = False
    free[6, 6] = True                                               # goal inside a closed C-obstacle ring
    r = lab.grid_astar(free, (0, 0), (6, 6), 8, "euclidean", wrap=True)
    assert not r["success"] and r["path"] is None


# ============================================================================ Checkpoint 4: the planner
@pytest.mark.parametrize("name", ["ward", "eod"])
def test_plan_2r(name, ward, eod):
    sc = {"ward": ward, "eod": eod}[name]
    res = lab.plan_2r(sc, k=K_LAB, connectivity=8, heuristic="euclidean", certified=True)
    assert res["success"]
    qp = np.asarray(res["q_path"])
    assert np.allclose(lab.wrap_angle(qp[0] - sc["q_start"]), 0) and np.allclose(lab.wrap_angle(qp[-1] - sc["q_goal"]), 0)
    steps = np.abs(np.diff(qp, axis=0))
    assert np.all(steps <= math.sqrt(2) * 2 * math.pi / K_LAB + 1e-9)     # continuous (unwrapped) path
    for qa, qb in zip(qp[:-1], qp[1:]):                                    # every segment free (dense check)
        for s in np.linspace(0, 1, 25):
            assert brute_clearance(qa + s * (qb - qa), sc) > -1e-3
    if not sc["wrap"]:
        assert np.all(np.abs(qp) <= math.pi + 1e-9)                         # never through the cable stop
    assert lab.path_length(qp) > 0


def test_unwrap_and_length():
    qs = np.array([[3.0, 0.0], [-3.1, 0.0], [-3.0, 0.1]])
    u = lab.unwrap_path(qs)
    assert np.allclose(u[1], [2 * math.pi - 3.1, 0.0])
    assert math.isclose(lab.path_length(u), (2 * math.pi - 6.1) + math.hypot(0.1, 0.1))
