"""Auto-checks for SBEG 361 Lab 9 -- analytic and numerical IK of the course arm; all-solutions search.

Usage:   LAB_IMPL=starter  python -m pytest -q test_lab.py     (your work, file starter.py)
         LAB_IMPL=solution python -m pytest -q test_lab.py     (reference; default)
Ground truth: the official `modern_robotics` library (FKinSpace, FKinBody, IKinBody, IKinSpace,
MatrixExp6), the MR Example 6.1 iteration table (printed p. 230), the 2R formulas of MR pp. 218-219,
and the Paden-Kahan definitions of MLS pp. 99-103 (checked by forward substitution).
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

d2r = np.deg2rad


def wrap(a):
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


def model(name):
    S, M, info = lab.load_robot(name)
    return S, M, mr.Adjoint(mr.TransInv(M)) @ S, info


def rot_point(w, r, th, p):
    """e^[xi th] p for a zero-pitch unit twist with direction w through r (independent of the lab)."""
    w = np.asarray(w, float) / np.linalg.norm(w)
    S = np.r_[w, -np.cross(w, r)]
    return (mr.MatrixExp6(mr.VecTose3(S * th)) @ np.r_[p, 1.0])[:3]


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def contains(sols, q, tol=1e-6):
    return any(np.all(np.abs(wrap(np.asarray(s) - q)) < tol) for s in sols)


# ---------------------------------------------------------------- Checkpoint 1: Paden-Kahan subproblems
def test_pk1_workshop_value():
    """Workshop P2(a): axis z through (0.1, 0, 0), p = (0.3, 0, 0.2), q = (0.1, 0.2, 0.2) -> 90 deg."""
    assert lab.pk1([0, 0, 1], [0.1, 0, 0], [0.3, 0, 0.2], [0.1, 0.2, 0.2]) == pytest.approx(np.pi / 2, abs=1e-12)


def test_pk1_random():
    rng = np.random.default_rng(1)
    for _ in range(20):
        w, r, p = unit(rng.normal(size=3)), rng.normal(size=3), rng.normal(size=3)
        th = rng.uniform(-3.0, 3.0)
        q = rot_point(w, r, th, p)
        assert wrap(lab.pk1(w, r, p, q) - th) == pytest.approx(0, abs=1e-9)


def test_pk1_no_solution():
    with pytest.raises(ValueError):
        lab.pk1([0, 0, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0.5])      # different heights along the axis
    with pytest.raises(ValueError):
        lab.pk1([0, 0, 1], [0, 0, 0], [1, 0, 0], [0, 2, 0])        # different radii


def test_pk2_workshop_values():
    """Workshop P2(b): pan (z) and tilt (-y) axes through the origin."""
    sols = lab.pk2([0, 0, 1], [0, -1, 0], [0, 0, 0], [0.1, 0, 0], [0.05, 0.05, np.sqrt(0.005)])
    assert len(sols) == 2
    assert contains(sols, d2r([45, 45])) and contains(sols, d2r([-135, 135]))


def test_pk2_random():
    rng = np.random.default_rng(2)
    for _ in range(20):
        w1, w2, r, p = unit(rng.normal(size=3)), unit(rng.normal(size=3)), rng.normal(size=3), rng.normal(size=3)
        t1, t2 = rng.uniform(-3, 3, 2)
        q = rot_point(w1, r, t1, rot_point(w2, r, t2, p))
        sols = lab.pk2(w1, w2, r, p, q)
        assert 1 <= len(sols) <= 2
        assert contains(sols, [t1, t2], 1e-7)
        for a, b in sols:
            np.testing.assert_allclose(rot_point(w1, r, a, rot_point(w2, r, b, p)), q, atol=1e-9)


def test_pk2_no_solution():
    assert lab.pk2([0, 0, 1], [0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 2]) == []   # |q - r| != |p - r|


def test_pk3_workshop_values():
    """Workshop P2(c): elbow axis y through (0.30, 0, 0), wrist p = (0.65, 0, 0), shoulder q = 0, delta = 0.5."""
    sols = lab.pk3([0, 1, 0], [0.30, 0, 0], [0.65, 0, 0], [0, 0, 0], 0.5)
    np.testing.assert_allclose(sorted(sols), d2r([-79.71344, 79.71344]), atol=1e-6)


def test_pk3_random():
    rng = np.random.default_rng(3)
    for _ in range(20):
        w, r, p, q = unit(rng.normal(size=3)), rng.normal(size=3), rng.normal(size=3), rng.normal(size=3)
        th = rng.uniform(-3, 3)
        delta = np.linalg.norm(q - rot_point(w, r, th, p))
        sols = lab.pk3(w, r, p, q, delta)
        assert 1 <= len(sols) <= 2 and contains(sols, [th], 1e-6)
        for a in sols:
            assert np.linalg.norm(q - rot_point(w, r, a, p)) == pytest.approx(delta, abs=1e-9)
    assert lab.pk3([0, 0, 1], [0, 0, 0], [1, 0, 0], [3, 0, 0], 0.5) == []   # circle and sphere do not meet


# ---------------------------------------------------------------- Checkpoint 2: analytic IK
def test_ik_2r_values():
    """Quiz 9 Q1 numbers (planar 2R, L = 0.6, 0.4) and the MR p. 218 righty/lefty convention."""
    sols = lab.ik_2r(0.6, 0.4, 0.6, 0.4)
    assert len(sols) == 2
    np.testing.assert_allclose(sols[0], [0, np.pi / 2], atol=1e-9)              # righty: theta2 > 0
    np.testing.assert_allclose(sols[1], [2 * np.arctan2(0.4, 0.6), -np.pi / 2], atol=1e-9)
    assert lab.ik_2r(1.2, 0.0, 0.6, 0.4) == [] and lab.ik_2r(0.1, 0.1, 0.6, 0.4) == []


def test_ik_2r_fk():
    rng = np.random.default_rng(4)
    for _ in range(20):
        q = rng.uniform(-3, 3, 2)
        x, y = 0.6 * np.cos(q[0]) + 0.4 * np.cos(q.sum()), 0.6 * np.sin(q[0]) + 0.4 * np.sin(q.sum())
        sols = lab.ik_2r(x, y, 0.6, 0.4)
        assert contains(sols, q, 1e-7)


@pytest.mark.parametrize("name", ["course_arm", "desktop_6r"])
def test_ik_course_arm_fk(name):
    S, M, B, info = model(name)
    rng = np.random.default_rng(5)
    for _ in range(25):
        q = rng.uniform(-np.pi, np.pi, 6)
        T = mr.FKinSpace(M, S, q)
        sols = lab.ik_course_arm(T, info["geometry"])
        assert 1 <= len(sols) <= 8
        assert contains(sols, q, 1e-7), "the configuration that produced T must be among the solutions"
        for th in sols:
            np.testing.assert_allclose(mr.FKinSpace(M, S, th), T, atol=1e-10)
            assert np.all(np.asarray(th) >= -np.pi - 1e-12) and np.all(np.asarray(th) < np.pi + 1e-12)


P8 = np.deg2rad([0, -90, 60, 0, 45, 0])   # pose P8 of lecture example WE-A7: 8 real branches


def test_ik_course_arm_solution_counts():
    """q_home: 4 solutions (the wrist-flip branches put joint 4 out of the 2R's reach); P8: 8."""
    S, M, B, info = model("course_arm")
    for q, n in [(info["q_home"], 4), (P8, 8)]:
        sols = lab.ik_course_arm(mr.FKinSpace(M, S, q), info["geometry"])
        assert len(sols) == n
        assert len({tuple(np.round(wrap(s), 6)) for s in sols}) == n


def test_ik_course_arm_unreachable():
    S, M, B, info = model("course_arm")
    T = np.eye(4)
    T[:3, 3] = [0.5, 0.0, 0.1]                      # 0.5 m away: beyond the 0.28 m reach
    assert lab.ik_course_arm(T, info["geometry"]) == []


# ---------------------------------------------------------------- Checkpoint 3: Newton-Raphson
def test_newton_raphson_scalar_table():
    """Workshop P4(a): g = 0.48 sin(theta) - 0.24 from theta0 = 0."""
    th, hist = lab.newton_raphson(lambda t: 0.48 * np.sin(t) - 0.24, lambda t: 0.48 * np.cos(t), 0.0,
                                  tol=1e-12, max_iter=10)
    angles = [float(np.rad2deg(np.squeeze(h[1]))) for h in hist]
    np.testing.assert_allclose(angles[:4], [0.0, 28.6479, 29.9912, 30.0], atol=1e-4)
    assert hist[0][0] == 0 and abs(float(np.squeeze(hist[-1][2]))) <= 1e-12


def test_newton_raphson_vector_2r():
    """Workshop P4(c): coordinate NR for the 2R arm of MR Example 6.1 converges to the OTHER solution."""
    xd = np.array([0.5 * np.sqrt(3) - 0.5, 0.5 + 0.5 * np.sqrt(3)])
    f = lambda t: np.array([np.cos(t[0]) + np.cos(t[0] + t[1]), np.sin(t[0]) + np.sin(t[0] + t[1])])
    J = lambda t: np.array([[-np.sin(t[0]) - np.sin(t[0] + t[1]), -np.sin(t[0] + t[1])],
                            [np.cos(t[0]) + np.cos(t[0] + t[1]), np.cos(t[0] + t[1])]])
    th, hist = lab.newton_raphson(lambda t: f(t) - xd, J, d2r([0, 30]), tol=1e-10, max_iter=20)
    np.testing.assert_allclose(hist[1][1], [-np.sqrt(3), 30 * np.pi / 180 + 4.7320508], atol=1e-6)
    np.testing.assert_allclose(wrap(th), d2r([120, -90]), atol=1e-8)


def test_ik_newton_body_mr_example_6_1():
    """MR Example 6.1 table (p. 230): (0, 30) -> (34.23, 79.18) -> (29.98, 90.22) -> (30.00, 90.00)."""
    S, M, B, info = model("mr_2r_point")
    Tsd = mr.FKinBody(M, B, d2r([30, 90]))
    th, ok, table = lab.ik_newton_body(B, M, Tsd, d2r([0, 30]), 1e-3, 1e-4)
    assert ok and len(table) == 4
    got = np.array([np.rad2deg(r["theta"]) for r in table])
    np.testing.assert_allclose(got, [[0, 30], [34.23, 79.18], [29.98, 90.22], [30.00, 90.00]], atol=6e-3)
    np.testing.assert_allclose(table[0]["V_b"][[2, 3, 4]], [1.571, 0.498, 1.858], atol=6e-4)
    np.testing.assert_allclose([table[1]["w_err"], table[1]["v_err"]], [0.115, 0.131], atol=6e-4)


@pytest.mark.parametrize("name", ["course_arm", "eod_6r_ugv"])
def test_ik_newton_body_matches_mr(name):
    S, M, B, info = model(name)
    rng = np.random.default_rng(6)
    for _ in range(6):
        q = rng.uniform(-2, 2, 6)
        q0 = q + rng.uniform(-0.3, 0.3, 6)
        T = mr.FKinSpace(M, S, q)
        th, ok, table = lab.ik_newton_body(B, M, T, q0, 1e-6, 1e-7)
        th_mr, ok_mr = mr.IKinBody(B, M, T, q0, 1e-6, 1e-7)
        assert ok == ok_mr
        np.testing.assert_allclose(th, th_mr, atol=1e-8)
        assert table[0]["i"] == 0 and np.allclose(table[0]["theta"], q0)
        if ok:
            np.testing.assert_allclose(mr.FKinSpace(M, S, th), T, atol=1e-6)


def test_ik_newton_space_matches_mr():
    S, M, B, info = model("course_arm")
    rng = np.random.default_rng(7)
    for _ in range(6):
        q = rng.uniform(-2, 2, 6)
        q0 = q + rng.uniform(-0.3, 0.3, 6)
        T = mr.FKinSpace(M, S, q)
        th, ok, table = lab.ik_newton_space(S, M, T, q0, 1e-6, 1e-7)
        th_mr, ok_mr = mr.IKinSpace(S, M, T, q0, 1e-6, 1e-7)
        assert ok == ok_mr
        np.testing.assert_allclose(th, th_mr, atol=1e-8)


def test_ik_newton_reports_failure():
    """An unreachable pose: NR must stop after max_iter and report failure, not raise."""
    S, M, B, info = model("course_arm")
    T = np.eye(4)
    T[:3, 3] = [0.5, 0.0, 0.1]
    th, ok, table = lab.ik_newton_body(B, M, T, info["q_home"], 1e-3, 1e-4, max_iter=20)
    assert not ok and len(table) == 21


# ---------------------------------------------------------------- Checkpoint 4: all-solutions search
def test_unique_solutions():
    a = np.array([0.1, 3.1, -1.0])
    sols = lab.unique_solutions([a, a + 2 * np.pi * np.array([1, 0, -1]), a + 1e-6, a + 0.1])
    assert len(sols) == 2


def test_all_solutions_search_finds_analytic_set():
    S, M, B, info = model("course_arm")
    analytic_p8 = lab.ik_course_arm(mr.FKinSpace(M, S, P8), info["geometry"])
    for q, n in [(P8, 8), (info["q_home"], 4)]:
        T = mr.FKinSpace(M, S, q)
        analytic = lab.ik_course_arm(T, info["geometry"])
        sols, counts, nconv = lab.all_solutions_search(S, M, T, n_starts=200, seed=0)
        assert len(sols) == len(counts) and sum(counts) == nconv <= 200
        for s in sols:
            assert contains(analytic, s, 1e-6), "every numerical solution must be one of the analytic ones"
        if len(sols) < n:
            # a branch with a very small basin can be missed by 200 starts on some CPUs (round-off in the Newton
            # steps differs between processors); more starts must then find it
            sols, _, _ = lab.all_solutions_search(S, M, T, n_starts=2000, seed=1)
            for s in sols:
                assert contains(analytic, s, 1e-6), "every numerical solution must be one of the analytic ones"
        assert len(sols) == n, "enough random starts find every analytic branch of this pose"
    sols20, counts20, _ = lab.all_solutions_search(S, M, mr.FKinSpace(M, S, P8), n_starts=20, seed=0)
    # 6 of 8 on the reference computer. One start lies near a singularity (theta3 ~ 0), so the branch it reaches
    # depends on round-off: some processors find all 8 with these 20 starts (as the C++ build does). Same checks as C++.
    assert 1 <= len(sols20) <= 8 and len(sols20) == len(counts20)
    for s in sols20:
        assert contains(analytic_p8, s, 1e-6), "every numerical solution must be one of the analytic ones"


def test_select_solution():
    lim = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    sols = [np.array([0.5, 0.5]), np.array([-0.2, 0.1]), np.array([2.0, 0.0])]
    np.testing.assert_allclose(lab.select_solution(sols, [-0.3, 0.0], lim), [-0.2, 0.1])
    assert lab.select_solution([np.array([2.0, 0.0])], [0, 0], lim) is None
    # an angle may need a 2 pi shift to fall inside the limits
    lim2 = np.array([[0.0, 6.0]])
    np.testing.assert_allclose(lab.select_solution([np.array([-1.0])], [5.0], lim2), [2 * np.pi - 1.0])


# ---------------------------------------------------------------- Checkpoint 5: biopsy needle guide
def test_needle_guide_pose():
    T = lab.needle_guide_pose([0.2, 0.06, 0.02], [0, 0, -1], 0.01, x_hint=(0, -1, 0))
    np.testing.assert_allclose(T[:3, 2], [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(T[:3, 0], [0, -1, 0], atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], [0.2, 0.06, 0.03], atol=1e-12)
    np.testing.assert_allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-12)
    assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0)
    T2 = lab.needle_guide_pose([0, 0, 0], [1, 0, 0], 0.0, x_hint=(1, 0, 0))   # hint parallel to the axis
    np.testing.assert_allclose(T2[:3, 0], [0, 1, 0], atol=1e-12)


def test_biopsy_plan():
    """Checkpoint 5 values: analytic solutions, feasible solutions, selected solution for T1..T5."""
    S, M, B, info = model("desktop_6r")
    lim, qh = np.array(info["limits"]), np.array(info["q_home"])
    expected_n = [4, 4, 4, 4, 8]
    expected_feasible = [4, 3, 4, 2, 3]
    for k, tgt in enumerate(lab.load_biopsy_targets()):
        T = lab.needle_guide_pose(tgt["entry"], tgt["axis"], 0.01, x_hint=(0, -1, 0))
        sols = lab.ik_course_arm(T, info["geometry"])
        assert len(sols) == expected_n[k], tgt["id"]
        feas = [s for s in sols if lab.select_solution([s], qh, lim) is not None]
        assert len(feas) == expected_feasible[k], tgt["id"]
        best = lab.select_solution(sols, qh, lim)
        np.testing.assert_allclose(mr.FKinSpace(M, S, best), T, atol=1e-10)
        assert np.all(best >= lim[:, 0] - 1e-12) and np.all(best <= lim[:, 1] + 1e-12)
