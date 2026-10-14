"""Auto-checks for SBEG 361 Lab 8 -- statics, manipulability maps of an assistive arm, workspace design.

Usage:   LAB_IMPL=starter  python -m pytest -q test_lab.py     (your work, file starter.py; default)
         LAB_IMPL=lab8_<ID> python -m pytest -q test_lab.py    (your renamed file)
Ground truth: the official `modern_robotics` library (JacobianSpace, JacobianBody, FKinSpace, Adjoint),
independent moment-arm formulas, and the textbook numbers of MR Ch. 5 (2R arm with L1 = L2 = 1 at
theta = (0, pi/4) and (0, 3 pi/4), printed p. 171; MR Eqs. 5.26-5.27 and pp. 196-197).
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
RNG = np.random.default_rng(10)


def model(name):
    S, M, info = lab.load_robot(name)
    return S, M, mr.Adjoint(mr.TransInv(M)) @ S, info


def rand_q(n, k=5, seed=3):
    rng = np.random.default_rng(seed)
    return [rng.uniform(-np.pi, np.pi, n) for _ in range(k)]


# ---------------------------------------------------------------- Checkpoint 1: statics
def test_joint_torques_is_JT_F():
    J = RNG.normal(size=(6, 5))
    F = RNG.normal(size=6)
    assert np.allclose(lab.joint_torques(J, F), J.T @ F)


@pytest.mark.parametrize("name", ["course_arm", "eod_6r_ugv", "srs_7r_feeding"])
def test_space_and_body_torques_agree(name):
    S, M, B, _ = model(name)
    for q in rand_q(S.shape[1]):
        T = mr.FKinSpace(M, S, q)
        Fb = RNG.normal(size=6)
        Fs = lab.body_to_space_wrench(Fb, T)
        assert np.allclose(Fs, mr.Adjoint(mr.TransInv(T)).T @ Fb, atol=1e-10)
        tb = lab.joint_torques(mr.JacobianBody(B, q), Fb)
        ts = lab.joint_torques(mr.JacobianSpace(S, q), Fs)
        assert np.allclose(tb, ts, atol=1e-10)


def test_power_balance():
    """tau . thetadot = F . V for every thetadot (MR p. 188)."""
    S, M, B, _ = model("course_arm")
    q = np.array([0.3, -0.7, 1.2, -1.9, -1.2, 0.5])
    Jb = mr.JacobianBody(B, q)
    F = RNG.normal(size=6)
    for _ in range(5):
        qd = RNG.normal(size=6)
        assert np.isclose(lab.joint_torques(Jb, F) @ qd, F @ (Jb @ qd))


def test_holding_wrench_moment_arms():
    """tau_i = w_i . ((p - q_i) x f) for revolute joint i with axis (w_i, q_i) at the current pose."""
    S, M, _, info = model("course_arm")
    q = np.array(info["q_home"])
    m, g = 0.25, np.array([0, 0, -9.81])
    T = mr.FKinSpace(M, S, q)
    p = T[:3, 3]
    F = lab.holding_wrench_space(m, p, g)
    assert np.allclose(F[3:], [0, 0, m * 9.81])
    assert np.allclose(F[:3], np.cross(p, F[3:]))
    Js = mr.JacobianSpace(S, q)
    tau = lab.joint_torques(Js, F)
    for i in range(6):
        w, v = Js[:3, i], Js[3:, i]
        qi = np.cross(w, v)                       # point on the axis (|w| = 1)
        assert np.isclose(tau[i], w @ np.cross(p - qi, F[3:]), atol=1e-12)


def test_exoskeleton_load_torques():
    """Workshop P1: arm exoskeleton (0.30, 0.35) m holding 2 kg at the hand, gravity -y."""
    S, M, _, info = model("arm_exo_2r")
    g = np.array([0, -9.81, 0])
    for q, expected in [((0, 90), (5.886, 0.0)), ((0, 0), (12.753, 6.867)), ((90, 0), (0.0, 0.0))]:
        th = d2r(np.array(q, float))
        p = mr.FKinSpace(M, S, th)[:3, 3]
        tau = lab.joint_torques(mr.JacobianSpace(S, th), lab.holding_wrench_space(2.0, p, g))
        assert np.allclose(tau, expected, atol=1e-3)


def test_wrench_from_torques():
    J = lab.jacobian_2r(d2r([0.0, 90.0]), 0.30, 0.35)
    assert np.allclose(lab.wrench_from_torques(J, [5.0, 2.0]), [-5.7143, 10.0], atol=1e-4)
    S, M, B, _ = model("course_arm")
    q = np.array([0.3, -0.7, 1.2, -1.9, -1.2, 0.5])
    Jb = mr.JacobianBody(B, q)
    F = RNG.normal(size=6)
    assert np.allclose(lab.wrench_from_torques(Jb, Jb.T @ F), F)
    with pytest.raises(ValueError):
        lab.wrench_from_torques(np.ones((6, 5)), np.ones(5))          # not square (n != 6)
    with pytest.raises(ValueError):
        lab.wrench_from_torques(lab.jacobian_2r([0.4, 0.0], 0.35, 0.35), [1.0, 1.0])   # singular


def test_wrench_null_space():
    S, M, B, _ = model("hapkit")
    N = lab.wrench_null_space(B)
    assert N.shape == (6, 5)                       # MR p. 189: 1-DOF door resists a 5-D wrench space
    assert np.allclose(B.T @ N, 0, atol=1e-12) and np.allclose(N.T @ N, np.eye(5), atol=1e-10)
    S, M, B, info = model("course_arm")
    q = np.array(info["q_home"]); q[2] = 0.0       # elbow straight
    N = lab.wrench_null_space(mr.JacobianBody(B, q))
    assert N.shape == (6, 1)
    assert np.allclose(mr.JacobianBody(B, q).T @ N, 0, atol=1e-9)


# ---------------------------------------------------------------- Checkpoint 2: ellipsoids and measures
def test_ellipsoid_mr_2r():
    """MR p. 171 (Figs. 5.3, 5.5): 2R, L1 = L2 = 1."""
    a, V = lab.manipulability_ellipsoid(lab.jacobian_2r([0, np.pi / 4], 1, 1))
    assert np.allclose(a, [2.0731, 0.3411], atol=1e-4)
    a, V = lab.manipulability_ellipsoid(lab.jacobian_2r([0, 3 * np.pi / 4], 1, 1))
    assert np.allclose(a, [1.0731, 0.6589], atol=1e-4)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_ellipsoid_reconstructs_A(seed):
    J = np.random.default_rng(seed).normal(size=(3, 6))
    a, V = lab.manipulability_ellipsoid(J)
    assert np.all(np.diff(a) <= 1e-12)                        # longest first
    assert np.allclose(V.T @ V, np.eye(3), atol=1e-10)
    assert np.allclose(V @ np.diag(a ** 2) @ V.T, J @ J.T, atol=1e-10)
    f, Vf = lab.force_ellipsoid(J)
    assert np.allclose(Vf, V) and np.allclose(f * a, 1.0)     # same axes, reciprocal lengths (MR p. 197)


def test_force_ellipsoid_singular_is_infinite():
    f, _ = lab.force_ellipsoid(lab.jacobian_2r([0.3, 0.0], 0.35, 0.35))
    assert np.isinf(f[-1]) and np.isfinite(f[0])


def test_measures_values():
    J = lab.jacobian_2r(d2r([0.0, 90.0]), 0.35, 0.35)          # Workshop P2
    m = lab.manipulability_measures(J)
    assert np.isclose(m["mu1"], 2.6180, atol=1e-4) and np.isclose(m["mu2"], 6.8541, atol=1e-4)
    assert np.isclose(m["mu3"], 0.1225, atol=1e-6)
    for q in rand_q(2, 6):
        J = lab.jacobian_2r(q, 0.35, 0.35)
        m = lab.manipulability_measures(J)
        assert np.isclose(m["mu3"], abs(np.linalg.det(J)))     # square J: sqrt(det JJ^T) = |det J|
        assert np.isclose(m["mu2"], m["mu1"] ** 2) and m["mu1"] >= 1 - 1e-12


def test_measures_singular():
    m = lab.manipulability_measures(lab.jacobian_2r([0.2, 0.0], 1, 1))
    assert np.isinf(m["mu1"]) and np.isinf(m["mu2"]) and np.isclose(m["mu3"], 0, atol=1e-12)


def test_volume_product_constant():
    """mu3(A) mu3(B) = 1 with B = A^-1 (MR p. 197)."""
    for q in rand_q(2, 6, seed=5):
        J = lab.jacobian_2r(q, 0.35, 0.35)
        a, _ = lab.manipulability_ellipsoid(J)
        f, _ = lab.force_ellipsoid(J)
        assert np.isclose(np.prod(a) * np.prod(f), 1.0)


def test_angular_measures_frame_independent_linear_not():
    """J_omega of J_s and J_b differ by a rotation (same ellipsoid shape); J_v does not (MR p. 196)."""
    S, M, B, _ = model("course_arm")
    q = np.array([0.3, -0.7, 1.2, -1.9, -1.2, 0.5])
    Js, Jb = mr.JacobianSpace(S, q), mr.JacobianBody(B, q)
    ms, mb = (lab.manipulability_measures(lab.select_rows(J, "angular")) for J in (Js, Jb))
    assert np.isclose(ms["mu1"], mb["mu1"]) and np.isclose(ms["mu3"], mb["mu3"])
    ls, lb = (lab.manipulability_measures(lab.select_rows(J, "linear")) for J in (Js, Jb))
    assert not np.isclose(ls["mu3"], lb["mu3"], rtol=1e-3)


# ---------------------------------------------------------------- Checkpoint 3: singularities
@pytest.mark.parametrize("name", ["course_arm", "eod_6r_ugv"])
def test_rank_space_equals_body(name):
    S, M, B, info = model(name)
    qs = rand_q(6) + [np.zeros(6), np.array(info["q_home"])]
    for q in qs:
        rs = lab.singularity_report(mr.JacobianSpace(S, q))["rank"]
        rb = lab.singularity_report(mr.JacobianBody(B, q))["rank"]
        assert rs == rb == np.linalg.matrix_rank(mr.JacobianSpace(S, q), tol=1e-6)


def test_eod_singularity_cases():
    """Workshop P3: stretched (Cases I + II), wrist (Case I), regular, shoulder (Case III)."""
    S, M, B, _ = model("eod_6r_ugv")
    elbow_x = 0.10 + 0.80 * np.cos(np.radians(45.0))         # theta2 = -45 deg (inside [-60, 110] deg)
    th3 = 45.0 - np.degrees(np.arccos(-elbow_x / 0.75))        # -107.57 deg: wrist centre on axis 1
    cases = [((0, 0, 0, 0, 0, 0), 4), ((0, -30, 60, 0, 0, 0), 5), ((0, -30, 60, 0, 45, 0), 6),
             ((0, -45, th3, 0, 45, 0), 5)]
    for q, rank in cases:
        rep = lab.singularity_report(mr.JacobianSpace(S, d2r(np.array(q, float))))
        assert rep["rank"] == rank
    rep = lab.singularity_report(mr.JacobianSpace(S, d2r(np.array([0, -30, 60, 0, 45, 0.]))))
    assert np.isclose(rep["sigma_min"], 0.29938, atol=1e-4)


def test_lost_twists_and_null_motions():
    S, M, B, _ = model("eod_6r_ugv")
    q = d2r(np.array([0, -30, 60, 0, 0, 0.]))
    J = mr.JacobianSpace(S, q)
    rep = lab.singularity_report(J)
    U, N = rep["lost_twists"], rep["null_motions"]
    assert U.shape == (6, 1) and N.shape == (6, 1)
    assert np.allclose(J.T @ U, 0, atol=1e-9) and np.allclose(J @ N, 0, atol=1e-9)
    n = N[:, 0] / N[3, 0]
    assert np.allclose(n, [0, 0, 0, 1, 0, -1], atol=1e-9)    # forearm roll against gripper roll


def test_singularity_independent_of_tool_frame():
    """MR Fig. 5.10(b): moving {b} (needle guide) does not move the singularities."""
    Sa, Ma, _, info = model("course_arm")
    Sd, Md, _, _ = model("desktop_6r")
    for q3 in [0.0, 0.3]:
        q = np.array(info["q_home"]); q[2] = q3
        ra = lab.singularity_report(mr.JacobianSpace(Sa, q))["rank"]
        rd = lab.singularity_report(mr.JacobianSpace(Sd, q))["rank"]
        assert ra == rd == (5 if q3 == 0.0 else 6)


# ---------------------------------------------------------------- Checkpoint 4: maps and design (rehab_2r)
def test_map_2r_formula_and_nan():
    xs = np.linspace(-0.8, 0.8, 9)
    ys = np.linspace(-0.8, 0.8, 9)
    mp = lab.manipulability_map_2r(0.35, 0.35, xs, ys, "mu3")
    assert mp.shape == (9, 9)
    for a, y in enumerate(ys):
        for b, x in enumerate(xs):
            r2 = x * x + y * y
            if r2 > 0.7 ** 2:
                assert np.isnan(mp[a, b])
            else:
                c2 = (r2 - 2 * 0.35 ** 2) / (2 * 0.35 ** 2)
                assert np.isclose(mp[a, b], 0.35 * 0.35 * np.sqrt(max(0.0, 1 - c2 * c2)), atol=1e-12)


def test_map_respects_limits_and_base():
    _, _, _, info = model("rehab_2r")
    lim = np.array(info["limits"])
    mp = lab.manipulability_map_2r(0.35, 0.35, [0.0, 0.0], [-0.3, 0.3], "mu3", 1, lim)
    assert np.isnan(mp[0, 0]) and not np.isnan(mp[1, 0])     # theta1 in [0, pi]: no targets behind the base
    m0 = lab.manipulability_map_2r(0.35, 0.35, [0.1], [0.4], "mu1")[0, 0]
    m1 = lab.manipulability_map_2r(0.35, 0.35, [0.1], [0.3], "mu1", base=(0.0, -0.1))[0, 0]
    assert np.isclose(m0, m1)                                  # same point relative to the base


def test_region_score_values():
    _, _, _, info = model("rehab_2r")
    L1, L2 = info["L"]
    lim = np.array(info["limits"])
    pts = lab.load_targets()
    assert pts.shape == (25, 2)
    w, fr = lab.region_score(L1, L2, pts, "mu3", 1, lim, base=(0.0, -0.10))
    assert np.isclose(w, 0.106088, atol=1e-5) and fr == 1.0
    w, fr = lab.region_score(L1, L2, pts, "mu3", 1, lim, base=(0.0, 0.0))
    assert np.isclose(fr, 0.92)
    w, fr = lab.region_score(L1, L2, pts, "mu1", 1, lim, base=(0.0, -0.10))
    assert np.isclose(w, 3.87994, atol=1e-4)


def test_design_search():
    _, _, _, info = model("rehab_2r")
    lim = np.array(info["limits"])
    pts = lab.load_targets()
    ratios = np.round(np.arange(0.30, 0.71, 0.05), 2)
    offsets = [0.0, 0.05, 0.10, 0.15, 0.20]
    rows, best = lab.design_search(0.70, ratios, offsets, pts, lim)
    assert len(rows) == 45
    assert np.isclose(best["ratio"], 0.5) and np.isclose(best["b"], 0.10)
    assert np.isclose(best["worst"], 0.106088, atol=1e-5)
    rows, best = lab.design_search(0.70, ratios, offsets, pts, lim, measure="mu1")
    assert np.isclose(best["ratio"], 0.6) and np.isclose(best["b"], 0.0)
    assert np.isclose(best["worst"], 2.29716, atol=1e-4)
    _, none = lab.design_search(0.40, [0.5], [0.0], pts, lim)
    assert none is None                                        # too short to reach every target


# ---------------------------------------------------------------- Checkpoint 5: spatial arms
def test_feeding_path_measures():
    S, M, _, _ = model("srs_7r_feeding")
    t, Q = lab.load_feeding_csv()
    assert Q.shape == (100, 7)
    m = lab.measures_along_path(S, M, Q, "linear")
    assert np.isclose(m["mu3"][0], 0.135794, atol=1e-5)
    assert np.isclose(m["mu3"][-1], 0.014930, atol=1e-5)
    assert np.argmin(m["mu3"]) == 99
    assert np.isclose(m["mu1"][-1], 57.617, atol=1e-2)
    ma = lab.measures_along_path(S, M, Q[[0, -1]], "angular")
    assert np.isclose(ma["mu1"][-1], 1.3132, atol=1e-4)       # orientation stays well conditioned


def test_joint_space_map_course_arm():
    S, M, B, info = model("course_arm")
    q0 = np.array(info["q_home"])
    g3 = d2r(np.array([-30.0, 0.0, 30.0]))
    g5 = d2r(np.array([-90.0, 0.0, 45.0]))
    mp = lab.joint_space_map(S, M, q0, 2, 4, g3, g5, "sigma_min", "all")
    assert mp.shape == (3, 3)
    assert np.allclose(mp[:, 1], 0, atol=1e-9)                 # elbow straight (theta3 = 0)
    assert np.allclose(mp[1, :], 0, atol=1e-9)                 # wrist singular (theta5 = 0)
    one = lab.joint_space_map(S, M, q0, 2, 4, [q0[2]], [q0[4]], "mu3", "all")
    assert np.isclose(one[0, 0], abs(np.linalg.det(mr.JacobianBody(B, q0))))
    with pytest.raises(ValueError):
        lab.select_rows(np.eye(6), "upper")
