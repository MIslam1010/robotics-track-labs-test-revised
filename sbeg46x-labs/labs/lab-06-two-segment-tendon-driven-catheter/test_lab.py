"""Auto-checks for SBEG 46X Lab 6 (constant-curvature kinematics of a two-segment tendon-driven catheter).

    LAB_IMPL=starter  python -m pytest -q test_lab.py   # untouched starter: every test that calls your code fails
                                                        # with NotImplementedError; the 2 data tests pass

LAB_IMPL is the name of the module to test (set it to starter, or to the name of your own copy of starter.py); it
is imported from this folder.
Ground truth: modern_robotics (MatrixExp6, VecTose3, TransInv, Adjoint), closed-form arc geometry, central finite
differences written in this file, and the hand values of the Week 6 workshop.
"""
import importlib
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import modern_robotics as mr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("LAB_IMPL", "starter")
lab = importlib.import_module(os.environ["LAB_IMPL"])

CAT = lab.load_catheter()
RNG = np.random.default_rng(461)
Q_TEST = [np.array([0.010, 0.4, -0.7, 1.1, 0.3]),
          np.array([0.000, -1.2, 0.2, 0.0, 0.0]),
          np.array([0.025, 0.05, 0.02, -0.6, 1.5])]


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def ref_segment(theta_v, ell):
    """Independent closed form of one CC segment written with numerically stable sinc-type factors
    (Rodrigues' formula MR Eq. 3.51 for R; the arc for p), valid also at and near a straight segment."""
    tx, ty = float(theta_v[0]), float(theta_v[1])
    th = math.hypot(tx, ty)
    a = 1.0 - th**2 / 6 if th < 1e-4 else math.sin(th) / th                      # sin(th)/th
    b = 0.5 - th**2 / 24 if th < 1e-4 else 2 * math.sin(th / 2) ** 2 / th**2    # (1 - cos th)/th^2
    w = np.array([-ty, tx, 0.0])
    W = mr.VecToso3(w)
    R = np.eye(3) + a * W + b * W @ W
    p = ell * np.array([b * tx, b * ty, a])
    return mr.RpToTrans(R, p)


def ref_fk(q):
    T = CAT["T_base"].copy() @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, q[0]], [0, 0, 0, 1.0]])
    return T @ ref_segment(q[1:3], CAT["ell"][0]) @ ref_segment(q[3:5], CAT["ell"][1])


def fd_space_jacobian(q, h=1e-6):
    T = ref_fk(q)
    Ti = mr.TransInv(T)
    J = np.zeros((6, 5))
    for i in range(5):
        e = np.zeros(5)
        e[i] = h
        dT = (ref_fk(q + e) - ref_fk(q - e)) / (2 * h)
        J[:, i] = mr.se3ToVec(dT @ Ti)
    return J, T


# ============================================================================ data
def test_catheter_json():
    assert CAT["ell"].tolist() == [0.040, 0.025] and CAT["d"].tolist() == [0.0010, 0.0010]
    assert CAT["EI"].tolist() == [1.0e-4, 0.6e-4]
    assert np.allclose(np.diff(CAT["sigma_deg"], axis=1), 120.0)
    R = CAT["T_base"][:3, :3]
    assert np.allclose(R.T @ R, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)


def test_data_files_load():
    ring = lab.load_csv("ablation_ring.csv")
    assert set(ring) == {"t", "x", "y", "z"} and len(ring["t"]) == 801
    log = lab.load_csv("tendon_log.csv")
    assert len(log["pose"]) == 5 and "dl6" in log


# ============================================================================ Part A -- one segment
def test_arc_transform_hand_values():
    """Workshop P1: kappa = 10 1/m, ell = pi/20 m -> a quarter circle of radius 0.1 m."""
    T = lab.arc_transform(10.0, 0.0, math.pi / 20)
    assert np.allclose(T[:3, 3], [0.1, 0.0, 0.1], atol=1e-12)
    assert np.allclose(T[:3, :3], ry(math.pi / 2), atol=1e-12)
    T = lab.arc_transform(10.0, math.pi / 2, math.pi / 20)
    assert np.allclose(T[:3, 3], [0.0, 0.1, 0.1], atol=1e-12)
    T = lab.arc_transform(0.0, 1.3, 0.03)                       # straight
    assert np.allclose(T, np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.03], [0, 0, 0, 1]]), atol=1e-12)


def test_arc_transform_matches_mr_exponential():
    for _ in range(20):
        k, phi, ell = RNG.uniform(0.5, 60), RNG.uniform(-math.pi, math.pi), RNG.uniform(0.01, 0.06)
        S = np.array([-k * math.sin(phi), k * math.cos(phi), 0, 0, 0, 1.0])   # twist per unit arc length
        T_ref = mr.MatrixExp6(mr.VecTose3(S * ell))
        assert np.allclose(lab.arc_transform(k, phi, ell), T_ref, atol=1e-12)


def test_arc_tip_z_axis_is_the_backbone_tangent():
    k, phi, ell, h = 18.0, 0.7, 0.05, 1e-7
    T = lab.arc_transform(k, phi, ell)
    tangent = (lab.arc_transform(k, phi, ell + h)[:3, 3] - lab.arc_transform(k, phi, ell - h)[:3, 3]) / (2 * h)
    assert np.allclose(tangent, T[:3, 2], atol=1e-7)
    assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-12)


def test_bend_arc_roundtrip():
    k, phi, th = lab.bend_to_arc([0.0, 0.0], 0.04)
    assert k == 0.0 and th == 0.0 and phi == 0.0
    k, phi, th = lab.bend_to_arc([-0.3, 0.4], 0.04)
    assert th == pytest.approx(0.5) and k == pytest.approx(12.5) and phi == pytest.approx(math.atan2(0.4, -0.3))
    assert np.allclose(lab.arc_to_bend(k, phi, 0.04), [-0.3, 0.4])


def test_segment_twist_and_exponential():
    theta_v, ell = np.array([0.6, -0.8]), 0.03
    V = lab.segment_twist(theta_v, ell)
    assert np.allclose(V, [0.8, 0.6, 0, 0, 0, 0.03])
    k, phi, _ = lab.bend_to_arc(theta_v, ell)
    assert np.allclose(lab.segment_transform_exp(theta_v, ell), lab.arc_transform(k, phi, ell), atol=1e-12)
    assert np.allclose(lab.segment_transform_exp(theta_v, ell), ref_segment(theta_v, ell), atol=1e-12)


def test_tendon_lengths_hand_values():
    """Workshop P2: d = 1 mm, ell = 40 mm, bent 90 deg towards phi = 0."""
    sig = np.radians([0, 120, 240])
    l = lab.tendon_lengths([math.pi / 2, 0.0], 0.040, 0.001, sig)
    assert np.allclose(l, 0.040 - 0.001 * math.pi / 2 * np.cos(sig), atol=1e-15)
    assert l[0] == pytest.approx(0.040 - 0.001 * math.pi / 2)
    assert np.sum(l - 0.040) == pytest.approx(0.0, abs=1e-15)          # 3 equally spaced tendons: zero sum


def test_bend_from_tendons_roundtrip():
    sig = np.radians([0, 120, 240])
    for _ in range(10):
        theta_v = RNG.uniform(-2, 2, 2)
        dl = lab.tendon_lengths(theta_v, 0.04, 0.001, sig) - 0.04
        assert np.allclose(lab.bend_from_tendons(dl, 0.001, sig), theta_v, atol=1e-12)
        s = np.column_stack([np.cos(sig), np.sin(sig)])
        assert np.allclose(-(2 / (3 * 0.001)) * s.T @ dl, theta_v, atol=1e-12)    # closed form


def test_tension_allocation():
    sig = np.radians([0, 120, 240])
    s = np.column_stack([np.cos(sig), np.sin(sig)])
    for _ in range(10):
        m = RNG.uniform(-0.01, 0.01, 2)
        t = lab.tension_allocation(m, 0.001, sig)
        assert t.min() == pytest.approx(0.0, abs=1e-12) and np.all(t >= -1e-12)
        assert np.allclose(0.001 * s.T @ t, m, atol=1e-14)


def test_segment_tensions_single_tendon_hand_value():
    """Bending towards a tendon needs only that tendon: t = EI theta / (ell d)."""
    sig = np.radians([0, 120, 240])
    t = lab.segment_tensions([0.5, 0.0], 0.04, 0.001, sig, 1.0e-4)
    assert np.allclose(t, [1.0e-4 * 0.5 / (0.04 * 0.001), 0, 0], atol=1e-12)        # 1.25 N


# ============================================================================ Part B -- two-segment catheter
def test_catheter_fk_matches_independent_product():
    for q in Q_TEST + [RNG.uniform([0, -2, -2, -2, -2], [0.03, 2, 2, 2, 2]) for _ in range(5)]:
        assert np.allclose(lab.catheter_fk(q, CAT), ref_fk(q), atol=1e-12)


def test_catheter_fk_straight():
    T = lab.catheter_fk([0.012, 0, 0, 0, 0], CAT)
    z = CAT["T_base"][:3, 2]
    assert np.allclose(T[:3, 3], CAT["T_base"][:3, 3] + (0.012 + 0.065) * z, atol=1e-12)
    assert np.allclose(T[:3, :3], CAT["T_base"][:3, :3], atol=1e-12)


def test_catheter_tendon_lengths():
    l0 = lab.catheter_tendon_lengths(np.zeros(5), CAT)
    assert np.allclose(l0, [0.040] * 3 + [0.065] * 3)
    q = Q_TEST[0]
    q2 = q.copy()
    q2[0] = 0.0
    assert np.allclose(lab.catheter_tendon_lengths(q, CAT), lab.catheter_tendon_lengths(q2, CAT))
    s2 = np.column_stack([np.cos(CAT["sigma"][1]), np.sin(CAT["sigma"][1])])
    dl2 = lab.catheter_tendon_lengths(q, CAT)[3:] - 0.065
    assert np.allclose(dl2, -0.001 * s2 @ (q[1:3] + q[3:5]), atol=1e-15)


def test_catheter_tendon_jacobian():
    A = lab.catheter_tendon_jacobian(CAT)
    assert A.shape == (6, 5) and np.allclose(A[:, 0], 0)
    q, h = Q_TEST[2], 1e-6
    for i in range(5):
        e = np.zeros(5)
        e[i] = h
        col = (lab.catheter_tendon_lengths(q + e, CAT) - lab.catheter_tendon_lengths(q - e, CAT)) / (2 * h)
        assert np.allclose(A[:, i], col, atol=1e-9)


def test_bend_from_catheter_tendons_and_log():
    for q in Q_TEST:
        dl = lab.catheter_tendon_lengths(q, CAT) - lab.catheter_tendon_lengths(np.zeros(5), CAT)
        assert np.allclose(lab.bend_from_catheter_tendons(dl, CAT), q[1:], atol=1e-12)
    log = lab.load_csv("tendon_log.csv")
    truth = np.array([[0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.2], [0.4, -0.6, 0.8, 0.3],
                      [-1.0, 0.5, 1.5, -0.4], [1.6, 1.0, -0.9, 0.9]])
    for k in range(5):
        dl = np.array([log[f"dl{i + 1}"][k] for i in range(6)])
        assert np.allclose(lab.bend_from_catheter_tendons(dl, CAT), truth[k], atol=0.03)   # 5 um encoder noise


def test_catheter_tensions_equilibrium():
    assert np.allclose(lab.catheter_tensions(np.zeros(5), CAT), 0)
    s = [np.column_stack([np.cos(CAT["sigma"][j]), np.sin(CAT["sigma"][j])]) for j in range(2)]
    for q in Q_TEST:
        t = lab.catheter_tensions(q, CAT)
        assert np.all(t >= -1e-12) and t[3:].min() == pytest.approx(0, abs=1e-12)
        m2 = 0.001 * s[1].T @ t[3:]
        m1 = 0.001 * s[0].T @ t[:3] + m2
        assert np.allclose(m2, CAT["EI"][1] / CAT["ell"][1] * q[3:5], atol=1e-12)
        assert np.allclose(m1, CAT["EI"][0] / CAT["ell"][0] * q[1:3], atol=1e-12)


def test_dexp_left_matches_finite_difference():
    V, Vd, h = np.array([0.3, -0.9, 0.2, 0.01, 0.02, 0.04]), np.array([0.5, 0.1, -0.2, 0.03, -0.01, 0.02]), 1e-6
    dT = (mr.MatrixExp6(mr.VecTose3(V + h * Vd)) - mr.MatrixExp6(mr.VecTose3(V - h * Vd))) / (2 * h)
    ref = mr.se3ToVec(dT @ mr.TransInv(mr.MatrixExp6(mr.VecTose3(V))))
    assert np.allclose(lab.dexp_left(V) @ Vd, ref, atol=1e-8)
    assert np.allclose(lab.dexp_left(np.zeros(6)), np.eye(6))


def test_space_jacobian_matches_finite_difference():
    for q in Q_TEST:
        Jfd, _ = fd_space_jacobian(q)
        assert np.allclose(lab.catheter_jacobian_space(q, CAT), Jfd, atol=1e-7)


def test_space_jacobian_insertion_column():
    J = lab.catheter_jacobian_space(Q_TEST[0], CAT)
    assert np.allclose(J[:, 0], mr.Adjoint(CAT["T_base"]) @ [0, 0, 0, 0, 0, 1])


def test_tip_position_jacobian():
    for q in Q_TEST:
        Jp = lab.tip_position_jacobian(q, CAT)
        h = 1e-6
        for i in range(5):
            e = np.zeros(5)
            e[i] = h
            col = (ref_fk(q + e)[:3, 3] - ref_fk(q - e)[:3, 3]) / (2 * h)
            assert np.allclose(Jp[:, i], col, atol=1e-8)
        # same thing from the body Jacobian: pdot = R_sb J_b,v qdot
        T = ref_fk(q)
        Jb = mr.Adjoint(mr.TransInv(T)) @ lab.catheter_jacobian_space(q, CAT)
        assert np.allclose(Jp, T[:3, :3] @ Jb[3:], atol=1e-12)


# ============================================================================ Part C -- resolved-rate control
def test_clamp_q():
    q = lab.clamp_q([0.05, 3.0, 4.0, 0.1, 0.0], CAT)
    assert q[0] == pytest.approx(0.030)
    assert np.linalg.norm(q[1:3]) == pytest.approx(CAT["theta_max"]) and q[1] / q[2] == pytest.approx(0.75)
    assert np.allclose(q[3:], [0.1, 0.0])
    assert lab.clamp_q([-0.01, 0, 0, 0, 0], CAT)[0] == 0.0


def test_dls_inverse():
    J = RNG.normal(size=(3, 5))
    assert np.allclose(lab.dls_inverse(J, 0.0), np.linalg.pinv(J), atol=1e-12)
    lam = 0.1
    assert np.allclose(lab.dls_inverse(J, lam), np.linalg.solve(J.T @ J + lam**2 * np.eye(5), J.T), atol=1e-12)


def test_resolved_rate_reaches_a_target():
    q_goal = np.array([0.018, 0.9, -0.4, -0.5, 0.8])
    p_goal = ref_fk(q_goal)[:3, 3]
    out = lab.go_to(np.array([0.010, 0.2, 0.0, 0.2, 0.0]), p_goal, CAT, dt=0.01, K=5.0, steps=600)
    assert out["err"][-1] < 1e-6
    assert np.all(np.diff(out["err"][:200]) < 1e-9) or out["err"][200] < 0.5 * out["err"][0]
    assert np.all(np.linalg.norm(out["q"][:, 1:3], axis=1) <= CAT["theta_max"] + 1e-12)


def test_track_ablation_ring():
    ring = lab.load_csv("ablation_ring.csv")
    P = np.column_stack([ring["x"], ring["y"], ring["z"]])
    start = lab.go_to(np.array([0.010, 0.0, 0.3, 0.0, 0.3]), P[0], CAT, dt=0.01, K=5.0, steps=600)
    assert start["err"][-1] < 1e-6
    res = lab.track_path(start["q"][-1], ring["t"], P, CAT, K=20.0)
    assert res["q"].shape == (801, 5) and res["tendon_dl"].shape == (801, 6)
    assert res["err"].max() < 0.3e-3                                     # 0.3 mm
    for qq in res["q"][::50]:
        assert np.allclose(lab.catheter_fk(qq, CAT), ref_fk(qq), atol=1e-12)
    assert np.all((res["q"][:, 0] >= 0) & (res["q"][:, 0] <= 0.030))
    assert np.allclose(res["tendon_dl"][:, :3].sum(axis=1), 0, atol=1e-12)
