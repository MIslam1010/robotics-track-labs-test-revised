"""Auto-checks for SBEG 46X Lab 7 (Cosserat-rod statics vs constant-curvature and Euler-Bernoulli models; tip-force
estimation for a tendon-driven catheter).

    LAB_IMPL=starter  pytest -q test_lab.py      # untouched starter: every test that calls your code fails with
                                                 # NotImplementedError; the 2 data tests pass

LAB_IMPL is the name of the module to test (set it to starter, or to the name of your own copy of starter.py); it
is imported from this folder.
Ground truth: closed-form results written out independently in this file -- the constant-curvature arc, the
exponential of a twist (modern_robotics.MatrixExp6), the Euler-Bernoulli cantilever formulas, the classical
large-deflection elastica of a cantilever with a transverse tip force (elliptic integrals, scipy.special), the
tip-angle compliance of an arc derived in Workshop 7 Problem 5 -- and hand values from Workshop 7.
"""
import importlib
import math
import os
import sys
from pathlib import Path

import modern_robotics as mr
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import ellipe, ellipeinc, ellipk, ellipkinc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("LAB_IMPL", "starter")
lab = importlib.import_module(os.environ["LAB_IMPL"])

CAT = lab.load_catheter()
L, EI, R = CAT["L"], CAT["EI"], CAT["r"]
mm = 1e-3


# ============================================================================ independent references
def elastica_tip(alpha):
    """Cantilever of unit length and unit EI with a transverse (dead) tip force alpha = P L^2/EI, perpendicular to the
    undeformed rod.  Classical elliptic-integral solution: with k^2 = (1 + sin th_L)/2 and phi1 = asin(1/(sqrt2 k)),
    sqrt(alpha) = K(k) - F(phi1, k);  x_L = sqrt(2 sin th_L / alpha);  y_L = 1 - 2 (E(k) - E(phi1, k)) / sqrt(alpha)."""
    def eq(th):
        m = (1 + math.sin(th)) / 2
        phi1 = math.asin(1 / math.sqrt(2 * m))
        return ellipk(m) - ellipkinc(phi1, m) - math.sqrt(alpha)
    th = brentq(eq, 1e-9, math.pi / 2 - 1e-9)
    m = (1 + math.sin(th)) / 2
    phi1 = math.asin(1 / math.sqrt(2 * m))
    return np.array([math.sqrt(2 * math.sin(th) / alpha), 1 - 2 * (ellipe(m) - ellipeinc(phi1, m)) / math.sqrt(alpha), th])


def arc(kappa, ell):
    return np.array([math.sin(kappa * ell) / kappa, (1 - math.cos(kappa * ell)) / kappa, kappa * ell])


# ============================================================================ data
def test_catheter_json():
    assert (L, EI, R) == (0.10, 1.0e-3, 1.2e-3)
    assert CAT["tau_nominal"] == 6.0
    assert CAT["sigma_p"] == pytest.approx(1e-4) and CAT["sigma_theta"] == pytest.approx(math.radians(0.5))


def test_log_loads():
    log = lab.load_csv("catheter_log.csv")
    assert set(log) == {"t", "tau", "x_meas", "y_meas", "theta_meas", "Fx_ref", "Fy_ref"}
    assert len(log["t"]) == 40 and np.all((log["tau"] >= 4) & (log["tau"] <= 10))
    assert np.max(np.hypot(log["Fx_ref"], log["Fy_ref"])) < 0.13


# ============================================================================ Part A
def test_A1_cc_tip_closed_form_and_straight_limit():
    np.testing.assert_allclose(lab.cc_tip(7.2, L), arc(7.2, L), atol=1e-12)
    np.testing.assert_allclose(lab.cc_tip(-5.0, 0.04), arc(-5.0, 0.04), atol=1e-12)
    np.testing.assert_allclose(lab.cc_tip(0.0, L), [L, 0, 0], atol=1e-15)
    np.testing.assert_allclose(lab.cc_tip(1e-9, L), [L, 0.5e-9 * L**2, 1e-10], atol=1e-14)
    # Workshop 7 P1: tip at (91.58, 34.47) mm, 0.72 rad
    np.testing.assert_allclose(lab.cc_tip(7.2, L) / [mm, mm, 1], [91.58, 34.47, 0.72], atol=5e-3)


def test_A2_A5_curvature_models():
    assert lab.tendon_curvature(6.0, R, EI) == pytest.approx(7.2)
    assert lab.pneumatic_curvature(5e4, 50e-6, 4e-3, 2e-3) == pytest.approx(5.0)
    assert lab.pneumatic_curvature(1.5e5, 50e-6, 4e-3, 2e-3) == pytest.approx(15.0)


def test_A3_cc_transform_is_the_exponential_of_a_twist():
    T = lab.cc_transform(7.2, L)
    assert T.shape == (4, 4)
    p = arc(7.2, L)
    Tref = mr.RpToTrans(np.array([[math.cos(0.72), -math.sin(0.72), 0], [math.sin(0.72), math.cos(0.72), 0], [0, 0, 1]]),
                        np.array([p[0], p[1], 0.0]))
    np.testing.assert_allclose(T, Tref, atol=1e-12)
    # composition: e^{[V] l1} e^{[V] l2} = e^{[V](l1 + l2)}
    np.testing.assert_allclose(lab.cc_transform(7.2, 0.03) @ lab.cc_transform(7.2, 0.07), T, atol=1e-12)


def test_A4_pcc_tip():
    # Workshop 7 P4: two pneumatic segments
    np.testing.assert_allclose(lab.pcc_tip([5.0, 15.0], [0.04, 0.04]) / [mm, mm, 1], [74.313, 22.877, 0.8], atol=1e-3)
    # equal curvatures = one arc; a single segment = cc_tip; straight segments add up
    np.testing.assert_allclose(lab.pcc_tip([10.0, 10.0], [0.04, 0.04]), arc(10.0, 0.08), atol=1e-12)
    np.testing.assert_allclose(lab.pcc_tip([3.0], [0.05]), arc(3.0, 0.05), atol=1e-12)
    np.testing.assert_allclose(lab.pcc_tip([0.0, 0.0], [0.02, 0.03]), [0.05, 0, 0], atol=1e-12)
    # independent: SE(3) product of the exponentials (MR Eq. 3.88)
    T = np.eye(4)
    for k, l in [(4.0, 0.03), (-8.0, 0.02), (12.0, 0.025)]:
        T = T @ mr.MatrixExp6(mr.VecTose3(np.array([0, 0, k, 1, 0, 0]) * l))
    z = lab.pcc_tip([4.0, -8.0, 12.0], [0.03, 0.02, 0.025])
    np.testing.assert_allclose(z[:2], T[:2, 3], atol=1e-12)
    assert z[2] == pytest.approx(math.atan2(T[1, 0], T[0, 0]))


# ============================================================================ Part B
def test_B1_rod_rhs_values_and_vectorised():
    z = np.array([0.01, 0.02, 0.3, 0.1, -0.2, 0.004])
    d = lab.rod_rhs(0.0, z, EI, f=(0.5, -1.0), kappa0=2.0)
    c, s = math.cos(0.3), math.sin(0.3)
    np.testing.assert_allclose(d, [c, s, 0.004 / EI + 2.0, -0.5, 1.0, -(c * -0.2 - s * 0.1)], atol=1e-12)
    Z = np.tile(z[:, None], (1, 5))
    D = lab.rod_rhs(np.zeros(5), Z, EI, f=(0.5, -1.0), kappa0=2.0)
    assert np.shape(D) == (6, 5)
    np.testing.assert_allclose(np.asarray(D)[:, 3], d, atol=1e-12)


def test_B2_shooting_residual_linear_case():
    """Straight rod, tip force P: m(s) = sigma - P s exactly (n constant), so R(sigma) = sigma - P L (Workshop P2)."""
    P = 1e-4                   # tiny load: geometry stays straight to 1e-6
    for sigma in (0.0, 2e-5):
        r = lab.shooting_residual([0.0, P, sigma], L, EI, F=(0.0, P), M=0.0)
        np.testing.assert_allclose(r, [0, 0, sigma - P * L], atol=1e-9)


def test_B3_pure_tip_moment_gives_the_cc_arc():
    sol = lab.solve_rod_shooting(L, EI, F=(0, 0), M=R * 6.0)
    np.testing.assert_allclose(lab.tip_pose(sol), arc(7.2, L), atol=1e-8)
    np.testing.assert_allclose(sol["m"], R * 6.0, atol=1e-10)
    assert len(sol["s"]) == 101 and sol["s"][-1] == pytest.approx(L)


@pytest.mark.parametrize("alpha", [0.1, 1.0, 2.0])
def test_B3_shooting_matches_the_elastica(alpha):
    P = alpha * EI / L**2
    sol = lab.solve_rod_shooting(L, EI, F=(0, P))
    np.testing.assert_allclose(lab.tip_pose(sol) / [L, L, 1], elastica_tip(alpha), atol=1e-7)
    assert sol["m"][0] == pytest.approx(P * sol["x"][-1], abs=1e-9)          # base moment = P x_L
    np.testing.assert_allclose(sol["ny"], P, atol=1e-12)


def test_B3_small_loads_reduce_to_euler_bernoulli():
    P, w = 1e-3, 1e-2          # alpha = 0.01
    z = lab.tip_pose(lab.solve_rod_shooting(L, EI, F=(0, P)))
    assert z[1] == pytest.approx(P * L**3 / (3 * EI), rel=2e-3)
    zw = lab.tip_pose(lab.solve_rod_shooting(L, EI, f=(0, w)))
    assert zw[1] == pytest.approx(w * L**4 / (8 * EI), rel=2e-3)
    assert zw[2] == pytest.approx(w * L**3 / (6 * EI), rel=2e-3)


def test_B3_distributed_load_force_balance():
    """Workshop/Practice P3 arm: n(0) = F + f L exactly; self-weight + tip camera."""
    P, w = 0.05 * 9.81, 0.5
    sol = lab.solve_rod_shooting(0.40, 0.50, F=(0, -P), f=(0, -w))
    np.testing.assert_allclose([sol["nx"][0], sol["ny"][0]], [0, -(P + w * 0.40)], atol=1e-9)
    np.testing.assert_allclose(lab.tip_pose(sol) / [mm, mm, 1], [399.138, -24.042, -0.08889], atol=2e-3)


def test_B4_solve_bvp_agrees_with_shooting_and_elastica():
    P = 1.0 * EI / L**2
    b = lab.solve_rod_bvp(L, EI, F=(0, P))
    np.testing.assert_allclose(lab.tip_pose(b) / [L, L, 1], elastica_tip(1.0), atol=1e-6)
    # tendon + oblique tip force + weight, against shooting
    F, f, M = (0.03, -0.05), (0.0, -0.2), R * 6.0
    b = lab.solve_rod_bvp(L, EI, F=F, M=M, f=f)
    s = lab.solve_rod_shooting(L, EI, F=F, M=M, f=f)
    np.testing.assert_allclose(lab.tip_pose(b), lab.tip_pose(s), atol=1e-7)
    np.testing.assert_allclose(b["m"], s["m"], atol=1e-7)


def test_B5_euler_bernoulli():
    np.testing.assert_allclose(lab.euler_bernoulli_tip(L, EI, P=0.01), [L, 1 / 3 * 1e-2, 0.05], atol=1e-12)
    np.testing.assert_allclose(lab.euler_bernoulli_tip(0.4, 0.5, P=-0.4905, w=-0.5), [0.4, -0.024128, -0.08914667],
                               atol=1e-7)
    np.testing.assert_allclose(lab.euler_bernoulli_tip(L, EI, M=7.2e-3), [L, 0.036, 0.72], atol=1e-12)


# ============================================================================ Part C
def test_C1_catheter_tip_unloaded_is_cc_and_loaded_unbends():
    np.testing.assert_allclose(lab.catheter_tip(6.0, (0, 0), CAT), arc(7.2, L), atol=1e-8)
    d = lab.contact_direction(0.72)
    z = lab.catheter_tip(6.0, 0.02 * d, CAT)
    assert z[2] < 0.72 and z[1] < arc(7.2, L)[1]


def test_C2_compare_models():
    out = lab.compare_models(6.0, [0.0, 0.05, 0.1], CAT)
    assert out["cosserat"].shape == (3, 3) and out["eb"].shape == (3, 3)
    np.testing.assert_allclose(out["cc"], arc(7.2, L), atol=1e-10)
    assert out["err_cc"][0] == pytest.approx(0.0, abs=1e-8)
    assert np.all(np.diff(out["err_cc"]) > 0)
    # EB about the straight rod already misses the unloaded arc: |(L, r tau L^2/2EI) - CC tip|
    assert out["err_eb"][0] == pytest.approx(math.hypot(L - arc(7.2, L)[0], 0.036 - arc(7.2, L)[1]), abs=1e-8)
    # independent check of one Cosserat point with the BVP solver
    zb = lab.tip_pose(lab.solve_rod_bvp(L, EI, F=0.1 * lab.contact_direction(0.72), M=R * 6.0))
    np.testing.assert_allclose(out["cosserat"][2], zb, atol=1e-7)


# ============================================================================ Part D
def test_D1_compliance_straight_is_euler_bernoulli():
    C = lab.compliance_matrix(0.0, CAT)
    np.testing.assert_allclose(C, [[0, 0], [0, L**3 / (3 * EI)], [0, L**2 / (2 * EI)]], rtol=1e-4, atol=1e-9)


def test_D1_compliance_bent_matches_closed_forms():
    """Workshop P5 / Practice P5: tip-angle compliance of an arc for a force along the tip normal and tangent."""
    C = lab.compliance_matrix(6.0, CAT)
    k, th = 7.2, 0.72
    t, nu = np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)])
    assert (C @ nu)[2] == pytest.approx((1 - math.cos(k * L)) / (k**2 * EI), rel=1e-4)
    assert (C @ t)[2] == pytest.approx((k * L - math.sin(k * L)) / (k**2 * EI), rel=1e-4)


def test_D2_linear_estimate_noise_free():
    d = lab.contact_direction(0.72)
    t = np.array([math.cos(0.72), math.sin(0.72)])
    # contact model (force along d): one unknown, accurate for a small pure normal force
    z = lab.catheter_tip(6.0, 0.005 * d, CAT)
    Fn = lab.estimate_force_linear(z, 6.0, CAT, direction=d)
    assert Fn[0] * d[1] - Fn[1] * d[0] == pytest.approx(0.0, abs=1e-12)               # along d
    assert np.dot(Fn, d) == pytest.approx(0.005, rel=0.01)
    # both components: only for a very small force (ill-conditioned, Practice P5)
    F = 2e-4 * d + 0.8e-4 * t
    z = lab.catheter_tip(6.0, F, CAT)
    np.testing.assert_allclose(lab.estimate_force_linear(z, 6.0, CAT), F, atol=2e-5)


def test_D3_nonlinear_estimate_recovers_a_large_force():
    d = lab.contact_direction(0.72)
    F = 0.10 * d
    z = lab.catheter_tip(6.0, F, CAT)
    np.testing.assert_allclose(lab.estimate_force_nl(z, 6.0, CAT, direction=d), F, atol=1e-6)
    np.testing.assert_allclose(lab.estimate_force_nl(z, 6.0, CAT), F, atol=1e-5)
    # Workshop P5 (c): only the measured angle matters for the hand estimate; the linear one is 2 % low here
    lin = lab.estimate_force_linear(z, 6.0, CAT, direction=d)
    assert np.dot(lin, d) < 0.10


def test_D4_D5_bench_log():
    log = lab.load_csv("catheter_log.csv")
    sub = {k: v[:12] for k, v in log.items()}
    Fref = np.c_[sub["Fx_ref"], sub["Fy_ref"]]
    th = sub["tau"] * R / EI * L
    e_nl = lab.force_errors(lab.estimate_log(sub, CAT, "nl", True), Fref, th)
    e_lin = lab.force_errors(lab.estimate_log(sub, CAT, "linear", True), Fref, th)
    e_2d = lab.force_errors(lab.estimate_log(sub, CAT, "linear", False), Fref, th)
    assert set(e_nl) == {"rms", "rms_normal", "rms_tangential"}
    assert e_nl["rms_normal"] < 0.010 < e_lin["rms_normal"]                 # nonlinear model beats the linear one
    assert e_2d["rms_tangential"] > 10 * e_nl["rms_normal"]                 # 2-D estimate: tangential ill-conditioned


def test_D5_force_errors_decomposition():
    E = lab.force_errors([[0.1, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.1]], [0.0, 0.0])
    assert E["rms"] == pytest.approx(0.1)
    assert E["rms_tangential"] == pytest.approx(math.sqrt(0.005))
    assert E["rms_normal"] == pytest.approx(math.sqrt(0.005))
