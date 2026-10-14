"""Auto-checks for SBEG 46X Lab 10 (CBF-QP safety filter versus saturation on an exoskeleton joint and the Hapkit).

    LAB_IMPL=starter  python -m pytest -q test_lab.py    # untouched starter: every test that calls your code
                                                         # fails with NotImplementedError; the 5 data tests pass
    LAB_IMPL=lab10_<ID> python -m pytest -q test_lab.py  # your renamed file

LAB_IMPL is the name of the module to test; it is imported from this folder.
Ground truth: brute-force minimisation over a fine grid (1-D QP), enumeration of the active sets of the KKT conditions
(2-D QP), direct evaluation of the joint and handle dynamics (the barrier condition must hold with equality on the
bound), the joint inertia and gravity torque of the shared course models (modern_robotics MassMatrix / GravityForces
via course_models), hand-checked Workshop numbers, and the reference runs in data/expected.json.
"""
import importlib
import itertools
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
lab = importlib.import_module(os.environ.get("LAB_IMPL", "starter"))
import course_models as cm  # noqa: E402

P = json.loads((HERE / "data" / "lab10.json").read_text())
E, KN, H = P["elbow"], P["knee"], P["hapkit"]
EXP = json.loads((HERE / "data" / "expected.json").read_text())
RNG = np.random.default_rng(4610)


def kkt_qp(u_nom, A, b, W):
    """Exact solution of min (u-u_nom)'W(u-u_nom) s.t. A u >= b by enumerating active sets (small problems)."""
    u_nom = np.asarray(u_nom, float)
    best, best_cost = None, np.inf
    m = len(b)
    Wi = np.linalg.inv(W)
    for k in range(m + 1):
        for act in itertools.combinations(range(m), k):
            act = list(act)
            if act:
                Aa, ba = A[act], b[act]
                S = Aa @ Wi @ Aa.T
                if abs(np.linalg.det(S)) < 1e-12:
                    continue
                lam = np.linalg.solve(S, ba - Aa @ u_nom)
                if np.any(lam < -1e-10):
                    continue
                u = u_nom + Wi @ Aa.T @ lam
            else:
                u = u_nom.copy()
            if np.all(A @ u >= b - 1e-9):
                c = (u - u_nom) @ W @ (u - u_nom)
                if c < best_cost:
                    best, best_cost = u, c
    return best


# ------------------------------------------------------------------------------ data and provided code (pass)
def test_params_match_course_models():
    for name, j in (("elbow", E), ("knee", KN)):
        assert abs(j["M"] - cm.joint_inertia(name)) < 1e-9
        gs, gc = cm.gravity_coefficients(name)
        assert abs(j["g_s"] - gs) < 1e-8 and abs(j["g_c"] - gc) < 1e-8
        assert j["tau_max"] <= cm.joint_model(name)["tau_rated"]
        lo, hi = cm.joint_model(name)["limits"]
        assert lo <= j["rom"][0] < j["rom"][1] <= hi
        assert tuple(j["mech_limits"]) == (lo, hi)
    hl = cm.hapkit_linear()
    assert abs(H["m_h"] - hl["m_h"]) < 1e-5 and abs(H["k_g"] - hl["k_g"]) < 1e-4
    assert H["force_max"] == hl["force_max"] == 2.0 and H["x_range"] == hl["x_range"] == 0.040
    assert 0 < H["x_max"] < H["x_range"]


def test_same_joint_as_lab8():
    assert (E["M"], E["b"], E["g_s"], E["tau_max"], tuple(E["rom"])) == (0.0778, 0.1, 3.3354, 6.0, (0.2, 1.6))


def test_provided_baselines():
    assert lab.saturate_toward_limit(1.7, 0.5, 0.2, 1.6) == 0.0
    assert lab.saturate_toward_limit(1.7, -0.5, 0.2, 1.6) == -0.5
    assert lab.saturate_toward_limit(1.0, 0.5, 0.2, 1.6) == 0.5
    assert lab.virtual_wall_force(0.030, 0.0, 0.025, K=200.0) == pytest.approx(-1.0)
    assert lab.virtual_wall_force(0.020, 0.1, 0.025, K=200.0) == 0.0


def test_provided_solve_qp():
    z, info = lab.solve_qp(2 * np.eye(2), np.array([-2.0, -4.0]), np.array([[1.0, 1.0]]), np.array([1.0]))
    assert info["success"] and np.allclose(z, [0.0, 1.0], atol=1e-6)


def test_starter_does_not_import_solution():
    src = Path(lab.__file__).read_text()
    assert "import solution" not in src and "from solution" not in src


# ------------------------------------------------------------------------------ Part A
def test_cbf_filter_1d_closed_form_and_grid():
    for _ in range(40):
        u_nom, h, Lf, Lg, al = RNG.normal(0, 2), abs(RNG.normal(0, 1)), RNG.normal(0, 2), RNG.normal(0, 2), 2.0
        if abs(Lg) < 0.05:
            continue
        u = lab.cbf_filter_1d(u_nom, h, Lf, Lg, al)
        assert Lf + Lg * u + al * h >= -1e-9
        grid = np.linspace(u_nom - 50, u_nom + 50, 200001)
        ok = grid[Lf + Lg * grid + al * h >= 0]
        assert abs(u - ok[np.argmin(np.abs(ok - u_nom))]) < 1e-3
    assert lab.cbf_filter_1d(1.0, 0.1, 5.0, 0.0, 2.0) == 1.0           # Lg = 0: the input cannot act
    assert lab.cbf_filter_1d(0.1, 0.4, 0.0, -1.0, 2.0) == pytest.approx(0.1)   # already safe: unchanged


def test_cbf_filter_1d_workshop_rail():
    # Workshop Problem 1: ankle trainer theta' = u, h = 0.35 - theta, alpha = 4, u_nom = 0.5
    assert lab.cbf_filter_1d(0.5, 0.35 - 0.30, 0.0, -1.0, 4.0) == pytest.approx(0.2)
    assert lab.cbf_filter_1d(0.5, 0.35 - 0.20, 0.0, -1.0, 4.0) == pytest.approx(0.5)


def test_filter_interval():
    assert lab.filter_interval(3.0, -1.0, 2.0) == (2.0, True, True)
    u, act, feas = lab.filter_interval(0.5, -1.0, 2.0)
    assert u == 0.5 and not act and feas
    u, act, feas = lab.filter_interval(0.0, 1.0, 0.5)
    assert u == 0.5 and not feas
    u, act, feas = lab.filter_interval(-3.0, -np.inf, np.inf)
    assert u == -3.0 and feas


def test_cbf_qp_matches_kkt_enumeration():
    for _ in range(15):
        u_nom = RNG.normal(0, 1, 2)
        A = RNG.normal(0, 1, (2, 2))
        b = RNG.normal(0, 1, 2) + A @ u_nom * 0.0 + 0.5
        L = RNG.normal(0, 1, (2, 2))
        W = L @ L.T + 0.5 * np.eye(2)
        ref = kkt_qp(u_nom, A, b, W)
        if ref is None:
            continue
        u = np.asarray(lab.cbf_qp(u_nom, A, b, W=W)).ravel()
        assert np.allclose(u, ref, atol=1e-5)


def test_cbf_qp_one_input_equals_closed_form():
    for u_nom, h, Lf, Lg in ((0.5, 0.05, 0.0, -1.0), (2.0, 0.3, -1.0, 0.7), (-1.0, 0.2, 0.5, 2.0)):
        u = float(np.asarray(lab.cbf_qp([u_nom], [[Lg]], [-Lf - 4.0 * h])).ravel()[0])
        assert u == pytest.approx(lab.cbf_filter_1d(u_nom, h, Lf, Lg, 4.0), abs=1e-6)


def test_cbf_qp_with_input_bounds():
    u = np.asarray(lab.cbf_qp([1.0, 1.0], [[-1.0, -1.0]], [-0.5], u_min=[0.4, -1.0], u_max=[1.0, 1.0])).ravel()
    assert np.allclose(u, [0.4, 0.1], atol=1e-6)


# ------------------------------------------------------------------------------ Part B
def joint_acc(j, th, om, tau, tau_h):
    return (tau + tau_h - j["b"] * om - lab.gravity(th, j)) / j["M"]


@pytest.mark.parametrize("j", [E, KN], ids=["elbow", "knee"])
def test_rom_bounds_make_second_order_condition_tight(j):
    lo, hi = j["rom"]
    for _ in range(20):
        th, om, th_h = RNG.uniform(lo, hi), RNG.normal(0, 1), RNG.normal(0, 1)
        a1, a2 = RNG.uniform(2, 15), RNG.uniform(2, 15)
        t_lo, t_hi = lab.rom_torque_bounds(th, om, j, a1, a2, tau_h=th_h)
        acc = joint_acc(j, th, om, t_hi, th_h)          # upper barrier h = hi - th: h'' = -acc, h' = -om
        assert -acc + (a1 + a2) * (-om) + a1 * a2 * (hi - th) == pytest.approx(0.0, abs=1e-7)
        acc = joint_acc(j, th, om, t_lo, th_h)          # lower barrier h = th - lo
        assert acc + (a1 + a2) * om + a1 * a2 * (th - lo) == pytest.approx(0.0, abs=1e-7)
        assert t_lo < t_hi


def test_rom_bounds_workshop_knee():
    # Workshop Problem 2: knee of load_exo_2r at theta = 0.5 rad, omega = 0.4 rad/s, alpha1 = alpha2 = 5
    _, t_hi = lab.rom_torque_bounds(0.5, 0.4, KN, 5.0, 5.0)
    g = KN["g_s"] * math.sin(0.5) + KN["g_c"] * math.cos(0.5)
    assert t_hi == pytest.approx(KN["M"] * (25 * 0.1 - 10 * 0.4) + KN["b"] * 0.4 + g, abs=1e-9)
    assert t_hi == pytest.approx(-185.88, abs=0.01)


def test_joint_safety_filter():
    tau, info = lab.joint_safety_filter(5.0, (-1.0, 2.0), (-6.0, 6.0))
    assert tau == 2.0 and info["active"] and info["feasible"]
    tau, info = lab.joint_safety_filter(1.0, (-1.0, 2.0), (-6.0, 6.0))
    assert tau == 1.0 and not info["active"] and info["feasible"]
    tau, info = lab.joint_safety_filter(0.0, (-np.inf, -8.0), (-6.0, 6.0))   # needs more braking than available
    assert tau == -6.0 and not info["feasible"]


def test_aan_torque():
    assert lab.aan_torque(0.03, 0.0, 20.0, 0.5, 0.05) == 0.0
    assert lab.aan_torque(0.25, 0.0, 20.0, 0.5, 0.05) == pytest.approx(-4.0)
    assert lab.aan_torque(-0.25, 0.2, 20.0, 0.5, 0.05) == pytest.approx(4.0 - 0.1)


def test_violation_metrics():
    t = np.arange(1, 1001) * 0.001
    s = 1.5 + 0.2 * np.sin(2 * np.pi * t)
    m = lab.violation_metrics(t, s, 0.2, 1.6)
    assert m["max_violation"] == pytest.approx(0.1, abs=1e-5)
    assert m["time_outside"] == pytest.approx(1 / 3, abs=0.003)
    assert m["s_max"] == pytest.approx(1.7, abs=1e-5)
    assert lab.violation_metrics(t, s, -np.inf, 2.0)["max_violation"] == 0.0


# ------------------------------------------------------------------------------ Part C
def test_hapkit_bound_tight_and_robust():
    for _ in range(20):
        x, v, Fh = RNG.uniform(-0.02, 0.025), RNG.normal(0, 0.05), RNG.uniform(0, 1.7)
        F = lab.hapkit_force_bound(x, v, H, 50.0, 50.0, Fh)
        acc = (F + Fh - H["b_h"] * v - H["k_g"] * x) / H["m_h"]   # worst-case operator force F_h = +F_h_max
        assert -acc - 100.0 * v + 2500.0 * (H["x_max"] - x) == pytest.approx(0.0, abs=1e-7)
    F, info = lab.hapkit_safety_filter(0.0, 0.0, 0.0, H, 50.0, 50.0, 1.7)
    assert F == 0.0 and not info["active"]
    F, info = lab.hapkit_safety_filter(0.0, 0.025, 0.0, H, 50.0, 50.0, 3.0)
    assert F == -2.0 and not info["feasible"]


# ------------------------------------------------------------------------------ closed-loop runs
def check_run(key, log, s, lo, hi, tol_s=2e-3):
    m = lab.violation_metrics(log["t"], s, lo, hi)
    ref = EXP[key]
    assert m["max_violation"] == pytest.approx(ref["max_violation"], abs=tol_s)
    return m


def test_elbow_step_saturation_vs_cbf():
    ms = {}
    for mode in ("none", "sat", "cbf"):
        log = lab.simulate_joint(E, "step", mode)
        ms[mode] = check_run(f"elbow_step_{mode}_measured", log, log["theta"], *E["rom"])
        assert np.max(np.abs(log["tau"])) <= E["tau_max"] + 1e-9
    assert ms["none"]["max_violation"] > 0.2 and ms["sat"]["max_violation"] > 0.1
    assert ms["cbf"]["max_violation"] < 1e-4


def test_elbow_spasm_needs_the_measured_or_bounded_patient_torque():
    ms = {}
    for mode, thm in (("sat", "measured"), ("cbf", "measured"), ("cbf", "zero"), ("cbf", "robust")):
        log = lab.simulate_joint(E, "spasm", mode, thm)
        ms[(mode, thm)] = check_run(f"elbow_spasm_{mode}_{thm}", log, log["theta"], *E["rom"])
        if thm == "robust":
            assert np.max(np.abs(log["tau_h"])) <= E["spasm"]["tau_h_max"]
    assert ms[("sat", "measured")]["max_violation"] > 0.05
    assert ms[("cbf", "measured")]["max_violation"] < 1e-4
    assert ms[("cbf", "zero")]["max_violation"] > 0.05
    assert ms[("cbf", "robust")]["max_violation"] < 1e-4
    assert ms[("cbf", "robust")]["s_max"] < ms[("cbf", "measured")]["s_max"] - 0.05   # conservative


def test_knee_load_carriage_with_torque_rate_limit():
    log = lab.simulate_joint(KN, "step", "cbf")
    check_run("knee_step_cbf_measured", log, log["theta"], *KN["rom"])
    assert np.max(np.abs(np.diff(log["tau"]))) <= KN["tau_rate"] * KN["dt"] + 1e-9
    log = lab.simulate_joint(KN, "step", "sat")
    assert lab.violation_metrics(log["t"], log["theta"], *KN["rom"])["max_violation"] > 0.5   # knee buckles


def test_hapkit_forbidden_region():
    out = {}
    for case in ("push", "hard_push"):
        for mode in ("wall", "cbf"):
            log = lab.simulate_hapkit(H, case, mode)
            out[(case, mode)] = (check_run(f"hapkit_{case}_{mode}", log, log["x"], -np.inf, H["x_max"], 2e-4), log)
            assert np.max(np.abs(log["F"])) <= H["force_max"] + 1e-12
    assert out[("push", "cbf")][0]["max_violation"] < 1e-5
    assert out[("push", "wall")][0]["max_violation"] > 0.003
    assert not np.all(out[("hard_push", "cbf")][1]["feasible"])     # the operator out-pushes the 2 N motor
    assert out[("hard_push", "cbf")][0]["max_violation"] > 0.005
