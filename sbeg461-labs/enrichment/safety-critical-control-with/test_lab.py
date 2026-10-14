"""Auto-checks for the SBEG 461 enrichment lab "Safety-critical control with control barrier functions".

    LAB_IMPL=solution python -m pytest -q test_lab.py   # reference: all pass
    LAB_IMPL=starter  python -m pytest -q test_lab.py   # untouched starter: every test that calls your code fails
                                                        # with NotImplementedError; the 3 model/data tests pass

LAB_IMPL is the name of the module to test (default "solution"); it is imported from this folder.
Ground truth: modern_robotics (MassMatrix, VelQuadraticForces, FKinSpace, JacobianSpace, ForwardDynamics),
finite differences of the barrier along exact double-integrator motion, an independent active-set enumeration
of the small QP, the Week 13 numbers (apparent mass 1.063 kg at the home pose, 0.204 m/s and 0.300 m/s speed
limits) and closed-loop properties that any correct implementation must have (forward invariance, speed cap).
"""
import functools
import importlib
import itertools
import json
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
import course_models as cm  # noqa: E402

RH = cm.rehab_2r()
DATA = json.loads((HERE / "data" / "scenario.json").read_text())


@functools.lru_cache(maxsize=None)
def run(name: str, filter_on: bool = True, alpha1=None, v_max=None, model_scale: float = 1.0):
    scn = lab.load_scenario(name)
    if alpha1 is not None:
        scn["alpha1"] = alpha1
    if v_max is not None:
        scn["v_max"] = v_max
    return scn, lab.simulate(scn, filter_on=filter_on, model_scale=model_scale)


# ============================================================================ model and data (provided code)
def test_model_matches_modern_robotics():
    rng = np.random.default_rng(3)
    for _ in range(6):
        q, qd = rng.uniform([0.2, 0.3], [2.9, 2.7]), rng.normal(size=2)
        Mq, c = lab.dynamics(q, qd)
        assert np.allclose(Mq, mr.MassMatrix(q, RH["Mlist"], RH["Glist"], RH["Slist"]), atol=1e-12)
        assert np.allclose(c, mr.VelQuadraticForces(q, qd, RH["Mlist"], RH["Glist"], RH["Slist"]), atol=1e-12)
        assert np.allclose(mr.GravityForces(q, RH["g"], RH["Mlist"], RH["Glist"], RH["Slist"]), 0, atol=1e-12)
        T = mr.FKinSpace(RH["M"], RH["Slist"], q)
        assert np.allclose(lab.fk(q), T[:2, 3], atol=1e-12)
        Js = mr.JacobianSpace(RH["Slist"], q)                    # p' = v_s + w_s x p
        Jp = Js[3:, :] - mr.VecToso3(T[:3, 3]) @ Js[:3, :]
        assert np.allclose(lab.jacobian(q), Jp[:2, :], atol=1e-12)
        h = 1e-6                                                 # J' q' by a central difference along q'
        Jdot = (lab.jacobian(q + h * qd) - lab.jacobian(q - h * qd)) / (2 * h)
        assert np.allclose(lab.jdot_qdot(q, qd), Jdot @ qd, atol=1e-7)
    # Week 13 WE12 numbers at the home pose
    assert np.allclose(lab.dynamics(RH["q_home"], [0, 0])[0], [[0.3545, 0.0628], [0.0628, 0.0657]], atol=1e-4)


def test_scenario_file_and_start_pose():
    for name in ("static", "approaching"):
        scn = lab.load_scenario(name)
        for key in ("dt", "T_move", "t_end", "p_A", "p_B", "Kp", "Kd", "r_keepout", "alpha1", "alpha2", "gamma",
                    "v_max", "m_H", "k_contact", "F_max", "person_p0", "person_p1", "person_t0", "person_t1"):
            assert key in scn, key
        assert scn["Kd"] ** 2 == pytest.approx(4 * scn["Kp"])     # critically damped error dynamics
        for p in (scn["p_A"], scn["p_B"]):
            q = lab.ik(p)
            assert np.allclose(lab.fk(q), p, atol=1e-12)
            assert np.all(q >= RH["limits"][:, 0]) and np.all(q <= RH["limits"][:, 1])
    assert DATA["common"]["v_max"] == 0.20                         # the Week 13 value (H3) to be re-checked


def test_person_and_reference_profiles():
    scn = lab.load_scenario("approaching")
    p0, v0, a0 = lab.person_state(0.0, scn)
    p1, v1, a1 = lab.person_state(5.0, scn)
    assert np.allclose(p0, scn["person_p0"]) and np.allclose(p1, scn["person_p1"])
    assert np.allclose(v0, 0) and np.allclose(v1, 0) and np.allclose(a0, 0)
    t, h = 0.83, 1e-5
    assert np.allclose((lab.person_state(t + h, scn)[0] - lab.person_state(t - h, scn)[0]) / (2 * h),
                       lab.person_state(t, scn)[1], atol=1e-6)
    assert np.allclose((lab.reference(t + h, scn)[1] - lab.reference(t - h, scn)[1]) / (2 * h),
                       lab.reference(t, scn)[2], atol=1e-5)


# ============================================================================ Part A -- barrier and filter
def test_barrier_terms_values_and_second_derivative():
    p, v = np.array([0.10, 0.50]), np.array([-0.20, 0.05])
    p_o, v_o = np.array([0.02, 0.47]), np.array([0.03, -0.01])
    d, dd, n, kappa = lab.barrier_terms(p, v, p_o, v_o, 0.06)
    e, w = p - p_o, v - v_o
    assert d == pytest.approx(np.hypot(*e) - 0.06)
    assert np.allclose(n, e / np.hypot(*e)) and dd == pytest.approx(n @ w)
    assert kappa >= 0
    # exact motion with constant accelerations u (handle) and a_o (person): d'' by finite differences
    u, a_o, h = np.array([0.7, -0.4]), np.array([0.1, 0.2]), 1e-4

    def dist(s):
        return np.hypot(*(p + v * s + 0.5 * u * s * s - (p_o + v_o * s + 0.5 * a_o * s * s))) - 0.06
    ddd_fd = (dist(h) - 2 * dist(0.0) + dist(-h)) / h**2
    assert ddd_fd == pytest.approx(n @ (u - a_o) + kappa, rel=1e-5)
    assert (dist(h) - dist(-h)) / (2 * h) == pytest.approx(dd, rel=1e-7)


def test_cbf_bound_definitions():
    scn = lab.load_scenario("static")
    p, v = np.array([0.12, 0.52]), np.array([-0.25, -0.02])
    p_o, v_o, a_o = np.array([0.01, 0.48]), np.zeros(2), np.zeros(2)
    n, b, b1, b2 = lab.cbf_bound(p, v, p_o, v_o, a_o, scn)
    d, dd, n2, kappa = lab.barrier_terms(p, v, p_o, v_o, scn["r_keepout"])
    a1, a2, g = scn["alpha1"], scn["alpha2"], scn["gamma"]
    assert np.allclose(n, n2)
    assert b1 == pytest.approx(-kappa - (a1 + a2) * dd - a1 * a2 * d)
    assert b2 == pytest.approx(-kappa - g * (scn["v_max"] + dd))
    assert b == pytest.approx(max(b1, b2))
    # on the boundary n.u = b1 the second-order condition holds with equality: psi1' + a2 psi1 = 0
    u = b1 * n
    ddd = n @ u + kappa
    assert (ddd + a1 * dd) + a2 * (dd + a1 * d) == pytest.approx(0.0, abs=1e-12)
    # a moving person adds n.a_o to both bounds
    a_o2 = np.array([0.3, -0.5])
    _, _, b1m, b2m = lab.cbf_bound(p, v, p_o, v_o, a_o2, scn)
    assert b1m - b1 == pytest.approx(n @ a_o2) and b2m - b2 == pytest.approx(n @ a_o2)


def test_safety_filter_closed_form():
    a = np.array([0.6, 0.8])
    u_nom = np.array([1.0, -2.0])
    u, lam = lab.safety_filter(u_nom, a, -5.0)                 # a.u_nom = -1 >= -5: inactive
    assert lam == 0.0 and np.allclose(u, u_nom)
    u, lam = lab.safety_filter(u_nom, a, 0.5)                  # active: projection onto a.u = 0.5
    assert a @ u == pytest.approx(0.5) and lam == pytest.approx(1.5)
    du = u - u_nom
    assert abs(du[0] * a[1] - du[1] * a[0]) < 1e-12                 # correction along a (KKT)
    rng = np.random.default_rng(7)                             # optimality against random feasible points
    for _ in range(200):
        x = rng.normal(size=2) * 3
        if a @ x >= 0.5:
            assert np.linalg.norm(x - u_nom) >= np.linalg.norm(u - u_nom) - 1e-12
    u, lam = lab.safety_filter(u_nom, 2 * a, 1.0)              # a non-unit normal gives the same u
    assert np.allclose(u, lab.safety_filter(u_nom, a, 0.5)[0])


def _qp_enumerate(u_nom, A, b):
    """Independent solution of min |u - u_nom|^2 s.t. A u >= b (2-D, few rows) by enumerating active sets."""
    best, best_cost = None, np.inf
    m = len(b)
    for k in range(0, 3):
        for S in itertools.combinations(range(m), k):
            if k == 0:
                u = u_nom.copy()
            else:
                AS, bS = A[list(S)], b[list(S)]
                G = AS @ AS.T
                if abs(np.linalg.det(G)) < 1e-12:
                    continue
                lam = np.linalg.solve(G, bS - AS @ u_nom)
                if np.any(lam < -1e-12):
                    continue
                u = u_nom + AS.T @ lam
            if np.all(A @ u >= b - 1e-10) and np.sum((u - u_nom) ** 2) < best_cost:
                best, best_cost = u, np.sum((u - u_nom) ** 2)
    return best


def test_qp_with_scipy():
    rng = np.random.default_rng(11)
    for _ in range(10):                                        # one constraint: equals the closed form
        a, u_nom, b = rng.normal(size=2), rng.normal(size=2) * 2, rng.normal()
        assert np.allclose(lab.safety_filter_qp(u_nom, a[None, :], [b]), lab.safety_filter(u_nom, a, b)[0],
                           atol=1e-6)
    for _ in range(10):                                        # two or three constraints: active-set enumeration
        m = rng.integers(2, 4)
        A = rng.normal(size=(m, 2))
        u_nom = rng.normal(size=2) * 2
        x_feas = rng.normal(size=2)
        b = A @ x_feas - rng.uniform(0, 1, size=m)             # feasible by construction
        assert np.allclose(lab.safety_filter_qp(u_nom, A, b), _qp_enumerate(u_nom, A, b), atol=1e-6)


# ============================================================================ Part B -- the controller
def test_nominal_accel():
    scn = lab.load_scenario("static")
    t = 0.7
    pd, vd, ad = lab.reference(t, scn)
    assert np.allclose(lab.nominal_accel(t, pd, vd, scn), ad)
    dp, dv = np.array([0.01, -0.02]), np.array([0.05, 0.03])
    expect = ad + scn["Kd"] * (-dv) + scn["Kp"] * (-dp)
    assert np.allclose(lab.nominal_accel(t, pd + dp, vd + dv, scn), expect)


def test_computed_torque_gives_the_commanded_handle_acceleration():
    rng = np.random.default_rng(5)
    for _ in range(5):
        q, qd, u = rng.uniform([0.3, 0.4], [2.5, 2.4]), rng.normal(size=2), rng.normal(size=2)
        tau = lab.computed_torque(q, qd, u)
        qdd = mr.ForwardDynamics(q, qd, tau, RH["g"], np.zeros(6), RH["Mlist"], RH["Glist"], RH["Slist"])
        assert np.allclose(lab.jacobian(q) @ qdd + lab.jdot_qdot(q, qd), u, atol=1e-9)
        assert np.allclose(lab.computed_torque(q, qd, u, model_scale=1.2), 1.2 * tau)


def test_nominal_run_tracks_but_enters_the_keep_out_disc():
    for name in ("static", "approaching"):
        scn, log = run(name, filter_on=False)
        m = lab.metrics(log, scn)
        assert m["max_dev"] < 5e-5 and m["final_error"] < 5e-5  # exact model: p'' = u_nom (only the 1 ms hold)
        assert m["min_d"] < -0.01                                # the unfiltered handle enters the disc
        assert m["max_approach"] > 0.3 and not log["active"].any()
    scn, log = run("static", filter_on=False)
    assert lab.metrics(log, scn)["min_d"] == pytest.approx(-0.0285, abs=5e-4)  # path passes 31.5 mm from the centre


def test_filter_keeps_out_and_caps_the_approach_speed():
    for name in ("static", "approaching"):
        scn, log = run(name)
        m = lab.metrics(log, scn)
        assert m["min_d"] >= 0.0                                  # never inside the keep-out disc
        assert m["max_approach"] <= scn["v_max"] + 1e-3           # speed cap (1 kHz sampling tolerance)
        assert 0.1 < m["active_fraction"] < 0.5                   # the filter acts only near the hand
        assert m["final_error"] < 2e-3                            # still completes the movement
        assert m["max_tau_ratio"] < 1.0                           # within the rated torques
        q = log["q"]
        assert np.all(q >= RH["limits"][:, 0]) and np.all(q <= RH["limits"][:, 1])


def test_forward_invariance_of_the_barriers():
    for name in ("static", "approaching"):
        scn, log = run(name)
        psi1 = -log["approach"] + scn["alpha1"] * log["d"]       # psi1 = d' + a1 d
        h_v = scn["v_max"] - log["approach"]
        assert psi1.min() >= -1e-3 and h_v.min() >= -1e-3
        # where the filter is inactive the command is the nominal one; where active it differs
        same = np.all(np.isclose(log["u"], log["u_nom"]), axis=1)
        assert np.array_equal(same, ~log["active"])


def test_keep_out_only_reaches_the_boundary_without_crossing():
    scn, log = run("static", alpha1=10.0, v_max=1e3)
    m = lab.metrics(log, scn)
    assert 0.0 <= m["min_d"] < 0.006
    assert m["max_approach"] > 0.25                              # no speed cap in this run


def test_qp_and_closed_form_runs_agree():
    scn = dict(lab.load_scenario("static"), t_end=1.6)
    a = lab.simulate(scn)
    b = lab.simulate(scn, use_qp=True)
    assert np.allclose(a["p"], b["p"], atol=1e-6)


def test_model_error_keeps_the_handle_out_of_the_disc():
    for s in (0.7, 1.3):
        scn, log = run("static", model_scale=s)
        assert lab.metrics(log, scn)["min_d"] > 0.0


# ============================================================================ Part C -- force limits near a person
def test_apparent_mass_and_pfl_speed():
    assert lab.apparent_mass(RH["q_home"], [1.0, 0.0]) == pytest.approx(1.063, abs=5e-4)    # Week 13 WE12
    q, n = np.array([0.7, 1.3]), np.array([0.6, -0.8])
    J = lab.jacobian(q)
    Mmr = mr.MassMatrix(q, RH["Mlist"], RH["Glist"], RH["Slist"])
    Lam = np.linalg.inv(J @ np.linalg.inv(Mmr) @ J.T)
    lo, hi = np.linalg.eigvalsh(Lam)
    m = lab.apparent_mass(q, n)
    assert lo - 1e-12 <= m <= hi + 1e-12
    assert m == pytest.approx(1.0 / (n @ np.linalg.inv(Lam) @ n))
    assert lab.pfl_speed_limit(6.0, 10.0, 50_000.0, 130.0) == pytest.approx(0.300, abs=5e-4)  # Week 13 WE6
    assert lab.pfl_speed_limit(1.0633, 2.0, 5000.0, 12.0) == pytest.approx(0.204, abs=5e-4)  # Week 13 WE12


def test_design_speed_limit():
    v, m_max, v_exact = lab.design_speed_limit(lab.load_scenario("static"))
    assert m_max == pytest.approx(2.476, abs=2e-3)
    assert v_exact == pytest.approx(0.1613, abs=2e-4)
    assert v == pytest.approx(0.16)


def test_week13_speed_is_not_conservative_but_the_designed_one_is():
    scn, log = run("approaching")                                # v_max = 0.20 from the home pose along x
    assert lab.peak_contact_force(log, scn).max() > scn["F_max"]
    v = lab.design_speed_limit(scn)[0]
    for name in ("static", "approaching"):
        scn, log = run(name, v_max=v)
        assert lab.metrics(log, scn)["min_d"] >= 0.0
        assert lab.peak_contact_force(log, scn).max() <= scn["F_max"]
