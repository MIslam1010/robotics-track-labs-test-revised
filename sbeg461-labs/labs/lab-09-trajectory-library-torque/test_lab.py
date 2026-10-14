"""Auto-checks for SBEG 461 Lab 9 (trajectory library, torque feasibility, time-optimal time scaling).

Usage:   LAB_IMPL=starter  python -m pytest -q test_lab.py     (your work, file starter.py)
         LAB_IMPL=solution python -m pytest -q test_lab.py     (reference; default)
Ground truth: modern_robotics (CubicTimeScaling, QuinticTimeScaling, JointTrajectory, InverseDynamics,
InverseDynamicsTrajectory), the numbers printed in MR Secs. 9.2-9.3 (Figs. 9.3, 9.4, 9.7; Eqs. 9.16-9.29),
scipy.interpolate.BSpline, a brute-force bisection on the trajectory duration, the closed-form MR 2R dynamics
(MR Eqs. 8.9-8.10), closed-form phase-plane results (elbow exoskeleton, demining gantry, MR 2R arm) and toppra.
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
lab = importlib.import_module(os.environ.get("LAB_IMPL", "starter"))

RNG = np.random.default_rng(461)
PI = math.pi
TAU_2R = np.array([20.0, 10.0])
F_GANTRY = np.array([120.0, 60.0])


@pytest.fixture(scope="module")
def arm():
    return lab.load_course_arm(payload=True, derated=True)


@pytest.fixture(scope="module")
def tasks():
    return lab.load_tasks()


@pytest.fixture(scope="module")
def two_r():
    return lab.mr_2r_model(), lab.StraightLinePath([0.0, 0.0], [PI / 2, PI / 2])


@pytest.fixture(scope="module")
def gantry():
    return lab.gantry_model(), lab.SemicirclePath(0.3)


@pytest.fixture(scope="module")
def elbow():
    return lab.elbow_exo_model(), lab.StraightLinePath([0.0], [2 * PI / 3])


@pytest.fixture(scope="module")
def arm_task():
    task = lab.load_topp_task()
    r = lab.course_arm_model(task["payload_kg"])
    return task, r, lab.waypoint_path(task["waypoints"])


@pytest.fixture(scope="module")
def hw_topp(arm_task):
    task, r, path = arm_task
    tl = task["torque_fraction"] * r["tau_max"]
    return lab.topp(r, path, tl, qdot_max=task["hw_speed_fraction"] * r["qdot_max"],
                    qddot_max=task["hw_qddot_max"], N=task["grid_N"])


def M2(th):
    c2 = math.cos(th[1])
    return np.array([[3 + 2 * c2, 1 + c2], [1 + c2, 1.0]])


def c2r(th, dth):
    s2 = math.sin(th[1])
    return np.array([-s2 * (2 * dth[0] * dth[1] + dth[1] ** 2), s2 * dth[0] ** 2])


def quintic_samples(path, N=401):
    """Q, Qd, Qdd of the quintic time scaling of `path` with T = 1 s (input of min_time_scale)."""
    traj = lab.quintic_traj(path, 1.0)
    t = np.linspace(0.0, 1.0, N)
    return traj(t, 0), traj(t, 1), traj(t, 2)


# ---------------------------------------------------------------- Part A: point-to-point time scalings

@pytest.mark.parametrize("T", [0.7, 2.0, 5.0])
def test_cubic_matches_modern_robotics_and_mr_fig_9_3(T):
    t = np.linspace(0, T, 41)
    s, sd, sdd = lab.cubic_time_scaling(T, t)
    assert np.allclose(s, [mr.CubicTimeScaling(T, ti) for ti in t], atol=1e-12)
    assert np.isclose(np.max(sd), 3 / (2 * T), atol=1e-9)              # MR Fig. 9.3: peak 3/(2T) at T/2
    assert np.isclose(sdd[0], 6 / T**2) and np.isclose(sdd[-1], -6 / T**2)
    assert np.allclose(np.gradient(s, t)[1:-1], sd[1:-1], atol=0.02 / T)


@pytest.mark.parametrize("T", [0.7, 2.0, 5.0])
def test_quintic_matches_modern_robotics_and_mr_fig_9_4(T):
    t = np.linspace(0, T, 2001)
    s, sd, sdd = lab.quintic_time_scaling(T, t)
    assert np.allclose(s, [mr.QuinticTimeScaling(T, ti) for ti in t], atol=1e-12)
    assert np.isclose(np.max(sd), 15 / (8 * T), rtol=1e-6)              # MR Fig. 9.4
    assert np.isclose(np.max(sdd), 10 / (np.sqrt(3) * T**2), rtol=1e-5)
    assert abs(sd[0]) < 1e-12 and abs(sdd[0]) < 1e-12 and abs(sdd[-1]) < 1e-9
    assert np.allclose(np.gradient(sd, t)[1:-1], sdd[1:-1], atol=0.01 / T**2)


def test_time_scalings_accept_scalars_and_clip():
    for f in (lab.cubic_time_scaling, lab.quintic_time_scaling):
        s, sd, sdd = f(2.0, 1.0)
        assert np.isclose(float(s), 0.5)
        assert np.isclose(float(f(2.0, 3.0)[0]), 1.0) and np.isclose(float(f(2.0, -1.0)[0]), 0.0)


def test_trapezoid_three_stage_mr_eqs_9_16_to_9_24():
    v, a = 0.5, 1.25                                  # v^2/a = 0.2 <= 1: three stages
    T, ta, vp = lab.trapezoid_params(v, a)
    assert np.isclose(T, (a + v**2) / (v * a)) and np.isclose(T, 2.4)
    assert np.isclose(ta, 0.4) and np.isclose(vp, 0.5)
    t = np.linspace(0, T, 4801)
    s, sd, sdd = lab.trapezoid_time_scaling(v, a, t)
    assert np.isclose(s[-1], 1.0) and np.isclose(s[0], 0.0)
    assert np.isclose(np.max(sd), v) and np.isclose(np.max(sdd), a) and np.isclose(np.min(sdd), -a)
    assert np.all(np.diff(s) >= -1e-12)
    assert np.allclose(np.gradient(s, t), sd, atol=2e-3)
    k = np.searchsorted(t, 1.0)                       # cruise phase, MR Eq. 9.21
    assert np.isclose(s[k], v * t[k] - v**2 / (2 * a))


def test_trapezoid_bang_bang_when_v2_over_a_exceeds_1():
    v, a = 2.0, 1.0                                   # v^2/a = 4 > 1 (MR Ex. 9.10, 9.13)
    T, ta, vp = lab.trapezoid_params(v, a)
    assert np.isclose(T, 2.0) and np.isclose(ta, 1.0) and np.isclose(vp, 1.0)
    s, sd, _ = lab.trapezoid_time_scaling(v, a, np.linspace(0, T, 1001))
    assert np.isclose(s[-1], 1.0) and np.isclose(np.max(sd), 1.0)


@pytest.mark.parametrize("method", [3, 5])
def test_joint_trajectory_matches_modern_robotics(method):
    q0, q1 = RNG.uniform(-2, 2, 6), RNG.uniform(-2, 2, 6)
    t, Q, Qd, Qdd = lab.joint_trajectory(q0, q1, 1.7, 35, method)
    assert Q.shape == (35, 6) and Qd.shape == (35, 6) and Qdd.shape == (35, 6)
    assert np.allclose(t, np.linspace(0, 1.7, 35))
    assert np.allclose(Q, np.array(mr.JointTrajectory(q0, q1, 1.7, 35, method)), atol=1e-12)
    dtm = t[1] - t[0]
    assert np.allclose(np.gradient(Q, dtm, axis=0)[2:-2], Qd[2:-2], atol=0.05)


def test_joint_trajectory_trapezoid_respects_limits(tasks):
    q0, q1 = tasks["A"], tasks["B"]
    vlim, alim = np.full(6, 1.3962634), np.array(tasks["qddot_lim"])
    v, a, T = lab.trapezoid_from_limits(q0, q1, vlim, alim)      # provided; MR Eqs. 9.14-9.15
    assert np.isclose(v, 1.1636, atol=1e-4) and np.isclose(a, 3.3333, atol=1e-4)
    assert np.isclose(T, 1.2085, atol=1e-4)
    t, Q, Qd, Qdd = lab.joint_trajectory(q0, q1, None, 501, "trap", v=v, a=a)
    assert np.isclose(t[-1], T) and np.allclose(Q[-1], q1) and np.allclose(Q[0], q0)
    assert np.all(np.abs(Qd) <= vlim + 1e-9) and np.all(np.abs(Qdd) <= alim + 1e-9)
    assert np.isclose(np.max(np.abs(Qd[:, 0])), vlim[0])            # joint 1 saturates (largest move)


# ---------------------------------------------------------------- Part B: via points and B-splines

def test_via_cubic_coefficients_formula_and_continuity():
    beta, bdot, Tm = [0.0, 1.0, 0.5, 2.0], [0.0, 0.8, -0.3, 0.0], [0.0, 1.0, 2.5, 3.0]
    A = lab.via_cubic_coefficients(beta, bdot, Tm)
    assert A.shape == (3, 4)
    for j in range(3):                                       # each segment meets its four constraints
        dT = Tm[j + 1] - Tm[j]
        a0, a1, a2, a3 = A[j]
        assert np.isclose(a0, beta[j]) and np.isclose(a1, bdot[j])
        assert np.isclose(a0 + a1 * dT + a2 * dT**2 + a3 * dT**3, beta[j + 1])
        assert np.isclose(a1 + 2 * a2 * dT + 3 * a3 * dT**2, bdot[j + 1])


def test_via_trajectory_mr_figure_9_7a():
    # MR p. 337: vias (0,0), (0,1), (1,1), (1,0) at T = 0, 1, 2, 3, velocities (0,0), (1,0), (0,-1), (0,0)
    P = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], float)
    V = np.array([[0, 0], [1, 0], [0, -1], [0, 0]], float)
    Tm = [0, 1, 2, 3]
    B, Bd, Bdd = lab.via_trajectory(P, V, Tm, np.array([0, 1, 2, 3]))
    assert np.allclose(B, P) and np.allclose(Bd, V)
    t = np.linspace(0, 3, 3001)
    B, Bd, _ = lab.via_trajectory(P, V, Tm, t)
    assert np.allclose(np.gradient(B, t, axis=0)[1:-1], Bd[1:-1], atol=2e-3)
    assert B[:, 0].min() < -0.05            # x dips below 0 in segment 1 (Fig. 9.7a): not in the convex hull


def test_uniform_cubic_bspline_matches_scipy_and_convex_hull():
    from scipy.interpolate import BSpline
    P = RNG.uniform(-1, 1, (7, 2))
    n = len(P)
    spl = BSpline(np.arange(n + 4, dtype=float), P, 3)      # uniform knots; valid on [3, n]
    for u in np.linspace(0, n - 3, 37):
        p, dp, ddp = lab.uniform_cubic_bspline(P, u)
        assert np.allclose(p, spl(u + 3), atol=1e-12)
        assert np.allclose(dp, spl.derivative(1)(u + 3), atol=1e-10)
        assert np.allclose(ddp, spl.derivative(2)(u + 3), atol=1e-9)
        i = min(int(np.floor(u)), n - 4)                    # local convex hull: inside the box of P_i..P_i+3
        box = P[i:i + 4]
        assert np.all(p >= box.min(axis=0) - 1e-12) and np.all(p <= box.max(axis=0) + 1e-12)


# ---------------------------------------------------------------- Part C: torque feasibility on the course arm

def test_course_arm_data_and_gravity_at_A(arm, tasks):
    assert arm["Slist"].shape == (6, 6) and len(arm["Mlist"]) == 7 and len(arm["Glist"]) == 6
    assert np.allclose(arm["tau_max"], 0.6 * np.array([1.8, 3.0, 1.5, 0.6, 0.6, 0.4]))
    assert np.allclose(arm["qdot_max"], 0.5 * np.deg2rad(160.0))
    r = lab.course_arm_model(0.25)                           # same payload rule as the course robot models
    assert all(np.allclose(a, b, atol=1e-15) for a, b in zip(r["Glist"], arm["Glist"]))
    assert np.allclose(r["tau_max"], [1.8, 3.0, 1.5, 0.6, 0.6, 0.4])
    G = lab.torque_profile(arm, tasks["A"][None, :], np.zeros((1, 6)), np.zeros((1, 6)))[0]
    assert np.allclose(G, [0.0, -1.140, -0.570, -0.242, 0.0, 0.0], atol=1e-3)     # checkpoint of Part C
    assert np.allclose(G, mr.InverseDynamics(tasks["A"], np.zeros(6), np.zeros(6), arm["g"], np.zeros(6),
                                             arm["Mlist"], arm["Glist"], arm["Slist"]), atol=1e-12)


def test_torque_profile_matches_modern_robotics(arm, tasks):
    t, Q, Qd, Qdd = lab.joint_trajectory(tasks["A"], tasks["B"], 1.8, 25, 5)
    Tau = lab.torque_profile(arm, Q, Qd, Qdd)
    ref = np.array(mr.InverseDynamicsTrajectory(Q, Qd, Qdd, arm["g"], np.zeros((25, 6)),
                                                arm["Mlist"], arm["Glist"], arm["Slist"]))
    assert Tau.shape == (25, 6) and np.allclose(Tau, ref, atol=1e-10)


def _bisect_min_T(arm, q0, q1, method, N=201):
    def ok(T):
        _, Q, Qd, Qdd = lab.joint_trajectory(q0, q1, T, N, method)
        Tau = np.array(mr.InverseDynamicsTrajectory(Q, Qd, Qdd, arm["g"], np.zeros((N, 6)),
                                                    arm["Mlist"], arm["Glist"], arm["Slist"]))
        return np.all(np.abs(Qd) <= arm["qdot_max"] + 1e-12) and np.all(np.abs(Tau) <= arm["tau_max"] + 1e-12)
    lo, hi = 0.01, 10.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi


@pytest.mark.parametrize("goal, method, binding", [("B", 5, "velocity"), ("D", 5, "torque"), ("D", 3, "torque")])
def test_min_time_scale_equals_bisection(arm, tasks, goal, method, binding):
    _, Q, Qd, Qdd = lab.joint_trajectory(tasks["A"], tasks[goal], 1.0, 201, method)
    res = lab.min_time_scale(arm, Q, Qd, Qdd)
    assert np.isclose(res["c"], max(res["c_vel"], res["c_tau"])) and res["c_acc"] == 0.0
    assert res["binding"][0] == binding
    assert np.isclose(res["c"], _bisect_min_T(arm, tasks["A"], tasks[goal], method), rtol=1e-6)


def test_min_time_scale_checkpoint_values(arm, tasks):
    _, Q, Qd, Qdd = lab.joint_trajectory(tasks["A"], tasks["B"], 1.0, 401, 5)
    r = lab.min_time_scale(arm, Q, Qd, Qdd)
    assert np.isclose(r["c_vel"], 1.6114, atol=1e-3) and r["binding"] == ("velocity", 0)
    _, Q, Qd, Qdd = lab.joint_trajectory(tasks["A"], tasks["D"], 1.0, 401, 5)
    r = lab.min_time_scale(arm, Q, Qd, Qdd)
    assert np.isclose(r["c_tau"], 0.2348, atol=1e-3) and r["binding"] == ("torque", 3)   # joint 4!
    # with |qdd| <= 4 rad/s^2: c_acc = sqrt(0.1 * (10/sqrt(3)) / 4) (quintic peak sdd = 10/sqrt(3), MR Fig. 9.4)
    r = lab.min_time_scale(arm, Q, Qd, Qdd, qddot_max=np.array(tasks["qddot_lim"]))
    assert np.isclose(r["c_acc"], math.sqrt(0.1 * 10 / math.sqrt(3) / 4.0), rtol=1e-4)
    assert r["binding"] == ("acceleration", 1) and np.isclose(r["c"], r["c_acc"])


def test_min_time_scale_rejects_static_overload(arm, tasks):
    heavy = dict(arm, tau_max=0.1 * arm["tau_max"])           # cannot even hold the arm against gravity
    _, Q, Qd, Qdd = lab.joint_trajectory(tasks["A"], tasks["B"], 1.0, 21, 5)
    with pytest.raises(ValueError):
        lab.min_time_scale(heavy, Q, Qd, Qdd)


def test_via_trajectory_on_course_arm_is_feasible_and_stretch_works(arm, tasks):
    P = np.array([tasks["A"], tasks["W"], tasks["B"]])
    Tm = tasks["via_times"]
    V = lab.heuristic_via_velocities(P, Tm)
    t = np.linspace(Tm[0], Tm[-1], 481)
    B, Bd, Bdd = lab.via_trajectory(P, V, Tm, t)
    res = lab.min_time_scale(arm, B, Bd, Bdd)
    assert res["c"] < 1.0                                     # the given timing is feasible
    c = res["c"]                                              # stretched trajectory touches a limit
    Tau = lab.torque_profile(arm, B, Bd / c, Bdd / c**2)
    out = lab.check_limits(arm, Bd / c, Tau)
    assert out["ok"] and np.isclose(max(out["vel_ratio"].max(), out["tau_ratio"].max()), 1.0, atol=1e-9)


# ---------------------------------------------------------------- Part D: time-optimal time scaling (MR Sec. 9.4)

def test_path_dynamics_2r_closed_form(two_r):
    """Straight line: theta'' = 0, so m = M(theta) Delta and c = c(theta, Delta) (MR Eqs. 9.32, 8.9-8.10)."""
    model, path = two_r
    D = np.array([PI / 2, PI / 2])
    for s in (0.2, 0.5, 0.9):
        m, c, g = lab.path_dynamics(model, path, s)
        th = s * D
        assert np.allclose(m, M2(th) @ D, atol=1e-9)
        assert np.allclose(c, c2r(th, D), atol=1e-9)
        assert np.allclose(g, 0.0, atol=1e-12)


def test_path_dynamics_reproduces_rnea_on_course_arm(arm_task):
    """m sdd + c sdot^2 + g = ID(theta, theta' sdot, theta' sdd + theta'' sdot^2) for any (sdot, sdd)."""
    _, r, path = arm_task
    rng = np.random.default_rng(6)
    for s in (0.13, 0.5, 0.77):
        m, c, g = lab.path_dynamics(r, path, s)
        for _ in range(3):
            sd, sdd = rng.uniform(0, 2), rng.uniform(-5, 5)
            th, d1, d2 = path(s, 0), path(s, 1), path(s, 2)
            tau = mr.InverseDynamics(th, d1 * sd, d1 * sdd + d2 * sd**2, r["g"], np.zeros(6),
                                     r["Mlist"], r["Glist"], r["Slist"])
            assert np.allclose(m * sdd + c * sd**2 + g, tau, atol=1e-10)


def test_accel_limits_cone_example():
    """Motion-cone example of Quiz 8, Q3: m = (2, -0.5), c = (0.4, 0.3), g = (1, 0), |tau| <= (5, 2), sdot = 2
    -> L = -1.6, U = 1.2; sdot = 3 is inadmissible (sdot_lim^2 = 7.5)."""
    m, c, g, t = [2.0, -0.5], [0.4, 0.3], [1.0, 0.0], np.array([5.0, 2.0])
    L, U = lab.accel_limits(m, c, g, 2.0, -t, t)
    assert L == pytest.approx(-1.6) and U == pytest.approx(1.2)
    L, U = lab.accel_limits(m, c, g, 3.0, -t, t)
    assert L > U
    L, U = lab.accel_limits(m, c, g, math.sqrt(7.5) - 1e-6, -t, t)
    assert L <= U


def test_accel_limits_zero_inertia():
    """A zero component of m(s) gives a pure velocity constraint (MR Sec. 9.4.4)."""
    m, c, g, t = [0.0, 2.0], [1.0, 0.0], [0.0, 0.0], np.array([1.0, 1.0])
    L, U = lab.accel_limits(m, c, g, 0.9, -t, t)
    assert L == pytest.approx(-0.5) and U == pytest.approx(0.5)
    L, U = lab.accel_limits(m, c, g, 1.1, -t, t)
    assert L > U


def test_velocity_limit_2r_closed_form(two_r):
    """MR 2R arm at s = 0.5: the VLC is where the lower bound of joint 1 meets the upper bound of joint 2:
    (-20 - c1 x)/m1 = (10 - c2 x)/m2, x = sdot^2."""
    model, path = two_r
    m, c, _ = lab.path_dynamics(model, path, 0.5)
    x = (10 / m[1] + 20 / m[0]) / (c[1] / m[1] - c[0] / m[0])
    assert lab.velocity_limit(model, path, 0.5, -TAU_2R, TAU_2R) == pytest.approx(math.sqrt(x), abs=1e-6)
    assert math.sqrt(x) == pytest.approx(2.1546, abs=1e-4)


def test_velocity_limit_gantry_closed_form(gantry):
    """Demining gantry on the MR Fig. 9.9 semicircle: closed-form VLC values at s = 0, 0.25, 0.5."""
    model, path = gantry
    R, Mx, my, fx, fy = 0.3, 40.0, 15.0, 120.0, 60.0
    a = math.sqrt(0.5)
    A1, A2 = fx / (Mx * PI * R * a), fy / (my * PI * R * a)
    vl = lambda s: lab.velocity_limit(model, path, s, -F_GANTRY, F_GANTRY)
    assert vl(0.25) == pytest.approx(math.sqrt((A1 + A2) / (2 * PI)), abs=1e-6)
    assert vl(0.5) == pytest.approx(math.sqrt(fy / (my * PI**2 * R)), abs=1e-6)
    assert vl(0.0) == pytest.approx(math.sqrt(fx / (Mx * PI**2 * R)), abs=1e-6)


def test_velocity_limit_kinematic_and_course_arm(two_r, arm_task):
    model, path = two_r
    big = np.array([1e6, 1e6])
    v = lab.velocity_limit(model, path, 0.3, -big, big, qdot_max=[2.0, 2.0])
    assert v == pytest.approx(2 / (PI / 2), abs=1e-6)
    task, r, apath = arm_task                                  # checkpoint D1: s = 0.5, the middle waypoint
    tl = task["torque_fraction"] * r["tau_max"]
    assert lab.velocity_limit(r, apath, 0.5, -tl, tl) == pytest.approx(5.59, abs=0.01)
    vmax = task["hw_speed_fraction"] * r["qdot_max"]
    assert lab.velocity_limit(r, apath, 0.5, -tl, tl, qdot_max=vmax) == pytest.approx(0.931, abs=1e-3)


def test_topp_2r_closed_form(two_r):
    """MR 2R arm, |tau| <= (20, 10): the single-switch (bang-bang) solution of Lecture B takes 1.368 s."""
    model, path = two_r
    tp = lab.topp(model, path, TAU_2R, N=400)
    assert tp["T"] == pytest.approx(1.3683, rel=0.01)
    assert len(tp["s_grid"]) == 401 and len(tp["sdot_grid"]) == 401
    assert tp["sdot_grid"][0] == pytest.approx(0.0, abs=1e-6) and tp["sdot_grid"][-1] == pytest.approx(0.0, abs=1e-6)


def test_topp_elbow_closed_form(elbow):
    """Elbow exoskeleton, |tau| <= 10 N m: closed-form bang-bang time 0.3885 s; with |qdot| <= 3 rad/s 0.7466 s."""
    model, path = elbow
    assert lab.topp(model, path, [10.0], N=400)["T"] == pytest.approx(0.388528, rel=2e-3)
    assert lab.topp(model, path, [10.0], qdot_max=[3.0], N=400)["T"] == pytest.approx(0.7466, abs=1e-3)


def test_topp_gantry(gantry):
    model, path = gantry
    tp = lab.topp(model, path, F_GANTRY, N=400)
    assert tp["T"] == pytest.approx(1.1655, abs=3e-3)
    sd = np.interp(0.5, tp["s_grid"], tp["sdot_grid"])
    assert sd <= lab.velocity_limit(model, path, 0.5, -F_GANTRY, F_GANTRY) + 1e-3


def test_topp_torque_scaling(two_r):
    """g = 0 and pure torque limits: scaling the limits by k scales T by 1/sqrt(k)."""
    model, path = two_r
    T1 = lab.topp(model, path, TAU_2R, N=300)["T"]
    T2 = lab.topp(model, path, 0.6 * TAU_2R, N=300)["T"]
    assert T2 == pytest.approx(T1 / math.sqrt(0.6), rel=5e-3)


def test_topp_course_arm_respects_limits(arm_task, hw_topp):
    task, r, _ = arm_task
    tl = task["torque_fraction"] * r["tau_max"]
    vmax = task["hw_speed_fraction"] * r["qdot_max"]
    pu = lab.peak_usage(r, hw_topp["traj"], hw_topp["T"], tl, vmax, N=2000)
    assert np.all(pu["vel_ratio"] <= 1.01)
    assert np.all(pu["tau_ratio"] <= 1.03)
    t = np.linspace(0, hw_topp["T"], 2001)
    assert np.max(np.abs(hw_topp["traj"](t, 2))) <= task["hw_qddot_max"] * 1.01
    assert hw_topp["T"] == pytest.approx(1.5656, abs=0.01)
    q = hw_topp["traj"](np.array([0.0, hw_topp["T"]]), 0)
    assert np.allclose(q, np.array(task["waypoints"])[[0, -1]], atol=1e-6)


def test_min_time_scale_quintic_elbow_closed_form(elbow):
    """The quintic of Part A on a Part D model: torque-limited 0.5231 s; speed-limited (3 rad/s):
    T = (15/8) D / 3 = 1.3090 s (MR Fig. 9.4 peak speed)."""
    model, path = elbow
    Q, Qd, Qdd = quintic_samples(path)
    rob = dict(model, tau_max=np.array([10.0]), qdot_max=np.array([np.inf]))
    r = lab.min_time_scale(rob, Q, Qd, Qdd)
    assert r["c"] == pytest.approx(0.52312, abs=1e-3) and r["binding"] == ("torque", 0)
    r = lab.min_time_scale(dict(rob, qdot_max=np.array([3.0])), Q, Qd, Qdd)
    assert r["c"] == pytest.approx(15 / 8 * (2 * PI / 3) / 3, abs=1e-3) and r["binding"] == ("velocity", 0)


@pytest.mark.parametrize("case, speed, acc, T_topp, T_quintic, binding", [
    ("sim", 1.0, None, 0.6151, 1.1005, "velocity"),
    ("hw", 0.5, 4.0, 1.5656, 2.2964, "acceleration"),
    ("first run", 0.25, 4.0, 2.5025, 4.4020, "velocity")])
def test_topp_beats_quintic_on_course_arm(arm_task, case, speed, acc, T_topp, T_quintic, binding):
    task, r, path = arm_task
    tl = task["torque_fraction"] * r["tau_max"]
    vmax = speed * r["qdot_max"]
    tp = lab.topp(r, path, tl, qdot_max=vmax, qddot_max=acc, N=task["grid_N"])
    Q, Qd, Qdd = quintic_samples(path)
    res = lab.min_time_scale(dict(r, tau_max=tl, qdot_max=vmax), Q, Qd, Qdd,
                             qddot_max=None if acc is None else np.full(6, acc))
    assert tp["T"] == pytest.approx(T_topp, abs=2e-3)
    assert res["c"] == pytest.approx(T_quintic, abs=2e-3) and res["binding"][0] == binding
    assert 1.2 < res["c"] / tp["T"] < 2.0


# ---------------------------------------------------------------- Part E: the command file (provided functions)

def test_command_file_for_course_arm(arm_task, hw_topp):
    """The hw TOPP plan stretched by slow = 2, sampled at the 20 ms servo period, passes the checklist check at
    25 % rated speed with the 10 deg margin; the unstretched plan does not."""
    task, r, _ = arm_task
    t, q = lab.command_table(hw_topp["traj"], hw_topp["T"], dt=task["servo_period_s"], slow=2.0)
    assert len(t) == 158 and t[-1] == pytest.approx(3.131, abs=2e-3) and q.shape == (158, 6)
    ok = lab.check_command(t, q, r, task["first_run_speed_fraction"], task["joint_margin_deg"])
    assert ok["ok"] and 0.9 < ok["peak_speed_ratio"] <= 1.01
    t1, q1 = lab.command_table(hw_topp["traj"], hw_topp["T"], dt=task["servo_period_s"], slow=1.0)
    bad = lab.check_command(t1, q1, r, task["first_run_speed_fraction"], task["joint_margin_deg"])
    assert not bad["ok"] and bad["peak_speed_ratio"] > 1.8
