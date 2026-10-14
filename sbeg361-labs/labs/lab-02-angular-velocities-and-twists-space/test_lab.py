"""Auto-checks for SBEG 361 Lab 2 -- angular velocities and twists.

Usage:   LAB_IMPL=starter  python -m pytest -q test_lab.py     (your work, file starter.py)
         LAB_IMPL=solution python -m pytest -q test_lab.py     (reference; default)
Ground truth: the official `modern_robotics` library (Lynch & Park), the results of MR Sections 3.2.2 and
3.3.2 (Prop. 3.8, Prop. 3.9, Prop. 3.21, Prop. 3.22, Example 3.23) and finite differences of the data.
"""
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import modern_robotics as mr
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
lab = importlib.import_module(os.environ.get("LAB_IMPL", "starter"))

RNG = np.random.default_rng(361)


def random_T(seed):
    rng = np.random.default_rng(seed)
    return mr.RpToTrans(Rotation.random(random_state=seed).as_matrix(), rng.normal(size=3))


RANDOM_T = [random_T(s) for s in range(20)]


def load_nav():
    """Built inside each test (not as a fixture) so that each test starts from the static tree."""
    return lab.load_tree_json(HERE / "nav_tree.json")


@pytest.fixture(scope="module")
def track():
    return lab.load_track_csv(HERE / "tool_track.csv")


# ------------------------------------------------------------- Checkpoint 1: angular velocity
def test_angular_velocities_space_and_body():
    """[omega_s] = Rdot R^-1, [omega_b] = R^-1 Rdot (MR Eqs. 3.37-3.38); omega_s = R omega_b (Eq. 3.35)."""
    for T in RANDOM_T[:10]:
        R = T[:3, :3]
        w_b = RNG.normal(size=3)
        Rdot = R @ mr.VecToso3(w_b)                    # a frame turning with body angular velocity w_b
        w_s, w_b2 = lab.angular_velocities(R, Rdot)
        np.testing.assert_allclose(w_b2, w_b, atol=1e-12)
        np.testing.assert_allclose(w_s, R @ w_b, atol=1e-12)
        np.testing.assert_allclose(mr.VecToso3(w_s), R @ mr.VecToso3(w_b) @ R.T, atol=1e-12)   # Prop. 3.8


def test_angular_velocity_turntable_example():
    """R(t) = Rot(z, a t) Rot(x, b t): a turntable (rate a about z_s) carrying a body that spins about its own
    x-axis (rate b).  Product rule: omega_s = a z_s + b (R x_s) and omega_b = a (R^T z_s) + b x_b."""
    a, b, t = 0.5, 2.0, 1.0
    R = mr.MatrixExp3(mr.VecToso3([0, 0, a * t])) @ mr.MatrixExp3(mr.VecToso3([b * t, 0, 0]))
    Rdot = mr.VecToso3([0, 0, a]) @ R + R @ mr.VecToso3([b, 0, 0])
    w_s, w_b = lab.angular_velocities(R, Rdot)
    np.testing.assert_allclose(w_s, np.array([0, 0, a]) + b * R[:, 0], atol=1e-12)
    np.testing.assert_allclose(w_b, R.T @ [0, 0, a] + np.array([b, 0, 0]), atol=1e-12)
    np.testing.assert_allclose(w_s, [1.755165, 0.958851, 0.5], atol=1e-6)
    np.testing.assert_allclose(w_b, [2.0, 0.454649, -0.208073], atol=1e-6)
    assert np.isclose(np.linalg.norm(w_s), np.linalg.norm(w_b))        # same vector, two coordinate frames


# ------------------------------------------------------------- Checkpoint 2: se(3) and adjoint
def test_vec_to_se3_matches_mr():
    for _ in range(5):
        V = RNG.normal(size=6)
        np.testing.assert_allclose(lab.vec_to_se3(V), mr.VecTose3(V), atol=1e-12)
        np.testing.assert_allclose(lab.se3_to_vec(mr.VecTose3(V)), V, atol=1e-12)


def test_adjoint_matches_mr_and_properties():
    for T1, T2 in zip(RANDOM_T[:10], RANDOM_T[10:]):
        A1 = lab.adjoint(T1)
        np.testing.assert_allclose(A1, mr.Adjoint(T1), atol=1e-12)
        np.testing.assert_allclose(A1 @ lab.adjoint(T2), lab.adjoint(T1 @ T2), atol=1e-10)            # (3.77)
        np.testing.assert_allclose(np.linalg.inv(A1), lab.adjoint(mr.TransInv(T1)), atol=1e-10)       # (3.78)
        V = RNG.normal(size=6)
        # the adjoint law in its two forms: T [V] T^-1 = [Ad_T V] (Def. 3.20) and R [w] R^T = [R w] (Prop. 3.8)
        np.testing.assert_allclose(lab.vec_to_se3(A1 @ V), T1 @ mr.VecTose3(V) @ mr.TransInv(T1), atol=1e-10)
        R = T1[:3, :3]
        np.testing.assert_allclose((A1 @ V)[:3], R @ V[:3], atol=1e-12)
        np.testing.assert_allclose(lab.vec_to_se3(A1 @ V)[:3, :3], R @ mr.VecToso3(V[:3]) @ R.T, atol=1e-10)


# ------------------------------------------------------------- Checkpoint 3: twists and point velocities
def test_twists_from_T_Tdot_mr_example_3_23():
    """MR Example 3.23 (printed pp. 99-100): the three-wheeled car."""
    T_sb = np.array([[-1, 0, 0, 4], [0, 1, 0, 0.4], [0, 0, -1, 0], [0, 0, 0, 1.0]])
    V_b = np.array([0, 0, -2, 2.8, 4, 0])
    Tdot = T_sb @ mr.VecTose3(V_b)
    V_s, V_b2 = lab.twists_from_T_Tdot(T_sb, Tdot)
    np.testing.assert_allclose(V_s, [0, 0, 2, -2, -4, 0], atol=1e-12)
    np.testing.assert_allclose(V_b2, V_b, atol=1e-12)
    np.testing.assert_allclose(mr.Adjoint(T_sb) @ V_b2, V_s, atol=1e-12)


def test_twists_and_angular_velocities_agree():
    """Prop. 3.22 with random data: V_s = [Ad_T] V_b, and the angular parts are Week 3's omega_s, omega_b."""
    for T in RANDOM_T[:10]:
        V_s = RNG.normal(size=6)
        Tdot = mr.VecTose3(V_s) @ T
        V_s2, V_b = lab.twists_from_T_Tdot(T, Tdot)
        np.testing.assert_allclose(V_s2, V_s, atol=1e-12)
        np.testing.assert_allclose(V_b, mr.Adjoint(mr.TransInv(T)) @ V_s, atol=1e-10)              # (3.84)
        w_s, w_b = lab.angular_velocities(T[:3, :3], Tdot[:3, :3])
        np.testing.assert_allclose(w_s, V_s2[:3], atol=1e-12)
        np.testing.assert_allclose(w_b, V_b[:3], atol=1e-12)
        np.testing.assert_allclose(V_b[3:], T[:3, :3].T @ Tdot[:3, 3], atol=1e-12)                 # v_b = R^T pdot


def test_point_velocity():
    """xdot = omega x x + v in {s} and in {b}; v_s is the velocity of the body point at the {s} origin (Eq. 3.73)."""
    T_sb = np.array([[-1, 0, 0, 4], [0, 1, 0, 0.4], [0, 0, -1, 0], [0, 0, 0, 1.0]])   # MR Example 3.23
    V_b = np.array([0, 0, -2, 2.8, 4, 0])
    V_s = mr.Adjoint(T_sb) @ V_b
    Tdot = T_sb @ mr.VecTose3(V_b)
    np.testing.assert_allclose(lab.point_velocity(V_s, [0, 0, 0]), V_s[3:], atol=1e-12)
    np.testing.assert_allclose(lab.point_velocity(V_s, T_sb[:3, 3]), Tdot[:3, 3], atol=1e-12)   # pdot
    np.testing.assert_allclose(lab.point_velocity(V_s, T_sb[:3, 3]), [-2.8, 4, 0], atol=1e-12)
    np.testing.assert_allclose(lab.point_velocity(V_s, [2, -1, 0]), [0, 0, 0], atol=1e-12)     # on the screw axis
    x_b = RNG.normal(size=(6, 3))                                   # six body points, b coordinates
    x_s = lab.transform_point(T_sb, x_b)
    xdot_s = (Tdot @ np.c_[x_b, np.ones(6)].T).T[:, :3]             # d/dt (T x_b) = Tdot x_b
    np.testing.assert_allclose(lab.point_velocity(V_s, x_s), xdot_s, atol=1e-12)
    np.testing.assert_allclose(lab.point_velocity(V_b, x_b), xdot_s @ T_sb[:3, :3], atol=1e-12)   # R^T xdot_s


# ------------------------------------------------------------- Checkpoint 4: screws
def test_screw_to_twist_and_back():
    V = lab.screw_to_twist([2, -1, 0], [0, 0, 1], 0.0, 2.0)          # MR Example 3.23, V_s
    np.testing.assert_allclose(V, [0, 0, 2, -2, -4, 0], atol=1e-12)
    sc = lab.twist_to_screw(V)
    assert np.isclose(sc["thetadot"], 2) and np.isclose(sc["h"], 0)
    np.testing.assert_allclose(sc["q"], [2, -1, 0], atol=1e-12)
    np.testing.assert_allclose(sc["S"] * sc["thetadot"], V, atol=1e-12)
    for _ in range(10):                                              # random screws, q off the origin
        s = RNG.normal(size=3); s /= np.linalg.norm(s)
        q = RNG.normal(size=3); h = RNG.normal(); td = abs(RNG.normal()) + 0.1
        sc = lab.twist_to_screw(lab.screw_to_twist(q, s, h, td))
        assert np.isclose(sc["h"], h) and np.isclose(sc["thetadot"], td)
        np.testing.assert_allclose(sc["s_hat"], s, atol=1e-10)
        assert lab.point_line_distance(q, sc["q"], s) < 1e-10      # same line
        assert abs(sc["q"] @ s) < 1e-10                            # q is the point closest to the origin


def test_pure_translation_and_zero_twist():
    sc = lab.twist_to_screw([0, 0, 0, 0, 0, 0.05])
    assert np.isinf(sc["h"]) and sc["q"] is None and np.isclose(sc["thetadot"], 0.05)
    np.testing.assert_allclose(sc["S"], [0, 0, 0, 0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(lab.screw_to_twist(None, [0, 0, 2], np.inf, 0.05), [0, 0, 0, 0, 0, 0.05])
    with pytest.raises(ValueError):
        lab.twist_to_screw(np.zeros(6))


def test_pitch_is_frame_invariant():
    V = RNG.normal(size=6)
    h = lab.twist_to_screw(V)["h"]
    for T in RANDOM_T[:5]:
        assert np.isclose(lab.twist_to_screw(mr.Adjoint(T) @ V)["h"], h)


# ------------------------------------------------------------- Checkpoint 5: navigated screwdriver, EOD gripper
def test_body_twist_from_tracking_data(track):
    t, Ts = track
    dt = t[1] - t[0]
    Vs, Vb = lab.twists_from_samples(Ts, dt)
    assert Vs.shape == Vb.shape == (len(t) - 2, 6)
    exact = mr.se3ToVec(mr.MatrixLog6(mr.TransInv(Ts[0]) @ Ts[1])) / dt     # constant screw motion
    np.testing.assert_allclose(Vb, np.tile(exact, (len(Vb), 1)), rtol=0, atol=5e-4)
    for k in (0, 150, len(Vb) - 1):                                          # V_s = [Ad_Tsb] V_b
        np.testing.assert_allclose(Vs[k], mr.Adjoint(Ts[k + 1]) @ Vb[k], atol=1e-9)


def test_screw_axis_and_lead_of_the_screwdriver(track):
    nav = load_nav()
    t, Ts = track
    Vs, Vb = lab.twists_from_samples(Ts, t[1] - t[0])
    sc_b = lab.twist_to_screw(Vb.mean(axis=0))
    assert np.isclose(sc_b["h"] * 2 * np.pi * 1000, 2.5, atol=0.01)          # lead in mm/rev
    assert np.isclose(sc_b["thetadot"], np.pi / 2, rtol=1e-3)
    np.testing.assert_allclose(sc_b["s_hat"], [0, 0, -1], atol=1e-6)
    np.testing.assert_allclose(sc_b["q"], [0.05, 0, 0], atol=1e-6)
    V_ct = lab.adjoint(nav.lookup("ct_image", "tracker")) @ Vs.mean(axis=0)
    sc = lab.twist_to_screw(V_ct)
    T_plan = nav.lookup("ct_image", "plan_entry")
    assert np.isclose(np.rad2deg(lab.axis_angle_between(sc["s_hat"], T_plan[:3, 2])), 2.0, atol=0.01)
    assert np.isclose(lab.point_line_distance(T_plan[:3, 3], sc["q"], sc["s_hat"]) * 1000, 1.0, atol=0.01)


def test_tool_tip_velocity(track):
    """The marker origin moves at ~78.5 mm/s, the tip (on the screw axis) only at the axial 0.625 mm/s."""
    nav = load_nav()
    t, Ts = track
    dt = t[1] - t[0]
    Vs, Vb = lab.twists_from_samples(Ts, dt)
    tip_m = nav.lookup("tool_marker", "tool_tip")[:3, 3]
    np.testing.assert_allclose(lab.point_velocity(Vb.mean(axis=0), [0, 0, 0]) * 1000, [0, 78.527, -0.625], atol=1e-3)
    np.testing.assert_allclose(lab.point_velocity(Vb.mean(axis=0), tip_m) * 1000, [0, 0, -0.625], atol=1e-3)
    tips = np.array([lab.transform_point(T, tip_m) for T in Ts])               # tip track, tracker coordinates
    fd = (tips[2:] - tips[:-2]) / (2 * dt)                                   # central difference of the tip track
    v_tip = np.array([lab.point_velocity(Vs[k], tips[k + 1]) for k in range(len(Vs))])
    np.testing.assert_allclose(v_tip, fd, atol=1e-9)
    nav.add("tracker", "tool_marker", Ts[0])
    R_plan_tracker = nav.lookup("plan_entry", "tracker")[:3, :3]
    v_plan = R_plan_tracker @ v_tip[199]                                     # at t = 4 s, in plan coordinates
    np.testing.assert_allclose(v_plan * 1000, [0, -0.021812, 0.624619], atol=1e-5)    # mm/s; 2 deg off the plan


def test_eod_gripper_twist_in_camera_frame():
    """EOD robot (protective use): the gripper turns about its approach axis z_g at 0.5 rad/s (opening a lid).
    Twist in the mast-camera and base frames via the adjoint (MR Eq. 3.83 generalised: V_c = [Ad_Tcd] V_d)."""
    eod = lab.load_tree_json(HERE / "eod_tree.json")
    V_g = np.array([0, 0, 0.5, 0, 0, 0])
    T_cg = eod.lookup("mast_camera", "gripper")
    V_c = lab.adjoint(T_cg) @ V_g
    np.testing.assert_allclose(V_c, [0, 0, 0.5, 0.2, 0, 0], atol=1e-9)
    sc = lab.twist_to_screw(V_c)
    assert np.isclose(sc["h"], 0, atol=1e-12) and np.isclose(sc["thetadot"], 0.5)
    np.testing.assert_allclose(sc["s_hat"], T_cg[:3, 2], atol=1e-9)
    assert lab.point_line_distance(T_cg[:3, 3], sc["q"], sc["s_hat"]) < 1e-9     # axis through the gripper origin
    T_bg = eod.lookup("base_link", "gripper")
    V_base = lab.adjoint(T_bg) @ V_g
    np.testing.assert_allclose(V_base, [0.5, 0, 0, 0, 0.2, 0], atol=1e-9)
    r_g = np.array([0, 0.04, 0.1])                                   # a point on a jaw, gripper coordinates
    v_g = lab.point_velocity(V_g, r_g)
    np.testing.assert_allclose(v_g, [-0.02, 0, 0], atol=1e-12)
    np.testing.assert_allclose(lab.point_velocity(V_c, T_cg[:3, :3] @ r_g + T_cg[:3, 3]), T_cg[:3, :3] @ v_g,
                               atol=1e-12)
