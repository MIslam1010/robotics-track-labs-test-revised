"""Auto-checks for SBEG 361 Lab 1 (rotations and transformations: SO(3), SE(3), transform trees).

Run from this folder:
    LAB_IMPL=starter  python -m pytest -q      # your work (file starter.py)
Results are compared with the official modern_robotics library wherever it has an equivalent, with the
worked examples of MR Chapter 3 (Fig. 3.7, Fig. 3.14, Fig. 3.15, Example 3.19) and with scipy.spatial.transform.
"""
import importlib
import os
import sys

import numpy as np
import pytest
import modern_robotics as mr
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
lab = importlib.import_module(os.environ.get("LAB_IMPL", "starter"))

DATA_DIR = os.path.join(HERE, "data")
DATA = os.path.join(DATA_DIR, "tracker_orientations.csv")
CORRUPTED = {4, 9, 13, 17}
ANGLES = [0.0, 0.3, -1.2, np.pi / 2, np.pi, 2.7]
RNG = np.random.default_rng(361)   # consumed by test_mr_fig_3_14_frames_and_point, then test_distance_preserved
S2 = 1 / np.sqrt(2)


def mr_rot(axis, theta):
    """Reference rotation from modern_robotics: exp([w_hat] theta)."""
    return mr.MatrixExp3(mr.VecToso3(np.asarray(axis, float) * theta))


def random_T(seed):
    rng = np.random.default_rng(seed)
    return mr.RpToTrans(Rotation.random(random_state=seed).as_matrix(), rng.normal(size=3))


RANDOM_T = [random_T(s) for s in range(20)]

# MR Fig. 3.14 (printed p. 90)
T_SB_314 = mr.RpToTrans(np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0.0]]), [0, -2, 0])
T_SC_314 = mr.RpToTrans(np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0.0]]), [-1, 1, 0])
T_BC_314 = mr.RpToTrans(np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0.0]]), [0, -3, -1])

# MR Example 3.19 (printed pp. 93-95)
T_DB = np.array([[0, 0, -1, 250], [0, -1, 0, -150], [-1, 0, 0, 200], [0, 0, 0, 1.0]])
T_DE = np.array([[0, 0, -1, 300], [0, -1, 0, 100], [-1, 0, 0, 120], [0, 0, 0, 1.0]])
T_AD = np.array([[0, 0, -1, 400], [0, -1, 0, 50], [-1, 0, 0, 300], [0, 0, 0, 1.0]])
T_BC = np.array([[0, -S2, -S2, 30], [0, S2, -S2, -40], [1, 0, 0, 25], [0, 0, 0, 1.0]])
T_CE_MR = np.array([[0, 0, 1, -75], [-S2, S2, 0, -260 * S2], [-S2, -S2, 0, 160 * S2], [0, 0, 0, 1.0]])


def load_nav():
    """Built inside each test (not as a fixture) so that the starter fails as a test, not an error."""
    return lab.load_tree_json(os.path.join(DATA_DIR, "nav_tree.json"))


@pytest.fixture(scope="module")
def track():
    return lab.load_track_csv(os.path.join(DATA_DIR, "tool_track.csv"))


# ---------------- Part A: building R and T ----------------
@pytest.mark.parametrize("theta", ANGLES)
def test_elementary_rotations_match_mr(theta):
    np.testing.assert_allclose(lab.rot_x(theta), mr_rot([1, 0, 0], theta), atol=1e-12)
    np.testing.assert_allclose(lab.rot_y(theta), mr_rot([0, 1, 0], theta), atol=1e-12)
    np.testing.assert_allclose(lab.rot_z(theta), mr_rot([0, 0, 1], theta), atol=1e-12)


def test_axis_angle_matches_mr():
    rng = np.random.default_rng(2)
    for _ in range(25):
        w = rng.normal(size=3)
        w /= np.linalg.norm(w)
        th = rng.uniform(-np.pi, np.pi)
        np.testing.assert_allclose(lab.rot_axis_angle(w, th), mr_rot(w, th), atol=1e-12)


def test_axis_angle_identities():
    w = np.array([1.0, 2.0, 2.0]) / 3.0
    np.testing.assert_allclose(lab.rot_axis_angle(w, 0.7), lab.rot_axis_angle(-w, -0.7), atol=1e-12)  # MR p. 72
    np.testing.assert_allclose(lab.rot_axis_angle(w, 0.0), np.eye(3), atol=1e-12)
    np.testing.assert_allclose(lab.rot_axis_angle(w, 0.9) @ w, w, atol=1e-12)  # axis is unchanged


def test_axis_angle_rejects_non_unit():
    with pytest.raises(ValueError):
        lab.rot_axis_angle([1.0, 1.0, 0.0], 0.5)


def test_rp_to_trans_and_back():
    for T in RANDOM_T[:5]:
        R, p = mr.TransToRp(T)
        np.testing.assert_allclose(lab.rp_to_trans(R, p), T, atol=1e-12)
        R2, p2 = lab.trans_to_rp(T)
        np.testing.assert_allclose(R2, R, atol=1e-12)
        np.testing.assert_allclose(p2, p, atol=1e-12)


# ---------------- Part B: group property tests, SO(3) and SE(3) ----------------
def test_so3_errors_values():
    e, d = lab.so3_errors(np.eye(3))
    assert e == pytest.approx(0.0) and d == pytest.approx(1.0)
    e, d = lab.so3_errors(2.0 * np.eye(3))
    assert e == pytest.approx(np.sqrt(27.0)) and d == pytest.approx(8.0)
    with pytest.raises(ValueError):
        lab.so3_errors(np.eye(2))


def test_is_rotation_cases():
    Rc = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)  # MR Fig. 3.7
    swap = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], float)  # det = -1
    assert lab.is_rotation(Rc) and mr.TestIfSO3(Rc)
    assert not lab.is_rotation(swap) and not mr.TestIfSO3(swap)
    assert not lab.is_rotation(1.05 * Rc)
    assert not lab.is_rotation(np.eye(2))
    mr_example = np.array([[1.0, 0.0, 0.0], [0.0, 0.1, -0.95], [0.0, 1.0, 0.1]])  # MR library docstring
    assert not lab.is_rotation(mr_example) and not mr.TestIfSO3(mr_example)


def test_group_properties():
    """Closure, inverse = transpose, non-commutativity, length preservation (MR Props. 3.3-3.6)."""
    rng = np.random.default_rng(5)
    for _ in range(20):
        A = mr_rot(rng.normal(size=3) / 1.7, 1.0)
        B = mr_rot(rng.normal(size=3) / 1.7, 1.0)
        assert lab.is_rotation(A @ B)
        np.testing.assert_allclose(lab.rot_inv(A), mr.RotInv(A), atol=1e-12)
        np.testing.assert_allclose(lab.rot_inv(A) @ A, np.eye(3), atol=1e-12)
        x = rng.normal(size=3)
        assert np.linalg.norm(A @ x) == pytest.approx(np.linalg.norm(x))
    X, Y = lab.rot_x(np.pi / 2), lab.rot_y(np.pi / 2)
    assert not np.allclose(X @ Y, Y @ X)
    Z1, Z2 = lab.rot_z(0.4), lab.rot_z(1.1)
    np.testing.assert_allclose(Z1 @ Z2, Z2 @ Z1, atol=1e-12)  # same-axis (planar) rotations commute


def test_project_to_so3():
    M = np.array([[0.675, 0.150, 0.720], [0.370, 0.771, -0.511], [-0.630, 0.619, 0.472]])
    np.testing.assert_allclose(lab.project_to_so3(M), mr.ProjectToSO3(M), atol=1e-8)
    reflect = np.diag([1.0, 1.0, -1.0])
    R = lab.project_to_so3(reflect)
    assert lab.is_rotation(R)


def test_is_se3():
    """The SE(3) twin of test_is_rotation_cases, plus closure (MR Prop. 3.16)."""
    assert lab.is_se3(RANDOM_T[0])
    assert lab.is_se3(RANDOM_T[0] @ RANDOM_T[1])                      # closure
    bad = RANDOM_T[0].copy(); bad[:3, :3] *= -1           # det R = -1 (a reflection)
    assert not lab.is_se3(bad)
    bad2 = RANDOM_T[0].copy(); bad2[3, 0] = 0.1           # wrong last row
    assert not lab.is_se3(bad2)
    assert not lab.is_se3(np.eye(3))


def test_trans_inv_matches_mr():
    """The SE(3) twin of rot_inv: T^-1 = (R^T, -R^T p) (MR Eq. 3.64)."""
    for T in RANDOM_T:
        np.testing.assert_allclose(lab.trans_inv(T), mr.TransInv(T), atol=1e-12)
        np.testing.assert_allclose(lab.trans_inv(T) @ T, np.eye(4), atol=1e-12)


# ---------------- Part C: the three uses of R and T ----------------
def test_mr_figure_3_7():
    Rb = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    Rc = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)
    pa = np.array([1.0, 1.0, 0.0])
    np.testing.assert_allclose(lab.change_frame(lab.rot_inv(Rb), pa), [1, -1, 0], atol=1e-12)
    np.testing.assert_allclose(lab.change_frame(lab.rot_inv(Rc), pa), [0, -1, -1], atol=1e-12)
    np.testing.assert_allclose(Rb, lab.rot_z(np.pi / 2), atol=1e-12)
    np.testing.assert_allclose(lab.rotate_in_body(Rb, lab.rot_y(-np.pi / 2)), Rc, atol=1e-12)
    np.testing.assert_allclose(lab.rotate_in_fixed(Rb, lab.rot_x(np.pi / 2)), Rc, atol=1e-12)


def test_eod_base_pan_and_wrist():
    """Defence checkpoint: EOD robot gripper, base yaw (fixed z_s) vs wrist roll (body x_b)."""
    R_sb = lab.rot_z(np.pi / 6) @ lab.rot_y(np.pi / 4)
    pan = lab.rot_z(np.pi / 2)
    roll = lab.rot_x(np.pi / 3)
    np.testing.assert_allclose(lab.rotate_in_fixed(R_sb, pan), pan @ R_sb, atol=1e-12)
    np.testing.assert_allclose(lab.rotate_in_body(R_sb, roll), R_sb @ roll, atol=1e-12)
    # roll about the gripper's own x axis leaves that axis (first column) unchanged
    np.testing.assert_allclose(lab.rotate_in_body(R_sb, roll)[:, 0], R_sb[:, 0], atol=1e-12)


def test_mr_fig_3_14_frames_and_point():
    """T_bc = T_sb^-1 T_sc (MR p. 90) and v_b = (0, 0, 1.5) -> v_s."""
    T_bc = lab.trans_inv(T_SB_314) @ T_SC_314
    np.testing.assert_allclose(T_bc, T_BC_314, atol=1e-12)
    np.testing.assert_allclose(lab.transform_point(T_SB_314, [0, 0, 1.5]), [1.5, -2, 0], atol=1e-12)
    pts = RNG.normal(size=(7, 3))
    expected = (T_SB_314 @ np.c_[pts, np.ones(7)].T).T[:, :3]
    np.testing.assert_allclose(lab.transform_point(T_SB_314, pts), expected, atol=1e-12)


def test_mr_fig_3_15_fixed_vs_body_frame():
    """Pre-multiplication = fixed-frame displacement, post-multiplication = body-frame (MR Eqs. 3.66-3.67)."""
    T = lab.rp_to_trans(mr.MatrixExp3(mr.VecToso3([0, 0, np.pi / 2])), [0, 2, 0])
    T_sb = np.array([[0, 0, 1, 0], [0, -1, 0, -2], [1, 0, 0, 0], [0, 0, 0, 1.0]])
    np.testing.assert_allclose(lab.transform_in_fixed(T_sb, T),
                               [[0, 1, 0, 2], [0, 0, 1, 2], [1, 0, 0, 0], [0, 0, 0, 1]], atol=1e-12)
    np.testing.assert_allclose(lab.transform_in_body(T_sb, T),
                               [[0, 0, 1, 0], [-1, 0, 0, -4], [0, -1, 0, 0], [0, 0, 0, 1]], atol=1e-12)
    # the rotation block follows the SO(3) rule of MR Eqs. 3.23-3.24
    R, R_sb = T[:3, :3], T_sb[:3, :3]
    np.testing.assert_allclose(lab.transform_in_fixed(T_sb, T)[:3, :3], lab.rotate_in_fixed(R_sb, R), atol=1e-12)
    np.testing.assert_allclose(lab.transform_in_body(T_sb, T)[:3, :3], lab.rotate_in_body(R_sb, R), atol=1e-12)


def test_distance_preserved():
    """MR Prop. 3.18(a): ||Tx - Ty|| = ||x - y||."""
    x, y = RNG.normal(size=3), RNG.normal(size=3)
    for T in RANDOM_T[:5]:
        assert np.isclose(np.linalg.norm(lab.transform_point(T, x) - lab.transform_point(T, y)),
                          np.linalg.norm(x - y))


def test_planar_compose():
    P = lab.rot_z(np.pi / 3)[:2, :2]
    Q = lab.rot_z(np.pi / 6)[:2, :2]
    R, r = lab.planar_compose(P, [2.0, 1.0], Q, [1.0, 0.5])
    np.testing.assert_allclose(R, [[0, -1], [1, 0]], atol=1e-12)
    np.testing.assert_allclose(r, [2.0 + 0.5 - np.sqrt(3) / 4, 1.0 + np.sqrt(3) / 2 + 0.25], atol=1e-12)
    # the same composition as a product of 3x3 SE(2) matrices (MR Eq. 3.63)
    T_sb, T_bc = lab.planar_to_se2(P, [2.0, 1.0]), lab.planar_to_se2(Q, [1.0, 0.5])
    np.testing.assert_allclose(T_sb[2], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(T_sb @ T_bc, lab.planar_to_se2(R, r), atol=1e-12)


# ---------------- Part D: the transform tree ----------------
def test_tree_mr_example_3_19():
    tree = lab.TransformTree()
    tree.add("a", "d", T_AD)
    tree.add("d", "b", T_DB)
    tree.add("d", "e", T_DE)
    tree.add("b", "c", T_BC)
    np.testing.assert_allclose(tree.lookup("c", "e"), T_CE_MR, atol=1e-9)
    np.testing.assert_allclose(tree.lookup("a", "e"), T_AD @ T_DE, atol=1e-9)
    np.testing.assert_allclose(tree.lookup("a", "c"), T_AD @ T_DB @ T_BC, atol=1e-9)
    np.testing.assert_allclose(tree.lookup("e", "c"), mr.TransInv(T_CE_MR), atol=1e-9)
    np.testing.assert_allclose(tree.lookup("b", "b"), np.eye(4), atol=1e-12)
    assert tree.path("c", "e") == ["c", "b", "d", "e"]


def test_tree_rejects_bad_edges():
    tree = lab.TransformTree()
    tree.add("world", "a", RANDOM_T[0])
    tree.add("a", "b", RANDOM_T[1])
    tree.add("a", "b", RANDOM_T[2])                                   # update is allowed
    np.testing.assert_allclose(tree.lookup("a", "b"), RANDOM_T[2], atol=1e-12)
    with pytest.raises(ValueError):
        tree.add("world", "b", RANDOM_T[3])                          # second parent
    with pytest.raises(ValueError):
        tree.add("b", "world", RANDOM_T[3])                          # cycle
    with pytest.raises(ValueError):
        tree.add("b", "c", 2 * RANDOM_T[3])                          # not in SE(3)
    with pytest.raises(KeyError):
        tree.lookup("world", "nowhere")
    tree.add("other_root", "z", RANDOM_T[4])
    with pytest.raises(LookupError):
        tree.lookup("world", "z")


def test_quaternion_order_and_json_loader():
    nav = load_nav()
    for s in range(5):
        r = Rotation.random(random_state=s)
        np.testing.assert_allclose(lab.quat_xyzw_to_rot(r.as_quat()), r.as_matrix(), atol=1e-12)
    assert {"theatre", "tracker", "patient_ref", "ct_image", "plan_entry", "tool_marker",
            "tool_tip"} <= nav.frames


def test_eod_tree_package_in_gripper():
    eod = lab.load_tree_json(os.path.join(DATA_DIR, "eod_tree.json"))
    T_gp = eod.lookup("gripper", "package")
    np.testing.assert_allclose(T_gp[:3, 3], [0.1, -0.1, 0.5], atol=1e-9)
    np.testing.assert_allclose(T_gp[:3, :3], [[0, 0, -1], [0, 1, 0], [1, 0, 0]], atol=1e-9)
    np.testing.assert_allclose(eod.lookup("base_link", "package")[:3, 3], [1.4, -0.1, 0.3], atol=1e-9)


# ---------------- Part E: applications ----------------
def test_tracker_classification():
    Rs = lab.load_tracker_csv(DATA)
    assert Rs.shape == (20, 3, 3)
    ok = lab.classify_samples(Rs)
    assert set(np.flatnonzero(~np.asarray(ok))) == CORRUPTED
    ref = np.array([mr.TestIfSO3(R) for R in Rs])
    assert np.array_equal(np.asarray(ok, bool), ref)


def test_tracker_projection_and_tool():
    Rs = lab.load_tracker_csv(DATA)
    ok = np.asarray(lab.classify_samples(Rs), bool)
    for R in Rs[ok]:
        Rp = lab.project_to_so3(R)
        np.testing.assert_allclose(Rp, mr.ProjectToSO3(R), atol=1e-9)
        assert lab.so3_errors(Rp)[0] < 1e-12
    R_mt = lab.rot_x(np.pi / 6)  # marker-to-tip calibration (illustrative)
    R_st = lab.tool_orientations(Rs[ok], R_mt)
    assert R_st.shape == (ok.sum(), 3, 3)
    np.testing.assert_allclose(R_st[0], Rs[ok][0] @ R_mt, atol=1e-12)


def test_tool_tip_in_ct_at_start(track):
    nav = load_nav()
    t, Ts = track
    nav.add("tracker", "tool_marker", Ts[0])
    T = nav.lookup("ct_image", "tool_tip")
    direct = (mr.TransInv(nav.lookup("tracker", "ct_image")) @ Ts[0] @ nav.lookup("tool_marker", "tool_tip"))
    np.testing.assert_allclose(T, direct, atol=1e-12)
    np.testing.assert_allclose(T[:3, 3] * 1000, [142.0, 84.035, 210.262], atol=1e-2)       # mm
    tip_plan = nav.lookup("plan_entry", "tool_tip")[:3, 3] * 1000
    assert np.isclose(np.linalg.norm(tip_plan), 1.0, atol=1e-3)                            # 1.0 mm entry error
    assert nav.path("ct_image", "tool_tip") == ["ct_image", "patient_ref", "tracker", "tool_marker", "tool_tip"]


def test_tip_depth_along_plan(track):
    """Update the tracker -> tool_marker edge for every sample and follow the tip in the plan frame."""
    nav = load_nav()
    t, Ts = track
    tips = []
    for T in Ts:
        nav.add("tracker", "tool_marker", T)
        tips.append(nav.lookup("plan_entry", "tool_tip")[:3, 3] * 1000)              # mm
    tips = np.array(tips)
    assert tips.shape == (len(t), 3)
    np.testing.assert_allclose(tips[-1], [0.0, 0.826, 4.997], atol=1e-2)            # t = 8 s
    assert np.all(np.diff(tips[:, 2]) > 0)                                          # depth grows monotonically
    lateral = np.hypot(tips[:, 0], tips[:, 1])
    assert np.isclose(lateral[0], 1.0, atol=1e-3) and np.isclose(lateral[-1], 0.826, atol=1e-3)
