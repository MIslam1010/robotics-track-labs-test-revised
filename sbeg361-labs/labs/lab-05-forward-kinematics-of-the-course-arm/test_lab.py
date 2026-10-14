"""Auto-checks for SBEG 361 Lab 5 -- PoE forward kinematics of the course arm.

Usage:   LAB_IMPL=starter  python -m pytest -q test_lab.py     (your work, file starter.py)
         LAB_IMPL=solution python -m pytest -q test_lab.py     (reference; default)
Ground truth: the official `modern_robotics` library (FKinSpace, FKinBody, MatrixExp6, Adjoint),
the worked examples of MR Chapter 4 (Examples 4.1, 4.3-4.7; printed pp. 141-150), the UR5 URDF of
MR Sec. 4.2 (pp. 153-154), scipy.spatial.transform, Pinocchio (if installed) and the course-arm
model in this folder (course_arm_model.py, course_arm_reference()).
"""
import csv
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import modern_robotics as mr
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
sys.path.insert(0, str(HERE))
lab = importlib.import_module(os.environ.get("LAB_IMPL", "starter"))
from course_arm_model import COURSE_ARM_GEOMETRY, course_arm_reference  # noqa: E402

RNG = np.random.default_rng(361)
P_TOOL = (0.0, 0.0, 0.06)


# ------------------------------------------------------------------ textbook robots (MR Ch. 4)
def mr_ex41(L1=1.0, L2=1.0):        # MR Example 4.1, 3R spatial chain (p. 141-142)
    S = np.array([[0, 0, 1, 0, 0, 0], [0, -1, 0, 0, 0, -L1], [1, 0, 0, 0, -L2, 0]], float).T
    M = np.array([[0, 0, 1, L1], [0, 1, 0, 0], [-1, 0, 0, -L2], [0, 0, 0, 1]], float)
    return M, S


def mr_ex43(L=1.0):                 # MR Example 4.3, 6R open chain (pp. 143-144)
    S = np.array([[0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0],
                  [-1, 0, 0, 0, 0, L], [-1, 0, 0, 0, 0, 2 * L], [0, 1, 0, 0, 0, 0]], float).T
    M = np.eye(4)
    M[1, 3] = 3 * L
    return M, S


B_EX46 = np.array([[0, 0, 1, -3, 0, 0], [0, 1, 0, 0, 0, 0], [-1, 0, 0, 0, 0, -3],
                   [-1, 0, 0, 0, 0, -2], [-1, 0, 0, 0, 0, -1], [0, 1, 0, 0, 0, 0]], float).T  # MR Ex. 4.6, L = 1


def mr_ex44(L1=1.0, L2=0.5):        # MR Example 4.4, RRPRRR (pp. 144-145)
    S = np.array([[0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0],
                  [0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, -L1], [0, 1, 0, 0, 0, 0]], float).T
    M = np.eye(4)
    M[1, 3] = L1 + L2
    return M, S


def ur_type(W1, W2, L1, L2, H1, H2):   # MR Example 4.5 screw table (p. 145)
    S = np.array([[0, 0, 1, 0, 0, 0], [0, 1, 0, -H1, 0, 0], [0, 1, 0, -H1, 0, L1],
                  [0, 1, 0, -H1, 0, L1 + L2], [0, 0, -1, -W1, L1 + L2, 0], [0, 1, 0, H2 - H1, 0, L1 + L2]], float).T
    M = np.array([[-1, 0, 0, L1 + L2], [0, 0, 1, W1 + W2], [0, 1, 0, H1 - H2], [0, 0, 0, 1]], float)
    return M, S


UR5_PRINTED = dict(W1=0.109, W2=0.082, L1=0.425, L2=0.392, H1=0.089, H2=0.095)             # MR Fig. 4.6
UR5_URDF = dict(W1=0.10915, W2=0.0823, L1=0.425, L2=0.39225, H1=0.089159, H2=0.09465)      # MR URDF p. 153


def wam():                          # MR Example 4.7 (pp. 148-150), body screw axes
    L1, L2, L3, W1 = 0.550, 0.300, 0.060, 0.045
    B = np.array([[0, 0, 1, 0, 0, 0], [0, 1, 0, L1 + L2 + L3, 0, 0], [0, 0, 1, 0, 0, 0],
                  [0, 1, 0, L2 + L3, 0, W1], [0, 0, 1, 0, 0, 0], [0, 1, 0, L3, 0, 0], [0, 0, 1, 0, 0, 0]], float).T
    M = np.eye(4)
    M[2, 3] = L1 + L2 + L3
    return M, B


def rrp():                          # MR Exercise 5.7 RRP robot (Fig. 5.19, p. 204)
    S = np.array([[0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 2, 0], [0, 0, 0, 0, 1, 0]], float).T
    M = np.array([[-1, 0, 0, 0], [0, 0, 1, 3], [0, 1, 0, 2], [0, 0, 0, 1]], float)
    return M, S


ROBOTS = {"ex41": mr_ex41(0.4, 0.3), "ex43": mr_ex43(0.2), "ex44": mr_ex44(), "rrp": rrp(),
          "ur5": ur_type(**UR5_URDF), "course_arm": course_arm_reference()[:2]}


def read_tip_checks():
    with open(DATA / "tip_checks.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    Qs = np.array([[float(r[f"servo_{i + 1}"]) for i in range(6)] for r in rows])
    P = np.array([[float(r[k]) for k in ("x_mm", "y_mm", "z_mm")] for r in rows])
    return Qs, P


# ------------------------------------------------------------- Checkpoint 1: screw axes and exp
def test_screw_axis_mr_example_4_1():
    L1, L2 = 0.4, 0.3
    _, S = mr_ex41(L1, L2)
    np.testing.assert_allclose(lab.screw_axis((0, 0, 1), (0, 0, 0)), S[:, 0], atol=1e-12)
    np.testing.assert_allclose(lab.screw_axis((0, -1, 0), (L1, 0, 0)), S[:, 1], atol=1e-12)
    np.testing.assert_allclose(lab.screw_axis((1, 0, 0), (0, 0, -L2)), S[:, 2], atol=1e-12)
    # any point on the axis gives the same screw
    np.testing.assert_allclose(lab.screw_axis((1, 0, 0), (5.0, 0, -L2)), S[:, 2], atol=1e-12)


def test_screw_axis_prismatic_and_errors():
    np.testing.assert_allclose(lab.screw_axis(v=(0, 1, 0)), [0, 0, 0, 0, 1, 0], atol=1e-12)
    with pytest.raises(ValueError):
        lab.screw_axis((0, 0, 2), (0, 0, 0))          # not a unit vector
    with pytest.raises(ValueError):
        lab.screw_axis(v=(0, 0, 0.5))
    with pytest.raises(ValueError):
        lab.screw_axis()


def test_screw_exp_matches_mr():
    for _ in range(20):
        w = RNG.normal(size=3)
        w /= np.linalg.norm(w)
        S = np.r_[w, RNG.normal(size=3)]
        th = RNG.uniform(-np.pi, np.pi)
        np.testing.assert_allclose(lab.screw_exp(S, th), mr.MatrixExp6(mr.VecTose3(S * th)), atol=1e-10)
    v = np.array([0, 0.6, 0.8])
    np.testing.assert_allclose(lab.screw_exp(np.r_[0, 0, 0, v], 0.25),
                               mr.MatrixExp6(mr.VecTose3(np.r_[0, 0, 0, v] * 0.25)), atol=1e-12)
    np.testing.assert_allclose(lab.screw_exp(np.r_[0, 0, 1, 0, 0, 0], 0.0), np.eye(4), atol=1e-12)


def test_mr_example_4_5_exponentials():
    """The two matrix exponentials printed in MR Example 4.5 (p. 146), rounded UR5 values."""
    _, S = ur_type(**UR5_PRINTED)
    E2 = np.array([[0, 0, -1, 0.089], [0, 1, 0, 0], [1, 0, 0, 0.089], [0, 0, 0, 1]])
    E5 = np.array([[0, 1, 0, 0.708], [-1, 0, 0, 0.926], [0, 0, 1, 0], [0, 0, 0, 1]])
    np.testing.assert_allclose(lab.screw_exp(S[:, 1], -np.pi / 2), E2, atol=5e-4)
    np.testing.assert_allclose(lab.screw_exp(S[:, 4], np.pi / 2), E5, atol=5e-4)


# ------------------------------------------------------------- Checkpoint 2: PoE FK, space and body
@pytest.mark.parametrize("name", list(ROBOTS))
def test_fk_space_matches_mr(name):
    M, S = ROBOTS[name]
    for _ in range(10):
        th = RNG.uniform(-np.pi, np.pi, S.shape[1])
        np.testing.assert_allclose(lab.fk_space(M, S, th), mr.FKinSpace(M, S, th), atol=1e-10)
    np.testing.assert_allclose(lab.fk_space(M, S, np.zeros(S.shape[1])), M, atol=1e-12)


def test_ur5_mr_example_4_5():
    M, S = ur_type(**UR5_PRINTED)
    T = lab.fk_space(M, S, [0, -np.pi / 2, 0, 0, np.pi / 2, 0])
    T_mr = np.array([[0, -1, 0, 0.095], [1, 0, 0, 0.109], [0, 0, 1, 0.988], [0, 0, 0, 1]])
    np.testing.assert_allclose(T, T_mr, atol=5e-4)


def test_rrp_check_value():
    M, S = rrp()
    T = lab.fk_space(M, S, [np.pi / 2, np.pi / 2, 1.0])
    np.testing.assert_allclose(T, [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 6], [0, 0, 0, 1]], atol=1e-12)


def test_body_screws_mr_example_4_6():
    M, S = mr_ex43(1.0)
    np.testing.assert_allclose(lab.body_screws(S, M), B_EX46, atol=1e-12)


@pytest.mark.parametrize("name", list(ROBOTS))
def test_fk_body_equals_space(name):
    M, S = ROBOTS[name]
    B = mr.Adjoint(mr.TransInv(M)) @ S
    for _ in range(10):
        th = RNG.uniform(-np.pi, np.pi, S.shape[1])
        np.testing.assert_allclose(lab.fk_body(M, B, th), mr.FKinBody(M, B, th), atol=1e-10)
        np.testing.assert_allclose(lab.fk_body(M, B, th), lab.fk_space(M, S, th), atol=1e-10)


def test_wam_mr_example_4_7():
    M, B = wam()
    T = lab.fk_body(M, B, [0, np.pi / 4, 0, -np.pi / 4, 0, -np.pi / 2, 0])
    T_mr = np.array([[0, 0, -1, 0.3157], [0, 1, 0, 0], [1, 0, 0, 0.6571], [0, 0, 0, 1]])
    np.testing.assert_allclose(T, T_mr, atol=5e-5)


def test_course_arm_screws():
    M_ref, S_ref, _ = course_arm_reference()
    M, S = lab.course_arm_screws(**COURSE_ARM_GEOMETRY)
    np.testing.assert_allclose(S, S_ref, atol=1e-12)
    np.testing.assert_allclose(M, M_ref, atol=1e-12)
    Mt, _ = lab.course_arm_screws(**COURSE_ARM_GEOMETRY, tool=0.06)
    np.testing.assert_allclose(Mt[:3, 3], [0.206, 0.169, 0.0561], atol=1e-12)   # S3 needle guide


def test_course_arm_mr_example_4_5_pose():
    """Course arm at the pose of MR Example 4.5: p = (H2, W1, H1 + L1 + L2 + W2)."""
    M, S = lab.course_arm_screws(**COURSE_ARM_GEOMETRY)
    T = lab.fk_space(M, S, [0, -np.pi / 2, 0, 0, np.pi / 2, 0])
    np.testing.assert_allclose(T, [[0, -1, 0, 0.0751], [1, 0, 0, 0.0634], [0, 0, 1, 0.3828], [0, 0, 0, 1]],
                               atol=1e-12)


@pytest.mark.local_only  # needs files outside a single-lab repository; not graded
def test_course_arm_copy_is_identical_to_shared():
    # Instructor-side consistency check; skipped in the student copy of the lab.
    shared = HERE.parents[3] / "shared"
    if not (shared / "course_robots.py").exists():
        pytest.skip("shared model not available (student copy of the lab)")
    sys.path.insert(0, str(shared))
    cr = importlib.import_module("course_robots")
    for tool in (0.0, 0.06):
        r = cr.course_arm(tool=tool)
        M, S, q = course_arm_reference(tool)
        np.testing.assert_array_equal(S, r["Slist"])
        np.testing.assert_array_equal(M, r["M"])
        np.testing.assert_array_equal(q, r["axis_points"])
    assert (DATA / "course_arm.urdf").read_text() == (shared / "course_arm.urdf").read_text()


# ------------------------------------------------------------- Checkpoint 3: URDF -> PoE
def test_rpy_to_rot_matches_scipy():
    for _ in range(20):
        rpy = RNG.uniform(-np.pi, np.pi, 3)
        np.testing.assert_allclose(lab.rpy_to_rot(rpy), Rotation.from_euler("xyz", rpy).as_matrix(), atol=1e-12)
    np.testing.assert_allclose(lab.rpy_to_rot([0, np.pi / 2, 0]), [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], atol=1e-12)


def test_urdf_course_arm():
    joints = lab.read_urdf_joints(DATA / "course_arm.urdf")
    M, S, names, lims = lab.poe_from_urdf(joints, "tool0")
    M_ref, S_ref, _ = course_arm_reference()
    np.testing.assert_allclose(S, S_ref, atol=1e-9)
    np.testing.assert_allclose(M, M_ref, atol=1e-9)
    assert names == ["j1", "j2", "j3", "j4", "j5", "j6"]
    np.testing.assert_allclose(np.rad2deg(lims[:, 1]), [165] * 5 + [175], atol=1e-6)


def test_urdf_ur5_from_mr_text():
    """The UR5 URDF of MR Sec. 4.2 gives the screw table of MR Example 4.5 (full-precision values)."""
    joints = lab.read_urdf_joints(DATA / "ur5_mr.urdf")
    M, S, names, lims = lab.poe_from_urdf(joints, "ee_link")
    M_ref, S_ref = ur_type(**UR5_URDF)
    np.testing.assert_allclose(S, S_ref, atol=1e-8)
    np.testing.assert_allclose(M, M_ref, atol=1e-8)
    assert len(names) == 6 and np.all(np.isinf(lims))


def test_urdf_matches_pinocchio():
    pin = pytest.importorskip("pinocchio")
    model = pin.buildModelFromUrdf(str(DATA / "course_arm.urdf"))
    data = model.createData()
    fid = model.getFrameId("tool0")
    M, S, _, _ = lab.poe_from_urdf(lab.read_urdf_joints(DATA / "course_arm.urdf"), "tool0")
    for _ in range(5):
        q = RNG.uniform(-2.5, 2.5, 6)
        pin.framesForwardKinematics(model, data, q)
        np.testing.assert_allclose(lab.fk_space(M, S, q), data.oMf[fid].homogeneous, atol=1e-9)


# ------------------------------------------------------------- Checkpoint 4: simulation vs hardware
def test_servo_to_model():
    off = lab.load_offsets(DATA / "servo_offsets.json")
    np.testing.assert_allclose(off, [0, 90, 0, 90, 0, 0])
    q = lab.servo_to_model([0, 0, 0, 0, 0, 0], off)
    np.testing.assert_allclose(q, np.deg2rad([0, -90, 0, -90, 0, 0]), atol=1e-12)
    Q = lab.servo_to_model(np.zeros((3, 6)), off)
    assert Q.shape == (3, 6)


def test_servo_zero_is_candle_pose():
    """Servo zero (arm straight up): tool flange above joint 1 at H1 + L1 + L2 + H2 (illustrative calibration)."""
    M, S, _ = course_arm_reference()
    off = lab.load_offsets(DATA / "servo_offsets.json")
    p = lab.tool_points(M, S, lab.servo_to_model(np.zeros((1, 6)), off))[0]
    g = COURSE_ARM_GEOMETRY
    np.testing.assert_allclose(p, [0, g["W1"] + g["W2"], g["H1"] + g["L1"] + g["L2"] + g["H2"]], atol=1e-12)


def test_tool_points_matches_mr():
    M, S, _ = course_arm_reference()
    Q = RNG.uniform(-2, 2, (4, 6))
    P = lab.tool_points(M, S, Q, P_TOOL)
    for q, p in zip(Q, P):
        np.testing.assert_allclose(p, (mr.FKinSpace(M, S, q) @ np.r_[P_TOOL, 1])[:3], atol=1e-10)


def test_compare_log():
    log = lab.load_joint_log(DATA / "course_arm_log.csv")
    off = lab.load_offsets(DATA / "servo_offsets.json")
    M, S, _ = course_arm_reference()
    res = lab.compare_log(log, M, S, off, P_TOOL)
    k0 = int(np.argmax(log["stop"] > 0.5))
    assert res["n_samples"] == k0 == 900
    e = log["meas"][:k0] - log["cmd"][:k0]
    np.testing.assert_allclose(res["joint_rms_deg"], np.sqrt(np.mean(e ** 2, axis=0)), atol=1e-12)
    np.testing.assert_allclose(res["joint_mean_deg"][1], -0.50, atol=0.05)          # gravity droop of joint 2
    pb = np.r_[P_TOOL, 1]
    err = [1e3 * np.linalg.norm((mr.FKinSpace(M, S, np.deg2rad(m - off)) @ pb)[:3]
                                - (mr.FKinSpace(M, S, np.deg2rad(c - off)) @ pb)[:3])
           for c, m in zip(log["cmd"][:k0], log["meas"][:k0])]
    np.testing.assert_allclose(res["task_err_mm"], err, atol=1e-9)
    np.testing.assert_allclose(res["task_rms_mm"], np.sqrt(np.mean(np.square(err))), atol=1e-9)
    np.testing.assert_allclose(res["task_max_mm"], max(err), atol=1e-9)
    assert 3.0 < res["task_rms_mm"] < 4.5 and res["task_max_mm"] < 10.0


def test_tip_checks_and_joint3_offset():
    Qs, P = read_tip_checks()
    off = lab.load_offsets(DATA / "servo_offsets.json")
    M, S, _ = course_arm_reference()
    Q = lab.servo_to_model(Qs, off)
    e = lab.tip_errors_mm(M, S, Q, P, P_TOOL)
    pb = np.r_[P_TOOL, 1]
    e_ref = [np.linalg.norm(1e3 * (mr.FKinSpace(M, S, q) @ pb)[:3] - p) for q, p in zip(Q, P)]
    np.testing.assert_allclose(e, e_ref, atol=1e-9)
    assert 4.0 < np.sqrt(np.mean(e ** 2)) < 6.0
    fits = [lab.fit_joint_offset(M, S, Q, P, j, P_TOOL) for j in range(6)]
    best = int(np.argmin([f[1] for f in fits]))
    assert best == 2                                                    # joint 3 (0-based index 2)
    assert abs(fits[2][0] - 1.5) < 0.1 and fits[2][1] < 1.0
    assert sorted(f[1] for f in fits)[1] > 2.0                          # clearly better than any other joint
