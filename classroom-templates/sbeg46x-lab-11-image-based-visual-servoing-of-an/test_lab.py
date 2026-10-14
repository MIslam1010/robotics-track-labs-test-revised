"""Auto-checks for SBEG 46X Lab 11 (image-based visual servoing of an endoscope-holding arm, simulated pinhole camera).

    LAB_IMPL=starter    python -m pytest -q test_lab.py   # untouched starter: every test that calls your code fails
                                                          # with NotImplementedError; the data test passes
    LAB_IMPL=lab11_<ID> python -m pytest -q test_lab.py   # your renamed file

LAB_IMPL is the name of the module to test; it is imported from this folder.
Ground truth: the worked numbers of Corke, Robotics, Vision and Control (Sec. 11.1, pp. 254-258, and Sec. 15.2.1,
pp. 460-462), the camera matrix K [R | p], central finite differences of the projection under the matrix exponential
of the camera twist (scipy.linalg.expm), modern_robotics (MatrixExp6, MatrixLog6, JacobianBody, FKinBody), the
course model ur5_laparoscope (course_models.py), closed-form cases and the reference values in data/expected.json.
"""
import importlib
import json
import math
import os
import sys
from pathlib import Path

import modern_robotics as mr
import numpy as np
import pytest
from scipy.linalg import expm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
lab = importlib.import_module(os.environ.get("LAB_IMPL", "starter"))
import course_models as cm  # noqa: E402

SC = json.loads((HERE / "data" / "scene.json").read_text())
EXP = json.loads((HERE / "data" / "expected.json").read_text())
RNG = np.random.default_rng(1111)
CAM_I, CAM_E = SC["inspection_camera"], SC["endoscope_camera"]
K_I = np.array([[800.0, 0, 512], [0, 800.0, 512], [0, 0, 1]])        # Corke's default camera
K_E = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])        # endoscope camera
IB = SC["ibvs"]


def rand_pose(rng, centre=(0.0, 0.0, 0.0), spread=0.3, ang=0.4):
    """Random camera pose near `centre` with a rotation of at most about `ang` rad (draws 6 numbers)."""
    w = rng.uniform(-ang, ang, 3)
    p = np.asarray(centre, float) + rng.uniform(-spread, spread, 3)
    return mr.RpToTrans(mr.MatrixExp3(mr.VecToso3(w)), p)


def pix(K, Pc):
    """Pixels of camera-frame points (N, 3) with the camera matrix K [I | 0]."""
    q = (K @ (Pc / Pc[:, 2:3]).T).T
    return q[:, :2]


def scene_setup():
    P, xy = lab.target_points(SC)
    Tg = lab.pose_from_spec(IB["T_wc_star"])
    T_cst = mr.TransInv(Tg) @ lab.pose_from_spec(SC["target"]["T_wt"])
    return P, xy, Tg, np.array(EXP["p_star"]), T_cst


# ------------------------------------------------------------------------------ data (provided code only)
def test_data_files_consistent():
    P, p = lab.load_calib()
    assert P.shape == (EXP["calib"]["n_points"], 3) and p.shape == (EXP["calib"]["n_points"], 2)
    assert np.all((p[:, 0] > 0) & (p[:, 0] < CAM_E["W"]) & (p[:, 1] > 0) & (p[:, 1] < CAM_E["H"]))
    Pt, xy = lab.target_points(SC)
    assert Pt.shape == (4, 3) and np.allclose(Pt[:, 2], 3.0) and np.isclose(np.ptp(xy[:, 0]), SC["target"]["side"])
    T = np.array(SC["calib"]["T_wc"])
    assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(T[:3, :3]), 1)


# ------------------------------------------------------------------------------ Part A: camera and calibration
def test_intrinsic_matrix():
    K = lab.intrinsic_matrix(0.015, 10e-6, 10e-6, 640, 512)             # Corke p. 257 (cam.C)
    np.testing.assert_allclose(K, [[1500, 0, 640], [0, 1500, 512], [0, 0, 1]], atol=1e-9)
    np.testing.assert_allclose(lab.camera_K(CAM_E), K_E, atol=1e-9)
    np.testing.assert_allclose(lab.intrinsic_matrix(0.004, 5e-6, 4e-6, 100, 50), [[800, 0, 100], [0, 1000, 50], [0, 0, 1]],
                               atol=1e-9)


def test_project_points_corke_examples():
    K = np.array([[1500.0, 0, 640], [0, 1500.0, 512], [0, 0, 1]])
    p, Z = lab.project_points(K, np.eye(4), [0.3, 0.4, 3.0])           # Corke p. 256: (790, 712)
    assert p.shape == (1, 2) and Z.shape == (1,)
    np.testing.assert_allclose(p, [[790, 712]], atol=1e-9)
    np.testing.assert_allclose(Z, [3.0], atol=1e-12)
    p, _ = lab.project_points(K, lab.trans([-0.5, 0, 0]), [0.3, 0.4, 3.0])   # camera 0.5 m to the left (p. 255)
    np.testing.assert_allclose(p, [[1040, 712]], atol=1e-9)
    grid = np.array([[x, y, 1.0] for x in (-0.1, 0, 0.1) for y in (-0.1, 0, 0.1)])   # p. 258, mkgrid(3, 0.2)
    p, _ = lab.project_points(K, np.eye(4), grid)
    np.testing.assert_allclose(p[:, 0], [490] * 3 + [640] * 3 + [790] * 3, atol=1e-9)
    np.testing.assert_allclose(p[:, 1], [362, 512, 662] * 3, atol=1e-9)
    p0, _ = lab.project_points(K_I, np.eye(4), [1, 1, 5])                # Corke p. 460: (672, 672) and (656, 672)
    px, _ = lab.project_points(K_I, lab.trans([0.1, 0, 0]), [1, 1, 5])
    np.testing.assert_allclose(np.r_[p0[0], px[0]], [672, 672, 656, 672], atol=1e-9)


def test_project_points_rejects_points_behind():
    with pytest.raises(ValueError):
        lab.project_points(K_I, np.eye(4), [[0, 0, 1.0], [0.2, 0.1, -0.5]])
    with pytest.raises(ValueError):
        lab.project_points(K_I, lab.trans([0, 0, 2.0]), [0.1, 0.1, 1.0])


def test_project_points_matches_camera_matrix():
    for _ in range(5):
        T = rand_pose(RNG)
        P = RNG.uniform(-0.5, 0.5, (6, 3)) + [0, 0, 3.0]
        p, Z = lab.project_points(K_I, T, P)
        Pc = (mr.TransInv(T) @ np.c_[P, np.ones(6)].T).T[:, :3]
        np.testing.assert_allclose(p, pix(K_I, Pc), atol=1e-9)
        np.testing.assert_allclose(Z, Pc[:, 2], atol=1e-12)


def test_camera_matrix_dlt_exact_data():
    P, _ = lab.load_calib()
    T = np.array(SC["calib"]["T_wc"])
    C_true = K_E @ mr.TransInv(T)[:3, :]
    p = pix(K_E, (mr.TransInv(T) @ np.c_[P, np.ones(len(P))].T).T[:, :3])
    C = lab.camera_matrix_dlt(P, p)
    assert C.shape == (3, 4)
    np.testing.assert_allclose(C, C_true, atol=1e-7 * np.abs(C_true).max())
    K, T_cw = lab.decompose_camera_matrix(C)
    np.testing.assert_allclose(K, K_E, atol=1e-6)
    np.testing.assert_allclose(T_cw, mr.TransInv(T), atol=1e-9)
    assert lab.reprojection_rms(C, P, p) < 1e-6
    with pytest.raises(ValueError):
        lab.camera_matrix_dlt(P[:5], p[:5])


def test_camera_matrix_dlt_noisy_data_and_reprojection():
    P, p = lab.load_calib()
    C = lab.camera_matrix_dlt(P, p)
    E = EXP["calib"]
    np.testing.assert_allclose(C, E["C"], atol=1e-8 * np.abs(np.array(E["C"])).max())
    K, _ = lab.decompose_camera_matrix(C)
    np.testing.assert_allclose(K, E["K"], atol=1e-6)
    assert lab.reprojection_rms(C, P, p) == pytest.approx(E["rms"])
    i6 = E["idx6"]
    assert lab.reprojection_rms(lab.camera_matrix_dlt(P[i6], p[i6]), P, p) == pytest.approx(E["rms6_all"])


def test_reprojection_rms_hand_value():
    C = np.c_[K_E, np.zeros(3)]
    P = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]])                    # pixels (320, 240) and (380, 240)
    p = np.array([[323.0, 244.0], [380.0, 240.0]])                      # errors 5 px and 0 px
    assert lab.reprojection_rms(C, P, p) == pytest.approx(math.sqrt(12.5))
    assert lab.reprojection_rms(-2.0 * C, P, p) == pytest.approx(math.sqrt(12.5))     # scale-free


# ------------------------------------------------------------------------------ Part B: interaction matrix
def test_interaction_matrix_corke_values():
    L = lab.interaction_matrix([[672, 672]], 5.0, K_I)                  # Corke p. 462, columns in MR order
    np.testing.assert_allclose(L, [[32, -832, 160, -160, 0, 32], [832, -32, -160, 0, -160, 32]], atol=1e-9)
    L2 = lab.interaction_matrix(np.array([[672, 672], [512, 512]]), np.array([5.0, 1.0]), K_I)
    assert L2.shape == (4, 6)
    np.testing.assert_allclose(L2[2:], [[0, -800, 0, -800, 0, 0], [800, 0, 0, 0, -800, 0]], atol=1e-9)


def test_interaction_matrix_finite_differences():
    eps = 1e-6
    for _ in range(5):
        T = rand_pose(RNG)
        P = RNG.uniform(-0.5, 0.5, (3, 3)) + [0, 0, 3.0]
        V = RNG.normal(0, 1, 6)
        Pc = (mr.TransInv(T) @ np.c_[P, np.ones(3)].T).T[:, :3]
        p, Z = pix(K_I, Pc), Pc[:, 2]

        def img(s):
            Tn = T @ expm(mr.VecTose3(V * s))
            return pix(K_I, (mr.TransInv(Tn) @ np.c_[P, np.ones(3)].T).T[:, :3]).reshape(-1)
        fd = (img(eps) - img(-eps)) / (2 * eps)
        np.testing.assert_allclose(lab.interaction_matrix(p, Z, K_I) @ V, fd, atol=1e-4)


def test_interaction_matrix_rank_and_null_space():
    L = lab.interaction_matrix([[512, 512]], 2.0, K_I)                   # point on the optical axis
    assert np.linalg.matrix_rank(L) == 2
    for V in ([0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 1, 0, -2.0, 0, 0], [1, 0, 0, 0, 2.0, 0]):
        np.testing.assert_allclose(L @ np.array(V, float), 0, atol=1e-9)  # roll, along the ray, Tx/Ry and Ty/Rx pairs
    p3 = np.array([[400, 300], [700, 450], [480, 720]])
    assert np.linalg.matrix_rank(lab.interaction_matrix(p3, 2.0, K_I)) == 6
    pc = np.array([[300, 300], [500, 500], [700, 700]])                  # collinear: singular
    assert np.linalg.matrix_rank(lab.interaction_matrix(pc, 2.0, K_I), tol=1e-6) < 6


# ------------------------------------------------------------------------------ Part C: IBVS
def test_ibvs_velocity_law():
    p3 = np.array([[400.0, 300], [700, 450], [480, 720]])
    ps3 = np.array([[420.0, 330], [690, 430], [500, 700]])
    Z = np.array([2.0, 2.5, 1.8])
    V = lab.ibvs_velocity(p3, ps3, Z, K_I, 0.7)
    assert V.shape == (6,)
    np.testing.assert_allclose(lab.interaction_matrix(p3, Z, K_I) @ V, -0.7 * (p3 - ps3).reshape(-1), atol=1e-8)
    np.testing.assert_allclose(lab.ibvs_velocity(p3, p3, Z, K_I, 0.7), 0, atol=1e-12)
    P, xy, Tg, ps, _ = scene_setup()
    p, Z = lab.project_points(K_I, lab.pose_from_spec(SC["start_poses"]["corke"]), P)
    L = lab.interaction_matrix(p, Z, K_I)
    np.testing.assert_allclose(lab.ibvs_velocity(p, ps, Z, K_I, 1.0), -np.linalg.pinv(L) @ (p - ps).reshape(-1),
                               atol=1e-9)


def test_camera_step_matches_expm():
    T = lab.pose_from_spec(SC["start_poses"]["corke"])
    V = np.array([0.3, -0.2, 0.5, 1.0, -0.4, 0.8])
    np.testing.assert_allclose(lab.camera_step(T, V, 0.1), T @ expm(mr.VecTose3(V * 0.1)), atol=1e-12)
    np.testing.assert_allclose(lab.camera_step(T, np.zeros(6), 0.1), T, atol=1e-15)


def test_run_ibvs_corke_example():
    P, xy, Tg, ps, _ = scene_setup()
    T0 = lab.pose_from_spec(SC["start_poses"]["corke"])
    r = lab.run_ibvs(T0, P, ps, K_I, IB["lam"], IB["h"], n_max=IB["n_max"], tol=IB["tol"])
    E = EXP["ibvs_depth"][0]
    n = r["n_iter"]
    assert r["converged"] and abs(n - E["n_iter"]) <= 1
    assert r["T"].shape == (n + 1, 4, 4) and r["p"].shape == (n + 1, 4, 2) and r["Z"].shape == (n + 1, 4)
    assert r["err"].shape == (n + 1,) and r["V"].shape == (n, 6) and r["cond"].shape == (n,)
    assert r["err"][-1] <= IB["tol"] < r["err"][-2]
    np.testing.assert_allclose(r["T"][0], T0, atol=1e-15)
    assert np.linalg.norm(r["T"][-1][:3, 3] - Tg[:3, 3]) < 3e-3
    np.testing.assert_allclose(r["T"][-1][:3, :3], np.eye(3), atol=3e-3)
    m = lab.servo_metrics(r, CAM_I)
    assert m["straightness"] == pytest.approx(E["straightness"], rel=1e-4)
    assert r["cond"][0] == pytest.approx(E["cond0"], rel=1e-6)


def test_run_ibvs_depth_estimates():
    P, xy, Tg, ps, _ = scene_setup()
    T0 = lab.pose_from_spec(SC["start_poses"]["corke"])
    for E in EXP["ibvs_depth"][1:]:
        r = lab.run_ibvs(T0, P, ps, K_I, IB["lam"], IB["h"], Z_hat=E["Z_hat"], n_max=IB["n_max"], tol=IB["tol"])
        assert r["converged"] and abs(r["n_iter"] - E["n_iter"]) <= 1, E["Z_hat"]
        assert lab.servo_metrics(r, CAM_I)["straightness"] == pytest.approx(E["straightness"], rel=1e-3)


def test_run_ibvs_stops_and_model_errors():
    P, xy, Tg, ps, _ = scene_setup()
    T0 = lab.pose_from_spec(SC["start_poses"]["corke"])
    r = lab.run_ibvs(T0, P, ps, K_I, IB["lam"], IB["h"], n_max=5, tol=IB["tol"])
    assert not r["converged"] and r["n_iter"] == 5 and r["T"].shape == (6, 4, 4) and r["V"].shape == (5, 6)
    r0 = lab.run_ibvs(Tg, P, ps, K_I, IB["lam"], IB["h"], n_max=5, tol=IB["tol"])
    assert r0["converged"] and r0["n_iter"] == 0 and r0["V"].shape == (0, 6)
    Kb = np.array(EXP["k_bad"]["K_hat"])
    rb = lab.run_ibvs(T0, P, ps, K_I, IB["lam"], IB["h"], n_max=IB["n_max"], tol=IB["tol"], K_hat=Kb)
    assert rb["converged"] and np.linalg.norm(rb["T"][-1][:3, 3] - Tg[:3, 3]) < 3e-3


# ------------------------------------------------------------------------------ Part D: pose and PBVS
def test_homography_dlt_exact():
    P, xy, Tg, ps, T_cst = scene_setup()
    T = lab.pose_from_spec(SC["start_poses"]["corke"])
    T_ct = mr.TransInv(T) @ lab.pose_from_spec(SC["target"]["T_wt"])
    p = pix(K_I, (T_ct @ np.c_[xy, np.zeros(4), np.ones(4)].T).T[:, :3])
    H = lab.homography_dlt(xy, p)
    H_true = K_I @ np.c_[T_ct[:3, 0], T_ct[:3, 1], T_ct[:3, 3]]
    np.testing.assert_allclose(H, H_true / H_true[2, 2], rtol=1e-8, atol=1e-8)
    with pytest.raises(ValueError):
        lab.homography_dlt(xy[:3], p[:3])


def test_pose_from_homography_recovers_pose():
    xy = np.array([[-0.25, -0.25], [-0.25, 0.25], [0.25, 0.25], [0.25, -0.25], [0.1, 0.0]])
    for _ in range(5):
        T_ct = rand_pose(RNG, centre=(0, 0, 2.0), spread=0.4, ang=0.6)
        p = pix(K_I, (T_ct @ np.c_[xy, np.zeros(5), np.ones(5)].T).T[:, :3])
        np.testing.assert_allclose(lab.pose_from_homography(lab.homography_dlt(xy, p), K_I), T_ct, atol=1e-8)
    H = K_I @ np.c_[np.eye(3)[:, :2], [0.1, 0.2, 2.0]]
    np.testing.assert_allclose(lab.pose_from_homography(-3.0 * H, K_I), lab.trans([0.1, 0.2, 2.0]), atol=1e-12)


def test_pbvs_velocity_one_unit_step():
    P, xy, Tg, ps, T_cst = scene_setup()
    for T in (lab.pose_from_spec(SC["start_poses"]["corke"]), Tg @ lab.rotz(2.0) @ lab.trans([0.2, -0.1, 0.3])):
        T_ct = mr.TransInv(T) @ lab.pose_from_spec(SC["target"]["T_wt"])
        V = lab.pbvs_velocity(T_ct, T_cst, 0.5)
        assert V.shape == (6,)
        np.testing.assert_allclose(T @ expm(mr.VecTose3(V * 2.0)), Tg, atol=1e-9)    # 1/lambda = 2 s
    np.testing.assert_allclose(lab.pbvs_velocity(T_cst, T_cst, 0.5), 0, atol=1e-12)


def test_run_pbvs_and_field_of_view():
    P, xy, Tg, ps, T_cst = scene_setup()
    T0 = lab.pose_from_spec(SC["start_poses"]["corke"])
    r = lab.run_pbvs(T0, P, xy, T_cst, ps, K_I, IB["lam"], IB["h"], IB["n_max"], IB["tol"])
    n = r["n_iter"]
    assert r["converged"] and abs(n - EXP["pbvs_corke"]["n_iter"]) <= 1
    assert r["T"].shape == (n + 1, 4, 4) and r["V"].shape == (n, 6) and r["err"][-1] <= IB["tol"]
    Tf = lab.pose_from_spec(SC["start_poses"]["fov"])
    mi = lab.servo_metrics(lab.run_ibvs(Tf, P, ps, K_I, IB["lam"], IB["h"], n_max=IB["n_max"], tol=IB["tol"]), CAM_I)
    mp = lab.servo_metrics(lab.run_pbvs(Tf, P, xy, T_cst, ps, K_I, IB["lam"], IB["h"], IB["n_max"], IB["tol"]), CAM_I)
    assert mi["min_margin"] == pytest.approx(EXP["fov"]["ibvs"]["min_margin"], rel=1e-4) and mi["min_margin"] > 0
    assert mp["min_margin"] == pytest.approx(EXP["fov"]["pbvs"]["min_margin"], rel=1e-4) and mp["min_margin"] < 0


def test_pbvs_calibration_error():
    P, xy, Tg, ps, T_cst = scene_setup()
    T0 = lab.pose_from_spec(SC["start_poses"]["corke"])
    E = EXP["k_bad"]["pbvs"]
    r = lab.run_pbvs(T0, P, xy, T_cst, ps, K_I, IB["lam"], IB["h"], IB["n_max"], IB["tol"],
                     K_hat=np.array(EXP["k_bad"]["K_hat"]))
    assert not r["converged"] and r["n_iter"] == IB["n_max"]
    assert r["err"][-1] == pytest.approx(E["final_err"], rel=1e-6)
    np.testing.assert_allclose(r["T"][-1][:3, 3], E["final_pos"], atol=1e-6)


# ------------------------------------------------------------------------------ Part E: endoscope through a port
def test_rcm_twist_basis_keeps_port_still():
    for _ in range(5):
        ell = RNG.uniform(0.02, 0.2)
        xi = RNG.normal(0, 1, 4)
        W = lab.rcm_twist_basis(ell)
        assert W.shape == (6, 4) and np.linalg.matrix_rank(W) == 4
        V = W @ xi
        r = np.array([0.0, 0.0, -ell])
        v_port = V[3:] + np.cross(V[:3], r)                  # velocity of the scope point that is in the port
        np.testing.assert_allclose(v_port, [0.0, 0.0, xi[3]], atol=1e-12)        # along the scope axis only
        np.testing.assert_allclose(V[:3], xi[:3], atol=1e-15)
        assert V[5] == pytest.approx(xi[3])


def test_rcm_ibvs_velocity_law():
    p, ps = np.array([[430.0, 180.0]]), np.array([[320.0, 240.0]])
    for d_dot in (0.0, 0.01):
        xi, V = lab.rcm_ibvs_velocity(p, ps, 0.08, K_E, 2.0, 0.1, d_dot)
        assert xi.shape == (4,) and V.shape == (6,)
        assert xi[2] == 0.0 and xi[3] == pytest.approx(d_dot)
        np.testing.assert_allclose(V, lab.rcm_twist_basis(0.1) @ xi, atol=1e-15)
        np.testing.assert_allclose(lab.interaction_matrix(p, 0.08, K_E) @ V, -2.0 * (p - ps).reshape(-1), atol=1e-8)
    xi, V = lab.rcm_ibvs_velocity(ps, ps, 0.08, K_E, 2.0, 0.1)
    np.testing.assert_allclose(V, 0, atol=1e-15)


def test_joint_rates():
    rob = cm.ur5_laparoscope()
    for _ in range(3):
        th = rob["q_home"] + RNG.uniform(-0.2, 0.2, 6)
        V = RNG.normal(0, 0.05, 6)
        Jb = mr.JacobianBody(rob["Blist"], th)
        qd = lab.joint_rates(rob["Blist"], th, V)
        np.testing.assert_allclose(Jb @ qd, V, atol=1e-10)
        qs = lab.joint_rates(rob["Blist"], th, 50 * V, 0.5)
        assert np.max(np.abs(qs)) == pytest.approx(0.5)
        np.testing.assert_allclose(qs * np.max(np.abs(50 * qd)) / 0.5, 50 * qd, atol=1e-9)
        np.testing.assert_allclose(lab.joint_rates(rob["Blist"], th, V, 10.0), qd, atol=1e-12)


def test_rcm_errors():
    st = lab.endoscope_setup(SC)
    ell, e = lab.rcm_errors(st["T0"], st["port"])
    assert ell == pytest.approx(SC["endoscope"]["ell0"]) and np.linalg.norm(e) < 1e-12 and e.shape == (2,)
    T = st["T0"] @ lab.trans([0.002, -0.001, 0.01])                       # move 2 mm right, 1 mm up, 1 cm deeper
    ell, e = lab.rcm_errors(T, st["port"])
    assert ell == pytest.approx(0.11) and np.allclose(e, [-0.002, 0.001], atol=1e-12)


@pytest.mark.parametrize("case", ["default", "delay_unstable"])
def test_endoscope_closed_loop(case):
    if case == "default":
        E, kw, rel = EXP["endoscope"]["default"], {}, 1e-6
    else:
        E = EXP["endoscope"]["delay"][2]
        kw, rel = dict(lam=E["lam"], delay=E["delay"]), 1e-3
    m = lab.endoscope_metrics(lab.simulate_endoscope(SC, **kw))
    for k in ("rms_err", "max_err", "max_e_lat", "ell_min", "peak_qdot"):
        assert m[k] == pytest.approx(E[k], rel=rel), k
    if case == "default":
        assert m["final_still_err"] < 1.0 and m["view_frac"] == 1.0 and m["sat_frac"] == 0.0
    else:
        assert m["sat_frac"] > 0.3


def test_endoscope_rcm_drift_correction():
    m0 = lab.endoscope_metrics(lab.simulate_endoscope(SC))
    m1 = lab.endoscope_metrics(lab.simulate_endoscope(SC, k_rcm=5.0))
    assert m1["max_e_lat"] < 0.2 * m0["max_e_lat"] and m0["max_e_lat"] < 1e-3
    assert m1["max_e_lat"] == pytest.approx(EXP["endoscope"]["k_rcm"]["max_e_lat"], rel=1e-6)
    assert m1["rms_err"] == pytest.approx(m0["rms_err"], rel=0.02)
