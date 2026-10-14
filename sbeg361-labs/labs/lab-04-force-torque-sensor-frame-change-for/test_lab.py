"""Auto-checks for SBEG 361 Lab 4 -- exponential coordinates on SE(3), wrenches and F/T sensing.

Usage:   LAB_IMPL=starter  python -m pytest -q test_lab.py     (your work, file starter.py)
         LAB_IMPL=solution python -m pytest -q test_lab.py     (reference; default)
Ground truth: the official `modern_robotics` library (MatrixExp6, MatrixLog6, Adjoint, TransInv),
MR Example 3.26 (p. 105-106) and Example 3.28 (pp. 108-109), and the noise-free data model in
make_data.py.
"""
import importlib
import math
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
import make_data as md  # noqa: E402  (noise-free truth of the synthetic data)

S2 = 1 / math.sqrt(2)


def random_T(seed):
    rng = np.random.default_rng(seed)
    return mr.RpToTrans(Rotation.random(random_state=seed).as_matrix(), rng.normal(size=3))


def random_twist(seed):
    return np.random.default_rng(1000 + seed).normal(size=6)


T_TIP = mr.RpToTrans(np.eye(3), md.LAP_TIP)   # T_ft: tool tip {t} seen from the sensor {f}


def lap_calibration():
    _, Ts, Fs = lab.load_pose_wrench_csv(HERE / "lap_calib.csv")
    return lab.estimate_payload([T[:3, :3] for T in Ts], Fs)


# ===================================================== Checkpoint 1: exp and log on SE(3)
def test_matrix_exp6_matches_mr():
    for k in range(20):
        V = random_twist(k)
        se3 = mr.VecTose3(V)
        np.testing.assert_allclose(lab.matrix_exp6(se3), mr.MatrixExp6(se3), atol=1e-9)
    # pure translation, MR Eq. 3.89
    se3 = mr.VecTose3([0, 0, 0, 0.03, 0.04, 0])
    np.testing.assert_allclose(lab.matrix_exp6(se3), mr.RpToTrans(np.eye(3), [0.03, 0.04, 0]), atol=1e-12)


def test_exp6_screw_through_a_point():
    """Screw through a point (Week 4 revision): axis z through q = (1, 0, 0), theta = 90 deg, pitch 0 and 0.1 m/rad."""
    for h, pz in ((0.0, 0.0), (0.1, 0.1 * math.pi / 2)):
        S = np.r_[0, 0, 1, -np.cross([0, 0, 1], [1, 0, 0]) + h * np.array([0, 0, 1])]
        T = lab.matrix_exp6(mr.VecTose3(S) * math.pi / 2)
        np.testing.assert_allclose(T, [[0, -1, 0, 1], [1, 0, 0, -1], [0, 0, 1, pz], [0, 0, 0, 1]], atol=1e-12)


def test_matrix_log6_matches_mr_and_inverts_exp():
    for k in range(20):
        T = random_T(k)
        L = lab.matrix_log6(T)
        assert L.shape == (4, 4) and np.allclose(L[3], 0)
        np.testing.assert_allclose(L, mr.MatrixLog6(T), atol=1e-8)
        np.testing.assert_allclose(mr.MatrixExp6(L), T, atol=1e-9)


def test_matrix_log6_special_cases():
    np.testing.assert_allclose(lab.matrix_log6(np.eye(4)), np.zeros((4, 4)), atol=1e-12)
    T = mr.RpToTrans(np.eye(3), [0.4, 0, 0.3])                       # case (a): R = I
    np.testing.assert_allclose(lab.se3_to_vec(lab.matrix_log6(T)), [0, 0, 0, 0.4, 0, 0.3], atol=1e-12)
    for R in (np.diag([1.0, -1, -1]), np.diag([-1.0, 1, -1]), np.diag([-1.0, -1, 1])):   # theta = pi
        T = mr.RpToTrans(R, [0.1, 0.2, -0.3])
        L = lab.matrix_log6(T)
        assert abs(np.linalg.norm(lab.so3_to_vec(L[:3, :3])) - math.pi) < 1e-9
        np.testing.assert_allclose(mr.MatrixExp6(L), T, atol=1e-9)


def test_log6_mr_example_3_26():
    """MR Example 3.26 (pp. 104-106): S = (0, 0, 1, 3.37, -3.37, 0), theta = pi/6."""
    c30, s30, c60, s60 = math.cos(math.pi / 6), 0.5, 0.5, math.sin(math.pi / 3)
    Tsb = np.array([[c30, -s30, 0, 1], [s30, c30, 0, 2], [0, 0, 1, 0], [0, 0, 0, 1]])
    Tsc = np.array([[c60, -s60, 0, 2], [s60, c60, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
    S, theta = lab.screw_between(Tsb, Tsc, "space")
    assert abs(theta - math.pi / 6) < 1e-9
    np.testing.assert_allclose(S, [0, 0, 1, 3.3660, -3.3660, 0], atol=1e-4)


def test_screw_between_space_and_body():
    for k in range(10):
        T1, T2 = random_T(k), random_T(k + 50)
        Ss, ts = lab.screw_between(T1, T2, "space")
        Sb, tb = lab.screw_between(T1, T2, "body")
        assert abs(np.linalg.norm(Ss[:3]) - 1) < 1e-9 and abs(ts - tb) < 1e-9
        np.testing.assert_allclose(mr.MatrixExp6(mr.VecTose3(Ss) * ts) @ T1, T2, atol=1e-9)
        np.testing.assert_allclose(T1 @ mr.MatrixExp6(mr.VecTose3(Sb) * tb), T2, atol=1e-9)
        np.testing.assert_allclose(Sb, mr.Adjoint(mr.TransInv(T1)) @ Ss, atol=1e-9)   # S_b = [Ad_{T1^-1}] S_s
    S, th = lab.screw_between(np.eye(4), mr.RpToTrans(np.eye(3), [0, 0.3, 0.4]))
    np.testing.assert_allclose(S, [0, 0, 0, 0, 0.6, 0.8], atol=1e-12)
    assert abs(th - 0.5) < 1e-12
    with pytest.raises(ValueError):
        lab.screw_between(np.eye(4), np.eye(4), "world")


def test_screw_path_needle_guide():
    """Needle-guide screw path (Lab 3 / Week 4 revision): half-way along the screw from T1 to T2."""
    T1 = mr.RpToTrans(np.eye(3), [0.3, 0, 0.1])
    T2 = mr.RpToTrans(mr.MatrixExp3(mr.VecToso3([0, 0, math.pi / 2])), [0.2, 0.3, 0.15])
    np.testing.assert_allclose(lab.screw_path(T1, T2, 0.0), T1, atol=1e-12)
    np.testing.assert_allclose(lab.screw_path(T1, T2, 1.0), T2, atol=1e-12)
    Tm = lab.screw_path(T1, T2, 0.5)
    np.testing.assert_allclose(Tm[:3, 3], [0.1 + 0.3 * S2, 0.1 + 0.1 * S2, 0.125], atol=1e-9)


# ===================================================== Checkpoint 2: wrenches and the adjoint transpose
def test_point_force_wrench_and_power():
    F = lab.point_force_wrench([1, 0, -2], [0, 0, 0.3])
    np.testing.assert_allclose(F, [0, 0.3, 0, 1, 0, -2], atol=1e-12)
    assert isinstance(lab.power([1, 2, 3, 4, 5, 6], [1, 0, 0, 0, 0, 1]), float)
    assert lab.power([1, 2, 3, 4, 5, 6], [1, 0, 0, 0, 0, 1]) == 7.0


def test_change_wrench_frame_mr_example_3_28():
    """MR Example 3.28 (pp. 108-109): F_f = [Ad_Thf]^T F_h + [Ad_Taf]^T F_a = (0, 0, -0.75, 0, -6, 0)."""
    Thf = np.array([[1, 0, 0, -0.1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    Taf = np.array([[1, 0, 0, -0.25], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1.0]])
    Ff = lab.change_wrench_frame([0, 0, 0, 0, -5, 0], Thf) + lab.change_wrench_frame([0, 0, 0, 0, 0, 1], Taf)
    np.testing.assert_allclose(Ff, [0, 0, -0.75, 0, -6, 0], atol=1e-12)


def test_power_is_frame_independent():
    for k in range(10):
        T_ab = random_T(k)
        F_a, V_b = random_twist(k), random_twist(k + 30)
        F_b = lab.change_wrench_frame(F_a, T_ab)
        np.testing.assert_allclose(F_b, mr.Adjoint(T_ab).T @ F_a, atol=1e-12)
        V_a = mr.Adjoint(T_ab) @ V_b
        assert abs(lab.power(V_b, F_b) - lab.power(V_a, F_a)) < 1e-9         # MR Eq. 3.94
        np.testing.assert_allclose(lab.change_wrench_frame(F_b, mr.TransInv(T_ab)), F_a, atol=1e-9)


def test_wrench_to_screw():
    h, q, s, mag = lab.wrench_to_screw([0, 5.886, 0, 0, 0, -19.62])          # Workshop 4b
    assert abs(h) < 1e-12 and abs(mag - 19.62) < 1e-12
    np.testing.assert_allclose(q, [0.3, 0, 0], atol=1e-12)
    np.testing.assert_allclose(s, [0, 0, -1], atol=1e-12)
    for k in range(10):                                                     # recompose and invariance
        F = random_twist(k)
        h, q, s, mag = lab.wrench_to_screw(F)
        np.testing.assert_allclose(np.r_[np.cross(q, mag * s) + h * mag * s, mag * s], F, atol=1e-9)
        h2 = lab.wrench_to_screw(lab.change_wrench_frame(F, random_T(k)))[0]
        assert abs(h - h2) < 1e-9                                           # pitch is frame invariant
    h, q, s, mag = lab.wrench_to_screw([0, 0, 2, 0, 0, 0])
    assert h == math.inf and q is None and mag == 2
    with pytest.raises(ValueError):
        lab.wrench_to_screw(np.zeros(6))


def test_gravity_wrench_workshop_3():
    """Workshop 3a: 0.6 kg tool, CoM (0, 0, 0.10) in {f}, R_sf = Rot(x, 120 deg)."""
    R = mr.MatrixExp3(mr.VecToso3([2 * math.pi / 3, 0, 0]))
    F = lab.gravity_wrench(0.6, [0, 0, 0.10], R)
    np.testing.assert_allclose(F, [0.6 * 9.81 * math.sin(2 * math.pi / 3) * 0.1, 0, 0,
                                   0, -0.6 * 9.81 * math.sin(2 * math.pi / 3), 0.6 * 9.81 * 0.5], atol=1e-12)


# ===================================================== Checkpoint 3: laparoscope -- calibration and tip force
def test_estimate_payload_exact_data():
    rng = np.random.default_rng(7)
    m, c, b = 1.3, np.array([0.02, -0.01, 0.15]), rng.normal(size=6)
    Rs = [Rotation.random(random_state=s).as_matrix() for s in range(5)]
    Fs = [b + np.r_[np.cross(c, m * R.T @ md.G_S), m * R.T @ md.G_S] for R in Rs]
    m_e, c_e, b_e = lab.estimate_payload(Rs, Fs)
    assert abs(m_e - m) < 1e-9
    np.testing.assert_allclose(c_e, c, atol=1e-9)
    np.testing.assert_allclose(b_e, b, atol=1e-9)


def test_laparoscope_calibration():
    m, r, b = lap_calibration()
    assert abs(m - md.LAP_MASS) < 0.005                         # 0.600 kg
    np.testing.assert_allclose(r, md.LAP_COM, atol=0.002)       # (0, 0, 0.10) m
    np.testing.assert_allclose(b, md.LAP_BIAS, atol=0.01)


def test_compensate_removes_gravity_and_bias():
    m, r, b = lap_calibration()
    _, Ts, Fs = lab.load_pose_wrench_csv(HERE / "lap_calib.csv")
    for T, F in zip(Ts, Fs):
        Fc = lab.compensate(F, m, r, T[:3, :3], b)
        assert np.all(np.abs(Fc[3:]) < 0.08) and np.all(np.abs(Fc[:3]) < 0.006)


def test_tip_force_during_palpation():
    m, r, b = lap_calibration()
    t, Ts, Fs = lab.load_pose_wrench_csv(HERE / "lap_palpation.csv")
    Ft = lab.tip_forces(Ts, Fs, m, r, b, T_TIP)
    _, _, f_true, _ = md.palpation_truth()
    k = t < 7.0                                                  # presses 1 and 2 (no port force)
    rms = np.sqrt(np.mean(np.sum((Ft[k, 3:] - f_true[k]) ** 2, axis=1)))
    assert rms < 0.06
    peak = np.linalg.norm(Ft[k, 3:], axis=1).max()
    assert abs(peak - np.linalg.norm([0.3, -0.1, -2.0])) < 0.1   # 2.025 N


def test_port_force_detected_by_tip_moment():
    """A single point contact at the tip produces m_t = 0; the third press also has a port force."""
    m, r, b = lap_calibration()
    t, Ts, Fs = lab.load_pose_wrench_csv(HERE / "lap_palpation.csv")
    Ft = lab.tip_forces(Ts, Fs, m, r, b, T_TIP)
    F = md.press_profile(t)
    mt = np.linalg.norm(Ft[:, :3], axis=1)
    assert mt[(t < 7) & (F > 1.5)].mean() < 0.015
    assert mt[(t >= 7) & (F > 1.5)].mean() > 0.04


# ===================================================== Checkpoint 4: EOD gripper and package
def eod_estimates():
    d = lab.load_json(HERE / "eod_lift.json")
    Rs = [np.array(x["T_sf"])[:3, :3] for x in d["empty"]]
    mg, rg, bg = lab.estimate_payload(Rs, [x["F"] for x in d["empty"]])
    mt, rt, bt = lab.estimate_payload(Rs, [x["F"] for x in d["loaded"]])
    return d, (mg, rg, bg), (mt, rt, bt)


def test_eod_gripper_and_package_identification():
    _, (mg, rg, _), (mt, rt, _) = eod_estimates()
    assert abs(mg - md.GRIPPER_MASS) < 0.02
    np.testing.assert_allclose(rg, md.GRIPPER_COM, atol=0.003)
    mp = mt - mg
    rp = (mt * rt - mg * rg) / mp                                # composite centre of mass
    assert abs(mp - md.PACKAGE_MASS) < 0.02
    np.testing.assert_allclose(rp, md.PACKAGE_COM, atol=0.005)


def test_eod_wrench_at_turret_and_front_edge():
    d, (mg, rg, _), (mt, rt, bt) = eod_estimates()
    T_sf = np.array(d["lift"]["T_sf"])
    Fp = lab.compensate(d["lift"]["F"], mg, rg, T_sf[:3, :3], bt)  # package wrench in {f}
    F_s = lab.change_wrench_frame(Fp, mr.TransInv(T_sf))        # F_s = [Ad_{T_fs}]^T F_f
    exact = mr.Adjoint(mr.TransInv(T_sf)).T @ md.gravity(md.PACKAGE_MASS, md.PACKAGE_COM, T_sf[:3, :3])
    np.testing.assert_allclose(F_s, exact, atol=0.4)
    assert abs(F_s[1] - 37.1) < 0.5                             # pitching moment at the turret base
    F_e = lab.change_wrench_frame(F_s, np.array(d["front_edge"]["T_se"]))
    assert abs(F_e[1] - 26.3) < 0.5                             # about the front track edge


def test_eod_power_duality():
    d, (mg, rg, _), (mt, rt, bt) = eod_estimates()
    T_sf = np.array(d["lift"]["T_sf"])
    Fp = lab.compensate(d["lift"]["F"], mg, rg, T_sf[:3, :3], bt)
    V_f = np.array(d["lift"]["V_f"])
    P_f = lab.power(V_f, Fp)
    P_s = lab.power(mr.Adjoint(T_sf) @ V_f, lab.change_wrench_frame(Fp, mr.TransInv(T_sf)))
    assert abs(P_f - P_s) < 1e-9
    assert -7.3 < P_f < -6.8                                    # gravity absorbs about 7 W while lifting
