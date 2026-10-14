"""Auto-checks for SBEG 46X Lab 9 (IMU orientation fusion for an exoskeleton thigh segment: complementary vs EKF).

    LAB_IMPL=starter  python -m pytest -q test_lab.py    # untouched starter: every test that calls your code
                                                         # fails with NotImplementedError; the 4 data tests pass
    LAB_IMPL=lab9_<ID> python -m pytest -q test_lab.py   # your renamed file

LAB_IMPL is the name of the module to test; it is imported from this folder.
Ground truth: the thigh-IMU pose and ideal accelerometer of the shared course model gait_exo_2r (modern_robotics
FKinSpace via course_models), rotation matrices from modern_robotics MatrixExp3, closed-form responses of first-order
filters, the information form of the Kalman update, the Astrom-Murray predictor form (Eq. 7.22), finite-difference
Jacobians, the hip-encoder reference of the recording and the reference results in data/expected.json.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
lab = importlib.import_module(os.environ.get("LAB_IMPL", "starter"))
import course_models as cm  # noqa: E402

META = json.loads((HERE / "data" / "imu_meta.json").read_text())
EXP = json.loads((HERE / "data" / "expected.json").read_text())
REC = lab.load_recording()
T, REF = REC["t"], REC["theta_enc"]
RNG = np.random.default_rng(910)
G = 9.81


def rot_from_axis_angle(w_hat, phi):
    return mr.MatrixExp3(mr.VecToso3(np.asarray(w_hat, float) * phi))


def quat_from_axis_angle(w_hat, phi):
    w_hat = np.asarray(w_hat, float) / np.linalg.norm(w_hat)
    return np.r_[math.cos(phi / 2), w_hat * math.sin(phi / 2)]          # MR Eq. B.9


def random_unit(n=3):
    v = RNG.standard_normal(n)
    return v / np.linalg.norm(v)


# ------------------------------------------------------------------------------ data and provided code (pass)
def test_recording_format():
    assert set(REC) == {"t", "gx", "gy", "gz", "ax", "ay", "az", "theta_enc"}
    assert len(T) == EXP["N"] and np.allclose(np.diff(T), META["h"], atol=1e-9)
    assert META["r_imu"] == cm.R_IMU


def test_course_model_imu_pose():
    """T_sb from the PoE of gait_exo_2r equals Rot(z, theta) at p = r (sin theta, -cos theta, 0)."""
    for th in RNG.uniform(-0.5, 1.2, 5):
        T_sb = cm.imu_pose(th)
        R, p = mr.TransToRp(T_sb)
        c, s = math.cos(th), math.sin(th)
        assert np.allclose(R, [[c, -s, 0], [s, c, 0], [0, 0, 1]], atol=1e-12)
        assert np.allclose(p, cm.R_IMU * np.array([math.sin(th), -math.cos(th), 0.0]), atol=1e-12)


def test_recording_at_rest_matches_model():
    """While the leg hangs (t < 8 s) the accelerometer mean equals the ideal reading of the course model."""
    rest = T < 8.0
    f_model = cm.imu_specific_force(0.0, 0.0, 0.0)
    f_meas = np.array([REC["ax"][rest].mean(), REC["ay"][rest].mean(), REC["az"][rest].mean()])
    assert np.allclose(f_meas, f_model, atol=0.02)
    assert np.allclose(f_model, [0.0, G, 0.0], atol=1e-12)


def test_provided_quat_from_accel():
    q = lab.quat_from_accel([G * math.sin(0.3), G * math.cos(0.3), 0.0])
    R = rot_from_axis_angle(q[1:] / np.linalg.norm(q[1:]), 2 * math.acos(q[0]))
    assert np.allclose(R.T @ [0, 1, 0], [math.sin(0.3), math.cos(0.3), 0.0], atol=1e-12)


# ------------------------------------------------------------------------------ Part A
def test_A1_accel_tilt_inverts_the_course_model():
    th = RNG.uniform(-0.5, 1.2, 8)
    f = np.array([cm.imu_specific_force(x, 0.0, 0.0) for x in th])
    assert np.allclose(lab.accel_tilt(f[:, 0], f[:, 1]), th, atol=1e-12)
    assert np.allclose(lab.accel_tilt(3 * f[:, 0], 3 * f[:, 1]), th, atol=1e-12)    # scale-free


def test_A1_accel_tilt_error_during_swing():
    """Dynamic acceleration (r theta_dot^2, r theta_ddot) biases the accelerometer tilt: Workshop Problem 3b."""
    f = cm.imu_specific_force(math.radians(20.0), 3.0, 20.0)
    assert abs(math.degrees(lab.accel_tilt(f[0], f[1])) - 33.72) < 0.01


def test_A2_gyro_integrate_forward_euler():
    t = np.arange(0, 1.0, 0.01)
    w = RNG.standard_normal(len(t))
    out = lab.gyro_integrate(t, w, theta0=0.2, bias=0.1)
    ref = [0.2]
    for k in range(1, len(t)):
        ref.append(ref[-1] + 0.01 * (w[k] - 0.1))
    assert np.allclose(out, ref, atol=1e-12)
    # constant rate: exact ramp
    assert np.allclose(lab.gyro_integrate(t, np.full(len(t), 0.5)), 0.5 * t, atol=1e-12)


def test_A3_cf_alpha():
    assert abs(lab.cf_alpha(0.5, 0.01) - 0.5 / 0.51) < 1e-15
    assert abs(lab.cf_alpha(0.0, 0.01)) < 1e-15


def test_A4_complementary_step_and_bias_responses():
    h, tau = 0.01, 0.5
    t = np.arange(0, 10.0, h)
    a = tau / (tau + h)
    # accelerometer step, gyro silent: theta_k = 1 - a^k  (first-order low-pass of the accelerometer)
    th = lab.complementary_filter(t, np.zeros(len(t)), np.ones(len(t)), tau=tau, theta0=0.0)
    assert np.allclose(th, 1 - a ** np.arange(len(t)), atol=1e-12)
    # constant gyro bias b, segment still: steady error a h b / (1 - a) = tau b
    b = 0.02
    th = lab.complementary_filter(t, np.full(len(t), b), np.zeros(len(t)), tau=tau, theta0=0.0)
    assert abs(th[-1] - tau * b) < 1e-6
    # and the bias argument removes it
    th = lab.complementary_filter(t, np.full(len(t), b), np.zeros(len(t)), tau=tau, theta0=0.0, bias=b)
    assert np.allclose(th, 0.0, atol=1e-15)


def test_A4_complementary_on_recording():
    th_acc = lab.accel_tilt(REC["ax"], REC["ay"])
    th = lab.complementary_filter(T, REC["gz"], th_acc, tau=0.5)
    assert abs(lab.rms_deg(th, REF) - EXP["rms_deg"]["cf_tau05"]) < 1e-6
    assert lab.rms_deg(th, REF) < 0.3 * lab.rms_deg(th_acc, REF)


def test_A5_bias_at_rest():
    b = lab.bias_at_rest(T, REC["gz"], 8.0)
    assert abs(b - REC["gz"][T < 8.0].mean()) < 1e-12
    assert abs(b - EXP["true_gyro_bias_z_start"]) < 0.002


# ------------------------------------------------------------------------------ Part B
def test_B1_kf_predict():
    A = RNG.standard_normal((3, 3))
    B = RNG.standard_normal((3, 1))
    L = RNG.standard_normal((3, 3))
    P = L @ L.T + np.eye(3)
    Q = np.diag([0.1, 0.2, 0.3])
    x, u = RNG.standard_normal(3), 0.7
    xp, Pp = lab.kf_predict(x, P, A, Q, B=B, u=u)
    assert np.allclose(xp, A @ x + B[:, 0] * u, atol=1e-12)
    assert np.allclose(Pp, A @ P @ A.T + Q, atol=1e-12)
    xp, _ = lab.kf_predict(x, P, A, Q)
    assert np.allclose(xp, A @ x, atol=1e-12)


def test_B2_kf_update_information_form():
    """Covariance form == information form: P+ = (P^-1 + C^T R^-1 C)^-1, x+ = P+ (P^-1 x + C^T R^-1 y)."""
    L = RNG.standard_normal((3, 3))
    P = L @ L.T + 0.5 * np.eye(3)
    C = RNG.standard_normal((2, 3))
    R = np.diag([0.3, 0.8])
    x, y = RNG.standard_normal(3), RNG.standard_normal(2)
    xn, Pn, nu, S = lab.kf_update(x, P, y, C, R)
    Pi = np.linalg.inv(np.linalg.inv(P) + C.T @ np.linalg.inv(R) @ C)
    assert np.allclose(Pn, Pi, atol=1e-10)
    assert np.allclose(xn, Pi @ (np.linalg.solve(P, x) + C.T @ np.linalg.solve(R, y)), atol=1e-10)
    assert np.allclose(nu, y - C @ x, atol=1e-12) and np.allclose(S, C @ P @ C.T + R, atol=1e-12)


def test_B2_kf_matches_astrom_murray_predictor_form():
    """update then predict == Astrom & Murray Eq. 7.22: L = A P C^T (R + C P C^T)^-1,
    P[k+1] = (A - L C) P (A - L C)^T + Q + L R L^T  (their R_v = Q, R_w = R)."""
    A = np.array([[1.0, -0.01], [0.0, 1.0]])
    C = np.array([[1.0, 0.0]])
    Q, R = np.diag([1e-4, 1e-6]), np.array([[4e-4]])
    P = np.diag([0.01, 0.002])
    x, y = np.array([0.1, 0.0]), np.array([0.12])
    xu, Pu, _, _ = lab.kf_update(x, P, y, C, R)
    xp, Pp = lab.kf_predict(xu, Pu, A, Q)
    Lg = A @ P @ C.T @ np.linalg.inv(R + C @ P @ C.T)
    P_am = (A - Lg @ C) @ P @ (A - Lg @ C).T + Q + Lg @ R @ Lg.T
    assert np.allclose(Pp, P_am, atol=1e-14)
    assert np.allclose(xp, A @ x + Lg @ (y - C @ x), atol=1e-14)


def test_B2_scalar_workshop_problem_1():
    x, P = np.array([0.0]), np.array([[16.0]])
    for y in (3.0, 1.0, 2.5):
        x, P = lab.kf_predict(x, P, [[1.0]], [[2.0]])
        x, P, _, _ = lab.kf_update(x, P, [y], [[1.0]], [[4.0]])
    assert abs(x[0] - 2.0782) < 1e-4 and abs(P[0, 0] - 2.0664) < 1e-4


def test_B3_accel_model_and_jacobian():
    for th in RNG.uniform(-0.5, 1.2, 5):
        x = np.array([th, 0.03])
        hx, H = lab.accel_model(x)
        assert np.allclose(hx, cm.imu_specific_force(th, 0.0, 0.0)[:2], atol=1e-12)
        eps = 1e-6
        Hfd = np.c_[(lab.accel_model([th + eps, 0.03])[0] - lab.accel_model([th - eps, 0.03])[0]) / (2 * eps),
                    np.zeros(2)]
        assert np.allclose(H, Hfd, atol=1e-6)


def test_B4_ekf_converges_on_static_data():
    """Segment still at 25 deg, gyro reads only its bias: the EKF learns both the angle and the bias."""
    h, th0, b = 0.01, math.radians(25.0), 0.03
    t = np.arange(0, 30.0, h)
    gz = np.full(len(t), b)
    out = lab.ekf_tilt(t, gz, np.full(len(t), G * math.sin(th0)), np.full(len(t), G * math.cos(th0)),
                       0.006, 0.02, 0.5, x0=[0.0, 0.0], P0=np.diag([0.5, 0.05 ** 2]))
    assert abs(out["theta"][-1] - th0) < 1e-3 and abs(out["bias"][-1] - b) < 1e-3
    assert np.all(np.linalg.eigvalsh(out["P"][-1]) > 0)


def test_B4_ekf_gate_zero_is_dead_reckoning():
    out = lab.ekf_tilt(T[:500], REC["gz"][:500], REC["ax"][:500], REC["ay"][:500], 0.006, 0.02, 1.5, gate=0.0)
    assert out["rejected"] == 499
    th0 = math.atan2(REC["ax"][0], REC["ay"][0])
    assert np.allclose(out["theta"], lab.gyro_integrate(T[:500], REC["gz"][:500], th0), atol=1e-12)


def test_B4_ekf_on_recording():
    out = lab.ekf_tilt(T, REC["gz"], REC["ax"], REC["ay"], 0.006, 0.02, 1.5)
    assert abs(lab.rms_deg(out["theta"], REF) - EXP["rms_deg"]["ekf_tuned"]) < 1e-6
    assert lab.rms_deg(out["theta"], REF) < 1.2
    assert abs(out["bias"][-1] - EXP["true_gyro_bias_z_end"]) < 0.005
    assert np.isnan(out["nis"][0]) and np.all(np.isfinite(out["nis"][1:]))
    naive = lab.ekf_tilt(T, REC["gz"], REC["ax"], REC["ay"], 0.006, 0.02, 0.06)
    assert lab.rms_deg(naive["theta"], REF) > 3 * lab.rms_deg(out["theta"], REF)    # R too small: trusts swing


# ------------------------------------------------------------------------------ Part C
def test_C2_quat_to_rot_matches_matrix_exp():
    for _ in range(5):
        w, phi = random_unit(), RNG.uniform(0, math.pi)
        assert np.allclose(lab.quat_to_rot(quat_from_axis_angle(w, phi)), rot_from_axis_angle(w, phi), atol=1e-12)


def test_C1_quat_mult_is_rotation_product():
    for _ in range(5):
        (w1, a1), (w2, a2) = (random_unit(), RNG.uniform(0, 3)), (random_unit(), RNG.uniform(0, 3))
        n = lab.quat_mult(quat_from_axis_angle(w1, a1), quat_from_axis_angle(w2, a2))
        assert abs(np.linalg.norm(n) - 1) < 1e-12
        assert np.allclose(lab.quat_to_rot(n), rot_from_axis_angle(w1, a1) @ rot_from_axis_angle(w2, a2), atol=1e-12)


def test_C3_quat_exp():
    for _ in range(5):
        w = RNG.standard_normal(3) * 3
        assert np.allclose(lab.quat_to_rot(lab.quat_exp(w, 0.01)), mr.MatrixExp3(mr.VecToso3(w * 0.01)), atol=1e-12)
    assert np.allclose(lab.quat_exp(np.zeros(3), 0.01), [1, 0, 0, 0])


def test_C5_thigh_angle_from_quat_ignores_heading():
    for th in RNG.uniform(-0.5, 1.2, 5):
        q = quat_from_axis_angle([0, 0, 1], th)                        # R_sb = Rot(z, theta)
        assert abs(lab.thigh_angle_from_quat(q) - th) < 1e-12
        q_turned = lab.quat_mult(quat_from_axis_angle([0, 1, 0], 0.7), q)   # turn about the vertical y_s
        assert abs(lab.thigh_angle_from_quat(q_turned) - th) < 1e-12


def test_C4_mahony_static_learns_tilt_and_observable_bias():
    h, th0 = 0.01, math.radians(30.0)
    t = np.arange(0, 60.0, h)
    b = np.array([0.0, 0.0, 0.02])
    acc = np.tile([G * math.sin(th0), G * math.cos(th0), 0.0], (len(t), 1))
    out = lab.mahony_filter(t, np.tile(b, (len(t), 1)), acc, kp=1.0, ki=0.1,
                            q0=np.array([1.0, 0.0, 0.0, 0.0]))
    assert abs(lab.thigh_angle_from_quat(out["q"][-1]) - th0) < 1e-3
    assert abs(out["bias"][-1, 2] - 0.02) < 1e-3


def test_C4_mahony_on_recording():
    out = lab.mahony_filter(T, np.c_[REC["gx"], REC["gy"], REC["gz"]], np.c_[REC["ax"], REC["ay"], REC["az"]],
                            kp=1.0, ki=0.1)
    th = np.array([lab.thigh_angle_from_quat(q) for q in out["q"]])
    assert abs(lab.rms_deg(th, REF) - EXP["rms_deg"]["mahony"]) < 1e-6
    assert lab.rms_deg(th, REF) < 2.5
    assert abs(out["bias"][-1, 2] - EXP["true_gyro_bias_z_end"]) < 0.01
    # heading (about the vertical) is not observable from the accelerometer: it drifts
    assert abs(math.degrees(lab.heading_from_quat(out["q"][-1]))) > 10.0
