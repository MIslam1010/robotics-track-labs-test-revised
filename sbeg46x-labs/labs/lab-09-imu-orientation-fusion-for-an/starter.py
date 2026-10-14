"""SBEG 46X Lab 9 -- IMU orientation fusion for an exoskeleton thigh segment: complementary filter vs EKF.
STARTER CODE -- you edit this file.  Its functions join the module `estimation` of your library sbeg46x_medrob.
Write every function marked TODO (keep the names and signatures); the helpers in the "Provided" sections are given
(do not change them).  Run the checks from your Lab code folder with:   LAB_IMPL=starter python -m pytest -q test_lab.py
(rename the file to lab9_<ID>.py before you submit and run LAB_IMPL=lab9_<ID>).

Segment and sensor (course model gait_exo_2r, thigh on the test stand; see course_models.py):
    theta = hip flexion angle (rad), 0 with the leg hanging, positive = flexion (thigh forward).
    IMU on the thigh cuff, r_imu = 0.20 m below the hip; sensor frame {b} with R_sb = Rot(z, theta):
    x_b forward (perpendicular to the thigh), y_b up the thigh toward the hip, z_b = hip axis.
    Gyro:          omega_z = theta_dot + b + noise                    (rad/s)
    Accelerometer: ax = g sin(theta) + r theta_ddot + noise,  ay = g cos(theta) + r theta_dot^2 + noise   (m/s^2)
    (specific force f_b = R_sb^T (a - g) in sensor axes: at rest the accelerometer reads +g "up", so
    theta = atan2(ax, ay)).
Discrete time: samples k = 0..N-1, period h; the gyro sample k propagates the state from k-1 to k and the
accelerometer sample k corrects it (as in the lecture notes).  The course-project baseline (sbeg46x_project.estimation)
uses the previous gyro sample and a sensor frame turned by 90 deg (x along the segment); both differences are
harmless but keep them in mind when you port this code.
Kalman filter notation (Astrom & Murray Sec. 7.4, written in predict/update form; Corke Sec. 6.1):
    x_k = A x_{k-1} + B u_{k-1} + v,  v ~ N(0, Q);     y_k = C x_k + w,  w ~ N(0, R)
    predict  x- = A x + B u,  P- = A P A^T + Q
    update   S = C P- C^T + R,  L = P- C^T S^-1,  x = x- + L (y - C x-),  P = (I - L C) P-
    (gain L as in Astrom & Murray and the lecture notes; many texts write K)
Quaternions: q = (q0, q1, q2, q3) = (cos(phi/2), w_hat sin(phi/2)) (MR Eq. B.9), product MR Eq. B.15 (q*p <-> R_q R_p),
rotation matrix MR Eq. B.12.  Body-frame gyro rates: R_k = R_{k-1} exp([omega] h)  <->  q_k = q_{k-1} * exp_q(omega h).
All values are illustrative.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
G = 9.81


# ==========================================================================================
# Provided (do not edit): data loading, error metrics, initial quaternion
# ==========================================================================================
def load_meta(path=DATA / "imu_meta.json") -> dict:
    """Recording metadata (data/imu_meta.json): sample period h, r_imu, sensor noise specifications, phases."""
    with open(path) as f:
        return json.load(f)


def load_recording(path=DATA / "thigh_imu.csv") -> dict:
    """Thigh IMU recording (data/thigh_imu.csv) as a dict of arrays: t (s), gx, gy, gz (rad/s), ax, ay, az (m/s^2)
    and theta_enc (rad), the hip-joint encoder of the test stand (reference only; NOT an input of the filters)."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def rms_deg(est, ref, mask=None) -> float:
    """RMS of (est - ref) in degrees, optionally over a boolean mask."""
    e = np.asarray(est, float) - np.asarray(ref, float)
    if mask is not None:
        e = e[np.asarray(mask, bool)]
    return float(np.degrees(np.sqrt(np.mean(e ** 2))))


def quat_from_accel(acc) -> np.ndarray:
    """Initial attitude from one accelerometer sample at rest: the rotation that takes the world 'up' axis
    y_s = (0, 1, 0) onto the measured up direction in the sensor frame (heading set to zero).  Returns q_sb
    (sensor frame {b} seen from {s})."""
    a = np.asarray(acc, float) / np.linalg.norm(acc)
    up = np.array([0.0, 1.0, 0.0])
    # R_sb^T up = a  <=>  R_sb a = up: rotate a onto up about axis a x up
    ax = np.cross(a, up)
    s, c = np.linalg.norm(ax), float(a @ up)
    if s < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0]) if c > 0 else np.array([0.0, 0.0, 0.0, 1.0])
    phi = math.atan2(s, c)
    return np.r_[math.cos(phi / 2), ax / s * math.sin(phi / 2)]


# ==========================================================================================
# Part A -- tilt from the accelerometer, gyro integration and the complementary filter
# ==========================================================================================
def accel_tilt(ax, ay):
    """A1. Thigh angle from the accelerometer alone, theta = atan2(ax, ay) (rad); exact only at rest."""
    # TODO A1: write this function (see the docstring and the lab handout).
    raise NotImplementedError("TODO A1: accel_tilt")


def gyro_integrate(t, gyro, theta0=0.0, bias=0.0):
    """A2. Dead reckoning: theta_0 = theta0, theta_k = theta_{k-1} + (t_k - t_{k-1}) (gyro_k - bias)."""
    # TODO A2: write this function (see the docstring and the lab handout).
    raise NotImplementedError("TODO A2: gyro_integrate")


def cf_alpha(tau, h):
    """A3. Blending factor of the discrete first-order complementary filter, alpha = tau / (tau + h)."""
    # TODO A3: write this function (see the docstring and the lab handout).
    raise NotImplementedError("TODO A3: cf_alpha")


def complementary_filter(t, gyro, theta_acc, tau=0.5, theta0=None, bias=0.0):
    """A4. theta_k = alpha (theta_{k-1} + h (gyro_k - bias)) + (1 - alpha) theta_acc_k, alpha = tau/(tau + h).
    theta_0 = theta_acc_0 unless theta0 is given."""
    # TODO A4: write this function (see the docstring and the lab handout).
    raise NotImplementedError("TODO A4: complementary_filter")


def bias_at_rest(t, gyro, t_end):
    """A5. Gyro bias estimated as the mean gyro reading while the segment is still (t < t_end)."""
    # TODO A5: write this function (see the docstring and the lab handout).
    raise NotImplementedError("TODO A5: bias_at_rest")


# ==========================================================================================
# Part B -- linear Kalman filter and the EKF with gyro-bias state
# ==========================================================================================
def kf_predict(x, P, A, Q, B=None, u=None):
    """B1. Kalman predict: x- = A x + B u, P- = A P A^T + Q."""
    # TODO B1: write this function (see the docstring and the lab handout).
    raise NotImplementedError("TODO B1: kf_predict")


def kf_update(x, P, y, C, R):
    """B2. Kalman update with measurement y = C x + w, w ~ N(0, R).
    Returns (x, P, nu, S): updated state and covariance, innovation nu = y - C x- and its covariance S."""
    # TODO B2: write this function (see the docstring and the lab handout).
    raise NotImplementedError("TODO B2: kf_update")


def accel_model(x, g=G):
    """B3. EKF measurement model of the accelerometer for the state x = (theta, b):
    h(x) = (g sin theta, g cos theta) and its Jacobian H = dh/dx (2 x 2)."""
    # TODO B3: write this function (see the docstring and the lab handout).
    raise NotImplementedError("TODO B3: accel_model")


def ekf_tilt(t, gyro, ax, ay, sigma_gyro, sigma_bias, sigma_acc, x0=None, P0=None, gate=None, g=G):
    """B4. EKF for x = (theta, b) with the gyro as input and the 2-axis accelerometer as measurement.
       predict: theta- = theta + h (gyro_k - b), b- = b;  A = [[1, -h], [0, 1]], B = [h, 0]^T,
                Q = diag((h sigma_gyro)^2, (h sigma_bias)^2)    (sigma_bias: rate of the slow bias walk, rad/s^2;
                as in the lecture notes Sec. 10.2)
       update:  y = (ax, ay), linearised with accel_model at x-;  R = sigma_acc^2 I.
                If gate is given, the update is skipped when the NIS nu^T S^-1 nu exceeds gate.
    x0 defaults to (accel tilt of sample 0, 0) and P0 to diag((5 deg)^2, (0.05 rad/s)^2).
    Returns dict(theta, bias, P (N x 2 x 2), nis (N, NaN at k = 0), rejected (number of gated updates))."""
    # TODO B4: write this function (see the docstring and the lab handout).
    raise NotImplementedError("TODO B4: ekf_tilt")


# ==========================================================================================
# Part C -- quaternions and a 3-D explicit complementary filter (Mahony, Hamel & Pflimlin 2008)
# ==========================================================================================
def quat_mult(q, p):
    """C1. Quaternion product n = q * p of MR Eq. B.15 (the quaternion of R_q R_p)."""
    # TODO C1: write this function (see the docstring and the lab handout).
    raise NotImplementedError("TODO C1: quat_mult")


def quat_to_rot(q):
    """C2. Rotation matrix of a unit quaternion, MR Eq. B.12."""
    # TODO C2: write this function (see the docstring and the lab handout).
    raise NotImplementedError("TODO C2: quat_to_rot")


def quat_exp(omega, h):
    """C3. Quaternion of the rotation exp([omega] h): (cos(|omega| h / 2), omega_hat sin(|omega| h / 2)) (MR Eq. B.9)."""
    # TODO C3: write this function (see the docstring and the lab handout).
    raise NotImplementedError("TODO C3: quat_exp")


def mahony_filter(t, gyro3, acc3, kp=1.0, ki=0.1, q0=None, up=(0.0, 1.0, 0.0)):
    """C4. Explicit complementary filter on SO(3) with gyro-bias estimation (Mahony, Hamel & Pflimlin 2008):
        v_hat = R_hat^T up             (predicted 'up' direction in the sensor frame)
        v     = acc / |acc|            (measured 'up' direction: the accelerometer reads +g up at rest)
        e     = v x v_hat              (correction rate, rad/s)
        q_k   = q_{k-1} * exp_q((gyro_k - b_hat + kp e) h),   b_hat <- b_hat - ki e h
    with v from the accelerometer sample k and v_hat from q_{k-1}.  q0 defaults to quat_from_accel(acc3[0]).
    Returns dict(q (N x 4, q_sb), bias (N x 3))."""
    # TODO C4: write this function (see the docstring and the lab handout).
    raise NotImplementedError("TODO C4: mahony_filter")


def thigh_angle_from_quat(q):
    """C5. Signed thigh inclination from the estimated attitude q_sb: the thigh axis pointing to the knee is
    d = -R_sb y_b (second column of R, negated); theta = atan2(sign(d_x) sqrt(d_x^2 + d_z^2), -d_y).
    Uses only the tilt of the thigh, so it is not affected by the (unobservable) heading about the vertical."""
    # TODO C5: write this function (see the docstring and the lab handout).
    raise NotImplementedError("TODO C5: thigh_angle_from_quat")


def heading_from_quat(q):
    """Provided. Heading (rotation about the vertical y_s) of the thigh's flexion axis z_i, in rad: 0 when the
    hip axis is along z_s, positive about +y_s (vertical).  Shows the drift that the accelerometer cannot correct."""
    z = quat_to_rot(q)[:, 2]
    return math.atan2(z[0], z[2])
