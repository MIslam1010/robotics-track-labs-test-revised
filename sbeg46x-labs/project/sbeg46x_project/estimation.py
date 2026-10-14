"""State estimation (Scenarios 3 and 6).

Provided BASELINES: backward-difference velocity, accelerometer tilt, gyro integration, the
complementary filter (Lab 9), the open-loop gait-phase timer and the phase-portrait phase.
TEAM TODO (port your Lab 9 code): ``KalmanFilter`` (linear predict / update; Astrom-Murray §7.4,
Corke §6.1 for the EKF form).
"""
from __future__ import annotations

import numpy as np


def backward_difference(t, x) -> np.ndarray:
    t, x = np.asarray(t, float), np.asarray(x, float)
    v = np.zeros_like(x)
    v[1:] = np.diff(x) / np.diff(t)
    return v


def accel_tilt(ax, ay) -> np.ndarray:
    """Segment angle from the accelerometer alone (valid when the motion accelerations are small):
    theta = atan2(-ay, ax) for the sensor axes of ``devices.imu_from_angle``."""
    return np.arctan2(-np.asarray(ay, float), np.asarray(ax, float))


def gyro_integrate(t, gyro, theta0=0.0) -> np.ndarray:
    t, w = np.asarray(t, float), np.asarray(gyro, float)
    out = np.empty_like(w)
    out[0] = theta0
    out[1:] = theta0 + np.cumsum(w[:-1] * np.diff(t))
    return out


def complementary_filter(t, gyro, theta_acc, tau=0.5, theta0=None) -> np.ndarray:
    """First-order complementary filter: high-pass the integrated gyro, low-pass the accelerometer
    tilt, with crossover time constant tau (s). BASELINE of Scenario 6 (gyro bias not estimated)."""
    t, w, ta = np.asarray(t, float), np.asarray(gyro, float), np.asarray(theta_acc, float)
    th = np.empty_like(w)
    th[0] = ta[0] if theta0 is None else theta0
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        a = tau / (tau + dt)
        th[k] = a * (th[k - 1] + w[k - 1] * dt) + (1 - a) * ta[k]
    return th


def phase_timer(t, f_nominal, t0=0.0) -> np.ndarray:
    """BASELINE gait phase: open-loop timer at the nominal cadence (fraction of the cycle in [0, 1))."""
    return ((np.asarray(t, float) - t0) * f_nominal) % 1.0


def phase_portrait(theta, omega, theta_mid, omega_scale) -> np.ndarray:
    """Gait phase (fraction of the cycle in [0, 1)) from the phase portrait of a segment angle:
    phi = atan2(-omega / omega_scale, theta - theta_mid) / (2 pi). theta_mid and omega_scale
    normalise the orbit into a circle (e.g. the running mean of theta and the ratio of the angle and
    rate amplitudes). Works only on good estimates of theta and omega."""
    p = np.arctan2(-np.asarray(omega, float) / omega_scale, np.asarray(theta, float) - theta_mid)
    return (p / (2 * np.pi)) % 1.0


class KalmanFilter:
    """Linear Kalman filter x[k+1] = A x[k] + B u[k] + w,  y[k] = C x[k] + v,  w ~ N(0, Qw), v ~ N(0, Rv).
    TEAM TODO (port Lab 9): implement ``predict`` and ``update`` (Astrom-Murray §7.4)."""

    def __init__(self, A, B, C, Qw, Rv, x0, P0):
        self.A, self.B, self.C = np.atleast_2d(A), np.atleast_2d(B), np.atleast_2d(C)
        self.Qw, self.Rv = np.atleast_2d(Qw), np.atleast_2d(Rv)
        self.x, self.P = np.asarray(x0, float).copy(), np.atleast_2d(P0).astype(float).copy()

    def predict(self, u=None):
        raise NotImplementedError("KalmanFilter.predict: port your Lab 9 code")

    def update(self, y, C=None, Rv=None):
        raise NotImplementedError("KalmanFilter.update: port your Lab 9 code")
