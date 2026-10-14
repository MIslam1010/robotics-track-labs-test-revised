"""Emulator of one course arm for the simulation station of Hardware rotation 1 (Week 8).

`SimArm` has the same interface as `vendor_arm.VendorArm` (connect, time, wait_until, send, read, stop, reset,
close), so `run_session.py` and your `run_logged` run unchanged on the emulator and on a real arm.  It also offers
`measure_tip`, which plays the role of the grid and vertical scale (read from camera photos) at the hardware station.

What the emulator models (ILLUSTRATIVE VALUES, chosen to resemble a low-cost smart-servo arm):

  servo loop     each joint moves towards its target with a first-order lag (time constant 50 ms) and at most
                 the rated speed; under load it settles short of the target by tau_g / k (gravity droop that the
                 servo's own encoder SEES, as in Lab 5);
  gear backlash  a dead band between the servo shaft and the link (the encoder does NOT see it);
  zero error     the link's true zero differs from the calibrated one by a few tenths of a degree to ~1.5 deg
                 (the encoder does NOT see it either);
  encoder        noise and 12-bit quantisation of the reported angle;
  stop           `stop()` latches: the target is frozen where the joint will coast to (about 30 ms of motion).

The hidden "truth" of each emulated arm is in ARMS below.  Reading it before you have analysed your data spoils the
exercise; in the oral check you will be asked to explain your estimates from the data, not from this table.
Time is VIRTUAL: `wait_until(t)` integrates the dynamics up to t immediately (no real waiting).
"""
import numpy as np
import modern_robotics as mr

from course_arm_model import ENC_RES_DEG, course_arm_dynamics, course_arm_reference
from safety_guard import SafetyGuard

OFFSETS_DEG = np.array([0.0, 90.0, 0.0, 90.0, 0.0, 0.0])   # servo zero = arm straight up ("candle")

#: hidden parameters of the emulated arms (degrees; stiffness in N m/rad) -- ILLUSTRATIVE VALUES
ARMS = {
    "A5": dict(zero_err=(0.0, -0.8, 1.2, 0.0, 0.0, 0.0), backlash=(0.2, 0.5, 0.6, 0.3, 0.3, 0.3), seed=8),
    "S1": dict(zero_err=(0.0, 0.0, 1.5, 0.0, 0.0, 0.0), backlash=(0.2, 0.4, 0.5, 0.3, 0.3, 0.3), seed=11),
    "S2": dict(zero_err=(0.0, -1.2, 0.0, 0.0, 0.0, 0.0), backlash=(0.2, 0.6, 0.4, 0.3, 0.3, 0.3), seed=12),
    "S3": dict(zero_err=(0.0, 0.7, -0.9, 0.0, 0.0, 0.0), backlash=(0.3, 0.5, 0.5, 0.4, 0.3, 0.3), seed=13),
    "S4": dict(zero_err=(0.0, 0.0, -1.3, 0.0, 0.0, 0.0), backlash=(0.2, 0.8, 0.6, 0.3, 0.3, 0.3), seed=14),
    "S5": dict(zero_err=(0.0, 1.0, 0.6, 0.0, 0.0, 0.0), backlash=(0.2, 0.4, 0.4, 0.3, 0.3, 0.3), seed=15),
    "S6": dict(zero_err=(0.0, -0.5, -0.5, 0.0, 0.0, 0.0), backlash=(0.2, 0.5, 0.7, 0.3, 0.3, 0.3), seed=16),
}
STIFFNESS = np.array([40.0, 80.0, 60.0, 30.0, 30.0, 30.0])   # servo position-loop stiffness, N m / rad
TAU_SERVO = 0.05          # s
RATED_DEG_S = 160.0
NOISE_DEG = 0.05
PROBE_SIGMA_MM = 0.4      # external measurement (grid + vertical scale photos), 1 sigma per axis
DT = 0.002                # integration step, s
GRAVITY_EVERY = 5         # gravity torque update every 5 steps (10 ms)


class SimArm:
    """Emulated course arm.  Angles in SERVO degrees (q_model = q_servo - offset)."""

    def __init__(self, arm_id="S1", offsets_deg=OFFSETS_DEG, start_servo_deg=None, guard=None,
                 noise=True, seed=None):
        if arm_id not in ARMS:
            raise KeyError(f"unknown emulated arm '{arm_id}'; choose one of {sorted(ARMS)}")
        p = ARMS[arm_id]
        self.arm_id = arm_id
        self.offsets = np.asarray(offsets_deg, float)
        self.zero_err = np.asarray(p["zero_err"], float)
        self.backlash = np.asarray(p["backlash"], float)
        self.rng = np.random.default_rng(p["seed"] if seed is None else seed)
        self.noise = noise
        self.guard = guard if guard is not None else SafetyGuard(self.offsets)
        self._M, self._S, _ = course_arm_reference()
        self._Mlist, self._Glist, self._g = course_arm_dynamics()
        q0 = self.offsets + np.array([0.0, -60.0, 90.0, -120.0, -90.0, 0.0]) if start_servo_deg is None \
            else np.asarray(start_servo_deg, float)
        self.motor = q0.astype(float).copy()      # servo-shaft angle (what the encoder measures)
        self.link = q0.astype(float).copy()       # link-side angle before the zero error (after backlash)
        self.target = q0.astype(float).copy()
        self.stopped = False
        self._t = 0.0
        self._k = 0
        self._droop = self._gravity_droop()
        self.motor = self.target - self._droop        # power-up: the servos have settled under gravity
        self.link = self.motor.copy()
        self._droop = self._gravity_droop()
        self.connected = False

    # ------------------------------------------------------------------ interface shared with VendorArm
    def connect(self):
        self.connected = True
        return self

    def close(self):
        self.connected = False

    def time(self):
        return self._t

    def send(self, q_servo_deg):
        """Command new target angles (servo degrees).  Ignored while a stop is latched."""
        if self.stopped:
            return False
        q = self.guard.check(self._t, q_servo_deg)
        self.target = np.asarray(q, float).copy()
        return True

    def read(self):
        """Angles reported by the servos (servo degrees): shaft angle + noise, 12-bit quantised."""
        q = self.motor.copy()
        if self.noise:
            q = q + self.rng.normal(0.0, NOISE_DEG, 6)
        return np.round(q / ENC_RES_DEG) * ENC_RES_DEG

    def stop(self):
        """Latched stop: freeze the target where the joints coast to, refuse further motion commands."""
        v = self._velocity()
        self.target = self.motor + v * 0.03
        self.stopped = True
        self.guard.trip("operator stop")

    def reset(self):
        """Operator reset after a stop (the arm stays where it is)."""
        self.stopped = False
        self.target = self.motor.copy()
        self.guard.reset(self._t, self.motor)

    def wait_until(self, t):
        """Advance the virtual clock to time t (s), integrating the joint dynamics."""
        while self._t < t - 1e-12:
            h = min(DT, t - self._t)
            self._step(h)

    # ------------------------------------------------------------------ external measurement (emulated)
    def true_model_angles(self):
        """TRUE link angles in model radians (hidden from the encoder: backlash and zero error included)."""
        return np.deg2rad(self.link + self.zero_err - self.offsets)

    def measure_tip(self, p_tool=(0.0, 0.0, 0.06), sigma_mm=PROBE_SIGMA_MM):
        """Tool-point position in {s} (mm) as measured on the grid / vertical scale photos, with measurement noise."""
        T = mr.FKinSpace(self._M, self._S, self.true_model_angles())
        p = 1e3 * (T @ np.r_[np.asarray(p_tool, float), 1.0])[:3]
        if self.noise and sigma_mm > 0:
            p = p + self.rng.normal(0.0, sigma_mm, 3)
        return np.round(p, 1)

    # ------------------------------------------------------------------ internals
    def _gravity_droop(self):
        tau = mr.GravityForces(self.true_model_angles(), self._g, self._Mlist, self._Glist, self._S)
        return np.rad2deg(np.asarray(tau) / STIFFNESS)

    def _velocity(self):
        v = (self.target - self._droop - self.motor) / TAU_SERVO
        return np.clip(v, -RATED_DEG_S, RATED_DEG_S)

    def _step(self, h):
        if self._k % GRAVITY_EVERY == 0:
            self._droop = self._gravity_droop()
        self.motor = self.motor + h * self._velocity()
        half = self.backlash / 2
        gap = self.motor - self.link
        self.link = np.where(gap > half, self.motor - half, np.where(gap < -half, self.motor + half, self.link))
        self._t += h
        self._k += 1
