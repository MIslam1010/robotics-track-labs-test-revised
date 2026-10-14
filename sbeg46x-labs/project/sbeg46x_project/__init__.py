"""SBEG 46X Medical Robotics -- course project starter package.

Six project scenarios (three biomedical, three defence: protective / medical / logistic uses only)
built on the course hardware: Hapkit-type 1-DOF haptic devices (two per station), the course arm
(myCobot-280-class desktop 6R) and a low-cost IMU.

Modules
-------
devices       -- Hapkit model, operator / patient hands, servo axis of the course arm, IMU model,
                 single-device simulator
teleop        -- delayed network, teleoperation architectures (baseline position-position; wave
                 variables and TDPA wired to TEAM TODO functions), bilateral simulator
tissue        -- layered needle-insertion model (Scenario 1), force sensor, virtual wall
arm           -- course-arm kinematics (PoE), tool transforms, RCM / clearance geometry, servo model
registration  -- transforms, FRE / TRE; TEAM TODO: pivot calibration, SVD registration, TRE prediction
optctrl       -- c2d, discrete LQR, SciPy QP fallback; TEAM TODO: LinearMPC
estimation    -- baselines (complementary filter, gait-phase timer, phase portrait); TEAM TODO: KalmanFilter
safety        -- baseline limits; TEAM TODO: CBF filters
tasks         -- scenario test conditions, runs and metrics behind the acceptance tests
baselines     -- the baseline design of each scenario
hardware      -- hardware-log reading and sim-vs-hardware comparison (SBEG 461 log format + aux columns)
hwtasks       -- required hardware run: provisional HW thresholds, identification (free decay, servo
                 step), replay simulations of a logged command, reference signals
metrics       -- RMS and helpers

All numerical parameters are ILLUSTRATIVE VALUES for teaching; they are not data of any real
product, patient or soldier.
"""
from . import arm, devices, estimation, hardware, hwtasks, metrics, optctrl, registration, safety, teleop, tissue  # noqa: F401

__all__ = ["arm", "devices", "estimation", "hardware", "hwtasks", "metrics", "optctrl", "registration", "safety",
           "teleop", "tissue"]
