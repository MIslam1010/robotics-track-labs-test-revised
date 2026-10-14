"""BASELINE designs of the six scenarios. They are deliberately simple; most fail several acceptance
tests (scenario_sheets.md gives their numbers). They implement the same interfaces as
``team_design.py``, so ``tasks.sK_acceptance(baseline)`` runs on them."""
from __future__ import annotations

import numpy as np

from . import arm, tasks, teleop


# ------------------------------------------------------------------------------------------- S1
def s1_architecture():
    """Position-position teleoperation (Lab 3 "pp"), no governor, no safety filter."""
    return teleop.PositionPosition()


# ------------------------------------------------------------------------------------------- S2
class S2Baseline:
    """Drawing values instead of calibration and registration; stop-and-go quintic approach;
    joint-space quintic insertion from the pre-insertion pose to the final pose (no RCM control,
    no safety filter)."""

    def calibrate_tool(self, R_list, p_list):
        return tasks.S2["T_ft_nominal"].copy()

    def register(self, P_p, P_s):
        return tasks.S2["T_sp_nominal"].copy()

    def tre_prediction(self, P_p, p_target, fle_rms):
        return float("nan")

    def plan_approach(self, q_start, q_goal, v_max, a_max):
        T = arm.quintic_duration(q_start, q_goal, v_max, a_max)
        t = np.arange(0.0, T + 1e-9, 1e-3)
        return t, np.array([arm.quintic(q_start, q_goal, T, tt)[0] for tt in t])

    def insertion_rate(self, t, q, ctx):
        tgt = np.asarray(ctx["target"], float)
        if t == 0.0 or np.linalg.norm(tgt - self.tgt) > 1e-9:      # (re)plan on a new target
            self.t0, self.q0, self.tgt = t, np.array(q, float), tgt.copy()
            self.T = tasks.S2["T_insert"] - t
            T_end = arm.line_pose(ctx["entry"] + tasks.S2["depth"] * ctx["axis"], ctx["axis"], (0, -1, 0))
            self.q1, _ = arm.ik_tool(T_end, q, ctx["T_ft"])
        return arm.quintic(self.q0, self.q1, self.T, t - self.t0)[1]


# ------------------------------------------------------------------------------------------- S3
class S3Baseline:
    """Transparent device (no assistance) with the range of motion as a stiff virtual wall
    (K = 200 N/m at +-25 mm); velocity from the firmware's filtered backward difference; no
    patient-force estimate (logged as 0)."""

    def controller(self, case):
        from . import devices, safety
        est = devices.VelocityEstimator()
        x_rom = tasks.S3["x_rom"]

        def ctrl(t, x, sens):
            v = est(x)
            ctrl.log["t"].append(t)
            ctrl.log["v_hat"].append(v)
            ctrl.log["F_u_hat"].append(0.0)
            return safety.virtual_wall_force(x, v, -x_rom, x_rom, K=200.0)
        ctrl.log = dict(t=[], v_hat=[], F_u_hat=[])
        return ctrl


# ------------------------------------------------------------------------------------------- S6
class S6Baseline:
    """Open-loop gait-phase timer at the nominal cadence; thigh angle from the complementary filter
    (Lab 9, no bias estimate); assistance F_d = -F_load * stance(phase) sent open loop to the SEA
    motor (u = F_d); no range safety filter beyond the device watchdog."""

    def controller(self, imu):
        from . import estimation
        th = estimation.complementary_filter(imu["t"], imu["gyro"],
                                             estimation.accel_tilt(imu["ax"], imu["ay"]), tau=0.5)
        f0 = tasks.S6["f0"]

        def ctrl(t, x, sea):
            k = min(int(t * tasks.S6["imu_rate"]), len(th) - 1)
            ph = estimation.phase_timer(t, f0)
            Fd = -tasks.S6["F_load"] * float(tasks.s6_stance(ph))
            for key, val in zip(("t", "phase", "theta", "F_d"), (t, ph, th[k], Fd)):
                ctrl.log[key].append(val)
            return Fd
        ctrl.log = dict(t=[], phase=[], theta=[], F_d=[])
        return ctrl


# --------------------------------------------------------------------------------------- S4, S5
class ClippedPositionPosition(teleop.PositionPosition):
    """BASELINE for the course-arm follower: position-position with the follower command clipped at
    the (estimated) boundary x_lim; the operator feels the wall only through the coupling force
    Kp (x_d - x_s) - Kd v_s computed from the UNclipped command."""

    def __init__(self, x_lim, **kw):
        super().__init__(**kw)
        self.x_lim = x_lim

    def setpoint(self, k, t, x_d, x_s, v_s, f_meas):
        return min(x_d, self.x_lim)

    def follower(self, k, t, h, pkt, x_s, v_s, f_meas):
        x_new = self.alpha * pkt[0]
        dx = 0.0 if self.x_d is None else x_new - self.x_d
        self.x_d = x_new
        r = self.setpoint(k, t, self.x_d, x_s, v_s, f_meas)
        F_c = self.Kp * (self.x_d - x_s) - self.Kd * v_s
        return F_c, r, [F_c, 0.0, 0.0], F_c * dx / h


class S4Baseline:
    """Object pose from the camera estimate (8 mm too far, no registration); clipped P-P."""

    def register(self, P_model, P_s):
        T = tasks.S4["T_so_true"].copy()
        T[0, 3] += tasks.S4["camera_error"]
        return T

    def architecture(self, x_lim):
        a = tasks.S4["alpha"]
        return ClippedPositionPosition(x_lim + tasks.S4["camera_error"] * 0, Kp=50.0, Kd=1.0, alpha=a, beta=a)


class S5Baseline:
    """Box trainer at its nominal pose (no registration); clipped P-P without a fixture; tip-only
    resolved-rate insertion (no RCM constraint)."""

    def register(self, P_model, P_s):
        return tasks.S5["T_sk_nominal"].copy()

    def architecture(self, x_lim, fixture=None):
        a = tasks.S5["alpha"]
        return ClippedPositionPosition(x_lim, Kp=50.0, Kd=1.0, alpha=a, beta=a)

    def insertion_rate(self, t, q, ctx):
        T = arm.fk_tool(q, ctx["T_ft"])
        J = arm.point_jacobian(q, T[:3, 3])
        v = ctx["v_d"] + 5.0 * (ctx["p_d"] - T[:3, 3])
        return np.linalg.pinv(J) @ v
