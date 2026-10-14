"""SBEG 461 Lab 6 -- multi-joint control of the course arm: decentralized PID, PD + gravity compensation,
feedforward + PD, computed torque and task-space computed torque, with and without model error.
STARTER CODE: complete every function marked TODO (keep names and signatures).
Provided functions (do not edit): load_config, arm_model, make_case, tau_limits, time_grid, quintic_reference,
cartesian_reference, rk4_step, Plant, simulate_arm, _diag.

Notation follows Modern Robotics (MR) Ch. 11: theta_e = theta_d - theta, error integral eint = int theta_e dt,
dynamics tau = M(theta) theta'' + h(theta, theta')  (MR Eq. 11.36), h = c + g + friction.  "model" is the
CONTROLLER's model {M~, h~}; "plant" is the simulated robot.  In code q = theta.  Units: SI (s, rad, N m).

Parts
  A. Gains: computed-torque gains from (zeta, wn) for the unit-inertia error dynamics (MR Eq. 11.33), the
     integral-gain bound, decentralized per-joint gains from the diagonal of M (Lab 5 rules, MR Eq. 11.26)
  B. Joint-space control laws: decentralized PID (MR Eq. 11.23 per joint), PID + gravity compensation
     (MR Eq. 11.38), feedforward + PID (MR Eq. 11.32 + feedback; Corke Sec. 9.4.3.1), computed torque
     (MR Eq. 11.37, pseudocode MR Fig. 11.20), the error energy (MR Eq. 11.41), a controller object that
     keeps the error integral, and tracking metrics (MR Fig. 11.19: error and control effort int tau^T tau dt)
  C. Task-space computed torque (MR Sec. 11.4.3, Eq. 11.47): time derivative of the body Jacobian
     (MR Exercise 8.6b), configuration and velocity errors, the feedforward acceleration
     d/dt([Ad_{X^-1 X_d}] V_d), task-space dynamics Lambda, eta (MR Eqs. 8.90-8.91) and the controller

These functions become `sbeg461_dynamics/control.py` (next to Lab 5's JointPID) in your cumulative library.
"""
from __future__ import annotations

import json
from pathlib import Path

import modern_robotics as mr
import numpy as np

import course_arm as ca

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


# ==========================================================================================
# Provided (do not edit): configuration, robot models, references, plant simulator
# ==========================================================================================
def load_config() -> dict:
    """data/lab06_config.json (illustrative values; see make_data.py)."""
    return json.loads((DATA / "lab06_config.json").read_text())


def arm_model(payload: float = 0.0, mass_scale: float = 1.0, friction=None) -> dict:
    """The course arm (course_arm.py, identical to the shared model) as a dict with Slist, Blist, M, Mlist, Glist,
    g, tau_max, q_home and b (viscous friction vector, zeros if friction is None).  `payload` (kg) is a point mass
    at the tool point; `mass_scale` multiplies every spatial inertia G_i (masses and inertias, CoMs unchanged)."""
    r = ca.course_arm(payload=payload)
    r["Glist"] = [mass_scale * np.asarray(G, float) for G in r["Glist"]]
    r["Blist"] = mr.Adjoint(mr.TransInv(r["M"])) @ r["Slist"]
    r["b"] = np.zeros(6) if friction is None else np.asarray(friction, float)
    r["mass_scale"] = float(mass_scale)
    r["payload"] = float(payload)
    return r


def make_case(name: str, cfg: dict | None = None):
    """(plant, model) for a model-error case of the configuration: the plant is always the course arm with the
    0.25 kg payload and the configured friction; the model follows cfg['cases'][name]."""
    cfg = load_config() if cfg is None else cfg
    c = cfg["cases"][name]
    plant = arm_model(cfg["payload"], 1.0, cfg["friction_b"])
    model = arm_model(c["model_payload"], c["model_mass_scale"], cfg["friction_b"] if c["model_friction"] else None)
    return plant, model


def tau_limits(robot: dict, cfg: dict | None = None) -> np.ndarray:
    """Derated torque limits: tau_limit_fraction (0.6) x rated torque (hardware rule in Course resources: Course robot models and hardware safety checklist)."""
    cfg = load_config() if cfg is None else cfg
    return cfg["tau_limit_fraction"] * np.asarray(robot["tau_max"], float)


def time_grid(T: float, dt: float) -> np.ndarray:
    n = int(round(T / dt))
    return np.arange(n + 1) * dt


def quintic_reference(q0, q1, T: float, t):
    """Rest-to-rest quintic time scaling s(t) = 10 tau^3 - 15 tau^4 + 6 tau^5, tau = t/T (MR Eq. 9.12 type),
    q = q0 + s (q1 - q0), held at q1 for t > T.  Returns qd, dqd, ddqd as arrays (len(t), n)."""
    q0, q1, t = np.asarray(q0, float), np.asarray(q1, float), np.asarray(t, float)
    tau = np.clip(t / T, 0.0, 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    sd = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T
    sdd = (60 * tau - 180 * tau**2 + 120 * tau**3) / T**2
    sd = np.where(t > T, 0.0, sd)
    sdd = np.where(t > T, 0.0, sdd)
    d = q1 - q0
    return q0 + np.outer(s, d), np.outer(sd, d), np.outer(sdd, d)


def cartesian_reference(X0, dp_s, T: float, t):
    """Straight-line tool motion with fixed orientation: X_d(t) = (R0, p0 + s(t) dp_s), quintic s(t), held after T.
    Returns (Xd list of 4x4, Vd (N, 6) body twists of X_d, dVd (N, 6) their time derivatives).
    Body twist of X_d = (R, p): omega_b = 0, v_b = R^T p'  (MR Eq. 3.83);  dv_b/dt = R^T p''."""
    X0 = np.asarray(X0, float)
    R0, p0 = X0[:3, :3], X0[:3, 3]
    z = np.zeros(3)
    pd, dpd, ddpd = quintic_reference(z, np.asarray(dp_s, float), T, t)
    Xd = [mr.RpToTrans(R0, p0 + p) for p in pd]
    Vd = np.hstack([np.zeros_like(dpd), dpd @ R0])          # rows: R0^T p'
    dVd = np.hstack([np.zeros_like(ddpd), ddpd @ R0])
    return Xd, Vd, dVd


def rk4_step(f, x, h: float) -> np.ndarray:
    k1 = f(x)
    k2 = f(x + 0.5 * h * k1)
    k3 = f(x + 0.5 * h * k2)
    k4 = f(x + h * k3)
    return x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def _pinocchio_model(robot: dict):
    """Pinocchio model from the MR lists (Lab 2 `pinocchio_model`, extended to a general spatial inertia G_i, e.g.
    with a payload).  Joint frame i: orientation of CoM frame {i} at home, origin on joint axis i."""
    import pinocchio as pin

    model = pin.Model()
    model.gravity.linear = np.asarray(robot["g"], float)
    Mi, J_prev, parent = np.eye(4), np.eye(4), 0
    for i in range(robot["n"]):
        Mi = Mi @ robot["Mlist"][i]
        R, p = mr.TransToRp(Mi)
        w, v = robot["Slist"][:3, i], robot["Slist"][3:, i]
        qa = np.cross(w, v)                                   # point on the axis (|w| = 1)
        J = mr.RpToTrans(R, qa)
        T_rel = mr.TransInv(J_prev) @ J
        jid = model.addJoint(parent, pin.JointModelRevoluteUnaligned(R.T @ w),
                             pin.SE3(T_rel[:3, :3], T_rel[:3, 3]), f"joint{i + 1}")
        G = np.asarray(robot["Glist"][i], float)
        m = G[3, 3]
        S = G[:3, 3:] / m
        c = np.array([S[2, 1], S[0, 2], S[1, 0]])            # CoM in {i}
        cx = mr.VecToso3(c)
        Ic = G[:3, :3] + m * cx @ cx                          # rotational inertia about the CoM, {i} axes
        model.appendBodyToJoint(jid, pin.Inertia(m, R.T @ (p - qa) + c, Ic), pin.SE3.Identity())
        J_prev, parent = J, jid
    return model


class Plant:
    """Forward dynamics of the simulated robot:  q'' = M(q)^-1 (tau - h(q, q')), h including viscous friction b q'.
    Uses Pinocchio's ABA when available (fast), else modern_robotics.ForwardDynamics (slow but identical)."""

    def __init__(self, robot: dict, use_pinocchio: bool = True):
        self.r = robot
        self.b = np.asarray(robot.get("b", np.zeros(robot["n"])), float)
        self.pin = None
        if use_pinocchio:
            try:
                import pinocchio as pin
                self.pin = pin
                self.model = _pinocchio_model(robot)
                self.data = self.model.createData()
            except ImportError:                                # pragma: no cover
                self.pin = None

    def accel(self, q, dq, tau) -> np.ndarray:
        tau_net = np.asarray(tau, float) - self.b * np.asarray(dq, float)
        if self.pin is not None:
            return np.array(self.pin.aba(self.model, self.data, np.asarray(q, float), np.asarray(dq, float), tau_net))
        r = self.r
        return mr.ForwardDynamics(q, dq, tau_net, r["g"], np.zeros(6), r["Mlist"], r["Glist"], r["Slist"])


def simulate_arm(plant: dict | Plant, controller, t, q0, dq0=None, substeps: int = 2, tau_limit=None) -> dict:
    """Closed-loop simulation (the Lab 3/Lab 5 loop for n joints).  controller.reset() first; at every sample k the
    controller sees the exact state and returns tau_k = controller(k, q_k, dq_k); the actuators apply tau_k clipped
    to +-tau_limit (if given), held constant until t_{k+1} (zero-order hold); the plant is integrated with
    `substeps` RK4 steps.  Returns dict(t, q, dq, tau) with arrays of shape (len(t), n)."""
    P = plant if isinstance(plant, Plant) else Plant(plant)
    t = np.asarray(t, float)
    n = len(q0)
    dt = t[1] - t[0]
    h = dt / substeps
    controller.reset()
    x = np.r_[np.asarray(q0, float), np.zeros(n) if dq0 is None else np.asarray(dq0, float)]
    Q, DQ, TAU = np.empty((len(t), n)), np.empty((len(t), n)), np.empty((len(t), n))
    lim = None if tau_limit is None else np.asarray(tau_limit, float)
    for k in range(len(t)):
        Q[k], DQ[k] = x[:n], x[n:]
        tau = np.asarray(controller(k, x[:n].copy(), x[n:].copy()), float)
        if lim is not None:
            tau = np.clip(tau, -lim, lim)
        TAU[k] = tau
        if k + 1 < len(t):
            f = lambda s, tau=tau: np.r_[s[n:], P.accel(s[:n], s[n:], tau)]  # noqa: E731
            for _ in range(substeps):
                x = rk4_step(f, x, h)
    return dict(t=t, q=Q, dq=DQ, tau=TAU)


def _diag(K, n: int) -> np.ndarray:
    """Gain given as a scalar (k I) or a vector (diag) -> vector of length n."""
    K = np.asarray(K, float)
    return np.full(n, float(K)) if K.ndim == 0 else K


# ==========================================================================================
# Part A -- gains
# ==========================================================================================
def ct_gains(zeta: float, wn: float):
    """Computed-torque PD gains (scalars, K = k I) giving each joint the error dynamics
    e'' + kd e' + kp e = 0 (MR Eq. 11.33 with Ki = 0) with the target zeta and wn:  kp = wn^2, kd = 2 zeta wn.
    Units s^-2 and s^-1 (acceleration gains: the model M~ turns them into torques).  Returns (kp, kd)."""
    # TODO: kp = wn^2, kd = 2 zeta wn
    raise NotImplementedError("ct_gains")


def ct_ki_max(kp: float, kd: float) -> float:
    """Stability bound for computed torque with integral action: the error dynamics (MR Eq. 11.33) has the
    characteristic polynomial s^3 + kd s^2 + kp s + ki, stable iff kp, kd > 0 and 0 < ki < kd kp."""
    # TODO: Routh condition for s^3 + kd s^2 + kp s + ki
    raise NotImplementedError("ct_ki_max")


def decentralized_gains(model: dict, q_ref, zeta: float, wn: float, ki_fraction: float = 0.0):
    """Independent-joint (decentralized, MR Sec. 11.4.2.1) PID gains: joint j is treated as the Lab 5 single joint
    with inertia M_jj = M~(q_ref)[j, j] and friction b_j = model['b'][j]:
        Kp_j = M_jj wn^2,  Kd_j = 2 zeta wn M_jj - b_j  (MR Eq. 11.26),  Ki_j = ki_fraction (b_j + Kd_j) Kp_j / M_jj
    (MR Eq. 11.30 bound).  Returns (Kp, Ki, Kd) as arrays of length n."""
    # TODO: diagonal of mr.MassMatrix at q_ref, then the Lab 5 pd_gains rule per joint
    raise NotImplementedError("decentralized_gains")


# ==========================================================================================
# Part B -- joint-space control laws
# ==========================================================================================
def model_inverse_dynamics(model: dict, q, dq, ddq) -> np.ndarray:
    """M~(q) ddq + h~(q, dq) of the controller model with ONE RNEA call (MR Sec. 8.3), h~ = c~ + g~ + b~ dq."""
    # TODO: mr.InverseDynamics of the model (Ftip = 0) plus model["b"] * dq
    raise NotImplementedError("model_inverse_dynamics")


def pid_torque(q, dq, eint, qd, dqd, Kp, Ki, Kd) -> np.ndarray:
    """Decentralized PID (MR Eq. 11.23 at every joint, no model):  tau = Kp e + Ki eint + Kd e',
    e = qd - q, e' = dqd - dq.  Gains: scalars or length-n vectors (diagonal matrices)."""
    # TODO: e = qd - q, e_dot = dqd - dq; use _diag() so scalar and vector gains both work
    raise NotImplementedError("pid_torque")


def pd_gravity_torque(q, dq, eint, qd, dqd, Kp, Ki, Kd, model: dict) -> np.ndarray:
    """PID control with gravity compensation (MR Eq. 11.38):  tau = Kp e + Ki eint + Kd e' + g~(q)."""
    # TODO: pid_torque(...) + mr.GravityForces of the MODEL at the measured q
    raise NotImplementedError("pd_gravity_torque")


def feedforward_pd_torque(q, dq, eint, qd, dqd, ddqd, Kp, Ki, Kd, model: dict) -> np.ndarray:
    """Feedforward plus decentralized PID (MR Eq. 11.32 plus feedback; Corke Sec. 9.4.3.1):
    tau = M~(qd) ddqd + h~(qd, dqd) + Kp e + Ki eint + Kd e'.  The model is evaluated on the REFERENCE."""
    # TODO: model_inverse_dynamics at the REFERENCE (qd, dqd, ddqd) + pid_torque(...)
    raise NotImplementedError("feedforward_pd_torque")


def computed_torque(q, dq, eint, qd, dqd, ddqd, kp, ki, kd, model: dict) -> np.ndarray:
    """Computed torque control (MR Eq. 11.37, pseudocode MR Fig. 11.20):
        tau = M~(q) (ddqd + Kp e + Ki eint + Kd e') + h~(q, dq),
    with gains scalars (k I, MR p. 434) or vectors (diagonal).  Hint: M~(q) a + h~(q, dq) is the inverse dynamics of
    the model at (q, dq, a) -- one RNEA call, no mass matrix needed."""
    # TODO: commanded acceleration a = ddqd + Kp e + Ki eint + Kd e_dot, then model_inverse_dynamics(model, q, dq, a)
    raise NotImplementedError("computed_torque")


def error_energy(q, dq, qd, Kp, model: dict) -> float:
    """Virtual error energy for setpoint control (MR Eq. 11.41, qd' = 0):
    V = 1/2 e^T Kp e + 1/2 dq^T M~(q) dq,  e = qd - q,  Kp scalar or vector (diagonal)."""
    # TODO: 0.5 e^T Kp e + 0.5 dq^T M(q) dq with mr.MassMatrix of the model
    raise NotImplementedError("error_energy")


class JointController:
    """Sampled joint-space controller for simulate_arm.

        ctrl = JointController(law, t, qd, dqd, ddqd, Kp, Ki, Kd, model=None)
        ctrl.reset();  tau = ctrl(k, q, dq)

    law is "pid" (pid_torque), "pd_gravity" (pd_gravity_torque), "feedforward" (feedforward_pd_torque) or
    "computed_torque" (computed_torque; then Kp, Ki, Kd are the acceleration gains kp, ki, kd).  t, qd, dqd, ddqd
    are the reference samples (arrays of len(t) rows).  At every call, as in MR Fig. 11.20:
        e = qd[k] - q ;  eint = eint + e dt (dt = t[1] - t[0]) ;  tau = law(q, dq, eint, qd[k], dqd[k], [ddqd[k],] ...)
    reset() sets eint = 0.  Attribute eint holds the current error integral."""

    LAWS = ("pid", "pd_gravity", "feedforward", "computed_torque")

    def __init__(self, law: str, t, qd, dqd, ddqd, Kp, Ki, Kd, model: dict | None = None):
        if law not in self.LAWS:
            raise ValueError(f"law must be one of {self.LAWS}")
        if law != "pid" and model is None:
            raise ValueError(f"law {law!r} needs a model")
        self.law = law
        self.t = np.asarray(t, float)
        self.dt = float(self.t[1] - self.t[0])
        self.qd, self.dqd, self.ddqd = (np.atleast_2d(np.asarray(x, float)) for x in (qd, dqd, ddqd))
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.model = model
        self.reset()

    def reset(self) -> None:
        # TODO: set the integral state to zeros
        raise NotImplementedError("reset")

    def __call__(self, k: int, q, dq) -> np.ndarray:
        # TODO: update the integral (as in MR Fig. 11.20), then call the control law
        raise NotImplementedError("__call__")


def run_metrics(t, qd, q, tau) -> dict:
    """Metrics of a run (MR Fig. 11.19):  with e(t) = qd(t) - q(t) (rows) and |.| the Euclidean norm,
        max_err   = max_t |e(t)|            (rad)
        rms_err   = sqrt(mean_t |e(t)|^2)   (rad)
        final_err = |e(t_end)|              (rad)
        effort    = int tau^T tau dt        (N^2 m^2 s, trapezoidal rule)
        peak_tau  = max_t |tau_j(t)| per joint (array, N m)."""
    # TODO: norms over joints with np.linalg.norm(..., axis=1); trapezoidal rule for the effort
    raise NotImplementedError("run_metrics")


# ==========================================================================================
# Part C -- task-space computed torque (MR Sec. 11.4.3)
# ==========================================================================================
def body_jacobian_dot(Blist, q, dq) -> np.ndarray:
    """Time derivative of the body Jacobian (MR Exercise 8.6b):  column i,
        d/dt J_bi = sum_{j > i} [ad_{J_bi}] J_bj dq_j,  with J_b = JacobianBody(Blist, q)."""
    # TODO: loop over columns i and j > i; mr.ad(J[:, i]) @ J[:, j] * dq[j]
    raise NotImplementedError("body_jacobian_dot")


def task_errors(X, V, Xd, Vd):
    """Configuration and velocity errors of MR Sec. 11.4.3 (p. 436), both in the end-effector frame at X:
        [X_e] = log(X^-1 X_d),   V_e = [Ad_{X^-1 X_d}] V_d - V.   Returns (Xe, Ve) as 6-vectors (omega first)."""
    # TODO: mr.TransInv, mr.MatrixLog6, mr.se3ToVec, mr.Adjoint
    raise NotImplementedError("task_errors")


def task_feedforward_accel(X, V, Xd, Vd, dVd) -> np.ndarray:
    """The feedforward term d/dt([Ad_{X^-1 X_d}] V_d) of MR Eq. 11.47 (body twists V of X and V_d of X_d).
    With A = X^-1 X_d one finds  d/dt(Ad_A V_d) = Ad_A dV_d - [ad_V] Ad_A V_d ;  at X = X_d, V = V_d it is dV_d."""
    # TODO: A = X^-1 Xd; W = Ad_A Vd; return Ad_A dVd - ad_V W
    raise NotImplementedError("task_feedforward_accel")


def model_task_space_dynamics(model: dict, q, dq):
    """Task-space dynamics of the model in the end-effector (body) frame (MR Eqs. 8.90-8.91), square J_b:
        Lambda = J^-T M J^-1,   eta = J^-T h(q, dq) - Lambda Jdot dq   (V = J dq).   Returns (Lambda, eta)."""
    # TODO: J = JacobianBody, M = MassMatrix, h = model_inverse_dynamics(model, q, dq, 0)
    raise NotImplementedError("model_task_space_dynamics")


def task_space_ct(q, dq, Xint, Xd, Vd, dVd, kp, ki, kd, model: dict) -> np.ndarray:
    """Task-space computed torque (MR Eq. 11.47):
        tau = J_b^T ( Lambda~ (a) + eta~ ),  a = d/dt([Ad_{X^-1 X_d}] V_d) + Kp X_e + Ki Xint + Kd V_e,
    with X = FKinBody(M, Blist, q), V = J_b dq, (X_e, V_e) = task_errors(...), Xint the integral of X_e, gains
    scalars or 6-vectors.  For a square, non-singular J_b this equals
        tau = M~(q) J_b^-1 (a - Jdot_b dq) + h~(q, dq)    (substitute MR Eqs. 8.90-8.91),
    i.e. the inverse dynamics of the model at the joint acceleration J_b^-1 (a - Jdot_b dq): one RNEA call."""
    # TODO: X, J, V; errors; a; joint acceleration np.linalg.solve(J, a - Jdot dq); one RNEA call
    raise NotImplementedError("task_space_ct")


class TaskController:
    """Sampled task-space computed-torque controller for simulate_arm.

        ctrl = TaskController(t, Xd, Vd, dVd, kp, ki, kd, model);  ctrl.reset();  tau = ctrl(k, q, dq)

    At every call: X = FKinBody(model M, Blist, q), V = J_b dq, Xe from task_errors, Xint = Xint + Xe dt, then
    tau = task_space_ct(q, dq, Xint, Xd[k], Vd[k], dVd[k], kp, ki, kd, model).  reset() sets Xint = 0."""

    def __init__(self, t, Xd, Vd, dVd, kp, ki, kd, model: dict):
        self.t = np.asarray(t, float)
        self.dt = float(self.t[1] - self.t[0])
        self.Xd, self.Vd, self.dVd = list(Xd), np.asarray(Vd, float), np.asarray(dVd, float)
        self.kp, self.ki, self.kd = kp, ki, kd
        self.model = model
        self.reset()

    def reset(self) -> None:
        # TODO: set the integral state to zeros
        raise NotImplementedError("reset")

    def __call__(self, k: int, q, dq) -> np.ndarray:
        # TODO: update the integral (as in MR Fig. 11.20), then call the control law
        raise NotImplementedError("__call__")
