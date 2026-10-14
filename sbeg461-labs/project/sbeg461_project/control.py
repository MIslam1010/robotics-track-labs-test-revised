"""Controllers. BASELINES are complete; the team's controllers are TODO stubs.

Every controller is a closure ctrl(t, q, qd) -> tau. `model` is the controller's model of the
robot (a Scenario, possibly with parameter error from Scenario.scaled / with_payload), which may
differ from the simulated plant.
"""
from __future__ import annotations

import numpy as np

from .dyn import gravity


# ----------------------------------------------------------------------------- kinematic helpers
def tip_position(model, q) -> np.ndarray:
    return model.fk(q)[:3, 3]


def tip_jacobian(model, q) -> np.ndarray:
    """3 x n Jacobian of the {b}-origin velocity in {s} coordinates: p_dot = R_sb J_v,b q_dot."""
    T = model.fk(q)
    return T[:3, :3] @ model.jacobian_body(q)[3:, :]


# ----------------------------------------------------------------------------- baselines
def pd_gravity(model, Kp, Kd, ref, g=None):
    """BASELINE PD + gravity compensation, MR Eq. 11.38 with K_i = 0:
    tau = Kp (q_d - q) + Kd (qd_d - qd) + g~(q). `ref(t)` returns (q_d, qd_d, qdd_d)."""
    Kp, Kd = np.atleast_1d(Kp), np.atleast_1d(Kd)

    def ctrl(t, q, qd):
        q_d, v_d, _ = ref(t)
        return Kp * (q_d - q) + Kd * (v_d - qd) + gravity(model, q, g)
    return ctrl


def task_stiffness(model, K, B, x_ref, dims=(0, 1, 2), gravity_comp=True, K_rot=0.0, B_rot=0.0):
    """BASELINE task-space stiffness/damping without inertia compensation (a 'simple impedance'):
    tau = J_v^T (K (x_d - x) + B (xd_d - xd)) [+ K_rot (phi_d - phi) - B_rot phi_dot for planar
    robots, phi = sum(q)] + g~(q).
    x_ref(t) returns (x_d, xd_d, xdd_d) for the coordinates `dims` of the tip position in {s};
    for planar robots x_ref may also return phi_d as a 4th item."""
    K = np.broadcast_to(np.asarray(K, float), (len(dims),))
    B = np.broadcast_to(np.asarray(B, float), (len(dims),))
    dims = list(dims)

    def ctrl(t, q, qd):
        r = x_ref(t)
        xd_, vd_ = np.asarray(r[0], float), np.asarray(r[1], float)
        Jv = tip_jacobian(model, q)[dims, :]
        x = tip_position(model, q)[dims]
        v = Jv @ qd
        f = K * (xd_ - x) + B * (vd_ - v)
        tau = Jv.T @ f
        if K_rot or B_rot:
            phi_d = r[3] if len(r) > 3 else 0.0
            tau = tau + np.ones(model.n) * (K_rot * (phi_d - np.sum(q)) - B_rot * np.sum(qd))
        if gravity_comp:
            tau = tau + gravity(model, q)
        return tau
    return ctrl


def tunnel_stiffness(model, A, B, Kn, Bn, Bt, dims=(0, 1)):
    """BASELINE S1 virtual tunnel: spring Kn and damper Bn normal to the segment AB, light viscosity
    Bt along it, no inertia compensation. Force on the hand f = -Kn d n - Bn (n.v) n - Bt (t.v) t,
    d = signed distance to the line, n = unit normal, t = unit tangent."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    tvec = (B - A) / np.linalg.norm(B - A)
    nvec = np.array([-tvec[1], tvec[0]])
    dims = list(dims)

    def ctrl(t, q, qd):
        Jv = tip_jacobian(model, q)[dims, :]
        x = tip_position(model, q)[dims]
        v = Jv @ qd
        d = (x - A) @ nvec
        f = -Kn * d * nvec - Bn * (v @ nvec) * nvec - Bt * (v @ tvec) * tvec
        return Jv.T @ f + gravity(model, q)
    return ctrl


# ----------------------------------------------------------------------------- team work (TODO)
def computed_torque(model, Kp, Kd, ref, Ki=0.0):
    """TODO (team): computed-torque controller, MR Eq. 11.37:
    tau = M~(q) (qdd_d + Kp e + Ki int(e) + Kd e_dot) + h~(q, qd)."""
    raise NotImplementedError("computed_torque: team implementation required")


def task_space_computed_torque(model, Kp, Kd, X_ref):
    """TODO (team): task-space motion control, MR Eq. 11.47, with X_e = log(X^-1 X_d) and
    V_e = [Ad_{X^-1 X_d}] V_d - V. X_ref(t) returns (X_d, V_d, Vdot_d) with twists in {d}."""
    raise NotImplementedError("task_space_computed_torque: team implementation required")


def impedance(model, Mv, Bv, Kv, x_ref):
    """TODO (team): impedance control, MR Eq. 11.64-11.65, rendering M x'' + B x' + K x = f_ext
    about the reference x_ref(t). Discuss the M = 0 simplification and stiff-environment limits."""
    raise NotImplementedError("impedance: team implementation required")


def admittance(model, Mv, Bv, Kv, f_sensed, x_ref):
    """TODO (team, optional): admittance control, MR Eq. 11.66, using a sensed wrist force."""
    raise NotImplementedError("admittance: team implementation required")
