"""SBEG 461 Lab 2 -- Newton-Euler inverse dynamics (RNEA), verified against modern_robotics and Pinocchio.  STARTER CODE.

Fill in every function marked # TODO (checkpoints 1-6, in order), then run
    LAB_IMPL=starter pytest -q test_lab.py
Functions marked "(Provided.)" are complete -- read them, do not change them.
You may NOT call mr.InverseDynamics, mr.MassMatrix, ... or Pinocchio inside checkpoints 1-4 and 6: those are the
references your code is tested against.  You MAY use the modern_robotics helpers Adjoint, ad, TransInv,
MatrixExp6, VecTose3, RpToTrans, TransToRp (or the equivalents from your own SBEG 361 library).

Notation follows Modern Robotics (MR) Sections 8.2-8.3:
    twists V = (omega, v), wrenches F = (m, f)  -- ANGULAR FIRST (Pinocchio uses LINEAR first!)
    G_b = [[I_b, 0], [0, m I]]                              (MR Eq. 8.32)
    G_a = [Ad_{T_ba}]^T G_b [Ad_{T_ba}]                      (MR Eq. 8.42)
    F_b = G_b Vdot_b - [ad_{V_b}]^T G_b V_b                  (MR Eq. 8.40)
    A_i = [Ad_{M_i^{-1}}] S_i,   M_i = M_{0,i}
    T_{i,i-1} = exp(-[A_i] theta_i) M_{i,i-1}                (MR Eq. 8.50)
    V_i = [Ad_{T_{i,i-1}}] V_{i-1} + A_i dtheta_i            (MR Eq. 8.51)
    Vdot_i = [Ad_{T_{i,i-1}}] Vdot_{i-1} + [ad_{V_i}] A_i dtheta_i + A_i ddtheta_i      (MR Eq. 8.52)
    F_i = [Ad_{T_{i+1,i}}]^T F_{i+1} + G_i Vdot_i - [ad_{V_i}]^T G_i V_i               (MR Eq. 8.53)
    tau_i = F_i^T A_i                                       (MR Eq. 8.54)
    V_0 = 0,  Vdot_0 = (0, -g)  (gravity as a fictitious base acceleration),  F_{n+1} = F_tip
F_tip is the wrench the END-EFFECTOR APPLIES TO THE ENVIRONMENT, expressed in {n+1} (MR p. 292).

Robot data (data/robots.json) is in modern_robotics list format: Mlist = [M_{0,1}, ..., M_{n,n+1}],
Glist = [G_1, ..., G_n] (centre-of-mass frames), Slist = space screw axes as columns, g = gravity in {0}.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import modern_robotics as mr

DATA = Path(__file__).resolve().parent / "data"


# ===================================================================================== provided helpers
def load_robot(name: str) -> dict:
    """Return robot `name` from data/robots.json as numpy arrays.  (Provided.)

    Keys: Mlist (list of n+1 4x4), Glist (list of n 6x6), Slist (6 x n), g (3,), n, description.
    """
    with open(DATA / "robots.json") as fh:
        r = json.load(fh)[name]
    Slist = np.array(r["Slist"], float)
    return {"Mlist": [np.array(M, float) for M in r["Mlist"]],
            "Glist": [np.array(G, float) for G in r["Glist"]],
            "Slist": Slist, "g": np.array(r["g"], float), "n": Slist.shape[1],
            "description": r["description"], "name": name}


def load_states(name: str) -> list:
    """Return the test states of robot `name` from data/states.csv as (theta, dtheta, ddtheta, Ftip).  (Provided.)"""
    out = []
    with open(DATA / "states.csv") as fh:
        for row in csv.reader(line for line in fh if not line.startswith("#")):
            if row[0] != name:
                continue
            x = np.array([float(v) for v in row[1:]])
            n = (len(x) - 6) // 3
            out.append((x[:n], x[n:2 * n], x[2 * n:3 * n], x[3 * n:]))
    return out


def load_trajectory(fname: str):
    """Read data/<fname> (columns t, theta1..n, dtheta1..n, ddtheta1..n).  Returns t, theta, dtheta, ddtheta.  (Provided.)"""
    rows = np.loadtxt(DATA / fname, delimiter=",", comments="#", skiprows=2)
    n = (rows.shape[1] - 1) // 3
    return rows[:, 0], rows[:, 1:1 + n], rows[:, 1 + n:1 + 2 * n], rows[:, 1 + 2 * n:]


# ===================================================================================== checkpoint 1: one rigid body
def spatial_inertia(I: np.ndarray, m: float) -> np.ndarray:
    """6x6 spatial inertia G_b of a body in a centre-of-mass frame (MR Eq. 8.32)."""
    # TODO: Build a 6x6 zero matrix and fill the two diagonal blocks.
    raise NotImplementedError("spatial_inertia")


def transform_spatial_inertia(G_b: np.ndarray, T_ba: np.ndarray) -> np.ndarray:
    """Spatial inertia G_a in frame {a}, given G_b in {b} and T_ba = configuration of {a} seen from {b} (MR Eq. 8.42)."""
    # TODO: One line with mr.Adjoint(T_ba).
    raise NotImplementedError("transform_spatial_inertia")


def rigid_body_wrench(G: np.ndarray, V: np.ndarray, Vdot: np.ndarray) -> np.ndarray:
    """Net wrench F = G Vdot - [ad_V]^T G V on one rigid body (MR Eq. 8.40; valid in any body-fixed frame, Eq. 8.43)."""
    # TODO: mr.ad(V) returns [ad_V]; remember the TRANSPOSE in the gyroscopic term.
    raise NotImplementedError("rigid_body_wrench")


# ===================================================================================== checkpoint 2: forward iterations
def link_screw_axes(Mlist, Slist) -> np.ndarray:
    """6 x n matrix whose column i is A_i = [Ad_{M_i^{-1}}] S_i, the joint-i screw axis in link frame {i}."""
    # TODO: Accumulate M_i = M_{0,1} M_{1,2} ... M_{i-1,i} in a loop; use mr.Adjoint and mr.TransInv.
    raise NotImplementedError("link_screw_axes")


def forward_iterations(theta, dtheta, ddtheta, g, Mlist, Slist) -> dict:
    """Base-to-tip pass of the RNEA (MR Eqs. 8.50-8.52).

    Returns a dict with
        "A":    6 x n link-frame screw axes,
        "T":    list of n+1 transforms; T[i] = T_{i+1,i} in MR's 1-based numbering for i < n, T[n] = M_{n+1,n},
        "V":    6 x (n+1), column 0 = V_0 = 0, column i = V_i,
        "Vdot": 6 x (n+1), column 0 = Vdot_0 = (0, -g), column i = Vdot_i.
    """
    # TODO: Loop i = 0..n-1 (Python indexing!).  Column 0 of V and Vdot is the base.  mr.MatrixExp6(mr.VecTose3(-A_i*theta_i)) gives exp(-[A_i] theta_i).  Do not forget T[n] = M_{n+1,n} = TransInv(Mlist[n]).
    raise NotImplementedError("forward_iterations")


# ===================================================================================== checkpoint 3: backward iterations
def backward_iterations(fwd: dict, Ftip, Glist):
    """Tip-to-base pass (MR Eqs. 8.53-8.54).

    Returns (tau, F): tau (n,), F 6 x (n+1) with column i-1 = F_i (wrench transmitted through joint i to link i,
    in {i}) for i = 1..n and the last column = F_{n+1} = Ftip.
    """
    # TODO: Loop i = n-1 down to 0.  The wrench from link i+1 is transported with [Ad_{T_{i+1,i}}]^T = mr.Adjoint(T[i+1]).T.  Use your rigid_body_wrench.
    raise NotImplementedError("backward_iterations")


def rnea(theta, dtheta, ddtheta, g, Ftip, Mlist, Glist, Slist) -> np.ndarray:
    """Joint forces/torques tau = M(theta) ddtheta + c(theta, dtheta) + g(theta) + J_b^T(theta) Ftip (MR Section 8.3.2)."""
    # TODO: Two lines: forward_iterations then backward_iterations.
    raise NotImplementedError("rnea")


# ===================================================================================== checkpoint 4: M, c, g and J^T F by RNEA calls
def dynamics_terms(theta, dtheta, g, Mlist, Glist, Slist):
    """Return (M, c, grav) using only calls to `rnea` (Featherstone Section 5.3, p. 94):
        grav = rnea(theta, 0, 0, g, 0);  c = rnea(theta, dtheta, 0, 0, 0);  column j of M = rnea(theta, 0, e_j, 0, 0).
    """
    # TODO: n + 2 calls of your rnea; switch gravity off (np.zeros(3)) where needed.
    raise NotImplementedError("dynamics_terms")


def tip_wrench_torques(theta, Ftip, Mlist, Glist, Slist) -> np.ndarray:
    """Joint torques that balance the tip wrench alone (= J_b^T Ftip, MR Eq. 5.26), computed with one RNEA call."""
    # TODO: One rnea call with zero velocity, acceleration and gravity.
    raise NotImplementedError("tip_wrench_torques")


# ===================================================================================== checkpoint 5: Pinocchio
def pinocchio_model(robot: dict):
    """Build a Pinocchio model from MR lists.  (Provided -- read it; you will need T_jn_ee.)

    For joint i we create a Pinocchio joint frame J_i with the ORIENTATION of link frame {i} at home and its ORIGIN
    on the joint axis (revolute) or at the link CoM (prismatic).  The CoM of link i is then at R_i^T (p_i - q_i) in J_i
    and the rotational inertia block of G_i can be used unchanged.  Returns (model, info) with
        info["T_jn_ee"] = configuration of the end-effector frame {n+1} seen from the last joint frame J_n (constant).
    Zero-pitch joints only.  Pinocchio stores spatial vectors LINEAR first.
    """
    import pinocchio as pin

    model = pin.Model()
    model.gravity.linear = np.asarray(robot["g"], float)
    Mi, J_prev, parent = np.eye(4), np.eye(4), 0
    for i in range(robot["n"]):
        Mi = Mi @ robot["Mlist"][i]
        R, p = mr.TransToRp(Mi)
        w, v = robot["Slist"][:3, i], robot["Slist"][3:, i]
        if np.linalg.norm(w) > 1e-12:
            q = np.cross(w, v)                                   # point on the axis closest to {0} (|w| = 1)
            jmodel = pin.JointModelRevoluteUnaligned(R.T @ w)
        else:
            q = p
            jmodel = pin.JointModelPrismaticUnaligned(R.T @ v)
        J = mr.RpToTrans(R, q)
        T_rel = mr.TransInv(J_prev) @ J
        jid = model.addJoint(parent, jmodel, pin.SE3(T_rel[:3, :3], T_rel[:3, 3]), f"joint{i + 1}")
        G = robot["Glist"][i]
        model.appendBodyToJoint(jid, pin.Inertia(float(G[3, 3]), R.T @ (p - q), G[:3, :3]), pin.SE3.Identity())
        J_prev, parent = J, jid
    M_ee = Mi @ robot["Mlist"][robot["n"]]
    return model, {"T_jn_ee": mr.TransInv(J_prev) @ M_ee}


def pinocchio_inverse_dynamics(model, info: dict, theta, dtheta, ddtheta, Ftip=None) -> np.ndarray:
    """tau from pin.rnea for the model built by `pinocchio_model`, including the MR tip wrench Ftip (in {n+1}).

    Pinocchio's fext[j] is the wrench applied ON body j, expressed in joint frame j, LINEAR first.
    MR's Ftip is the wrench applied BY the end-effector, in {n+1}, ANGULAR first.  So:
        F_on = -Ftip (in {n+1});  F_J = [Ad_{T_ee,Jn}]^T F_on (MR Eq. 3.98);  fext[n] = Force(linear=F_J[3:], angular=F_J[:3]).
    """
    import pinocchio as pin  # noqa: F401  (needed once you fill this in)

    # TODO: data = model.createData(); fext = [pin.Force.Zero() for _ in range(model.njoints)]; pin.Force(linear, angular).  Mind the three convention changes listed above.
    raise NotImplementedError("pinocchio_inverse_dynamics")


# ===================================================================================== checkpoint 6: application
def torque_profile(robot: dict, theta, dtheta, ddtheta, Ftip=None) -> np.ndarray:
    """N x n joint-torque history along a trajectory (rows = time samples), using your `rnea`."""
    # TODO: Loop over the rows (time samples) and call your rnea.
    raise NotImplementedError("torque_profile")


def peak_torques(tau: np.ndarray) -> np.ndarray:
    """Peak absolute torque of each joint over a trajectory (max over rows of |tau|)."""
    # TODO: np.max over axis 0 of np.abs(...).
    raise NotImplementedError("peak_torques")


# ===================================================================================== provided: timing for the reflection
def time_rnea(robot: dict, n_calls: int = 2000, impl=None) -> dict:
    """Mean time per inverse-dynamics call (microseconds) for your rnea, modern_robotics and Pinocchio.  (Provided.)"""
    impl = impl or rnea
    rng = np.random.default_rng(0)
    n = robot["n"]
    x = rng.normal(size=(3, n))
    args = (robot["g"], np.zeros(6), robot["Mlist"], robot["Glist"], robot["Slist"])
    out = {}
    for name, f in (("yours", lambda: impl(x[0], x[1], x[2], *args)),
                    ("modern_robotics", lambda: mr.InverseDynamics(x[0], x[1], x[2], *args))):
        t0 = time.perf_counter()
        for _ in range(n_calls):
            f()
        out[name] = 1e6 * (time.perf_counter() - t0) / n_calls
    try:
        import pinocchio as pin
        model, _ = pinocchio_model(robot)
        data = model.createData()
        t0 = time.perf_counter()
        for _ in range(n_calls):
            pin.rnea(model, data, x[0], x[1], x[2])
        out["pinocchio"] = 1e6 * (time.perf_counter() - t0) / n_calls
    except ImportError:
        pass
    return out


if __name__ == "__main__":
    for name in ("mr_2r_point", "arm_exo_2r", "eod_3r", "ur5"):
        rb = load_robot(name)
        print(f"{name:12s}", {k: f"{v:.1f} us" for k, v in time_rnea(rb, 500).items()})
