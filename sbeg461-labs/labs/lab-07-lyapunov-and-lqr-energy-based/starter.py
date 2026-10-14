"""SBEG 461 Lab 7 -- why PD + gravity compensation works (energy as a Lyapunov function, passivity) and optimal gains
(LQR) for the linearised planar 2R arm and the course arm.
STARTER: write every function marked TODO (Parts A and B); the provided section is complete.

Notation follows Modern Robotics (MR): dynamics tau = M(theta) theta'' + C(theta, theta') theta' + g(theta)
(MR Sec. 8.1.2, Eqs. 8.18-8.19) plus viscous friction B theta' in the plant; theta_e = theta_d - theta; in code
q = theta.  "model" is the CONTROLLER's model (M~, g~); "plant" is the simulated robot.  Units: SI (s, rad, N m, J).
Linear systems follow Astrom-Murray Sec. 6.3: x' = A x + B u, cost J = int (x^T Q x + u^T R u) dt
(Astrom-Murray Eq. 6.26; Q_x = Q, Q_u = R), algebraic Riccati equation (Astrom-Murray Eq. 6.27).

Parts
  A. Passivity and Lyapunov: partial derivatives of M, the Coriolis matrix from the Christoffel symbols
     (MR Eqs. 8.17, 8.19), M' and the skew-symmetry of M' - 2C (MR Proposition 8.1), the stored energy
     (kinetic + potential) of the passive manipulator, PD setpoint control with gravity compensation (MR Eq. 11.38,
     K_i = 0), the error energy V (MR Eq. 11.41) and its rate (MR Eqs. 11.42-11.43), and the equilibrium reached
     with a wrong gravity model
  B. LQR: linearisation of the manipulator equations about an equilibrium (Tedrake Sec. 3.4.1), Bryson's rule
     (Astrom-Murray Exercise 6.15), the Lyapunov equation A^T P + P A + Q = 0, LQR from the Riccati equation,
     Kleinman's iteration (Newton's method for the Riccati equation, started from the PD gains), the quadratic cost
     of any stabilising gain, and the state-feedback torque on the nonlinear arm

These functions become `sbeg461_dynamics/stability.py` in your cumulative library.
"""
from __future__ import annotations

import json
from pathlib import Path

import modern_robotics as mr
import numpy as np
import scipy.linalg

import robots

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


# ==========================================================================================
# Provided (do not edit): configuration, robots, dynamics, simulator
# ==========================================================================================
def load_config() -> dict:
    """data/lab07_config.json (illustrative values; see make_data.py)."""
    return json.loads((DATA / "lab07_config.json").read_text())


def plant_2r(cfg: dict | None = None, friction: bool = True) -> dict:
    """The planar 2R plant with the configured viscous friction (friction=False: b = 0)."""
    cfg = load_config() if cfg is None else cfg
    return robots.planar_2r(cfg["planar_2r"]["b"] if friction else (0.0, 0.0))


def plant_course_arm(cfg: dict | None = None) -> dict:
    """The course arm (no payload) with the configured viscous friction."""
    cfg = load_config() if cfg is None else cfg
    return robots.course_arm_model(cfg["course_arm"]["b"])


def scaled_model(robot: dict, mass_scale: float) -> dict:
    """A copy of `robot` whose masses and inertias are multiplied by mass_scale (a wrong controller model)."""
    r = dict(robot)
    r["Glist"] = [mass_scale * np.asarray(G, float) for G in robot["Glist"]]
    if robot.get("closed_form"):
        cf = dict(robot["closed_form"])
        for k in ("m1", "m2", "I1", "I2"):
            cf[k] = mass_scale * cf[k]
        r["closed_form"] = cf
    return r


def mass_matrix(robot: dict, q) -> np.ndarray:
    """M(q): closed form for the planar 2R, mr.MassMatrix otherwise."""
    q = np.asarray(q, float)
    cf = robot.get("closed_form")
    if cf:
        c2 = np.cos(q[1])
        m22 = cf["I2"] + cf["m2"] * cf["lc2"] ** 2
        m12 = m22 + cf["m2"] * cf["L1"] * cf["lc2"] * c2
        m11 = (cf["I1"] + cf["m1"] * cf["lc1"] ** 2 + cf["I2"]
               + cf["m2"] * (cf["L1"] ** 2 + cf["lc2"] ** 2 + 2 * cf["L1"] * cf["lc2"] * c2))
        return np.array([[m11, m12], [m12, m22]])
    return mr.MassMatrix(q, robot["Mlist"], robot["Glist"], robot["Slist"])


def coriolis_vector(robot: dict, q, dq) -> np.ndarray:
    """c(q, dq) = C(q, dq) dq (Coriolis and centripetal torques): closed form for the 2R, mr.VelQuadraticForces else."""
    q, dq = np.asarray(q, float), np.asarray(dq, float)
    cf = robot.get("closed_form")
    if cf:
        hh = -cf["m2"] * cf["L1"] * cf["lc2"] * np.sin(q[1])
        return np.array([hh * (2 * dq[0] * dq[1] + dq[1] ** 2), -hh * dq[0] ** 2])
    return mr.VelQuadraticForces(q, dq, robot["Mlist"], robot["Glist"], robot["Slist"])


def gravity_torque(robot: dict, q) -> np.ndarray:
    """g(q) = dP/dq: closed form for the 2R (gravity along -y), mr.GravityForces otherwise."""
    q = np.asarray(q, float)
    cf = robot.get("closed_form")
    if cf:
        c1, c12 = np.cos(q[0]), np.cos(q[0] + q[1])
        g2 = cf["m2"] * cf["lc2"] * cf["g0"] * c12
        return np.array([(cf["m1"] * cf["lc1"] + cf["m2"] * cf["L1"]) * cf["g0"] * c1 + g2, g2])
    return mr.GravityForces(q, robot["g"], robot["Mlist"], robot["Glist"], robot["Slist"])


def potential_energy(robot: dict, q) -> float:
    """P(q) = -sum_i m_i g^T p_i(q), p_i = position of the CoM of link i (zero level: the origin of {s})."""
    q = np.asarray(q, float)
    cf = robot.get("closed_form")
    if cf:
        y1, y12 = np.sin(q[0]), np.sin(q[0] + q[1])
        return float(cf["g0"] * (cf["m1"] * cf["lc1"] * y1 + cf["m2"] * (cf["L1"] * y1 + cf["lc2"] * y12)))
    T, Mhome, P = np.eye(4), np.eye(4), 0.0
    for i in range(robot["n"]):
        T = T @ mr.MatrixExp6(mr.VecTose3(robot["Slist"][:, i] * q[i]))
        Mhome = Mhome @ robot["Mlist"][i]
        P -= robot["Glist"][i][3, 3] * float(np.dot(robot["g"], (T @ Mhome)[:3, 3]))
    return P


def forward_dynamics(robot: dict, q, dq, tau) -> np.ndarray:
    """q'' = M(q)^-1 (tau - c(q, dq) - g(q) - b * dq) (MR Eq. 8.1 with viscous friction tau_fric = b dq, MR Eq. 11.20)."""
    dq = np.asarray(dq, float)
    rhs = np.asarray(tau, float) - coriolis_vector(robot, q, dq) - gravity_torque(robot, q) - robot["b"] * dq
    return np.linalg.solve(mass_matrix(robot, q), rhs)


def pd_gains_from_inertia(robot: dict, q, zeta: float, wn: float):
    """Diagonal PD gains from the diagonal of M(q) (Lab 5 rule per joint): Kp = diag(M_jj wn^2),
    Kd = diag(2 zeta wn M_jj - b_j).  Returns (Kp, Kd) as (n, n) matrices."""
    m = np.diag(mass_matrix(robot, q))
    return np.diag(m * wn**2), np.diag(2 * zeta * wn * m - robot["b"])


def is_hurwitz(A) -> bool:
    """True if every eigenvalue of A has a negative real part (x' = A x is asymptotically stable)."""
    return bool(np.max(np.linalg.eigvals(np.asarray(A, float)).real) < 0.0)


def time_grid(T: float, dt: float) -> np.ndarray:
    n = int(round(T / dt))
    return np.arange(n + 1) * dt


def rk4_step(f, x, h: float) -> np.ndarray:
    k1 = f(x)
    k2 = f(x + 0.5 * h * k1)
    k3 = f(x + 0.5 * h * k2)
    k4 = f(x + h * k3)
    return x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(plant: dict, policy, q0, dq0, T: float, dt: float) -> dict:
    """Closed loop: at t_k the controller computes tau_k = policy(t_k, q_k, dq_k), held over [t_k, t_k + dt]
    (zero-order hold, 1 / dt Hz); the plant is integrated with one RK4 step per sample.  Returns a dict with
    t (N+1,), q, dq (N+1, n) and tau (N+1, n) (the last row is the torque the controller would apply at T)."""
    n = plant["n"]
    t = time_grid(T, dt)
    x = np.r_[np.asarray(q0, float), np.asarray(dq0, float)]
    X = np.zeros((len(t), 2 * n))
    U = np.zeros((len(t), n))
    for k, tk in enumerate(t):
        X[k] = x
        U[k] = policy(tk, x[:n], x[n:])
        if k + 1 < len(t):
            u = U[k]
            x = rk4_step(lambda z: np.r_[z[n:], forward_dynamics(plant, z[:n], z[n:], u)], x, dt)
    return dict(t=t, q=X[:, :n], dq=X[:, n:], tau=U)


# ==========================================================================================
# Part A -- passivity of the manipulator and the Lyapunov function of PD + gravity compensation
# ==========================================================================================
def mass_matrix_partials(robot: dict, q, h: float = 1e-6) -> np.ndarray:
    """dM[k] = dM/dq_k at q by central differences of mass_matrix (step h).  Returns an (n, n, n) array."""
    # TODO: for each k: e = h * unit vector k; dM[k] = (mass_matrix(robot, q + e) - mass_matrix(robot, q - e)) / (2 h)
    # TODO: return an (n, n, n) array (np.zeros((n, n, n)) to start)
    raise NotImplementedError("mass_matrix_partials")


def coriolis_matrix(robot: dict, q, dq, h: float = 1e-6) -> np.ndarray:
    """Coriolis matrix C(q, dq) with c_ij = sum_k Gamma_ijk dq_k (MR Eq. 8.19), Christoffel symbols
    Gamma_ijk = 1/2 (dm_ij/dq_k + dm_ik/dq_j - dm_jk/dq_i) (MR Eq. 8.17) from mass_matrix_partials."""
    # TODO: dM = mass_matrix_partials(robot, q, h); dM[k, i, j] = d m_ij / d q_k
    # TODO: Gamma[i, j, k] = 0.5 * (dM[k, i, j] + dM[j, i, k] - dM[i, j, k])   (loops are fine for n <= 6)
    # TODO: C[i, j] = sum_k Gamma[i, j, k] dq[k]
    raise NotImplementedError("coriolis_matrix")


def mass_matrix_dot(robot: dict, q, dq, h: float = 1e-6) -> np.ndarray:
    """M'(q, dq) = dM/dt = sum_k (dM/dq_k) dq_k."""
    # TODO: sum over k of mass_matrix_partials(...)[k] * dq[k]
    raise NotImplementedError("mass_matrix_dot")


def stored_energy(robot: dict, q, dq) -> float:
    """Total mechanical energy E = 1/2 dq^T M(q) dq + P(q) (kinetic energy MR Eq. 8.14 + potential energy; L = K - P, MR Eq. 8.13)."""
    # TODO: kinetic 1/2 dq^T M dq (mass_matrix) + potential_energy(robot, q) (provided)
    raise NotImplementedError("stored_energy")


def pd_gravity_setpoint_torque(q, dq, qd, Kp, Kd, model: dict) -> np.ndarray:
    """PD setpoint control with gravity compensation (MR Eq. 11.38 with K_i = 0 and dqd = 0):
    tau = Kp (qd - q) - Kd dq + g~(q).  Kp, Kd are (n, n) gain matrices; g~ from the MODEL."""
    # TODO: Kp @ (qd - q) - Kd @ dq + gravity_torque(model, q)   (the MODEL's gravity, at the measured q)
    raise NotImplementedError("pd_gravity_setpoint_torque")


def lyapunov_energy(plant: dict, q, dq, qd, Kp) -> float:
    """Error energy V = 1/2 theta_e^T Kp theta_e + 1/2 dq^T M(q) dq (MR Eq. 11.41), M of the PLANT."""
    # TODO: e = qd - q;  0.5 e^T Kp e + 0.5 dq^T M(q) dq with mass_matrix(plant, q)
    raise NotImplementedError("lyapunov_energy")


def lyapunov_rate(plant: dict, q, dq, qd, Kp, tau) -> float:
    """dV/dt along the plant for the applied torque tau, from the first line of MR Eq. 11.42:
    V' = -dq^T Kp theta_e + dq^T M(q) q'' + 1/2 dq^T M'(q, dq) dq, with q'' = forward_dynamics(plant, q, dq, tau)."""
    # TODO: ddq = forward_dynamics(plant, q, dq, tau)
    # TODO: -dq^T Kp e + dq^T M ddq + 0.5 dq^T mass_matrix_dot(plant, q, dq) dq   (return a float)
    raise NotImplementedError("lyapunov_rate")


def equilibrium_offset(plant: dict, model: dict, qd, Kp, iters: int = 50, tol: float = 1e-12,
                       h: float = 1e-6) -> np.ndarray:
    """Rest configuration q_ss of PD + g~ when the model gravity is wrong: solve
    F(q) = Kp (qd - q) + g~(q) - g(q) = 0 by Newton's method from q = qd (Jacobian by central differences, step h).
    Raises ValueError if |F| > 1e-9 after `iters` iterations."""
    # TODO: F(q) = Kp @ (qd - q) + gravity_torque(model, q) - gravity_torque(plant, q)
    # TODO: Newton: q = qd; repeat: if |F(q)| < tol return q; J[:, k] = (F(q + h e_k) - F(q - h e_k)) / (2h);
    # TODO:         q = q - np.linalg.solve(J, F(q));  after the loop: raise ValueError if |F(q)| > 1e-9
    raise NotImplementedError("equilibrium_offset")


# ==========================================================================================
# Part B -- LQR for the linearised arm
# ==========================================================================================
def linearize(plant: dict, q_eq, h: float = 1e-6):
    """Linearise the plant about the equilibrium (q_eq, dq = 0, u_eq = g(q_eq)) (Tedrake Sec. 3.4.1):
    x = (q - q_eq, dq), u = tau - u_eq,  x' = A x + B u  with
    A = [[0, I], [-M^-1 dg/dq, -M^-1 diag(b)]],  B = [[0], [M^-1]]  (M, dg/dq at q_eq; dg/dq by central differences).
    Returns (A (2n, 2n), B (2n, n), u_eq (n,))."""
    # TODO: G[:, k] = (gravity_torque(plant, q_eq + h e_k) - gravity_torque(plant, q_eq - h e_k)) / (2h)
    # TODO: Minv = np.linalg.inv(mass_matrix(plant, q_eq));  use np.block / np.vstack and np.diag(plant['b'])
    # TODO: return A, B, gravity_torque(plant, q_eq)
    raise NotImplementedError("linearize")


def bryson_weights(x_max, u_max):
    """Bryson's rule (Astrom-Murray Exercise 6.15): Q = diag(1 / x_max^2), R = diag(1 / u_max^2)."""
    # TODO: np.diag(1 / x_max**2), np.diag(1 / u_max**2)  (convert to float arrays first)
    raise NotImplementedError("bryson_weights")


def lyapunov_solve(A, Q) -> np.ndarray:
    """Solve the Lyapunov equation A^T P + P A + Q = 0 for P with the Kronecker (vec) form
    (I kron A^T + A^T kron I) vec(P) = -vec(Q)  (vec stacks COLUMNS).  Returns the symmetric part of P."""
    # TODO: L = np.kron(I, A.T) + np.kron(A.T, I);  vec(Q) = Q.reshape(-1, order='F')
    # TODO: P = np.linalg.solve(L, -vec(Q)).reshape(n, n, order='F');  return 0.5 (P + P^T)
    # TODO: (do NOT call scipy here: the test compares with scipy.linalg.solve_continuous_lyapunov)
    raise NotImplementedError("lyapunov_solve")


def lqr(A, B, Q, R):
    """Infinite-horizon LQR (Astrom-Murray Eqs. 6.26-6.27): P from the algebraic Riccati equation
    A^T P + P A - P B R^-1 B^T P + Q = 0 (scipy.linalg.solve_continuous_are), K = R^-1 B^T P.  Returns (K, P)."""
    # TODO: P = scipy.linalg.solve_continuous_are(A, B, Q, R) (symmetrise it);  K = np.linalg.solve(R, B.T @ P)
    raise NotImplementedError("lqr")


def pd_state_feedback(model: dict, q_eq, Kp, Kd, h: float = 1e-6) -> np.ndarray:
    """The PD + gravity-compensation law written as linear state feedback about (q_eq, 0): linearising
    tau = Kp (q_eq - q) - Kd dq + g~(q) gives tau - g~(q_eq) = -K0 x with K0 = [Kp - dg~/dq(q_eq), Kd]
    (dg~/dq by central differences, step h).  Returns K0 (n, 2n)."""
    # TODO: G = dg~/dq at q_eq by central differences (gravity_torque(model, ...)), as in linearize
    # TODO: return np.hstack([Kp - G, Kd])
    raise NotImplementedError("pd_state_feedback")


def kleinman_lqr(A, B, Q, R, K0, iters: int = 50, tol: float = 1e-10):
    """Kleinman's iteration (Newton's method for the Riccati equation) from a STABILISING gain K0:
    repeat  A_k = A - B K_k;  solve A_k^T P_k + P_k A_k + Q + K_k^T R K_k = 0 (lyapunov_solve);
    K_{k+1} = R^-1 B^T P_k,  until ||P_k - P_{k-1}||_F <= tol max(1, ||P_k||_F).
    Returns (K, P, iterations).  Raises ValueError if A - B K0 is not Hurwitz."""
    # TODO: if not is_hurwitz(A - B K0): raise ValueError
    # TODO: for k = 1..iters: P = lyapunov_solve(A - B K, Q + K^T R K); K = np.linalg.solve(R, B.T @ P);
    # TODO:     stop (return K, P, k) when np.linalg.norm(P - P_old) <= tol * max(1, np.linalg.norm(P))
    raise NotImplementedError("kleinman_lqr")


def lqr_cost(A, B, Q, R, K, x0) -> float:
    """Cost J = int_0^inf (x^T Q x + u^T R u) dt of u = -K x from x(0) = x0 on x' = A x + B u:
    J = x0^T P_K x0 with (A - BK)^T P_K + P_K (A - BK) + Q + K^T R K = 0.  Raises ValueError if A - BK is not Hurwitz."""
    # TODO: Acl = A - B K; raise ValueError if not is_hurwitz(Acl)
    # TODO: P_K = lyapunov_solve(Acl, Q + K^T R K);  return float(x0 @ P_K @ x0)
    raise NotImplementedError("lqr_cost")


def state_feedback_torque(q, dq, q_eq, u_eq, K) -> np.ndarray:
    """tau = u_eq - K x with x = (q - q_eq, dq) (the LQR or PD law applied to the nonlinear arm)."""
    # TODO: x = np.r_[q - q_eq, dq];  return u_eq - K @ x
    raise NotImplementedError("state_feedback_torque")


def run_cost(sim: dict, q_eq, u_eq, Q, R) -> float:
    """Quadratic cost of a simulated run, int (x^T Q x + u^T R u) dt by the trapezoidal rule over sim['t'], with
    x_k = (q_k - q_eq, dq_k) and u_k = tau_k - u_eq."""
    # TODO: X = [sim['q'] - q_eq, sim['dq']] (np.hstack), U = sim['tau'] - u_eq
    # TODO: w_k = x_k^T Q x_k + u_k^T R u_k;  trapezoid: sum 0.5 (w[1:] + w[:-1]) * np.diff(sim['t'])
    raise NotImplementedError("run_cost")
