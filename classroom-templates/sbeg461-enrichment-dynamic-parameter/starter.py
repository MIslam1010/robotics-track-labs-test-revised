"""SBEG 461 enrichment lab -- dynamic parameter identification from hardware logs (STARTER CODE).

Complete every function marked TODO (keep names and signatures). Provided functions (do not edit): load_meta,
load_robot, n_params, param_names, inertia_matrix, spatial_inertia, nominal_parameters, link_frames_motion,
stack_regressor, joint_index, stack_torque, build_system, pi_to_log_cholesky, random_states.

Identify the inertial, rotor (armature) and friction parameters of the course arm and of the EOD arm 3R from
SIMULATED hardware logs in the course log format (commanded and measured joint positions, servo currents).

Notation follows Modern Robotics (MR) Ch. 8: theta = joint positions, tau = M(theta) theta'' + h(theta, theta')
(MR Eq. 8.1); link frames {i} and screw axes A_i as in the Newton-Euler algorithm (MR Sec. 8.3.2, Eqs. 8.50-8.54);
twists V = (w, v), wrenches F = (m, f); gravity enters as the base acceleration dV_0 = (0, -g). In code q = theta.

Parameter vector (13 n entries), in this order:
    pi_i  = [m, h_x, h_y, h_z, I_xx, I_xy, I_xz, I_yy, I_yz, I_zz]   for links i = 1..n   (10 n entries)
            h = m c (first moment, c = CoM in {i}); I = rotational inertia about the ORIGIN of {i}
    I_a,j  armature (reflected rotor) inertia of joint j                                      (n entries)
    b_v,j  viscous friction of joint j      (tau_fric = b_v theta' + b_c sgn(theta'), MR Fig. 8.15(c))   (n)
    b_c,j  Coulomb friction of joint j                                                         (n entries)

Model:  tau = Y(theta, theta', theta'') pi   with Y in R^{n x 13n}   (linear in the parameters).

Parts
  A. Linear-in-parameters form: inertial_parameters (G -> pi_i), wrench_regressor (6 x 10), regressor (n x 13n)
  B. Base parameters: base_parameters (numerical QR with column pivoting)
  C. Logs: process_log (resample, zero-phase filter, differentiate, current -> torque)
  D. Estimation: moving_rows, ols, wls, validate
  E. Excitation: fourier_trajectory, excitation_cost
  F. Physical consistency: pseudo_inertia, is_consistent, log_cholesky_to_pi, consistent_fit
  G. Payload: payload_estimate (EOD arm package mass from a log)
"""
from __future__ import annotations

import json
from pathlib import Path

import modern_robotics as mr
import numpy as np
from scipy import linalg, optimize, signal

import hwlog
import robots

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
NI = 10                                       # inertial parameters per link
LINK_NAMES = ["m", "hx", "hy", "hz", "Ixx", "Ixy", "Ixz", "Iyy", "Iyz", "Izz"]


# ==========================================================================================
# Provided (do not edit)
# ==========================================================================================
def load_meta() -> dict:
    """data/meta.json: sampling rate, torque constants, encoder and current resolution, log files
    (ILLUSTRATIVE values; the logs were simulated by the course team)."""
    return json.loads((DATA / "meta.json").read_text())


def load_robot(key: str) -> dict:
    """'course_arm' or 'eod_3r' from robots.py (the nominal, CAD-like model)."""
    return robots.get(key)


def n_params(n: int) -> int:
    return 13 * n


def param_names(n: int) -> list:
    """Names of the 13 n entries of pi, e.g. 'm1', 'hx1', ..., 'Izz6', 'Ia1', ..., 'bv1', ..., 'bc6'."""
    out = [f"{s}{i + 1}" for i in range(n) for s in LINK_NAMES]
    return out + [f"Ia{j + 1}" for j in range(n)] + [f"bv{j + 1}" for j in range(n)] + [f"bc{j + 1}" for j in range(n)]


def inertia_matrix(I6) -> np.ndarray:
    """3x3 symmetric matrix from [I_xx, I_xy, I_xz, I_yy, I_yz, I_zz]."""
    a, b, c, d, e, f = np.asarray(I6, float)
    return np.array([[a, b, c], [b, d, e], [c, e, f]])


def spatial_inertia(pi10) -> np.ndarray:
    """6x6 spatial inertia in {i} from pi_i = [m, h, I6] (I about the origin of {i}):
        G = [[I, [h]], [[h]^T, m 1]]
    which is MR Eq. 8.42 with h = m c and I = I_c - m [c][c] (Steiner, MR Eq. 8.27)."""
    pi10 = np.asarray(pi10, float)
    m, h = pi10[0], pi10[1:4]
    G = np.zeros((6, 6))
    G[:3, :3] = inertia_matrix(pi10[4:])
    G[:3, 3:] = mr.VecToso3(h)
    G[3:, :3] = mr.VecToso3(h).T
    G[3:, 3:] = m * np.eye(3)
    return G


def nominal_parameters(robot: dict, armature=None, b_viscous=None, b_coulomb=None) -> np.ndarray:
    """The full parameter vector pi (13 n) of a model: inertial part from robot['Glist'], the rest from the
    arguments (zeros if None)."""
    n = robot["n"]
    z = np.zeros(n)
    inert = np.concatenate([inertial_parameters(G) for G in robot["Glist"]])
    return np.concatenate([inert, z if armature is None else armature, z if b_viscous is None else b_viscous,
                           z if b_coulomb is None else b_coulomb])


def link_frames_motion(robot: dict, theta, dtheta, ddtheta):
    """Forward iterations of the Newton-Euler algorithm (MR Eqs. 8.50-8.52) with dV_0 = (0, -g):
    returns (A, T, V, dV): screw axes A_i in {i} (6 x n), T[i] = T_{i,i-1}, link twists V[i] and accelerations
    dV[i] (lists of n arrays). Provided: this is your Week 2 RNEA forward pass."""
    n = robot["n"]
    Mlist, Slist = robot["Mlist"], robot["Slist"]
    Mi = np.eye(4)
    A = np.zeros((6, n))
    for i in range(n):
        Mi = Mi @ Mlist[i]
        A[:, i] = mr.Adjoint(mr.TransInv(Mi)) @ Slist[:, i]
    T, V, dV = [], [], []
    Vp = np.zeros(6)
    dVp = np.r_[0.0, 0.0, 0.0, -np.asarray(robot["g"], float)]
    for i in range(n):
        Ti = mr.MatrixExp6(mr.VecTose3(A[:, i] * -theta[i])) @ mr.TransInv(Mlist[i])
        AdT = mr.Adjoint(Ti)
        Vi = AdT @ Vp + A[:, i] * dtheta[i]
        dVi = AdT @ dVp + mr.ad(Vi) @ A[:, i] * dtheta[i] + A[:, i] * ddtheta[i]
        T.append(Ti)
        V.append(Vi)
        dV.append(dVi)
        Vp, dVp = Vi, dVi
    return A, T, V, dV


def stack_regressor(robot: dict, Q, dQ, ddQ) -> np.ndarray:
    """Stack Y(theta_k, ...) for samples k = 0..N-1 (sample-major: rows k n + j). Returns (N n) x 13 n."""
    return np.vstack([regressor(robot, q, dq, ddq) for q, dq, ddq in zip(Q, dQ, ddQ)])


def joint_index(n_samples: int, n: int) -> np.ndarray:
    """Joint index of every row of the stacked system (sample-major): [0, 1, ..., n-1, 0, 1, ...]."""
    return np.tile(np.arange(n), n_samples)


def stack_torque(tau) -> np.ndarray:
    """(N, n) torques -> (N n,) vector in the same sample-major order as stack_regressor."""
    return np.asarray(tau, float).reshape(-1)


def build_system(robot: dict, data: dict, idx=None, v_min: float = 0.1):
    """Stacked system of a processed log (process_log output): returns (Y, tau, joint) restricted to the rows
    of moving joints (moving_rows) and, if idx is given, to the columns idx (e.g. base['idx_base'])."""
    Y = stack_regressor(robot, data["q"], data["dq"], data["ddq"])
    if idx is not None:
        Y = Y[:, idx]
    keep = moving_rows(data["dq"], v_min)
    return Y[keep], stack_torque(data["tau"])[keep], joint_index(len(data["q"]), robot["n"])[keep]


def pi_to_log_cholesky(pi10) -> np.ndarray:
    """Inverse of log_cholesky_to_pi for a physically consistent pi_i (pseudo-inertia J > 0):
    J = U U^T with U upper triangular, U = e^alpha [[e^d1, s12, s13, t1], [0, e^d2, s23, t2],
    [0, 0, e^d3, t3], [0, 0, 0, 1]];  returns [alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3]."""
    J = pseudo_inertia(pi10)
    P = np.eye(4)[::-1]
    U = P @ np.linalg.cholesky(P @ J @ P) @ P          # upper triangular, J = U U^T
    alpha = np.log(U[3, 3])
    U = U / U[3, 3]
    return np.array([alpha, np.log(U[0, 0]), np.log(U[1, 1]), np.log(U[2, 2]), U[0, 1], U[1, 2], U[0, 2],
                     U[0, 3], U[1, 3], U[2, 3]])


# ==========================================================================================
# Part A -- linear-in-parameters form
# ==========================================================================================
def inertial_parameters(G) -> np.ndarray:
    """pi_i = [m, h_x, h_y, h_z, I_xx, I_xy, I_xz, I_yy, I_yz, I_zz] from a 6x6 spatial inertia G written in
    {i} (inverse of spatial_inertia)."""
    # TODO: m = G[3, 3]; h from the skew block G[:3, 3:] (mr.so3ToVec); the six entries of I = G[:3, :3].
    raise NotImplementedError("inertial_parameters: TODO")


def wrench_regressor(V, dV) -> np.ndarray:
    """6 x 10 matrix Phi(V, dV) with  G dV - [ad_V]^T G V = Phi(V, dV) pi_i  (MR Eq. 8.43 in a link frame).
    Columns in the order of pi_i. With V = (w, v), dV = (dw, dv) and a = dv + w x v:
        moment = I dw + w x (I w) + h x a          force = m a + dw x h + w x (w x h)."""
    # TODO: Expand G dV - [ad_V]^T G V with G = spatial_inertia(pi) (MR Eqs. 8.39-8.40) and collect the
    #       coefficients of m, h and I. Check: Phi @ pi == G @ dV - mr.ad(V).T @ G @ V for random V, dV, pi.
    raise NotImplementedError("wrench_regressor: TODO")


def regressor(robot: dict, theta, dtheta, ddtheta) -> np.ndarray:
    """Y(theta, theta', theta'') in R^{n x 13n} with tau = Y pi (parameter order in the module docstring).
    Inertial part: tau_j = sum_{i >= j} A_j^T [Ad_{T_ij}]^T Phi(V_i, dV_i) pi_i, T_ij = T_{i,i-1} ... T_{j+1,j}
    (the backward iterations MR Eqs. 8.53-8.54 written as a sum). Then armature I_a,j theta''_j, viscous
    b_v,j theta'_j and Coulomb b_c,j sgn(theta'_j) columns."""
    # TODO: Use link_frames_motion (forward pass). For every link i, Phi_i = wrench_regressor(V[i], dV[i]); walk
    #       j = i, i-1, ..., 0 with T_ij (start at identity, then T_ij = T_ij @ T[j]) and fill row j of the
    #       10 columns of link i with A[:, j] @ Ad(T_ij)^T @ Phi_i. Then the armature, viscous and Coulomb columns.
    raise NotImplementedError("regressor: TODO")


# ==========================================================================================
# Part B -- base parameters
# ==========================================================================================
def random_states(robot: dict, n_samples: int, seed: int = 0):
    """Random (theta, theta', theta'') inside the joint limits (provided)."""
    rng = np.random.default_rng(seed)
    lo, hi = robot["limits"][:, 0], robot["limits"][:, 1]
    n = robot["n"]
    Q = rng.uniform(lo, hi, (n_samples, n))
    return Q, rng.uniform(-2, 2, (n_samples, n)), rng.uniform(-5, 5, (n_samples, n))


def base_parameters(robot: dict, n_samples: int = 60, tol: float = 1e-9, seed: int = 0) -> dict:
    """Numerical base parameters (Gautier's QR method). Build W = stacked Y at random states, scale its columns
    to unit norm, QR with column pivoting W P = Q [R1 R2], rank r from |R_kk| > tol |R_00|. Returns
        idx_base (r,), idx_dep (p - r,)  indices into pi (base and dependent parameters),
        K  (r x (p - r)) with  W[:, idx_dep] = W[:, idx_base] K   (in the original, unscaled columns),
        E  (r x p)       with  beta = E pi  (beta = pi_base + K pi_dep), so that Y pi = Y[:, idx_base] beta,
        rank r, names (names of the base parameters, each the parameter it is "regrouped" onto)."""
    # TODO: W = stack_regressor at random_states; divide each column by its norm (zero columns: by 1);
    #       scipy.linalg.qr(..., mode='economic', pivoting=True); rank from the diagonal of R; K from R1^-1 R2,
    #       then undo the column scaling; E with ones at idx_base and K at idx_dep.
    raise NotImplementedError("base_parameters: TODO")


# ==========================================================================================
# Part C -- from a log to (theta, theta', theta'', tau)
# ==========================================================================================
def process_log(log: dict, k, fs: float = 500.0, cutoff: float = 4.0, order: int = 4, decimate: int = 5,
                trim: float = 1.0) -> dict:
    """Resample the log (hwlog.load_log output) onto a uniform grid at fs, low-pass filter the measured
    positions AND the currents with the same zero-phase Butterworth filter (order, cutoff in Hz; filtfilt),
    differentiate the filtered positions twice (np.gradient), convert current to torque tau = k * cur,
    drop `trim` seconds at both ends (filter edge effects) and keep every `decimate`-th sample.
    Returns dict(t, q, dq, ddq, tau), arrays (N,) and (N, n)."""
    # TODO: Uniform grid tu = np.arange(t0, t_end, 1/fs); hwlog.resample of meas and cur; signal.butter +
    #       signal.filtfilt (axis=0) on BOTH; np.gradient twice; tau = k * filtered current; trim and decimate.
    raise NotImplementedError("process_log: TODO")


# ==========================================================================================
# Part D -- estimation and validation
# ==========================================================================================
def ols(Y, tau) -> dict:
    """Ordinary least squares beta = argmin |Y beta - tau|^2 (use np.linalg.lstsq). Returns dict(beta,
    residual (vector), sigma2 = |residual|^2 / (rows - cols), cov = sigma2 (Y^T Y)^-1,
    rel_std = 100 sqrt(diag cov) / |beta| (percent), cond = condition number of Y)."""
    # TODO: np.linalg.lstsq; residual; sigma2 with rows - cols degrees of freedom; cov = sigma2 inv(Y^T Y).
    raise NotImplementedError("ols: TODO")


def moving_rows(dq, v_min: float = 0.1) -> np.ndarray:
    """Row mask for the stacked system (sample-major, rows k n + j): True where |theta'_j| >= v_min at sample k.
    The Coulomb term sgn(theta') is not defined at rest and the real friction near zero speed (stiction,
    Stribeck effect, MR Fig. 8.15) is not in the model, so rows of slow joints are left out."""
    # TODO: np.abs(dq) >= v_min, flattened in sample-major order.
    raise NotImplementedError("moving_rows: TODO")


def wls(Y, tau, joint, n: int) -> dict:
    """Two-step weighted least squares (Gautier): OLS first; for each joint j, sigma_j^2 = sum of the squared
    OLS residuals of the rows of joint j / (number of those rows - cols/n); then OLS of (Y, tau) with every row
    of joint j divided by sigma_j. `joint` (rows,) is the joint index of each row (joint_index, then masked).
    Returns dict(beta, sigma (n,), cov = (Y^T S^-2 Y)^-1, rel_std (percent), residual (unweighted),
    cond (of the weighted Y))."""
    # TODO: Call ols first; sigma_j from the residual rows with joint == j; weights 1/sigma_j per row; lstsq again.
    raise NotImplementedError("wls: TODO")


def validate(Y, beta, tau, joint, n: int) -> dict:
    """Prediction on a (held-out) data set: tau_hat = Y beta. For each joint j (rows with joint == j):
    rms_err_j = RMS(tau_hat - tau) in N m and nrms_j = rms_err_j / RMS(tau). Returns dict(tau_hat (rows,),
    rms_err (n,), nrms (n,))."""
    # TODO: tau_hat = Y @ beta; per joint: RMS of the error over the rows with joint == j, and divided by RMS(tau).
    raise NotImplementedError("validate: TODO")


# ==========================================================================================
# Part E -- excitation trajectories
# ==========================================================================================
def fourier_trajectory(t, q0, a, b, wf: float):
    """Finite Fourier series excitation (Swevers et al. 1997) for each joint j, harmonics l = 1..L:
        theta_j(t) = q0_j + sum_l [ a_jl/(wf l) sin(wf l t) - b_jl/(wf l) cos(wf l t) ]
    a, b: (n, L) arrays; wf = 2 pi f_f (rad/s). Returns (q, dq, ddq), each (len(t), n). Periodic with period
    2 pi / wf, so averaging over periods and repeating are possible."""
    # TODO: Use np.outer(t, wf * l) for the sin/cos arguments; differentiate the series term by term.
    raise NotImplementedError("fourier_trajectory: TODO")


def excitation_cost(robot: dict, base: dict, q, dq, ddq, every: int = 10) -> float:
    """Condition number of the base regressor Y_b = Y[:, idx_base] stacked along a trajectory (every
    `every`-th sample), after scaling its columns to unit norm. Smaller is better."""
    # TODO: stack_regressor on every `every`-th sample, keep base['idx_base'] columns, normalise columns, np.linalg.cond.
    raise NotImplementedError("excitation_cost: TODO")


# ==========================================================================================
# Part F -- physical consistency
# ==========================================================================================
def pseudo_inertia(pi10) -> np.ndarray:
    """4x4 pseudo-inertia J = [[Sigma, h], [h^T, m]] with Sigma = (1/2) tr(I) 1 - I (the second moment of the
    mass distribution about the origin of {i}). A rigid body with this pi exists iff J is positive definite
    (Wensing, Kim and Slotine 2018)."""
    # TODO: Sigma = 0.5 trace(I) 1 - I with I = inertia_matrix(pi10[4:]); h = pi10[1:4]; m = pi10[0].
    raise NotImplementedError("pseudo_inertia: TODO")


def is_consistent(pi, n: int, tol: float = 0.0) -> np.ndarray:
    """Per-link physical consistency of a full parameter vector pi (13 n): True for link i if the smallest
    eigenvalue of its pseudo-inertia is > tol, AND I_a,i >= 0, b_v,i >= 0, b_c,i >= 0. Returns (n,) bool."""
    # TODO: np.linalg.eigvalsh(pseudo_inertia(...)).min() > tol for each link; and armature, friction >= 0.
    raise NotImplementedError("is_consistent: TODO")


def log_cholesky_to_pi(theta10) -> np.ndarray:
    """pi_i from the 10 log-Cholesky coordinates [alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3] (Rucker and
    Wensing 2022): U = e^alpha [[e^d1, s12, s13, t1], [0, e^d2, s23, t2], [0, 0, e^d3, t3], [0, 0, 0, 1]],
    J = U U^T, m = J_44, h = J[:3, 3], I = tr(Sigma) 1 - Sigma with Sigma = J[:3, :3]. Every theta10 gives a
    physically consistent pi_i."""
    # TODO: Build U, J = U @ U.T, then read m, h and Sigma from J and convert Sigma to I.
    raise NotImplementedError("log_cholesky_to_pi: TODO")


def consistent_fit(Y, tau, joint, n: int, prior, lam: float = 1e-3, sigma=None) -> dict:
    """Physically consistent estimate of the FULL vector pi:
        minimise  sum_rows ((Y pi - tau)/sigma_j)^2 + lam * sum_k ((pi_k - prior_k)/s_k)^2
    over pi = pi(x) with log-Cholesky coordinates for every link and I_a = e^x, b_v = e^x, b_c = e^x for the
    other 3n entries, so every iterate is consistent. s_k = |prior_k| + 1e-3 max|prior|. `prior` (13 n) must be
    consistent (it is the starting point); `joint` (rows,) joint index of each row; sigma (n,) per-joint noise
    levels, e.g. from wls (ones if None). The regulariser keeps the directions the data cannot see (the
    non-base combinations) at the prior.
    Speed-up: with the QR factorisation Yw = Q R, |Yw pi - tw|^2 = |R pi - Q^T tw|^2 + const, so the data
    term needs only 13 n residuals. Use scipy.optimize.least_squares. Returns dict(pi, x, cost, nfev)."""
    # TODO: x0 from pi_to_log_cholesky(prior link blocks) and log(prior[10n:]); residual = [R pi(x) - Q^T tw,
    #       sqrt(lam) (pi(x) - prior)/s]; scipy.optimize.least_squares(resid, x0, bounds=(x0 - 8, x0 + 8)).
    raise NotImplementedError("consistent_fit: TODO")


# ==========================================================================================
# Part G -- payload from a log
# ==========================================================================================
def payload_estimate(robot: dict, data: dict, model_pi, link: int | None = None, v_min: float = 0.1) -> dict:
    """Rigid payload added to link `link` (0-based; default: last link). The arm is known (model_pi, a full
    13 n vector: e.g. the consistent fit, or any pi with E pi = beta); the payload's own 10 parameters dpi
    enter linearly:
        tau - Y pi_arm = Y_link dpi,      Y_link = the 10 columns of Y that belong to `link`,
    over the rows of moving joints (build_system). Solve by least squares over the columns of Y_link that are not
    zero along the data (min-norm lstsq), and return dict(dpi (10,), mass, com (h/m in {link}), residual_rms (n,))."""
    # TODO: r = tau - Y @ model_pi; Y_link = Y[:, 10*link:10*link+10]; lstsq on its non-zero columns.
    raise NotImplementedError("payload_estimate: TODO")
