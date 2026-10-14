// SBEG 461 enrichment lab -- dynamic parameter identification from hardware logs (C++ track): declarations.
//
// The same functions as starter.py, in namespace `lab`, with Eigen types instead of NumPy arrays.  Write your code
// in starter.cpp.  Do not change these declarations: test_lab.cpp calls them.
//
// Identify the inertial, rotor (armature) and friction parameters of the course arm and of the EOD arm 3R from
// SIMULATED hardware logs in the course log format (commanded and measured joint positions, servo currents).
//
// Notation follows Modern Robotics (MR) Ch. 8: theta = joint positions, tau = M(theta) theta'' + h(theta, theta')
// (MR Eq. 8.1); link frames {i} and screw axes A_i as in the Newton-Euler algorithm (MR Sec. 8.3.2, Eqs.
// 8.50-8.54); twists V = (w, v), wrenches F = (m, f); gravity enters as the base acceleration dV_0 = (0, -g).
// In code q = theta.
//
// Parameter vector (13 n entries), in this order:
//     pi_i  = [m, h_x, h_y, h_z, I_xx, I_xy, I_xz, I_yy, I_yz, I_zz]   for links i = 1..n   (10 n entries)
//             h = m c (first moment, c = CoM in {i}); I = rotational inertia about the ORIGIN of {i}
//     I_a,j  armature (reflected rotor) inertia of joint j                                      (n entries)
//     b_v,j  viscous friction of joint j      (tau_fric = b_v theta' + b_c sgn(theta'), MR Fig. 8.15(c))   (n)
//     b_c,j  Coulomb friction of joint j                                                         (n entries)
// Model:  tau = Y(theta, theta', theta'') pi   with Y in R^{n x 13n}   (linear in the parameters).
//
// Python -> C++: a robot dict -> lab::Robot; a dict result -> a small struct with the same field names; a tuple
// -> a struct or std::pair; (N, n) time series -> Eigen::MatrixXd with ONE ROW PER SAMPLE; index arrays ->
// std::vector<int> (0-based); the joint index of the stacked rows -> Eigen::VectorXi; boolean arrays ->
// std::vector<bool>; meta.json -> sbeg::json; `link=None` -> link = -1; `sigma=None` -> an empty VectorXd.
// NumPy / SciPy calls of the Python lab have given C++ equivalents (helpers.cpp, see "Given helpers" below):
// np.linalg.lstsq -> lab::lstsq, np.linalg.cond -> lab::cond, np.gradient -> lab::gradient, signal.butter /
// signal.filtfilt -> lab::butter / lab::filtfilt, optimize.least_squares -> lab::least_squares.
#ifndef SBEG461_PARAM_ID_HPP
#define SBEG461_PARAM_ID_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include <sbeg/labio.hpp>  // sbeg::json (meta.json)

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace lab {

using Vector6d = sbeg::Vector6d;
using Matrix6d = sbeg::Matrix6d;

constexpr double PI = 3.14159265358979323846;
constexpr int NI = 10;  ///< inertial parameters per link
/// Names of the 10 inertial parameters of one link, in the order of pi_i.
inline const std::vector<std::string> LINK_NAMES = {"m", "hx", "hy", "hz", "Ixx", "Ixy", "Ixz", "Iyy", "Iyz", "Izz"};

// ===========================================================================================================
// Data structures
// ===========================================================================================================

/// A robot model in Modern Robotics list form (MR Sec. 8.3.2): robots.py dict keys -> fields.
/// Slist 6 x n (axes are the COLUMNS), Mlist = [M_01, ..., M_n,n+1] (n + 1 transforms; link frames {i} at the link
/// CoMs at home), Glist = [G_1, ..., G_n] (spatial inertias in {i}), g (gravity in {s}), limits n x 2 (rad,
/// [min, max] per joint), qdot_max (rad/s; empty for the EOD arm, None in Python), tau_max (N m), q_home.
struct Robot {
    std::string name;
    std::string key;
    int n = 0;
    Eigen::MatrixXd Slist;
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    std::vector<Eigen::Matrix4d> Mlist;
    std::vector<Matrix6d> Glist;
    Eigen::Vector3d g = Eigen::Vector3d::Zero();
    Eigen::MatrixXd limits;
    Eigen::VectorXd qdot_max;
    Eigen::VectorXd tau_max;
    Eigen::VectorXd q_home;
};

/// A hardware log (hwlog.load_log): t (N), cmd and meas (N x n), cur (N x n, empty if the log has no currents),
/// force (N, empty if absent).  Rows sorted by time, duplicate time stamps dropped.
struct HwLog {
    Eigen::VectorXd t;
    Eigen::MatrixXd cmd;
    Eigen::MatrixXd meas;
    Eigen::MatrixXd cur;
    Eigen::VectorXd force;
};

/// Joint states along a trajectory: q, dq, ddq, each N x n (row k = sample k).
/// Returned by random_states (Python: a tuple) and fourier_trajectory (Python: a tuple (q, dq, ddq)).
struct Trajectory {
    Eigen::MatrixXd q;
    Eigen::MatrixXd dq;
    Eigen::MatrixXd ddq;
};

/// process_log output (Python dict(t, q, dq, ddq, tau)): t (N), q, dq, ddq, tau (N x n).
struct ProcessedLog {
    Eigen::VectorXd t;
    Eigen::MatrixXd q;
    Eigen::MatrixXd dq;
    Eigen::MatrixXd ddq;
    Eigen::MatrixXd tau;
};

/// link_frames_motion output (Python tuple (A, T, V, dV)): A (6 x n, axis A_i in {i} = column i),
/// T[i] = T_{i,i-1}, V[i] and dV[i] the twist and acceleration of link i in {i}.
struct LinkFrames {
    Eigen::MatrixXd A;
    std::vector<Eigen::Matrix4d> T;
    std::vector<Vector6d> V;
    std::vector<Vector6d> dV;
};

/// build_system output (Python tuple (Y, tau, joint)): stacked regressor, torque vector and joint index per row.
struct LinearSystem {
    Eigen::MatrixXd Y;
    Eigen::VectorXd tau;
    Eigen::VectorXi joint;
};

/// base_parameters output (Python dict): idx_base (r), idx_dep (p - r) indices into pi (0-based), K (r x (p - r)),
/// E (r x p), rank r, names (names of the base parameters).
struct BaseParams {
    std::vector<int> idx_base;
    std::vector<int> idx_dep;
    Eigen::MatrixXd K;
    Eigen::MatrixXd E;
    int rank = 0;
    std::vector<std::string> names;
};

/// ols output (Python dict(beta, residual, sigma2, cov, rel_std, cond)).
struct OlsResult {
    Eigen::VectorXd beta;
    Eigen::VectorXd residual;
    double sigma2 = 0.0;
    Eigen::MatrixXd cov;
    Eigen::VectorXd rel_std;
    double cond = 0.0;
};

/// wls output (Python dict(beta, sigma, cov, residual, rel_std, cond)).
struct WlsResult {
    Eigen::VectorXd beta;
    Eigen::VectorXd sigma;
    Eigen::MatrixXd cov;
    Eigen::VectorXd residual;
    Eigen::VectorXd rel_std;
    double cond = 0.0;
};

/// validate output (Python dict(tau_hat, rms_err, nrms)).
struct Validation {
    Eigen::VectorXd tau_hat;
    Eigen::VectorXd rms_err;
    Eigen::VectorXd nrms;
};

/// consistent_fit output (Python dict(pi, x, cost, nfev)).
struct ConsistentFit {
    Eigen::VectorXd pi;
    Eigen::VectorXd x;
    double cost = 0.0;
    int nfev = 0;
};

/// payload_estimate output (Python dict(dpi, mass, com, residual_rms)).
struct Payload {
    Eigen::VectorXd dpi;
    double mass = 0.0;
    Eigen::Vector3d com = Eigen::Vector3d::Zero();
    Eigen::VectorXd residual_rms;
};

// ===========================================================================================================
// Given code (do not edit): robots.cpp (= robots.py), hwlog.cpp (= hwlog.py), helpers.cpp (NumPy / SciPy
// equivalents) and provided.cpp (the provided functions of starter.py)
// ===========================================================================================================

// ---- robots.cpp: the course arm and the EOD arm 3R (numbers identical to robots.py)
/// THE course arm: desktop UR-type 6R (myCobot-280 class), MR Example 4.5 layout (pp. 145-146).
Robot course_arm();
/// Vertical-plane shoulder / elbow / wrist-pitch arm of a tracked EOD robot (gravity -y, joints along z).
Robot eod_3r();
/// "course_arm" or "eod_3r" (robots.get); throws std::invalid_argument for any other key.
Robot get_robot(const std::string& key);
/// Homogeneous transform with identity rotation and position p (robots.trans).
Eigen::Matrix4d trans(const Eigen::Vector3d& p);

// ---- hwlog.cpp: hardware-log helpers (hwlog.py)
/// Read a hardware log (header t, cmd_1..n, meas_1..n [, cur_1..n] [, force]); rows sorted by time, duplicate time
/// stamps dropped.  Throws std::runtime_error if the file or the columns are missing.
HwLog load_log(const std::string& path);
/// Linear interpolation of the columns of x_src(t_src) (N x m) onto t_dst; points outside [t_src(0), t_src(N-1)]
/// are NaN (np.interp per column + NaN outside).
Eigen::MatrixXd resample(const Eigen::VectorXd& t_src, const Eigen::MatrixXd& x_src, const Eigen::VectorXd& t_dst);
/// Root-mean-square of a vector (hwlog.rms).
double rms(const Eigen::VectorXd& x);
/// RMS(predicted - measured) / RMS(measured) (hwlog.normalised_rms).
double normalised_rms(const Eigen::VectorXd& predicted, const Eigen::VectorXd& measured);

// ---- helpers.cpp: NumPy / SciPy equivalents
/// Minimum-norm least-squares solution of A x = b (np.linalg.lstsq(A, b, rcond=None)[0]; SVD, singular values
/// below eps * max(rows, cols) * s_max ignored).
Eigen::VectorXd lstsq(const Eigen::MatrixXd& A, const Eigen::VectorXd& b);
/// 2-norm condition number s_max / s_min (np.linalg.cond).
double cond(const Eigen::MatrixXd& A);
/// np.sign: -1, 0 or +1 for each entry.
Eigen::VectorXd sign(const Eigen::VectorXd& x);
/// np.gradient(X, h, axis=0): central differences inside, one-sided first differences at the two ends.
Eigen::MatrixXd gradient(const Eigen::MatrixXd& X, double h);
/// Digital Butterworth low-pass filter (scipy.signal.butter(order, wn)): returns (b, a), wn = cutoff / Nyquist.
std::pair<Eigen::VectorXd, Eigen::VectorXd> butter(int order, double wn);
/// Zero-phase filtering of every column of X (scipy.signal.filtfilt(b, a, X, axis=0) with its defaults:
/// odd extension of 3 max(len(a), len(b)) samples, initial conditions from lfilter_zi).
Eigen::MatrixXd filtfilt(const Eigen::VectorXd& b, const Eigen::VectorXd& a, const Eigen::MatrixXd& X);

/// Result of lab::least_squares: solution x, cost = 0.5 |r(x)|^2 (as SciPy), number of residual evaluations.
struct LeastSquaresResult {
    Eigen::VectorXd x;
    double cost = 0.0;
    int nfev = 0;
};
/// Bounded nonlinear least squares  min 0.5 |r(x)|^2  with lb <= x <= ub (replaces scipy.optimize.least_squares
/// with jac= and bounds=): damped Gauss-Newton (Levenberg-Marquardt) steps with SciPy's 'trf' scaling near the
/// bounds; every iterate is kept strictly inside the bounds.  fun(x) -> r (m), jac(x) -> J (m x len(x)).  Stops
/// with SciPy's default tolerances (ftol = xtol = gtol = 1e-8) or after max_nfev evaluations of fun.
LeastSquaresResult least_squares(const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& fun,
                                 const std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>& jac,
                                 const Eigen::VectorXd& x0, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
                                 int max_nfev = 2000);

// ---- provided.cpp: the provided functions of starter.py
/// data/meta.json: sampling rate, torque constants, encoder and current resolution, log files
/// (ILLUSTRATIVE values; the logs were simulated by the course team).
sbeg::json load_meta();
/// Path of a file of the lab's data folder, e.g. data_path("eod_3r_excite.csv").
std::string data_path(const std::string& file);
/// "course_arm" or "eod_3r" (the nominal, CAD-like model).
Robot load_robot(const std::string& key);
/// 13 n.
int n_params(int n);
/// Names of the 13 n entries of pi, e.g. "m1", "hx1", ..., "Izz6", "Ia1", ..., "bv1", ..., "bc6".
std::vector<std::string> param_names(int n);
/// 3x3 symmetric matrix from [I_xx, I_xy, I_xz, I_yy, I_yz, I_zz] (a 6-vector).
Eigen::Matrix3d inertia_matrix(const Eigen::VectorXd& I6);
/// 6x6 spatial inertia in {i} from pi_i = [m, h, I6] (I about the origin of {i}):
///     G = [[I, [h]], [[h]^T, m 1]]
/// which is MR Eq. 8.42 with h = m c and I = I_c - m [c][c] (Steiner, MR Eq. 8.27).
Matrix6d spatial_inertia(const Eigen::VectorXd& pi10);
/// The full parameter vector pi (13 n) of a model: inertial part from robot.Glist, the rest from the arguments
/// (zeros if empty).
Eigen::VectorXd nominal_parameters(const Robot& robot, const Eigen::VectorXd& armature = Eigen::VectorXd(),
                                   const Eigen::VectorXd& b_viscous = Eigen::VectorXd(),
                                   const Eigen::VectorXd& b_coulomb = Eigen::VectorXd());
/// Forward iterations of the Newton-Euler algorithm (MR Eqs. 8.50-8.52) with dV_0 = (0, -g): screw axes A_i in
/// {i} (6 x n), T[i] = T_{i,i-1}, link twists V[i] and accelerations dV[i].  This is your Week 2 RNEA forward pass.
LinkFrames link_frames_motion(const Robot& robot, const Eigen::VectorXd& theta, const Eigen::VectorXd& dtheta,
                              const Eigen::VectorXd& ddtheta);
/// Stack Y(theta_k, ...) for samples k = 0..N-1 (sample-major: rows k n + j).  Q, dQ, ddQ: N x n.
/// Returns (N n) x 13 n.
Eigen::MatrixXd stack_regressor(const Robot& robot, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& dQ,
                                const Eigen::MatrixXd& ddQ);
/// Joint index of every row of the stacked system (sample-major): [0, 1, ..., n-1, 0, 1, ...].
Eigen::VectorXi joint_index(int n_samples, int n);
/// (N, n) torques -> (N n) vector in the same sample-major order as stack_regressor (row by row).
Eigen::VectorXd stack_torque(const Eigen::MatrixXd& tau);
/// Stacked system of a processed log (process_log output): (Y, tau, joint) restricted to the rows of moving
/// joints (moving_rows) and, if idx is not empty, to the columns idx (e.g. base.idx_base).
LinearSystem build_system(const Robot& robot, const ProcessedLog& data, const std::vector<int>& idx = {},
                          double v_min = 0.1);
/// Inverse of log_cholesky_to_pi for a physically consistent pi_i (pseudo-inertia J > 0):
/// J = U U^T with U upper triangular, U = e^alpha [[e^d1, s12, s13, t1], [0, e^d2, s23, t2],
/// [0, 0, e^d3, t3], [0, 0, 0, 1]];  returns [alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3].
Eigen::VectorXd pi_to_log_cholesky(const Eigen::VectorXd& pi10);
/// Random (theta, theta', theta'') inside the joint limits: theta uniform in the limits, theta' in [-2, 2],
/// theta'' in [-5, 5] (N x n each).  C++: a portable generator (std::mt19937_64 with this seed), so the states
/// are NOT the same numbers as NumPy's default_rng in Python.
Trajectory random_states(const Robot& robot, int n_samples, unsigned seed = 0);

// ===========================================================================================================
// Part A -- linear-in-parameters form
// ===========================================================================================================

/// pi_i = [m, h_x, h_y, h_z, I_xx, I_xy, I_xz, I_yy, I_yz, I_zz] (a 10-vector) from a 6x6 spatial inertia G
/// written in {i} (inverse of spatial_inertia).
Eigen::VectorXd inertial_parameters(const Matrix6d& G);

/// 6 x 10 matrix Phi(V, dV) with  G dV - [ad_V]^T G V = Phi(V, dV) pi_i  (MR Eq. 8.43 in a link frame).
/// Columns in the order of pi_i. With V = (w, v), dV = (dw, dv) and a = dv + w x v:
///     moment = I dw + w x (I w) + h x a          force = m a + dw x h + w x (w x h).
Eigen::MatrixXd wrench_regressor(const Vector6d& V, const Vector6d& dV);

/// Y(theta, theta', theta'') in R^{n x 13n} with tau = Y pi (parameter order at the top of this file).
/// Inertial part: tau_j = sum_{i >= j} A_j^T [Ad_{T_ij}]^T Phi(V_i, dV_i) pi_i, T_ij = T_{i,i-1} ... T_{j+1,j}
/// (the backward iterations MR Eqs. 8.53-8.54 written as a sum). Then armature I_a,j theta''_j, viscous
/// b_v,j theta'_j and Coulomb b_c,j sgn(theta'_j) columns.
Eigen::MatrixXd regressor(const Robot& robot, const Eigen::VectorXd& theta, const Eigen::VectorXd& dtheta,
                          const Eigen::VectorXd& ddtheta);

// ===========================================================================================================
// Part B -- base parameters
// ===========================================================================================================

/// Numerical base parameters (Gautier's QR method). Build W = stacked Y at random_states(robot, n_samples, seed),
/// scale its columns to unit norm, QR with column pivoting W P = Q [R1 R2], rank r from |R_kk| > tol |R_00|.
/// Returns idx_base (r), idx_dep (p - r)  indices into pi (base and dependent parameters; idx_base sorted),
///         K  (r x (p - r)) with  W[:, idx_dep] = W[:, idx_base] K   (in the original, unscaled columns),
///         E  (r x p)       with  beta = E pi  (beta = pi_base + K pi_dep), so that Y pi = Y[:, idx_base] beta,
///         rank r, names (names of the base parameters, each the parameter it is "regrouped" onto).
BaseParams base_parameters(const Robot& robot, int n_samples = 60, double tol = 1e-9, unsigned seed = 0);

// ===========================================================================================================
// Part C -- from a log to (theta, theta', theta'', tau)
// ===========================================================================================================

/// Resample the log (load_log output) onto a uniform grid at fs, low-pass filter the measured positions AND the
/// currents with the same zero-phase Butterworth filter (order, cutoff in Hz; filtfilt), differentiate the
/// filtered positions twice (gradient), convert current to torque tau = k * cur (k: torque constant per joint),
/// drop `trim` seconds at both ends (filter edge effects) and keep every `decimate`-th sample.
ProcessedLog process_log(const HwLog& log, const Eigen::VectorXd& k, double fs = 500.0, double cutoff = 4.0,
                         int order = 4, int decimate = 5, double trim = 1.0);

// ===========================================================================================================
// Part D -- estimation and validation
// ===========================================================================================================

/// Ordinary least squares beta = argmin |Y beta - tau|^2 (use lab::lstsq). Returns beta, residual (vector),
/// sigma2 = |residual|^2 / (rows - cols), cov = sigma2 (Y^T Y)^-1, rel_std = 100 sqrt(diag cov) / |beta|
/// (percent), cond = condition number of Y.
OlsResult ols(const Eigen::MatrixXd& Y, const Eigen::VectorXd& tau);

/// Row mask for the stacked system (sample-major, rows k n + j): true where |theta'_j| >= v_min at sample k.
/// The Coulomb term sgn(theta') is not defined at rest and the real friction near zero speed (stiction,
/// Stribeck effect, MR Fig. 8.15) is not in the model, so rows of slow joints are left out.
std::vector<bool> moving_rows(const Eigen::MatrixXd& dq, double v_min = 0.1);

/// Two-step weighted least squares (Gautier): OLS first; for each joint j, sigma_j^2 = sum of the squared OLS
/// residuals of the rows of joint j / (number of those rows - cols/n); then OLS of (Y, tau) with every row of
/// joint j divided by sigma_j. `joint` (rows) is the joint index of each row (joint_index, then masked).
/// Returns beta, sigma (n), cov = (Y^T S^-2 Y)^-1, rel_std (percent), residual (unweighted), cond (of the
/// weighted Y).
WlsResult wls(const Eigen::MatrixXd& Y, const Eigen::VectorXd& tau, const Eigen::VectorXi& joint, int n);

/// Prediction on a (held-out) data set: tau_hat = Y beta. For each joint j (rows with joint == j):
/// rms_err_j = RMS(tau_hat - tau) in N m and nrms_j = rms_err_j / RMS(tau). Returns tau_hat (rows), rms_err (n),
/// nrms (n).
Validation validate(const Eigen::MatrixXd& Y, const Eigen::VectorXd& beta, const Eigen::VectorXd& tau,
                    const Eigen::VectorXi& joint, int n);

// ===========================================================================================================
// Part E -- excitation trajectories
// ===========================================================================================================

/// Finite Fourier series excitation (Swevers et al. 1997) for each joint j, harmonics l = 1..L:
///     theta_j(t) = q0_j + sum_l [ a_jl/(wf l) sin(wf l t) - b_jl/(wf l) cos(wf l t) ]
/// a, b: n x L matrices; wf = 2 pi f_f (rad/s). Returns (q, dq, ddq), each len(t) x n. Periodic with period
/// 2 pi / wf, so averaging over periods and repeating are possible.
Trajectory fourier_trajectory(const Eigen::VectorXd& t, const Eigen::VectorXd& q0, const Eigen::MatrixXd& a,
                              const Eigen::MatrixXd& b, double wf);

/// Condition number of the base regressor Y_b = Y[:, idx_base] stacked along a trajectory (every `every`-th
/// sample, starting with sample 0), after scaling its columns to unit norm. Smaller is better.
double excitation_cost(const Robot& robot, const BaseParams& base, const Eigen::MatrixXd& q,
                       const Eigen::MatrixXd& dq, const Eigen::MatrixXd& ddq, int every = 10);

// ===========================================================================================================
// Part F -- physical consistency
// ===========================================================================================================

/// 4x4 pseudo-inertia J = [[Sigma, h], [h^T, m]] with Sigma = (1/2) tr(I) 1 - I (the second moment of the mass
/// distribution about the origin of {i}). A rigid body with this pi exists iff J is positive definite
/// (Wensing, Kim and Slotine 2018).
Eigen::Matrix4d pseudo_inertia(const Eigen::VectorXd& pi10);

/// Per-link physical consistency of a full parameter vector pi (13 n): true for link i if the smallest
/// eigenvalue of its pseudo-inertia is > tol, AND I_a,i >= 0, b_v,i >= 0, b_c,i >= 0. Returns n booleans.
std::vector<bool> is_consistent(const Eigen::VectorXd& pi, int n, double tol = 0.0);

/// pi_i from the 10 log-Cholesky coordinates [alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3] (Rucker and
/// Wensing 2022): U = e^alpha [[e^d1, s12, s13, t1], [0, e^d2, s23, t2], [0, 0, e^d3, t3], [0, 0, 0, 1]],
/// J = U U^T, m = J_44, h = J[:3, 3], I = tr(Sigma) 1 - Sigma with Sigma = J[:3, :3]. Every theta10 gives a
/// physically consistent pi_i.
Eigen::VectorXd log_cholesky_to_pi(const Eigen::VectorXd& theta10);

/// Physically consistent estimate of the FULL vector pi:
///     minimise  sum_rows ((Y pi - tau)/sigma_j)^2 + lam * sum_k ((pi_k - prior_k)/s_k)^2
/// over pi = pi(x) with log-Cholesky coordinates for every link and I_a = e^x, b_v = e^x, b_c = e^x for the
/// other 3n entries, so every iterate is consistent. s_k = |prior_k| + 1e-3 max|prior|. `prior` (13 n) must be
/// consistent (it is the starting point); `joint` (rows) joint index of each row; sigma (n) per-joint noise
/// levels, e.g. from wls (ones if empty). The regulariser keeps the directions the data cannot see (the
/// non-base combinations) at the prior.
/// Speed-up: with the QR factorisation Yw = Q R, |Yw pi - tw|^2 = |R pi - Q^T tw|^2 + const, so the data
/// term needs only 13 n residuals. Use lab::least_squares with the bounds x0 -/+ 8. Returns pi, x, cost, nfev.
ConsistentFit consistent_fit(const Eigen::MatrixXd& Y, const Eigen::VectorXd& tau, const Eigen::VectorXi& joint,
                             int n, const Eigen::VectorXd& prior, double lam = 1e-3,
                             const Eigen::VectorXd& sigma = Eigen::VectorXd());

// ===========================================================================================================
// Part G -- payload from a log
// ===========================================================================================================

/// Rigid payload added to link `link` (0-based; -1 = default: last link). The arm is known (model_pi, a full
/// 13 n vector: e.g. the consistent fit, or any pi with E pi = beta); the payload's own 10 parameters dpi
/// enter linearly:
///     tau - Y pi_arm = Y_link dpi,      Y_link = the 10 columns of Y that belong to `link`,
/// over the rows of moving joints (build_system). Solve by least squares over the columns of Y_link that are
/// not zero along the data (min-norm lstsq), and return dpi (10), mass, com (h/m in {link}), residual_rms (n).
Payload payload_estimate(const Robot& robot, const ProcessedLog& data, const Eigen::VectorXd& model_pi,
                         int link = -1, double v_min = 0.1);

}  // namespace lab

#endif  // SBEG461_PARAM_ID_HPP
