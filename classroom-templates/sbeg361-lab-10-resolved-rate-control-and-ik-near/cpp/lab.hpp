// SBEG 361 Lab 10 -- Resolved-rate control, damped least squares, redundancy and a hexapod (C++ track): declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp. Do not change these declarations: test_lab.cpp calls them.
//
// Notation follows Lynch & Park, *Modern Robotics* (MR):
//   pseudo-inverse  J^+ = J^T (J J^T)^-1 (fat, full rank), (J^T J)^-1 J^T (tall)      (MR Sec. 6.2.2, p. 227)
//   inverse velocity kinematics  thetadot = J^+(theta) V_d                             (MR Eq. 6.7, p. 231)
//   weighted pseudo-inverse  W^-1 J^T (J W^-1 J^T)^-1  (W = M(theta) on MR p. 232)
//   redundancy  thetadot = J^+ xdot + (I - J^+ J) thetadot_0   (null-space projector I - J^+ J)
//   damped least squares  J^#(lam) = J^T (J J^T + lam^2 I)^-1 = sum_i sigma_i/(sigma_i^2 + lam^2) v_i u_i^T
//   task-space feedforward + P control  V_b = [Ad_{X^-1 X_d}] V_d + k_p X_e,  [X_e] = log(X^-1 X_d)
//                                                                                     (MR Eq. 11.16 with K_i = 0, p. 421)
//   Newton-Raphson IK in the body frame  theta <- theta + J_b^+ V_b                    (MR Sec. 6.2.2, pp. 228-229)
//   Stewart-Gough platform  s_i = |p + R b_i - a_i|,  sdot = J_s^-1 V_s,
//   J_s^-1 = [ -n_1 x q_1 ... -n_6 x q_6 ; n_1 ... n_6 ]^T, q_i = a_i                    (MR Secs. 7.1.2, 7.2.1, pp. 247-252)
// Twists V = (omega, v): angular part first.
//
// Python -> C++: screw lists and Jacobians -> Eigen::MatrixXd (6 x n, one axis or column per joint, as in Python);
// theta -> Eigen::VectorXd; a list of 4x4 poses -> std::vector<Eigen::Matrix4d>; an (N, n) array -> Eigen::MatrixXd
// (row = sample); a tuple or a dict of results -> a small struct with the same field names; an optional argument
// that is None in Python -> an empty Eigen::VectorXd or std::nullopt.
#ifndef SBEG361_LAB10_HPP
#define SBEG361_LAB10_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lab {

/// Same threshold as modern_robotics.NearZero.
constexpr double TOL = 1e-6;

/// A robot model of data/robots.json (Python: the tuple (Slist, M, info)).
struct Robot {
    Eigen::MatrixXd Slist;    ///< 6 x n space screw axes (columns)
    Eigen::Matrix4d M;        ///< home configuration of {b}
    std::string name;         ///< info["name"]
    Eigen::MatrixXd limits;   ///< n x 2 joint limits (rad); empty if null in the file
    Eigen::VectorXd qdot_max; ///< rated joint speeds (rad/s); empty if null in the file
    Eigen::VectorXd q_home;   ///< home joint vector (rad); empty if null in the file
};

/// The couch hexapod of data/hexapod.json (Python: the tuple (A, B, info)).
struct Hexapod {
    Eigen::MatrixXd A;        ///< 6 x 3: base joint points a_i in {s} (row i = a_i)
    Eigen::MatrixXd B;        ///< 6 x 3: platform joint points b_i in {b} (row i = b_i)
    std::string name;         ///< info["name"]
    double h0 = 0.0;          ///< home height of {b} above {s} (info["params"]["h0"])
    Eigen::Vector2d stroke;   ///< (s_min, s_max) leg length range (m)
    double leg_speed_max = 0.0;
    double rated_leg_force = 0.0;
};

/// A joint log in the project's hwcompare.py format (Python: the tuple (t, cmd_deg, meas_deg)).
struct JointLog {
    Eigen::VectorXd t;        ///< N sample times (s)
    Eigen::MatrixXd cmd_deg;  ///< N x n commanded joints (deg), row = sample
    Eigen::MatrixXd meas_deg; ///< N x n measured joints (deg), row = sample
};

// ---------------------------------------------------------------------------
// Helpers (complete; from Labs 2-7) -- given in starter.cpp
// ---------------------------------------------------------------------------

/// 3x3 skew-symmetric matrix [omega] (MR Eq. 3.30).
Eigen::Matrix3d vec_to_so3(const Eigen::Vector3d& omega);

/// omega from [omega] (reads the entries (2,1), (0,2), (1,0)).
Eigen::Vector3d so3_to_vec(const Eigen::Matrix3d& so3mat);

/// 4x4 matrix [V] of a twist V = (omega, v) (MR Eq. 3.71).
Eigen::Matrix4d vec_to_se3(const sbeg::Vector6d& V);

/// V = (omega, v) from a 4x4 matrix [V].
sbeg::Vector6d se3_to_vec(const Eigen::Matrix4d& se3mat);

/// T^-1 = (R^T, -R^T p) (MR Eq. 3.64).
Eigen::Matrix4d trans_inv(const Eigen::Matrix4d& T);

/// [Ad_T] = [[R, 0], [[p]R, R]] (MR Def. 3.20).
sbeg::Matrix6d adjoint(const Eigen::Matrix4d& T);

/// Rodrigues' formula (MR Eq. 3.51); below TOL the first-order form I + [omega].
Eigen::Matrix3d matrix_exp3(const Eigen::Matrix3d& so3mat);

/// e^[V] for [V] in se(3) (MR Eq. 3.88).  For a rotation below TOL the first-order form
/// (I + [omega], v) is used (modern_robotics would drop the rotation; Newton iterations on tiny
/// corrections need it).
Eigen::Matrix4d matrix_exp6(const Eigen::Matrix4d& se3mat);

/// [w]theta = log R, theta in [0, pi] (MR Sec. 3.2.3.3).
Eigen::Matrix3d matrix_log3(const Eigen::Matrix3d& R);

/// [V] = log T (MR Sec. 3.3.3.2).
Eigen::Matrix4d matrix_log6(const Eigen::Matrix4d& T);

/// T(theta) = e^[S1]th1 ... e^[Sn]thn M (MR Eq. 4.14).
Eigen::Matrix4d fk_space(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist, const Eigen::VectorXd& theta);

/// B_i = [Ad_{M^-1}] S_i (MR Eq. 4.16).
Eigen::MatrixXd body_screws(const Eigen::MatrixXd& Slist, const Eigen::Matrix4d& M);

/// J_s(theta) (MR Eq. 5.11) -- Lab 7.
Eigen::MatrixXd jacobian_space(const Eigen::MatrixXd& Slist, const Eigen::VectorXd& theta);

/// J_b(theta) (MR Eq. 5.18) -- Lab 7.
Eigen::MatrixXd jacobian_body(const Eigen::MatrixXd& Blist, const Eigen::VectorXd& theta);

/// A model of data/robots.json (numbers identical to the course robot models).
/// An empty path means the lab's data/robots.json.
Robot load_robot(const std::string& name, const std::string& path = "");

/// The hexapod of data/hexapod.json (A, B: 6 x 3, row i = a_i / b_i). An empty path means the lab's data file.
Hexapod load_hexapod(const std::string& path = "");

/// (t, cmd_deg, meas_deg) from a log in the project's hwcompare.py format
/// (header t, cmd_1..cmd_n, meas_1..meas_n[, stop]; angles in degrees).
JointLog read_joint_log(const std::string& path);

/// Write a command (and optionally measurement) log in the hwcompare.py format (4 decimals, as in Python).
/// An empty meas_deg writes the commanded joints as the measurement.
void write_joint_log(const std::string& path, const Eigen::VectorXd& t, const Eigen::MatrixXd& cmd_deg,
                     const Eigen::MatrixXd& meas_deg = Eigen::MatrixXd());

/// Derivative of every column of Y (N x m, row = sample) with respect to t (N samples, possibly non-uniform):
/// the same numbers as np.gradient(Y, t, axis=0) (second-order central differences inside, one-sided at the ends).
/// (C++ only: replaces np.gradient.)
Eigen::MatrixXd gradient(const Eigen::MatrixXd& Y, const Eigen::VectorXd& t);

// ---------------------------------------------------------------------------
// Checkpoint 1 -- inverse velocity kinematics
// ---------------------------------------------------------------------------

/// Moore-Penrose pseudo-inverse from the SVD J = U Sigma V^T: J^+ = V Sigma^+ U^T,
/// where Sigma^+ inverts every singular value > tol * sigma_max and sets the others to 0
/// (MR Sec. 6.2.2, p. 227: minimum-norm / least-squares solution of J y = z).  J is m x n, J^+ is n x m.
Eigen::MatrixXd pinv_svd(const Eigen::MatrixXd& J, double tol = 1e-10);

/// Damped least-squares inverse J^#(lam) = J^T (J J^T + lam^2 I)^-1 (n x m).
/// It minimises |J thetadot - V|^2 + lam^2 |thetadot|^2 and equals (J^T J + lam^2 I)^-1 J^T.
/// lam = 0 must return the pseudo-inverse (use pinv_svd).
Eigen::MatrixXd dls_inverse(const Eigen::MatrixXd& J, double lam);

/// (sigma, gain): the singular values of J (descending) and the gains sigma/(sigma^2 + lam^2)
/// that J^#(lam) applies along each singular direction (1/sigma when lam = 0; +infinity where sigma = 0).
std::pair<Eigen::VectorXd, Eigen::VectorXd> dls_singular_gains(const Eigen::MatrixXd& J, double lam);

/// Damping that switches on only near a singularity:
/// lam = 0 if sigma_min >= eps, else lam = lam_max * sqrt(1 - (sigma_min/eps)^2).
double adaptive_lambda(double sigma_min, double eps, double lam_max);

// ---------------------------------------------------------------------------
// Checkpoint 2 -- redundancy and null space
// ---------------------------------------------------------------------------

/// P = I - J^+ J (n x n): projects any joint velocity onto the null space of J.
Eigen::MatrixXd null_projector(const Eigen::MatrixXd& J);

/// W^-1 J^T (J W^-1 J^T)^-1: the thetadot with J thetadot = V that minimises
/// (1/2) thetadot^T W thetadot (W symmetric positive definite; W = mass matrix on MR p. 232).
Eigen::MatrixXd weighted_pinv(const Eigen::MatrixXd& J, const Eigen::MatrixXd& W);

/// thetadot = J^+ xdot + (I - J^+ J) qdot0 (primary task + secondary motion in the null space).
Eigen::VectorXd redundant_rates(const Eigen::MatrixXd& J, const Eigen::VectorXd& xdot, const Eigen::VectorXd& qdot0);

/// Gradient of h(theta) = 1/2 sum_i ((theta_i - c_i)/r_i)^2, c_i = mid-range, r_i = range
/// (limits: n x 2, row i = (min_i, max_i)).  qdot0 = -k grad h moves the joints towards the middle of their ranges.
Eigen::VectorXd joint_centering_gradient(const Eigen::VectorXd& theta, const Eigen::MatrixXd& limits);

// ---------------------------------------------------------------------------
// Checkpoint 3 -- resolved-rate control
// ---------------------------------------------------------------------------

/// Result of line_trajectory (Python: the tuple (t, Xd, Vd)).
struct LineTrajectory {
    Eigen::VectorXd t;                ///< N times 0, dt, ..., t_f
    std::vector<Eigen::Matrix4d> Xd;  ///< N desired poses
    Eigen::MatrixXd Vd;               ///< N x 6 desired body twists (row = sample)
};

/// Straight line at constant speed and constant orientation, starting at T0.
/// Returns (t, Xd, Vd): times 0, dt, ..., t_f (t_f = |displacement|/speed, last sample exactly at t_f),
/// the desired poses X_d(t) and the desired BODY twists V_d(t) (N x 6),
/// [V_d] = X_d^-1 Xdot_d = (0, R0^T u speed) while moving and 0 at the last sample.
LineTrajectory line_trajectory(const Eigen::Matrix4d& T0, const Eigen::Vector3d& displacement, double speed,
                               double dt);

/// Result of resolved_rate_sim (Python: the dict with the same keys).  N = number of samples.
struct RateSimResult {
    Eigen::MatrixXd theta;      ///< N x n joint angles (row = sample)
    Eigen::MatrixXd qdot;       ///< N x n joint rates (last row zeros)
    Eigen::VectorXd pos_err;    ///< N position errors |p_d - p| (m)
    Eigen::VectorXd ang_err;    ///< N orientation errors |log(R^T R_d)| (rad)
    Eigen::VectorXd sigma_min;  ///< N smallest singular values of J_b
    Eigen::VectorXd lam;        ///< N damping factors used (last entry 0)
};

/// Simulate body-frame resolved-rate control with feedforward and proportional feedback:
///     V_b = [Ad_{X^-1 X_d}] V_d + kp * X_e,   [X_e] = log(X^-1 X_d)          (MR Eq. 11.16, K_i = 0)
///     thetadot = J_b^#(lam) V_b                                              (MR Eq. 6.7, damped)
/// If eps and lam_max are given, lam = adaptive_lambda(sigma_min(J_b), eps, lam_max) at every step.
/// If qdot_max is given (non-empty: one entry = the same limit for every joint, or n entries) and some
/// |thetadot_i| > qdot_max_i, scale the WHOLE vector down (keeps its direction).
/// Integrate with explicit Euler: theta_{k+1} = theta_k + thetadot_k dt.  Vd is N x 6 (row = sample).
RateSimResult resolved_rate_sim(const Eigen::MatrixXd& Slist, const Eigen::Matrix4d& M, const Eigen::VectorXd& theta0,
                                const std::vector<Eigen::Matrix4d>& Xd, const Eigen::MatrixXd& Vd, double dt,
                                double kp = 0.0, double lam = 0.0, const Eigen::VectorXd& qdot_max = Eigen::VectorXd(),
                                std::optional<double> eps = std::nullopt, std::optional<double> lam_max = std::nullopt);

// ---------------------------------------------------------------------------
// Checkpoint 4 -- IK near singularities
// ---------------------------------------------------------------------------

/// Result of ik_newton (Python: the tuple (theta, success, iterations, err)).
struct IKNewtonResult {
    Eigen::VectorXd theta;
    bool success = false;
    int iterations = 0;
    std::vector<std::pair<double, double>> err;  ///< (|omega_b|, |v_b|) of every iterate, len = iterations + 1
};

/// Newton-Raphson IK in the body frame (MR Sec. 6.2.2) with an optional damped step:
///     V_b = log(T(theta)^-1 T_d);  theta <- theta + J_b^#(lam) V_b   (lam = 0: MR's J_b^+).
/// Stop when |omega_b| < eomg and |v_b| < ev, or after max_iter steps.
/// Returns (theta, success, iterations, err) where err is a list of (|omega_b|, |v_b|) for
/// every iterate including the initial guess (len = iterations + 1).
IKNewtonResult ik_newton(const Eigen::MatrixXd& Slist, const Eigen::Matrix4d& M, const Eigen::Matrix4d& T_d,
                         const Eigen::VectorXd& theta0, double lam = 0.0, double eomg = 1e-6, double ev = 1e-6,
                         int max_iter = 100);

// ---------------------------------------------------------------------------
// Checkpoint 5 -- hexapod (closed chain)
// ---------------------------------------------------------------------------

/// Leg lengths s_i = |p + R b_i - a_i| of a Stewart-Gough platform (MR Sec. 7.1.2, p. 248).
/// A, B: k x 3 (row i = a_i, b_i); returns k lengths.
Eigen::VectorXd hexapod_leg_lengths(const Eigen::Vector3d& p, const Eigen::Matrix3d& R, const Eigen::MatrixXd& A,
                                    const Eigen::MatrixXd& B);

/// k x 6 inverse Jacobian J_s^-1 with sdot = J_s^-1 V_s (space twist V_s = (omega_s, v_s)).
/// Row i = ( (-n_i x a_i)^T , n_i^T ), n_i = unit vector from a_i to the platform joint
/// (MR Sec. 7.2.1, p. 252, with q_i = a_i).
Eigen::MatrixXd hexapod_inverse_jacobian(const Eigen::Vector3d& p, const Eigen::Matrix3d& R, const Eigen::MatrixXd& A,
                                         const Eigen::MatrixXd& B);

/// Result of hexapod_forward_kinematics (Python: the tuple (p, R, iterations)).
struct HexapodFKResult {
    Eigen::Vector3d p;
    Eigen::Matrix3d R;
    int iterations = 0;
};

/// Forward kinematics by Newton-Raphson on the leg lengths: from the guess T = (R0, p0),
/// repeat  ds = s - s(T);  V_s = (J_s^-1)^-1 ds;  T <- e^[V_s] T  until max|ds| < tol.
/// Returns (p, R, iterations).
HexapodFKResult hexapod_forward_kinematics(const Eigen::VectorXd& s, const Eigen::MatrixXd& A,
                                           const Eigen::MatrixXd& B, const Eigen::Vector3d& p0,
                                           const Eigen::Matrix3d& R0, double tol = 1e-12, int max_iter = 50);

// ---------------------------------------------------------------------------
// Checkpoint 6 -- hardware session (sim first)
// ---------------------------------------------------------------------------

/// Result of tracking_report (Python: the dict with the same keys).
struct TrackingReport {
    Eigen::VectorXd joint_rms_deg;     ///< per joint, RMS of |cmd - meas|
    Eigen::VectorXd joint_max_deg;     ///< per joint, max of |cmd - meas|
    double task_rms_mm = 0.0;          ///< RMS tool-point distance |p(cmd) - p(meas)| (mm)
    double task_max_mm = 0.0;          ///< max tool-point distance (mm)
    Eigen::VectorXd peak_speed_deg_s;  ///< per joint, max |d meas / dt| (deg/s)
};

/// Hardware acceptance numbers from a joint log (degrees), as in project starter/hwcompare.py:
///   joint_rms_deg, joint_max_deg (per joint, |cmd - meas|),
///   task_rms_mm, task_max_mm (tool point |p(cmd) - p(meas)| by PoE FK, millimetres),
///   peak_speed_deg_s (per joint, measured, from gradient(meas, t)).
/// cmd_deg, meas_deg: N x n (row = sample).
TrackingReport tracking_report(const Eigen::VectorXd& t, const Eigen::MatrixXd& cmd_deg,
                               const Eigen::MatrixXd& meas_deg, const Eigen::MatrixXd& Slist,
                               const Eigen::Matrix4d& M);

}  // namespace lab

#endif  // SBEG361_LAB10_HPP
