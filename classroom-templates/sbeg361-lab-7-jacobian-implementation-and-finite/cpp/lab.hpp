// SBEG 361 Lab 7 -- Space, body and analytic Jacobians; finite-difference verification (C++ track): declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp. Do not change these declarations: test_lab.cpp calls them.
//
// Notation follows Lynch & Park, *Modern Robotics* (MR), Section 5.1:
//   T(theta) = e^[S1]th1 ... e^[Sn]thn M = M e^[B1]th1 ... e^[Bn]thn          (MR Eqs. 4.14, 4.16)
//   V_s = J_s(theta) thetadot,  J_s1 = S1,  J_si = [Ad_{e^[S1]th1...e^[S(i-1)]th(i-1)}] S_i   (MR Eqs. 5.10-5.11)
//   V_b = J_b(theta) thetadot,  J_bn = Bn,  J_bi = [Ad_{e^-[Bn]thn...e^-[B(i+1)]th(i+1)}] B_i (MR Eqs. 5.17-5.18)
//   J_b = [Ad_Tbs] J_s,  J_s = [Ad_Tsb] J_b                                             (MR Eqs. 5.22-5.23)
//   analytic Jacobian for q = (r, x), r = exponential coordinates of R_sb:
//   J_a = [[A(r)^-1, 0], [0, R_sb]] J_b,  A(r) = I - (1-cos|r|)/|r|^2 [r] + (|r|-sin|r|)/|r|^3 [r]^2  (MR Eqs. 5.24-5.25)
//   velocity of the origin of {b}:  pdot = v_s + w_s x p = R_sb v_b                (MR Eqs. 3.69, 3.73)
//
// Python -> C++: ValueError -> std::invalid_argument; screw lists Slist / Blist and Jacobians -> Eigen::MatrixXd
// (6 x n, one axis or column per joint, as in Python); theta -> Eigen::VectorXd; the forward-kinematics callable
// `fk` -> lab::FkFunction (any lambda  [&](const Eigen::VectorXd& q) -> Eigen::Matrix4d { ... });
// the scheme string "forward" / "central" -> std::string; an (N, 3) or (N, 6) array -> Eigen::MatrixXd (row = sample).
#ifndef SBEG361_LAB7_HPP
#define SBEG361_LAB7_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace lab {

/// Same threshold as modern_robotics.NearZero.
constexpr double TOL = 1e-6;

/// Forward kinematics as a function of the joint vector: theta -> 4x4 T_sb.
using FkFunction = std::function<Eigen::Matrix4d(const Eigen::VectorXd&)>;

/// A robot model of data/robots.json (Python: the tuple (Slist, M, info)).
struct Robot {
    Eigen::MatrixXd Slist;    ///< 6 x n space screw axes (columns)
    Eigen::Matrix4d M;        ///< home configuration of {b}
    std::string name;         ///< info["name"]
    std::string joint_types;  ///< info["joint_types"], e.g. "RRRRRR"
    Eigen::MatrixXd limits;   ///< n x 2 joint limits (rad or m); empty if null in the file
    Eigen::VectorXd qdot_max; ///< rated joint speeds; empty if null in the file (e.g. "rrp")
    Eigen::VectorXd q_home;   ///< home joint vector; empty if null in the file
};

// ---------------------------------------------------------------------------
// Helpers (complete; from Labs 2-5) -- given in starter.cpp
// ---------------------------------------------------------------------------

/// 3x3 skew-symmetric matrix [omega] (MR Eq. 3.30).
Eigen::Matrix3d vec_to_so3(const Eigen::Vector3d& omega);

/// omega from [omega] (reads the entries (2,1), (0,2), (1,0)).
Eigen::Vector3d so3_to_vec(const Eigen::Matrix3d& so3mat);

/// 4x4 matrix [V] of a twist V = (omega, v) (MR Eq. 3.71).
Eigen::Matrix4d vec_to_se3(const sbeg::Vector6d& V);

/// V = (omega, v) from a 4x4 matrix [V] (entries read as in MR se3ToVec).
sbeg::Vector6d se3_to_vec(const Eigen::Matrix4d& se3mat);

/// T^-1 = (R^T, -R^T p) (MR Eq. 3.64).
Eigen::Matrix4d trans_inv(const Eigen::Matrix4d& T);

/// 6x6 adjoint [Ad_T] = [[R, 0], [[p]R, R]] (MR Def. 3.20).
sbeg::Matrix6d adjoint(const Eigen::Matrix4d& T);

/// Rodrigues' formula e^[w]theta (MR Eq. 3.51).
Eigen::Matrix3d matrix_exp3(const Eigen::Matrix3d& so3mat);

/// [w]theta = log R, theta in [0, pi] (MR Algorithm, Sec. 3.2.3.3).
Eigen::Matrix3d matrix_log3(const Eigen::Matrix3d& R);

/// e^[S]theta for a twist matrix (MR Eqs. 3.88-3.89).
Eigen::Matrix4d matrix_exp6(const Eigen::Matrix4d& se3mat);

/// [S]theta = log T (MR Algorithm, Sec. 3.3.3.2).
Eigen::Matrix4d matrix_log6(const Eigen::Matrix4d& T);

/// Space-form PoE forward kinematics (MR Eq. 4.14).
Eigen::Matrix4d fk_space(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist, const Eigen::VectorXd& theta);

/// Body-form PoE forward kinematics (MR Eq. 4.16).
Eigen::Matrix4d fk_body(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Blist, const Eigen::VectorXd& theta);

/// B_i = [Ad_{M^-1}] S_i (MR Eq. 4.16).
Eigen::MatrixXd body_screws(const Eigen::MatrixXd& Slist, const Eigen::Matrix4d& M);

/// A model of data/robots.json (numbers identical to the course robot models).
/// An empty path means the lab's data/robots.json.
Robot load_robot(const std::string& name, const std::string& path = "");

/// (t, Q) from data/teleop_needle.csv; Q is N x 6 (rad). An empty path means the lab's data file.
std::pair<Eigen::VectorXd, Eigen::MatrixXd> load_teleop_csv(const std::string& path = "");

/// Least-squares slope of log(err) against log(h) (the observed order of accuracy).
double fit_slope(const Eigen::VectorXd& hs, const Eigen::VectorXd& errs);

/// Derivative of every column of Y (N x m, row = sample) with respect to t (N samples, possibly non-uniform):
/// the same numbers as np.gradient(Y, t, axis=0) (second-order central differences inside, one-sided at the ends).
/// (C++ only: replaces np.gradient.)
Eigen::MatrixXd gradient(const Eigen::MatrixXd& Y, const Eigen::VectorXd& t);

// ---------------------------------------------------------------------------
// Checkpoint 1 -- geometric Jacobians
// ---------------------------------------------------------------------------

/// Space Jacobian J_s(theta), 6 x n (MR Def. 5.1, Eq. 5.11).
/// Column 1 is S_1; column i is [Ad_{T_(i-1)}] S_i with T_(i-1) = e^[S1]th1 ... e^[S(i-1)]th(i-1).
Eigen::MatrixXd jacobian_space(const Eigen::MatrixXd& Slist, const Eigen::VectorXd& theta);

/// Body Jacobian J_b(theta), 6 x n (MR Def. 5.4, Eq. 5.18).
/// Column n is B_n; column i is [Ad_{e^-[Bn]thn ... e^-[B(i+1)]th(i+1)}] B_i.
Eigen::MatrixXd jacobian_body(const Eigen::MatrixXd& Blist, const Eigen::VectorXd& theta);

/// J_b = [Ad_{T_bs}] J_s (MR Eq. 5.22).
Eigen::MatrixXd body_from_space(const Eigen::MatrixXd& Js, const Eigen::Matrix4d& T_sb);

/// J_s = [Ad_{T_sb}] J_b (MR Eq. 5.23).
Eigen::MatrixXd space_from_body(const Eigen::MatrixXd& Jb, const Eigen::Matrix4d& T_sb);

/// 3 x n Jacobian of the velocity of a point p (in {s}) fixed to the end-effector:
/// pdot = v_s + w_s x p = (J_v - [p] J_w) thetadot, where J_w, J_v are the top and bottom rows of J_s.
Eigen::MatrixXd point_jacobian(const Eigen::MatrixXd& Js, const Eigen::Vector3d& p);

// ---------------------------------------------------------------------------
// Checkpoint 2 -- finite-difference checks
// ---------------------------------------------------------------------------

/// Numerical space Jacobian from finite differences of T(theta).
/// fk: callable theta -> 4x4 T_sb.  For each joint i, approximate dT/dtheta_i by
///   forward:  (T(theta + h e_i) - T(theta)) / h
///   central:  (T(theta + h e_i) - T(theta - h e_i)) / (2h)
/// and take column i = se3_to_vec( dT/dtheta_i  T(theta)^-1 )      ([V_s] = Tdot T^-1, MR Eq. 3.74).
/// Throw std::invalid_argument for any other scheme.
Eigen::MatrixXd fd_jacobian_space(const FkFunction& fk, const Eigen::VectorXd& theta, double h = 1e-6,
                                  const std::string& scheme = "central");

/// Numerical body Jacobian: column i = se3_to_vec( T^-1 dT/dtheta_i ) ([V_b] = T^-1 Tdot, MR Eq. 3.69).
/// Throw std::invalid_argument for a scheme other than "forward" / "central".
Eigen::MatrixXd fd_jacobian_body(const FkFunction& fk, const Eigen::VectorXd& theta, double h = 1e-6,
                                 const std::string& scheme = "central");

/// Space Jacobian from the matrix log of a finite joint step:
/// column i = se3_to_vec( log( T(theta + h e_i) T(theta)^-1 ) ) / h.
/// Exact (up to rounding) for any 0 < |h| < pi: moving one joint is a screw motion about J_si.
Eigen::MatrixXd log_jacobian_space(const FkFunction& fk, const Eigen::VectorXd& theta, double h = 0.1);

/// Max-abs error of fd_jacobian_space(fk, theta, h, scheme) against J_exact for every h in hs
/// (one entry per h).
Eigen::VectorXd fd_error_study(const FkFunction& fk, const Eigen::MatrixXd& J_exact, const Eigen::VectorXd& theta,
                               const Eigen::VectorXd& hs, const std::string& scheme = "central");

// ---------------------------------------------------------------------------
// Checkpoint 3 -- analytic Jacobian
// ---------------------------------------------------------------------------

/// A(r) with omega_b = A(r) rdot for exponential coordinates r of R_sb (MR Example 5.5).
Eigen::Matrix3d a_matrix(const Eigen::Vector3d& r);

/// 6 x n analytic Jacobian of q = (r, x): r = exponential coordinates of R_sb, x = p_sb
/// (MR Eq. 5.25): J_a = [[A(r)^-1, 0], [0, R_sb]] J_b.
Eigen::MatrixXd analytic_jacobian(const Eigen::MatrixXd& Jb, const Eigen::Matrix4d& T_sb);

/// q = (r, x): exponential coordinates of the rotation and the position of {b}.  (Given.)
sbeg::Vector6d minimal_coords(const Eigen::Matrix4d& T);

/// Central finite differences of q(theta) = minimal_coords(fk(theta)) (valid for |r| < pi).
Eigen::MatrixXd fd_analytic_jacobian(const FkFunction& fk, const Eigen::VectorXd& theta, double h = 1e-6);

// ---------------------------------------------------------------------------
// Checkpoint 4 -- teleoperation record
// ---------------------------------------------------------------------------

/// N x 3 matrix of tool-tip (origin of {b}) velocities pdot_k = point_jacobian(J_s(q_k), p_k) qdot_k
/// (row k = sample k; Q and Qdot are N x n, row = sample).
Eigen::MatrixXd tip_velocity_series(const Eigen::MatrixXd& Slist, const Eigen::Matrix4d& M, const Eigen::MatrixXd& Q,
                                    const Eigen::MatrixXd& Qdot);

}  // namespace lab

#endif  // SBEG361_LAB7_HPP
