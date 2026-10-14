// SBEG 461 Lab 4 -- URDF dynamics, RNEA timing and the constrained 2R arm (C++ track): declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp. The provided helpers of starter.py are given in helpers.cpp. Do not change these
// declarations: test_lab.cpp calls them.
//
// Notation follows Modern Robotics (MR): twists V = (omega, v), wrenches F = (m, f), screw axes S,
// Mlist = [M01, M12, ..., M_{n,n+1}] (link CoM frames at theta = 0), Glist = [G1, ..., Gn] (6x6 spatial
// inertias in the CoM frames, MR Eq. 8.32), Slist = 6 x n space screw axes (axes are the COLUMNS).
//
// Parts
//   A. URDF -> (Mlist, Glist, Slist)                         MR Sec. 8.8, Sec. 4.2
//   B. Recursive Newton-Euler inverse dynamics (your own)     MR Eqs. 8.50-8.54  + timing helper
//   C. Task-space dynamics                                    MR Eqs. 8.89-8.91
//   D. Constrained dynamics of the 2R arm in a channel        MR Sec. 8.7, Eqs. 8.93-8.100, Example 8.5
//      + Baumgarte stabilisation (Featherstone Sec. 8.3, beyond MR)
//
// Python -> C++: ValueError -> std::invalid_argument; a tuple -> std::pair or a small struct with the same field
// names; a list of 4x4 / 6x6 arrays -> std::vector<Eigen::Matrix4d> / std::vector<Matrix6d>; a Python callable ->
// std::function (pass a lambda).
#ifndef SBEG461_LAB4_HPP
#define SBEG461_LAB4_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include <cstddef>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace lab {

using Vector6d = sbeg::Vector6d;
using Matrix6d = sbeg::Matrix6d;
using TransformList = std::vector<Eigen::Matrix4d>;  // Mlist (same type as mr::TransformList)
using InertiaList = std::vector<Matrix6d>;           // Glist (same type as mr::InertiaList)

// ===========================================================================
// Provided helpers (given in helpers.cpp, do not edit) -- the small SE(3) toolkit of starter.py,
// so your code does not depend on modern_robotics (mrcpp); the tests use mrcpp as ground truth.
// ===========================================================================

/// [w] for a 3-vector (MR Eq. 3.30).
Eigen::Matrix3d skew(const Eigen::Vector3d& w);

/// URDF fixed-axis roll-pitch-yaw: R = Rot(z, yaw) Rot(y, pitch) Rot(x, roll).
Eigen::Matrix3d rpy_to_R(const Eigen::Vector3d& rpy);

/// Homogeneous transform from R and p (Python: RpToTrans).
Eigen::Matrix4d rp_to_trans(const Eigen::Matrix3d& R, const Eigen::Vector3d& p);

/// T^-1 (Python: TransInv).
Eigen::Matrix4d trans_inv(const Eigen::Matrix4d& T);

/// [Ad_T] (MR Def. 3.20) (Python: Adjoint).
Matrix6d adjoint(const Eigen::Matrix4d& T);

/// [ad_V] (MR Eq. 8.38).
Matrix6d ad(const Vector6d& V);

/// e^{[S] theta} for a screw axis S with ||omega|| = 1 or (omega = 0, ||v|| = 1) (MR Prop. 3.25)
/// (Python: MatrixExp6(S, theta)).
Eigen::Matrix4d matrix_exp6(const Vector6d& S, double theta);

/// One joint of a serial-chain URDF, as returned by read_urdf_chain (the dict keys of the Python version).
struct UrdfJoint {
    std::string name;
    std::string type;                                     // "revolute" | "continuous" | "fixed"
    Eigen::Matrix4d T_origin = Eigen::Matrix4d::Identity();  // child-link frame in the parent-link frame (joint at 0)
    Eigen::Vector3d axis = Eigen::Vector3d::Zero();       // unit joint axis in the child-link frame (0 for fixed)
    std::string child;                                    // name of the child link
    double mass = 0.0;                                    // 0 if the child link has no <inertial>
    Eigen::Matrix4d T_com = Eigen::Matrix4d::Identity();  // inertial (CoM) frame of the child link in the child-link frame
    Eigen::Matrix3d I = Eigen::Matrix3d::Zero();          // rotational inertia about the CoM, in the inertial frame
};

/// Parse a serial-chain URDF into a list of joints ordered from the base to the tip (see UrdfJoint).
std::vector<UrdfJoint> read_urdf_chain(const std::string& path);

/// Read states_arm6.csv -> (N, 18) matrix (row = theta(6), thetadot(6), thetaddot(6)).
Eigen::MatrixXd load_states(const std::string& csv_path);

/// Write the plain-text files read by the Step 3 program rnea_bench (main.cpp):
///   model file : line 1 = n; then n+1 matrices M_{i-1,i} (4 rows of 4); n matrices G_i (6 rows of 6);
///                then n screw axes S_i (one row of 6 each).
///   states file: one state per line, 3n numbers (theta, thetadot, thetaddot).
void export_for_cpp(const std::string& model_path, const std::string& states_path, const TransformList& Mlist,
                    const InertiaList& Glist, const Eigen::MatrixXd& Slist, const Eigen::MatrixXd& states);

/// Time your rnea and mr::inverse_dynamics (mrcpp) with your time_per_call on the first n_states states.
/// Returns name -> microseconds per call. (Python: benchmark; Pinocchio and Python are timed in Python.)
std::map<std::string, double> benchmark(const std::string& urdf_path, const std::string& csv_path,
                                        int n_states = 200);

// ===========================================================================
// Part A -- URDF -> MR dynamic parameters (MR Sec. 8.8)
// ===========================================================================

/// (Mlist, Glist, Slist) of a serial chain (Python: the returned tuple).
struct MRModel {
    TransformList Mlist;    // n + 1 matrices M_{i-1,i}
    InertiaList Glist;      // n spatial inertias G_i
    Eigen::MatrixXd Slist;  // 6 x n, column i = S_i
};

/// Return (Mlist, Glist, Slist) for the serial chain in a URDF file.
///
/// Frames: {0} = base_link frame; {i} = CoM (inertial) frame of moving link i; {n+1} = frame of the last
/// link reached through fixed joints (the tool frame). All poses at theta = 0.
///   M_{i-1,i} = T_{0,i-1}^{-1} T_{0,i}   (Mlist has n + 1 entries),
///   G_i = diag(I_i, m_i I_3)            (MR Eq. 8.32, I_i about the CoM in the inertial frame),
///   S_i = (omega_i, -omega_i x q_i)     (omega_i = joint axis in {0}, q_i = joint-frame origin in {0}).
/// Moving links are those that are the child of a revolute/continuous joint. Fixed joints after the last
/// moving joint only move the tool frame (their links are assumed massless).
MRModel urdf_to_mr(const std::string& path);

// ===========================================================================
// Part B -- RNEA (MR Eqs. 8.50-8.54) and timing
// ===========================================================================

/// Joint torques tau = M(theta) thetaddot + h(theta, thetadot) + J^T Ftip by the recursive
/// Newton-Euler algorithm (same interface as mr::inverse_dynamics / modern_robotics.InverseDynamics).
///
/// Forward:  T_{i,i-1} = e^{-[A_i] theta_i} M_{i,i-1}                         (8.50)
///           V_i = Ad_{T_{i,i-1}} V_{i-1} + A_i thetadot_i                       (8.51)
///           Vdot_i = Ad_{T_{i,i-1}} Vdot_{i-1} + ad_{V_i} A_i thetadot_i + A_i thetaddot_i   (8.52)
/// Backward: F_i = Ad^T_{T_{i+1,i}} F_{i+1} + G_i Vdot_i - ad^T_{V_i} G_i V_i   (8.53)
///           tau_i = F_i^T A_i                                                    (8.54)
/// with V_0 = 0, Vdot_0 = (0, -g) (gravity as a fictitious base acceleration) and F_{n+1} = Ftip.
Eigen::VectorXd rnea(const Eigen::VectorXd& thetalist, const Eigen::VectorXd& dthetalist,
                     const Eigen::VectorXd& ddthetalist, const Eigen::Vector3d& g, const Vector6d& Ftip,
                     const TransformList& Mlist, const InertiaList& Glist, const Eigen::MatrixXd& Slist);

/// Median over `repeats` runs of the mean wall-clock time (seconds) of one call.
/// Python: time_per_call(fn, arg_tuples, repeats) calls fn(*args) for every args in arg_tuples. C++: fn(k) makes
/// the k-th call (k = 0 .. n_calls - 1), e.g. a lambda that calls rnea with the k-th state.
/// Use std::chrono::steady_clock. Throw std::invalid_argument if n_calls == 0 or repeats < 1.
double time_per_call(const std::function<void(std::size_t)>& fn, std::size_t n_calls, int repeats = 3);

// ===========================================================================
// Part C -- Task-space dynamics (MR Sec. 8.6)
// ===========================================================================

/// Lambda = J^-T M J^-1 (8.90) and eta = J^-T h - Lambda Jdot J^-1 V with J^-1 V = thetadot (8.91).
/// J must be square and invertible. Returns (Lambda, eta).
std::pair<Eigen::MatrixXd, Eigen::VectorXd> task_space_dynamics(const Eigen::MatrixXd& M, const Eigen::VectorXd& h,
                                                                const Eigen::MatrixXd& J, const Eigen::MatrixXd& Jdot,
                                                                const Eigen::VectorXd& dthetalist);

// ===========================================================================
// Part D -- The MR 2R arm (Fig. 8.1) and constrained dynamics (Sec. 8.7)
// ===========================================================================

/// M(theta) and h(theta, thetadot) = c + g of the MR 2R arm with point masses at the link ends
/// (MR Eqs. 8.9-8.10); gravity of magnitude g acts along -y. Returns (M, h).
std::pair<Eigen::Matrix2d, Eigen::Vector2d> twor_dynamics(const Eigen::Vector2d& theta, const Eigen::Vector2d& dtheta,
                                                          double m1 = 1.0, double m2 = 1.0, double L1 = 1.0,
                                                          double L2 = 1.0, double g = 0.0);

/// Tip Jacobian J (for (xdot, ydot)) and its time derivative Jdot (MR Example 8.5). Returns (J, Jdot).
std::pair<Eigen::Matrix2d, Eigen::Matrix2d> twor_jacobian(const Eigen::Vector2d& theta, const Eigen::Vector2d& dtheta,
                                                          double L1 = 1.0, double L2 = 1.0);

/// Tip in a frictionless vertical channel x = const (MR Example 8.5).
/// Holonomic constraint L1 c1 + L2 c12 = x_c  ->  A(theta) thetadot = 0 with A = d x / d theta (1 x 2).
/// Returns (A, Adot), both 1 x 2 matrices.
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> channel_constraint(const Eigen::Vector2d& theta,
                                                               const Eigen::Vector2d& dtheta, double L1 = 1.0,
                                                               double L2 = 1.0);

/// Solve tau = M thetaddot + h + A^T lambda with Adot thetadot + A thetaddot = 0 (MR Eqs. 8.93-8.98).
/// A, Adot are k x n. Returns (thetaddot (n), lam (k)).
std::pair<Eigen::VectorXd, Eigen::VectorXd> constrained_forward_dynamics(const Eigen::MatrixXd& M,
                                                                         const Eigen::VectorXd& h,
                                                                         const Eigen::MatrixXd& A,
                                                                         const Eigen::MatrixXd& Adot,
                                                                         const Eigen::VectorXd& dtheta,
                                                                         const Eigen::VectorXd& tau);

/// tau = M thetaddot + h + A^T lambda (MR Eq. 8.93); thetaddot must satisfy (8.95).
Eigen::VectorXd constrained_inverse_dynamics(const Eigen::MatrixXd& M, const Eigen::VectorXd& h,
                                             const Eigen::MatrixXd& A, const Eigen::VectorXd& ddtheta,
                                             const Eigen::VectorXd& lam);

/// P = I - A^T (A M^-1 A^T)^-1 A M^-1 (MR Eq. 8.100), rank n - k.
Eigen::MatrixXd constraint_projector(const Eigen::MatrixXd& M, const Eigen::MatrixXd& A);

/// tau_fun(t, theta, dtheta) -> joint torques (2).
using TorqueFunction = std::function<Eigen::Vector2d(double, const Eigen::Vector2d&, const Eigen::Vector2d&)>;

/// Result of simulate_channel_2r (Python: the returned dict, same keys).
struct ChannelSim {
    Eigen::VectorXd t;       // (N+1)
    Eigen::MatrixXd theta;   // (N+1) x 2, row k = theta at t(k)
    Eigen::MatrixXd dtheta;  // (N+1) x 2
    Eigen::VectorXd lam;     // (N+1)
    Eigen::VectorXd e;       // (N+1)
};

/// RK4 simulation of the 2R arm whose tip is held in the channel x = x_c.
///
/// tau_fun(t, theta, dtheta) -> joint torques (2).
/// Constraint error e = L1 c1 + L2 c12 - x_c, edot = A thetadot. With alpha = beta = 0 the acceleration
/// constraint is MR Eq. 8.95; with alpha, beta > 0 it becomes edd + 2 alpha edot + beta^2 e = 0
/// (Baumgarte stabilisation, Featherstone Sec. 8.3), i.e. the lambda equation (8.98) gets the extra term
/// + 2 alpha edot + beta^2 e on the right-hand side.
///
/// Returns t (N+1), theta (N+1, 2), dtheta (N+1, 2), lam (N+1), e (N+1) with N = round(T / dt).
ChannelSim simulate_channel_2r(const Eigen::Vector2d& theta0, const Eigen::Vector2d& dtheta0,
                               const TorqueFunction& tau_fun, double T, double dt, double x_c = 1.0,
                               double alpha = 0.0, double beta = 0.0, double m1 = 1.0, double m2 = 1.0,
                               double L1 = 1.0, double L2 = 1.0, double g = 0.0);

}  // namespace lab

#endif  // SBEG461_LAB4_HPP
