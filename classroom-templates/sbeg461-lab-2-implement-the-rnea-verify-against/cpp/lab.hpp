// SBEG 461 Lab 2 -- Newton-Euler inverse dynamics (RNEA), verified against modern_robotics and Pinocchio
// (C++ track): declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp. Do not change these declarations: test_lab.cpp calls them.
// You may NOT call mr::inverse_dynamics, mr::mass_matrix, ... or pin_rnea inside checkpoints 1-4 and 6: those are
// the references your code is tested against.  You MAY use the mrcpp helpers mr::adjoint, mr::ad, mr::trans_inv,
// mr::matrix_exp6, mr::vec_to_se3, mr::rp_to_trans, mr::trans_to_rp (or the equivalents from your SBEG 361 library).
//
// Notation follows Modern Robotics (MR) Sections 8.2-8.3:
//     twists V = (omega, v), wrenches F = (m, f)  -- ANGULAR FIRST (Pinocchio uses LINEAR first!)
//     G_b = [[I_b, 0], [0, m I]]                              (MR Eq. 8.32)
//     G_a = [Ad_{T_ba}]^T G_b [Ad_{T_ba}]                      (MR Eq. 8.42)
//     F_b = G_b Vdot_b - [ad_{V_b}]^T G_b V_b                  (MR Eq. 8.40)
//     A_i = [Ad_{M_i^{-1}}] S_i,   M_i = M_{0,i}
//     T_{i,i-1} = exp(-[A_i] theta_i) M_{i,i-1}                (MR Eq. 8.50)
//     V_i = [Ad_{T_{i,i-1}}] V_{i-1} + A_i dtheta_i            (MR Eq. 8.51)
//     Vdot_i = [Ad_{T_{i,i-1}}] Vdot_{i-1} + [ad_{V_i}] A_i dtheta_i + A_i ddtheta_i      (MR Eq. 8.52)
//     F_i = [Ad_{T_{i+1,i}}]^T F_{i+1} + G_i Vdot_i - [ad_{V_i}]^T G_i V_i               (MR Eq. 8.53)
//     tau_i = F_i^T A_i                                       (MR Eq. 8.54)
//     V_0 = 0,  Vdot_0 = (0, -g)  (gravity as a fictitious base acceleration),  F_{n+1} = F_tip
// F_tip is the wrench the END-EFFECTOR APPLIES TO THE ENVIRONMENT, expressed in {n+1} (MR p. 292).
//
// Robot data (data/robots.json) is in modern_robotics list format: Mlist = [M_{0,1}, ..., M_{n,n+1}],
// Glist = [G_1, ..., G_n] (centre-of-mass frames), Slist = space screw axes as COLUMNS (6 x n), g = gravity in {0}.
//
// Python -> C++: a dict -> a small struct with the same field names; a tuple -> std::pair or a struct; a list of
// matrices -> std::vector; an (N, n) trajectory array -> Eigen::MatrixXd (ROW k = time sample k); Ftip=None ->
// std::nullopt (an empty std::optional); KeyError / ValueError -> std::invalid_argument.
#ifndef SBEG461_LAB2_HPP
#define SBEG461_LAB2_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include "pinlite.hpp"  // given: the Pinocchio stand-in for Checkpoint 5

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lab {

using sbeg::Matrix6d;
using sbeg::Vector6d;

// ---------------------------------------------------------------------------
// Provided helpers (given in starter.cpp)
// ---------------------------------------------------------------------------

/// A robot of data/robots.json (the Python dict returned by load_robot).
struct Robot {
    std::vector<Eigen::Matrix4d> Mlist;  ///< n+1 transforms M_{i-1,i}
    std::vector<Matrix6d> Glist;         ///< n spatial inertias (CoM frames)
    Eigen::MatrixXd Slist;               ///< 6 x n, space screw axes as columns
    Eigen::Vector3d g;                   ///< gravity in {0}
    int n = 0;                           ///< number of joints
    std::string description;
    std::string name;
};

/// One row of data/states.csv.
struct State {
    Eigen::VectorXd theta, dtheta, ddtheta;
    Vector6d Ftip;
};

/// The content of a trajectory file: t (N), theta, dtheta, ddtheta (N x n, row = time sample).
struct Trajectory {
    Eigen::VectorXd t;
    Eigen::MatrixXd theta, dtheta, ddtheta;
};

/// Return robot `name` from data/robots.json.  (Provided.)  Throws std::invalid_argument for an unknown name.
Robot load_robot(const std::string& name);

/// Return the test states of robot `name` from data/states.csv as (theta, dtheta, ddtheta, Ftip).  (Provided.)
std::vector<State> load_states(const std::string& name);

/// Read data/<fname> (columns t, theta1..n, dtheta1..n, ddtheta1..n).  (Provided.)
Trajectory load_trajectory(const std::string& fname);

// ---------------------------------------------------------------------------
// Checkpoint 1 -- one rigid body
// ---------------------------------------------------------------------------

/// 6x6 spatial inertia G_b of a body in a centre-of-mass frame (MR Eq. 8.32).
Matrix6d spatial_inertia(const Eigen::Matrix3d& I, double m);

/// Spatial inertia G_a in frame {a}, given G_b in {b} and T_ba = configuration of {a} seen from {b} (MR Eq. 8.42).
Matrix6d transform_spatial_inertia(const Matrix6d& G_b, const Eigen::Matrix4d& T_ba);

/// Net wrench F = G Vdot - [ad_V]^T G V on one rigid body (MR Eq. 8.40; valid in any body-fixed frame, Eq. 8.43).
Vector6d rigid_body_wrench(const Matrix6d& G, const Vector6d& V, const Vector6d& Vdot);

// ---------------------------------------------------------------------------
// Checkpoint 2 -- forward iterations
// ---------------------------------------------------------------------------

/// 6 x n matrix whose column i is A_i = [Ad_{M_i^{-1}}] S_i, the joint-i screw axis in link frame {i}.
Eigen::MatrixXd link_screw_axes(const std::vector<Eigen::Matrix4d>& Mlist, const Eigen::MatrixXd& Slist);

/// Result of forward_iterations (the Python dict with keys "A", "T", "V", "Vdot").
struct ForwardPass {
    Eigen::MatrixXd A;                 ///< 6 x n link-frame screw axes
    std::vector<Eigen::Matrix4d> T;    ///< n+1 transforms; T[i] = T_{i+1,i} (MR 1-based) for i < n, T[n] = M_{n+1,n}
    Eigen::MatrixXd V;                 ///< 6 x (n+1), column 0 = V_0 = 0, column i = V_i
    Eigen::MatrixXd Vdot;              ///< 6 x (n+1), column 0 = Vdot_0 = (0, -g), column i = Vdot_i
};

/// Base-to-tip pass of the RNEA (MR Eqs. 8.50-8.52).
ForwardPass forward_iterations(const Eigen::VectorXd& theta, const Eigen::VectorXd& dtheta,
                               const Eigen::VectorXd& ddtheta, const Eigen::Vector3d& g,
                               const std::vector<Eigen::Matrix4d>& Mlist, const Eigen::MatrixXd& Slist);

// ---------------------------------------------------------------------------
// Checkpoint 3 -- backward iterations and full RNEA
// ---------------------------------------------------------------------------

/// Tip-to-base pass (MR Eqs. 8.53-8.54).
/// Returns (tau, F): tau (n), F 6 x (n+1) with column i-1 = F_i (wrench transmitted through joint i to link i,
/// in {i}) for i = 1..n and the last column = F_{n+1} = Ftip.
std::pair<Eigen::VectorXd, Eigen::MatrixXd> backward_iterations(const ForwardPass& fwd, const Vector6d& Ftip,
                                                                const std::vector<Matrix6d>& Glist);

/// Joint forces/torques tau = M(theta) ddtheta + c(theta, dtheta) + g(theta) + J_b^T(theta) Ftip (MR Section 8.3.2).
Eigen::VectorXd rnea(const Eigen::VectorXd& theta, const Eigen::VectorXd& dtheta, const Eigen::VectorXd& ddtheta,
                     const Eigen::Vector3d& g, const Vector6d& Ftip, const std::vector<Eigen::Matrix4d>& Mlist,
                     const std::vector<Matrix6d>& Glist, const Eigen::MatrixXd& Slist);

// ---------------------------------------------------------------------------
// Checkpoint 4 -- M, c, g and J^T F by RNEA calls
// ---------------------------------------------------------------------------

/// The Python tuple (M, c, grav).
struct DynamicsTerms {
    Eigen::MatrixXd M;
    Eigen::VectorXd c;
    Eigen::VectorXd grav;
};

/// Return (M, c, grav) using only calls to `rnea` (Featherstone Section 5.3, p. 94):
///     grav = rnea(theta, 0, 0, g, 0);  c = rnea(theta, dtheta, 0, 0, 0);  column j of M = rnea(theta, 0, e_j, 0, 0).
DynamicsTerms dynamics_terms(const Eigen::VectorXd& theta, const Eigen::VectorXd& dtheta, const Eigen::Vector3d& g,
                             const std::vector<Eigen::Matrix4d>& Mlist, const std::vector<Matrix6d>& Glist,
                             const Eigen::MatrixXd& Slist);

/// Joint torques that balance the tip wrench alone (= J_b^T Ftip, MR Eq. 5.26), computed with one RNEA call.
Eigen::VectorXd tip_wrench_torques(const Eigen::VectorXd& theta, const Vector6d& Ftip,
                                   const std::vector<Eigen::Matrix4d>& Mlist, const std::vector<Matrix6d>& Glist,
                                   const Eigen::MatrixXd& Slist);

// ---------------------------------------------------------------------------
// Checkpoint 5 -- Pinocchio conventions (pinocchio_model, pin_rnea: given in pinlite.hpp / pinlite.cpp)
// ---------------------------------------------------------------------------

/// tau from pin_rnea for the model built by `pinocchio_model`, including the MR tip wrench Ftip (in {n+1}).
/// pin_rnea's fext[j] is the wrench applied ON body j, expressed in joint frame j, LINEAR first.
/// MR's Ftip is the wrench applied BY the end-effector, in {n+1}, ANGULAR first.  So:
///     F_on = -Ftip (in {n+1});  F_J = [Ad_{T_ee,Jn}]^T F_on (MR Eq. 3.98);  fext[n] = (F_J[3:6], F_J[0:3]).
/// Ftip = std::nullopt (Python None): no external force.
Eigen::VectorXd pinocchio_inverse_dynamics(const PinModel& model, const PinInfo& info, const Eigen::VectorXd& theta,
                                           const Eigen::VectorXd& dtheta, const Eigen::VectorXd& ddtheta,
                                           const std::optional<Vector6d>& Ftip = std::nullopt);

// ---------------------------------------------------------------------------
// Checkpoint 6 -- application: torque profiles
// ---------------------------------------------------------------------------

/// N x n joint-torque history along a trajectory (rows = time samples), using your `rnea`.
/// Ftip = std::nullopt (Python None) means a zero tip wrench.
Eigen::MatrixXd torque_profile(const Robot& robot, const Eigen::MatrixXd& theta, const Eigen::MatrixXd& dtheta,
                               const Eigen::MatrixXd& ddtheta, const std::optional<Vector6d>& Ftip = std::nullopt);

/// Peak absolute torque of each joint over a trajectory (max over rows of |tau|).
Eigen::VectorXd peak_torques(const Eigen::MatrixXd& tau);

}  // namespace lab

#endif  // SBEG461_LAB2_HPP
