// SBEG 46X Lab 6 -- Constant-curvature (CC) kinematics of a two-segment tendon-driven catheter: arc parameters,
// tendon-length and tendon-tension maps, the tip Jacobian and resolved-rate tip control (C++ track): declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp (the functions marked TODO A1 ... C4). Do not change these declarations:
// test_lab.cpp calls them. The provided functions (load_catheter, load_csv, vee, trans_z, split_q,
// catheter_backbone, gradient_rows) are in helpers.cpp (given code, do not edit).
// These functions become the `continuum` module of your cumulative library sbeg46x_medrob.
//
// Notation and units: SI (m, rad, N).  Twists V = (omega, v), angular first (MR Ch. 3).  T = e^[V] uses
// mr::matrix_exp6 (MR Eq. 3.88 when ||omega|| = 1; matrix_exp6 handles any V).
//
// Conventions used everywhere
//   * One CC segment of length ell bends in a plane through its base z-axis.  Arc parameters (kappa, phi, ell):
//     curvature kappa >= 0, bending-plane angle phi measured from the base x-axis, length ell.  Bending angle
//     theta = kappa * ell.
//   * Bending-angle vector theta_v = (theta_x, theta_y) = theta (cos phi, sin phi): it is zero for a straight
//     segment and has no singularity there (phi is undefined only for the arc parameters, not for theta_v).
//   * Twist-free CC map (Webster & Jones 2010):  R = Rot(z, phi) Rot(y, theta) Rot(z, -phi),
//     p = ((1 - cos theta)/kappa) (cos phi, sin phi, 0) + (sin theta / kappa) z_hat.
//     The same transform is the exponential of the constant body twist V = (-theta_y, theta_x, 0, 0, 0, ell):
//     T = e^[V]  (unit-speed travel along the backbone, angular rate kappa about the axis (-sin phi, cos phi, 0)).
//   * Tendon i of a segment runs parallel to the backbone at radius d and angle sigma_i in the (untwisted)
//     cross-section frame.  Its length in the segment is  l_i = ell - d (theta_v . s_i),  s_i = (cos sigma_i,
//     sin sigma_i).  A tendon that ends at segment 2 also runs through segment 1 at the same angle.
//   * Tendon tension t_i >= 0 (a tendon can only pull).  With bending energy U = sum_j EI_j |theta_v_j|^2 /
//     (2 ell_j) the equilibrium (virtual work, MLS Ch. 6 Sec. 4) is (EI_j / ell_j) theta_v_j = d sum t_i s_i
//     (sum over the tendons crossing segment j).
//   * Catheter joint vector q = (z_ins, theta_1x, theta_1y, theta_2x, theta_2y): insertion of the steerable
//     section out of the sheath along the sheath z-axis, then segment 1, then segment 2.
//     T_tip(q) = T_base Trans(0, 0, z_ins) e^[V1] e^[V2].
//
// Python -> C++: the catheter dict -> lab::Catheter; a tuple -> a small struct (fields named as in Python);
// a result dict -> a struct (fields named as the dict keys); an (N, k) array -> Eigen::MatrixXd with one ROW per
// sample; W=None -> W = Vector5d::Ones() (the default argument).
#ifndef SBEG46X_LAB6_HPP
#define SBEG46X_LAB6_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include <sbeg/labio.hpp>

#include <string>
#include <utility>
#include <vector>

namespace lab {

using Vector5d = Eigen::Matrix<double, 5, 1>;   ///< q = (z_ins, theta_1x, theta_1y, theta_2x, theta_2y)
using Vector6d = Eigen::Matrix<double, 6, 1>;   ///< twist (omega, v), or the 6 tendon lengths
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix23d = Eigen::Matrix<double, 2, 3>;  ///< one row per segment, one column per tendon
using Matrix65d = Eigen::Matrix<double, 6, 5>;  ///< space Jacobian, tendon Jacobian dl/dq
using Matrix35d = Eigen::Matrix<double, 3, 5>;  ///< tip-position Jacobian

/// Catheter parameters from data/catheter.json (illustrative values) -- the `cat` dict of the Python lab.
struct Catheter {
    std::string description;
    Eigen::Vector2d ell;        ///< segment lengths (m)
    Eigen::Vector2d d;          ///< tendon radius of each segment (m)
    Matrix23d sigma_deg;        ///< tendon angles (deg): row j = segment j, column i = tendon i
    Matrix23d sigma;            ///< the same in radians
    Eigen::Vector2d EI;         ///< bending stiffness of each segment (N m^2)
    double theta_max = 0.0;     ///< limit on |theta_v_j| (rad)
    Eigen::Vector2d z_range;    ///< insertion limits (m)
    Eigen::Matrix4d T_base;     ///< sheath exit frame in the atrium frame {s}
};

/// (kappa, phi, theta) returned by bend_to_arc.
struct ArcParams {
    double kappa;
    double phi;
    double theta;
};

/// (z, theta_v1, theta_v2) returned by split_q.
struct SplitQ {
    double z;
    Eigen::Vector2d theta_v1;
    Eigen::Vector2d theta_v2;
};

/// Result of go_to: q (steps+1) x 5 (one row per step), err (steps+1).
struct GoToResult {
    Eigen::MatrixXd q;
    Eigen::VectorXd err;
};

/// Result of track_path: q (N x 5), p (N x 3), err (N), tendon_dl (N x 6).
struct TrackResult {
    Eigen::MatrixXd q;
    Eigen::MatrixXd p;
    Eigen::VectorXd err;
    Eigen::MatrixXd tendon_dl;
};

// ---------------------------------------------------------------------------
// Provided (helpers.cpp, do not edit): data files and small helpers
// ---------------------------------------------------------------------------

/// Catheter parameters from data/catheter.json.
Catheter load_catheter();

/// data/<name> as a table: header (column names) and data (one row per line); column k of name h is
/// table.data.col(table.col("h")).
sbeg::CsvTable load_csv(const std::string& name);

/// 6-vector (omega, v) of a 4x4 matrix whose top-left block is (approximately) skew-symmetric.
Vector6d vee(const Eigen::Matrix4d& se3mat);

/// Trans(0, 0, z).
Eigen::Matrix4d trans_z(double z);

/// q -> (z, theta_v1, theta_v2).
SplitQ split_q(const Vector5d& q);

/// B1 (plot helper). (2n + 2) x 3 backbone points from the sheath exit to the tip (uses segment_transform_exp).
Eigen::MatrixXd catheter_backbone(const Vector5d& q, const Catheter& cat, int n = 20);

/// np.gradient(P, t, axis=0): derivative of every column of P (N x k, one row per time t_k) with respect to t,
/// second-order central differences inside (non-uniform spacing allowed), one-sided first-order at both ends.
Eigen::MatrixXd gradient_rows(const Eigen::MatrixXd& P, const Eigen::VectorXd& t);

// ---------------------------------------------------------------------------
// Part A -- one constant-curvature segment
// ---------------------------------------------------------------------------

/// A1. Robot-independent map: arc parameters -> T in SE(3) (closed form, twist-free).
/// kappa = 0 must return a straight segment (pure translation ell along z).
Eigen::Matrix4d arc_transform(double kappa, double phi, double ell);

/// A2. Bending-angle vector theta_v = (theta_x, theta_y) -> (kappa, phi, theta).  For theta_v = 0 return phi = 0.
ArcParams bend_to_arc(const Eigen::Vector2d& theta_v, double ell);

/// A2 (inverse). (kappa, phi, ell) -> theta_v.
Eigen::Vector2d arc_to_bend(double kappa, double phi, double ell);

/// A3. Constant body twist V of the segment such that T = e^[V] (MR exponential), V = (omega, v).
Vector6d segment_twist(const Eigen::Vector2d& theta_v, double ell);

/// A3. T of the segment from the MR matrix exponential of segment_twist.
Eigen::Matrix4d segment_transform_exp(const Eigen::Vector2d& theta_v, double ell);

/// A4. Robot-specific map: lengths of the tendons (angles sigma, radius d) inside one segment.
/// One entry per tendon (sigma has one entry per tendon).
Eigen::VectorXd tendon_lengths(const Eigen::Vector2d& theta_v, double ell, double d, const Eigen::VectorXd& sigma);

/// A5. Inverse robot-specific map: bending-angle vector theta_v from tendon length CHANGES dl_i = l_i - ell
/// (least squares; exact when the dl come from tendon_lengths).  For three tendons at 120 deg this is
/// theta_v = -(2 / (3 d)) sum_i dl_i s_i.
Eigen::Vector2d bend_from_tendons(const Eigen::VectorXd& dl, double d, const Eigen::VectorXd& sigma);

/// A6. Non-negative tendon tensions t (one per tendon) with d sum t_i s_i = m and min(t) = 0.
/// m is the 2-vector bending moment required in the cross-section.  Equally spaced tendons (n >= 3):
/// minimum-norm solution t_i = (2 / (n d)) s_i . m, then add the same constant to every tendon (it adds no
/// moment) so that the smallest tension is zero.
Eigen::VectorXd tension_allocation(const Eigen::Vector2d& m, double d, const Eigen::VectorXd& sigma);

/// A6. Tensions that hold one segment at bending-angle vector theta_v: (EI / ell) theta_v = d sum t_i s_i.
Eigen::VectorXd segment_tensions(const Eigen::Vector2d& theta_v, double ell, double d, const Eigen::VectorXd& sigma,
                                 double EI);

// ---------------------------------------------------------------------------
// Part B -- the two-segment catheter
// ---------------------------------------------------------------------------

/// B1. Tip pose T_tip = T_base Trans(0,0,z_ins) e^[V1] e^[V2].
Eigen::Matrix4d catheter_fk(const Vector5d& q, const Catheter& cat);

/// B2. Path lengths of the 6 tendons (3 ending at segment 1, then 3 ending at segment 2) measured from the
/// sheath exit; segment-2 tendons pass through segment 1 at their own angle.  Insertion does not change them
/// (the tendons slide with the catheter).
Vector6d catheter_tendon_lengths(const Vector5d& q, const Catheter& cat);

/// B2. 6 x 5 matrix dl/dq (constant: the tendon map is linear in theta_v).  Its transpose is the MLS coupling
/// matrix P (MLS Ch. 6 Sec. 4) restricted to the bending coordinates.
Matrix65d catheter_tendon_jacobian(const Catheter& cat);

/// B2 (inverse). (theta_v1, theta_v2) as a 4-vector from the 6 tendon length changes l - l(straight).
Eigen::Vector4d bend_from_catheter_tendons(const Vector6d& dl, const Catheter& cat);

/// B3. Non-negative tensions (6) that hold the catheter at q (no external load).  Segment 2 is held by
/// its own tendons; those tendons also bend segment 1, so the segment-1 tendons supply only the rest:
/// (EI2/ell2) theta_v2 = d2 sum t_(2,i) s_(2,i);
/// (EI1/ell1) theta_v1 = d1 sum t_(1,i) s_(1,i) + d2 sum t_(2,i) s_(2,i).
Vector6d catheter_tensions(const Vector5d& q, const Catheter& cat);

/// B4 helper. Left Jacobian of the SE(3) exponential, sum_k ad_V^k / (k+1)!  (6 x 6, ad = mr::ad).
/// (d/dt e^[V]) e^-[V] = [dexp_left(V) Vdot].
Matrix6d dexp_left(const Vector6d& V, int terms = 40);

/// B4. 6 x 5 space Jacobian J_s(q): V_s = J_s(q) qdot, where [V_s] = Tdot T^-1 (MR Sec. 5.1 definition).
Matrix65d catheter_jacobian_space(const Vector5d& q, const Catheter& cat);

/// B5. 3 x 5 Jacobian of the tip position p(q) in the space frame: pdot = [-[p]  I] V_s.
Matrix35d tip_position_jacobian(const Vector5d& q, const Catheter& cat);

// ---------------------------------------------------------------------------
// Part C -- resolved-rate tip control
// ---------------------------------------------------------------------------

/// C1. Enforce limits: z_ins in z_range, |theta_v_j| <= theta_max (scale theta_v_j radially).
Vector5d clamp_q(const Vector5d& q, const Catheter& cat);

/// C1. Damped least-squares inverse J^T (J J^T + lam^2 I)^-1 (lam = 0 gives the pseudoinverse).
/// J is m x n (any size); the result is n x m.
Eigen::MatrixXd dls_inverse(const Eigen::MatrixXd& J, double lam);

/// C2. One step of resolved-rate position control
/// qdot = W J^T (J W J^T + lam^2 I)^-1 (pdot_des + K (p_des - p)),   q <- clamp(q + qdot dt).
/// W (5) are joint weights (default: 1 for every coordinate).
Vector5d resolved_rate_step(const Vector5d& q, const Eigen::Vector3d& p_des, const Eigen::Vector3d& pdot_des,
                            const Catheter& cat, double dt, double K = 20.0, double lam = 1e-3,
                            const Vector5d& W = Vector5d::Ones());

/// C3. Regulate the tip to a fixed point (pdot_des = 0).  Returns q ((steps+1) x 5) and err (steps+1).
GoToResult go_to(const Vector5d& q0, const Eigen::Vector3d& p_goal, const Catheter& cat, double dt = 0.01,
                 double K = 5.0, int steps = 400, double lam = 1e-3, const Vector5d& W = Vector5d::Ones());

/// C4. Track a sampled tip path P (N x 3) at times t (N) with feed-forward pdot from finite differences
/// (gradient_rows).  Returns q (N x 5), p (N x 3), err (N), tendon_dl (N x 6) where tendon_dl = l(q) - l(straight).
TrackResult track_path(const Vector5d& q0, const Eigen::VectorXd& t, const Eigen::MatrixXd& P, const Catheter& cat,
                       double K = 20.0, double lam = 1e-3, const Vector5d& W = Vector5d::Ones());

}  // namespace lab

#endif  // SBEG46X_LAB6_HPP
