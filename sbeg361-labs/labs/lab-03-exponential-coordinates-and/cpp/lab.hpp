// SBEG 361 Lab 3 -- Exponential coordinates and logarithms on SO(3) and SE(3); IMU orientation (C++ track):
// declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp. Do not change these declarations: test_lab.cpp calls them.
//
// Notation follows Lynch & Park, *Modern Robotics* (MR), Sections 3.2.3, 3.3.2.2-3.3.3 and Appendix B.3.
// SO(3) and SE(3) side by side:
//   [w]          3x3 skew-symmetric matrix of w in R^3                         (MR Eq. 3.30)
//   [S]          4x4 matrix [[ [w], v ], [0, 0]] of S = (w, v), angular first  (MR Eq. 3.85)
//   e^[w]theta   Rodrigues: I + sin(t)[w] + (1 - cos t)[w]^2                    (MR Eq. 3.51)
//   e^[S]theta   [[e^[w]theta, G(theta) v], [0, 1]]                             (MR Prop. 3.25, Eq. 3.88)
//   G(theta)     I theta + (1 - cos theta)[w] + (theta - sin theta)[w]^2          (MR Eq. 3.87)
//   e^[S]theta   [[I, v theta], [0, 1]]  if w = 0, |v| = 1                       (MR Eq. 3.89)
//   log R        three-case algorithm                                           (MR Sec. 3.2.3.3, Eqs. 3.58-3.61)
//   log T        two-case algorithm, v = G^-1(theta) p                          (MR Sec. 3.3.3.2, Eqs. 3.91-3.92)
//   G^-1(theta)  I/theta - [w]/2 + (1/theta - cot(theta/2)/2)[w]^2               (MR Eq. 3.92)
//   q            unit quaternion (q0, q1, q2, q3), scalar first                 (MR Eqs. B.9-B.12, B.15)
//                stored as an Eigen::Vector4d in this order (NOT Eigen::Quaterniond, whose storage order differs).
//
// Python -> C++: ValueError -> std::invalid_argument; a tuple -> std::pair; a dict -> a small struct (fields named
// as the Python keys); a 6-vector -> sbeg::Vector6d; an (N, 3) array of gyro samples -> Eigen::MatrixXd with N rows
// and 3 columns (row k = sample k); an (N+1, 3, 3) array -> std::vector<Eigen::Matrix3d>; an (N, 4, 4) array of
// poses -> std::vector<Eigen::Matrix4d>.
#ifndef SBEG361_LAB3_HPP
#define SBEG361_LAB3_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include <string>
#include <utility>
#include <vector>

namespace lab {

using sbeg::Vector6d;

/// Same threshold as modern_robotics NearZero.
constexpr double TOL = 1e-6;

// ---------------------------------------------------------------------------
// given helpers (Labs 1-2): complete in starter.cpp, you may use them
// ---------------------------------------------------------------------------

/// Build T = [[R, p], [0, 1]] (MR Eq. 3.62).
Eigen::Matrix4d rp_to_trans(const Eigen::Matrix3d& R, const Eigen::Vector3d& p);

/// Split T into (R, p).
std::pair<Eigen::Matrix3d, Eigen::Vector3d> trans_to_rp(const Eigen::Matrix4d& T);

/// Inverse of T in SE(3) without a general matrix inverse (MR Eq. 3.64).
Eigen::Matrix4d trans_inv(const Eigen::Matrix4d& T);

/// [V] = [[ [omega], v ], [0, 0]] of a 6-vector V = (omega, v) (MR Eq. 3.85).
Eigen::Matrix4d vec_to_se3(const Vector6d& V);

/// 6-vector (omega, v) of a 4x4 matrix [V] in se(3).
Vector6d se3_to_vec(const Eigen::Matrix4d& se3mat);

/// Result of twist_to_screw (the keys of the Python dict).
struct ScrewParams {
    Vector6d S;             ///< normalised screw axis (MR Def. 3.24)
    double thetadot;        ///< speed (or distance) about/along the axis (>= 0)
    double h;               ///< pitch (infinity for a pure translation)
    Eigen::Vector3d s_hat;  ///< unit axis direction
    Eigen::Vector3d q;      ///< point on the axis closest to the frame origin (zero if has_q is false)
    bool has_q;             ///< false for a pure translation (Python: q is None)
};

/// Screw interpretation of a twist or of exponential coordinates S theta (MR Sec. 3.3.2.2, Def. 3.24):
/// q = omega x v / |omega|^2, h = omega.v / |omega|^2. Throws std::invalid_argument for the zero twist.
ScrewParams twist_to_screw(const Vector6d& V);

/// A tracker log: times and poses.
struct PoseTrack {
    Eigen::VectorXd t;               ///< times (s)
    std::vector<Eigen::Matrix4d> Ts; ///< poses
};

/// Load a tracker log (columns t, q0..q3, px, py, pz and the noisy nq0..nq3, npx..npz).
/// Uses YOUR quat_to_rot (Step 5).
PoseTrack load_pose_track(const std::string& path, bool noisy = false);

// ---------------------------------------------------------------------------
// so(3) <-> R^3
// ---------------------------------------------------------------------------

/// Return the 3x3 skew-symmetric matrix [omega] (MR Eq. 3.30).
/// Example: vec_to_so3((1, 2, 3)) -> [[0, -3, 2], [3, 0, -1], [-2, 1, 0]]
Eigen::Matrix3d vec_to_so3(const Eigen::Vector3d& omega);

/// Return the 3-vector omega from a skew-symmetric matrix [omega].
Eigen::Vector3d so3_to_vec(const Eigen::Matrix3d& so3mat);

/// Split exponential coordinates w_hat*theta into (w_hat, theta), theta >= 0.
/// Throw std::invalid_argument if theta < TOL (axis undefined).
std::pair<Eigen::Vector3d, double> axis_ang3(const Eigen::Vector3d& expc3);

// ---------------------------------------------------------------------------
// exponentials: SO(3) and SE(3)
// ---------------------------------------------------------------------------

/// Matrix exponential of [w_hat]theta in so(3) by Rodrigues' formula (MR Eq. 3.51).
Eigen::Matrix3d matrix_exp3(const Eigen::Matrix3d& so3mat);

/// G(theta) = I theta + (1 - cos theta)[w] + (theta - sin theta)[w]^2 for a unit axis w (MR Eq. 3.87).
Eigen::Matrix3d g_matrix(const Eigen::Vector3d& omega_hat, double theta);

/// G^-1(theta) = I/theta - [w]/2 + (1/theta - cot(theta/2)/2)[w]^2, 0 < theta <= pi (MR Eq. 3.92).
Eigen::Matrix3d g_inv_matrix(const Eigen::Vector3d& omega_hat, double theta);

/// T = e^{[S]theta} for the 4x4 matrix [S]theta (MR Prop. 3.25, Eqs. 3.88-3.89); any element of se(3):
/// theta = |omega theta| if omega != 0, otherwise the pure translation case (Eq. 3.89).
Eigen::Matrix4d matrix_exp6(const Eigen::Matrix4d& se3mat);

// ---------------------------------------------------------------------------
// logarithms: SO(3) and SE(3)
// ---------------------------------------------------------------------------

/// Matrix logarithm [w_hat]theta of R in SO(3), theta in [0, pi] (MR Sec. 3.2.3.3), all three cases:
///   (a) R = I               -> zero matrix
///   (b) tr R = -1 (theta=pi) -> Eqs. 3.58 / 3.59 / 3.60 (the first feasible one)
///   (c) otherwise           -> theta = arccos((tr R - 1)/2), [w_hat] = (R - R^T)/(2 sin theta)
Eigen::Matrix3d matrix_log3(const Eigen::Matrix3d& R);

/// [S]theta = log(T) (MR Sec. 3.3.3.2 algorithm, Eqs. 3.91-3.92), theta in [0, pi].
Eigen::Matrix4d matrix_log6(const Eigen::Matrix4d& T);

// ---------------------------------------------------------------------------
// screw motions
// ---------------------------------------------------------------------------

/// Unit screw axis S and distance theta that carry T1 to T2.
///   frame="space":  T2 = e^{[S]theta} T1   (S in {s}; log of T2 T1^-1)
///   frame="body":   T2 = T1 e^{[S]theta}   (S in the moving frame at T1; log of T1^-1 T2)
/// S is normalised as in MR Def. 3.24. For T1 = T2 returns (zero, 0.0). Throws std::invalid_argument for an
/// unknown frame.
std::pair<Vector6d, double> screw_between(const Eigen::Matrix4d& T1, const Eigen::Matrix4d& T2,
                                          const std::string& frame = "space");

/// Pose a fraction s in [0, 1] of the way along the constant screw motion from T1 to T2:
/// T(s) = e^{[S] s theta} T1 (MR Sec. 3.3.3).
Eigen::Matrix4d screw_path(const Eigen::Matrix4d& T1, const Eigen::Matrix4d& T2, double s);

/// Cumulative SIGNED rotation angle theta and advance d along the screw axis of a tracked screw motion.
/// For every step k = 1..N-1: (S_k, dtheta_k) = screw_between(Ts[k-1], Ts[k], "space"). The reference direction
/// w_ref is the omega of the step with the LARGEST dtheta; a step whose omega_k . w_ref < 0 counts negative.
/// Advance of step k = (omega_k . v_k) * signed dtheta_k. Steps with dtheta < TOL add nothing. Throws
/// std::invalid_argument if a step translates without rotating. Returns (theta, d), both of size N, from 0.
std::pair<Eigen::VectorXd, Eigen::VectorXd> screw_motion_profile(const std::vector<Eigen::Matrix4d>& Ts);

// ---------------------------------------------------------------------------
// quaternions (MR App. B.3); q = (q0, q1, q2, q3), scalar first
// ---------------------------------------------------------------------------

/// Unit quaternion q of R, with q0 >= 0 (MR Eqs. B.10-B.11; printed B.11 has a typo: last entry is r21 - r12).
/// For q0 <= 0.5 (theta >= 120 deg) build q from the matrix log instead (Eq. B.9).
Eigen::Vector4d rot_to_quat(const Eigen::Matrix3d& R);

/// Rotation matrix of a unit quaternion (MR Eq. B.12). Normalise q first.
Eigen::Matrix3d quat_to_rot(const Eigen::Vector4d& q);

/// Quaternion of the product R_q R_p (MR Eq. B.15).
Eigen::Vector4d quat_mul(const Eigen::Vector4d& q, const Eigen::Vector4d& p);

/// Conjugate (= inverse for a unit quaternion): the rotation R_q^T.
Eigen::Vector4d quat_conj(const Eigen::Vector4d& q);

// ---------------------------------------------------------------------------
// IMU orientation
// ---------------------------------------------------------------------------

/// Integrate body-frame gyro samples with the exponential update R_{k+1} = R_k exp([omega_b,k] dt).
/// omega_b has N rows and 3 columns (row k = sample k). Return N+1 matrices; the first one is R0.
std::vector<Eigen::Matrix3d> integrate_gyro_exp(const Eigen::Matrix3d& R0, const Eigen::MatrixXd& omega_b,
                                                double dt);

/// First-order (forward Euler) update R_{k+1} = R_k (I + [omega_b,k] dt) -- for comparison.
/// Same input/output shapes as integrate_gyro_exp.
std::vector<Eigen::Matrix3d> integrate_gyro_euler(const Eigen::Matrix3d& R0, const Eigen::MatrixXd& omega_b,
                                                  double dt);

/// Exponential coordinates w_hat*theta (in the current body frame) of the rotation that takes
/// R_cur to R_des, i.e. R_des = R_cur exp([w_hat] theta).
Eigen::Vector3d orientation_error(const Eigen::Matrix3d& R_cur, const Eigen::Matrix3d& R_des);

/// Angle (rad, in [0, pi]) of the relative rotation R1^T R2.
double rotation_angle(const Eigen::Matrix3d& R1, const Eigen::Matrix3d& R2);

/// Signed knee flexion angle (rad): y-component of log(R_thigh^T R_shank).
/// Convention for this lab: both IMUs have their y-axis along the knee flexion axis,
/// so R_shank = R_thigh Rot(y_hat, kappa).
double knee_flexion(const Eigen::Matrix3d& R_thigh, const Eigen::Matrix3d& R_shank);

}  // namespace lab

#endif  // SBEG361_LAB3_HPP
