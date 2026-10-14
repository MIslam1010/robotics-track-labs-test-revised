// SBEG 361 Lab 2 -- Angular velocities and twists (C++ track): declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp. Do not change these declarations: test_lab.cpp calls them.
//
// Notation follows Lynch & Park, *Modern Robotics* (MR), Sections 3.2.2 and 3.3.2:
//   [omega_s] = Rdot R^-1, [omega_b] = R^-1 Rdot                          (MR Eqs. 3.37-3.38, Prop. 3.9)
//   omega_s = R omega_b,  R [omega] R^T = [R omega]                       (MR Eq. 3.35, Prop. 3.8)
//   V = (omega, v)               twist, angular part FIRST                 (MR Eqs. 3.70, 3.74)
//   [V] = [[omega] v; 0 0]       4x4 se(3) matrix                          (MR Eq. 3.71)
//   [V_b] = T^-1 Tdot, [V_s] = Tdot T^-1                                  (MR Eqs. 3.81-3.82)
//   [Ad_T] = [[R 0]; [[p]R R]],  [Ad_T V] = T [V] T^-1                    (MR Def. 3.20, Eqs. 3.83-3.84)
//   velocity of a body point x (coordinates in the same frame as V): omega x x + v   (MR Eq. 3.73, Sec. 3.3.2)
//   screw {q, s_hat, h}: V = (s_hat, -s_hat x q + h s_hat) thetadot       (MR Sec. 3.3.2.2)
//
// The SE(3) basics and the tf2-style transform tree of Lab 1 are GIVEN (marked "Given."), so this lab is
// self-contained. Transform tree convention (identical to ROS 2 tf2): an edge parent -> child stores
// T_parent_child; lookup(target, source) returns T_target_source (tf2: buffer.lookup_transform(target, source, t)).
//
// Python -> C++: ValueError -> std::invalid_argument; KeyError / LookupError -> lab::KeyError / lab::LookupError
// (below; as in Python, KeyError is a kind of LookupError); a tuple -> std::pair; a dict -> a struct;
// None -> an empty std::optional; np.inf -> std::numeric_limits<double>::infinity() (lab::INF);
// an (N, 4, 4) array -> std::vector<Eigen::Matrix4d>; an (N, 6) or (N, 3) array -> Eigen::MatrixXd (row = sample).
#ifndef SBEG361_LAB2_HPP
#define SBEG361_LAB2_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lab {

using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

constexpr double TOL = 1e-6;  // same threshold as modern_robotics.NearZero
constexpr double INF = std::numeric_limits<double>::infinity();

/// Python LookupError: two frames that are not connected.
struct LookupError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
/// Python KeyError: an unknown frame name (a KeyError is also a LookupError, as in Python).
struct KeyError : LookupError {
    using LookupError::LookupError;
};

// ================================================================ GIVEN (from Lab 1)

/// 3x3 skew-symmetric matrix [omega] (MR Eq. 3.30).  (Given.)
Eigen::Matrix3d vec_to_so3(const Eigen::Vector3d& omega);

/// omega from [omega].  (Given.)
Eigen::Vector3d so3_to_vec(const Eigen::Matrix3d& so3mat);

/// Build T = [[R, p], [0, 1]] (MR Eq. 3.62).  (Given.)
Eigen::Matrix4d rp_to_trans(const Eigen::Matrix3d& R, const Eigen::Vector3d& p);

/// Split T into (R, p).  (Given.)
std::pair<Eigen::Matrix3d, Eigen::Vector3d> trans_to_rp(const Eigen::Matrix4d& T);

/// Inverse of T in SE(3) without a general matrix inverse (MR Eq. 3.64).  (Given.)
Eigen::Matrix4d trans_inv(const Eigen::Matrix4d& T);

/// True if T is a 4x4 element of SE(3): R^T R = I, det R = +1, last row (0, 0, 0, 1) (all within tol).  (Given.)
bool is_se3(const Eigen::MatrixXd& T, double tol = 1e-6);

/// Return T x = R x + p for one point (MR Eq. 3.65).  (Given.)
Eigen::Vector3d transform_point(const Eigen::Matrix4d& T, const Eigen::Vector3d& x);

/// The same for many points: X is N x 3 (one point per ROW).  (Given; Python: transform_point with x of shape (N, 3).)
Eigen::MatrixXd transform_points(const Eigen::Matrix4d& T, const Eigen::MatrixXd& X);

/// Rotation matrix from a tf2/ROS quaternion (x, y, z, w) -- SCALAR LAST.  Quaternions are taught in Week 4
/// (MR App. B.3, scalar FIRST there); this converter is given so that the tf2-style JSON files can be read.
/// The quaternion is normalised first (MR Eq. B.12).  (Given.)
Eigen::Matrix3d quat_xyzw_to_rot(const Eigen::Vector4d& q_xyzw);

/// A tree of named frames; each edge parent -> child stores T_parent_child (Lab 1).  (Given.)
/// Like tf2, every frame has at most one parent, so there is exactly one path between any two connected frames.
/// lookup(target, source) = (T_common_target)^-1 T_common_source (via the lowest common ancestor).
class TransformTree {
public:
    std::map<std::string, std::string> parent;  ///< child -> parent
    std::map<std::string, Eigen::Matrix4d> T;   ///< child -> T_parent_child

    /// All frame names (children and parents).
    std::set<std::string> frames() const;
    /// Add (or update, if the same parent->child edge exists) a transform; std::invalid_argument if invalid
    /// (second parent, parent == child, not in SE(3), cycle).
    void add(const std::string& parent_frame, const std::string& child, const Eigen::Matrix4d& T_parent_child);
    /// [frame, parent, grandparent, ..., root].
    std::vector<std::string> _ancestors(const std::string& frame) const;
    /// T_ancestor_frame, composed down the chain ancestor -> ... -> frame.
    Eigen::Matrix4d _to_ancestor(const std::string& frame, const std::string& ancestor) const;
    /// Lowest common ancestor of two frames (throws lab::KeyError / lab::LookupError if impossible).
    std::string _common(const std::string& target, const std::string& source) const;
    /// List of frames from target to source through their lowest common ancestor.
    std::vector<std::string> path(const std::string& target, const std::string& source) const;
    /// T_target_source: pose of `source` expressed in `target` (tf2 lookup_transform order).
    Eigen::Matrix4d lookup(const std::string& target, const std::string& source) const;
};

/// T from a tf2-style (translation, quaternion x,y,z,w) pair.  (Given.)
Eigen::Matrix4d transform_from_tf(const Eigen::Vector3d& translation, const Eigen::Vector4d& rotation_xyzw);

/// Build a TransformTree from a JSON file holding a list of tf2-style static transforms:
///   {"transforms": [{"parent": .., "child": .., "translation": [x,y,z], "rotation_xyzw": [qx,qy,qz,qw]}, ...]}
/// (Given.)
TransformTree load_tree_json(const std::string& path);

/// Read a tracked pose sequence. Columns: t, r11..r33 (row-major), px, py, pz.
/// Returns (t, Ts): the N times and the N poses.  (Given.)
std::pair<Eigen::VectorXd, std::vector<Eigen::Matrix4d>> load_track_csv(const std::string& path);

/// Angle (rad) between two direction vectors.  (Given.)
double axis_angle_between(const Eigen::Vector3d& a, const Eigen::Vector3d& b);

/// Distance from point x to the line through q with direction s_hat.  (Given.)
double point_line_distance(const Eigen::Vector3d& x, const Eigen::Vector3d& q, const Eigen::Vector3d& s_hat);

// ================================================================ Step 1: angular velocity

/// Return (omega_s, omega_b) of a rotating frame R = R_sb(t) with time derivative Rdot.
/// [omega_s] = Rdot R^-1 = Rdot R^T (space frame), [omega_b] = R^-1 Rdot = R^T Rdot (body frame)
/// (MR Eqs. 3.37-3.38).  R^-1 = R^T for a rotation matrix -- no general inverse is needed.
std::pair<Eigen::Vector3d, Eigen::Vector3d> angular_velocities(const Eigen::Matrix3d& R, const Eigen::Matrix3d& Rdot);

// ================================================================ Step 2: se(3) and the adjoint

/// 4x4 matrix [V] of the twist V = (omega, v) (MR Eq. 3.71).
Eigen::Matrix4d vec_to_se3(const Vector6d& V);

/// Twist V = (omega, v) from its 4x4 matrix [V].
Vector6d se3_to_vec(const Eigen::Matrix4d& se3mat);

/// 6x6 adjoint [Ad_T] = [[R, 0], [[p]R, R]] (MR Def. 3.20).
Matrix6d adjoint(const Eigen::Matrix4d& T);

// ================================================================ Step 3: twists and point velocities

/// Return (V_s, V_b) from a configuration T_sb and its time derivative (MR Eqs. 3.81-3.82).
std::pair<Vector6d, Vector6d> twists_from_T_Tdot(const Eigen::Matrix4d& T, const Eigen::Matrix4d& Tdot);

/// Velocity omega x x + v of the body point whose coordinates are x, for a twist V = (omega, v).
/// V and x must be expressed in the SAME frame {a}; the result is the velocity of that point in {a} coordinates.
/// With {a} = {s}: xdot_s = omega_s x x_s + v_s (from [V_s] = Tdot T^-1, MR Eqs. 3.72-3.74);
/// with {a} = {b}: R^T xdot_s = omega_b x x_b + v_b.
Eigen::Vector3d point_velocity(const Vector6d& V, const Eigen::Vector3d& x);

/// The same for many points: X is N x 3 (one point per ROW); return the N x 3 matrix of velocities.
/// (Python: point_velocity(V, x) with x of shape (N, 3).)
Eigen::MatrixXd point_velocities(const Vector6d& V, const Eigen::MatrixXd& X);

/// Estimate (V_s, V_b) at every interior sample of a pose sequence by central differences.
///   Ts: N poses T_sb(t_k) sampled every dt seconds.
///   Returns two (N-2) x 6 matrices: ROW k is the twist at sample k+1.
///   Tdot_k ~ (T_{k+1} - T_{k-1}) / (2 dt).  The numerical T^-1 Tdot is only approximately in se(3), so the
///   angular part is read from its skew-symmetric part (M - M^T)/2.
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> twists_from_samples(const std::vector<Eigen::Matrix4d>& Ts, double dt);

// ================================================================ Step 4: screws

/// Twist V = (s_hat thetadot, (-s_hat x q + h s_hat) thetadot) of a screw {q, s_hat, h} (MR Sec. 3.3.2.2).
/// Use h = lab::INF for a pure translation along s_hat (then q is ignored: pass any vector, e.g. Zero()).
Vector6d screw_to_twist(const Eigen::Vector3d& q, const Eigen::Vector3d& s_hat, double h, double thetadot = 1.0);

/// Result of twist_to_screw (the Python dict, same field names).
struct Screw {
    Vector6d S;                        ///< normalised screw axis (MR Def. 3.24)
    double thetadot = 0.0;             ///< speed about/along the axis (>= 0)
    double h = 0.0;                    ///< pitch (lab::INF for a pure translation)
    Eigen::Vector3d s_hat;             ///< unit axis direction
    std::optional<Eigen::Vector3d> q;  ///< point on the axis closest to the origin; empty (None) for a pure translation
};

/// Screw interpretation of a twist (MR Sec. 3.3.2.2, Def. 3.24).
///   q = omega x v / |omega|^2 is the point on the axis closest to the frame origin.
/// Throw std::invalid_argument for the zero twist.
Screw twist_to_screw(const Vector6d& V);

}  // namespace lab

#endif  // SBEG361_LAB2_HPP
