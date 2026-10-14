// SBEG 361 Lab 1 -- Rotations and transformations: SO(3)/SE(3) property tests and transform trees
// (C++ track): declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp. Do not change these declarations: test_lab.cpp calls them.
//
// Notation follows Lynch & Park, *Modern Robotics* (MR), Sections 3.1-3.3.1, rotations and rigid motions side by side:
//   * R_sb in SO(3): orientation of {b} in {s} (columns = x_b, y_b, z_b written in {s})     (MR Def. 3.1)
//   * T_sb = (R_sb, p_sb) in SE(3): configuration of {b} in {s}                              (MR Def. 3.13, Eq. 3.62)
//   * Rot(w_hat, theta): rotation operator about the unit axis w_hat by theta                (MR p. 72)
//   * inverses: R^-1 = R^T (MR Prop. 3.3);  T^-1 = (R^T, -R^T p)                             (MR Eq. 3.64)
//   * fixed vs body frame: R R_sb / R_sb R (MR Eqs. 3.23-3.24);  T T_sb / T_sb T              (MR Eqs. 3.66-3.67)
//   * planar frames (P, p), P in SO(2), and their 3x3 SE(2) matrices                         (MR Eqs. 3.8-3.9, 3.63)
//
// Transform tree convention (identical to ROS 2 tf2): an edge parent -> child stores T_parent_child, the pose of
// the child frame expressed in the parent frame. lookup(target, source) returns T_target_source
// (tf2: buffer.lookup_transform(target, source, t)).
//
// Python -> C++: ValueError -> std::invalid_argument; KeyError / LookupError -> lab::KeyError / lab::LookupError
// (below; as in Python, KeyError is a kind of LookupError); a tuple -> std::pair; an (N, 3, 3) or (N, 4, 4) array ->
// std::vector<Eigen::Matrix3d> / std::vector<Eigen::Matrix4d>; a boolean array -> std::vector<bool>;
// an (N, 3) array of points -> Eigen::MatrixXd (one point per row).
#ifndef SBEG361_LAB1_HPP
#define SBEG361_LAB1_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lab {

/// Python LookupError: two frames that are not connected.
struct LookupError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
/// Python KeyError: an unknown frame name (a KeyError is also a LookupError, as in Python).
struct KeyError : LookupError {
    using LookupError::LookupError;
};

// ---------------------------------------------------------------------------
// Part A -- building R and T
// ---------------------------------------------------------------------------

/// Rot(x_hat, theta) as a 3x3 matrix (theta in radians).  (Given as an example.)
Eigen::Matrix3d rot_x(double theta);

/// Rot(y_hat, theta) as a 3x3 matrix (theta in radians).  See MR p. 72.
Eigen::Matrix3d rot_y(double theta);

/// Rot(z_hat, theta) as a 3x3 matrix (theta in radians).  See MR p. 72.
Eigen::Matrix3d rot_z(double theta);

/// Rot(omega_hat, theta) from the explicit formula on MR p. 72.
/// omega_hat must be a unit 3-vector: throw std::invalid_argument if | ||omega_hat|| - 1 | > 1e-6.
Eigen::Matrix3d rot_axis_angle(const Eigen::Vector3d& omega_hat, double theta);

/// Build T = [[R, p], [0, 1]] (MR Eq. 3.62).
Eigen::Matrix4d rp_to_trans(const Eigen::Matrix3d& R, const Eigen::Vector3d& p);

/// Split T into (R, p).
std::pair<Eigen::Matrix3d, Eigen::Vector3d> trans_to_rp(const Eigen::Matrix4d& T);

// ---------------------------------------------------------------------------
// Part B -- group property tests: SO(3) and SE(3) side by side
// ---------------------------------------------------------------------------

/// Return (orth_err, det) for a 3x3 matrix (throw std::invalid_argument if R is not 3x3).
///   orth_err = Frobenius norm of R^T R - I   (MR Eq. 3.19)
///   det      = det(R)                         (MR Eq. 3.21 requires +1)
std::pair<double, double> so3_errors(const Eigen::MatrixXd& R);

/// True if R is (numerically) in SO(3): size 3x3, ||R^T R - I||_F < tol and |det R - 1| < tol.
/// Return false (do not throw) for a wrong size.
bool is_rotation(const Eigen::MatrixXd& R, double tol = 1e-3);

/// Inverse of a rotation matrix (MR Proposition 3.3).  Do NOT call .inverse().
Eigen::Matrix3d rot_inv(const Eigen::Matrix3d& R);

/// Closest rotation matrix to M in the Frobenius norm (SVD method).
/// M = U S V^T  ->  R = U diag(1, 1, d) V^T with d = det(U V^T), so that det R = +1 always.
Eigen::Matrix3d project_to_so3(const Eigen::Matrix3d& M);

/// True if T is a 4x4 element of SE(3): R^T R = I, det R = +1, last row (0, 0, 0, 1) (all within tol,
/// element by element as np.allclose).  T may have any size: return false (do not throw) if it is not 4x4.
bool is_se3(const Eigen::MatrixXd& T, double tol = 1e-6);

/// Inverse of T in SE(3) without a general matrix inverse (MR Eq. 3.64).  Do NOT call .inverse().
Eigen::Matrix4d trans_inv(const Eigen::Matrix4d& T);

// ---------------------------------------------------------------------------
// Part C -- the three uses of R and T (MR Sections 3.2.1, 3.3.1)
// ---------------------------------------------------------------------------

/// Change the reference frame of a vector: return p_a (subscript cancellation rule).
Eigen::Vector3d change_frame(const Eigen::Matrix3d& R_ab, const Eigen::Vector3d& p_b);

/// Return T x = R x + p for one point (MR Eq. 3.65).
Eigen::Vector3d transform_point(const Eigen::Matrix4d& T, const Eigen::Vector3d& x);

/// The same for many points: X is N x 3 (one point per ROW); return the N x 3 matrix of transformed points.
/// (Python: transform_point(T, x) with x of shape (N, 3).)
Eigen::MatrixXd transform_points(const Eigen::Matrix4d& T, const Eigen::MatrixXd& X);

/// Rotate {b} by R about an axis expressed in the fixed frame {s} (MR Eq. 3.23).
Eigen::Matrix3d rotate_in_fixed(const Eigen::Matrix3d& R_sb, const Eigen::Matrix3d& R);

/// Rotate {b} by R about an axis expressed in the body frame {b} (MR Eq. 3.24).
Eigen::Matrix3d rotate_in_body(const Eigen::Matrix3d& R_sb, const Eigen::Matrix3d& R);

/// Displace {b} by T = (R, p) expressed in the fixed frame {s} (MR Eq. 3.66).
Eigen::Matrix4d transform_in_fixed(const Eigen::Matrix4d& T_sb, const Eigen::Matrix4d& T);

/// Displace {b} by T = (R, p) expressed in the body frame {b} (MR Eq. 3.67).
Eigen::Matrix4d transform_in_body(const Eigen::Matrix4d& T_sb, const Eigen::Matrix4d& T);

/// 3x3 SE(2) matrix [[P, p], [0, 0, 1]] of a planar frame (P, p) (MR Def. 3.14, Eq. 3.63).
Eigen::Matrix3d planar_to_se2(const Eigen::Matrix2d& P, const Eigen::Vector2d& p);

/// Compose planar frames: given {b} in {s} as (P, p) and {c} in {b} as (Q, q),
/// return {c} in {s} as the pair (R, r)  (MR Eqs. 3.8-3.9).
std::pair<Eigen::Matrix2d, Eigen::Vector2d> planar_compose(const Eigen::Matrix2d& P, const Eigen::Vector2d& p,
                                                           const Eigen::Matrix2d& Q, const Eigen::Vector2d& q);

// ---------------------------------------------------------------------------
// Part D -- the transform tree (tf2-like)
// ---------------------------------------------------------------------------

/// Rotation matrix from a tf2/ROS quaternion (x, y, z, w) -- SCALAR LAST.  (Given.)
/// MR App. B.3 writes q = (q0, q1, q2, q3) with the scalar first; tf2 and SciPy store
/// (x, y, z, w) = (q1, q2, q3, q0).  The quaternion is normalised first, then MR Eq. B.12 is applied.
/// Unit quaternions are taught in Week 4 (Lab 3); here the function only reads the JSON files.
Eigen::Matrix3d quat_xyzw_to_rot(const Eigen::Vector4d& q_xyzw);

/// A tree of named frames; each edge parent -> child stores T_parent_child.
/// Like tf2, every frame has at most one parent, so there is exactly one path between any two connected frames.
/// lookup(target, source) composes transforms along that path:
///     T_target_source = (T_common_target)^-1 T_common_source  (computed via the lowest common ancestor).
class TransformTree {
public:
    std::map<std::string, std::string> parent;  ///< child -> parent
    std::map<std::string, Eigen::Matrix4d> T;   ///< child -> T_parent_child

    /// All frame names (children and parents).  (Given.)
    std::set<std::string> frames() const;

    /// Add (or update, if the same parent->child edge exists) a transform.
    /// Throw std::invalid_argument if `child` already has a different parent, if parent == child, if T is not in
    /// SE(3), or if the edge would create a cycle.
    void add(const std::string& parent_frame, const std::string& child, const Eigen::Matrix4d& T_parent_child);

    /// [frame, parent, grandparent, ..., root].  (Given.)
    std::vector<std::string> _ancestors(const std::string& frame) const;

    /// T_ancestor_frame, composed down the chain ancestor -> ... -> frame.
    Eigen::Matrix4d _to_ancestor(const std::string& frame, const std::string& ancestor) const;

    /// Lowest common ancestor of two frames (throws lab::KeyError / lab::LookupError if impossible).  (Given.)
    std::string _common(const std::string& target, const std::string& source) const;

    /// List of frames from target to source through their lowest common ancestor.  (Given.)
    std::vector<std::string> path(const std::string& target, const std::string& source) const;

    /// T_target_source: pose of `source` expressed in `target` (tf2 lookup_transform order).
    Eigen::Matrix4d lookup(const std::string& target, const std::string& source) const;
};

/// T from a tf2-style (translation, quaternion x,y,z,w) pair.  (Given; uses your rp_to_trans.)
Eigen::Matrix4d transform_from_tf(const Eigen::Vector3d& translation, const Eigen::Vector4d& rotation_xyzw);

/// Build a TransformTree from a JSON file holding a list of tf2-style static transforms:
///   {"transforms": [{"parent": .., "child": .., "translation": [x,y,z], "rotation_xyzw": [qx,qy,qz,qw]}, ...]}
/// (Given; uses your TransformTree::add.)
TransformTree load_tree_json(const std::string& path);

// ---------------------------------------------------------------------------
// Part E -- applications: tracker data cleaning and surgical navigation
// ---------------------------------------------------------------------------

/// Read the tracker file (given). Returns one 3x3 matrix per sample; each row of the file holds
/// sample_id, r11, r12, r13, r21, ..., r33 (row-major).
std::vector<Eigen::Matrix3d> load_tracker_csv(const std::string& path);

/// One flag per sample: true where sample k is a valid rotation matrix (use is_rotation).
std::vector<bool> classify_samples(const std::vector<Eigen::Matrix3d>& Rs, double tol = 1e-3);

/// Orientation of the tool-tip frame {t} in the tracker frame {s} for every sample.
/// Return one matrix per sample.  Which matrix goes first?  Use subscript cancellation.
std::vector<Eigen::Matrix3d> tool_orientations(const std::vector<Eigen::Matrix3d>& R_sm_list,
                                               const Eigen::Matrix3d& R_mt);

/// Read a tracked pose sequence (given). Columns: t, r11..r33 (row-major), px, py, pz.
/// Returns (t, Ts): the N times and the N poses.
std::pair<Eigen::VectorXd, std::vector<Eigen::Matrix4d>> load_track_csv(const std::string& path);

}  // namespace lab

#endif  // SBEG361_LAB1_HPP
