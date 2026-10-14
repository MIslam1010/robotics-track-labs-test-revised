// SBEG 361 Lab 6 -- Denavit-Hartenberg models vs the product of exponentials (C++ track): declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp. Do not change these declarations: test_lab.cpp calls them.
// (Part B of the lab -- fk.hpp, fk_starter.cpp, robot_data.cpp, main.cpp -- is a separate program, fk_compare,
// built by the same CMakeLists.txt; it does not use this file.)
//
// DH table format: the struct DHTable below (the Python dict with the same keys).
//   convention : "standard" (Corke Sec. 7.1, distal frames) or "modified" (MR App. C / Craig, proximal frames)
//   types      : string of 'R'/'P', one character per joint
//   d, a, alpha, offset : length-n vectors (m, rad).  In the MODIFIED convention row i stores a_{i-1} and
//                alpha_{i-1} (MR's columns alpha_{i-1}, a_{i-1}, d_i, phi_i).
//                Revolute joint i:  theta_i = offset_i + q_i, d_i constant.
//                Prismatic joint i: d_i(total) = d_i + q_i,   theta_i = offset_i constant.
//   base, tool : 4x4 constant transforms, T = base * A_1 * ... * A_n * tool.
//
// Python -> C++: ValueError -> std::invalid_argument; a tuple -> std::pair; a list of 4x4 arrays / an (N, 4, 4)
// array -> std::vector<Eigen::Matrix4d>; an (N, n) array of joint vectors -> Eigen::MatrixXd (row = sample).
#ifndef SBEG361_LAB6_HPP
#define SBEG361_LAB6_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include <functional>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace lab {

// ---------------------------------------------------------------------------
// Course-arm model and RCM holder (given; the same numbers as course_arm_model.py)
// ---------------------------------------------------------------------------
constexpr double PI = 3.14159265358979323846;
constexpr double W1 = 0.0634, W2 = 0.0456, L1 = 0.110, L2 = 0.096, H1 = 0.1312, H2 = 0.0751;  // m
constexpr double ENC_RES = 2 * PI / 4096;  // 12-bit encoders (0.088 deg)
constexpr double RCM_TIP0 = 0.10;          // m, tip depth below the port at q3 = 0

/// Space screw axes of the course arm as COLUMNS S_i = (w_i, v_i) (6 x 6), MR Eq. 4.14 (= SLIST).
inline Eigen::MatrixXd course_arm_slist() {
    Eigen::MatrixXd S(6, 6);
    // each row below is one screw axis (w, v); the matrix is transposed so that columns are S_i
    S << 0, 0, 1, 0, 0, 0,
         0, 1, 0, -H1, 0, 0,
         0, 1, 0, -H1, 0, L1,
         0, 1, 0, -H1, 0, L1 + L2,
         0, 0, -1, -W1, L1 + L2, 0,
         0, 1, 0, H2 - H1, 0, L1 + L2;
    S.transposeInPlace();
    return S;
}

/// T_sb of the course arm at theta = 0 (= M_HOME).
inline Eigen::Matrix4d course_arm_m() {
    Eigen::Matrix4d M;
    M << -1, 0, 0, L1 + L2,
          0, 0, 1, W1 + W2,
          0, 1, 0, H1 - H2,
          0, 0, 0, 1;
    return M;
}

/// "Ready" configuration of the course arm in rad (= Q_HOME).
inline Eigen::VectorXd q_home() {
    Eigen::VectorXd q(6);
    q << 0.0, -0.6, 1.4, -2.4, -1.57, 0.0;
    return q;
}

/// Remote centre of motion (trocar port) of the RCM holder in {s}, m (= RCM_PORT).
inline Eigen::Vector3d rcm_port() { return Eigen::Vector3d(0.45, 0.10, 0.30); }

// ---------------------------------------------------------------------------
// DH table
// ---------------------------------------------------------------------------

/// Eigen::VectorXd from a list of numbers (given), e.g.  t.d = vec({H1, 0, 0, W1, H2, W2});
/// Safer than `t.d << ...` (the comma initialiser needs t.d.resize(n) first and exactly n numbers).
inline Eigen::VectorXd vec(std::initializer_list<double> values) {
    Eigen::VectorXd v(static_cast<Eigen::Index>(values.size()));
    Eigen::Index i = 0;
    for (const double x : values) v(i++) = x;
    return v;
}

struct DHTable {
    std::string convention;                              ///< "standard" or "modified"
    std::string types;                                   ///< e.g. "RRPR"
    Eigen::VectorXd d, a, alpha, offset;                 ///< one entry per joint (m, rad)
    Eigen::Matrix4d base = Eigen::Matrix4d::Identity();  ///< constant base transform
    Eigen::Matrix4d tool = Eigen::Matrix4d::Identity();  ///< constant tool transform
};

// ---------------------------------------------------------------------------
// Provided helpers (Lab 5) -- given in starter.cpp
// ---------------------------------------------------------------------------

/// [w] (3x3 skew-symmetric matrix of w).
Eigen::Matrix3d skew(const Eigen::Vector3d& w);

/// e^{[S]theta} for a screw axis S = (w, v) with ||w|| = 1 or (w = 0, ||v|| = 1) (MR Prop. 3.25, Eq. 3.88).
/// Provided; equals Lab 3's matrix_exp6(se3mat) applied to [S]*theta.
Eigen::Matrix4d screw_exp(const sbeg::Vector6d& S, double theta);

/// Space-form PoE T = e^{[S1]th1} ... e^{[Sn]thn} M (MR Eq. 4.14); same signature as Lab 5's fk_space.
/// Slist: 6 x n, columns S_i.
Eigen::Matrix4d fk_space(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist, const Eigen::VectorXd& theta);

/// Trans(p): pure translation.
Eigen::Matrix4d trans(const Eigen::Vector3d& p);
/// Rot(z, t) as a 4x4 transform.
Eigen::Matrix4d rotz_h(double t);
/// Rot(x, t) as a 4x4 transform.
Eigen::Matrix4d rotx_h(double t);

/// (theta_i, d_i) actually used for every row at joint vector q (provided; = _joint_values in Python).
std::pair<Eigen::VectorXd, Eigen::VectorXd> joint_values(const DHTable& table, const Eigen::VectorXd& q);

// ---------------------------------------------------------------------------
// Step 1 -- link transforms
// ---------------------------------------------------------------------------

/// Standard DH link transform A_j = Rz(theta) Tz(d) Tx(a) Rx(alpha) (Corke Eqs. 7.1-7.2, p. 138).
/// Return the 4x4 matrix written out entry by entry (do not multiply four matrices).
Eigen::Matrix4d dh_std(double theta, double d, double a, double alpha);

/// Modified DH (MR App. C.3, p. 592): T_{i-1,i} = Rx(alpha_{i-1}) Tx(a_{i-1}) Tz(d_i) Rz(theta_i).
/// Return the 4x4 matrix written out entry by entry.
Eigen::Matrix4d dh_mod(double alpha_prev, double a_prev, double d, double theta);

// ---------------------------------------------------------------------------
// Step 2 -- FK from a table
// ---------------------------------------------------------------------------

/// The n link transforms A_1(q_1) ... A_n(q_n) for either convention.
/// Use joint_values, then dh_std or dh_mod depending on table.convention;
/// throw std::invalid_argument for an unknown convention.
std::vector<Eigen::Matrix4d> link_transforms(const DHTable& table, const Eigen::VectorXd& q);

/// T = base * A_1(q_1) ... A_n(q_n) * tool (Corke Eq. 7.4; MR Eq. C.1).
Eigen::Matrix4d fk_dh(const DHTable& table, const Eigen::VectorXd& q);

/// [T_{s,0}, T_{s,1}, ..., T_{s,n}]: the base frame and every link frame (tool excluded); n + 1 matrices.
std::vector<Eigen::Matrix4d> link_frames(const DHTable& table, const Eigen::VectorXd& q);

// ---------------------------------------------------------------------------
// Step 3 -- the course arm
// ---------------------------------------------------------------------------

/// Your hand-derived STANDARD DH table of the course arm (same zero pose and joint signs as
/// course_arm_slist() / course_arm_m()).  Use the constants H1, H2, L1, L2, W1, W2.  base = I.
/// Hint: at q = 0 the last DH frame and {b} differ by a constant rotation about z: put it in `tool`.
DHTable course_arm_dh();

/// Your hand-derived MODIFIED (MR App. C) DH table of the course arm (row i holds alpha_{i-1}, a_{i-1}).
DHTable course_arm_mdh();

// ---------------------------------------------------------------------------
// Step 4 -- conversions
// ---------------------------------------------------------------------------

/// (Slist, M) of the same robot (MR Eqs. C.11-C.13, written for both conventions); Slist is 6 x n (columns S_i).
/// The joint-i axis is the z axis of frame {i-1} (standard) or of frame {i} (modified), at q = 0.
/// Revolute: S_i = (z, -z x o); prismatic: S_i = (0, z), where z and o are that frame's z axis and origin.
/// M = T(0).  Use link_frames(table, zeros).
std::pair<Eigen::MatrixXd, Eigen::Matrix4d> poe_from_dh(const DHTable& table);

/// Standard -> modified table of the same robot (Corke Sec. 7.5.3, pp. 160-162).
/// Regroup  B [Rz Tz Tx Rx]_1 ... [Rz Tz Tx Rx]_n T  so that every group reads [Tx Rx Rz Tz]:
/// row i of the new table takes a_{i-1}, alpha_{i-1} from row i-1 of the old one (0 for row 1), and the
/// leftover Tx(a_n) Rx(alpha_n) of the last row moves into the tool.  Return a NEW table.
DHTable std_to_mod(const DHTable& table);

/// Modified -> standard table of the same robot (the inverse regrouping): the leading
/// Rx(alpha_0) Tx(a_0) moves into the base, row i takes a_i, alpha_i from row i+1, and a_n = alpha_n = 0.
DHTable mod_to_std(const DHTable& table);

// ---------------------------------------------------------------------------
// Step 5 -- RCM robot (the Robotics Toolbox part, rtb_robot, is Python only)
// ---------------------------------------------------------------------------

/// Standard DH table of the RRPR laparoscope holder of Workshop 7 Problem 2 (types "RRPR").
/// base = Trans(port); the tip is tip0 + q3 beyond the port along the shaft; tool = I.
DHTable rcm_dh(const Eigen::Vector3d& port = rcm_port(), double tip0 = RCM_TIP0);

/// Distance from the port to the instrument shaft (the line through T's origin along T's z axis).
double rcm_distance(const Eigen::Matrix4d& T, const Eigen::Vector3d& port = rcm_port());

// ---------------------------------------------------------------------------
// Step 6 -- comparison and timing helpers
// ---------------------------------------------------------------------------

/// Read a pose file: header line, then one pose per row as r11,r12,r13,r21,r22,r23,r31,r32,r33,px,py,pz.
/// Return one 4x4 matrix per row.
std::vector<Eigen::Matrix4d> read_pose_csv(const std::string& path);

/// (max position error [m], max rotation error [rad]) between two lists of poses of equal length
/// (throw std::invalid_argument if the lengths differ or the lists are empty).
/// Rotation error = angle of R_a^T R_b, from cos(angle) = (tr R - 1)/2 (MR Eq. 3.54); clip to [-1, 1].
std::pair<double, double> pose_errors(const std::vector<Eigen::Matrix4d>& Ta, const std::vector<Eigen::Matrix4d>& Tb);

/// Best-of-`repeat` mean wall-clock time of one call fn(q) over the rows of qs (seconds).
/// Use std::chrono::steady_clock.  Throw std::invalid_argument if qs has no rows or repeat < 1.
double time_fk(const std::function<Eigen::Matrix4d(const Eigen::VectorXd&)>& fn, const Eigen::MatrixXd& qs,
               int repeat = 3);

}  // namespace lab

#endif  // SBEG361_LAB6_HPP
