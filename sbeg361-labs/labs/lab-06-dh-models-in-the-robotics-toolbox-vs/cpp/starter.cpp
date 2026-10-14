// SBEG 361 Lab 6 -- DH models vs the product of exponentials (C++ STARTER).
//
// Fill in every function marked  // TODO  (replace the `throw std::logic_error("TODO: ...")` line).
// Do not change function names or signatures (they are declared in lab.hpp). Build and test from the cpp folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
// or from your Lab code folder:   python run_lab.py 6 --cpp
//
// Rules (as in starter.py): do NOT call mrcpp (mr::...) in this file; it is the reference the tests compare you
// with.  The provided helpers below (skew, screw_exp, fk_space, trans, rotz_h, rotx_h, joint_values) may be used.
//
// Eigen reminders:  Eigen::Matrix4d A;  A << a, b, c, d,  e, f, g, h,  ...;   (comma initialiser, row by row)
//   A * B (matrix product), A.block<3, 1>(0, 3) (translation), A.block<3, 3>(0, 0) (rotation), A.col(2),
//   v.cross(w), v.dot(w), v.norm(), Eigen::VectorXd::Zero(n), v.head(k), v.tail(k), R.trace(), std::acos.
//   A DHTable is filled field by field:  DHTable t;  t.convention = "standard";  t.types = "RRR";
//   t.d = vec({0.1, 0, 0});  ...  t.tool = rotz_h(PI);   (vec, PI, H1, L1, ... are given in lab.hpp)
#include "lab.hpp"

#include <sbeg/labio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace lab {

// ---------------------------------------------------------------------------
// Provided helpers (Lab 5) -- given, do not change (you may use them)
// ---------------------------------------------------------------------------

Eigen::Matrix3d skew(const Eigen::Vector3d& w) {
    Eigen::Matrix3d W;
    W << 0.0, -w(2), w(1),
         w(2), 0.0, -w(0),
         -w(1), w(0), 0.0;
    return W;
}

Eigen::Matrix4d screw_exp(const sbeg::Vector6d& S, double theta) {
    const Eigen::Vector3d w = S.head<3>(), v = S.tail<3>();
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    if (w.norm() < 1e-12) {  // prismatic: pure translation
        T.block<3, 1>(0, 3) = v * theta;
        return T;
    }
    const Eigen::Matrix3d W = skew(w), I = Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d R = I + std::sin(theta) * W + (1 - std::cos(theta)) * W * W;  // Rodrigues (MR Eq. 3.51)
    const Eigen::Matrix3d G = I * theta + (1 - std::cos(theta)) * W + (theta - std::sin(theta)) * W * W;
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = G * v;
    return T;
}

Eigen::Matrix4d fk_space(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist, const Eigen::VectorXd& theta) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    for (Eigen::Index i = 0; i < theta.size(); ++i) T = T * screw_exp(Slist.col(i), theta(i));
    return T * M;
}

Eigen::Matrix4d trans(const Eigen::Vector3d& p) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 1>(0, 3) = p;
    return T;
}

Eigen::Matrix4d rotz_h(double t) {
    const double c = std::cos(t), s = std::sin(t);
    Eigen::Matrix4d T;
    T << c, -s, 0, 0,
         s, c, 0, 0,
         0, 0, 1, 0,
         0, 0, 0, 1;
    return T;
}

Eigen::Matrix4d rotx_h(double t) {
    const double c = std::cos(t), s = std::sin(t);
    Eigen::Matrix4d T;
    T << 1, 0, 0, 0,
         0, c, -s, 0,
         0, s, c, 0,
         0, 0, 0, 1;
    return T;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> joint_values(const DHTable& table, const Eigen::VectorXd& q) {
    Eigen::VectorXd th = table.offset, d = table.d;
    for (std::size_t k = 0; k < table.types.size(); ++k) {
        const auto i = static_cast<Eigen::Index>(k);
        if (table.types[k] == 'R')
            th(i) += q(i);
        else
            d(i) += q(i);
    }
    return {th, d};
}

// ---------------------------------------------------------------------------
// Step 1 -- link transforms
// ---------------------------------------------------------------------------

Eigen::Matrix4d dh_std(double theta, double d, double a, double alpha) {
    // TODO: the standard DH matrix of Corke Eq. 7.2, written out entry by entry (do not multiply four matrices).
    throw std::logic_error("TODO: dh_std");
}

Eigen::Matrix4d dh_mod(double alpha_prev, double a_prev, double d, double theta) {
    // TODO: the modified DH matrix of MR Sec. C.3 (p. 592), written out entry by entry.
    throw std::logic_error("TODO: dh_mod");
}

// ---------------------------------------------------------------------------
// Step 2 -- FK from a table
// ---------------------------------------------------------------------------

std::vector<Eigen::Matrix4d> link_transforms(const DHTable& table, const Eigen::VectorXd& q) {
    // TODO: const auto [th, d] = joint_values(table, q);  then one dh_std (table.convention == "standard") or
    //       dh_mod ("modified") per row, pushed into a std::vector; any other convention:
    //       throw std::invalid_argument("unknown convention ...").
    throw std::logic_error("TODO: link_transforms");
}

Eigen::Matrix4d fk_dh(const DHTable& table, const Eigen::VectorXd& q) {
    // TODO: T = base * A_1 ... A_n * tool
    throw std::logic_error("TODO: fk_dh");
}

std::vector<Eigen::Matrix4d> link_frames(const DHTable& table, const Eigen::VectorXd& q) {
    // TODO: [T_s0 = base, T_s1, ..., T_sn]  (n + 1 matrices, tool excluded)
    throw std::logic_error("TODO: link_frames");
}

// ---------------------------------------------------------------------------
// Step 3 -- the course arm
// ---------------------------------------------------------------------------

DHTable course_arm_dh() {
    // TODO: your hand-derived STANDARD table (convention "standard", types "RRRRRR"); use H1, H2, L1, L2, W1, W2.
    //       base = I.  Hint: at q = 0 the last DH frame and {b} differ by a constant rotation about z: put it in tool.
    throw std::logic_error("TODO: course_arm_dh");
}

DHTable course_arm_mdh() {
    // TODO: your hand-derived MODIFIED (MR App. C) table: row i holds alpha_{i-1}, a_{i-1}.
    throw std::logic_error("TODO: course_arm_mdh");
}

// ---------------------------------------------------------------------------
// Step 4 -- conversions
// ---------------------------------------------------------------------------

std::pair<Eigen::MatrixXd, Eigen::Matrix4d> poe_from_dh(const DHTable& table) {
    // TODO: frames = link_frames(table, Eigen::VectorXd::Zero(n)).  For joint i take F = frames[i] (standard) or
    //       frames[i + 1] (modified); z = F.block<3, 1>(0, 2), o = F.block<3, 1>(0, 3).
    //       Revolute: S.col(i) = (z, -z x o); prismatic: (0, z).  M = fk_dh(table, zeros).  Return {S, M}.
    throw std::logic_error("TODO: poe_from_dh");
}

DHTable std_to_mod(const DHTable& table) {
    // TODO: DHTable out = table;  out.convention = "modified";
    //       out.a = (0, a_1, ..., a_{n-1}), out.alpha = (0, alpha_1, ..., alpha_{n-1});
    //       out.tool = Trans(a_n x_hat) * Rot(x, alpha_n) * table.tool   (use trans and rotx_h).
    throw std::logic_error("TODO: std_to_mod");
}

DHTable mod_to_std(const DHTable& table) {
    // TODO: the inverse regrouping: out.base = base * Rot(x, alpha_0) * Trans(a_0 x_hat);
    //       out.a = (a_1, ..., a_{n-1}, 0), out.alpha = (alpha_1, ..., alpha_{n-1}, 0).
    throw std::logic_error("TODO: mod_to_std");
}

// ---------------------------------------------------------------------------
// Step 5 -- RCM robot
// ---------------------------------------------------------------------------

DHTable rcm_dh(const Eigen::Vector3d& port, double tip0) {
    // TODO: standard table of the RRPR holder (Workshop 7 Problem 2): base = trans(port); the tip is tip0 + q3
    //       beyond the port along the shaft; tool = I.
    throw std::logic_error("TODO: rcm_dh");
}

double rcm_distance(const Eigen::Matrix4d& T, const Eigen::Vector3d& port) {
    // TODO: distance from port to the line through T's origin along T's z axis.
    throw std::logic_error("TODO: rcm_distance");
}

// ---------------------------------------------------------------------------
// Step 6 -- comparison and timing helpers
// ---------------------------------------------------------------------------

std::vector<Eigen::Matrix4d> read_pose_csv(const std::string& path) {
    // TODO: const sbeg::CsvTable t = sbeg::read_csv(path);   t.data is N x 12 (header skipped).
    //       Row k: R(i, c) = t.data(k, 3 * i + c) (row-major r11..r33), p(i) = t.data(k, 9 + i); last row 0 0 0 1.
    throw std::logic_error("TODO: read_pose_csv");
}

std::pair<double, double> pose_errors(const std::vector<Eigen::Matrix4d>& Ta, const std::vector<Eigen::Matrix4d>& Tb) {
    // TODO: throw std::invalid_argument if the sizes differ or are 0.  Max position error and max angle of
    //       R_a^T R_b: cos(angle) = (tr R - 1)/2 (MR Eq. 3.54); clip to [-1, 1] (std::min / std::max) before std::acos.
    throw std::logic_error("TODO: pose_errors");
}

double time_fk(const std::function<Eigen::Matrix4d(const Eigen::VectorXd&)>& fn, const Eigen::MatrixXd& qs,
               int repeat) {
    // TODO: throw std::invalid_argument if qs.rows() == 0 or repeat < 1.  For each repeat:
    //       auto t0 = std::chrono::steady_clock::now();  call fn(qs.row(i).transpose()) for every row;
    //       seconds = std::chrono::duration<double>(t1 - t0).count();  keep the smallest seconds / qs.rows().
    throw std::logic_error("TODO: time_fk");
}

}  // namespace lab
