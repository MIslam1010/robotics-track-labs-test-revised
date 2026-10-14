// SBEG 361 Lab 3 -- Exponential coordinates and logarithms on SO(3) and SE(3); IMU orientation (C++ STARTER).
//
// Fill in every function marked  // TODO  (replace the `throw std::logic_error("TODO: ...")` line; 21 functions).
// Do not change function names or signatures (they are declared in lab.hpp). Build and test from the cpp folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
// or from your Lab code folder:   python run_lab.py 3 --cpp
//
// Notation follows Lynch & Park, *Modern Robotics* (MR), Sections 3.2.3, 3.3.2.2-3.3.3 and Appendix B.3 (see
// lab.hpp). You may NOT call mrcpp (mr::...) or Eigen's Quaterniond / AngleAxisd inside this file -- the tests
// compare your answers with mrcpp, the C++ port of modern_robotics, and with numbers computed by the Python
// reference (modern_robotics and SciPy). The helpers from Labs 1-2 at the top of this file are complete.
//
// Eigen reminders:  Eigen::Matrix3d M;  M << a, b, c,  d, e, f,  g, h, i;   (comma initialiser, row by row)
//   M.transpose(), M.trace(), v.norm(), v.dot(w), v.cross(w), Eigen::Matrix3d::Identity(), ::Zero(), A * B,
//   M(i, j) (0-based), T.topLeftCorner<3, 3>() (R block), T.topRightCorner<3, 1>() (p column), V.head<3>(),
//   V.tail<3>(), omega_b.rows(), omega_b.row(k).transpose() (row k as a column Vector3d),
//   Eigen::Vector4d q(q0, q1, q2, q3), Eigen::VectorXd::Zero(N).  Use std::sqrt, std::sin, std::cos, std::tan,
//   std::acos, std::abs.
#include "lab.hpp"

#include <sbeg/labio.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace lab {

// ===========================================================================
// given helpers (Labs 1-2): complete, do not change
// ===========================================================================

Eigen::Matrix4d rp_to_trans(const Eigen::Matrix3d& R, const Eigen::Vector3d& p) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.topLeftCorner<3, 3>() = R;
    T.topRightCorner<3, 1>() = p;
    return T;
}

std::pair<Eigen::Matrix3d, Eigen::Vector3d> trans_to_rp(const Eigen::Matrix4d& T) {
    return {T.topLeftCorner<3, 3>(), T.topRightCorner<3, 1>()};
}

Eigen::Matrix4d trans_inv(const Eigen::Matrix4d& T) {
    const Eigen::Matrix3d R = T.topLeftCorner<3, 3>();
    const Eigen::Vector3d p = T.topRightCorner<3, 1>();
    return rp_to_trans(R.transpose(), -R.transpose() * p);
}

Eigen::Matrix4d vec_to_se3(const Vector6d& V) {
    Eigen::Matrix4d M;
    M << 0.0, -V(2), V(1), V(3),
         V(2), 0.0, -V(0), V(4),
         -V(1), V(0), 0.0, V(5),
         0.0, 0.0, 0.0, 0.0;
    return M;
}

Vector6d se3_to_vec(const Eigen::Matrix4d& m) {
    Vector6d V;
    V << m(2, 1), m(0, 2), m(1, 0), m(0, 3), m(1, 3), m(2, 3);
    return V;
}

ScrewParams twist_to_screw(const Vector6d& V) {
    const Eigen::Vector3d w = V.head<3>(), v = V.tail<3>();
    const double nw = w.norm(), nv = v.norm();
    ScrewParams s;
    if (nw > TOL) {
        s.S = V / nw;
        s.thetadot = nw;
        s.h = w.dot(v) / (nw * nw);
        s.s_hat = w / nw;
        s.q = w.cross(v) / (nw * nw);
        s.has_q = true;
        return s;
    }
    if (nv > TOL) {
        s.S << Eigen::Vector3d::Zero(), v / nv;
        s.thetadot = nv;
        s.h = std::numeric_limits<double>::infinity();
        s.s_hat = v / nv;
        s.q = Eigen::Vector3d::Zero();
        s.has_q = false;
        return s;
    }
    throw std::invalid_argument("zero twist has no screw axis");
}

PoseTrack load_pose_track(const std::string& path, bool noisy) {
    const sbeg::CsvTable T = sbeg::read_csv(path);
    const std::string n = noisy ? "n" : "";
    PoseTrack out;
    out.t = T.data.col(T.col("t"));
    for (Eigen::Index k = 0; k < T.data.rows(); ++k) {
        Eigen::Vector4d q;
        for (int i = 0; i < 4; ++i) q(i) = T.data(k, T.col(n + "q" + std::to_string(i)));
        const Eigen::Vector3d p(T.data(k, T.col(n + "px")), T.data(k, T.col(n + "py")), T.data(k, T.col(n + "pz")));
        out.Ts.push_back(rp_to_trans(quat_to_rot(q), p));  // uses YOUR quat_to_rot (Step 5)
    }
    return out;
}

// ===========================================================================
// so(3) <-> R^3
// ===========================================================================

Eigen::Matrix3d vec_to_so3(const Eigen::Vector3d& omega) {
    // TODO: build the matrix from the three components of omega (you wrote this in Lab 2: copy it)
    throw std::logic_error("TODO: vec_to_so3");
}

Eigen::Vector3d so3_to_vec(const Eigen::Matrix3d& so3mat) {
    // TODO: read omega back from entries (3,2), (1,3), (2,1) (1-based) -- in C++ indices start at 0
    throw std::logic_error("TODO: so3_to_vec");
}

std::pair<Eigen::Vector3d, double> axis_ang3(const Eigen::Vector3d& expc3) {
    // TODO: throw std::invalid_argument("rotation angle is (near) zero: axis undefined") if theta < TOL;
    //       return {w_hat, theta};
    throw std::logic_error("TODO: axis_ang3");
}

// ===========================================================================
// exponentials: SO(3) and SE(3)
// ===========================================================================

Eigen::Matrix3d matrix_exp3(const Eigen::Matrix3d& so3mat) {
    // TODO: Rodrigues' formula (MR Eq. 3.51).
    // Hint: theta = ||so3_to_vec(so3mat)||; if theta < TOL return I; otherwise [w_hat] = so3mat / theta.
    throw std::logic_error("TODO: matrix_exp3");
}

Eigen::Matrix3d g_matrix(const Eigen::Vector3d& omega_hat, double theta) {
    // TODO: MR Eq. 3.87, one line with vec_to_so3(omega_hat)
    throw std::logic_error("TODO: g_matrix");
}

Eigen::Matrix3d g_inv_matrix(const Eigen::Vector3d& omega_hat, double theta) {
    // TODO: MR Eq. 3.92; cot(x) = 1 / std::tan(x)
    throw std::logic_error("TODO: g_inv_matrix");
}

Eigen::Matrix4d matrix_exp6(const Eigen::Matrix4d& se3mat) {
    // TODO: split [S]theta into omega*theta (top-left 3x3 block) and v*theta (top-right column).
    // If |omega theta| < TOL use Eq. 3.89.  Otherwise theta = |omega theta|, w = (omega theta)/theta,
    // v = (v theta)/theta, rotation block = matrix_exp3(top-left block), translation = g_matrix(w, theta) * v.
    throw std::logic_error("TODO: matrix_exp6");
}

// ===========================================================================
// logarithms: SO(3) and SE(3)
// ===========================================================================

Eigen::Matrix3d matrix_log3(const Eigen::Matrix3d& R) {
    // TODO: implement all three cases of the MR algorithm:
    //   (a) R = I               -> zero matrix
    //   (b) tr R = -1 (theta=pi) -> Eqs. 3.58 / 3.59 / 3.60 (use the first feasible one)
    //   (c) otherwise           -> theta = arccos((tr R - 1)/2), [w_hat] = (R - R^T)/(2 sin theta)
    // Use a small tolerance (e.g. 1e-12) when comparing (tr R - 1)/2 with +1 and -1.
    throw std::logic_error("TODO: matrix_log3");
}

Eigen::Matrix4d matrix_log6(const Eigen::Matrix4d& T) {
    // TODO: so3 = matrix_log3(R).
    // Case (a), every |entry of so3| <= TOL: return [[0, p], [0, 0]].
    // Case (b): theta = |so3_to_vec(so3)|, w = so3_to_vec(so3)/theta, v = g_inv_matrix(w, theta) * p,
    // return [[so3, v theta], [0, 0]].
    throw std::logic_error("TODO: matrix_log6");
}

// ===========================================================================
// screw motions
// ===========================================================================

std::pair<Vector6d, double> screw_between(const Eigen::Matrix4d& T1, const Eigen::Matrix4d& T2,
                                          const std::string& frame) {
    // TODO: throw std::invalid_argument for a frame other than "space" / "body".
    // D = T2 T1^-1 (space) or T1^-1 T2 (body); St = se3_to_vec(matrix_log6(D)); theta = |omega part|,
    // or |v part| if that is < TOL; if both are < TOL return {Vector6d::Zero(), 0.0}; else {St / theta, theta}.
    throw std::logic_error("TODO: screw_between");
}

Eigen::Matrix4d screw_path(const Eigen::Matrix4d& T1, const Eigen::Matrix4d& T2, double s) {
    // TODO: use screw_between(T1, T2, "space"), vec_to_se3 and matrix_exp6.
    throw std::logic_error("TODO: screw_path");
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> screw_motion_profile(const std::vector<Eigen::Matrix4d>& Ts) {
    // TODO: first compute all steps screw_between(Ts[k-1], Ts[k], "space") (throw std::invalid_argument for a
    // step with dtheta >= TOL and |omega| < 0.5); w_ref = omega of the step with the largest dtheta; then
    // loop again: sign = +1 if omega_k . w_ref >= 0 else -1; theta(k) = theta(k-1) + sign*dtheta,
    // d(k) = d(k-1) + (omega_k . v_k)*sign*dtheta (steps with dtheta < TOL add nothing).
    throw std::logic_error("TODO: screw_motion_profile");
}

// ===========================================================================
// quaternions (MR App. B.3)
// ===========================================================================

Eigen::Vector4d rot_to_quat(const Eigen::Matrix3d& R) {
    // TODO: unit quaternion with q0 >= 0 (MR Eqs. B.10-B.11).
    // Note: printed Eq. B.11 has a typo in its last entry; it should read r21 - r12.
    // For q0 <= 0.5 (theta >= 120 deg) build q from the matrix log instead (Eq. B.9),
    // because dividing by 4*q0 becomes ill-conditioned near theta = pi.
    throw std::logic_error("TODO: rot_to_quat");
}

Eigen::Matrix3d quat_to_rot(const Eigen::Vector4d& q) {
    // TODO: MR Eq. B.12. Normalise q first.
    throw std::logic_error("TODO: quat_to_rot");
}

Eigen::Vector4d quat_mul(const Eigen::Vector4d& q, const Eigen::Vector4d& p) {
    // TODO: MR Eq. B.15
    throw std::logic_error("TODO: quat_mul");
}

Eigen::Vector4d quat_conj(const Eigen::Vector4d& q) {
    // TODO
    throw std::logic_error("TODO: quat_conj");
}

// ===========================================================================
// IMU orientation
// ===========================================================================

std::vector<Eigen::Matrix3d> integrate_gyro_exp(const Eigen::Matrix3d& R0, const Eigen::MatrixXd& omega_b,
                                                double dt) {
    // TODO: R_{k+1} = R_k exp([omega_b,k] dt) for every row k of omega_b.
    // Return a std::vector with omega_b.rows() + 1 matrices whose first entry is R0 (use push_back).
    throw std::logic_error("TODO: integrate_gyro_exp");
}

std::vector<Eigen::Matrix3d> integrate_gyro_euler(const Eigen::Matrix3d& R0, const Eigen::MatrixXd& omega_b,
                                                  double dt) {
    // TODO: R_{k+1} = R_k (I + [omega_b,k] dt); same shapes as integrate_gyro_exp
    throw std::logic_error("TODO: integrate_gyro_euler");
}

Eigen::Vector3d orientation_error(const Eigen::Matrix3d& R_cur, const Eigen::Matrix3d& R_des) {
    // TODO
    throw std::logic_error("TODO: orientation_error");
}

double rotation_angle(const Eigen::Matrix3d& R1, const Eigen::Matrix3d& R2) {
    // TODO
    throw std::logic_error("TODO: rotation_angle");
}

double knee_flexion(const Eigen::Matrix3d& R_thigh, const Eigen::Matrix3d& R_shank) {
    // TODO
    throw std::logic_error("TODO: knee_flexion");
}

}  // namespace lab
