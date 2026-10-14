// SBEG 361 Lab 4 -- Exponential coordinates on SE(3), wrenches and F/T sensor frame changes (C++ STARTER).
//
// Fill in every function marked  // TODO  (11 functions; replace the `throw std::logic_error("TODO: ...")` line).
// Do not change function names or signatures (they are declared in lab.hpp). Build and test from the cpp folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
// or from your Lab code folder:   python run_lab.py 4 --cpp
//
// Do NOT call mrcpp (the C++ port of modern_robotics) inside the TODO functions -- it is what the tests compare
// against.  The helpers from Labs 1-3 (vec_to_so3 ... adjoint), the loaders and tip_forces below are complete
// and may be used.  A least-squares solve with Eigen is allowed in estimate_payload.
// Notation: see lab.hpp (MR Sections 3.3.3 and 3.4; wrenches are (m, f), MOMENT FIRST).
//
// Eigen reminders:  Vector6d F;  F << m, f;   (stack two 3-vectors)   F.head<3>(), F.tail<3>(), F.segment<3>(i);
//   T.topLeftCorner<3, 3>() (R), T.topRightCorner<3, 1>() (p), A.transpose(), A * B, a.cross(b), a.dot(b),
//   v.norm(), Matrix3d::Identity(), Matrix4d::Zero().  Never store an Eigen product in `auto`: write the type.
#include "lab.hpp"

#include <sbeg/labio.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace lab {

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::Vector3d;

namespace {
constexpr double PI = 3.14159265358979323846;
}  // namespace

// ====================================================================== helpers from Labs 1-3 (given)
Matrix3d vec_to_so3(const Vector3d& omega) {
    Matrix3d W;
    W << 0.0, -omega(2), omega(1),
         omega(2), 0.0, -omega(0),
         -omega(1), omega(0), 0.0;
    return W;
}

Vector3d so3_to_vec(const Matrix3d& so3mat) { return Vector3d(so3mat(2, 1), so3mat(0, 2), so3mat(1, 0)); }

Matrix3d matrix_exp3(const Matrix3d& so3mat) {
    const double theta = so3_to_vec(so3mat).norm();
    if (theta < TOL) return Matrix3d::Identity();
    const Matrix3d W = so3mat / theta;
    return Matrix3d::Identity() + std::sin(theta) * W + (1.0 - std::cos(theta)) * W * W;
}

Matrix3d matrix_log3(const Matrix3d& R) {
    const double c = (R.trace() - 1.0) / 2.0;
    if (c >= 1.0 - 1e-12) return Matrix3d::Zero();
    if (c <= -1.0 + 1e-12) {
        // theta = pi: MR Eqs. 3.58-3.60 (choose a numerically safe column)
        Vector3d w;
        if (1.0 + R(2, 2) > TOL)
            w = Vector3d(R(0, 2), R(1, 2), 1.0 + R(2, 2)) / std::sqrt(2.0 * (1.0 + R(2, 2)));
        else if (1.0 + R(1, 1) > TOL)
            w = Vector3d(R(0, 1), 1.0 + R(1, 1), R(2, 1)) / std::sqrt(2.0 * (1.0 + R(1, 1)));
        else
            w = Vector3d(1.0 + R(0, 0), R(1, 0), R(2, 0)) / std::sqrt(2.0 * (1.0 + R(0, 0)));
        return vec_to_so3(PI * w);
    }
    const double theta = std::acos(std::max(-1.0, std::min(1.0, c)));
    return theta / (2.0 * std::sin(theta)) * (R - R.transpose());
}

Matrix4d rp_to_trans(const Matrix3d& R, const Vector3d& p) {
    Matrix4d T = Matrix4d::Identity();
    T.topLeftCorner<3, 3>() = R;
    T.topRightCorner<3, 1>() = p;
    return T;
}

std::pair<Matrix3d, Vector3d> trans_to_rp(const Matrix4d& T) {
    return {T.topLeftCorner<3, 3>(), T.topRightCorner<3, 1>()};
}

Matrix4d trans_inv(const Matrix4d& T) {
    const auto [R, p] = trans_to_rp(T);
    return rp_to_trans(R.transpose(), -R.transpose() * p);
}

Matrix4d vec_to_se3(const Vector6d& V) {
    Matrix4d M = Matrix4d::Zero();
    M.topLeftCorner<3, 3>() = vec_to_so3(V.head<3>());
    M.topRightCorner<3, 1>() = V.tail<3>();
    return M;
}

Vector6d se3_to_vec(const Matrix4d& se3mat) {
    Vector6d V;
    V << so3_to_vec(se3mat.topLeftCorner<3, 3>()), se3mat.topRightCorner<3, 1>();
    return V;
}

Matrix6d adjoint(const Matrix4d& T) {
    const auto [R, p] = trans_to_rp(T);
    Matrix6d A = Matrix6d::Zero();
    A.topLeftCorner<3, 3>() = R;
    A.bottomLeftCorner<3, 3>() = vec_to_so3(p) * R;
    A.bottomRightCorner<3, 3>() = R;
    return A;
}

// ====================================================================== Part A: exp and log on SE(3) (from Lab 3)
// You wrote matrix_exp6, matrix_log6, screw_between and screw_path in Lab 3 (Week 4): paste your tested Lab 3
// versions here and re-run the Checkpoint 1a/1b tests (Lab 4 Steps 1-2). The hints below summarise the method.
Matrix4d matrix_exp6(const Matrix4d& se3mat) {
    // TODO: Split [S]theta into omega*theta (so3_to_vec of the top-left 3x3 block) and v*theta (top-right 3x1
    // block); if |omega theta| < TOL use Eq. 3.89, else theta = |omega theta|, [w] = [omega theta]/theta,
    // v = (v theta)/theta and Eq. 3.88 with G(theta) of Eq. 3.87.  Reuse matrix_exp3 for the rotation block.
    // Eigen: se3mat.topLeftCorner<3, 3>(), se3mat.topRightCorner<3, 1>(), Matrix3d::Identity(), std::cos, std::sin.
    throw std::logic_error("TODO: matrix_exp6");
}

Matrix4d matrix_log6(const Matrix4d& T) {
    // TODO: Use matrix_log3(R). If it is zero (every |entry| <= TOL: so3.cwiseAbs().maxCoeff() <= TOL):
    // return [[0, p], [0, 0]] (case a). Otherwise theta = |omega theta|, [w] = log3/theta, G^-1 from Eq. 3.92
    // (cot = 1/std::tan), v = G^-1 p, and return [[ [w]theta, v theta ], [0, 0]] (start from Matrix4d::Zero()).
    throw std::logic_error("TODO: matrix_log6");
}

std::pair<Vector6d, double> screw_between(const Matrix4d& T1, const Matrix4d& T2, const std::string& frame) {
    // TODO: D = T2 T1^-1 (space) or T1^-1 T2 (body); St = se3_to_vec(matrix_log6(D)); theta = |omega part|
    // (St.head<3>().norm()), or |v part| (St.tail<3>()) if omega = 0; return {St / theta, theta}
    // ({Vector6d::Zero(), 0.0} if both are zero).  Throw std::invalid_argument for an unknown frame.
    throw std::logic_error("TODO: screw_between");
}

Matrix4d screw_path(const Matrix4d& T1, const Matrix4d& T2, double s) {
    // TODO: Use screw_between(T1, T2, "space") and matrix_exp6.
    throw std::logic_error("TODO: screw_path");
}

// ====================================================================== Part B: wrenches
Vector6d point_force_wrench(const Vector3d& f, const Vector3d& r) {
    // TODO: (r x f, f).  Eigen: r.cross(f); fill a Vector6d F with the comma initialiser  F << a, b;
    throw std::logic_error("TODO: point_force_wrench");
}

Vector6d change_wrench_frame(const Vector6d& F_a, const Matrix4d& T_ab) {
    // TODO: One line, with adjoint() from Lab 2 (and .transpose()).
    throw std::logic_error("TODO: change_wrench_frame");
}

double power(const Vector6d& V, const Vector6d& F) {
    // TODO: One line (V.dot(F)).
    throw std::logic_error("TODO: power");
}

WrenchScrew wrench_to_screw(const Vector6d& F) {
    // TODO: Two cases: f != 0 and f == 0 (pure moment: h = std::numeric_limits<double>::infinity(), leave q
    // empty); throw std::invalid_argument for F = 0.  m = F.head<3>(), f = F.tail<3>().
    // Fill a WrenchScrew w (w.h, w.q = Vector3d(...), w.s_hat, w.magnitude) and return it.
    throw std::logic_error("TODO: wrench_to_screw");
}

Vector6d gravity_wrench(double mass, const Vector3d& r_com, const Matrix3d& R_sf, const Vector3d& g_s) {
    // TODO: f_f = mass R_sf^T g_s, then point_force_wrench(f_f, r_com).
    throw std::logic_error("TODO: gravity_wrench");
}

PayloadEstimate estimate_payload(const std::vector<Matrix3d>& R_list, const Eigen::MatrixXd& F_list,
                                 const Vector3d& g_s) {
    // TODO: Throw std::invalid_argument unless F_list is K x 6 with K = R_list.size().  Build the 6K x 10 matrix A
    // (Eigen::MatrixXd::Zero(6 * K, 10)) and the 6K vector y as in lab.hpp (blocks: A.block<3, 3>(row, col) = ...;
    // y.segment<6>(6 * k) = F_list.row(k).transpose()), solve the least-squares problem with
    // x = A.completeOrthogonalDecomposition().solve(y)  (the Eigen equivalent of np.linalg.lstsq, allowed here),
    // then r_com = c / mass.  Return PayloadEstimate{mass, r_com, bias}.
    throw std::logic_error("TODO: estimate_payload");
}

Vector6d compensate(const Vector6d& F_meas, double mass, const Vector3d& r_com, const Matrix3d& R_sf,
                    const Vector6d& bias, const Vector3d& g_s) {
    // TODO: Reading - bias - gravity_wrench(...).
    throw std::logic_error("TODO: compensate");
}

// ====================================================================== loaders and analysis (given)
PoseWrenchData load_pose_wrench_csv(const std::string& path) {
    const sbeg::CsvTable tab = sbeg::read_csv(path);
    PoseWrenchData out;
    const bool has_t = std::find(tab.header.begin(), tab.header.end(), "t") != tab.header.end();
    const Eigen::Index K = tab.data.rows();
    if (has_t) out.t = tab.data.col(tab.col("t"));
    out.Fs.resize(K, 6);
    const char* wrench_cols[6] = {"mx", "my", "mz", "fx", "fy", "fz"};
    for (Eigen::Index k = 0; k < K; ++k) {
        Matrix3d R;
        for (int i = 0; i < 3; ++i)
            for (int jj = 0; jj < 3; ++jj)
                R(i, jj) = tab.data(k, tab.col("R" + std::to_string(i + 1) + std::to_string(jj + 1)));
        const Vector3d p(tab.data(k, tab.col("px")), tab.data(k, tab.col("py")), tab.data(k, tab.col("pz")));
        out.Ts.push_back(rp_to_trans(R, p));
        for (int c = 0; c < 6; ++c) out.Fs(k, c) = tab.data(k, tab.col(wrench_cols[c]));
    }
    return out;
}

sbeg::json load_json(const std::string& path) { return sbeg::load_json(path); }

Eigen::MatrixXd tip_forces(const std::vector<Matrix4d>& Ts, const Eigen::MatrixXd& Fs, double mass,
                           const Vector3d& r_com, const Vector6d& bias, const Matrix4d& T_ft, const Vector3d& g_s) {
    if (Fs.cols() != 6 || static_cast<Eigen::Index>(Ts.size()) != Fs.rows())
        throw std::invalid_argument("tip_forces: need one 6-column wrench row per pose");
    Eigen::MatrixXd Ft(Fs.rows(), 6);
    for (Eigen::Index k = 0; k < Fs.rows(); ++k) {
        const Matrix4d& T = Ts[static_cast<std::size_t>(k)];
        const Vector6d F = Fs.row(k).transpose();
        Ft.row(k) = change_wrench_frame(compensate(F, mass, r_com, T.topLeftCorner<3, 3>(), bias, g_s), T_ft)
                        .transpose();
    }
    return Ft;
}

}  // namespace lab
