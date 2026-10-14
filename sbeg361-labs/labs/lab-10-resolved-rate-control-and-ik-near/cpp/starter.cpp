// SBEG 361 Lab 10 -- Resolved-rate control, damped least squares, redundancy and a hexapod (C++ STARTER).
//
// Fill in every function marked  // TODO  (15 functions: replace the `throw std::logic_error("TODO: ...")` line).
// Do not change function names or signatures (they are declared in lab.hpp). Build and test from the cpp folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
// or from your Lab code folder:   python run_lab.py 10 --cpp
//
// You may NOT call mrcpp (mr::pinv, mr::jacobian_body, mr::ikin_body, ...) inside this file -- it is what the tests
// compare against. The helpers marked "complete" (from your Labs 2-7 library) may be used freely; replace them by
// your own versions if you prefer. (matrix_exp6 keeps the first-order rotation for angles below 1e-6 rad, unlike
// mr::matrix_exp6; Newton iterations on tiny corrections need it.) Notation: Lynch & Park, *Modern Robotics* (MR),
// see lab.hpp.
//
// Eigen reminders:  Eigen::JacobiSVD<Eigen::MatrixXd> svd(J, Eigen::ComputeThinU | Eigen::ComputeThinV);
//   svd.singularValues() (descending), svd.matrixU(), svd.matrixV() (V, not V^T); s.asDiagonal();
//   Eigen::MatrixXd::Identity(m, m), A.partialPivLu().solve(B) (= np.linalg.solve(A, B)), A.inverse(), J.transpose(),
//   J.rows(), J.cols(), T.topLeftCorner<3, 3>() (R), T.topRightCorner<3, 1>() (p), v.head<3>() / v.tail<3>(),
//   M.row(k) / M.col(i) (read or write), a.cross(b), v.norm(), v.cwiseAbs().maxCoeff(), (a.array() / b.array()),
//   M.colwise().maxCoeff() (= M.max(axis=0)), Eigen::Index for loop counters, std::vector::push_back.
//   A 6-vector for vec_to_se3: `const sbeg::Vector6d V = ...;`
#include "lab.hpp"

#include <sbeg/labio.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lab {

namespace {
constexpr double PI = 3.14159265358979323846;
}  // namespace

// ===================================================================== helpers (complete; from Labs 2-7)

Eigen::Matrix3d vec_to_so3(const Eigen::Vector3d& omega) {
    Eigen::Matrix3d m;
    m << 0.0, -omega(2), omega(1),
         omega(2), 0.0, -omega(0),
         -omega(1), omega(0), 0.0;
    return m;
}

Eigen::Vector3d so3_to_vec(const Eigen::Matrix3d& so3mat) {
    return Eigen::Vector3d(so3mat(2, 1), so3mat(0, 2), so3mat(1, 0));
}

Eigen::Matrix4d vec_to_se3(const sbeg::Vector6d& V) {
    Eigen::Matrix4d out = Eigen::Matrix4d::Zero();
    out.topLeftCorner<3, 3>() = vec_to_so3(V.head<3>());
    out.topRightCorner<3, 1>() = V.tail<3>();
    return out;
}

sbeg::Vector6d se3_to_vec(const Eigen::Matrix4d& se3mat) {
    sbeg::Vector6d V;
    V << se3mat(2, 1), se3mat(0, 2), se3mat(1, 0), se3mat(0, 3), se3mat(1, 3), se3mat(2, 3);
    return V;
}

Eigen::Matrix4d trans_inv(const Eigen::Matrix4d& T) {
    const Eigen::Matrix3d R = T.topLeftCorner<3, 3>();
    const Eigen::Vector3d p = T.topRightCorner<3, 1>();
    Eigen::Matrix4d out = Eigen::Matrix4d::Identity();
    out.topLeftCorner<3, 3>() = R.transpose();
    out.topRightCorner<3, 1>() = -R.transpose() * p;
    return out;
}

sbeg::Matrix6d adjoint(const Eigen::Matrix4d& T) {
    const Eigen::Matrix3d R = T.topLeftCorner<3, 3>();
    const Eigen::Vector3d p = T.topRightCorner<3, 1>();
    sbeg::Matrix6d out = sbeg::Matrix6d::Zero();
    out.topLeftCorner<3, 3>() = R;
    out.bottomRightCorner<3, 3>() = R;
    out.bottomLeftCorner<3, 3>() = vec_to_so3(p) * R;
    return out;
}

Eigen::Matrix3d matrix_exp3(const Eigen::Matrix3d& so3mat) {
    const double th = so3_to_vec(so3mat).norm();
    if (th < TOL) return Eigen::Matrix3d::Identity() + so3mat;
    const Eigen::Matrix3d K = so3mat / th;
    return Eigen::Matrix3d::Identity() + std::sin(th) * K + (1.0 - std::cos(th)) * K * K;
}

Eigen::Matrix4d matrix_exp6(const Eigen::Matrix4d& se3mat) {
    const Eigen::Matrix3d W = se3mat.topLeftCorner<3, 3>();
    const Eigen::Vector3d v = se3mat.topRightCorner<3, 1>();
    const double th = so3_to_vec(W).norm();
    Eigen::Matrix4d out = Eigen::Matrix4d::Identity();
    if (th < TOL) {  // first-order rotation (kept, unlike mr.MatrixExp6)
        out.topLeftCorner<3, 3>() += W;
        out.topRightCorner<3, 1>() = v;
        return out;
    }
    const Eigen::Matrix3d K = W / th;
    out.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity() + std::sin(th) * K + (1.0 - std::cos(th)) * K * K;
    const Eigen::Matrix3d G = Eigen::Matrix3d::Identity() * th + (1.0 - std::cos(th)) * K + (th - std::sin(th)) * K * K;
    out.topRightCorner<3, 1>() = G * (v / th);
    return out;
}

Eigen::Matrix3d matrix_log3(const Eigen::Matrix3d& R) {
    const double c = std::min(1.0, std::max(-1.0, (R.trace() - 1.0) / 2.0));
    if (c >= 1.0 - 1e-12) return Eigen::Matrix3d::Zero();
    if (c <= -1.0 + 1e-12) {
        Eigen::Vector3d w;
        if (1.0 + R(2, 2) > 1e-9)
            w = Eigen::Vector3d(R(0, 2), R(1, 2), 1.0 + R(2, 2)) / std::sqrt(2.0 * (1.0 + R(2, 2)));
        else if (1.0 + R(1, 1) > 1e-9)
            w = Eigen::Vector3d(R(0, 1), 1.0 + R(1, 1), R(2, 1)) / std::sqrt(2.0 * (1.0 + R(1, 1)));
        else
            w = Eigen::Vector3d(1.0 + R(0, 0), R(1, 0), R(2, 0)) / std::sqrt(2.0 * (1.0 + R(0, 0)));
        return vec_to_so3(PI * w);
    }
    const double th = std::acos(c);
    return th / (2.0 * std::sin(th)) * (R - R.transpose());
}

Eigen::Matrix4d matrix_log6(const Eigen::Matrix4d& T) {
    const Eigen::Matrix3d R = T.topLeftCorner<3, 3>();
    const Eigen::Vector3d p = T.topRightCorner<3, 1>();
    const Eigen::Matrix3d W = matrix_log3(R);
    Eigen::Matrix4d out = Eigen::Matrix4d::Zero();
    if (W.cwiseAbs().maxCoeff() <= 1e-8) {  // np.allclose(W, 0)
        out.topRightCorner<3, 1>() = p;
        return out;
    }
    const double th = so3_to_vec(W).norm();
    const Eigen::Matrix3d K = W / th;
    const Eigen::Matrix3d Ginv =
        Eigen::Matrix3d::Identity() / th - K / 2.0 + (1.0 / th - 0.5 / std::tan(th / 2.0)) * K * K;
    out.topLeftCorner<3, 3>() = W;
    out.topRightCorner<3, 1>() = (Ginv * p) * th;
    return out;
}

Eigen::Matrix4d fk_space(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist, const Eigen::VectorXd& theta) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    for (Eigen::Index i = 0; i < theta.size(); ++i) {
        const sbeg::Vector6d S = Slist.col(i) * theta(i);
        T = (T * matrix_exp6(vec_to_se3(S))).eval();
    }
    return T * M;
}

Eigen::MatrixXd body_screws(const Eigen::MatrixXd& Slist, const Eigen::Matrix4d& M) {
    return adjoint(trans_inv(M)) * Slist;
}

Eigen::MatrixXd jacobian_space(const Eigen::MatrixXd& Slist, const Eigen::VectorXd& theta) {
    Eigen::MatrixXd J = Slist;
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    for (Eigen::Index i = 1; i < Slist.cols(); ++i) {
        const sbeg::Vector6d S = Slist.col(i - 1) * theta(i - 1);
        T = (T * matrix_exp6(vec_to_se3(S))).eval();
        J.col(i) = adjoint(T) * Slist.col(i);
    }
    return J;
}

Eigen::MatrixXd jacobian_body(const Eigen::MatrixXd& Blist, const Eigen::VectorXd& theta) {
    const Eigen::Index n = Blist.cols();
    Eigen::MatrixXd J = Blist;
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    for (Eigen::Index i = n - 2; i >= 0; --i) {
        const sbeg::Vector6d B = Blist.col(i + 1) * -theta(i + 1);
        T = (T * matrix_exp6(vec_to_se3(B))).eval();
        J.col(i) = adjoint(T) * Blist.col(i);
    }
    return J;
}

Robot load_robot(const std::string& name, const std::string& path) {
    const sbeg::json all = sbeg::load_json(path.empty() ? sbeg::lab_path("../data/robots.json") : path);
    if (!all.contains(name)) throw std::invalid_argument("load_robot: no robot named '" + name + "'");
    const sbeg::json& d = all.at(name);
    Robot r;
    r.Slist = sbeg::as_mat(d.at("Slist"));
    r.M = sbeg::as<Eigen::Matrix4d>(d.at("M"));
    r.name = d.at("name").get<std::string>();
    // limits, qdot_max, q_home may be null in the file: left empty then.
    if (d.contains("limits") && !d.at("limits").is_null()) r.limits = sbeg::as_mat(d.at("limits"));
    if (d.contains("qdot_max") && !d.at("qdot_max").is_null()) r.qdot_max = sbeg::as_vec(d.at("qdot_max"));
    if (d.contains("q_home") && !d.at("q_home").is_null()) r.q_home = sbeg::as_vec(d.at("q_home"));
    return r;
}

Hexapod load_hexapod(const std::string& path) {
    const sbeg::json d = sbeg::load_json(path.empty() ? sbeg::lab_path("../data/hexapod.json") : path);
    Hexapod h;
    h.A = sbeg::as_mat(d.at("a"));
    h.B = sbeg::as_mat(d.at("b"));
    h.name = d.at("name").get<std::string>();
    h.h0 = d.at("params").at("h0").get<double>();
    h.stroke = sbeg::as<Eigen::Vector2d>(d.at("stroke"));
    h.leg_speed_max = d.at("leg_speed_max").get<double>();
    h.rated_leg_force = d.at("rated_leg_force").get<double>();
    return h;
}

JointLog read_joint_log(const std::string& path) {
    const sbeg::CsvTable tab = sbeg::read_csv(path);
    Eigen::Index n = 0;
    for (const std::string& h : tab.header)
        if (h.rfind("cmd_", 0) == 0) ++n;
    JointLog log;
    log.t = tab.data.col(0);
    log.cmd_deg = tab.data.middleCols(1, n);
    log.meas_deg = tab.data.middleCols(1 + n, n);
    return log;
}

void write_joint_log(const std::string& path, const Eigen::VectorXd& t, const Eigen::MatrixXd& cmd_deg,
                     const Eigen::MatrixXd& meas_deg) {
    const Eigen::MatrixXd& meas = meas_deg.size() == 0 ? cmd_deg : meas_deg;
    const Eigen::Index n = cmd_deg.cols();
    std::ofstream out(sbeg::fs_path(path), std::ios::binary);
    if (!out) throw std::runtime_error("cannot write file: " + path);
    out << "t";
    for (Eigen::Index i = 0; i < n; ++i) out << ",cmd_" << i + 1;
    for (Eigen::Index i = 0; i < n; ++i) out << ",meas_" << i + 1;
    out << "\r\n" << std::fixed << std::setprecision(4);  // csv.writer line ending, f"{x:.4f}"
    for (Eigen::Index k = 0; k < t.size(); ++k) {
        out << t(k);
        for (Eigen::Index i = 0; i < n; ++i) out << "," << cmd_deg(k, i);
        for (Eigen::Index i = 0; i < n; ++i) out << "," << meas(k, i);
        out << "\r\n";
    }
}

Eigen::MatrixXd gradient(const Eigen::MatrixXd& Y, const Eigen::VectorXd& t) {
    const Eigen::Index N = Y.rows();
    if (t.size() != N || N < 2) throw std::invalid_argument("gradient: need N >= 2 samples and t of length N");
    Eigen::MatrixXd D(N, Y.cols());
    D.row(0) = (Y.row(1) - Y.row(0)) / (t(1) - t(0));
    D.row(N - 1) = (Y.row(N - 1) - Y.row(N - 2)) / (t(N - 1) - t(N - 2));
    for (Eigen::Index k = 1; k + 1 < N; ++k) {
        const double h1 = t(k) - t(k - 1), h2 = t(k + 1) - t(k);
        const double a = -h2 / (h1 * (h1 + h2)), b = (h2 - h1) / (h1 * h2), c = h1 / (h2 * (h1 + h2));
        D.row(k) = a * Y.row(k - 1) + b * Y.row(k) + c * Y.row(k + 1);
    }
    return D;
}

// ===================================================================== Checkpoint 1 -- inverse velocity kinematics

Eigen::MatrixXd pinv_svd(const Eigen::MatrixXd& J, double tol) {
    // TODO: Eigen::JacobiSVD<Eigen::MatrixXd> svd(J, Eigen::ComputeThinU | Eigen::ComputeThinV);
    //       (= np.linalg.svd(J, full_matrices=False); svd.matrixV() is V, NOT V^T)
    // TODO: invert only the singular values > tol * s(0); J^+ = V diag(1/s) U^T
    // TODO: do NOT call mr::pinv here (the tests compare with it)
    throw std::logic_error("TODO: pinv_svd");
}

Eigen::MatrixXd dls_inverse(const Eigen::MatrixXd& J, double lam) {
    // TODO: lam == 0: return pinv_svd(J)
    // TODO: otherwise J^T (J J^T + lam^2 I)^-1 -- use A.partialPivLu().solve(I), not A.inverse(), if you can
    throw std::logic_error("TODO: dls_inverse");
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> dls_singular_gains(const Eigen::MatrixXd& J, double lam) {
    // TODO: s = Eigen::JacobiSVD<Eigen::MatrixXd>(J).singularValues()  (descending)
    // TODO: gain = s / (s^2 + lam^2); for lam = 0 the gain is 1/s (std::numeric_limits<double>::infinity() where s = 0)
    throw std::logic_error("TODO: dls_singular_gains");
}

double adaptive_lambda(double sigma_min, double eps, double lam_max) {
    // TODO: one if/else
    throw std::logic_error("TODO: adaptive_lambda");
}

// ===================================================================== Checkpoint 2 -- redundancy and null space

Eigen::MatrixXd null_projector(const Eigen::MatrixXd& J) {
    // TODO: I - J^+ J with your pinv_svd (I is n x n, n = J.cols())
    throw std::logic_error("TODO: null_projector");
}

Eigen::MatrixXd weighted_pinv(const Eigen::MatrixXd& J, const Eigen::MatrixXd& W) {
    // TODO: W^-1 J^T (J W^-1 J^T)^-1   (small matrices: .inverse() is fine here)
    throw std::logic_error("TODO: weighted_pinv");
}

Eigen::VectorXd redundant_rates(const Eigen::MatrixXd& J, const Eigen::VectorXd& xdot, const Eigen::VectorXd& qdot0) {
    // TODO: two terms: pinv_svd(J) * xdot and null_projector(J) * qdot0
    throw std::logic_error("TODO: redundant_rates");
}

Eigen::VectorXd joint_centering_gradient(const Eigen::VectorXd& theta, const Eigen::MatrixXd& limits) {
    // TODO: c = mid-range (limits.rowwise().mean()), r = range (limits.col(1) - limits.col(0)),
    //       gradient = (theta - c) / r^2 (element-wise: use .array())
    throw std::logic_error("TODO: joint_centering_gradient");
}

// ===================================================================== Checkpoint 3 -- resolved-rate control

LineTrajectory line_trajectory(const Eigen::Matrix4d& T0, const Eigen::Vector3d& displacement, double speed,
                               double dt) {
    // TODO: tf = |d| / speed, n = round(tf/dt) steps (std::llround), t = LinSpaced(n+1, 0, n dt) with t(n) = tf
    // TODO: X_d(t): orientation R0, position p0 + u * min(speed t, |d|)
    // TODO: V_d row k = (0, 0, 0, R0^T u speed) for all samples except the last (zero)
    throw std::logic_error("TODO: line_trajectory");
}

RateSimResult resolved_rate_sim(const Eigen::MatrixXd& Slist, const Eigen::Matrix4d& M, const Eigen::VectorXd& theta0,
                                const std::vector<Eigen::Matrix4d>& Xd, const Eigen::MatrixXd& Vd, double dt,
                                double kp, double lam, const Eigen::VectorXd& qdot_max, std::optional<double> eps,
                                std::optional<double> lam_max) {
    // TODO: for k in 0..N-1: X = fk_space(...); Xerr = trans_inv(X) * Xd[k]; store theta, errors, sigma_min
    //       (sigma_min = last entry of Eigen::JacobiSVD<Eigen::MatrixXd>(Jb).singularValues())
    // TODO: stop after storing the last sample (no step from it)
    // TODO: V_b = adjoint(Xerr) * V_d(k) + kp * se3_to_vec(matrix_log6(Xerr))   (V_d(k) = Vd.row(k).transpose())
    // TODO: lam_k = adaptive_lambda(...) if eps has a value (*eps, *lam_max), else lam;
    //       qdot = dls_inverse(J_b, lam_k) * V_b
    // TODO: rate limit (qdot_max non-empty; one entry = same limit for all joints):
    //       ratio = max_i |qdot_i| / qdot_max_i; if ratio > 1: qdot /= ratio
    // TODO: theta = theta + qdot * dt
    throw std::logic_error("TODO: resolved_rate_sim");
}

// ===================================================================== Checkpoint 4 -- IK near singularities

IKNewtonResult ik_newton(const Eigen::MatrixXd& Slist, const Eigen::Matrix4d& M, const Eigen::Matrix4d& T_d,
                         const Eigen::VectorXd& theta0, double lam, double eomg, double ev, int max_iter) {
    // TODO: B = body_screws(Slist, M); V_b = se3_to_vec(matrix_log6(trans_inv(fk_space(...)) * T_d))
    // TODO: append (|omega_b|, |v_b|) to err BEFORE testing convergence
    // TODO: theta += dls_inverse(jacobian_body(B, theta), lam) * V_b
    throw std::logic_error("TODO: ik_newton");
}

// ===================================================================== Checkpoint 5 -- hexapod (closed chain)

Eigen::VectorXd hexapod_leg_lengths(const Eigen::Vector3d& p, const Eigen::Matrix3d& R, const Eigen::MatrixXd& A,
                                    const Eigen::MatrixXd& B) {
    // TODO: d_i = p + R b_i - a_i (b_i = B.row(i).transpose()), s_i = |d_i|
    throw std::logic_error("TODO: hexapod_leg_lengths");
}

Eigen::MatrixXd hexapod_inverse_jacobian(const Eigen::Vector3d& p, const Eigen::Matrix3d& R, const Eigen::MatrixXd& A,
                                         const Eigen::MatrixXd& B) {
    // TODO: n_i = d_i / s_i; row i = ( -n_i.cross(a_i), n_i )
    throw std::logic_error("TODO: hexapod_inverse_jacobian");
}

HexapodFKResult hexapod_forward_kinematics(const Eigen::VectorXd& s, const Eigen::MatrixXd& A,
                                           const Eigen::MatrixXd& B, const Eigen::Vector3d& p0,
                                           const Eigen::Matrix3d& R0, double tol, int max_iter) {
    // TODO: T = (R0, p0); loop i = 0..max_iter: ds = s - hexapod_leg_lengths(...); stop if max|ds| < tol
    // TODO: V = hexapod_inverse_jacobian(...).partialPivLu().solve(ds); T = matrix_exp6(vec_to_se3(V)) * T
    //       (space twist: multiply on the LEFT)
    throw std::logic_error("TODO: hexapod_forward_kinematics");
}

// ===================================================================== Checkpoint 6 -- hardware session (sim first)

TrackingReport tracking_report(const Eigen::VectorXd& t, const Eigen::MatrixXd& cmd_deg,
                               const Eigen::MatrixXd& meas_deg, const Eigen::MatrixXd& Slist,
                               const Eigen::Matrix4d& M) {
    // TODO: e = |cmd - meas| per sample and joint; RMS and max of every column
    // TODO: tool points by fk_space of the rows of cmd and meas converted to radians (* PI / 180); distance in mm
    // TODO: peak speed: gradient(meas_deg, t).cwiseAbs().colwise().maxCoeff()
    throw std::logic_error("TODO: tracking_report");
}

}  // namespace lab
