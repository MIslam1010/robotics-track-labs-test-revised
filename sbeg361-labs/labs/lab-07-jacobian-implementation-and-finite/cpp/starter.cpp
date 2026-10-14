// SBEG 361 Lab 7 -- Space, body and analytic Jacobians; finite-difference verification (C++ STARTER).
//
// Fill in every function marked  // TODO  (13 functions: replace the `throw std::logic_error("TODO: ...")` line).
// Do not change function names or signatures (they are declared in lab.hpp). Build and test from the cpp folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
// or from your Lab code folder:   python run_lab.py 7 --cpp
//
// You may NOT call mrcpp (mr::jacobian_space, mr::jacobian_body, mr::adjoint, ...) inside this file -- it is what
// the tests compare against. The helpers marked "complete" (from your Labs 2-5 library) may be used freely; replace
// them by your own versions if you prefer. Notation: Lynch & Park, *Modern Robotics* (MR), Section 5.1 (see lab.hpp).
//
// Eigen reminders:  J.col(i) (column i, read or write), J.topRows(3) / J.bottomRows(3) (J_w / J_v of a 6 x n
//   Jacobian), T.topLeftCorner<3, 3>() (R), T.topRightCorner<3, 1>() (p), A * B (matrix product),
//   A.partialPivLu().solve(B) (= np.linalg.solve(A, B)), (A - B).cwiseAbs().maxCoeff() (= np.max(np.abs(A - B))),
//   Eigen::VectorXd::Zero(n) (= np.zeros(n)), Eigen::Index for loop counters.
//   A 6-vector for vec_to_se3: `const sbeg::Vector6d S = Slist.col(i) * theta(i);`
//   Calling the forward kinematics you are given: `const Eigen::Matrix4d T = fk(theta);`
#include "lab.hpp"

#include <sbeg/labio.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lab {

namespace {
constexpr double PI = 3.14159265358979323846;
}  // namespace

// ===================================================================== helpers (complete; from Labs 2-5)

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
    if (th < TOL) return Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d K = so3mat / th;
    return Eigen::Matrix3d::Identity() + std::sin(th) * K + (1.0 - std::cos(th)) * K * K;
}

Eigen::Matrix3d matrix_log3(const Eigen::Matrix3d& R) {
    const double c = (R.trace() - 1.0) / 2.0;
    if (c >= 1.0) return Eigen::Matrix3d::Zero();
    if (c <= -1.0) {
        Eigen::Vector3d w;
        if (std::abs(1.0 + R(2, 2)) > TOL)
            w = Eigen::Vector3d(R(0, 2), R(1, 2), 1.0 + R(2, 2)) / std::sqrt(2.0 * (1.0 + R(2, 2)));
        else if (std::abs(1.0 + R(1, 1)) > TOL)
            w = Eigen::Vector3d(R(0, 1), 1.0 + R(1, 1), R(2, 1)) / std::sqrt(2.0 * (1.0 + R(1, 1)));
        else
            w = Eigen::Vector3d(1.0 + R(0, 0), R(1, 0), R(2, 0)) / std::sqrt(2.0 * (1.0 + R(0, 0)));
        return vec_to_so3(PI * w);
    }
    const double th = std::acos(c);
    return th / (2.0 * std::sin(th)) * (R - R.transpose());
}

Eigen::Matrix4d matrix_exp6(const Eigen::Matrix4d& se3mat) {
    const Eigen::Matrix3d W = se3mat.topLeftCorner<3, 3>();
    const Eigen::Vector3d v = se3mat.topRightCorner<3, 1>();
    const double th = so3_to_vec(W).norm();
    Eigen::Matrix4d out = Eigen::Matrix4d::Identity();
    if (th < TOL) {
        out.topRightCorner<3, 1>() = v;
        return out;
    }
    const Eigen::Matrix3d K = W / th;
    out.topLeftCorner<3, 3>() = matrix_exp3(W);
    const Eigen::Matrix3d G = Eigen::Matrix3d::Identity() * th + (1.0 - std::cos(th)) * K + (th - std::sin(th)) * K * K;
    out.topRightCorner<3, 1>() = G * (v / th);
    return out;
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
    const double th = std::acos(std::min(1.0, std::max(-1.0, (R.trace() - 1.0) / 2.0)));
    const Eigen::Matrix3d Ginv =
        Eigen::Matrix3d::Identity() - W / 2.0 + (1.0 / th - 1.0 / std::tan(th / 2.0) / 2.0) * (W * W) / th;
    out.topLeftCorner<3, 3>() = W;
    out.topRightCorner<3, 1>() = Ginv * p;
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

Eigen::Matrix4d fk_body(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Blist, const Eigen::VectorXd& theta) {
    Eigen::Matrix4d T = M;
    for (Eigen::Index i = 0; i < theta.size(); ++i) {
        const sbeg::Vector6d B = Blist.col(i) * theta(i);
        T = (T * matrix_exp6(vec_to_se3(B))).eval();
    }
    return T;
}

Eigen::MatrixXd body_screws(const Eigen::MatrixXd& Slist, const Eigen::Matrix4d& M) {
    return adjoint(trans_inv(M)) * Slist;
}

Robot load_robot(const std::string& name, const std::string& path) {
    const sbeg::json all = sbeg::load_json(path.empty() ? sbeg::lab_path("../data/robots.json") : path);
    if (!all.contains(name)) throw std::invalid_argument("load_robot: no robot named '" + name + "'");
    const sbeg::json& d = all.at(name);
    Robot r;
    r.Slist = sbeg::as_mat(d.at("Slist"));
    r.M = sbeg::as<Eigen::Matrix4d>(d.at("M"));
    r.name = d.at("name").get<std::string>();
    r.joint_types = d.at("joint_types").get<std::string>();
    // limits, qdot_max, q_home may be null in the file (e.g. qdot_max of "rrp"): left empty then.
    if (d.contains("limits") && !d.at("limits").is_null()) r.limits = sbeg::as_mat(d.at("limits"));
    if (d.contains("qdot_max") && !d.at("qdot_max").is_null()) r.qdot_max = sbeg::as_vec(d.at("qdot_max"));
    if (d.contains("q_home") && !d.at("q_home").is_null()) r.q_home = sbeg::as_vec(d.at("q_home"));
    return r;
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> load_teleop_csv(const std::string& path) {
    const sbeg::CsvTable tab = sbeg::read_csv(path.empty() ? sbeg::lab_path("../data/teleop_needle.csv") : path);
    const Eigen::Index n = tab.data.cols();
    return {tab.data.col(0), tab.data.rightCols(n - 1)};
}

double fit_slope(const Eigen::VectorXd& hs, const Eigen::VectorXd& errs) {
    if (hs.size() != errs.size() || hs.size() < 2) throw std::invalid_argument("fit_slope: need >= 2 pairs");
    const Eigen::ArrayXd x = hs.array().log(), y = errs.array().log();
    const Eigen::ArrayXd dx = x - x.mean(), dy = y - y.mean();
    return (dx * dy).sum() / (dx * dx).sum();
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

// ===================================================================== Checkpoint 1 -- geometric Jacobians

Eigen::MatrixXd jacobian_space(const Eigen::MatrixXd& Slist, const Eigen::VectorXd& theta) {
    // TODO: Start with Js = copy of Slist; accumulate T = T * exp([S_(i-1)] th_(i-1)); column i = Ad_T S_i.
    throw std::logic_error("TODO: jacobian_space");
}

Eigen::MatrixXd jacobian_body(const Eigen::MatrixXd& Blist, const Eigen::VectorXd& theta) {
    // TODO: Mirror image of jacobian_space: walk from joint n-1 down to 1, accumulating exp(-[B_(i+1)] th_(i+1)).
    // (Use a signed loop counter: for (Eigen::Index i = n - 2; i >= 0; --i).)
    throw std::logic_error("TODO: jacobian_body");
}

Eigen::MatrixXd body_from_space(const Eigen::MatrixXd& Js, const Eigen::Matrix4d& T_sb) {
    // TODO: One line with adjoint() and trans_inv().
    throw std::logic_error("TODO: body_from_space");
}

Eigen::MatrixXd space_from_body(const Eigen::MatrixXd& Jb, const Eigen::Matrix4d& T_sb) {
    // TODO: One line with adjoint().
    throw std::logic_error("TODO: space_from_body");
}

Eigen::MatrixXd point_jacobian(const Eigen::MatrixXd& Js, const Eigen::Vector3d& p) {
    // TODO: pdot = v_s + w_s x p = v_s - [p] w_s; use vec_to_so3(p), Js.topRows(3) and Js.bottomRows(3).
    throw std::logic_error("TODO: point_jacobian");
}

// ===================================================================== Checkpoint 2 -- finite-difference checks

Eigen::MatrixXd fd_jacobian_space(const FkFunction& fk, const Eigen::VectorXd& theta, double h,
                                  const std::string& scheme) {
    // TODO: Loop over joints; build dT/dtheta_i with the chosen scheme; multiply by T(theta)^-1 on the RIGHT.
    // throw std::invalid_argument("scheme must be 'forward' or 'central'") for any other scheme.
    throw std::logic_error("TODO: fd_jacobian_space");
}

Eigen::MatrixXd fd_jacobian_body(const FkFunction& fk, const Eigen::VectorXd& theta, double h,
                                 const std::string& scheme) {
    // TODO: As fd_jacobian_space but T(theta)^-1 on the LEFT.
    throw std::logic_error("TODO: fd_jacobian_body");
}

Eigen::MatrixXd log_jacobian_space(const FkFunction& fk, const Eigen::VectorXd& theta, double h) {
    // TODO: Use matrix_log6 and se3_to_vec; divide by h.
    throw std::logic_error("TODO: log_jacobian_space");
}

Eigen::VectorXd fd_error_study(const FkFunction& fk, const Eigen::MatrixXd& J_exact, const Eigen::VectorXd& theta,
                               const Eigen::VectorXd& hs, const std::string& scheme) {
    // TODO: One loop over hs calling fd_jacobian_space; entry k = max-abs difference from J_exact.
    throw std::logic_error("TODO: fd_error_study");
}

// ===================================================================== Checkpoint 3 -- analytic Jacobian

Eigen::Matrix3d a_matrix(const Eigen::Vector3d& r) {
    // TODO: Return I for |r| < TOL; otherwise the MR Example 5.5 formula.
    throw std::logic_error("TODO: a_matrix");
}

Eigen::MatrixXd analytic_jacobian(const Eigen::MatrixXd& Jb, const Eigen::Matrix4d& T_sb) {
    // TODO: r = so3_to_vec(matrix_log3(R_sb)); top block A(r)^-1 J_w (use .partialPivLu().solve(...)),
    // bottom block R_sb J_v.
    throw std::logic_error("TODO: analytic_jacobian");
}

sbeg::Vector6d minimal_coords(const Eigen::Matrix4d& T) {
    // Given.
    const Eigen::Matrix3d R = T.topLeftCorner<3, 3>();
    sbeg::Vector6d q;
    q << so3_to_vec(matrix_log3(R)), T.topRightCorner<3, 1>();
    return q;
}

Eigen::MatrixXd fd_analytic_jacobian(const FkFunction& fk, const Eigen::VectorXd& theta, double h) {
    // TODO: Central differences of minimal_coords(fk(theta +- h e_i)).
    throw std::logic_error("TODO: fd_analytic_jacobian");
}

// ===================================================================== Checkpoint 4 -- teleoperation record

Eigen::MatrixXd tip_velocity_series(const Eigen::MatrixXd& Slist, const Eigen::Matrix4d& M, const Eigen::MatrixXd& Q,
                                    const Eigen::MatrixXd& Qdot) {
    // TODO: For each sample k (row of Q): p from fk_space, then point_jacobian(jacobian_space(...), p) * qdot;
    // row k of the N x 3 result.  (Q.row(k).transpose() is the joint vector of sample k.)
    throw std::logic_error("TODO: tip_velocity_series");
}

}  // namespace lab
