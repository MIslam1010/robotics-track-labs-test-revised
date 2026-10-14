// Auto-checks for SBEG 361 Lab 10 (resolved-rate control, DLS, redundancy, hexapod) -- C++ track.
// Mirrors test_lab.py test by test (same names, same order, same tolerances).
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*pinv*,*dls*"      (run only some tests, like pytest -k)
// Ground truth: mrcpp (the C++ port of modern_robotics: fkin_space, jacobian_body, ikin_body, matrix_exp6),
// finite differences of the hexapod leg lengths, the checkpoint values of the handout, and cpp_cases.json, which
// holds the NumPy reference values (np.linalg.pinv, lstsq, svd) and the random matrices that test_lab.py draws,
// so both languages are checked against the same numbers.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <string>
#include <utility>
#include <vector>

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using sbeg::as;
using sbeg::as_mat;
using sbeg::as_vec;
using sbeg::json;

namespace {
constexpr double PI = 3.14159265358979323846;

const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

double d2r(double deg) { return deg * PI / 180.0; }
double r2d(double rad) { return rad * 180.0 / PI; }

VectorXd vec(std::initializer_list<double> v) {
    VectorXd out(static_cast<Eigen::Index>(v.size()));
    Eigen::Index i = 0;
    for (double x : v) out(i++) = x;
    return out;
}

VectorXd deg2rad(std::initializer_list<double> v) { return vec(v) * (PI / 180.0); }

/// Matrix rank with an absolute singular-value tolerance (np.linalg.matrix_rank(P, tol=...)).
Eigen::Index matrix_rank(const MatrixXd& P, double tol) {
    Eigen::JacobiSVD<MatrixXd> svd(P);
    return (svd.singularValues().array() > tol).count();
}

/// 2-norm condition number (np.linalg.cond).
double cond(const MatrixXd& A) {
    Eigen::JacobiSVD<MatrixXd> svd(A);
    const VectorXd& s = svd.singularValues();
    return s(0) / s(s.size() - 1);
}

/// Largest |entry| of a matrix (0 for an empty one).
double max_abs(const MatrixXd& A) { return A.size() == 0 ? 0.0 : A.cwiseAbs().maxCoeff(); }

struct EodLine {
    lab::Robot robot;
    VectorXd q0;
    lab::LineTrajectory traj;
};

EodLine eod_line() {
    EodLine e;
    e.robot = lab::load_robot("eod_6r_ugv");
    e.q0 = deg2rad({0, -30, 60, 3, 5, 0.0});
    const Matrix4d T0 = mr::fkin_space(e.robot.M, e.robot.Slist, e.q0);
    e.traj = lab::line_trajectory(T0, Vector3d(-0.10, 0, 0), 0.05, 0.02);
    return e;
}

/// The hexapod test pose: p and R = exp([1.0, -0.5, 1.5] deg).
std::pair<Vector3d, Matrix3d> hexapod_pose() {
    const Vector3d p(0.004, -0.003, 0.252);
    const Matrix3d R = mr::matrix_exp3(mr::vec_to_so3(Vector3d(d2r(1.0), d2r(-0.5), d2r(1.5))));
    return {p, R};
}

Matrix4d course_arm_target(const Matrix4d& M, double dx) {
    Matrix4d Td = M;
    Td(0, 3) += dx;
    return Td;
}
}  // namespace

// ---------------------------------------------------------------- Checkpoint 1: pseudo-inverse and DLS
TEST_CASE("test_pinv_svd_matches_numpy") {
    for (const auto& group : cases()["pinv_svd_matches_numpy"]) {
        INFO("shape = " << group["shape"].dump());
        for (const auto& c : group["cases"]) {
            const MatrixXd J = as_mat(c["J"]);
            CHECK_MAT_NEAR(lab::pinv_svd(J), as_mat(c["pinv"]), 1e-10);
            CHECK_MAT_NEAR(lab::pinv_svd(J), mr::pinv(J), 1e-10);
        }
    }
}

TEST_CASE("test_pinv_rank_deficient") {
    // At a singularity J^+ still exists and gives the least-squares, minimum-norm solution.
    const json& c = cases()["pinv_rank_deficient"];
    MatrixXd J(2, 3);
    J << 1.0, 1.0, 0.0,
         2.0, 2.0, 0.0;  // rank 1
    const MatrixXd P = lab::pinv_svd(J);
    CHECK_MAT_NEAR(P, as_mat(c["pinv"]), 1e-12);
    const VectorXd z = vec({1.0, 0.0});
    REQUIRE(P.cols() == 2);
    CHECK_MAT_NEAR(P * z, as_vec(c["lstsq"]), 1e-12);
}

TEST_CASE("test_pinv_fat_is_right_inverse") {
    // MR p. 227: J^+ = J^T (J J^T)^-1 for a fat full-rank J, and J J^+ = I.
    for (const auto& c : cases()["pinv_fat_is_right_inverse"]) {
        const MatrixXd J = as_mat(c["J"]);
        const MatrixXd JJt = J * J.transpose();
        const MatrixXd P = lab::pinv_svd(J);
        CHECK_MAT_NEAR(P, J.transpose() * JJt.inverse(), 1e-10);
        REQUIRE(P.rows() == 3);
        CHECK_MAT_NEAR(J * P, MatrixXd::Identity(2, 2), 1e-10);
    }
}

TEST_CASE("test_dls_two_forms_and_limits") {
    for (const auto& c : cases()["dls_two_forms_and_limits"]) {
        const MatrixXd J = as_mat(c["J"]);
        const double lam = 0.3;
        const Eigen::Index n = J.cols();
        const MatrixXd A = J.transpose() * J + lam * lam * MatrixXd::Identity(n, n);
        const MatrixXd ref = A.partialPivLu().solve(MatrixXd(J.transpose()));
        CHECK_MAT_NEAR(lab::dls_inverse(J, lam), ref, 1e-10);
        CHECK_MAT_NEAR(lab::dls_inverse(J, lam), as_mat(c["dls"]), 1e-10);
        CHECK_MAT_NEAR(lab::dls_inverse(J, 0.0), as_mat(c["pinv"]), 1e-10);
        CHECK_MAT_NEAR(lab::dls_inverse(J, 1e-9), as_mat(c["pinv"]), 1e-6);
    }
}

TEST_CASE("test_dls_minimises_damped_cost") {
    // J^# V minimises |J x - V|^2 + lam^2 |x|^2 (stacked least-squares problem).
    const json& c = cases()["dls_minimises_damped_cost"];
    const MatrixXd J = as_mat(c["J"]);
    const VectorXd V = as_vec(c["V"]);
    const double lam = c["lam"].get<double>();
    const MatrixXd G = lab::dls_inverse(J, lam);
    REQUIRE(G.cols() == 6);
    CHECK_MAT_NEAR(G * V, as_vec(c["x"]), 1e-10);
}

TEST_CASE("test_dls_singular_gains") {
    const Eigen::Matrix2d D = vec({0.5, 0.01}).asDiagonal();
    const MatrixXd J = D;
    const std::pair<VectorXd, VectorXd> sg = lab::dls_singular_gains(J, 0.05);
    CHECK_MAT_NEAR(sg.first, vec({0.5, 0.01}), 1e-12);
    CHECK_MAT_NEAR(sg.second, vec({0.5 / 0.2525, 0.01 / 0.0026}), 1e-10);
    CHECK_MAT_NEAR(sg.second, as_vec(cases()["dls_singular_gains"]["g"]), 1e-12);
    const std::pair<VectorXd, VectorXd> sg0 = lab::dls_singular_gains(J, 0.0);
    CHECK_MAT_NEAR(sg0.second, vec({2.0, 100.0}), 1e-10);
    // the largest possible DLS gain is 1/(2 lam), at sigma = lam
    double gmax = 0.0;
    for (int i = 0; i <= 20000; i += 500) {
        const double x = 1e-4 + (1.0 - 1e-4) * i / 20000.0;  // np.linspace(1e-4, 1, 20001)[::500]
        MatrixXd J1(1, 1);
        J1(0, 0) = x;
        const VectorXd gi = lab::dls_singular_gains(J1, 0.05).second;
        REQUIRE(gi.size() == 1);
        gmax = std::max(gmax, gi(0));
    }
    CHECK(gmax <= 1.0 / (2 * 0.05) + 1e-9);
}

TEST_CASE("test_adaptive_lambda") {
    CHECK(lab::adaptive_lambda(0.06, 0.05, 0.04) == 0.0);
    CHECK(lab::adaptive_lambda(0.05, 0.05, 0.04) == 0.0);
    CHECK_APPROX(lab::adaptive_lambda(0.0, 0.05, 0.04), 0.04);
    CHECK_APPROX(lab::adaptive_lambda(0.03, 0.05, 0.04), 0.032);
}

// ---------------------------------------------------------------- Checkpoint 2: redundancy
TEST_CASE("test_null_projector_properties") {
    for (const auto& c : cases()["null_projector_properties"]) {
        const MatrixXd J = as_mat(c["J"]);
        const MatrixXd P = lab::null_projector(J);
        REQUIRE(P.rows() == J.cols());
        REQUIRE(P.cols() == J.cols());
        CHECK_MAT_NEAR(J * P, MatrixXd::Zero(J.rows(), J.cols()), 1e-10);  // no task motion
        CHECK_MAT_NEAR(P * P, P, 1e-10);                                     // projector
        CHECK_MAT_NEAR(P, P.transpose(), 1e-10);                             // orthogonal projector
        CHECK(matrix_rank(P, 1e-8) == J.cols() - J.rows());
    }
}

TEST_CASE("test_null_projector_workshop_p1") {
    // Workshop 12 P1: planar 3R at (0, 90, -90) deg, null space along (-1, 1, 2).
    MatrixXd J(2, 3);
    J << -0.4, -0.4, 0.0,
         0.75, 0.25, 0.25;
    const Vector3d n = Vector3d(-1.0, 1.0, 2.0) / std::sqrt(6.0);
    CHECK_MAT_NEAR(lab::null_projector(J), n * n.transpose(), 1e-12);
}

TEST_CASE("test_weighted_pinv") {
    const json& c = cases()["weighted_pinv"];
    const MatrixXd J = as_mat(c["J"]);
    const MatrixXd W = vec({4.0, 2.0, 1.0}).asDiagonal();
    const MatrixXd G = lab::weighted_pinv(J, W);
    REQUIRE(G.rows() == 3);
    REQUIRE(G.cols() == 2);
    CHECK_MAT_NEAR(J * G, MatrixXd::Identity(2, 2), 1e-10);
    CHECK_MAT_NEAR(lab::weighted_pinv(J, MatrixXd::Identity(3, 3)), mr::pinv(J), 1e-10);
    // among all solutions, G V has the smallest weighted norm
    const VectorXd V = vec({0.1, -0.05});
    const VectorXd x = G * V;
    const VectorXd nvec = as_vec(c["nvec"]);  // last row of V^T of np.linalg.svd(J)
    for (int i = 0; i <= 10; ++i) {
        const double a = -1.0 + 0.2 * i;
        const VectorXd y = x + a * nvec;
        CHECK(x.dot(W * x) <= y.dot(W * y) + 1e-12);
    }
}

TEST_CASE("test_redundant_rates") {
    for (const auto& c : cases()["redundant_rates"]) {
        const MatrixXd J = as_mat(c["J"]);
        const VectorXd xd = as_vec(c["xd"]), q0 = as_vec(c["q0"]);
        const VectorXd qd = lab::redundant_rates(J, xd, q0);
        REQUIRE(qd.size() == 7);
        CHECK_MAT_NEAR(J * qd, xd, 1e-10);  // task is met
        CHECK_MAT_NEAR(lab::redundant_rates(J, xd, VectorXd::Zero(7)), mr::pinv(J) * xd, 1e-10);
        CHECK_MAT_NEAR(lab::redundant_rates(J, xd, VectorXd::Zero(7)), as_vec(c["pinv_xd"]), 1e-10);
    }
}

TEST_CASE("test_joint_centering_gradient") {
    MatrixXd lim(3, 2);
    lim << -1.0, 1.0,
           0.0, 2.0,
           -2.0, 1.0;
    const VectorXd th = vec({0.5, 0.5, 0.0});
    const VectorXd c = lim.rowwise().mean();
    const VectorXd r = lim.col(1) - lim.col(0);
    auto h = [&](const VectorXd& q) { return 0.5 * ((q - c).array() / r.array()).square().sum(); };
    const VectorXd g = lab::joint_centering_gradient(th, lim);
    VectorXd fd(3);
    for (Eigen::Index i = 0; i < 3; ++i) {
        const VectorXd e = VectorXd::Unit(3, i) * 1e-6;
        fd(i) = (h(th + e) - h(th - e)) / 2e-6;
    }
    CHECK_MAT_NEAR(g, fd, 1e-8);
    CHECK_MAT_NEAR(lab::joint_centering_gradient(c, lim), VectorXd::Zero(3), 1e-15);
}

TEST_CASE("test_srs7r_self_motion") {
    // Null-space motion of the 7R feeding arm moves the elbow but not the spoon.
    const lab::Robot rb = lab::load_robot("srs_7r_feeding");
    VectorXd th = deg2rad({10, 30, 0, -70, 0, 40, 0.0});
    const Matrix4d T0 = mr::fkin_space(rb.M, rb.Slist, th);
    const MatrixXd B = mr::adjoint(mr::trans_inv(rb.M)) * rb.Slist;
    const VectorXd q0 = vec({0, 0, 0.5, 0, 0, 0, 0});
    for (int i = 0; i < 100; ++i) {
        const MatrixXd Jb = mr::jacobian_body(B, th);
        const VectorXd qd = lab::redundant_rates(Jb, VectorXd::Zero(6), q0);
        REQUIRE(qd.size() == 7);
        th = th + qd * 0.01;
    }
    const double err = (mr::fkin_space(rb.M, rb.Slist, th) - T0).norm();
    CHECK(err < 5e-4);                 // Euler drift only (2.1e-4); no task feedback here
    CHECK(std::abs(th(2)) > d2r(5));   // the shoulder roll ("elbow swivel") moved by several degrees
}

// ---------------------------------------------------------------- Checkpoint 3: resolved-rate control
TEST_CASE("test_line_trajectory") {
    const Matrix4d T0 = mr::rp_to_trans(mr::matrix_exp3(mr::vec_to_so3(Vector3d(0.3, -0.2, 0.5))),
                                        Vector3d(0.2, 0.1, 0.3));
    const lab::LineTrajectory tr = lab::line_trajectory(T0, Vector3d(0.03, 0.0, -0.04), 0.02, 0.01);
    REQUIRE(tr.t.size() == 251);
    REQUIRE(tr.Xd.size() == 251);
    REQUIRE(tr.Vd.rows() == 251);
    REQUIRE(tr.Vd.cols() == 6);
    CHECK(tr.t(0) == 0.0);
    CHECK_APPROX(tr.t(250), 2.5);
    CHECK_MAT_NEAR(Vector3d(tr.Xd[250].topRightCorner<3, 1>()), Vector3d(0.23, 0.1, 0.26), 1e-12);
    CHECK_MAT_NEAR(Matrix3d(tr.Xd[250].topLeftCorner<3, 3>()), Matrix3d(T0.topLeftCorner<3, 3>()), 1e-12);
    // body twist: X_d^-1 dX_d/dt by finite difference
    const Matrix4d Xdot = (tr.Xd[11] - tr.Xd[10]) / (tr.t(11) - tr.t(10));
    CHECK_MAT_NEAR(mr::se3_to_vec(mr::trans_inv(tr.Xd[10]) * Xdot), VectorXd(tr.Vd.row(10).transpose()), 1e-10);
    CHECK_MAT_NEAR(VectorXd(tr.Vd.row(10).transpose()), as_vec(cases()["line_trajectory"]["Vd_10"]), 1e-12);
    CHECK_MAT_NEAR(VectorXd(tr.Vd.row(250).transpose()), VectorXd::Zero(6), 1e-15);
}

TEST_CASE("test_resolved_rate_course_arm_line") {
    // Checkpoint 3/6: course arm, 0.06 m along +y_s at 0.02 m/s from q_home.
    const lab::Robot rb = lab::load_robot("course_arm");
    const VectorXd q0 = rb.q_home;
    const Matrix4d T0 = mr::fkin_space(rb.M, rb.Slist, q0);
    const lab::LineTrajectory tr = lab::line_trajectory(T0, Vector3d(0, 0.06, 0), 0.02, 0.02);
    const lab::RateSimResult o0 = lab::resolved_rate_sim(rb.Slist, rb.M, q0, tr.Xd, tr.Vd, 0.02, 0.0);
    const lab::RateSimResult o2 = lab::resolved_rate_sim(rb.Slist, rb.M, q0, tr.Xd, tr.Vd, 0.02, 2.0);
    const Eigen::Index N = static_cast<Eigen::Index>(tr.Xd.size());
    REQUIRE(N == 151);
    REQUIRE(o0.pos_err.size() == N);
    REQUIRE(o2.pos_err.size() == N);
    REQUIRE(o2.theta.rows() == N);
    CHECK_SCALAR_NEAR(o0.pos_err(N - 1), 7.56e-5, 3e-6);   // open-loop Euler drift
    CHECK_SCALAR_NEAR(o2.pos_err(N - 1), 1.50e-5, 1e-6);   // feedback removes most of it
    CHECK_SCALAR_NEAR(r2d(max_abs(o2.qdot)), 11.17, 0.02);
    CHECK_SCALAR_NEAR(o2.sigma_min.minCoeff(), 0.0389, 2e-4);
    const VectorXd th_end = o2.theta.row(N - 1).transpose();
    CHECK_MAT_NEAR(Vector3d(mr::fkin_space(rb.M, rb.Slist, th_end).topRightCorner<3, 1>()),
                   Vector3d(tr.Xd[static_cast<std::size_t>(N - 1)].topRightCorner<3, 1>()), 1e-4);
    // C++ only: the same final joints as the Python reference (well-conditioned line)
    CHECK_MAT_NEAR(th_end, as_vec(cases()["resolved_rate_course_arm_line"]["theta_end_kp2"]), 1e-8);
}

TEST_CASE("test_resolved_rate_reach_boundary_detected") {
    // +x_s line leaves the workspace: sigma_min collapses and the pseudo-inverse rates explode.
    const lab::Robot rb = lab::load_robot("course_arm");
    const VectorXd q0 = rb.q_home;
    const lab::LineTrajectory tr =
        lab::line_trajectory(mr::fkin_space(rb.M, rb.Slist, q0), Vector3d(0.06, 0, 0), 0.02, 0.02);
    const lab::RateSimResult o = lab::resolved_rate_sim(rb.Slist, rb.M, q0, tr.Xd, tr.Vd, 0.02, 0.0);
    REQUIRE(o.sigma_min.size() > 0);
    CHECK(o.sigma_min.minCoeff() < 1e-3);
    CHECK(r2d(max_abs(o.qdot)) > 1000);
}

TEST_CASE("test_resolved_rate_eod_wrist_singularity") {
    const EodLine e = eod_line();
    const MatrixXd& S = e.robot.Slist;
    const Matrix4d& M = e.robot.M;
    const auto& Xd = e.traj.Xd;
    const MatrixXd& Vd = e.traj.Vd;
    const lab::RateSimResult pinv = lab::resolved_rate_sim(S, M, e.q0, Xd, Vd, 0.02, 2.0, 0.0);
    const lab::RateSimResult sat =
        lab::resolved_rate_sim(S, M, e.q0, Xd, Vd, 0.02, 2.0, 0.0, VectorXd::Constant(1, d2r(30)));
    const lab::RateSimResult dls = lab::resolved_rate_sim(S, M, e.q0, Xd, Vd, 0.02, 2.0, 0.02);
    const lab::RateSimResult ada = lab::resolved_rate_sim(S, M, e.q0, Xd, Vd, 0.02, 2.0, 0.0, VectorXd(), 0.04, 0.04);
    REQUIRE(pinv.qdot.cols() == 6);
    REQUIRE(sat.pos_err.size() > 0);
    REQUIRE(ada.qdot.cols() == 6);
    REQUIRE(ada.lam.size() > 0);
    CHECK_SCALAR_NEAR(r2d(max_abs(pinv.qdot.col(3))), 944.2, 1.0);
    CHECK(r2d(max_abs(sat.qdot)) <= 30 + 1e-9);
    CHECK_SCALAR_NEAR(sat.pos_err(sat.pos_err.size() - 1) * 1000, 34.3, 0.2);
    CHECK_SCALAR_NEAR(r2d(max_abs(dls.qdot)), 14.61, 0.05);
    CHECK_SCALAR_NEAR(dls.pos_err.maxCoeff() * 1000, 2.72, 0.02);
    const double ada_wrist = std::max(max_abs(ada.qdot.col(3)), max_abs(ada.qdot.col(5)));
    CHECK_SCALAR_NEAR(r2d(ada_wrist), 6.48, 0.05);
    CHECK_SCALAR_NEAR(ada.pos_err.maxCoeff() * 1000, 2.75, 0.02);
    CHECK(ada.lam(0) == 0.0);  // damping only near the singularity
    CHECK(ada.lam.maxCoeff() > 0.039);
}

// ---------------------------------------------------------------- Checkpoint 4: IK near singularities
TEST_CASE("test_ik_newton_matches_mr_reachable") {
    const lab::Robot rb = lab::load_robot("course_arm");
    const Matrix4d Td = course_arm_target(rb.M, 0.05);
    const VectorXd q0 = vec({0, 0.3, -0.6, 0.3, 0.0, 0});
    const lab::IKNewtonResult r = lab::ik_newton(rb.Slist, rb.M, Td, q0, 0.0);
    const MatrixXd B = mr::adjoint(mr::trans_inv(rb.M)) * rb.Slist;
    const mr::IKResult ref = mr::ikin_body(B, rb.M, Td, q0, 1e-6, 1e-6);
    CHECK(r.success);
    CHECK(ref.success);
    CHECK(r.iterations == 4);
    CHECK(r.err.size() == static_cast<std::size_t>(r.iterations) + 1);
    REQUIRE(r.theta.size() == 6);
    CHECK_MAT_NEAR(mr::fkin_space(rb.M, rb.Slist, r.theta), Td, 1e-6);
    CHECK_MAT_NEAR(r.theta, ref.thetalist, 1e-6);
    CHECK_MAT_NEAR(r.theta, as_vec(cases()["ik_newton_matches_mr_reachable"]["theta"]), 1e-6);
}

TEST_CASE("test_ik_newton_damping_slows_reachable") {
    const lab::Robot rb = lab::load_robot("course_arm");
    const Matrix4d Td = course_arm_target(rb.M, 0.05);
    const VectorXd q0 = vec({0, 0.3, -0.6, 0.3, 0.0, 0});
    const std::vector<double> lams = {0.0, 0.01, 0.02, 0.05};
    const std::vector<int> expected = {4, 6, 10, 37};
    for (std::size_t i = 0; i < lams.size(); ++i) {
        INFO("lam = " << lams[i]);
        CHECK(lab::ik_newton(rb.Slist, rb.M, Td, q0, lams[i]).iterations == expected[i]);
    }
}

TEST_CASE("test_ik_newton_unreachable") {
    // Target 0.08 m beyond home along x_s: pseudo-inverse steps jump about, DLS settles at the
    // closest reachable pose (elbow straight) with a 14.4 mm residual.
    const lab::Robot rb = lab::load_robot("course_arm");
    const Matrix4d Td = course_arm_target(rb.M, 0.08);
    const VectorXd q0 = vec({0, 0.3, -0.6, 0.3, 0.0, 0});
    const lab::IKNewtonResult r0 = lab::ik_newton(rb.Slist, rb.M, Td, q0, 0.0);
    const lab::IKNewtonResult r2 = lab::ik_newton(rb.Slist, rb.M, Td, q0, 0.02);
    CHECK(!r0.success);
    CHECK(!r2.success);
    REQUIRE(!r0.err.empty());
    REQUIRE(r2.err.size() >= 2);
    double emax = 0.0;
    for (const auto& e : r0.err) emax = std::max(emax, e.second);
    // The pseudo-inverse iterates are chaotic: their peak depends on rounding (Python/NumPy: 0.460 m at iteration
    // 95; Eigen: about 0.39 m). The C++ check asks for jumps far beyond the 14 mm DLS residual, and the first
    // 8 iterates (before rounding differences grow; they include the 0.201 m jump at iteration 3) must equal
    // the Python reference.
    CHECK(emax > 0.1);
    const VectorXd first8 = as_vec(cases()["ik_newton_unreachable"]["v_err_first8"]);
    REQUIRE(r0.err.size() >= 8);
    for (Eigen::Index i = 0; i < 8; ++i) {
        INFO("iterate " << i);
        CHECK_SCALAR_NEAR(r0.err[static_cast<std::size_t>(i)].second, first8(i), 1e-6);
    }
    const std::size_t m = r2.err.size();
    CHECK_SCALAR_NEAR(r2.err[m - 1].second, 0.0144, 2e-4);
    CHECK(std::abs(r2.err[m - 1].second - r2.err[m - 2].second) < 1e-6);
    REQUIRE(r2.theta.size() == 6);
    CHECK(std::abs(r2.theta(2)) < 1e-3);
}

// ---------------------------------------------------------------- Checkpoint 5: hexapod
TEST_CASE("test_hexapod_leg_lengths_home") {
    const lab::Hexapod hx = lab::load_hexapod();
    const VectorXd s = lab::hexapod_leg_lengths(Vector3d(0, 0, 0.25), Matrix3d::Identity(), hx.A, hx.B);
    CHECK_MAT_NEAR(s, VectorXd::Constant(6, 0.297619), 1e-6);
    CHECK_MAT_NEAR(s, as_vec(cases()["hexapod"]["s_home"]), 1e-12);
}

TEST_CASE("test_hexapod_inverse_jacobian_fd") {
    // sdot = J^-1 V_s checked by differentiating s along a space-frame twist: T(e) = e^[V]e T.
    const lab::Hexapod hx = lab::load_hexapod();
    const Vector3d p = hexapod_pose().first;
    const Matrix3d R = hexapod_pose().second;
    const Matrix4d T = mr::rp_to_trans(R, p);
    const MatrixXd Ji = lab::hexapod_inverse_jacobian(p, R, hx.A, hx.B);
    REQUIRE(Ji.rows() == 6);
    REQUIRE(Ji.cols() == 6);
    const double h = 1e-6;
    for (Eigen::Index i = 0; i < 6; ++i) {
        INFO("twist direction " << i);
        const sbeg::Vector6d V = sbeg::Vector6d::Unit(i);
        const Matrix4d Tp = mr::matrix_exp6(mr::vec_to_se3(V * h)) * T;
        const Matrix4d Tm = mr::matrix_exp6(mr::vec_to_se3(-V * h)) * T;
        const VectorXd sp = lab::hexapod_leg_lengths(Tp.topRightCorner<3, 1>(), Tp.topLeftCorner<3, 3>(), hx.A, hx.B);
        const VectorXd sm = lab::hexapod_leg_lengths(Tm.topRightCorner<3, 1>(), Tm.topLeftCorner<3, 3>(), hx.A, hx.B);
        CHECK_MAT_NEAR(Ji * V, (sp - sm) / (2 * h), 1e-8);
    }
}

TEST_CASE("test_hexapod_jacobian_home_values") {
    const lab::Hexapod hx = lab::load_hexapod();
    const MatrixXd Ji = lab::hexapod_inverse_jacobian(Vector3d(0, 0, 0.25), Matrix3d::Identity(), hx.A, hx.B);
    REQUIRE(Ji.rows() == 6);
    REQUIRE(Ji.cols() == 6);
    CHECK_MAT_NEAR(VectorXd(Ji.row(0).transpose()), vec({-0.04376, -0.24817, -0.10080, -0.47790, -0.25692, 0.84000}),
                   1e-5);
    CHECK_SCALAR_NEAR(cond(Ji), 8.333, 1e-3);
}

TEST_CASE("test_hexapod_forward_kinematics") {
    const lab::Hexapod hx = lab::load_hexapod();
    const Vector3d p = hexapod_pose().first;
    const Matrix3d R = hexapod_pose().second;
    const VectorXd s = lab::hexapod_leg_lengths(p, R, hx.A, hx.B);
    const lab::HexapodFKResult fk =
        lab::hexapod_forward_kinematics(s, hx.A, hx.B, Vector3d(0, 0, 0.25), Matrix3d::Identity());
    CHECK_MAT_NEAR(fk.p, p, 1e-10);
    CHECK_MAT_NEAR(fk.R, R, 1e-10);
    CHECK(fk.iterations >= 2);
    CHECK(fk.iterations <= 6);
}

// ---------------------------------------------------------------- Checkpoint 6: hardware log
TEST_CASE("test_tracking_report_example_log") {
    const lab::Robot rb = lab::load_robot("course_arm");
    const lab::JointLog log = lab::read_joint_log(sbeg::lab_path("../data/hw_log_example.csv"));
    const lab::TrackingReport r = lab::tracking_report(log.t, log.cmd_deg, log.meas_deg, rb.Slist, rb.M);
    REQUIRE(r.joint_rms_deg.size() == 6);
    REQUIRE(r.joint_max_deg.size() == 6);
    REQUIRE(r.peak_speed_deg_s.size() == 6);
    CHECK_SCALAR_NEAR(r.joint_rms_deg.maxCoeff(), 0.362, 0.002);
    CHECK_SCALAR_NEAR(r.joint_max_deg.maxCoeff(), 0.761, 0.002);
    CHECK_SCALAR_NEAR(r.task_rms_mm, 1.21, 0.01);
    CHECK_SCALAR_NEAR(r.task_max_mm, 2.02, 0.01);
    CHECK_SCALAR_NEAR(r.peak_speed_deg_s.maxCoeff(), 15.38, 0.01);
    // C++ only: every entry equals the Python reference
    const json& c = cases()["tracking_report_example_log"];
    CHECK_MAT_NEAR(r.joint_rms_deg, as_vec(c["joint_rms_deg"]), 1e-9);
    CHECK_MAT_NEAR(r.peak_speed_deg_s, as_vec(c["peak_speed_deg_s"]), 1e-9);
}

TEST_CASE("test_tracking_report_zero_for_perfect_log") {
    const lab::Robot rb = lab::load_robot("course_arm");
    const VectorXd t = VectorXd::LinSpaced(11, 0.0, 1.0);
    const MatrixXd cmd = t * Eigen::RowVectorXd::Ones(6) * 10.0;
    const lab::TrackingReport r = lab::tracking_report(t, cmd, cmd, rb.Slist, rb.M);
    REQUIRE(r.joint_max_deg.size() == 6);
    CHECK(r.task_max_mm == 0.0);
    CHECK((r.joint_max_deg.array() == 0.0).all());
    CHECK_MAT_NEAR(r.peak_speed_deg_s, VectorXd::Constant(6, 10.0), 1e-9);
}
