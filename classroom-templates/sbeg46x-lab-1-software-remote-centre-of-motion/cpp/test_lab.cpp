// Auto-checks for SBEG 46X Lab 1 (software RCM resolved-rate control of the course arm with a laparoscope) -- C++
// track.  Mirrors test_lab.py test by test.
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*tool_pose*,*geometry*"      (run only some tests, like pytest -k)
// Reference values come from mrcpp (the C++ port of modern_robotics: fkin_space, jacobian_space, jacobian_body),
// finite differences of the forward kinematics, closed-form geometry and cpp_cases.json, which holds the numbers of
// the Python reference (including the random configurations that test_lab.py draws), so both languages are checked
// against the same values.
// Notation: Modern Robotics -- V = (w, v); point velocity pdot = v_s + w_s x p (MR Eq. 3.73).
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>
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

const lab::Robot& robot() {
    static const lab::Robot R = lab::course_arm_kinematics();
    return R;
}

const lab::Task& task() {
    static const lab::Task T = lab::load_task();
    return T;
}

/// THETAS of test_lab.py: theta0 and 4 random configurations around it (rng 461, stored in cpp_cases.json).
std::vector<VectorXd> thetas() {
    std::vector<VectorXd> out;
    for (const auto& c : cases()["tool_pose_matches_modern_robotics"]) out.push_back(as_vec(c["theta"]));
    return out;
}

Matrix4d fk_ref(const VectorXd& theta) { return mr::fkin_space(robot().M, robot().Slist, theta); }

VectorXd vec(std::initializer_list<double> v) {
    VectorXd x(static_cast<Eigen::Index>(v.size()));
    Eigen::Index i = 0;
    for (double e : v) x(i++) = e;
    return x;
}

/// numpy.linalg.matrix_rank(A, tol): number of singular values > tol.
int matrix_rank(const MatrixXd& A, double tol) {
    const Eigen::JacobiSVD<MatrixXd> svd(A);
    return static_cast<int>((svd.singularValues().array() > tol).count());
}

/// Last right-singular vector of A (numpy: np.linalg.svd(A)[2][-1]); for a 5 x 6 A of rank 5 it spans the null space.
VectorXd null_vector(const MatrixXd& A) {
    const Eigen::JacobiSVD<MatrixXd> svd(A, Eigen::ComputeFullV);
    return svd.matrixV().col(svd.matrixV().cols() - 1);
}

/// The simulation of test_lab.py: lab::simulate with the task's dt and gains.
lab::SimLog sim(const std::string& controller) {
    const lab::Task& t = task();
    return lab::simulate(robot(), t, controller, t.dt, t.gains.kp, t.gains.k_rcm, t.gains.lam);
}

int steps_total() {
    double T = 0.0;
    for (const auto& s : task().path) T += s.T;
    return static_cast<int>(std::lround(T / task().dt));
}
}  // namespace

// ---------------------------------------------------------------- Checkpoint 0: your SBEG 361 library
TEST_CASE("test_361_library_matches_modern_robotics") {
    // The kinematics backend (mrcpp in the given code, or your own SBEG 361 C++ functions) against mrcpp.
    // (In Python this test is skipped on the modern_robotics fallback; in C++ it always runs.)
    INFO("kinematics backend: " << lab::kinematics_backend());
    for (const VectorXd& th : thetas()) {
        CHECK_MAT_NEAR(lab::tool_pose(robot(), th), fk_ref(th), 1e-9);
        CHECK_MAT_NEAR(lab::jacobian_space(robot().Slist, th), mr::jacobian_space(robot().Slist, th), 1e-9);
    }
}

// ---------------------------------------------------------------- Part A
TEST_CASE("test_tool_pose_matches_modern_robotics") {
    for (const auto& c : cases()["tool_pose_matches_modern_robotics"]) {
        const VectorXd th = as_vec(c["theta"]);
        INFO("theta = " << th.transpose());
        CHECK_MAT_NEAR(lab::tool_pose(robot(), th), fk_ref(th), 1e-12);
        CHECK_MAT_NEAR(lab::tool_pose(robot(), th), as<Matrix4d>(c["T"]), 1e-12);
    }
}

TEST_CASE("test_start_pose_shaft_through_port") {
    const Matrix4d T = lab::tool_pose(robot(), task().theta0);
    const auto [p_tip, z] = lab::shaft(T);
    CHECK_MAT_NEAR(p_tip, (T.block<3, 1>(0, 3)), 1e-12);  // Python: assert_allclose (rtol 1e-7) on an exact copy
    CHECK_MAT_NEAR(z, (T.block<3, 1>(0, 2)), 1e-12);
    CHECK_MAT_NEAR(z, Vector3d(0, 0, -1), 1e-5);          // shaft straight down at the start
    const lab::RcmGeometry geo = lab::rcm_geometry(T, task().p_port);
    CHECK_SCALAR_NEAR(geo.depth, 0.03, 1e-6);             // 30 mm beyond the port
    CHECK(geo.dist < 1e-6);
}

TEST_CASE("test_rcm_geometry_by_hand") {
    // tip 0.064 m beyond the hole along z = (0.6, 0, -0.8): Practice Problem 4 numbers
    Matrix4d T = Matrix4d::Identity();
    const Vector3d z(0.6, 0.0, -0.8), x(0.0, 1.0, 0.0);
    T.block<3, 1>(0, 0) = x;
    T.block<3, 1>(0, 1) = z.cross(x);
    T.block<3, 1>(0, 2) = z;
    T.block<3, 1>(0, 3) = Vector3d(0.62, 0.05, 0.28);
    const lab::RcmGeometry geo = lab::rcm_geometry(T, Vector3d(0.58, 0.06, 0.33));
    CHECK_SCALAR_NEAR(geo.depth, 0.064, 1e-12);
    CHECK_MAT_NEAR(geo.p_c, Vector3d(0.5816, 0.05, 0.3312), 1e-12);
    CHECK_MAT_NEAR(geo.e, Vector3d(-0.0016, 0.01, -0.0012), 1e-12);
    CHECK_SCALAR_NEAR(geo.dist, std::sqrt(0.0016 * 0.0016 + 0.01 * 0.01 + 0.0012 * 0.0012), 1e-12);
    CHECK(std::abs(geo.e.dot(z)) < 1e-12);                // error is perpendicular to the shaft
    const json& c = cases()["rcm_geometry_by_hand"];
    CHECK_MAT_NEAR(geo.p_c, as<Vector3d>(c["p_c"]), 1e-12);
    CHECK_MAT_NEAR(geo.e, as<Vector3d>(c["e"]), 1e-12);
}

TEST_CASE("test_perp_basis") {
    for (const auto& c : cases()["perp_basis"]) {
        const Vector3d z = as<Vector3d>(c["z"]);
        INFO("z = " << z.transpose());
        const lab::Matrix32d N = lab::perp_basis(z);
        CHECK_MAT_NEAR(N.transpose() * N, Eigen::Matrix2d::Identity(), 1e-12);
        CHECK_MAT_NEAR(N.transpose() * z, Eigen::Vector2d::Zero(), 1e-12);
        Matrix3d B;
        B << N, z;
        CHECK_SCALAR_NEAR(B.determinant(), 1.0, 1e-12);
    }
}

// ---------------------------------------------------------------- Part B
TEST_CASE("test_point_jacobian_tip_vs_body_jacobian") {
    const MatrixXd Blist = mr::adjoint(mr::trans_inv(robot().M)) * robot().Slist;
    for (const VectorXd& th : thetas()) {
        INFO("theta = " << th.transpose());
        const Matrix4d T = fk_ref(th);
        const MatrixXd Js = mr::jacobian_space(robot().Slist, th);
        const MatrixXd Jb = mr::jacobian_body(Blist, th);
        // tip = origin of {b}: its velocity in {s} is R_sb v_b
        const MatrixXd expected = T.block<3, 3>(0, 0) * Jb.bottomRows(3);
        CHECK_MAT_NEAR(lab::point_jacobian(Js, T.block<3, 1>(0, 3)), expected, 1e-12);
    }
}

TEST_CASE("test_point_jacobian_finite_difference") {
    const Vector3d p_body(0.0, 0.0, -0.05);  // a shaft point 5 cm behind the tip, in {b}
    for (const VectorXd& th : thetas()) {
        INFO("theta = " << th.transpose());
        const Matrix4d T = fk_ref(th);
        const Vector3d p = T.block<3, 3>(0, 0) * p_body + T.block<3, 1>(0, 3);
        const MatrixXd Js = mr::jacobian_space(robot().Slist, th);
        const double eps = 1e-6;
        MatrixXd J_fd(3, 6);
        for (int i = 0; i < 6; ++i) {
            VectorXd d = VectorXd::Zero(6);
            d(i) = eps;
            const Matrix4d Tp = fk_ref(th + d), Tm = fk_ref(th - d);
            J_fd.col(i) = ((Tp.block<3, 3>(0, 0) * p_body + Tp.block<3, 1>(0, 3)) -
                           (Tm.block<3, 3>(0, 0) * p_body + Tm.block<3, 1>(0, 3))) / (2 * eps);
        }
        CHECK_MAT_NEAR(lab::point_jacobian(Js, p), J_fd, 1e-8);
    }
}

TEST_CASE("test_rcm_jacobian_shape_rank_and_rows") {
    const VectorXd th = task().theta0;
    const auto [A, info] = lab::rcm_jacobian(robot(), th, task().p_port);
    REQUIRE(A.rows() == 5);
    REQUIRE(A.cols() == 6);
    CHECK(matrix_rank(A, 1e-6) == 5);
    const MatrixXd Js = mr::jacobian_space(robot().Slist, th);
    const Matrix4d T = fk_ref(th);
    CHECK_MAT_NEAR(A.topRows(3), lab::point_jacobian(Js, T.block<3, 1>(0, 3)), 1e-12);
    CHECK_MAT_NEAR(A.topRows(3), as_mat(cases()["rcm_jacobian_shape_rank_and_rows"]["A_tip"]), 1e-12);
    CHECK_MAT_NEAR(info.T, T, 1e-12);
    const lab::Matrix32d N = info.N;
    CHECK_MAT_NEAR((N.transpose() * T.block<3, 1>(0, 2)), Eigen::Vector2d::Zero(), 1e-12);
    // the null space of A: the tip and the port point do not move sideways
    const VectorXd v = null_vector(A);
    const Vector3d p_c = info.geometry.p_c;
    CHECK_MAT_NEAR(lab::point_jacobian(Js, T.block<3, 1>(0, 3)) * v, Vector3d::Zero(), 1e-10);
    const Vector3d zz = T.block<3, 1>(0, 2);
    const Vector3d lat = (Matrix3d::Identity() - zz * zz.transpose()) * lab::point_jacobian(Js, p_c) * v;
    CHECK_MAT_NEAR(lat, Vector3d::Zero(), 1e-10);
}

TEST_CASE("test_rcm_rows_predict_lateral_port_motion") {
    // Rows 4-5 of A predict the first-order change of N^T e when the joints move (e = p_port - p_c).
    const VectorXd th = task().theta0 + vec({0.05, -0.03, 0.04, 0.02, -0.05, 0.1});
    const auto [A, info] = lab::rcm_jacobian(robot(), th, task().p_port);
    REQUIRE(A.rows() == 5);
    REQUIRE(A.cols() == 6);
    const lab::Matrix32d N = info.N;
    const VectorXd qdot = vec({0.2, -0.1, 0.3, 0.1, -0.2, 0.4});
    const double h = 1e-6;
    const Vector3d e_p = lab::rcm_geometry(fk_ref(th + h * qdot), task().p_port).e;
    const Vector3d e_m = lab::rcm_geometry(fk_ref(th - h * qdot), task().p_port).e;
    const Vector3d de = (e_p - e_m) / (2 * h);
    // p_c moves with the lateral part of the shaft point velocity (sliding along z does not change e)
    const VectorXd predicted = -(A.bottomRows(2) * qdot);
    CHECK_MAT_NEAR(N.transpose() * de, predicted, 2e-6);
}

TEST_CASE("test_dls_inverse") {
    for (const auto& c : cases()["dls_inverse"]) {
        const double lam = c["lam"].get<double>();
        INFO("lam = " << lam);
        const MatrixXd A = as_mat(c["A"]);
        const MatrixXd X = lab::dls_inverse(A, lam);
        if (lam == 0.0) {
            CHECK_MAT_NEAR(X, mr::pinv(A), 1e-10);
        } else {
            const Eigen::JacobiSVD<MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
            const VectorXd s = svd.singularValues();
            const VectorXd f = (s.array() / (s.array().square() + lam * lam)).matrix();
            const MatrixXd expected = svd.matrixV() * f.asDiagonal() * svd.matrixU().transpose();
            CHECK_MAT_NEAR(X, expected, 1e-10);
        }
        CHECK_MAT_NEAR(X, as_mat(c["X"]), 1e-10);
    }
}

// ---------------------------------------------------------------- Part C
TEST_CASE("test_rcm_rates_satisfy_both_tasks") {
    const VectorXd th = task().theta0;
    const Matrix4d T = fk_ref(th);
    const Vector3d p_tip = T.block<3, 1>(0, 3);
    const Vector3d p_d = p_tip + Vector3d(0.002, -0.001, 0.0), v_d(0.01, 0.0, -0.005);
    const VectorXd qdot = lab::rcm_rates(robot(), th, task().p_port, p_d, v_d, 5.0, 10.0);
    REQUIRE(qdot.size() == 6);
    const MatrixXd Js = mr::jacobian_space(robot().Slist, th);
    const Vector3d tip_task = v_d + 5.0 * (p_d - p_tip);
    CHECK_MAT_NEAR(lab::point_jacobian(Js, p_tip) * qdot, tip_task, 1e-10);
    const auto [A, info] = lab::rcm_jacobian(robot(), th, task().p_port);
    REQUIRE(A.rows() == 5);
    const VectorXd rcm_task = 10.0 * info.N.transpose() * info.geometry.e;
    CHECK_MAT_NEAR(A.bottomRows(2) * qdot, rcm_task, 1e-10);
    // minimum norm: no component in the null space of A
    const VectorXd v = null_vector(A);
    CHECK(std::abs(v.dot(qdot)) < 1e-10);
    // the minimum-norm solution is unique: the Python reference gives the same rates
    CHECK_MAT_NEAR(qdot, as_vec(cases()["rcm_rates_satisfy_both_tasks"]["qdot"]), 1e-8);
}

TEST_CASE("test_tip_only_rates") {
    const VectorXd th = task().theta0;
    const Matrix4d T = fk_ref(th);
    const Vector3d p_d = T.block<3, 1>(0, 3), v_d(0.0, 0.01, 0.0);
    const VectorXd qdot = lab::tip_only_rates(robot(), th, p_d, v_d, 5.0);
    const MatrixXd Jt = lab::point_jacobian(mr::jacobian_space(robot().Slist, th), T.block<3, 1>(0, 3));
    CHECK_MAT_NEAR(Jt * qdot, v_d, 1e-10);
    CHECK_MAT_NEAR(qdot, mr::pinv(Jt) * v_d, 1e-10);
    CHECK_MAT_NEAR(qdot, as_vec(cases()["tip_only_rates"]["qdot"]), 1e-10);
}

TEST_CASE("test_scale_rates") {
    const VectorXd qmax = VectorXd::Constant(6, 40.0 * PI / 180.0);
    const VectorXd q = vec({0.9, -0.3, 0.6, 0.2, 0.0, -0.1});
    const VectorXd out = lab::scale_rates(q, qmax);
    REQUIRE(out.size() == 6);
    const VectorXd expected = q * qmax(0) / 0.9;
    CHECK_MAT_NEAR(out, expected, 1e-12);  // Practice Problem 5(d)
    CHECK_APPROX((out.cwiseAbs().array() / qmax.array()).maxCoeff(), 1.0);
    const VectorXd small = 0.1 * q;
    CHECK_MAT_NEAR(lab::scale_rates(small, qmax), small, 1e-12);  // Python: rtol 1e-7 on an exact copy
}

TEST_CASE("test_simulation_rcm_tracks_and_keeps_the_port") {
    const lab::SimLog log = sim("rcm");
    const int K = steps_total();
    const json& c = cases()["simulation_rcm_tracks_and_keeps_the_port"];
    CHECK(K == c["K"].get<int>());
    CHECK(static_cast<int>(std::lround(task().path[0].T / task().dt)) == c["k_ins"].get<int>());
    REQUIRE(log.theta.rows() == K + 1);
    REQUIRE(log.theta.cols() == 6);
    REQUIRE(log.p_tip.rows() == K + 1);
    REQUIRE(log.p_tip.cols() == 3);
    REQUIRE(log.tip_err.size() == K + 1);
    REQUIRE(log.rcm_err.size() == K + 1);
    REQUIRE(log.depth.size() == K + 1);
    const auto& acc = task().acceptance;
    CHECK(1e3 * log.tip_err.maxCoeff() < acc.max_tip_error_mm);
    CHECK(1e3 * log.rcm_err.maxCoeff() < acc.max_rcm_error_mm);
    const int k_ins = static_cast<int>(std::lround(task().path[0].T / task().dt));
    CHECK_SCALAR_NEAR(log.depth(k_ins), 0.05, 5e-4);                                  // inserted to 50 mm
    CHECK_SCALAR_NEAR(log.depth.maxCoeff(), std::hypot(0.05, 0.01), 5e-4);            // circle: tip 10 mm off-axis
    CHECK_SCALAR_NEAR(log.depth(K), 0.03, 5e-4);                                      // retracted to 30 mm
    const MatrixXd& lim = robot().limits;
    bool inside = true;
    for (Eigen::Index k = 0; k <= K; ++k)
        for (Eigen::Index i = 0; i < 6; ++i)
            inside = inside && log.theta(k, i) > lim(i, 0) && log.theta(k, i) < lim(i, 1);
    CHECK(inside);
    const MatrixXd rates = (log.theta.bottomRows(K) - log.theta.topRows(K)).cwiseAbs() / task().dt;
    bool slow = true;
    for (Eigen::Index k = 0; k < K; ++k)
        for (Eigen::Index i = 0; i < 6; ++i) slow = slow && rates(k, i) <= robot().qdot_max(i) + 1e-9;
    CHECK(slow);
    // logged tip positions agree with modern_robotics
    for (int k : {0, K / 3, K}) {
        INFO("k = " << k);
        const VectorXd th_k = log.theta.row(k).transpose();
        const Vector3d p_ref = fk_ref(th_k).block<3, 1>(0, 3);
        const VectorXd p_k = log.p_tip.row(k).transpose();
        CHECK_MAT_NEAR(p_k, p_ref, 1e-12);
    }
}

TEST_CASE("test_simulation_tip_only_violates_the_port") {
    const lab::SimLog log_tip = sim("tip"), log_rcm = sim("rcm");
    REQUIRE(log_tip.tip_err.size() > 0);
    REQUIRE(log_tip.rcm_err.size() > 0);
    REQUIRE(log_rcm.rcm_err.size() > 0);
    CHECK(1e3 * log_tip.tip_err.maxCoeff() < task().acceptance.max_tip_error_mm);  // the tip task is fine ...
    CHECK(1e3 * log_tip.rcm_err.maxCoeff() > 5.0);                                 // ... but the port is torn
    CHECK(log_tip.rcm_err.maxCoeff() > 20 * log_rcm.rcm_err.maxCoeff());
}

// ---------------------------------------------------------------- Part D
TEST_CASE("test_fulcrum_ratio") {
    for (const auto& c : cases()["fulcrum_ratio"]) {
        const double L = c["L"].get<double>(), d = c["depth"].get<double>();
        INFO("L = " << L << ", depth = " << d);
        CHECK_SCALAR_NEAR(lab::fulcrum_ratio(L, d), -d / (L - d), 1e-12);
        CHECK_SCALAR_NEAR(lab::fulcrum_ratio(L, d), c["ratio"].get<double>(), 1e-12);
    }
    CHECK_SCALAR_NEAR(lab::fulcrum_ratio(0.36, 0.12), -0.5, 1e-12);
    CHECK_SCALAR_NEAR(lab::fulcrum_ratio(0.36, 0.24), -2.0, 1e-12);
    CHECK_SCALAR_NEAR(lab::fulcrum_ratio(0.36, 0.06), -0.2, 1e-12);
    CHECK_SCALAR_NEAR(lab::fulcrum_ratio(0.36, 0.18), -1.0, 1e-12);
}

TEST_CASE("test_scaled_tip_command") {
    const Matrix3d R = mr::matrix_exp3(mr::vec_to_so3(Vector3d(0, 0, PI / 2)));
    CHECK_MAT_NEAR(lab::scaled_tip_command(Vector3d(15.0, 0, 0), 1.0 / 3.0, R), Vector3d(0, 5.0, 0), 1e-12);
    CHECK_MAT_NEAR(lab::scaled_tip_command(Vector3d(3.0, -6.0, 9.0), 0.2), Vector3d(0.6, -1.2, 1.8), 1e-12);
}
