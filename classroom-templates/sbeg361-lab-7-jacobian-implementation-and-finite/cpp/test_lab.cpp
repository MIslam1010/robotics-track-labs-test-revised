// Auto-checks for SBEG 361 Lab 7 (Jacobians, finite-difference check) -- C++ track. Mirrors test_lab.py test by test.
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*jacobian_space*,*jacobian_body*"      (run only some tests, like pytest -k)
// Ground truth: mrcpp (the C++ port of modern_robotics: jacobian_space, jacobian_body, fkin_space, adjoint),
// MR Example 5.2 (RRRP space Jacobian, printed p. 179), MR Example 5.5 / Eq. 5.25, the 2R Jacobian of MR Eq. 5.1
// (p. 170, numbers on p. 171), and cpp_cases.json, which holds the numbers of the Python reference (including the
// random configurations that test_lab.py draws with NumPy), so both languages are checked against the same values.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <cmath>
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
const std::vector<std::string> ROBOTS = {"course_arm", "desktop_6r", "ur5", "eod_6r_ugv", "rrp"};

const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

struct Model {
    MatrixXd S;  // space screw axes (columns)
    Matrix4d M;
    MatrixXd B;  // body screw axes, [Ad_{M^-1}] S (computed with mrcpp)
};

/// Python: model(name) -> (S, M, B, info), with the given loader lab::load_robot.
Model model(const std::string& name) {
    const lab::Robot r = lab::load_robot(name);
    return {r.Slist, r.M, mr::adjoint(mr::trans_inv(r.M)) * r.Slist};
}

/// Reference forward kinematics (mr.FKinSpace), as the callable the FD functions take.
lab::FkFunction fk_of(const MatrixXd& S, const Matrix4d& M) {
    return [S, M](const VectorXd& q) -> Matrix4d { return mr::fkin_space(M, S, q); };
}

/// np.linalg.matrix_rank(A, tol): number of singular values > tol.
int matrix_rank(const MatrixXd& A, double tol) {
    const Eigen::JacobiSVD<MatrixXd> svd(A);
    const VectorXd s = svd.singularValues();
    int r = 0;
    for (Eigen::Index k = 0; k < s.size(); ++k)
        if (s(k) > tol) ++r;
    return r;
}

/// np.gradient(Y, t, axis=0) for non-uniform t (second order inside, first order at the ends).
MatrixXd np_gradient(const MatrixXd& Y, const VectorXd& t) {
    const Eigen::Index N = Y.rows();
    MatrixXd D(N, Y.cols());
    D.row(0) = (Y.row(1) - Y.row(0)) / (t(1) - t(0));
    D.row(N - 1) = (Y.row(N - 1) - Y.row(N - 2)) / (t(N - 1) - t(N - 2));
    for (Eigen::Index k = 1; k + 1 < N; ++k) {
        const double h1 = t(k) - t(k - 1), h2 = t(k + 1) - t(k);
        D.row(k) = -h2 / (h1 * (h1 + h2)) * Y.row(k - 1) + (h2 - h1) / (h1 * h2) * Y.row(k) +
                   h1 / (h2 * (h1 + h2)) * Y.row(k + 1);
    }
    return D;
}
}  // namespace

// ---------------------------------------------------------------- Checkpoint 1: geometric Jacobians
TEST_CASE("test_jacobian_space_matches_mr") {
    for (const std::string& name : ROBOTS) {
        const Model m = model(name);
        for (const auto& c : cases()["jacobian_space_matches_mr"][name]) {
            const VectorXd q = as_vec(c["q"]);
            INFO(name << ", q = " << q.transpose());
            const MatrixXd Js = lab::jacobian_space(m.S, q);
            CHECK_MAT_NEAR(Js, mr::jacobian_space(m.S, q), 1e-10);
            CHECK_MAT_NEAR(Js, as_mat(c["Js"]), 1e-10);
        }
    }
}

TEST_CASE("test_jacobian_body_matches_mr") {
    for (const std::string& name : ROBOTS) {
        const Model m = model(name);
        for (const auto& c : cases()["jacobian_body_matches_mr"][name]) {
            const VectorXd q = as_vec(c["q"]);
            INFO(name << ", q = " << q.transpose());
            const MatrixXd Jb = lab::jacobian_body(m.B, q);
            CHECK_MAT_NEAR(Jb, mr::jacobian_body(m.B, q), 1e-10);
            CHECK_MAT_NEAR(Jb, as_mat(c["Jb"]), 1e-10);
        }
    }
}

TEST_CASE("test_first_and_last_columns") {
    // J_s1 = S_1 and J_bn = B_n for every configuration (MR Defs. 5.1, 5.4).
    const Model m = model("course_arm");
    const Eigen::Index last = m.B.cols() - 1;
    for (const auto& qj : cases()["first_and_last_columns"]["q"]) {
        const VectorXd q = as_vec(qj);
        const MatrixXd Js = lab::jacobian_space(m.S, q);
        const MatrixXd Jb = lab::jacobian_body(m.B, q);
        CHECK_MAT_NEAR(Js.col(0), m.S.col(0), 1e-12);
        CHECK_MAT_NEAR(Jb.col(last), m.B.col(last), 1e-12);
    }
}

TEST_CASE("test_mr_example_5_2_rrrp") {
    // Space Jacobian of the spatial RRRP chain, MR Example 5.2 (Fig. 5.7, printed p. 179).
    const double L1 = 0.4, L2 = 0.3;
    MatrixXd S(6, 4);
    S.col(0) << 0, 0, 1, 0, 0, 0;
    S.col(1) << 0, 0, 1, 0, -L1, 0;
    S.col(2) << 0, 0, 1, 0, -(L1 + L2), 0;
    S.col(3) << 0, 0, 0, 0, 0, 1;
    for (const auto& c : cases()["mr_example_5_2_rrrp"]) {
        const VectorXd q = as_vec(c["q"]);
        const double c1 = std::cos(q(0)), s1 = std::sin(q(0));
        const double c12 = std::cos(q(0) + q(1)), s12 = std::sin(q(0) + q(1));
        MatrixXd expected(6, 4);
        expected << 0, 0, 0, 0,
                    0, 0, 0, 0,
                    1, 1, 1, 0,
                    0, L1 * s1, L1 * s1 + L2 * s12, 0,
                    0, -L1 * c1, -L1 * c1 - L2 * c12, 0,
                    0, 0, 0, 1;
        CHECK_MAT_NEAR(lab::jacobian_space(S, q), expected, 1e-12);
    }
}

TEST_CASE("test_dependence_on_joints") {
    // J_s does not depend on theta_n; J_b does not depend on theta_1 (MR Sec. 5.1.1-5.1.2).
    const Model m = model("course_arm");
    const VectorXd q = as_vec(cases()["dependence_on_joints"]["q"]);
    VectorXd q2 = q, q3 = q;
    q2(q2.size() - 1) += 1.0;
    q3(0) += 1.0;
    CHECK_MAT_NEAR(lab::jacobian_space(m.S, q2), lab::jacobian_space(m.S, q), 1e-12);
    CHECK_MAT_NEAR(lab::jacobian_body(m.B, q3), lab::jacobian_body(m.B, q), 1e-12);
}

TEST_CASE("test_space_body_relation") {
    // J_b = [Ad_Tbs] J_s and J_s = [Ad_Tsb] J_b (MR Eqs. 5.22-5.23).
    for (const std::string& name : ROBOTS) {
        const Model m = model(name);
        for (const auto& c : cases()["space_body_relation"][name]) {
            const VectorXd q = as_vec(c["q"]);
            INFO(name << ", q = " << q.transpose());
            const Matrix4d T = mr::fkin_space(m.M, m.S, q);
            const MatrixXd Js = mr::jacobian_space(m.S, q), Jb = mr::jacobian_body(m.B, q);
            CHECK_MAT_NEAR(lab::body_from_space(Js, T), Jb, 1e-10);
            CHECK_MAT_NEAR(lab::space_from_body(Jb, T), Js, 1e-10);
            CHECK_MAT_NEAR(lab::body_from_space(Js, T), as_mat(c["Jb"]), 1e-10);
        }
    }
}

TEST_CASE("test_point_jacobian_2r_mr_eq_5_1") {
    // For the planar 2R with L1 = L2 = 1 the point Jacobian equals MR Eq. 5.1; at (0, pi/4) it is
    // [[-0.71, -0.71], [1.71, 0.71]] (MR p. 171).
    MatrixXd S(6, 2);
    S.col(0) << 0, 0, 1, 0, 0, 0;
    S.col(1) << 0, 0, 1, 0, -1, 0;
    const Matrix4d M = mr::rp_to_trans(Matrix3d::Identity(), Vector3d(2.0, 0.0, 0.0));
    const std::vector<Eigen::Vector2d> qs = {Eigen::Vector2d(0.0, PI / 4), Eigen::Vector2d(0.3, -1.1),
                                             Eigen::Vector2d(2.0, 0.7)};
    for (const Eigen::Vector2d& q2 : qs) {
        const VectorXd q = q2;
        INFO("q = " << q.transpose());
        const Vector3d p = mr::fkin_space(M, S, q).topRightCorner<3, 1>();
        const MatrixXd Jp = lab::point_jacobian(mr::jacobian_space(S, q), p);
        const double s1 = std::sin(q(0)), c1 = std::cos(q(0)), s12 = std::sin(q.sum()), c12 = std::cos(q.sum());
        Eigen::Matrix2d J51;
        J51 << -s1 - s12, -s12, c1 + c12, c12;
        CHECK_MAT_NEAR(Jp.topRows(2), J51, 1e-12);
        CHECK_MAT_NEAR(Jp.row(2), Eigen::RowVector2d::Zero(), 1e-12);
    }
    const VectorXd q = Eigen::Vector2d(0.0, PI / 4);
    const Vector3d p = mr::fkin_space(M, S, q).topRightCorner<3, 1>();
    const MatrixXd Jp = lab::point_jacobian(mr::jacobian_space(S, q), p);
    Eigen::Matrix2d book;
    book << -0.7071, -0.7071, 1.7071, 0.7071;
    CHECK_MAT_NEAR(Jp.topRows(2), book, 1e-4);
    CHECK_MAT_NEAR(Jp, as_mat(cases()["point_jacobian_2r_mr_eq_5_1"]["Jp"]), 1e-12);
}

TEST_CASE("test_point_jacobian_equals_R_Jbv") {
    // pdot = v_s + w_s x p = R_sb v_b.
    const Model m = model("desktop_6r");
    for (const auto& qj : cases()["point_jacobian_equals_R_Jbv"]["q"]) {
        const VectorXd q = as_vec(qj);
        const Matrix4d T = mr::fkin_space(m.M, m.S, q);
        const MatrixXd Jp = lab::point_jacobian(mr::jacobian_space(m.S, q), T.topRightCorner<3, 1>());
        const MatrixXd expected = T.topLeftCorner<3, 3>() * mr::jacobian_body(m.B, q).bottomRows(3);
        CHECK_MAT_NEAR(Jp, expected, 1e-12);
    }
}

// ---------------------------------------------------------------- Checkpoint 2: finite differences
TEST_CASE("test_fd_space_and_body_central") {
    for (const std::string name : {"course_arm", "ur5", "eod_6r_ugv"}) {
        const Model m = model(name);
        const lab::FkFunction fk = fk_of(m.S, m.M);
        for (const auto& c : cases()["fd_space_and_body_central"][name]) {
            const VectorXd q = as_vec(c["q"]);
            INFO(name << ", q = " << q.transpose());
            CHECK_MAT_NEAR(lab::fd_jacobian_space(fk, q, 1e-5), mr::jacobian_space(m.S, q), 1e-8);
            CHECK_MAT_NEAR(lab::fd_jacobian_body(fk, q, 1e-5), mr::jacobian_body(m.B, q), 1e-8);
        }
    }
}

TEST_CASE("test_fd_forward_is_first_order_and_central_second_order") {
    // Halving h halves the forward-difference error and quarters the central-difference error.
    const Model m = model("course_arm");
    VectorXd q(6);
    q << 0.3, -0.7, 1.2, -1.9, -1.2, 0.5;
    const lab::FkFunction fk = fk_of(m.S, m.M);
    const MatrixXd Js = mr::jacobian_space(m.S, q);
    const VectorXd hs = Vector3d(4e-3, 2e-3, 1e-3);
    const VectorXd ef = lab::fd_error_study(fk, Js, q, hs, "forward");
    const VectorXd ec = lab::fd_error_study(fk, Js, q, hs, "central");
    REQUIRE(ef.size() == 3);
    REQUIRE(ec.size() == 3);
    for (Eigen::Index k = 0; k < 2; ++k) {
        INFO("h = " << hs(k) << " -> " << hs(k + 1));
        CHECK(1.8 < ef(k) / ef(k + 1));
        CHECK(ef(k) / ef(k + 1) < 2.2);
        CHECK(3.6 < ec(k) / ec(k + 1));
        CHECK(ec(k) / ec(k + 1) < 4.4);
    }
    const double sf = lab::fit_slope(hs, ef), sc = lab::fit_slope(hs, ec);
    CHECK(0.9 < sf);
    CHECK(sf < 1.1);
    CHECK(1.9 < sc);
    CHECK(sc < 2.1);
    const json& c = cases()["fd_forward_is_first_order_and_central_second_order"];
    const VectorXd ef_ref = as_vec(c["ef"]), ec_ref = as_vec(c["ec"]);
    CHECK_MAT_NEAR(ef, ef_ref, 1e-4 * ef_ref.minCoeff());  // same errors as the Python reference
    CHECK_MAT_NEAR(ec, ec_ref, 1e-4 * ec_ref.minCoeff());
}

TEST_CASE("test_fd_bad_scheme_raises") {
    const Model m = model("rrp");
    const VectorXd q = Vector3d(0.1, 0.2, 0.3);
    CHECK_THROWS_AS(lab::fd_jacobian_space(fk_of(m.S, m.M), q, 1e-5, "backward"), std::invalid_argument);
}

TEST_CASE("test_log_jacobian_is_exact") {
    // One joint moving alone is a screw motion about J_si: log(T(q + h e_i) T(q)^-1) = [J_si] h.
    const Model m = model("course_arm");
    const json& c = cases()["log_jacobian_is_exact"];
    for (const auto& hj : c["hs"]) {
        const double h = hj.get<double>();
        for (const auto& qj : c["q"]) {
            const VectorXd q = as_vec(qj);
            INFO("h = " << h << ", q = " << q.transpose());
            CHECK_MAT_NEAR(lab::log_jacobian_space(fk_of(m.S, m.M), q, h), mr::jacobian_space(m.S, q), 1e-9);
        }
    }
}

// ---------------------------------------------------------------- Checkpoint 3: analytic Jacobian
TEST_CASE("test_a_matrix_properties") {
    CHECK_MAT_NEAR(lab::a_matrix(Vector3d::Zero()), Matrix3d::Identity(), 1e-12);
    const Vector3d r(0.3, -0.5, 0.9);
    CHECK_MAT_NEAR(lab::a_matrix(r) * r, r, 1e-12);  // rotation about r itself: w_b = rdot
    const Matrix3d A = lab::a_matrix(Vector3d(0.0, 0.0, PI / 2));  // lecture notes WE-B5 value
    Matrix3d A_we;
    A_we << 0.6366, 0.6366, 0, -0.6366, 0.6366, 0, 0, 0, 1;
    CHECK_MAT_NEAR(A, A_we, 1e-4);
    CHECK_MAT_NEAR(A, as<Matrix3d>(cases()["a_matrix_properties"]["A_pi2"]), 1e-12);
}

TEST_CASE("test_a_matrix_against_finite_differences") {
    // w_b = A(r) rdot with [w_b] = R^T Rdot (MR Example 5.5, Exercise 5.10).
    for (const auto& c : cases()["a_matrix_against_finite_differences"]) {
        const Vector3d r = as<Vector3d>(c["r"]), rdot = as<Vector3d>(c["rdot"]);
        INFO("r = " << r.transpose() << ", rdot = " << rdot.transpose());
        const double h = 1e-6;
        const Matrix3d Rp = mr::matrix_exp3(mr::vec_to_so3(r + h * rdot));
        const Matrix3d Rm = mr::matrix_exp3(mr::vec_to_so3(r - h * rdot));
        const Matrix3d R = mr::matrix_exp3(mr::vec_to_so3(r));
        const Vector3d wb = mr::so3_to_vec(Matrix3d(R.transpose() * (Rp - Rm) / (2 * h)));
        CHECK_MAT_NEAR(lab::a_matrix(r) * rdot, wb, 1e-7);
        CHECK_MAT_NEAR(lab::a_matrix(r) * rdot, as<Vector3d>(c["wb"]), 1e-7);
    }
}

TEST_CASE("test_analytic_jacobian") {
    // Eq. 5.25 against central differences of q = (r, x), away from |r| = pi.
    for (const std::string name : {"course_arm", "eod_6r_ugv"}) {
        const Model m = model(name);
        const lab::FkFunction fk = fk_of(m.S, m.M);
        int checked = 0;
        for (const auto& c : cases()["analytic_jacobian"][name]) {
            const VectorXd q = as_vec(c["q"]);
            INFO(name << ", q = " << q.transpose());
            const Matrix4d T = fk(q);
            const Matrix3d R = T.topLeftCorner<3, 3>();
            if (mr::so3_to_vec(mr::matrix_log3(R)).norm() > 2.8) continue;
            const MatrixXd Ja = lab::analytic_jacobian(mr::jacobian_body(m.B, q), T);
            CHECK_MAT_NEAR(Ja, lab::fd_analytic_jacobian(fk, q, 1e-6), 1e-6);
            CHECK_MAT_NEAR(Ja, as_mat(c["Ja"]), 1e-9);
            ++checked;
        }
        CHECK(checked >= 4);
    }
}

TEST_CASE("test_analytic_position_rows_equal_point_jacobian") {
    const Model m = model("course_arm");
    VectorXd q(6);
    q << 0.2, -0.5, 1.0, -2.0, -1.5, 0.3;
    const Matrix4d T = mr::fkin_space(m.M, m.S, q);
    const MatrixXd Ja = lab::analytic_jacobian(mr::jacobian_body(m.B, q), T);
    const Vector3d p = T.topRightCorner<3, 1>();
    CHECK_MAT_NEAR(Ja.bottomRows(3), lab::point_jacobian(mr::jacobian_space(m.S, q), p), 1e-12);
}

// ---------------------------------------------------------------- Checkpoint 4: EOD singularity + teleop data
TEST_CASE("test_eod_wrist_singularity_rank") {
    // EOD arm at (0, -60, 60, 0, 0, 0) deg (shoulder at its -60 deg limit): axes 4 and 6 are
    // collinear -> rank 5 for J_s and J_b.
    const Model m = model("eod_6r_ugv");
    VectorXd q(6);
    q << 0.0, -PI / 3, PI / 3, 0.0, 0.0, 0.0;
    const MatrixXd Js = lab::jacobian_space(m.S, q), Jb = lab::jacobian_body(m.B, q);
    CHECK(matrix_rank(Js, 1e-9) == 5);
    CHECK(matrix_rank(Jb, 1e-9) == 5);
    CHECK_MAT_NEAR(Js.col(3), Js.col(5), 1e-12);
    sbeg::Vector6d col4;
    col4 << 1, 0, 0, 0, 0.2 + 0.8 * std::sin(PI / 3), 0;
    CHECK_MAT_NEAR(Js.col(3), col4, 1e-12);
}

TEST_CASE("test_teleop_tip_velocity") {
    const auto [t, Q] = lab::load_teleop_csv();
    const Model m = model("desktop_6r");
    const MatrixXd Qd = np_gradient(Q, t);
    const MatrixXd V = lab::tip_velocity_series(m.S, m.M, Q, Qd);
    REQUIRE(V.rows() == t.size());
    REQUIRE(V.cols() == 3);
    MatrixXd P(Q.rows(), 3);
    for (Eigen::Index k = 0; k < Q.rows(); ++k) {
        const VectorXd qk = Q.row(k).transpose();
        P.row(k) = mr::fkin_space(m.M, m.S, qk).topRightCorner<3, 1>().transpose();
    }
    const MatrixXd Pd = np_gradient(P, t);
    CHECK((V - Pd).cwiseAbs().maxCoeff() < 1e-4);  // Jacobian vs differentiated FK
    Eigen::Index kmax = 0;
    const double vmax = V.rowwise().norm().maxCoeff(&kmax);
    CHECK(std::abs(vmax - 0.0541) <= 5e-4);  // value quoted in the handout
    CHECK(std::abs(t(kmax) - 1.83) <= 0.02);
    const json& c = cases()["teleop_tip_velocity"];
    const std::vector<int> rows = c["rows"].get<std::vector<int>>();
    const MatrixXd V_ref = as_mat(c["V_rows"]), Qd_ref = as_mat(c["Qd_rows"]);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const Eigen::Index k = rows[i];
        const Eigen::Index ii = static_cast<Eigen::Index>(i);
        INFO("sample " << k);
        CHECK_MAT_NEAR(Qd.row(k), Qd_ref.row(ii), 1e-9);  // np.gradient reproduced
        CHECK_MAT_NEAR(V.row(k), V_ref.row(ii), 1e-9);
    }
}
