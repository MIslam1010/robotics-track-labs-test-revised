// Auto-checks for SBEG 46X Lab 7 (Cosserat-rod statics vs constant-curvature and Euler-Bernoulli models; tip-force
// estimation for a tendon-driven catheter) -- C++ track.  Mirrors test_lab.py test by test (same names, same order,
// same tolerances).
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="test_A*"      (run only some tests, like pytest -k)
// Ground truth: closed-form results written out independently in this file -- the constant-curvature arc, the
// exponential of a twist (mrcpp, the C++ port of modern_robotics), the Euler-Bernoulli cantilever formulas, the
// tip-angle compliance of an arc derived in Workshop 7 Problem 5 -- hand values from Workshop 7, and
// cpp_cases.json, which holds the classical large-deflection elastica of a cantilever (elliptic integrals, computed
// in Python with scipy.special) and the numbers of the Python reference, so both languages are checked against the
// same values.
// With the untouched starter, the two data tests (catheter.json and the bench log, read by the given code) pass, as
// in Python; every other test fails with "TODO: <function>".
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <set>
#include <string>
#include <vector>

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using sbeg::as;
using sbeg::as_mat;
using sbeg::as_vec;
using sbeg::json;

namespace {
const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

const lab::Catheter& CAT() {
    static const lab::Catheter c = lab::load_catheter();
    return c;
}

constexpr double mm = 1e-3;

/// The constant-curvature arc, written out independently (as arc() in test_lab.py).
Vector3d arc(double kappa, double ell) {
    return Vector3d(std::sin(kappa * ell) / kappa, (1 - std::cos(kappa * ell)) / kappa, kappa * ell);
}

/// x divided element-wise by [a, b, c] (Python: x / [L, L, 1] or x / [mm, mm, 1]).
Vector3d scaled(const Vector3d& x, double a, double b, double c) { return Vector3d(x(0) / a, x(1) / b, x(2) / c); }

/// np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol): |got - expected| <= atol + rtol |expected|.
void check_allclose(const MatrixXd& got, const MatrixXd& expected, double rtol, double atol) {
    REQUIRE(got.rows() == expected.rows());
    REQUIRE(got.cols() == expected.cols());
    for (Eigen::Index i = 0; i < got.rows(); ++i)
        for (Eigen::Index k = 0; k < got.cols(); ++k) {
            INFO("entry (" << i << ", " << k << "): got " << got(i, k) << ", expected " << expected(i, k));
            CHECK(std::abs(got(i, k) - expected(i, k)) <= atol + rtol * std::abs(expected(i, k)));
        }
}

/// got == pytest.approx(expected, rel=rel)  (absolute floor 1e-12 as pytest).
void check_rel(double got, double expected, double rel) {
    INFO("got " << got << ", expected " << expected << " (rel " << rel << ")");
    CHECK(std::abs(got - expected) <= std::max(rel * std::abs(expected), 1e-12));
}

/// The first n rows of every column of a log (Python: {k: v[:n] for k, v in log.items()}).
lab::Log head_rows(const lab::Log& log, Eigen::Index n) {
    lab::Log out;
    for (const auto& kv : log) out[kv.first] = kv.second.head(n);
    return out;
}
}  // namespace

// ============================================================================ data
TEST_CASE("test_catheter_json") {
    const lab::Catheter& c = CAT();
    CHECK(c.L == 0.10);
    CHECK(c.EI == 1.0e-3);
    CHECK(c.r == 1.2e-3);
    CHECK(c.tau_nominal == 6.0);
    CHECK_APPROX(c.sigma_p, 1e-4);
    CHECK_APPROX(c.sigma_theta, 0.5 * 3.14159265358979323846 / 180.0);
}

TEST_CASE("test_log_loads") {
    const lab::Log log = lab::load_csv("catheter_log.csv");
    std::set<std::string> keys;
    for (const auto& kv : log) keys.insert(kv.first);
    CHECK(keys == std::set<std::string>{"t", "tau", "x_meas", "y_meas", "theta_meas", "Fx_ref", "Fy_ref"});
    REQUIRE(keys.count("tau") == 1);
    CHECK(log.at("t").size() == 40);
    CHECK(log.at("tau").minCoeff() >= 4);
    CHECK(log.at("tau").maxCoeff() <= 10);
    const VectorXd Fmag = (log.at("Fx_ref").array().square() + log.at("Fy_ref").array().square()).sqrt().matrix();
    CHECK(Fmag.maxCoeff() < 0.13);
}

// ============================================================================ Part A
TEST_CASE("test_A1_cc_tip_closed_form_and_straight_limit") {
    const double L = CAT().L;
    CHECK_MAT_NEAR(lab::cc_tip(7.2, L), arc(7.2, L), 1e-12);
    CHECK_MAT_NEAR(lab::cc_tip(-5.0, 0.04), arc(-5.0, 0.04), 1e-12);
    CHECK_MAT_NEAR(lab::cc_tip(0.0, L), Vector3d(L, 0, 0), 1e-15);
    CHECK_MAT_NEAR(lab::cc_tip(1e-9, L), Vector3d(L, 0.5e-9 * L * L, 1e-10), 1e-14);
    // Workshop 7 P1: tip at (91.58, 34.47) mm, 0.72 rad
    CHECK_MAT_NEAR(scaled(lab::cc_tip(7.2, L), mm, mm, 1), Vector3d(91.58, 34.47, 0.72), 5e-3);
    for (const auto& c : cases()["A1_cc_tip_closed_form_and_straight_limit"]["cases"]) {
        const double k = c["kappa"].get<double>(), l = c["L"].get<double>();
        INFO("kappa = " << k << ", L = " << l);
        CHECK_MAT_NEAR(lab::cc_tip(k, l), as<Vector3d>(c["tip"]), 1e-14);
    }
}

TEST_CASE("test_A2_A5_curvature_models") {
    CHECK_APPROX(lab::tendon_curvature(6.0, CAT().r, CAT().EI), 7.2);
    CHECK_APPROX(lab::pneumatic_curvature(5e4, 50e-6, 4e-3, 2e-3), 5.0);
    CHECK_APPROX(lab::pneumatic_curvature(1.5e5, 50e-6, 4e-3, 2e-3), 15.0);
}

TEST_CASE("test_A3_cc_transform_is_the_exponential_of_a_twist") {
    const double L = CAT().L;
    const Matrix4d T = lab::cc_transform(7.2, L);
    const Vector3d p = arc(7.2, L);
    Matrix3d Rz;
    Rz << std::cos(0.72), -std::sin(0.72), 0, std::sin(0.72), std::cos(0.72), 0, 0, 0, 1;
    const Matrix4d Tref = mr::rp_to_trans(Rz, Vector3d(p(0), p(1), 0.0));
    CHECK_MAT_NEAR(T, Tref, 1e-12);
    // composition: e^{[V] l1} e^{[V] l2} = e^{[V](l1 + l2)}
    CHECK_MAT_NEAR(lab::cc_transform(7.2, 0.03) * lab::cc_transform(7.2, 0.07), T, 1e-12);
    CHECK_MAT_NEAR(T, as<Matrix4d>(cases()["A3_cc_transform_is_the_exponential_of_a_twist"]["T"]), 1e-12);
}

TEST_CASE("test_A4_pcc_tip") {
    // Workshop 7 P4: two pneumatic segments
    CHECK_MAT_NEAR(scaled(lab::pcc_tip({5.0, 15.0}, {0.04, 0.04}), mm, mm, 1), Vector3d(74.313, 22.877, 0.8), 1e-3);
    // equal curvatures = one arc; a single segment = cc_tip; straight segments add up
    CHECK_MAT_NEAR(lab::pcc_tip({10.0, 10.0}, {0.04, 0.04}), arc(10.0, 0.08), 1e-12);
    CHECK_MAT_NEAR(lab::pcc_tip({3.0}, {0.05}), arc(3.0, 0.05), 1e-12);
    CHECK_MAT_NEAR(lab::pcc_tip({0.0, 0.0}, {0.02, 0.03}), Vector3d(0.05, 0, 0), 1e-12);
    // independent: SE(3) product of the exponentials (MR Eq. 3.88)
    Matrix4d T = Matrix4d::Identity();
    const double kl[3][2] = {{4.0, 0.03}, {-8.0, 0.02}, {12.0, 0.025}};
    for (const auto& e : kl) {
        sbeg::Vector6d V;
        V << 0, 0, e[0], 1, 0, 0;
        T = (T * mr::matrix_exp6(mr::vec_to_se3(V * e[1]))).eval();
    }
    const Vector3d z = lab::pcc_tip({4.0, -8.0, 12.0}, {0.03, 0.02, 0.025});
    CHECK_MAT_NEAR(z.head<2>(), (T.block<2, 1>(0, 3)), 1e-12);
    CHECK_APPROX(z(2), std::atan2(T(1, 0), T(0, 0)));
    for (const auto& c : cases()["A4_pcc_tip"]["cases"]) {
        const std::vector<double> k = c["kappas"].get<std::vector<double>>(), l = c["lengths"].get<std::vector<double>>();
        CHECK_MAT_NEAR(lab::pcc_tip(k, l), as<Vector3d>(c["tip"]), 1e-12);
    }
}

// ============================================================================ Part B
TEST_CASE("test_B1_rod_rhs_values_and_vectorised") {
    const double EI = CAT().EI;
    lab::Vector6d z;
    z << 0.01, 0.02, 0.3, 0.1, -0.2, 0.004;
    const Vector2d f(0.5, -1.0);
    const lab::Vector6d d = lab::rod_rhs(0.0, z, EI, f, 2.0);
    const double c = std::cos(0.3), s = std::sin(0.3);
    lab::Vector6d expected;
    expected << c, s, 0.004 / EI + 2.0, -0.5, 1.0, -(c * -0.2 - s * 0.1);
    CHECK_MAT_NEAR(d, expected, 1e-12);
    // C++: rod_rhs takes one state; the "vectorised" part becomes: the same state at other s gives the same value
    // (the rod equations do not depend on s explicitly), as every column of the tiled array in Python.
    for (int k = 0; k < 5; ++k) CHECK_MAT_NEAR(lab::rod_rhs(0.02 * k, z, EI, f, 2.0), d, 1e-12);
    CHECK_MAT_NEAR(d, as_vec(cases()["B1_rod_rhs_values_and_vectorised"]["d"]), 1e-12);
}

TEST_CASE("test_B2_shooting_residual_linear_case") {
    // Straight rod, tip force P: m(s) = sigma - P s exactly (n constant), so R(sigma) = sigma - P L (Workshop P2).
    const double L = CAT().L, EI = CAT().EI;
    const double P = 1e-4;  // tiny load: geometry stays straight to 1e-6
    for (double sigma : {0.0, 2e-5}) {
        INFO("sigma = " << sigma);
        const Vector3d r = lab::shooting_residual(Vector3d(0.0, P, sigma), L, EI, Vector2d(0.0, P), 0.0);
        CHECK_MAT_NEAR(r, Vector3d(0, 0, sigma - P * L), 1e-9);
    }
    for (const auto& c : cases()["B2_shooting_residual_linear_case"]["cases"]) {
        const double sigma = c["sigma"].get<double>();
        CHECK_MAT_NEAR(lab::shooting_residual(Vector3d(0.0, P, sigma), L, EI, Vector2d(0.0, P), 0.0),
                       as<Vector3d>(c["r"]), 1e-12);
    }
}

TEST_CASE("test_B3_pure_tip_moment_gives_the_cc_arc") {
    const double L = CAT().L, EI = CAT().EI, R = CAT().r;
    const lab::RodSolution sol = lab::solve_rod_shooting(L, EI, Vector2d(0, 0), R * 6.0);
    CHECK_MAT_NEAR(lab::tip_pose(sol), arc(7.2, L), 1e-8);
    CHECK_MAT_NEAR(sol.m, VectorXd::Constant(sol.m.size(), R * 6.0), 1e-10);
    REQUIRE(sol.s.size() == 101);
    CHECK_APPROX(sol.s(100), L);
    CHECK_MAT_NEAR(lab::tip_pose(sol), as<Vector3d>(cases()["B3_pure_tip_moment_gives_the_cc_arc"]["tip"]), 1e-8);
}

TEST_CASE("test_B3_shooting_matches_the_elastica") {
    // Python: parametrized over alpha = 0.1, 1.0, 2.0 (3 pytest items); the elastica tips (elliptic integrals,
    // scipy.special) are stored in cpp_cases.json.
    const double L = CAT().L, EI = CAT().EI;
    for (const auto& c : cases()["B3_shooting_matches_the_elastica"]["cases"]) {
        const double alpha = c["alpha"].get<double>();
        INFO("alpha = " << alpha);
        const double P = alpha * EI / (L * L);
        const lab::RodSolution sol = lab::solve_rod_shooting(L, EI, Vector2d(0, P));
        CHECK_MAT_NEAR(scaled(lab::tip_pose(sol), L, L, 1), as<Vector3d>(c["elastica"]), 1e-7);
        REQUIRE(sol.m.size() > 0);
        REQUIRE(sol.x.size() > 0);
        CHECK_SCALAR_NEAR(sol.m(0), P * sol.x(sol.x.size() - 1), 1e-9);  // base moment = P x_L
        CHECK_MAT_NEAR(sol.ny, VectorXd::Constant(sol.ny.size(), P), 1e-12);
        CHECK_MAT_NEAR(lab::tip_pose(sol), as<Vector3d>(c["tip"]), 1e-8);
    }
}

TEST_CASE("test_B3_small_loads_reduce_to_euler_bernoulli") {
    const double L = CAT().L, EI = CAT().EI;
    const double P = 1e-3, w = 1e-2;  // alpha = 0.01
    const Vector3d z = lab::tip_pose(lab::solve_rod_shooting(L, EI, Vector2d(0, P)));
    check_rel(z(1), P * L * L * L / (3 * EI), 2e-3);
    const Vector3d zw = lab::tip_pose(lab::solve_rod_shooting(L, EI, Vector2d(0, 0), 0.0, Vector2d(0, w)));
    check_rel(zw(1), w * L * L * L * L / (8 * EI), 2e-3);
    check_rel(zw(2), w * L * L * L / (6 * EI), 2e-3);
    const json& c = cases()["B3_small_loads_reduce_to_euler_bernoulli"];
    CHECK_MAT_NEAR(z, as<Vector3d>(c["z"]), 1e-9);
    CHECK_MAT_NEAR(zw, as<Vector3d>(c["zw"]), 1e-9);
}

TEST_CASE("test_B3_distributed_load_force_balance") {
    // Workshop/Practice P3 arm: n(0) = F + f L exactly; self-weight + tip camera.
    const double P = 0.05 * 9.81, w = 0.5;
    const lab::RodSolution sol = lab::solve_rod_shooting(0.40, 0.50, Vector2d(0, -P), 0.0, Vector2d(0, -w));
    REQUIRE(sol.nx.size() > 0);
    REQUIRE(sol.ny.size() > 0);
    CHECK_MAT_NEAR(Vector2d(sol.nx(0), sol.ny(0)), Vector2d(0, -(P + w * 0.40)), 1e-9);
    CHECK_MAT_NEAR(scaled(lab::tip_pose(sol), mm, mm, 1), Vector3d(399.138, -24.042, -0.08889), 2e-3);
    CHECK_MAT_NEAR(lab::tip_pose(sol), as<Vector3d>(cases()["B3_distributed_load_force_balance"]["tip"]), 1e-8);
}

TEST_CASE("test_B4_solve_bvp_agrees_with_shooting_and_elastica") {
    const double L = CAT().L, EI = CAT().EI, R = CAT().r;
    const json& c = cases()["B4_solve_bvp_agrees_with_shooting_and_elastica"];
    const double P = 1.0 * EI / (L * L);
    const lab::RodSolution b1 = lab::solve_rod_bvp(L, EI, Vector2d(0, P));
    CHECK_MAT_NEAR(scaled(lab::tip_pose(b1), L, L, 1), as<Vector3d>(c["elastica_1"]), 1e-6);
    // tendon + oblique tip force + weight, against shooting
    const Vector2d F(0.03, -0.05), f(0.0, -0.2);
    const double M = R * 6.0;
    const lab::RodSolution b = lab::solve_rod_bvp(L, EI, F, M, f);
    const lab::RodSolution s = lab::solve_rod_shooting(L, EI, F, M, f);
    CHECK_MAT_NEAR(lab::tip_pose(b), lab::tip_pose(s), 1e-7);
    CHECK_MAT_NEAR(b.m, s.m, 1e-7);
    CHECK_MAT_NEAR(lab::tip_pose(b), as<Vector3d>(c["combined"]["tip"]), 1e-7);
    CHECK_MAT_NEAR(b.m, as_vec(c["combined"]["m"]), 1e-7);
}

TEST_CASE("test_B5_euler_bernoulli") {
    const double L = CAT().L, EI = CAT().EI;
    CHECK_MAT_NEAR(lab::euler_bernoulli_tip(L, EI, 0.01), Vector3d(L, 1.0 / 3 * 1e-2, 0.05), 1e-12);
    CHECK_MAT_NEAR(lab::euler_bernoulli_tip(0.4, 0.5, -0.4905, 0.0, -0.5), Vector3d(0.4, -0.024128, -0.08914667),
                   1e-7);
    CHECK_MAT_NEAR(lab::euler_bernoulli_tip(L, EI, 0.0, 7.2e-3), Vector3d(L, 0.036, 0.72), 1e-12);
    for (const auto& c : cases()["B5_euler_bernoulli"]["cases"])
        CHECK_MAT_NEAR(lab::euler_bernoulli_tip(c["L"].get<double>(), c["EI"].get<double>(), c["P"].get<double>(),
                                                c["M"].get<double>(), c["w"].get<double>()),
                       as<Vector3d>(c["tip"]), 1e-12);
}

// ============================================================================ Part C
TEST_CASE("test_C1_catheter_tip_unloaded_is_cc_and_loaded_unbends") {
    const double L = CAT().L;
    CHECK_MAT_NEAR(lab::catheter_tip(6.0, Vector2d(0, 0), CAT()), arc(7.2, L), 1e-8);
    const Vector2d d = lab::contact_direction(0.72);
    const Vector3d z = lab::catheter_tip(6.0, 0.02 * d, CAT());
    CHECK(z(2) < 0.72);
    CHECK(z(1) < arc(7.2, L)(1));
    CHECK_MAT_NEAR(z, as<Vector3d>(cases()["C1_catheter_tip_unloaded_is_cc_and_loaded_unbends"]["loaded"]), 1e-8);
}

TEST_CASE("test_C2_compare_models") {
    const double L = CAT().L, EI = CAT().EI, R = CAT().r;
    const lab::ModelComparison out = lab::compare_models(6.0, {0.0, 0.05, 0.1}, CAT());
    REQUIRE(out.cosserat.rows() == 3);
    REQUIRE(out.cosserat.cols() == 3);
    REQUIRE(out.eb.rows() == 3);
    REQUIRE(out.eb.cols() == 3);
    REQUIRE(out.err_cc.size() == 3);
    REQUIRE(out.err_eb.size() == 3);
    CHECK_MAT_NEAR(out.cc, arc(7.2, L), 1e-10);
    CHECK_SCALAR_NEAR(out.err_cc(0), 0.0, 1e-8);
    CHECK(out.err_cc(1) > out.err_cc(0));
    CHECK(out.err_cc(2) > out.err_cc(1));
    // EB about the straight rod already misses the unloaded arc: |(L, r tau L^2/2EI) - CC tip|
    CHECK_SCALAR_NEAR(out.err_eb(0), std::hypot(L - arc(7.2, L)(0), 0.036 - arc(7.2, L)(1)), 1e-8);
    // independent check of one Cosserat point with the BVP solver
    const Vector3d zb = lab::tip_pose(lab::solve_rod_bvp(L, EI, 0.1 * lab::contact_direction(0.72), R * 6.0));
    CHECK_MAT_NEAR(out.cosserat.row(2).transpose(), zb, 1e-7);
    const json& c = cases()["C2_compare_models"];
    CHECK_MAT_NEAR(out.cosserat, as_mat(c["cosserat"]), 1e-8);
    CHECK_MAT_NEAR(out.eb, as_mat(c["eb"]), 1e-12);
    CHECK_MAT_NEAR(out.err_cc, as_vec(c["err_cc"]), 1e-8);
    CHECK_MAT_NEAR(out.err_eb, as_vec(c["err_eb"]), 1e-8);
}

// ============================================================================ Part D
TEST_CASE("test_D1_compliance_straight_is_euler_bernoulli") {
    const double L = CAT().L, EI = CAT().EI;
    const lab::Matrix32d C = lab::compliance_matrix(0.0, CAT());
    lab::Matrix32d expected;
    expected << 0, 0, 0, L * L * L / (3 * EI), 0, L * L / (2 * EI);
    check_allclose(C, expected, 1e-4, 1e-9);
    CHECK_MAT_NEAR(C, as_mat(cases()["D1_compliance_straight_is_euler_bernoulli"]["C"]), 1e-6);
}

TEST_CASE("test_D1_compliance_bent_matches_closed_forms") {
    // Workshop P5 / Practice P5: tip-angle compliance of an arc for a force along the tip normal and tangent.
    const double L = CAT().L, EI = CAT().EI;
    const lab::Matrix32d C = lab::compliance_matrix(6.0, CAT());
    const double k = 7.2, th = 0.72;
    const Vector2d t(std::cos(th), std::sin(th)), nu(-std::sin(th), std::cos(th));
    check_rel((C * nu)(2), (1 - std::cos(k * L)) / (k * k * EI), 1e-4);
    check_rel((C * t)(2), (k * L - std::sin(k * L)) / (k * k * EI), 1e-4);
    CHECK_MAT_NEAR(C, as_mat(cases()["D1_compliance_bent_matches_closed_forms"]["C"]), 1e-6);
}

TEST_CASE("test_D2_linear_estimate_noise_free") {
    const Vector2d d = lab::contact_direction(0.72);
    const Vector2d t(std::cos(0.72), std::sin(0.72));
    // contact model (force along d): one unknown, accurate for a small pure normal force
    Vector3d z = lab::catheter_tip(6.0, 0.005 * d, CAT());
    const Vector2d Fn = lab::estimate_force_linear(z, 6.0, CAT(), d);
    CHECK_SCALAR_NEAR(Fn(0) * d(1) - Fn(1) * d(0), 0.0, 1e-12);  // along d
    check_rel(Fn.dot(d), 0.005, 0.01);
    // both components: only for a very small force (ill-conditioned, Practice P5)
    const Vector2d F = 2e-4 * d + 0.8e-4 * t;
    z = lab::catheter_tip(6.0, F, CAT());
    const Vector2d F2 = lab::estimate_force_linear(z, 6.0, CAT());
    CHECK_MAT_NEAR(F2, F, 2e-5);
    const json& c = cases()["D2_linear_estimate_noise_free"];
    CHECK_MAT_NEAR(Fn, as<Vector2d>(c["F_contact"]), 1e-8);
    CHECK_MAT_NEAR(F2, as<Vector2d>(c["F_2d"]), 1e-7);
}

TEST_CASE("test_D3_nonlinear_estimate_recovers_a_large_force") {
    const Vector2d d = lab::contact_direction(0.72);
    const Vector2d F = 0.10 * d;
    const Vector3d z = lab::catheter_tip(6.0, F, CAT());
    CHECK_MAT_NEAR(lab::estimate_force_nl(z, 6.0, CAT(), d), F, 1e-6);
    CHECK_MAT_NEAR(lab::estimate_force_nl(z, 6.0, CAT()), F, 1e-5);
    // Workshop P5 (c): only the measured angle matters for the hand estimate; the linear one is 2 % low here
    const Vector2d lin = lab::estimate_force_linear(z, 6.0, CAT(), d);
    CHECK(lin.dot(d) < 0.10);
    CHECK_MAT_NEAR(lin, as<Vector2d>(cases()["D3_nonlinear_estimate_recovers_a_large_force"]["F_lin_contact"]), 1e-7);
}

TEST_CASE("test_D4_D5_bench_log") {
    const double L = CAT().L, EI = CAT().EI, R = CAT().r;
    const lab::Log sub = head_rows(lab::load_csv("catheter_log.csv"), 12);
    MatrixXd Fref(12, 2);
    Fref << sub.at("Fx_ref"), sub.at("Fy_ref");
    const VectorXd th = sub.at("tau") * R / EI * L;
    const MatrixXd F_nl = lab::estimate_log(sub, CAT(), "nl", true);
    const MatrixXd F_lin = lab::estimate_log(sub, CAT(), "linear", true);
    const MatrixXd F_2d = lab::estimate_log(sub, CAT(), "linear", false);
    REQUIRE(F_nl.rows() == 12);
    REQUIRE(F_lin.rows() == 12);
    REQUIRE(F_2d.rows() == 12);
    const lab::ForceErrors e_nl = lab::force_errors(F_nl, Fref, th);
    const lab::ForceErrors e_lin = lab::force_errors(F_lin, Fref, th);
    const lab::ForceErrors e_2d = lab::force_errors(F_2d, Fref, th);
    CHECK(e_nl.rms_normal < 0.010);  // nonlinear model beats the linear one
    CHECK(0.010 < e_lin.rms_normal);
    CHECK(e_2d.rms_tangential > 10 * e_nl.rms_normal);  // 2-D estimate: tangential ill-conditioned
    const json& c = cases()["D4_D5_bench_log"];
    CHECK_MAT_NEAR(F_nl, as_mat(c["nl_contact"]), 1e-7);
    CHECK_MAT_NEAR(F_lin, as_mat(c["lin_contact"]), 1e-7);
    CHECK_MAT_NEAR(F_2d, as_mat(c["lin_2d"]), 1e-5);
    CHECK_SCALAR_NEAR(e_nl.rms_normal, c["e_nl"]["rms_normal"].get<double>(), 1e-7);
    CHECK_SCALAR_NEAR(e_lin.rms_normal, c["e_lin"]["rms_normal"].get<double>(), 1e-7);
}

TEST_CASE("test_D5_force_errors_decomposition") {
    MatrixXd Fe(2, 2), Fr(2, 2);
    Fe << 0.1, 0.0, 0.0, 0.0;
    Fr << 0.0, 0.0, 0.0, 0.1;
    const lab::ForceErrors E = lab::force_errors(Fe, Fr, VectorXd::Zero(2));
    CHECK_APPROX(E.rms, 0.1);
    CHECK_APPROX(E.rms_tangential, std::sqrt(0.005));
    CHECK_APPROX(E.rms_normal, std::sqrt(0.005));
}
