// Auto-checks for SBEG 361 Lab 9 (analytic and numerical IK of the course arm; all-solutions search) -- C++ track.
// Mirrors test_lab.py test by test.
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*pk*"      (run only some tests, like pytest -k)
// Ground truth: mrcpp (the C++ port of modern_robotics: fkin_space, fkin_body, ikin_body, ikin_space, matrix_exp6),
// the MR Example 6.1 iteration table (printed p. 230), the 2R formulas of MR pp. 218-219, the Paden-Kahan
// definitions of MLS pp. 99-103 (checked by forward substitution), and cpp_cases.json, which holds the numbers of
// the Python reference (including the random inputs that test_lab.py draws with NumPy and the NumPy random starts
// of the all-solutions search), so both languages are checked against the same values.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
constexpr double PI = 3.14159265358979323846;

const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

double d2r(double deg) { return deg * PI / 180.0; }

VectorXd d2r(std::vector<double> deg) {
    VectorXd v(static_cast<Eigen::Index>(deg.size()));
    for (std::size_t i = 0; i < deg.size(); ++i) v(static_cast<Eigen::Index>(i)) = d2r(deg[i]);
    return v;
}

double wrap(double a) {
    double r = std::fmod(a + PI, 2.0 * PI);
    if (r < 0.0) r += 2.0 * PI;
    return r - PI;
}

VectorXd wrap(const VectorXd& a) {
    VectorXd out(a.size());
    for (Eigen::Index i = 0; i < a.size(); ++i) out(i) = wrap(a(i));
    return out;
}

struct Model {
    MatrixXd S;  // space screw axes (columns)
    Matrix4d M;
    MatrixXd B;  // body screw axes, [Ad_{M^-1}] S (computed with mrcpp)
    lab::Robot info;
};

/// Python: model(name) -> (S, M, B, info), with the given loader lab::load_robot.
Model model(const std::string& name) {
    const lab::Robot r = lab::load_robot(name);
    return {r.Slist, r.M, mr::adjoint(mr::trans_inv(r.M)) * r.Slist, r};
}

/// e^[xi th] p for a zero-pitch unit twist with direction w through r (independent of the lab).
Vector3d rot_point(const Vector3d& w_in, const Vector3d& r, double th, const Vector3d& p) {
    const Vector3d w = w_in / w_in.norm();
    mr::Vector6d S;
    S << w, -w.cross(r);
    const Matrix4d T = mr::matrix_exp6(mr::vec_to_se3(S * th));
    return T.topLeftCorner<3, 3>() * p + T.topRightCorner<3, 1>();
}

/// True if some solution equals q up to wrapped differences below tol (Python: contains(sols, q, tol)).
bool contains(const std::vector<VectorXd>& sols, const VectorXd& q, double tol = 1e-6) {
    for (const VectorXd& s : sols)
        if (s.size() == q.size() && wrap(VectorXd(s - q)).cwiseAbs().maxCoeff() < tol) return true;
    return false;
}

std::vector<VectorXd> to_vectors(const std::vector<std::pair<double, double>>& pairs) {
    std::vector<VectorXd> out;
    for (const auto& [a, b] : pairs) out.push_back(Vector2d(a, b));
    return out;
}

std::vector<VectorXd> to_vectors(const std::vector<double>& angles) {
    std::vector<VectorXd> out;
    for (const double a : angles) out.push_back(VectorXd::Constant(1, a));
    return out;
}

const VectorXd& P8() {  // pose P8 of lecture example WE-A7: 8 real branches
    static const VectorXd q = d2r({0, -90, 60, 0, 45, 0});
    return q;
}
}  // namespace

// ---------------------------------------------------------------- Checkpoint 1: Paden-Kahan subproblems
TEST_CASE("test_pk1_workshop_value") {
    // Workshop P2(a): axis z through (0.1, 0, 0), p = (0.3, 0, 0.2), q = (0.1, 0.2, 0.2) -> 90 deg.
    CHECK_SCALAR_NEAR(lab::pk1(Vector3d(0, 0, 1), Vector3d(0.1, 0, 0), Vector3d(0.3, 0, 0.2), Vector3d(0.1, 0.2, 0.2)),
                      PI / 2, 1e-12);
}

TEST_CASE("test_pk1_random") {
    for (const auto& c : cases()["pk1_random"]) {
        const Vector3d w = as<Vector3d>(c["w"]), r = as<Vector3d>(c["r"]), p = as<Vector3d>(c["p"]);
        const double th = c["th"].get<double>();
        const Vector3d q = rot_point(w, r, th, p);
        INFO("th = " << th);
        CHECK_SCALAR_NEAR(wrap(lab::pk1(w, r, p, q) - th), 0.0, 1e-9);
    }
}

TEST_CASE("test_pk1_no_solution") {
    // different heights along the axis
    CHECK_THROWS_AS(lab::pk1(Vector3d(0, 0, 1), Vector3d(0, 0, 0), Vector3d(1, 0, 0), Vector3d(0, 1, 0.5)),
                    std::invalid_argument);
    // different radii
    CHECK_THROWS_AS(lab::pk1(Vector3d(0, 0, 1), Vector3d(0, 0, 0), Vector3d(1, 0, 0), Vector3d(0, 2, 0)),
                    std::invalid_argument);
}

TEST_CASE("test_pk2_workshop_values") {
    // Workshop P2(b): pan (z) and tilt (-y) axes through the origin.
    const auto sols = to_vectors(lab::pk2(Vector3d(0, 0, 1), Vector3d(0, -1, 0), Vector3d(0, 0, 0),
                                          Vector3d(0.1, 0, 0), Vector3d(0.05, 0.05, std::sqrt(0.005))));
    CHECK(sols.size() == 2);
    CHECK(contains(sols, d2r({45, 45})));
    CHECK(contains(sols, d2r({-135, 135})));
}

TEST_CASE("test_pk2_random") {
    for (const auto& c : cases()["pk2_random"]) {
        const Vector3d w1 = as<Vector3d>(c["w1"]), w2 = as<Vector3d>(c["w2"]);
        const Vector3d r = as<Vector3d>(c["r"]), p = as<Vector3d>(c["p"]);
        const double t1 = c["t1"].get<double>(), t2 = c["t2"].get<double>();
        const Vector3d q = rot_point(w1, r, t1, rot_point(w2, r, t2, p));
        INFO("t1 = " << t1 << ", t2 = " << t2);
        const auto pairs = lab::pk2(w1, w2, r, p, q);
        CHECK(pairs.size() >= 1);
        CHECK(pairs.size() <= 2);
        CHECK(contains(to_vectors(pairs), Vector2d(t1, t2), 1e-7));
        for (const auto& [a, b] : pairs) CHECK_MAT_NEAR(rot_point(w1, r, a, rot_point(w2, r, b, p)), q, 1e-9);
    }
}

TEST_CASE("test_pk2_no_solution") {
    // |q - r| != |p - r|
    CHECK(lab::pk2(Vector3d(0, 0, 1), Vector3d(0, 1, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), Vector3d(0, 0, 2))
              .empty());
}

TEST_CASE("test_pk3_workshop_values") {
    // Workshop P2(c): elbow axis y through (0.30, 0, 0), wrist p = (0.65, 0, 0), shoulder q = 0, delta = 0.5.
    std::vector<double> sols =
        lab::pk3(Vector3d(0, 1, 0), Vector3d(0.30, 0, 0), Vector3d(0.65, 0, 0), Vector3d(0, 0, 0), 0.5);
    std::sort(sols.begin(), sols.end());
    REQUIRE(sols.size() == 2);
    CHECK_MAT_NEAR(Vector2d(sols[0], sols[1]), d2r({-79.71344, 79.71344}), 1e-6);
}

TEST_CASE("test_pk3_random") {
    for (const auto& c : cases()["pk3_random"]) {
        const Vector3d w = as<Vector3d>(c["w"]), r = as<Vector3d>(c["r"]);
        const Vector3d p = as<Vector3d>(c["p"]), q = as<Vector3d>(c["q"]);
        const double th = c["th"].get<double>();
        const double delta = (q - rot_point(w, r, th, p)).norm();
        INFO("th = " << th << ", delta = " << delta);
        const std::vector<double> sols = lab::pk3(w, r, p, q, delta);
        CHECK(sols.size() >= 1);
        CHECK(sols.size() <= 2);
        CHECK(contains(to_vectors(sols), VectorXd::Constant(1, th), 1e-6));
        for (const double a : sols) CHECK_SCALAR_NEAR((q - rot_point(w, r, a, p)).norm(), delta, 1e-9);
    }
    // circle and sphere do not meet
    CHECK(lab::pk3(Vector3d(0, 0, 1), Vector3d(0, 0, 0), Vector3d(1, 0, 0), Vector3d(3, 0, 0), 0.5).empty());
}

// ---------------------------------------------------------------- Checkpoint 2: analytic IK
TEST_CASE("test_ik_2r_values") {
    // Quiz 9 Q1 numbers (planar 2R, L = 0.6, 0.4) and the MR p. 218 righty/lefty convention.
    const auto sols = lab::ik_2r(0.6, 0.4, 0.6, 0.4);
    REQUIRE(sols.size() == 2);
    CHECK_MAT_NEAR(Vector2d(sols[0].first, sols[0].second), Vector2d(0, PI / 2), 1e-9);  // righty: theta2 > 0
    CHECK_MAT_NEAR(Vector2d(sols[1].first, sols[1].second), Vector2d(2 * std::atan2(0.4, 0.6), -PI / 2), 1e-9);
    CHECK_MAT_NEAR(Vector2d(sols[0].first, sols[0].second), as_vec(cases()["ik_2r_values"]["sols"][0]), 1e-9);
    CHECK(lab::ik_2r(1.2, 0.0, 0.6, 0.4).empty());
    CHECK(lab::ik_2r(0.1, 0.1, 0.6, 0.4).empty());
}

TEST_CASE("test_ik_2r_fk") {
    for (const auto& c : cases()["ik_2r_fk"]) {
        const VectorXd q = as_vec(c["q"]);
        const double x = 0.6 * std::cos(q(0)) + 0.4 * std::cos(q.sum());
        const double y = 0.6 * std::sin(q(0)) + 0.4 * std::sin(q.sum());
        INFO("q = " << q.transpose());
        CHECK(contains(to_vectors(lab::ik_2r(x, y, 0.6, 0.4)), q, 1e-7));
    }
}

TEST_CASE("test_ik_course_arm_fk") {  // pytest: parametrized over course_arm, desktop_6r
    for (const std::string name : {"course_arm", "desktop_6r"}) {
        const Model m = model(name);
        for (const auto& qj : cases()["ik_course_arm_fk"]["q"]) {
            const VectorXd q = as_vec(qj);
            INFO(name << ", q = " << q.transpose());
            const Matrix4d T = mr::fkin_space(m.M, m.S, q);
            const std::vector<VectorXd> sols = lab::ik_course_arm(T, m.info.geometry);
            CHECK(sols.size() >= 1);
            CHECK(sols.size() <= 8);
            CHECK_MESSAGE(contains(sols, q, 1e-7), "the configuration that produced T must be among the solutions");
            for (const VectorXd& th : sols) {
                REQUIRE(th.size() == 6);
                CHECK_MAT_NEAR(mr::fkin_space(m.M, m.S, th), T, 1e-10);
                CHECK(th.minCoeff() >= -PI - 1e-12);
                CHECK(th.maxCoeff() < PI + 1e-12);
            }
        }
    }
}

TEST_CASE("test_ik_course_arm_solution_counts") {
    // q_home: 4 solutions (the wrist-flip branches put joint 4 out of the 2R's reach); P8: 8.
    const Model m = model("course_arm");
    const std::vector<std::pair<VectorXd, std::size_t>> poses = {{m.info.q_home, 4}, {P8(), 8}};
    for (const auto& pose : poses) {
        const VectorXd& q = pose.first;
        const std::size_t n = pose.second;
        INFO("q = " << q.transpose());
        const std::vector<VectorXd> sols = lab::ik_course_arm(mr::fkin_space(m.M, m.S, q), m.info.geometry);
        CHECK(sols.size() == n);
        std::set<std::vector<long long>> distinct;  // {tuple(np.round(wrap(s), 6)) for s in sols}
        for (const VectorXd& s : sols) {
            const VectorXd w = wrap(s);
            std::vector<long long> key;
            for (Eigen::Index k = 0; k < w.size(); ++k) key.push_back(std::llround(w(k) * 1e6));
            distinct.insert(key);
        }
        CHECK(distinct.size() == n);
    }
}

TEST_CASE("test_ik_course_arm_unreachable") {
    const Model m = model("course_arm");
    Matrix4d T = Matrix4d::Identity();
    T.topRightCorner<3, 1>() = Vector3d(0.5, 0.0, 0.1);  // 0.5 m away: beyond the 0.28 m reach
    CHECK(lab::ik_course_arm(T, m.info.geometry).empty());
}

// ---------------------------------------------------------------- Checkpoint 3: Newton-Raphson
TEST_CASE("test_newton_raphson_scalar_table") {
    // Workshop P4(a): g = 0.48 sin(theta) - 0.24 from theta0 = 0 (a scalar problem: vectors of length 1).
    const lab::ResidualFunction g = [](const VectorXd& t) -> VectorXd { return VectorXd::Constant(1, 0.48 * std::sin(t(0)) - 0.24); };
    const lab::JacobianFunction dg = [](const VectorXd& t) -> MatrixXd { return MatrixXd::Constant(1, 1, 0.48 * std::cos(t(0))); };
    const lab::NewtonResult res = lab::newton_raphson(g, dg, VectorXd::Zero(1), 1e-12, 10);
    REQUIRE(res.history.size() >= 4);
    const double expected[4] = {0.0, 28.6479, 29.9912, 30.0};
    for (std::size_t k = 0; k < 4; ++k) {
        REQUIRE(res.history[k].theta.size() == 1);
        CHECK_SCALAR_NEAR(res.history[k].theta(0) * 180.0 / PI, expected[k], 1e-4);
    }
    CHECK(res.history[0].k == 0);
    REQUIRE(res.history.back().g.size() == 1);
    CHECK(std::abs(res.history.back().g(0)) <= 1e-12);
}

TEST_CASE("test_newton_raphson_vector_2r") {
    // Workshop P4(c): coordinate NR for the 2R arm of MR Example 6.1 converges to the OTHER solution.
    const Vector2d xd(0.5 * std::sqrt(3.0) - 0.5, 0.5 + 0.5 * std::sqrt(3.0));
    const lab::ResidualFunction g = [xd](const VectorXd& t) -> VectorXd {
        return VectorXd(Vector2d(std::cos(t(0)) + std::cos(t(0) + t(1)), std::sin(t(0)) + std::sin(t(0) + t(1))) - xd);
    };
    const lab::JacobianFunction J = [](const VectorXd& t) -> MatrixXd {
        MatrixXd m(2, 2);
        m << -std::sin(t(0)) - std::sin(t(0) + t(1)), -std::sin(t(0) + t(1)),
             std::cos(t(0)) + std::cos(t(0) + t(1)), std::cos(t(0) + t(1));
        return m;
    };
    const lab::NewtonResult res = lab::newton_raphson(g, J, d2r({0, 30}), 1e-10, 20);
    REQUIRE(res.history.size() >= 2);
    CHECK_MAT_NEAR(res.history[1].theta, Vector2d(-std::sqrt(3.0), 30 * PI / 180 + 4.7320508), 1e-6);
    CHECK_MAT_NEAR(wrap(res.theta), d2r({120, -90}), 1e-8);
}

TEST_CASE("test_ik_newton_body_mr_example_6_1") {
    // MR Example 6.1 table (p. 230): (0, 30) -> (34.23, 79.18) -> (29.98, 90.22) -> (30.00, 90.00).
    const Model m = model("mr_2r_point");
    const Matrix4d Tsd = mr::fkin_body(m.M, m.B, d2r({30, 90}));
    const lab::IKTableResult res = lab::ik_newton_body(m.B, m.M, Tsd, d2r({0, 30}), 1e-3, 1e-4);
    CHECK(res.success);
    REQUIRE(res.table.size() == 4);
    MatrixXd got(4, 2), expected(4, 2);
    for (std::size_t k = 0; k < 4; ++k) {
        REQUIRE(res.table[k].theta.size() == 2);
        got.row(static_cast<Eigen::Index>(k)) = res.table[k].theta.transpose() * 180.0 / PI;
    }
    expected << 0, 30, 34.23, 79.18, 29.98, 90.22, 30.00, 90.00;
    CHECK_MAT_NEAR(got, expected, 6e-3);
    const sbeg::Vector6d& V0 = res.table[0].V;
    CHECK_MAT_NEAR(Vector3d(V0(2), V0(3), V0(4)), Vector3d(1.571, 0.498, 1.858), 6e-4);
    CHECK_MAT_NEAR(Vector2d(res.table[1].w_err, res.table[1].v_err), Vector2d(0.115, 0.131), 6e-4);
}

TEST_CASE("test_ik_newton_body_matches_mr") {  // pytest: parametrized over course_arm, eod_6r_ugv
    for (const std::string name : {"course_arm", "eod_6r_ugv"}) {
        const Model m = model(name);
        for (const auto& c : cases()["ik_newton_body_matches_mr"][name]) {
            const VectorXd q = as_vec(c["q"]), q0 = as_vec(c["q0"]);
            INFO(name << ", q = " << q.transpose());
            const Matrix4d T = mr::fkin_space(m.M, m.S, q);
            const lab::IKTableResult res = lab::ik_newton_body(m.B, m.M, T, q0, 1e-6, 1e-7);
            const mr::IKResult ref = mr::ikin_body(m.B, m.M, T, q0, 1e-6, 1e-7);
            CHECK(res.success == ref.success);
            CHECK_MAT_NEAR(res.theta, ref.thetalist, 1e-8);
            CHECK_MAT_NEAR(res.theta, as_vec(c["theta_mr"]), 1e-8);  // the Python modern_robotics result
            REQUIRE(!res.table.empty());
            CHECK(res.table[0].i == 0);
            CHECK(sbeg::compare_near(res.table[0].theta, q0, 1e-8).ok);
            if (res.success) CHECK_MAT_NEAR(mr::fkin_space(m.M, m.S, res.theta), T, 1e-6);
        }
    }
}

TEST_CASE("test_ik_newton_space_matches_mr") {
    const Model m = model("course_arm");
    for (const auto& c : cases()["ik_newton_space_matches_mr"]) {
        const VectorXd q = as_vec(c["q"]), q0 = as_vec(c["q0"]);
        INFO("q = " << q.transpose());
        const Matrix4d T = mr::fkin_space(m.M, m.S, q);
        const lab::IKTableResult res = lab::ik_newton_space(m.S, m.M, T, q0, 1e-6, 1e-7);
        const mr::IKResult ref = mr::ikin_space(m.S, m.M, T, q0, 1e-6, 1e-7);
        CHECK(res.success == ref.success);
        CHECK_MAT_NEAR(res.theta, ref.thetalist, 1e-8);
        CHECK_MAT_NEAR(res.theta, as_vec(c["theta_mr"]), 1e-8);  // the Python modern_robotics result
    }
}

TEST_CASE("test_ik_newton_reports_failure") {
    // An unreachable pose: NR must stop after max_iter and report failure, not throw.
    const Model m = model("course_arm");
    Matrix4d T = Matrix4d::Identity();
    T.topRightCorner<3, 1>() = Vector3d(0.5, 0.0, 0.1);
    const lab::IKTableResult res = lab::ik_newton_body(m.B, m.M, T, m.info.q_home, 1e-3, 1e-4, 20);
    CHECK(!res.success);
    CHECK(res.table.size() == 21);
}

// ---------------------------------------------------------------- Checkpoint 4: all-solutions search
TEST_CASE("test_unique_solutions") {
    const Vector3d a(0.1, 3.1, -1.0);
    const std::vector<VectorXd> in = {a, a + 2 * PI * Vector3d(1, 0, -1), a + Vector3d::Constant(1e-6),
                                      a + Vector3d::Constant(0.1)};
    CHECK(lab::unique_solutions(in).size() == 2);
}

TEST_CASE("test_all_solutions_search_finds_analytic_set") {
    // The 200 initial guesses are those of np.random.default_rng(0).uniform(-pi, pi, (200, 6)) (cpp_cases.json);
    // the 20-start call uses the first 20 of them, as Python does with n_starts=20, seed=0.
    const json& C = cases()["all_solutions_search_finds_analytic_set"];
    const MatrixXd starts = as_mat(C["starts"]);
    REQUIRE(starts.rows() == 200);
    REQUIRE(starts.cols() == 6);
    const Model m = model("course_arm");
    const std::vector<std::pair<VectorXd, std::size_t>> poses = {{P8(), 8}, {m.info.q_home, 4}};
    for (const auto& pose : poses) {
        const VectorXd& q = pose.first;
        const std::size_t n = pose.second;
        INFO("q = " << q.transpose());
        const Matrix4d T = mr::fkin_space(m.M, m.S, q);
        const std::vector<VectorXd> analytic = lab::ik_course_arm(T, m.info.geometry);
        const lab::SearchResult res = lab::all_solutions_search(m.S, m.M, T, starts);
        CHECK(res.solutions.size() == res.counts.size());
        int total = 0;
        for (const int k : res.counts) total += k;
        CHECK(total == res.n_converged);
        CHECK(res.n_converged <= 200);
        for (const VectorXd& s : res.solutions)
            CHECK_MESSAGE(contains(analytic, s, 1e-6), "every numerical solution must be one of the analytic ones");
        CHECK_MESSAGE(res.solutions.size() == n, "200 random starts find every analytic branch of this pose");
    }
    const lab::SearchResult r20 = lab::all_solutions_search(m.S, m.M, mr::fkin_space(m.M, m.S, P8()), starts.topRows(20));
    // Python: 6 of 8 on the reference machine, checked as 1 <= count < 8. Some starts lie near a singularity (start 12:
    // theta3 = -0.0007 rad, elbow stretched), so the first Newton step through the nearly singular J_b is huge and the
    // branch it ends on depends on round-off (SVD of Eigen vs LAPACK): the same 20 starts reach all 8 branches with
    // the C++ reference build. The C++ test therefore checks 1 <= count <= 8, the counts, the analytic membership of
    // every solution, and prints the count (run with -s to see it).
    const std::vector<VectorXd> analytic20 = lab::ik_course_arm(mr::fkin_space(m.M, m.S, P8()), m.info.geometry);
    CHECK(r20.solutions.size() == r20.counts.size());
    int total20 = 0;
    for (const int k : r20.counts) total20 += k;
    CHECK(total20 == r20.n_converged);
    CHECK(r20.n_converged <= 20);
    CHECK(r20.solutions.size() >= 1);
    CHECK(r20.solutions.size() <= 8);
    for (const VectorXd& s : r20.solutions)
        CHECK_MESSAGE(contains(analytic20, s, 1e-6), "every numerical solution must be one of the analytic ones");
    MESSAGE("20 starts (seed 0) found " << r20.solutions.size() << " of the 8 branches of P8");
}

TEST_CASE("test_select_solution") {
    MatrixXd lim(2, 2);
    lim << -1.0, 1.0, -1.0, 1.0;
    const std::vector<VectorXd> sols = {Vector2d(0.5, 0.5), Vector2d(-0.2, 0.1), Vector2d(2.0, 0.0)};
    const std::optional<VectorXd> best = lab::select_solution(sols, Vector2d(-0.3, 0.0), lim);
    REQUIRE(best.has_value());
    CHECK_MAT_NEAR(*best, Vector2d(-0.2, 0.1), 1e-8);  // assert_allclose rtol 1e-7
    CHECK(!lab::select_solution({Vector2d(2.0, 0.0)}, Vector2d(0, 0), lim).has_value());
    // an angle may need a 2 pi shift to fall inside the limits
    MatrixXd lim2(1, 2);
    lim2 << 0.0, 6.0;
    const std::optional<VectorXd> b2 =
        lab::select_solution({VectorXd::Constant(1, -1.0)}, VectorXd::Constant(1, 5.0), lim2);
    REQUIRE(b2.has_value());
    CHECK_MAT_NEAR(*b2, VectorXd::Constant(1, 2 * PI - 1.0), 1e-7);
}

// ---------------------------------------------------------------- Checkpoint 5: biopsy needle guide
TEST_CASE("test_needle_guide_pose") {
    const Matrix4d T = lab::needle_guide_pose(Vector3d(0.2, 0.06, 0.02), Vector3d(0, 0, -1), 0.01, Vector3d(0, -1, 0));
    CHECK_MAT_NEAR(T.col(2).head<3>(), Vector3d(0, 0, -1), 1e-12);
    CHECK_MAT_NEAR(T.col(0).head<3>(), Vector3d(0, -1, 0), 1e-12);
    CHECK_MAT_NEAR(T.col(3).head<3>(), Vector3d(0.2, 0.06, 0.03), 1e-12);
    const Eigen::Matrix3d R = T.topLeftCorner<3, 3>();
    CHECK_MAT_NEAR(R.transpose() * R, Eigen::Matrix3d::Identity(), 1e-12);
    CHECK_APPROX(R.determinant(), 1.0);
    // hint parallel to the axis
    const Matrix4d T2 = lab::needle_guide_pose(Vector3d(0, 0, 0), Vector3d(1, 0, 0), 0.0, Vector3d(1, 0, 0));
    CHECK_MAT_NEAR(T2.col(0).head<3>(), Vector3d(0, 1, 0), 1e-12);
}

TEST_CASE("test_biopsy_plan") {
    // Checkpoint 5 values: analytic solutions, feasible solutions, selected solution for T1..T5.
    const Model m = model("desktop_6r");
    const MatrixXd& lim = m.info.limits;
    const VectorXd& qh = m.info.q_home;
    const std::vector<int> expected_n = cases()["biopsy_plan"]["expected_n"].get<std::vector<int>>();
    const std::vector<int> expected_feasible = cases()["biopsy_plan"]["expected_feasible"].get<std::vector<int>>();
    const std::vector<lab::BiopsyTarget> targets = lab::load_biopsy_targets();
    REQUIRE(targets.size() == expected_n.size());
    for (std::size_t k = 0; k < targets.size(); ++k) {
        const lab::BiopsyTarget& tgt = targets[k];
        INFO(tgt.id);
        const Matrix4d T = lab::needle_guide_pose(tgt.entry, tgt.axis, 0.01, Vector3d(0, -1, 0));
        const std::vector<VectorXd> sols = lab::ik_course_arm(T, m.info.geometry);
        CHECK(static_cast<int>(sols.size()) == expected_n[k]);
        int feasible = 0;
        for (const VectorXd& s : sols)
            if (lab::select_solution({s}, qh, lim).has_value()) ++feasible;
        CHECK(feasible == expected_feasible[k]);
        const std::optional<VectorXd> best = lab::select_solution(sols, qh, lim);
        REQUIRE(best.has_value());
        CHECK_MAT_NEAR(mr::fkin_space(m.M, m.S, *best), T, 1e-10);
        CHECK(((*best).array() >= lim.col(0).array() - 1e-12).all());
        CHECK(((*best).array() <= lim.col(1).array() + 1e-12).all());
    }
}
