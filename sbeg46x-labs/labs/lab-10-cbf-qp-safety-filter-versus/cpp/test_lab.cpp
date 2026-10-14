// Auto-checks for SBEG 46X Lab 10 (CBF-QP safety filter versus saturation on an exoskeleton joint and the Hapkit) --
// C++ track.  Mirrors test_lab.py test by test (20 test cases, same names and order; the Python check that your file
// does not import the reference module has no C++ counterpart: the build itself compiles only your starter.cpp).
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*cbf_filter_1d*,*filter_interval*"      (run only some tests, like pytest -k)
// Ground truth (as in Python): brute-force minimisation over a fine grid (1-D QP), enumeration of the active sets of
// the KKT conditions (2-D QP, written in this file), direct evaluation of the joint and handle dynamics (the barrier
// condition must hold with equality on the bound), the joint inertia and gravity torque of the shared course models
// (modern_robotics MassMatrix / GravityForces via course_models.py, stored in cpp_cases.json), hand-checked Workshop
// numbers, and the reference runs in data/expected.json.  cpp_cases.json holds the random inputs of test_lab.py
// (NumPy generator, seed 4610) and the numbers of the Python reference, so both languages are checked against the
// same values.
//
// C++ only: the four data/provided-code tests (which pass with the untouched Python starter) also check one of your
// functions here (rom_torque_bounds, joint_safety_filter, filter_interval, cbf_qp), so every C++ test case fails
// until you have written something.
#include "lab.hpp"

#include <sbeg/labio.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using lab::INF;
using sbeg::as_mat;
using sbeg::as_vec;
using sbeg::json;

namespace {
constexpr double PI = 3.14159265358979323846;

const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

const json& expected() {
    static const json X = sbeg::load_json(sbeg::lab_path("../data/expected.json"));
    return X;
}

const lab::Params& params() {
    static const lab::Params P = lab::load_params();
    return P;
}

VectorXd vec(std::initializer_list<double> v) {
    VectorXd x(static_cast<Eigen::Index>(v.size()));
    Eigen::Index i = 0;
    for (double e : v) x(i++) = e;
    return x;
}

/// Exact solution of min (u-u_nom)'W(u-u_nom) s.t. A u >= b by enumerating active sets (as kkt_qp in test_lab.py).
/// Returns false if no KKT point exists.
bool kkt_qp(const VectorXd& u_nom, const MatrixXd& A, const VectorXd& b, const MatrixXd& W, VectorXd& best) {
    const int m = static_cast<int>(b.size());
    const MatrixXd Wi = W.inverse();
    double best_cost = INF;
    bool found = false;
    for (unsigned mask = 0; mask < (1u << m); ++mask) {
        std::vector<Eigen::Index> act;
        for (int i = 0; i < m; ++i)
            if (mask & (1u << i)) act.push_back(i);
        VectorXd u = u_nom;
        if (!act.empty()) {
            const Eigen::Index k = static_cast<Eigen::Index>(act.size());
            MatrixXd Aa(k, A.cols());
            VectorXd ba(k);
            for (Eigen::Index r = 0; r < k; ++r) {
                Aa.row(r) = A.row(act[static_cast<std::size_t>(r)]);
                ba(r) = b(act[static_cast<std::size_t>(r)]);
            }
            const MatrixXd S = Aa * Wi * Aa.transpose();
            if (std::abs(S.determinant()) < 1e-12) continue;
            const VectorXd lam = S.partialPivLu().solve(ba - Aa * u_nom);
            if (lam.minCoeff() < -1e-10) continue;
            u = u_nom + Wi * Aa.transpose() * lam;
        }
        if (((A * u - b).array() >= -1e-9).all()) {
            const double c = (u - u_nom).dot(W * (u - u_nom));
            if (c < best_cost) {
                best_cost = c;
                best = u;
                found = true;
            }
        }
    }
    return found;
}

double joint_acc(const lab::JointParams& j, double th, double om, double tau, double tau_h) {
    return (tau + tau_h - j.b * om - lab::gravity(th, j)) / j.M;
}

double max_abs(const VectorXd& v) { return v.cwiseAbs().maxCoeff(); }

/// check_run of test_lab.py: violation_metrics of the run compared with data/expected.json.
lab::ViolationMetrics check_run(const std::string& key, const VectorXd& t, const VectorXd& s, double lo, double hi,
                                double tol_s = 2e-3) {
    INFO("run " << key);
    const lab::ViolationMetrics m = lab::violation_metrics(t, s, lo, hi);
    CHECK_SCALAR_NEAR(m.max_violation, expected().at(key).at("max_violation").get<double>(), tol_s);
    return m;
}
}  // namespace

// ------------------------------------------------------------------------------ data and provided code
TEST_CASE("test_params_match_course_models") {
    const json& C = cases()["params_match_course_models"];
    const lab::Params& P = params();
    for (const std::string name : {"elbow", "knee"}) {
        const lab::JointParams& j = name == "elbow" ? P.elbow : P.knee;
        INFO("joint " << name);
        const json& c = C.at(name);
        CHECK_SCALAR_NEAR(j.M, c.at("M").get<double>(), 1e-9);
        CHECK_SCALAR_NEAR(j.g_s, c.at("g_s").get<double>(), 1e-8);
        CHECK_SCALAR_NEAR(j.g_c, c.at("g_c").get<double>(), 1e-8);
        CHECK(j.tau_max <= c.at("tau_rated").get<double>());
        const double lo = c.at("limits")[0].get<double>(), hi = c.at("limits")[1].get<double>();
        CHECK(lo <= j.rom.first);
        CHECK(j.rom.first < j.rom.second);
        CHECK(j.rom.second <= hi);
        CHECK(j.mech_limits.first == lo);
        CHECK(j.mech_limits.second == hi);
        // C++ only: in the middle of the range, at rest, the barrier bounds bracket the gravity torque, and the upper
        // one is g + M alpha1 alpha2 (theta_max - theta).
        const double mid = 0.5 * (j.rom.first + j.rom.second);
        const auto [t_lo, t_hi] = lab::rom_torque_bounds(mid, 0.0, j, j.cbf.alpha1, j.cbf.alpha2);
        const double g = lab::gravity(mid, j);
        CHECK(t_lo < g);
        CHECK(g < t_hi);
        CHECK_SCALAR_NEAR(t_hi, g + j.M * j.cbf.alpha1 * j.cbf.alpha2 * (j.rom.second - mid), 1e-9);
    }
    const json& hl = C.at("hapkit_linear");
    const lab::HapkitParams& H = P.hapkit;
    CHECK_SCALAR_NEAR(H.m_h, hl.at("m_h").get<double>(), 1e-5);
    CHECK_SCALAR_NEAR(H.k_g, hl.at("k_g").get<double>(), 1e-4);
    CHECK(H.force_max == 2.0);
    CHECK(hl.at("force_max").get<double>() == 2.0);
    CHECK(H.x_range == 0.040);
    CHECK(hl.at("x_range").get<double>() == 0.040);
    CHECK(0 < H.x_max);
    CHECK(H.x_max < H.x_range);
}

TEST_CASE("test_same_joint_as_lab8") {
    const lab::JointParams& E = params().elbow;
    CHECK(E.M == 0.0778);
    CHECK(E.b == 0.1);
    CHECK(E.g_s == 3.3354);
    CHECK(E.tau_max == 6.0);
    CHECK(E.rom.first == 0.2);
    CHECK(E.rom.second == 1.6);
    // C++ only: with no barrier, the filter is the Lab 8 actuator saturation at +-tau_max.
    const auto [tau, info] = lab::joint_safety_filter(E.tau_max + 1.0, {-INF, INF}, {-E.tau_max, E.tau_max});
    CHECK(tau == E.tau_max);
    CHECK(info.active);
    CHECK(info.feasible);
}

TEST_CASE("test_provided_baselines") {
    CHECK(lab::saturate_toward_limit(1.7, 0.5, 0.2, 1.6) == 0.0);
    CHECK(lab::saturate_toward_limit(1.7, -0.5, 0.2, 1.6) == -0.5);
    CHECK(lab::saturate_toward_limit(1.0, 0.5, 0.2, 1.6) == 0.5);
    CHECK_APPROX(lab::virtual_wall_force(0.030, 0.0, 0.025, 200.0), -1.0);
    CHECK(lab::virtual_wall_force(0.020, 0.1, 0.025, 200.0) == 0.0);
    // C++ only: a one-sided interval u <= 0 stops an outward command like the saturation at the limit.
    const lab::IntervalResult r = lab::filter_interval(0.5, -INF, 0.0);
    CHECK(r.u == 0.0);
    CHECK(r.active);
    CHECK(r.feasible);
}

TEST_CASE("test_provided_solve_qp") {
    MatrixXd A(1, 2);
    A << 1.0, 1.0;
    const auto [z, info] = lab::solve_qp(2 * MatrixXd::Identity(2, 2), vec({-2.0, -4.0}), A, vec({1.0}));
    CHECK(info.success);
    CHECK(sbeg::compare_near(z, vec({0.0, 1.0}), 1e-6).ok);
    // C++ only: the same problem as a CBF-QP: u_nom = (1, 2), constraint -u1 - u2 >= -1.
    MatrixXd Ac(1, 2);
    Ac << -1.0, -1.0;
    CHECK_MAT_NEAR(lab::cbf_qp(vec({1.0, 2.0}), Ac, vec({-1.0})), vec({0.0, 1.0}), 1e-6);
}

// ------------------------------------------------------------------------------ Part A
TEST_CASE("test_cbf_filter_1d_closed_form_and_grid") {
    const json& C = cases()["cbf_filter_1d_closed_form_and_grid"];
    REQUIRE(C.size() > 30);
    for (const auto& c : C) {
        const double u_nom = c["u_nom"].get<double>(), h = c["h"].get<double>(), Lf = c["Lf"].get<double>(),
                     Lg = c["Lg"].get<double>(), al = c["alpha"].get<double>();
        INFO("u_nom = " << u_nom << ", h = " << h << ", Lf = " << Lf << ", Lg = " << Lg);
        const double u = lab::cbf_filter_1d(u_nom, h, Lf, Lg, al);
        CHECK(Lf + Lg * u + al * h >= -1e-9);
        // grid = np.linspace(u_nom - 50, u_nom + 50, 200001): the feasible grid point closest to u_nom
        const int N = 200001;
        const double start = u_nom - 50, stop = u_nom + 50, step = (stop - start) / (N - 1);
        double best = INF, u_grid = 0.0;
        for (int i = 0; i < N; ++i) {
            const double g = i == N - 1 ? stop : start + i * step;
            if (Lf + Lg * g + al * h >= 0 && std::abs(g - u_nom) < best) {
                best = std::abs(g - u_nom);
                u_grid = g;
            }
        }
        CHECK(std::abs(u - u_grid) < 1e-3);
        CHECK_SCALAR_NEAR(u, c["u"].get<double>(), 1e-9);  // C++ only: the Python reference value
    }
    CHECK(lab::cbf_filter_1d(1.0, 0.1, 5.0, 0.0, 2.0) == 1.0);      // Lg = 0: the input cannot act
    CHECK_APPROX(lab::cbf_filter_1d(0.1, 0.4, 0.0, -1.0, 2.0), 0.1);  // already safe: unchanged
}

TEST_CASE("test_cbf_filter_1d_workshop_rail") {
    // Workshop Problem 1: ankle trainer theta' = u, h = 0.35 - theta, alpha = 4, u_nom = 0.5
    CHECK_APPROX(lab::cbf_filter_1d(0.5, 0.35 - 0.30, 0.0, -1.0, 4.0), 0.2);
    CHECK_APPROX(lab::cbf_filter_1d(0.5, 0.35 - 0.20, 0.0, -1.0, 4.0), 0.5);
}

TEST_CASE("test_filter_interval") {
    lab::IntervalResult r = lab::filter_interval(3.0, -1.0, 2.0);
    CHECK(r.u == 2.0);
    CHECK(r.active);
    CHECK(r.feasible);
    r = lab::filter_interval(0.5, -1.0, 2.0);
    CHECK(r.u == 0.5);
    CHECK(!r.active);
    CHECK(r.feasible);
    r = lab::filter_interval(0.0, 1.0, 0.5);
    CHECK(r.u == 0.5);
    CHECK(!r.feasible);
    r = lab::filter_interval(-3.0, -INF, INF);
    CHECK(r.u == -3.0);
    CHECK(r.feasible);
}

TEST_CASE("test_cbf_qp_matches_kkt_enumeration") {
    const json& C = cases()["cbf_qp_matches_kkt_enumeration"];
    REQUIRE(C.size() > 0);
    int i = 0;
    for (const auto& c : C) {
        INFO("case " << i++);
        const VectorXd u_nom = as_vec(c["u_nom"]), b = as_vec(c["b"]);
        const MatrixXd A = as_mat(c["A"]), W = as_mat(c["W"]);
        VectorXd ref;
        REQUIRE(kkt_qp(u_nom, A, b, W, ref));
        CHECK_MAT_NEAR(ref, as_vec(c["ref"]), 1e-9);  // this file's KKT enumeration = the Python one
        const VectorXd u = lab::cbf_qp(u_nom, A, b, W);
        CHECK_MAT_NEAR(u, ref, 1e-5);
    }
}

TEST_CASE("test_cbf_qp_one_input_equals_closed_form") {
    const double rows[3][4] = {{0.5, 0.05, 0.0, -1.0}, {2.0, 0.3, -1.0, 0.7}, {-1.0, 0.2, 0.5, 2.0}};
    for (const auto& r : rows) {
        const double u_nom = r[0], h = r[1], Lf = r[2], Lg = r[3];
        INFO("u_nom = " << u_nom << ", Lg = " << Lg);
        const VectorXd u = lab::cbf_qp(vec({u_nom}), MatrixXd::Constant(1, 1, Lg), vec({-Lf - 4.0 * h}));
        REQUIRE(u.size() == 1);
        CHECK_SCALAR_NEAR(u(0), lab::cbf_filter_1d(u_nom, h, Lf, Lg, 4.0), 1e-6);
    }
}

TEST_CASE("test_cbf_qp_with_input_bounds") {
    MatrixXd A(1, 2);
    A << -1.0, -1.0;
    const VectorXd u = lab::cbf_qp(vec({1.0, 1.0}), A, vec({-0.5}), MatrixXd(), vec({0.4, -1.0}), vec({1.0, 1.0}));
    CHECK_MAT_NEAR(u, vec({0.4, 0.1}), 1e-6);
}

// ------------------------------------------------------------------------------ Part B
TEST_CASE("test_rom_bounds_make_second_order_condition_tight") {  // Python: parametrized over [elbow, knee]
    const json& C = cases()["rom_bounds_make_second_order_condition_tight"];
    for (const std::string name : {"elbow", "knee"}) {
        const lab::JointParams& j = name == "elbow" ? params().elbow : params().knee;
        const double lo = j.rom.first, hi = j.rom.second;
        for (const auto& c : C.at(name)) {
            const double th = c["theta"].get<double>(), om = c["omega"].get<double>(), th_h = c["tau_h"].get<double>();
            const double a1 = c["alpha1"].get<double>(), a2 = c["alpha2"].get<double>();
            INFO("joint " << name << ", theta = " << th << ", omega = " << om << ", tau_h = " << th_h);
            const auto [t_lo, t_hi] = lab::rom_torque_bounds(th, om, j, a1, a2, th_h);
            double acc = joint_acc(j, th, om, t_hi, th_h);  // upper barrier h = hi - th: h'' = -acc, h' = -om
            CHECK_SCALAR_NEAR(-acc + (a1 + a2) * (-om) + a1 * a2 * (hi - th), 0.0, 1e-7);
            acc = joint_acc(j, th, om, t_lo, th_h);  // lower barrier h = th - lo
            CHECK_SCALAR_NEAR(acc + (a1 + a2) * om + a1 * a2 * (th - lo), 0.0, 1e-7);
            CHECK(t_lo < t_hi);
            CHECK_SCALAR_NEAR(t_lo, c["tau_lo"].get<double>(), 1e-9);  // C++ only: the Python reference values
            CHECK_SCALAR_NEAR(t_hi, c["tau_hi"].get<double>(), 1e-9);
        }
    }
}

TEST_CASE("test_rom_bounds_workshop_knee") {
    // Workshop Problem 2: knee of load_exo_2r at theta = 0.5 rad, omega = 0.4 rad/s, alpha1 = alpha2 = 5
    const lab::JointParams& KN = params().knee;
    const double t_hi = lab::rom_torque_bounds(0.5, 0.4, KN, 5.0, 5.0).second;
    const double g = KN.g_s * std::sin(0.5) + KN.g_c * std::cos(0.5);
    CHECK_SCALAR_NEAR(t_hi, KN.M * (25 * 0.1 - 10 * 0.4) + KN.b * 0.4 + g, 1e-9);
    CHECK_SCALAR_NEAR(t_hi, -185.88, 0.01);
}

TEST_CASE("test_joint_safety_filter") {
    auto [tau, info] = lab::joint_safety_filter(5.0, {-1.0, 2.0}, {-6.0, 6.0});
    CHECK(tau == 2.0);
    CHECK(info.active);
    CHECK(info.feasible);
    std::tie(tau, info) = lab::joint_safety_filter(1.0, {-1.0, 2.0}, {-6.0, 6.0});
    CHECK(tau == 1.0);
    CHECK(!info.active);
    CHECK(info.feasible);
    std::tie(tau, info) = lab::joint_safety_filter(0.0, {-INF, -8.0}, {-6.0, 6.0});  // needs more braking than available
    CHECK(tau == -6.0);
    CHECK(!info.feasible);
}

TEST_CASE("test_aan_torque") {
    CHECK(lab::aan_torque(0.03, 0.0, 20.0, 0.5, 0.05) == 0.0);
    CHECK_APPROX(lab::aan_torque(0.25, 0.0, 20.0, 0.5, 0.05), -4.0);
    CHECK_APPROX(lab::aan_torque(-0.25, 0.2, 20.0, 0.5, 0.05), 4.0 - 0.1);
}

TEST_CASE("test_violation_metrics") {
    VectorXd t(1000), s(1000);
    for (Eigen::Index i = 0; i < 1000; ++i) {
        t(i) = static_cast<double>(i + 1) * 0.001;
        s(i) = 1.5 + 0.2 * std::sin(2 * PI * t(i));
    }
    const lab::ViolationMetrics m = lab::violation_metrics(t, s, 0.2, 1.6);
    CHECK_SCALAR_NEAR(m.max_violation, 0.1, 1e-5);
    CHECK_SCALAR_NEAR(m.time_outside, 1.0 / 3.0, 0.003);
    CHECK_SCALAR_NEAR(m.s_max, 1.7, 1e-5);
    CHECK(lab::violation_metrics(t, s, -INF, 2.0).max_violation == 0.0);
}

// ------------------------------------------------------------------------------ Part C
TEST_CASE("test_hapkit_bound_tight_and_robust") {
    const lab::HapkitParams& H = params().hapkit;
    for (const auto& c : cases()["hapkit_bound_tight_and_robust"]) {
        const double x = c["x"].get<double>(), v = c["v"].get<double>(), Fh = c["F_h_max"].get<double>();
        INFO("x = " << x << ", v = " << v << ", F_h_max = " << Fh);
        const double F = lab::hapkit_force_bound(x, v, H, 50.0, 50.0, Fh);
        const double acc = (F + Fh - H.b_h * v - H.k_g * x) / H.m_h;  // worst-case operator force F_h = +F_h_max
        CHECK_SCALAR_NEAR(-acc - 100.0 * v + 2500.0 * (H.x_max - x), 0.0, 1e-7);
        CHECK_SCALAR_NEAR(F, c["F"].get<double>(), 1e-9);  // C++ only: the Python reference value
    }
    auto [F, info] = lab::hapkit_safety_filter(0.0, 0.0, 0.0, H, 50.0, 50.0, 1.7);
    CHECK(F == 0.0);
    CHECK(!info.active);
    std::tie(F, info) = lab::hapkit_safety_filter(0.0, 0.025, 0.0, H, 50.0, 50.0, 3.0);
    CHECK(F == -2.0);
    CHECK(!info.feasible);
}

// ------------------------------------------------------------------------------ closed-loop runs
TEST_CASE("test_elbow_step_saturation_vs_cbf") {
    const lab::JointParams& E = params().elbow;
    double viol_none = 0.0, viol_sat = 0.0, viol_cbf = 0.0;
    for (const std::string mode : {"none", "sat", "cbf"}) {
        const lab::JointLog log = lab::simulate_joint(E, "step", mode);
        const lab::ViolationMetrics m =
            check_run("elbow_step_" + mode + "_measured", log.t, log.theta, E.rom.first, E.rom.second);
        CHECK(max_abs(log.tau) <= E.tau_max + 1e-9);
        (mode == "none" ? viol_none : mode == "sat" ? viol_sat : viol_cbf) = m.max_violation;
    }
    CHECK(viol_none > 0.2);
    CHECK(viol_sat > 0.1);
    CHECK(viol_cbf < 1e-4);
}

TEST_CASE("test_elbow_spasm_needs_the_measured_or_bounded_patient_torque") {
    const lab::JointParams& E = params().elbow;
    const std::pair<std::string, std::string> runs[4] = {
        {"sat", "measured"}, {"cbf", "measured"}, {"cbf", "zero"}, {"cbf", "robust"}};
    lab::ViolationMetrics ms[4];
    for (int i = 0; i < 4; ++i) {
        const auto& [mode, thm] = runs[i];
        const lab::JointLog log = lab::simulate_joint(E, "spasm", mode, thm);
        ms[i] = check_run("elbow_spasm_" + mode + "_" + thm, log.t, log.theta, E.rom.first, E.rom.second);
        if (thm == "robust") CHECK(max_abs(log.tau_h) <= E.spasm.tau_h_max);
    }
    CHECK(ms[0].max_violation > 0.05);   // sat
    CHECK(ms[1].max_violation < 1e-4);   // cbf, measured
    CHECK(ms[2].max_violation > 0.05);   // cbf, zero
    CHECK(ms[3].max_violation < 1e-4);   // cbf, robust
    CHECK(ms[3].s_max < ms[1].s_max - 0.05);  // conservative
}

TEST_CASE("test_knee_load_carriage_with_torque_rate_limit") {
    const lab::JointParams& KN = params().knee;
    lab::JointLog log = lab::simulate_joint(KN, "step", "cbf");
    check_run("knee_step_cbf_measured", log.t, log.theta, KN.rom.first, KN.rom.second);
    REQUIRE(log.tau.size() > 1);
    const Eigen::Index n = log.tau.size();
    const VectorXd dtau = log.tau.tail(n - 1) - log.tau.head(n - 1);
    CHECK(max_abs(dtau) <= KN.tau_rate * KN.dt + 1e-9);
    log = lab::simulate_joint(KN, "step", "sat");
    CHECK(lab::violation_metrics(log.t, log.theta, KN.rom.first, KN.rom.second).max_violation > 0.5);  // knee buckles
}

TEST_CASE("test_hapkit_forbidden_region") {
    const lab::HapkitParams& H = params().hapkit;
    double viol[2][2] = {{0, 0}, {0, 0}};
    bool hard_cbf_all_feasible = true;
    const std::string case_names[2] = {"push", "hard_push"}, mode_names[2] = {"wall", "cbf"};
    for (int c = 0; c < 2; ++c) {
        for (int k = 0; k < 2; ++k) {
            const lab::HapkitLog log = lab::simulate_hapkit(H, case_names[c], mode_names[k]);
            viol[c][k] = check_run("hapkit_" + case_names[c] + "_" + mode_names[k], log.t, log.x, -INF, H.x_max, 2e-4)
                             .max_violation;
            CHECK(max_abs(log.F) <= H.force_max + 1e-12);
            if (c == 1 && k == 1)
                hard_cbf_all_feasible = std::all_of(log.feasible.begin(), log.feasible.end(), [](bool b) { return b; });
        }
    }
    CHECK(viol[0][1] < 1e-5);       // push, cbf
    CHECK(viol[0][0] > 0.003);      // push, wall
    CHECK(!hard_cbf_all_feasible);  // the operator out-pushes the 2 N motor
    CHECK(viol[1][1] > 0.005);      // hard push, cbf
}
