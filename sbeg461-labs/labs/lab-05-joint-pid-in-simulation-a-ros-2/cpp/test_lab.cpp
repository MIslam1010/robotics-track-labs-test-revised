// Auto-checks for SBEG 461 Lab 5 (error dynamics, joint PID in simulation, ROS 2 controller node logic) -- C++ track.
// Mirrors test_lab.py test by test (same names, same order, same tolerances).
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*overshoot*"        (run only some tests, like pytest -k)
// Ground truth: mrcpp (the C++ port of modern_robotics: mass matrix and gravity torques of the course arm), the
// matrix exponential of the companion matrix and SciPy's ODE solver (their values are in cpp_cases.json, computed
// in Python), the textbook values of MR pp. 415-427 and a loop implementation of PID without anti-windup written
// in this file. With the untouched starter, the two checks of the given code (plant data, ROS-free node classes)
// pass, as in Python; every other test fails with "TODO: <function>".
#include "lab.hpp"
#include "ros2/node_logic.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using Eigen::VectorXd;
using sbeg::json;

namespace {
constexpr double PI = 3.14159265358979323846;
constexpr double ENC = 2 * PI / 4096;

const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

/// Python list -> Eigen vector, e.g. vec({1.0, 20.0, 100.0}).
VectorXd vec(std::initializer_list<double> v) {
    VectorXd out(static_cast<Eigen::Index>(v.size()));
    Eigen::Index i = 0;
    for (double x : v) out(i++) = x;
    return out;
}

/// np.allclose(a, b, rtol, atol): same length and |a - b| <= atol + rtol |b| everywhere.
sbeg::NearResult allclose(const VectorXd& a, const VectorXd& b, double atol = 1e-8, double rtol = 1e-5) {
    sbeg::NearResult r;
    std::ostringstream s;
    if (a.size() != b.size()) {
        s << "length mismatch: got " << a.size() << ", expected " << b.size();
        r.msg = s.str();
        return r;
    }
    r.ok = true;
    for (Eigen::Index i = 0; i < a.size(); ++i) {
        if (!(std::abs(a(i) - b(i)) <= atol + rtol * std::abs(b(i)))) {
            r.ok = false;
            s << "first mismatch at index " << i << ": got " << a(i) << ", expected " << b(i) << " (atol " << atol
              << ", rtol " << rtol << ")";
            break;
        }
    }
    r.msg = s.str();
    return r;
}

#define CHECK_ALLCLOSE(a, b, atol)                                         \
    do {                                                                   \
        const ::sbeg::NearResult sbeg_ac_ = allclose((a), (b), (atol));    \
        CHECK_MESSAGE(sbeg_ac_.ok, "allclose(" #a ", " #b ")\n  " << sbeg_ac_.msg); \
    } while (false)

/// e as a time series of the given function evaluated elementwise (np.exp(-t / tau) etc.).
VectorXd decay(const VectorXd& t, double rate) { return (-rate * t.array()).exp().matrix(); }

/// np.std(np.diff(x)) (population standard deviation of the first differences).
double std_of_diff(const VectorXd& x) {
    REQUIRE(x.size() >= 2);
    const VectorXd d = x.tail(x.size() - 1) - x.head(x.size() - 1);
    const double m = d.mean();
    return std::sqrt((d.array() - m).square().mean());
}

double last(const VectorXd& v) { return v(v.size() - 1); }

double max_abs_tail(const VectorXd& v, Eigen::Index n) { return v.tail(n).cwiseAbs().maxCoeff(); }
}  // namespace

// ============================================================================ data and infrastructure
TEST_CASE("test_plant_data_matches_modern_robotics") {
    // Course-arm plants: M = MassMatrix[j, j] and g(q) = GravityForces[j] with the other joints at q_home.
    // Checks the GIVEN loaders (passes with the untouched starter, as in Python).
    const json& c = cases()["plant_data_matches_modern_robotics"];
    const json& arm = c["arm"];
    const Eigen::MatrixXd Slist = sbeg::as_mat(arm["Slist"]);
    const mr::TransformList Mlist = sbeg::as_list<Eigen::Matrix4d>(arm["Mlist"]);
    const mr::InertiaList Glist = sbeg::as_list<mr::Matrix6d>(arm["Glist"]);
    const Eigen::Vector3d g = sbeg::as<Eigen::Vector3d>(arm["g"]);
    const VectorXd q_home = sbeg::as_vec(arm["q_home"]);
    const VectorXd tau_rated = sbeg::as_vec(arm["tau_max"]);
    REQUIRE(c["cases"].size() == 2);  // arm_j1, arm_j2
    for (const auto& k : c["cases"]) {
        const std::string name = k["name"].get<std::string>();
        const int jj = k["j"].get<int>();
        INFO("plant " << name);
        const lab::Plant p = lab::load_plant(name);
        CHECK_MAT_NEAR(p.locked_q, q_home, sbeg::approx_tol(1.0));
        const VectorXd qjs = sbeg::as_vec(k["qj"]);
        REQUIRE(qjs.size() == 4);
        for (Eigen::Index i = 0; i < qjs.size(); ++i) {
            VectorXd q = q_home;
            q(jj) = qjs(i);
            const Eigen::MatrixXd Mm = mr::mass_matrix(q, Mlist, Glist, Slist);
            const VectorXd gf = mr::gravity_forces(q, g, Mlist, Glist, Slist);
            CHECK_SCALAR_NEAR(p.M, Mm(jj, jj), 1e-12);  // M_jj does not depend on q_j
            CHECK_SCALAR_NEAR(lab::plant_gravity(p, qjs(i)), gf(jj), 1e-12);
            CHECK_SCALAR_NEAR(p.M, k["M"][static_cast<std::size_t>(i)].get<double>(), 1e-12);
            CHECK_SCALAR_NEAR(lab::plant_gravity(p, qjs(i)), k["g"][static_cast<std::size_t>(i)].get<double>(), 1e-12);
        }
        REQUIRE(p.tau_max.has_value());
        CHECK_APPROX(*p.tau_max, 0.6 * tau_rated(jj));
    }
    const lab::Plant knee = lab::load_plant("knee"), eod = lab::load_plant("eod_wrist");
    CHECK_APPROX(knee.M, 0.045 + 2.4 * 0.21 * 0.21);
    CHECK_APPROX(knee.g_sin, 2.4 * 9.81 * 0.21);
    CHECK_APPROX(eod.M, 0.008 + 3.5 * 0.10 * 0.10);
    CHECK_APPROX(eod.g_cos, 3.5 * 9.81 * 0.10);
}

TEST_CASE("test_ros2_files_compile_and_import_without_ros") {
    // C++ form of the Python check: the ROS-free node classes (ros2/node_logic.hpp) compile into this program and the
    // given parts run without ROS (passes with the untouched starter, as in Python).
    const auto hold = node_logic::make_reference("hold", 0.3);
    CHECK(hold(5.0).first == 0.3);
    const auto step = node_logic::make_reference("step", -0.6, -0.3, 0.5);
    CHECK(step(0.49).first == -0.6);
    CHECK(step(0.5).first == -0.3);
    CHECK_THROWS_AS(node_logic::make_reference("ramp", 0.0), std::invalid_argument);
    node_logic::JointSimLogic sim(lab::load_plant("arm_j2"), 0.005, -0.6, 0.0, 10);
    sim.apply(100.0);
    CHECK(std::get<2>(sim.state()) == 1.8);  // clipped to the plant's tau_max
    sim.step();
    CHECK(sim.k == 1);
}

// ============================================================================ Part A: error dynamics
TEST_CASE("test_second_order_params") {
    auto [z1, w1] = lab::second_order_params(20.0, 100.0);  // MR p. 418, Ki = 100
    CHECK_APPROX(z1, 1.0);
    CHECK_APPROX(w1, 10.0);
    auto [z2, w2] = lab::second_order_params(20.0, 400.0);  // case III
    CHECK_APPROX(z2, 0.5);
    CHECK_APPROX(w2, 20.0);
    auto [z, wn] = lab::second_order_params(20.0, 44.4);
    CHECK_SCALAR_NEAR(z, 1.5, 1e-3);
    CHECK_APPROX(wn, std::sqrt(44.4));
    CHECK_THROWS_AS(lab::second_order_params(1.0, 0.0), std::invalid_argument);
}

TEST_CASE("test_char_roots_and_stability") {
    const Eigen::VectorXcd r = lab::char_roots(vec({1.0, 20.0, 100.0}));
    REQUIRE(r.size() == 2);
    for (Eigen::Index i = 0; i < r.size(); ++i) {
        INFO("root " << r(i));
        CHECK(std::abs(r(i) - std::complex<double>(-10.0, 0.0)) <= 1e-6 + 1e-5 * 10.0);  // np.allclose, atol 1e-6
    }
    CHECK(lab::is_stable(vec({1, 1, 1})));
    CHECK_FALSE(lab::is_stable(vec({1, -1, 1})));
    CHECK_FALSE(lab::is_stable(vec({1, 0, 4})));
    // cubic: stable iff all coefficients > 0 and a2 a1 > a0 (MR p. 410); 50 random cubics (rng 1 of test_lab.py)
    REQUIRE(cases()["char_roots_and_stability"].size() == 50);
    for (const auto& c : cases()["char_roots_and_stability"]) {
        const double a2 = c["a2"].get<double>(), a1 = c["a1"].get<double>(), a0 = c["a0"].get<double>();
        INFO("s^3 + " << a2 << " s^2 + " << a1 << " s + " << a0);
        CHECK(lab::is_stable(vec({1.0, a2, a1, a0})) == (a2 * a1 > a0));
    }
}

TEST_CASE("test_unit_error_response_matches_matrix_exponential") {
    // Python: parametrized over zeta = 0, 0.2, 0.5, 1, 1.5, 3 (6 pytest items); reference e = [1 0] expm(A t) e_1.
    const json& all = cases()["unit_error_response_matches_matrix_exponential"];
    REQUIRE(all.size() == 6);
    for (const auto& c : all) {
        const double zeta = c["zeta"].get<double>(), wn = c["wn"].get<double>();
        const VectorXd t = sbeg::as_vec(c["t"]);  // np.linspace(0, 2, 41)
        REQUIRE(t.size() == 41);
        INFO("zeta = " << zeta);
        CHECK_ALLCLOSE(lab::unit_error_response(zeta, wn, t), sbeg::as_vec(c["e"]), 1e-9);
    }
}

TEST_CASE("test_overshoot_formula_and_inverse") {
    CHECK_SCALAR_NEAR(lab::overshoot_from_zeta(0.1), 73, 0.5);  // MR p. 415
    CHECK_SCALAR_NEAR(lab::overshoot_from_zeta(0.5), 16, 0.5);
    CHECK_SCALAR_NEAR(lab::overshoot_from_zeta(0.8), 1.5, 0.05);
    CHECK(lab::overshoot_from_zeta(1.0) == 0.0);
    CHECK(lab::overshoot_from_zeta(2.0) == 0.0);
    for (double z : {0.05, 0.3, 0.7, 0.95}) {
        INFO("zeta = " << z);
        CHECK_SCALAR_NEAR(lab::zeta_from_overshoot(lab::overshoot_from_zeta(z)), z, 1e-12);
    }
}

TEST_CASE("test_step_metrics_first_and_second_order") {
    const VectorXd t = VectorXd::LinSpaced(300001, 0.0, 3.0);
    lab::StepMetrics m = lab::step_metrics(t, decay(t, 1.0 / 0.5));
    CHECK(m.overshoot == 0.0);
    CHECK_SCALAR_NEAR(m.ess, 0.0, 1e-2);
    CHECK_SCALAR_NEAR(lab::step_metrics(t, decay(t, 1.0 / 0.5), 0.0).settling_time, 0.5 * std::log(50.0), 1e-4);
    const json& ref = cases()["step_metrics_first_and_second_order"];
    REQUIRE(ref.size() == 3);
    std::size_t idx = 0;
    for (double z : {0.2, 0.5, 0.8}) {
        INFO("zeta = " << z);
        const VectorXd e = lab::unit_error_response(z, 10.0, t);
        REQUIRE(e.size() == t.size());
        m = lab::step_metrics(t, e, 0.0);
        CHECK_SCALAR_NEAR(m.overshoot, lab::overshoot_from_zeta(z), 1e-3);
        Eigen::Index bad = -1;
        for (Eigen::Index k = 0; k < e.size(); ++k)
            if (std::abs(e(k)) > 0.02) bad = k;
        REQUIRE(bad >= 0);
        REQUIRE(bad + 1 < t.size());
        CHECK_APPROX(m.settling_time, t(bad + 1));
        // the same numbers as the Python reference
        CHECK_SCALAR_NEAR(m.overshoot, ref[idx]["overshoot"].get<double>(), 1e-6);
        CHECK_APPROX(m.settling_time, ref[idx]["settling_time"].get<double>());
        ++idx;
    }
    // offset response: e from 2 to ess = 1 with overshoot
    const VectorXd e = (1.0 + lab::unit_error_response(0.5, 10.0, t).array()).matrix();
    CHECK_SCALAR_NEAR(lab::step_metrics(t, e).overshoot, 16.3, 0.1);
    const VectorXd t100 = t.head(100);
    CHECK(std::isnan(lab::step_metrics(t100, decay(t100, 1.0), 0.0).settling_time));
}

// ============================================================================ Part B: velocity inputs
TEST_CASE("test_pi_velocity_gains") {
    auto [kp1, ki1] = lab::pi_velocity_gains(1.0, 10.0);
    CHECK_APPROX(kp1, 20.0);
    CHECK_APPROX(ki1, 100.0);
    auto [kp2, ki2] = lab::pi_velocity_gains(0.5, 20.0);
    CHECK_APPROX(kp2, 20.0);
    CHECK_APPROX(ki2, 400.0);
}

TEST_CASE("test_p_control_setpoint_and_ramp") {
    const double dt = 1e-3, kp = 20.0;
    const VectorXd t = lab::time_grid(2.0, dt);
    const VectorXd Z = VectorXd::Zero(t.size());
    lab::VelocitySim r = lab::simulate_velocity_joint(t, Z, Z, kp, 0.0, -1.0);
    VectorXd decay_ref(t.size());
    for (Eigen::Index k = 0; k < t.size(); ++k) decay_ref(k) = std::pow(1 - kp * dt, static_cast<double>(k));
    CHECK_ALLCLOSE(r.e, decay_ref, 1e-8);  // discrete first-order decay
    CHECK_ALLCLOSE(r.u, (kp * r.e).eval(), 1e-8);
    const auto [qd, vd] = lab::ramp_reference(t, 0.0, 0.5);
    r = lab::simulate_velocity_joint(t, qd, vd, kp, 0.0, 0.0);
    CHECK_SCALAR_NEAR(last(r.e), 0.5 / kp, 1e-9);  // ess = c / Kp (MR p. 417)
}

TEST_CASE("test_pi_and_feedforward_remove_ramp_error") {
    const VectorXd t = lab::time_grid(3.0, 1e-3);
    const auto [qd, vd] = lab::ramp_reference(t, 0.0, 0.5);
    lab::VelocitySim r = lab::simulate_velocity_joint(t, qd, vd, 20.0, 100.0, -0.2);
    CHECK(std::abs(last(r.e)) < 1e-6);
    r = lab::simulate_velocity_joint(t, qd, vd, 20.0, 0.0, 0.0, true);
    CHECK(r.e.cwiseAbs().maxCoeff() < 1e-12);  // MR Eq. 11.15, no initial error
    r = lab::simulate_velocity_joint(t, qd, vd, 20.0, 100.0, -0.2, true);
    CHECK(std::abs(last(r.e)) < 1e-6);
}

TEST_CASE("test_velocity_saturation_and_anti_windup") {
    const VectorXd t = lab::time_grid(4.0, 1e-3);
    const auto [qd, vd] = lab::step_reference(t, 0.0, 1.0);
    const lab::VelocitySim r = lab::simulate_velocity_joint(t, qd, vd, 20.0, 100.0, 0.0, false, 1.4);
    CHECK(r.u.cwiseAbs().maxCoeff() <= 1.4 + 1e-12);
    // reference loop without anti-windup
    double q = 0.0, E = 0.0, emin = 1.0;
    for (Eigen::Index k = 0; k < t.size(); ++k) {
        const double e = qd(k) - q;
        E += e * 1e-3;
        q += std::clamp(20 * e + 100 * E, -1.4, 1.4) * 1e-3;
        emin = std::min(emin, qd(k) - q);
    }
    CHECK(r.e.minCoeff() > emin + 0.05);  // far less overshoot
    CHECK(std::abs(last(r.e)) < 1e-4);
}

// ============================================================================ Part C: torque PID
TEST_CASE("test_pd_gains_reproduce_mr_fig_11_13") {
    const auto [kp, kd] = lab::pd_gains(0.5, 0.1, 1.0, 2.1);
    CHECK_APPROX(kp, 2.205);  // MR Fig. 11.13 caption
    CHECK_APPROX(kd, 2.0);
    CHECK_APPROX(lab::ki_max(0.5, 0.1, kp, kd), 2.1 * 2.205 / 0.5);
    CHECK_ALLCLOSE(lab::pid_char_poly(0.5, 0.1, kp, 0.0, kd), vec({1, 4.2, 4.41}), 1e-8);
    CHECK_ALLCLOSE(lab::pid_char_poly(0.5, 0.1, kp, 1.0, kd), vec({1, 4.2, 4.41, 2.0}), 1e-8);
}

TEST_CASE("test_ki_bound_is_the_stability_boundary") {
    // Python: parametrized over the plants (4 pytest items).
    for (const char* name : {"arm_j2", "knee", "eod_wrist", "mr_link"}) {
        INFO("plant " << name);
        const lab::Plant p = lab::load_plant(name);
        const auto [kp, kd] = lab::pd_gains(p.M, p.b, 1.0, 15.0);
        const double km = lab::ki_max(p.M, p.b, kp, kd);
        CHECK(lab::is_stable(lab::pid_char_poly(p.M, p.b, kp, 0.9 * km, kd)));
        CHECK_FALSE(lab::is_stable(lab::pid_char_poly(p.M, p.b, kp, 1.1 * km, kd)));
    }
}

TEST_CASE("test_pd_gravity_error") {
    const std::vector<std::tuple<const char*, double, double>> runs = {
        {"mr_link", 0.0, 2.205}, {"arm_j2", -0.6, 9.62}, {"eod_wrist", 0.3, 10.0}, {"knee", 1.0, 50.0}};
    for (const auto& run : runs) {
        const std::string name = std::get<0>(run);
        const double qd = std::get<1>(run), kp = std::get<2>(run);
        INFO("plant " << name);
        const lab::Plant p = lab::load_plant(name);
        const double e = lab::pd_gravity_error(p, kp, qd);
        CHECK_SCALAR_NEAR(kp * e, lab::plant_gravity(p, qd - e), 1e-10);  // MR Eq. 11.27 at rest
    }
    CHECK(lab::pd_gravity_error(lab::load_plant("arm_j1"), 5.0, 0.3) == 0.0);
}

TEST_CASE("test_joint_pid_terms_and_differencing") {
    lab::JointPID c(2.0, 3.0, 0.5, 0.01);
    c.reset(0.0);
    double tau = c.update(1.0, 0.0, 0.0);  // first call: qdot = (0 - 0)/dt = 0
    CHECK_APPROX(tau, 2.0 * 1 + 3.0 * 0.01 + 0.5 * 0);
    tau = c.update(1.0, 0.2, 0.1);  // qdot = 10, e = 0.9, e_dot = -9.8, eint = 0.019
    CHECK_APPROX(tau, 2 * 0.9 + 3 * 0.019 + 0.5 * (0.2 - 10.0));
    CHECK_APPROX(c.eint, 0.019);
    CHECK_APPROX(c.terms.P, 1.8);
    CHECK_APPROX(c.terms.D, -4.9);
    tau = c.update(1.0, 0.0, 0.2, 1.0, 0.7);  // qdot = 1.0, tau_ff = 0.7
    CHECK_APPROX(tau, 2 * 0.8 + 3 * (0.019 + 0.008) + 0.5 * (-1.0) + 0.7);
    c.reset();
    CHECK(c.eint == 0.0);
    CHECK_APPROX(c.update(0.0, 0.0, 5.0), -10 + 3 * (-0.05));
}

TEST_CASE("test_joint_pid_saturation_and_anti_windup") {
    lab::JointPID c(10.0, 50.0, 0.0, 0.01, 1.0);
    c.reset(0.0);
    for (int i = 0; i < 20; ++i) {
        CHECK_APPROX(c.update(1.0, 0.0, 0.0, 0.0), 1.0);
        CHECK(c.saturated);
    }
    CHECK_APPROX(c.eint, 0.0);  // clamped
    lab::JointPID c2(10.0, 50.0, 0.0, 0.01, 1.0, false);
    c2.reset(0.0);
    for (int i = 0; i < 20; ++i) c2.update(1.0, 0.0, 0.0, 0.0);
    CHECK_APPROX(c2.eint, 0.2);
    // leaving saturation: an error of the opposite sign integrates even while clipped
    c.eint = 0.5;
    c.update(0.0, 0.0, 0.2, 0.0);
    CHECK_APPROX(c.eint, 0.5 - 0.002);
}

TEST_CASE("test_simulate_joint_linear_matches_analytic") {
    lab::Plant p = lab::load_plant("arm_j1");
    p.tau_max.reset();  // Python: dict(load_plant("arm_j1"), tau_max=None)
    const double zeta = 0.6, wn = 12.0;
    const auto [kp, kd] = lab::pd_gains(p.M, p.b, zeta, wn);
    const VectorXd t = lab::time_grid(1.5, 1e-4);
    const auto [qd, vd] = lab::step_reference(t, 0.0, 0.0);
    lab::JointPID ctrl(kp, 0.0, kd, 1e-4);
    const lab::JointSim r = lab::simulate_joint(p, ctrl, t, qd, vd, -0.4);
    CHECK_ALLCLOSE((r.e / 0.4).eval(), lab::unit_error_response(zeta, wn, t), 2e-3);
    CHECK(r.q(0) == -0.4);
    CHECK(r.dq(0) == 0.0);
    CHECK_ALLCLOSE(r.tau, (kp * r.e - kd * r.dq).eval(), 1e-8);
}

TEST_CASE("test_simulate_joint_matches_continuous_ode_with_gravity") {
    const lab::Plant p = lab::load_plant("eod_wrist");
    const auto [kp, kd] = lab::pd_gains(p.M, p.b, 1.0, 15.0);
    const VectorXd t = lab::time_grid(1.0, 2e-4);
    const auto [qd, vd] = lab::step_reference(t, 0.5, 0.5);
    lab::JointPID ctrl(kp, 0.0, kd, 2e-4);
    const lab::JointSim r = lab::simulate_joint(p, ctrl, t, qd, vd, 0.0);
    // reference: solve_ivp(rtol 1e-10) of the continuous closed loop, computed in Python (cpp_cases.json)
    const json& c = cases()["simulate_joint_matches_continuous_ode_with_gravity"];
    const VectorXd q_ode = sbeg::as_vec(c["q_ode"]);
    REQUIRE(r.q.size() == q_ode.size());
    CHECK((q_ode - r.q).cwiseAbs().maxCoeff() < 2e-3);
    CHECK_SCALAR_NEAR(last(r.e), lab::pd_gravity_error(p, kp, 0.5), 1e-4);
    CHECK_SCALAR_NEAR(last(r.e), c["e_final"].get<double>(), 1e-9);  // the Python reference, same algorithm
}

TEST_CASE("test_mr_fig_11_13_pd_and_pid") {
    const lab::Plant p = lab::load_plant("mr_link");
    const VectorXd t = lab::time_grid(10.0, 1e-3);
    const auto [qd, vd] = lab::step_reference(t, 0.0, 0.0);
    lab::JointPID c_pd(2.205, 0.0, 2.0, 1e-3), c_pid(2.205, 1.0, 2.0, 1e-3);
    const lab::JointSim pd = lab::simulate_joint(p, c_pd, t, qd, vd, -PI / 2);
    const lab::JointSim pid = lab::simulate_joint(p, c_pid, t, qd, vd, -PI / 2);
    CHECK_SCALAR_NEAR(last(pd.e), lab::pd_gravity_error(p, 2.205, 0.0), 1e-4);
    CHECK(std::abs(last(pid.e)) < 0.01);
    CHECK(pid.e.minCoeff() < -0.05);  // overshoot, then e -> 0
    const json& c = cases()["mr_fig_11_13_pd_and_pid"];
    CHECK_SCALAR_NEAR(last(pd.e), c["pd_e_final"].get<double>(), 1e-9);
    CHECK_SCALAR_NEAR(last(pid.e), c["pid_e_final"].get<double>(), 1e-9);
    CHECK_SCALAR_NEAR(pid.e.minCoeff(), c["pid_e_min"].get<double>(), 1e-9);
}

TEST_CASE("test_arm_shoulder_power_on_sag_and_integral_action") {
    const lab::Plant p = lab::load_plant("arm_j2");
    const auto [kp, kd] = lab::pd_gains(p.M, p.b, 1.0, 20.0);
    const VectorXd t = lab::time_grid(3.0, 1e-3);
    const auto [qd, vd] = lab::step_reference(t, -0.6, -0.6);
    lab::JointPID c_pd(kp, 0.0, kd, 1e-3, p.tau_max);
    const lab::JointSim pd = lab::simulate_joint(p, c_pd, t, qd, vd, -0.6);
    CHECK_SCALAR_NEAR(last(pd.e), lab::pd_gravity_error(p, kp, -0.6), 1e-5);
    const double ki = 0.1 * lab::ki_max(p.M, p.b, kp, kd);
    lab::JointPID c_pid(kp, ki, kd, 1e-3, p.tau_max);
    const lab::JointSim pid = lab::simulate_joint(p, c_pid, t, qd, vd, -0.6);
    CHECK(std::abs(last(pid.e)) < 1e-4);
    CHECK(pid.e.minCoeff() > last(pd.e));
    REQUIRE(p.tau_max.has_value());
    CHECK(pid.tau.cwiseAbs().maxCoeff() <= *p.tau_max + 1e-12);
    const json& c = cases()["arm_shoulder_power_on_sag_and_integral_action"];
    CHECK_SCALAR_NEAR(last(pd.e), c["pd_e_final"].get<double>(), 1e-9);
    CHECK_SCALAR_NEAR(pid.e.minCoeff(), c["pid_e_min"].get<double>(), 1e-9);
}

TEST_CASE("test_integral_gain_above_bound_diverges_in_simulation") {
    const lab::Plant p = lab::load_plant("arm_j2");
    const auto [kp, kd] = lab::pd_gains(p.M, p.b, 1.0, 20.0);
    const double km = lab::ki_max(p.M, p.b, kp, kd);
    const VectorXd t = lab::time_grid(5.0, 1e-3);
    const auto [qd, vd] = lab::step_reference(t, -0.6, -0.6);
    lab::JointPID c_ok(kp, 0.9 * km, kd, 1e-3), c_bad(kp, 1.1 * km, kd, 1e-3);
    const lab::JointSim ok = lab::simulate_joint(p, c_ok, t, qd, vd, -0.6);
    const lab::JointSim bad = lab::simulate_joint(p, c_bad, t, qd, vd, -0.6);
    CHECK(max_abs_tail(ok.e, 1000) < 0.02);
    CHECK(1.0 < max_abs_tail(bad.e, 1000));
}

TEST_CASE("test_encoder_differencing_makes_torque_chatter") {
    const lab::KneeSwing ks = lab::load_knee_swing();
    const lab::Plant p = lab::load_plant("knee");
    const auto [kp, kd] = lab::pd_gains(p.M, p.b, 1.0, 30.0);
    lab::JointPID ca(kp, 0, kd, 1e-3, p.tau_max), cb(kp, 0, kd, 1e-3, p.tau_max);
    const lab::JointSim a = lab::simulate_joint(p, ca, ks.t, ks.qd, ks.qd_dot, ks.qd(0));
    const lab::JointSim b = lab::simulate_joint(p, cb, ks.t, ks.qd, ks.qd_dot, ks.qd(0), 0.0, 4, "difference", ENC);
    CHECK(a.e.cwiseAbs().maxCoeff() < 5.0 * PI / 180.0);
    CHECK(std_of_diff(b.tau) > 10 * std_of_diff(a.tau));
}

TEST_CASE("test_ros_logic_matches_offline_simulation") {
    const double dt = 0.005;
    const int sub = 10;
    const lab::Plant p = lab::load_plant("arm_j2");
    node_logic::PidNodeLogic logic(node_logic::default_controller("arm_j2", 1.0, 20.0, dt, -1.0, 30.0), "joint2",
                                   "step", -0.3, 0.5);
    node_logic::JointSimLogic sim(p, dt, -0.6, 0.0, sub);
    auto [q, dq, tau0] = sim.state();
    (void)tau0;
    VectorXd Q(400);
    std::vector<double> dbg;
    for (Eigen::Index i = 0; i < 400; ++i) {
        Q(i) = q;
        double tau = 0.0;
        std::tie(tau, dbg) = logic.on_joint_state({"joint2"}, {q}, {dq});
        sim.apply(tau);
        std::tie(q, dq, std::ignore) = sim.step();
    }
    const VectorXd t = lab::time_grid(399 * dt, dt);
    const auto [qd, vd] = lab::step_reference(t, -0.6, -0.3, 0.5);
    const auto [kp, kd] = lab::pd_gains(p.M, p.b, 1.0, 20.0);
    lab::JointPID ctrl(kp, 30.0, kd, dt, p.tau_max);
    const lab::JointSim ref = lab::simulate_joint(p, ctrl, t, qd, vd, -0.6, 0.0, sub);
    CHECK_ALLCLOSE(Q, ref.q, 1e-12);
    CHECK_ALLCLOSE(ref.q, sbeg::as_vec(cases()["ros_logic_matches_offline_simulation"]["q"]), 1e-9);
    REQUIRE(dbg.size() == 9);
    CHECK_APPROX(dbg[0], 399 * dt);
}
