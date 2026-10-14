// Auto-checks for SBEG 461 Lab 7 (Lyapunov stability and passivity of PD + gravity compensation; LQR) -- C++ track.
// Mirrors test_lab.py test by test (test_starter_compiles has no C++ counterpart: building is the check).
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*lyapunov*"      (run only some tests, like pytest -k)
// Ground truth: mrcpp (the C++ port of modern_robotics: mass_matrix, vel_quadratic_forces, gravity_forces), the
// closed-form 2R model, finite differences, the identities of MR Sec. 8.1.2 and Sec. 11.4.2.2 and of LQR theory, and
// cpp_cases.json, which holds the numbers of the Python reference implementation and scipy.linalg (including the random
// inputs drawn by test_lab.py), so both languages are checked against the same values.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using sbeg::as_mat;
using sbeg::as_vec;
using sbeg::json;

namespace {
const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

/// The configuration (Python CFG = lab.load_config()), loaded with the given helper.
const json& cfg() {
    static const json C = lab::load_config();
    return C;
}

double num(const json& j) { return j.get<double>(); }

const json& c2() { return cfg().at("planar_2r"); }
VectorXd q_eq() { return as_vec(c2().at("q_eq")); }
double dt() { return num(cfg().at("dt")); }

lab::Robot p2r(bool friction = true) { return lab::plant_2r(cfg(), friction); }
lab::Robot arm() { return lab::plant_course_arm(cfg()); }
lab::Robot robot_by_key(const std::string& key) { return key == "planar_2r" ? p2r() : arm(); }

lab::PDGains pd_gains(const lab::Robot& r, const VectorXd& q) {
    return lab::pd_gains_from_inertia(r, q, num(c2().at("pd").at("zeta")), num(c2().at("pd").at("wn")));
}

std::pair<MatrixXd, MatrixXd> weights_2r() {
    const json& L = c2().at("lqr");
    VectorXd xm(4);
    xm << num(L.at("q_max")), num(L.at("q_max")), num(L.at("dq_max")), num(L.at("dq_max"));
    return lab::bryson_weights(xm, as_vec(L.at("tau_max")));
}

VectorXd vec(std::initializer_list<double> v) {
    VectorXd x(static_cast<Eigen::Index>(v.size()));
    Eigen::Index i = 0;
    for (double e : v) x(i++) = e;
    return x;
}

MatrixXd diag(std::initializer_list<double> v) { return vec(v).asDiagonal(); }

double max_real_eig(const MatrixXd& A) { return Eigen::EigenSolver<MatrixXd>(A, false).eigenvalues().real().maxCoeff(); }

double max_abs(const MatrixXd& A) { return A.cwiseAbs().maxCoeff(); }

/// Lyapunov energy along a run (row k of sim.q, sim.dq).
VectorXd energy_along(const lab::Robot& r, const lab::SimResult& s, const VectorXd& qd, const MatrixXd& Kp) {
    VectorXd V(s.t.size());
    for (Eigen::Index k = 0; k < s.t.size(); ++k)
        V(k) = lab::lyapunov_energy(r, s.q.row(k).transpose(), s.dq.row(k).transpose(), qd, Kp);
    return V;
}

/// Independent check: Jacobians of f(x, u) = (dq, forward_dynamics) by central differences.
std::pair<MatrixXd, MatrixXd> fd_jacobians(const lab::Robot& r, const VectorXd& qe, const VectorXd& ue, double h = 1e-6) {
    const Eigen::Index n = r.n;
    auto f = [&](const VectorXd& x, const VectorXd& u) {
        VectorXd d(2 * n);
        d << x.tail(n), lab::forward_dynamics(r, x.head(n), x.tail(n), u);
        return d;
    };
    VectorXd x0(2 * n);
    x0 << qe, VectorXd::Zero(n);
    MatrixXd A(2 * n, 2 * n), B(2 * n, n);
    for (Eigen::Index k = 0; k < 2 * n; ++k) {
        VectorXd e = VectorXd::Zero(2 * n);
        e(k) = h;
        A.col(k) = (f(x0 + e, ue) - f(x0 - e, ue)) / (2 * h);
    }
    for (Eigen::Index k = 0; k < n; ++k) {
        VectorXd e = VectorXd::Zero(n);
        e(k) = h;
        B.col(k) = (f(x0, ue + e) - f(x0, ue - e)) / (2 * h);
    }
    return {A, B};
}
}  // namespace

// ============================================================================ Part A: passivity and Lyapunov
TEST_CASE("test_models_and_stored_energy") {
    const lab::Robot r = p2r();
    for (const json& c : cases().at("models_and_stored_energy").at("planar_2r")) {
        const VectorXd q = as_vec(c.at("q")), dq = as_vec(c.at("dq"));
        CHECK_MAT_NEAR(lab::mass_matrix(r, q), mr::mass_matrix(q, r.Mlist, r.Glist, r.Slist), 1e-12);
        CHECK_MAT_NEAR(lab::coriolis_vector(r, q, dq), mr::vel_quadratic_forces(q, dq, r.Mlist, r.Glist, r.Slist), 1e-12);
        CHECK_MAT_NEAR(lab::gravity_torque(r, q), mr::gravity_forces(q, r.g, r.Mlist, r.Glist, r.Slist), 1e-12);
        CHECK_MAT_NEAR(lab::mass_matrix(r, q), as_mat(c.at("M")), 1e-12);
        const double y1 = 0.3 * std::sin(q(0)), y2 = 0.6 * std::sin(q(0)) + 0.2 * std::sin(q(0) + q(1));
        const double E = 0.5 * dq.dot(mr::mass_matrix(q, r.Mlist, r.Glist, r.Slist) * dq) + 9.81 * (2.0 * y1 + 1.0 * y2);
        CHECK_SCALAR_NEAR(lab::stored_energy(r, q, dq), E, 1e-12 * std::abs(E));
        CHECK_APPROX(lab::stored_energy(r, q, dq), num(c.at("E")));
    }
    const lab::Robot a = arm();
    const VectorXd q = a.q_home, dq = VectorXd::Constant(6, 0.3);
    const double E = 0.5 * dq.dot(mr::mass_matrix(q, a.Mlist, a.Glist, a.Slist) * dq) + lab::potential_energy(a, q);
    CHECK_SCALAR_NEAR(lab::stored_energy(a, q, dq), E, 1e-12 * std::abs(E));
    CHECK_APPROX(lab::stored_energy(a, q, dq), num(cases().at("models_and_stored_energy").at("course_arm").at("E")));
}

TEST_CASE("test_mass_matrix_partials") {
    const json& C = cases().at("mass_matrix_partials");
    const lab::Robot r = p2r();
    const VectorXd q = as_vec(C.at("q"));
    const std::vector<MatrixXd> dM = lab::mass_matrix_partials(r, q);
    REQUIRE(dM.size() == 2);
    const double a = -r.cf.m2 * r.cf.L1 * r.cf.lc2 * std::sin(q(1));
    MatrixXd ref1(2, 2);
    ref1 << 2 * a, a, a, 0.0;
    CHECK_MAT_NEAR(dM[0], MatrixXd::Zero(2, 2), 1e-8);
    CHECK_MAT_NEAR(dM[1], ref1, 1e-8);
    const auto refs = sbeg::as_mats(C.at("dM"));
    CHECK_MAT_NEAR(dM[1], refs.at(1), 1e-8);
    const lab::Robot ar = arm();
    const VectorXd qa = as_vec(C.at("q_arm"));
    const std::vector<MatrixXd> dMa = lab::mass_matrix_partials(ar, qa);
    REQUIRE(dMa.size() == 6);
    CHECK_MAT_NEAR(dMa[2], as_mat(C.at("dM_arm_k2_fd")), 1e-8);
    CHECK_MAT_NEAR(dMa[0], MatrixXd::Zero(6, 6), 1e-9);
}

TEST_CASE("test_coriolis_matrix_reproduces_velocity_forces") {
    const json& C = cases().at("coriolis_matrix_reproduces_velocity_forces");
    for (const json& c : C.at("cases")) {
        const lab::Robot r = robot_by_key(c.at("robot").get<std::string>());
        const VectorXd q = as_vec(c.at("q")), dq = as_vec(c.at("dq"));
        INFO("robot = " << r.key);
        const MatrixXd Cm = lab::coriolis_matrix(r, q, dq);
        REQUIRE(Cm.rows() == r.n);
        REQUIRE(Cm.cols() == r.n);
        CHECK_MAT_NEAR(Cm * dq, mr::vel_quadratic_forces(q, dq, r.Mlist, r.Glist, r.Slist), 1e-7);
        CHECK_MAT_NEAR(Cm * dq, as_vec(c.at("c")), 1e-7);
    }
    const lab::Robot r = p2r();
    const VectorXd q = vec({0.3, 1.1}), dq = vec({0.7, -0.4});
    const double hh = -r.cf.m2 * r.cf.L1 * r.cf.lc2 * std::sin(q(1));
    MatrixXd Cref(2, 2);
    Cref << hh * dq(1), hh * (dq(0) + dq(1)), -hh * dq(0), 0.0;   // textbook 2R Coriolis matrix
    CHECK_MAT_NEAR(lab::coriolis_matrix(r, q, dq), Cref, 1e-8);
}

TEST_CASE("test_mdot_minus_2c_is_skew_symmetric") {
    for (const json& c : cases().at("mdot_minus_2c_is_skew_symmetric").at("cases")) {
        const lab::Robot r = robot_by_key(c.at("robot").get<std::string>());
        const VectorXd q = as_vec(c.at("q")), dq = as_vec(c.at("dq")), x = as_vec(c.at("x"));
        INFO("robot = " << r.key);
        const MatrixXd N = lab::mass_matrix_dot(r, q, dq) - 2 * lab::coriolis_matrix(r, q, dq);
        const double scale = std::max(1.0, max_abs(N));
        CHECK(max_abs(N + N.transpose()) < 1e-7 * scale);
        CHECK(std::abs(x.dot(N * x)) < 1e-7 * scale * x.dot(x));
    }
}

TEST_CASE("test_mass_matrix_dot_matches_time_derivative") {
    for (const json& c : cases().at("mass_matrix_dot_matches_time_derivative").at("cases")) {
        const lab::Robot r = robot_by_key(c.at("robot").get<std::string>());
        const VectorXd q = as_vec(c.at("q")), dq = as_vec(c.at("dq"));
        const double h = 1e-5;
        const MatrixXd ref = (lab::mass_matrix(r, q + h * dq) - lab::mass_matrix(r, q - h * dq)) / (2 * h);
        INFO("robot = " << r.key);
        CHECK_MAT_NEAR(lab::mass_matrix_dot(r, q, dq), ref, 1e-8);
        CHECK_MAT_NEAR(lab::mass_matrix_dot(r, q, dq), as_mat(c.at("Mdot_fd")), 1e-8);
    }
}

TEST_CASE("test_manipulator_is_passive") {
    const VectorXd tau = vec({3.0, 0.5});
    for (bool friction : {false, true}) {
        INFO("friction = " << friction);
        const lab::Robot r = p2r(friction);
        const lab::SimResult s = lab::simulate(
            r, [&](double, const VectorXd&, const VectorXd&) { return tau; }, vec({0.2, 0.4}), vec({0.0, 0.0}), 1.5,
            0.001);
        const Eigen::Index N = s.t.size();
        const double E0 = lab::stored_energy(r, s.q.row(0).transpose(), s.dq.row(0).transpose());
        const double E1 = lab::stored_energy(r, s.q.row(N - 1).transpose(), s.dq.row(N - 1).transpose());
        const double supplied = tau.dot((s.q.row(N - 1) - s.q.row(0)).transpose());
        double lost = 0.0;
        for (Eigen::Index k = 0; k + 1 < N; ++k) {
            const double d0 = s.dq.row(k).cwiseProduct(s.dq.row(k)).dot(r.b.transpose());
            const double d1 = s.dq.row(k + 1).cwiseProduct(s.dq.row(k + 1)).dot(r.b.transpose());
            lost += 0.5 * (d0 + d1) * (s.t(k + 1) - s.t(k));
        }
        CHECK_SCALAR_NEAR(E1 - E0, supplied - lost, friction ? 1e-5 : 1e-8);
        const json& ref = cases().at("manipulator_is_passive").at(friction ? "friction" : "no_friction");
        CHECK_SCALAR_NEAR(E1 - E0, num(ref.at("dE")), 1e-6);
        if (friction) CHECK(lost > 0.1);
    }
}

TEST_CASE("test_pd_gravity_setpoint_torque") {
    const lab::Robot r = p2r();
    const MatrixXd Kp = diag({50.0, 4.0}), Kd = diag({10.0, 1.0});
    const VectorXd q = vec({0.5, 1.0}), dq = vec({0.2, -0.3}), qd = q_eq();
    const VectorXd tau = lab::pd_gravity_setpoint_torque(q, dq, qd, Kp, Kd, r);
    CHECK_MAT_NEAR(tau, Kp * (qd - q) - Kd * dq + mr::gravity_forces(q, r.g, r.Mlist, r.Glist, r.Slist), 1e-12);
    const lab::Robot m08 = lab::scaled_model(r, 0.8);
    const VectorXd tau08 = lab::pd_gravity_setpoint_torque(q, dq, qd, Kp, Kd, m08);
    CHECK_MAT_NEAR(tau - tau08, 0.2 * lab::gravity_torque(r, q), 1e-12);
    CHECK_MAT_NEAR(tau08, as_vec(cases().at("pd_gravity_setpoint_torque").at("tau08")), 1e-12);
}

TEST_CASE("test_lyapunov_energy_is_positive_definite") {
    const json& C = cases().at("lyapunov_energy_is_positive_definite");
    const lab::Robot r = p2r();
    const VectorXd qe = q_eq();
    const MatrixXd Kp = pd_gains(r, qe).Kp;
    CHECK_MAT_NEAR(Kp, as_mat(C.at("Kp")), 1e-12);
    CHECK_SCALAR_NEAR(lab::lyapunov_energy(r, qe, VectorXd::Zero(2), qe, Kp), 0.0, 1e-15);
    for (const json& c : C.at("cases")) {
        const VectorXd q = as_vec(c.at("q")), dq = as_vec(c.at("dq"));
        const double V = lab::lyapunov_energy(r, q, dq, qe, Kp);
        const VectorXd e = qe - q;
        CHECK_APPROX(V, 0.5 * e.dot(Kp * e) + 0.5 * dq.dot(lab::mass_matrix(r, q) * dq));
        CHECK_APPROX(V, num(c.at("V")));
        CHECK(V > 0);
    }
    const VectorXd q0 = as_vec(c2().at("lyapunov").at("q0"));
    CHECK_APPROX(lab::lyapunov_energy(r, q0, VectorXd::Zero(2), qe, Kp), 0.5 * (qe - q0).dot(Kp * (qe - q0)));
}

TEST_CASE("test_lyapunov_rate_is_minus_dq_kd_dq") {
    const json& C = cases().at("lyapunov_rate_is_minus_dq_kd_dq");
    const lab::Robot r = p2r();
    const VectorXd qe = q_eq();
    const lab::PDGains g = pd_gains(r, qe);
    const MatrixXd B = r.b.asDiagonal();
    for (const json& c : C.at("cases")) {
        const VectorXd q = as_vec(c.at("q")), dq = as_vec(c.at("dq")), ta = as_vec(c.at("tau_any"));
        const VectorXd tau = lab::pd_gravity_setpoint_torque(q, dq, qe, g.Kp, g.Kd, r);
        const double ref = -dq.dot((g.Kd + B) * dq);
        CHECK_SCALAR_NEAR(lab::lyapunov_rate(r, q, dq, qe, g.Kp, tau), ref, std::max(1e-6 * std::abs(ref), 1e-9));
        const double ref_any = dq.dot(ta - lab::gravity_torque(r, q) - B * dq - g.Kp * (qe - q));
        CHECK_SCALAR_NEAR(lab::lyapunov_rate(r, q, dq, qe, g.Kp, ta), ref_any,
                          std::max(1e-6 * std::abs(ref_any), 1e-9));
        CHECK_SCALAR_NEAR(lab::lyapunov_rate(r, q, dq, qe, g.Kp, ta), num(c.at("rate_any")),
                          std::max(1e-6 * std::abs(ref_any), 1e-9));
    }
    const lab::Robot a = arm();
    const lab::PDGains ga = lab::pd_gains_from_inertia(a, a.q_home, 1.0, 8.0);
    const VectorXd q = as_vec(C.at("arm").at("q")), dq = as_vec(C.at("arm").at("dq"));
    const VectorXd tau = lab::pd_gravity_setpoint_torque(q, dq, a.q_home, ga.Kp, ga.Kd, a);
    const double ref = -dq.dot((ga.Kd + MatrixXd(a.b.asDiagonal())) * dq);
    CHECK_APPROX(lab::lyapunov_rate(a, q, dq, a.q_home, ga.Kp, tau), ref);
}

TEST_CASE("test_energy_decreases_along_closed_loop") {
    const lab::Robot r = p2r();
    const VectorXd qe = q_eq();
    const lab::PDGains g = pd_gains(r, qe);
    const json& L = c2().at("lyapunov");
    const lab::SimResult s = lab::simulate(
        r, [&](double, const VectorXd& q, const VectorXd& dq) { return lab::pd_gravity_setpoint_torque(q, dq, qe, g.Kp, g.Kd, r); },
        as_vec(L.at("q0")), as_vec(L.at("dq0")), num(L.at("T")), dt());
    const VectorXd V = energy_along(r, s, qe, g.Kp);
    const Eigen::Index N = V.size();
    CHECK_SCALAR_NEAR(V(0), 2.586473, 1e-6 * 2.586473);
    CHECK((V.tail(N - 1) - V.head(N - 1)).maxCoeff() <= 1e-12);
    CHECK(V(N - 1) < 1e-12);
    CHECK((s.q.row(N - 1).transpose() - qe).norm() < 1e-6);
    const Eigen::Index k = 100;
    const double dV = (V(k + 1) - V(k - 1)) / (2 * dt());
    const double rate = lab::lyapunov_rate(r, s.q.row(k).transpose(), s.dq.row(k).transpose(), qe, g.Kp,
                                           s.tau.row(k).transpose());
    CHECK_SCALAR_NEAR(dV, rate, 2e-3 * std::abs(rate));
    CHECK_APPROX(rate, num(cases().at("energy_decreases_along_closed_loop").at("rate_k100")));
}

TEST_CASE("test_kd_zero_conserves_energy_lasalle_fails") {
    const lab::Robot r = p2r(false);
    const VectorXd qe = q_eq();
    const MatrixXd Kp = pd_gains(r, qe).Kp, Kd = MatrixXd::Zero(2, 2);
    const json& L = c2().at("lyapunov");
    std::vector<double> drift;
    for (double h : {dt(), dt() / 4}) {
        INFO("dt = " << h);
        const lab::SimResult s = lab::simulate(
            r, [&](double, const VectorXd& q, const VectorXd& dq) { return lab::pd_gravity_setpoint_torque(q, dq, qe, Kp, Kd, r); },
            as_vec(L.at("q0")), as_vec(L.at("dq0")), 2.0, h);
        const VectorXd V = energy_along(r, s, qe, Kp);
        const Eigen::Index N = V.size();
        CHECK((V.array() - V(0)).minCoeff() > -1e-9);
        const auto last = static_cast<Eigen::Index>(std::llround(0.5 / h));
        double max_err = 0.0;
        for (Eigen::Index k = N - last; k < N; ++k) max_err = std::max(max_err, (s.q.row(k).transpose() - qe).norm());
        CHECK(max_err > 0.5);
        drift.push_back(V(N - 1) - V(0));
        const Eigen::Index k = N / 3;
        const VectorXd qk = s.q.row(k).transpose(), dqk = s.dq.row(k).transpose();
        CHECK_SCALAR_NEAR(lab::lyapunov_rate(r, qk, dqk, qe, Kp, lab::pd_gravity_setpoint_torque(qk, dqk, qe, Kp, Kd, r)),
                          0.0, 1e-8);
    }
    REQUIRE(drift.size() == 2);
    CHECK_SCALAR_NEAR(drift[0], 0.997701, 1e-4 * 0.997701);
    CHECK_SCALAR_NEAR(drift[0], num(cases().at("kd_zero_conserves_energy_lasalle_fails").at("drift_dt")), 1e-8);
    CHECK(drift[1] < 0.3 * drift[0]);
}

TEST_CASE("test_wrong_gravity_model_offset") {
    const lab::Robot r = p2r();
    const lab::Robot model = lab::scaled_model(r, num(c2().at("wrong_model_mass_scale")));
    const VectorXd qe = q_eq();
    const lab::PDGains g = pd_gains(r, qe);
    const VectorXd q_ss = lab::equilibrium_offset(r, model, qe, g.Kp);
    const VectorXd F = g.Kp * (qe - q_ss) + lab::gravity_torque(model, q_ss) - lab::gravity_torque(r, q_ss);
    CHECK(F.norm() < 1e-10);
    CHECK_MAT_NEAR(q_ss - qe, vec({-0.039760, -0.014271}), 1e-6);
    CHECK_MAT_NEAR(q_ss, as_vec(cases().at("wrong_gravity_model_offset").at("q_ss")), 1e-10);
    const json& L = c2().at("lyapunov");
    const lab::SimResult s = lab::simulate(
        r, [&](double, const VectorXd& q, const VectorXd& dq) { return lab::pd_gravity_setpoint_torque(q, dq, qe, g.Kp, g.Kd, model); },
        as_vec(L.at("q0")), as_vec(L.at("dq0")), num(L.at("T")), dt());
    CHECK_MAT_NEAR(s.q.row(s.q.rows() - 1).transpose(), q_ss, 1e-5);
    CHECK_THROWS_AS(lab::equilibrium_offset(r, model, qe, g.Kp, 1), std::invalid_argument);
}

// ============================================================================ Part B: LQR
TEST_CASE("test_linearize_matches_finite_differences") {
    const json& C = cases().at("linearize_matches_finite_differences");
    for (const lab::Robot& r : {p2r(), arm()}) {
        INFO("robot = " << r.key);
        const VectorXd qe = r.key == "planar_2r" ? q_eq() : r.q_home;
        const lab::Linearization L = lab::linearize(r, qe);
        const Eigen::Index n = r.n;
        REQUIRE(L.A.rows() == 2 * n);
        REQUIRE(L.A.cols() == 2 * n);
        REQUIRE(L.B.rows() == 2 * n);
        REQUIRE(L.B.cols() == n);
        CHECK_MAT_NEAR(L.u_eq, mr::gravity_forces(qe, r.g, r.Mlist, r.Glist, r.Slist), 1e-10);
        CHECK_MAT_NEAR(lab::forward_dynamics(r, qe, VectorXd::Zero(n), L.u_eq), VectorXd::Zero(n), 1e-9);
        const auto [Aref, Bref] = fd_jacobians(r, qe, L.u_eq);
        CHECK_MAT_NEAR(L.A, Aref, 1e-5 * std::max(1.0, max_abs(Aref)));
        CHECK_MAT_NEAR(L.B, Bref, 1e-5 * std::max(1.0, max_abs(Bref)));
        const MatrixXd Apy = as_mat(C.at(r.key).at("A"));
        CHECK_MAT_NEAR(L.A, Apy, 1e-6 * std::max(1.0, max_abs(Apy)));
    }
}

TEST_CASE("test_linearize_hanging_and_hold_pose") {
    const lab::Linearization Lh = lab::linearize(p2r(false), as_vec(c2().at("q_hang")));
    const Eigen::VectorXcd ev = Eigen::EigenSolver<MatrixXd>(Lh.A, false).eigenvalues();
    CHECK_MAT_NEAR(ev.real(), VectorXd::Zero(ev.size()), 1e-6);
    std::vector<double> w;
    for (Eigen::Index i = 0; i < ev.size(); ++i) w.push_back(std::abs(ev(i).imag()));
    std::sort(w.begin(), w.end());
    REQUIRE(w.size() == 4);
    CHECK_MAT_NEAR(Eigen::Map<const VectorXd>(w.data(), 4), vec({3.86986, 3.86986, 9.33178, 9.33178}), 1e-3);
    const lab::Linearization L = lab::linearize(p2r(), q_eq());
    CHECK_SCALAR_NEAR(max_real_eig(L.A), 6.5706, 1e-3);
    CHECK_FALSE(lab::is_hurwitz(L.A));
}

TEST_CASE("test_bryson_weights") {
    const auto [Q, R] = lab::bryson_weights(vec({0.05, 0.05, 0.5, 0.5}), vec({5.0, 2.0}));
    CHECK_MAT_NEAR(Q, diag({400.0, 400.0, 4.0, 4.0}), 1e-9);
    CHECK_MAT_NEAR(R, diag({0.04, 0.25}), 1e-12);
}

TEST_CASE("test_lyapunov_solve_matches_scipy") {
    for (const json& c : cases().at("lyapunov_solve_matches_scipy")) {
        const MatrixXd A = as_mat(c.at("A")), Q = as_mat(c.at("Q")), Pref = as_mat(c.at("P"));
        INFO("n = " << A.rows());
        const MatrixXd P = lab::lyapunov_solve(A, Q);
        REQUIRE(P.rows() == A.rows());
        REQUIRE(P.cols() == A.rows());
        CHECK_MAT_NEAR(P, Pref, 1e-10);
        CHECK_MAT_NEAR(A.transpose() * P + P * A + Q, MatrixXd::Zero(A.rows(), A.rows()), 1e-9);
        CHECK_MAT_NEAR(P, MatrixXd(P.transpose()), 1e-12);
        CHECK(Eigen::SelfAdjointEigenSolver<MatrixXd>(P).eigenvalues().minCoeff() > 0);
    }
}

TEST_CASE("test_lqr_solves_riccati") {
    const lab::Linearization L = lab::linearize(p2r(), q_eq());
    const auto [Q, R] = weights_2r();
    const auto [K, P] = lab::lqr(L.A, L.B, Q, R);
    REQUIRE(K.rows() == 2);
    REQUIRE(K.cols() == 4);
    const MatrixXd res = L.A.transpose() * P + P * L.A - P * L.B * R.partialPivLu().solve(L.B.transpose() * P) + Q;
    CHECK(max_abs(res) < 1e-8 * max_abs(Q));
    CHECK_MAT_NEAR(P, MatrixXd(P.transpose()), 1e-9);
    CHECK(Eigen::SelfAdjointEigenSolver<MatrixXd>(P).eigenvalues().minCoeff() > 0);
    CHECK_MAT_NEAR(K, R.partialPivLu().solve(L.B.transpose() * P), 1e-10);
    CHECK(lab::is_hurwitz(L.A - L.B * K));
    MatrixXd Kref(2, 4);
    Kref << 106.933, -18.295, 16.263, -1.006, 10.246, 41.185, 1.878, 4.395;
    CHECK_MAT_NEAR(K, Kref, 2e-3);
    CHECK_MAT_NEAR(K, as_mat(cases().at("lqr_solves_riccati").at("K")), 1e-8);
    CHECK_MAT_NEAR(P, as_mat(cases().at("lqr_solves_riccati").at("P")), 1e-9);
    const MatrixXd Acl = L.A - L.B * K, Qcl = Q + K.transpose() * R * K;
    CHECK_MAT_NEAR(Acl.transpose() * P + P * Acl + Qcl, MatrixXd::Zero(4, 4), 1e-7 * max_abs(Qcl));
}

TEST_CASE("test_pd_state_feedback_is_linearised_pd") {
    const lab::Robot r = p2r();
    const VectorXd qe = q_eq();
    const lab::PDGains g = pd_gains(r, qe);
    const MatrixXd K0 = lab::pd_state_feedback(r, qe, g.Kp, g.Kd);
    REQUIRE(K0.rows() == 2);
    REQUIRE(K0.cols() == 4);
    auto tau = [&](const VectorXd& x) { return lab::pd_gravity_setpoint_torque(qe + x.head(2), x.tail(2), qe, g.Kp, g.Kd, r); };
    const double h = 1e-6;
    MatrixXd J(2, 4);
    for (Eigen::Index k = 0; k < 4; ++k) {
        VectorXd e = VectorXd::Zero(4);
        e(k) = h;
        J.col(k) = (tau(e) - tau(-e)) / (2 * h);
    }
    CHECK_MAT_NEAR(K0, -J, 1e-6);
    CHECK_MAT_NEAR(K0, as_mat(cases().at("pd_state_feedback_is_linearised_pd").at("K0")), 1e-6);
    const lab::Linearization L = lab::linearize(r, qe);
    CHECK(lab::is_hurwitz(L.A - L.B * K0));
}

TEST_CASE("test_kleinman_converges_to_lqr") {
    const lab::Robot r = p2r();
    const VectorXd qe = q_eq();
    const lab::Linearization L = lab::linearize(r, qe);
    const auto [Q, R] = weights_2r();
    const lab::PDGains g = pd_gains(r, qe);
    const MatrixXd K0 = lab::pd_state_feedback(r, qe, g.Kp, g.Kd);
    const lab::KleinmanResult kr = lab::kleinman_lqr(L.A, L.B, Q, R, K0);
    const json& C = cases().at("kleinman_converges_to_lqr");
    const MatrixXd Pref = as_mat(cases().at("lqr_solves_riccati").at("P"));   // scipy.linalg.solve_continuous_are
    CHECK_MAT_NEAR(kr.P, Pref, 1e-8 * max_abs(Pref));
    CHECK_MAT_NEAR(kr.K, as_mat(cases().at("lqr_solves_riccati").at("K")), 1e-8 * max_abs(kr.K));
    CHECK(kr.iterations >= 3);
    CHECK(kr.iterations <= 15);
    CHECK(kr.iterations == C.at("iterations").get<int>());
    CHECK_THROWS_AS(lab::kleinman_lqr(L.A, L.B, Q, R, MatrixXd::Zero(2, 4)), std::invalid_argument);
}

TEST_CASE("test_lqr_cost_is_optimal") {
    const json& C = cases().at("lqr_cost_is_optimal");
    const lab::Robot r = p2r();
    const VectorXd qe = q_eq();
    const lab::Linearization L = lab::linearize(r, qe);
    const auto [Q, R] = weights_2r();
    const auto [K, P] = lab::lqr(L.A, L.B, Q, R);
    const VectorXd x0 = as_vec(c2().at("lqr").at("x0"));
    const double J = lab::lqr_cost(L.A, L.B, Q, R, K, x0);
    CHECK_SCALAR_NEAR(J, x0.dot(P * x0), 1e-8 * J);
    CHECK_SCALAR_NEAR(J, 0.254473, 1e-5 * 0.254473);
    const lab::PDGains g = pd_gains(r, qe);
    const double Jpd = lab::lqr_cost(L.A, L.B, Q, R, lab::pd_state_feedback(r, qe, g.Kp, g.Kd), x0);
    CHECK_SCALAR_NEAR(Jpd, 0.310228, 1e-5 * 0.310228);
    for (const json& c : C.at("perturbed")) {
        const MatrixXd Kp = as_mat(c.at("K"));
        if (lab::is_hurwitz(L.A - L.B * Kp)) {
            const double Jp = lab::lqr_cost(L.A, L.B, Q, R, Kp, x0);
            CHECK(Jp >= J - 1e-12);
            if (!c.at("J").is_null()) CHECK_APPROX(Jp, num(c.at("J")));
        }
    }
    CHECK_THROWS_AS(lab::lqr_cost(L.A, L.B, Q, R, MatrixXd::Zero(2, 4), x0), std::invalid_argument);
}

TEST_CASE("test_state_feedback_torque_and_nonlinear_closed_loop") {
    const lab::Robot r = p2r();
    const VectorXd qe = q_eq();
    const lab::Linearization L = lab::linearize(r, qe);
    const auto [Q, R] = weights_2r();
    const auto [K, P] = lab::lqr(L.A, L.B, Q, R);
    const VectorXd tau = lab::state_feedback_torque(vec({0.65, 0.85}), vec({0.1, -0.2}), qe, L.u_eq, K);
    CHECK_MAT_NEAR(tau, L.u_eq - K * vec({0.05, -0.05, 0.1, -0.2}), 1e-12);
    const VectorXd x0 = as_vec(c2().at("lqr").at("x0"));
    const lab::PDGains g = pd_gains(r, qe);
    const std::vector<std::pair<std::string, MatrixXd>> laws = {{"lqr", K}, {"pd", lab::pd_state_feedback(r, qe, g.Kp, g.Kd)}};
    std::vector<double> costs;
    for (const auto& law : laws) {
        INFO("law = " << law.first);
        const MatrixXd& KK = law.second;
        const lab::SimResult s = lab::simulate(
            r, [&](double, const VectorXd& q, const VectorXd& dq) { return lab::state_feedback_torque(q, dq, qe, L.u_eq, KK); },
            qe + x0.head(2), x0.tail(2), num(c2().at("lqr").at("T")), dt());
        CHECK((s.q.row(s.q.rows() - 1).transpose() - qe).norm() < 1e-5);
        const double c = lab::run_cost(s, qe, L.u_eq, Q, R);
        const double Jlin = lab::lqr_cost(L.A, L.B, Q, R, KK, x0);
        CHECK_SCALAR_NEAR(c, Jlin, 0.02 * Jlin);
        CHECK_SCALAR_NEAR(c, num(cases().at("state_feedback_torque_and_nonlinear_closed_loop").at("run_cost_" + law.first)),
                          1e-8);
        costs.push_back(c);
    }
    REQUIRE(costs.size() == 2);
    CHECK(costs[0] < costs[1]);
    CHECK_SCALAR_NEAR(costs[0], 0.252415, 1e-4 * 0.252415);
}

TEST_CASE("test_course_arm_lqr") {
    const lab::Robot a = arm();
    const VectorXd qe = a.q_home;
    const lab::Linearization L = lab::linearize(a, qe);
    const json& Lc = cfg().at("course_arm").at("lqr");
    VectorXd xm(12);
    xm << VectorXd::Constant(6, num(Lc.at("q_max"))), VectorXd::Constant(6, num(Lc.at("dq_max")));
    const auto [Q, R] = lab::bryson_weights(xm, num(Lc.at("tau_fraction")) * a.tau_max);
    const auto [K, P] = lab::lqr(L.A, L.B, Q, R);
    REQUIRE(K.rows() == 6);
    REQUIRE(K.cols() == 12);
    const MatrixXd res = L.A.transpose() * P + P * L.A - P * L.B * R.partialPivLu().solve(L.B.transpose() * P) + Q;
    CHECK(max_abs(res) < 1e-7 * max_abs(Q));
    CHECK(lab::is_hurwitz(L.A - L.B * K));
    const MatrixXd Kpy = as_mat(cases().at("course_arm_lqr").at("K"));
    CHECK_MAT_NEAR(K, Kpy, 1e-6 * max_abs(Kpy));
    const json& pdc = cfg().at("course_arm").at("pd");
    const lab::PDGains g = lab::pd_gains_from_inertia(a, qe, num(pdc.at("zeta")), num(pdc.at("wn")));
    const lab::KleinmanResult kr = lab::kleinman_lqr(L.A, L.B, Q, R, lab::pd_state_feedback(a, qe, g.Kp, g.Kd));
    CHECK_MAT_NEAR(kr.K, K, 1e-7 * max_abs(K));
    CHECK(kr.iterations <= 30);
}
