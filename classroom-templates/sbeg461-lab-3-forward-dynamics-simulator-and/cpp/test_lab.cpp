// Auto-checks for SBEG 461 Lab 3 (forward-dynamics simulator, integrator comparison) -- C++ track.
// Mirrors test_lab.py test by test (a parametrized Python test is one test case looping over the robots).
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*mass*"            (run only some tests, like pytest -k)
// Ground truth: mrcpp (the C++ port of modern_robotics: mass_matrix, gravity_forces, forward_dynamics,
// forward_dynamics_trajectory = explicit Euler, MR Sec. 8.5), MR Eq. 8.10 (2R point-mass mass matrix), the SciPy
// DOP853 reference stored in cpp_cases.json, the MR Exercise 8.15(b) trajectory in data/ur5_fall_reference.csv, and
// the numbers of the Python reference in cpp_cases.json (the random states of test_lab.py are stored there).
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <cmath>
#include <initializer_list>
#include <string>
#include <vector>

using Eigen::Matrix2d;
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
const std::vector<std::string> ROBOTS = {"mr_2r_point", "leg_2r", "ur5"};

const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

lab::Robot robot(const std::string& name) { return lab::load_robot(name); }

VectorXd vec2(double a, double b) {
    VectorXd v(2);
    v << a, b;
    return v;
}

double deg2rad(double d) { return d * PI / 180.0; }

/// np.isclose(a, b, atol) with NumPy's default rtol = 1e-5.
bool isclose(double a, double b, double atol) { return std::abs(a - b) <= atol + 1e-5 * std::abs(b); }

/// Total energy K + P along a simulated trajectory.
VectorXd total_energy(const lab::SimResult& s, const lab::Robot& R) {
    VectorXd E(s.t.size());
    for (Eigen::Index k = 0; k < s.t.size(); ++k) {
        const VectorXd th = s.theta.row(k).transpose(), dth = s.dtheta.row(k).transpose();
        const auto KP = lab::energy(th, dth, R.g, R.Mlist, R.Glist, R.Slist);
        E(k) = KP.first + KP.second;
    }
    return E;
}

/// Damped oscillator x = (theta, dtheta) of the integrator tests.
Matrix2d a_osc() {
    Matrix2d A;
    A << 0.0, 1.0, -25.0, -0.4;
    return A;
}

VectorXd f_osc(double /*t*/, const VectorXd& x) { return a_osc() * x; }

/// Require a simulation result of N+1 rows and n joints before indexing it.
void require_shape(const lab::SimResult& s, Eigen::Index N, Eigen::Index n) {
    REQUIRE(s.t.size() == N + 1);
    REQUIRE(s.theta.rows() == N + 1);
    REQUIRE(s.theta.cols() == n);
    REQUIRE(s.dtheta.rows() == N + 1);
    REQUIRE(s.dtheta.cols() == n);
}
}  // namespace

// ------------------------------------------------------------------------- checkpoint 1: M(theta)
TEST_CASE("test_mass_matrix_rnea_matches_mr") {
    for (const auto& name : ROBOTS) {
        INFO("robot = " << name);
        const lab::Robot R = robot(name);
        for (const auto& c : cases()["mass_matrix_rnea_matches_mr"][name]) {
            const VectorXd th = as_vec(c["theta"]);
            const MatrixXd M = lab::mass_matrix_rnea(th, R.Mlist, R.Glist, R.Slist);
            CHECK_MAT_NEAR(M, mr::mass_matrix(th, R.Mlist, R.Glist, R.Slist), 1e-10);
            CHECK_MAT_NEAR(M, as_mat(c["M"]), 1e-10);
        }
    }
}

TEST_CASE("test_mass_matrix_rnea_uses_exactly_n_calls") {
    const lab::Robot R = robot("ur5");
    std::vector<VectorXd> dths, ddths;
    std::vector<Vector3d> gs;
    const lab::RneaFunction counting_rnea = [&](const VectorXd& th, const VectorXd& dth, const VectorXd& ddth,
                                                const Vector3d& g, const mr::Vector6d& Ftip,
                                                const mr::TransformList& Mlist, const mr::InertiaList& Glist,
                                                const MatrixXd& Slist) {
        dths.push_back(dth);
        ddths.push_back(ddth);
        gs.push_back(g);
        return mr::inverse_dynamics(th, dth, ddth, g, Ftip, Mlist, Glist, Slist);
    };
    const json& c = cases()["mass_matrix_rnea_uses_exactly_n_calls"];
    const VectorXd th = as_vec(c["theta"]);
    const MatrixXd M = lab::mass_matrix_rnea(th, R.Mlist, R.Glist, R.Slist, counting_rnea);
    CHECK_MESSAGE(dths.size() == 6, "one RNEA call per column");
    for (std::size_t k = 0; k < dths.size(); ++k) {
        INFO("call " << k);
        CHECK_MAT_NEAR(dths[k], VectorXd::Zero(dths[k].size()), 1e-8);
        CHECK_MAT_NEAR(gs[k], Vector3d::Zero(), 1e-8);
        CHECK(isclose(ddths[k].sum(), 1.0, 1e-8));
    }
    CHECK_MAT_NEAR(M, mr::mass_matrix(th, R.Mlist, R.Glist, R.Slist), 1e-10);
    CHECK_MAT_NEAR(M, as_mat(c["M"]), 1e-10);
}

TEST_CASE("test_mass_matrix_closed_form_matches_mr") {
    for (const auto& name : ROBOTS) {
        INFO("robot = " << name);
        const lab::Robot R = robot(name);
        for (const auto& c : cases()["mass_matrix_closed_form_matches_mr"][name]) {
            const VectorXd th = as_vec(c["theta"]);
            const MatrixXd M = lab::mass_matrix_closed_form(th, R.Mlist, R.Glist, R.Slist);
            CHECK_MAT_NEAR(M, mr::mass_matrix(th, R.Mlist, R.Glist, R.Slist), 1e-10);
            CHECK_MAT_NEAR(M, as_mat(c["M"]), 1e-10);
            REQUIRE(M.rows() == M.cols());
            const MatrixXd Mt = M.transpose();
            CHECK_MAT_NEAR(M, Mt, 1e-12);
            if (name != "mr_2r_point") {
                const Eigen::SelfAdjointEigenSolver<MatrixXd> eig(M);
                CHECK(eig.eigenvalues().minCoeff() > 0.0);
            }
        }
    }
}

TEST_CASE("test_mass_matrix_mr_eq_8_9") {
    // MR Eq. 8.10 with L1 = L2 = m1 = m2 = 1: M = [[3 + 2 c2, 1 + c2], [1 + c2, 1]].
    const lab::Robot R = robot("mr_2r_point");
    for (const double t2 : {0.0, PI / 2, 2.0}) {
        INFO("theta2 = " << t2);
        const double c2 = std::cos(t2);
        Matrix2d expected;
        expected << 3 + 2 * c2, 1 + c2, 1 + c2, 1.0;
        CHECK_MAT_NEAR(lab::mass_matrix_closed_form(vec2(0.3, t2), R.Mlist, R.Glist, R.Slist), expected, 1e-12);
        CHECK_MAT_NEAR(lab::mass_matrix_rnea(vec2(0.3, t2), R.Mlist, R.Glist, R.Slist), expected, 1e-12);
    }
}

// ------------------------------------------------------------------------- checkpoint 2: FD, energy
TEST_CASE("test_forward_dynamics_matches_mr") {
    for (const auto& name : ROBOTS) {
        INFO("robot = " << name);
        const lab::Robot R = robot(name);
        for (const auto& c : cases()["forward_dynamics_matches_mr"][name]) {
            const VectorXd th = as_vec(c["theta"]), dth = as_vec(c["dtheta"]), tau = as_vec(c["tau"]);
            const mr::Vector6d Ftip = as<mr::Vector6d>(c["Ftip"]);
            const VectorXd dd = lab::forward_dynamics(th, dth, tau, R.g, Ftip, R.Mlist, R.Glist, R.Slist);
            CHECK_MAT_NEAR(dd, mr::forward_dynamics(th, dth, tau, R.g, Ftip, R.Mlist, R.Glist, R.Slist), 1e-8);
            CHECK_MAT_NEAR(dd, as_vec(c["ddtheta"]), 1e-8);
            // round trip through inverse dynamics (MR Eq. 8.77)
            REQUIRE(dd.size() == th.size());
            CHECK_MAT_NEAR(mr::inverse_dynamics(th, dth, dd, R.g, Ftip, R.Mlist, R.Glist, R.Slist), tau, 1e-8);
        }
    }
}

TEST_CASE("test_energy_kinetic_and_gravity") {
    // K = 1/2 dth^T M dth and dP/dtheta = g(theta) (gravity torques from mrcpp).
    for (const auto& name : ROBOTS) {
        INFO("robot = " << name);
        const lab::Robot R = robot(name);
        const Eigen::Index n = R.Slist.cols();
        for (const auto& c : cases()["energy_kinetic_and_gravity"][name]) {
            const VectorXd th = as_vec(c["theta"]), dth = as_vec(c["dtheta"]);
            const auto KP = lab::energy(th, dth, R.g, R.Mlist, R.Glist, R.Slist);
            const double K_mr = 0.5 * dth.dot(mr::mass_matrix(th, R.Mlist, R.Glist, R.Slist) * dth);
            CHECK(isclose(KP.first, K_mr, 1e-10));
            CHECK_SCALAR_NEAR(KP.first, c["K"].get<double>(), 1e-9);
            CHECK_SCALAR_NEAR(KP.second, c["P"].get<double>(), 1e-9);
            const double eps = 1e-6;
            VectorXd grad(n);
            for (Eigen::Index i = 0; i < n; ++i) {
                const VectorXd e = VectorXd::Unit(n, i);
                const VectorXd thp = th + eps * e, thm = th - eps * e;
                grad(i) = (lab::energy(thp, dth, R.g, R.Mlist, R.Glist, R.Slist).second -
                           lab::energy(thm, dth, R.g, R.Mlist, R.Glist, R.Slist).second) /
                          (2 * eps);
            }
            CHECK_MAT_NEAR(grad, mr::gravity_forces(th, R.g, R.Mlist, R.Glist, R.Slist), 1e-6);
        }
    }
}

TEST_CASE("test_energy_values_leg") {
    // Leg at theta = (-70, -40) deg, dtheta = (2, -3): K = 1.47994 J, P = -27.50766 J (workshop Problem 1).
    const lab::Robot R = robot("leg_2r");
    const auto KP = lab::energy(vec2(deg2rad(-70), deg2rad(-40)), vec2(2.0, -3.0), R.g, R.Mlist, R.Glist, R.Slist);
    CHECK(isclose(KP.first, 1.4799422, 1e-6));
    CHECK(isclose(KP.second, -27.5076597, 1e-6));
    CHECK_SCALAR_NEAR(KP.first, cases()["energy_values_leg"]["K"].get<double>(), 1e-9);
    CHECK_SCALAR_NEAR(KP.second, cases()["energy_values_leg"]["P"].get<double>(), 1e-9);
}

// ------------------------------------------------------------------------- checkpoint 3: integrators
TEST_CASE("test_euler_step_linear") {
    const VectorXd x = vec2(0.3, -0.2);
    const double h = 0.01;
    const VectorXd expected = (Matrix2d::Identity() + h * a_osc()) * x;
    CHECK_MAT_NEAR(lab::euler_step(f_osc, 0.0, x, h), expected, 1e-14);
}

TEST_CASE("test_semi_implicit_step_linear") {
    const VectorXd x = vec2(0.3, -0.2);
    const double h = 0.01;
    const double v = x(1) + h * (a_osc() * x)(1);
    CHECK_MAT_NEAR(lab::semi_implicit_euler_step(f_osc, 0.0, x, h), vec2(x(0) + h * v, v), 1e-14);
}

TEST_CASE("test_rk4_step_linear_is_fourth_order_taylor") {
    const VectorXd x = vec2(0.3, -0.2);
    const double h = 0.05;
    const Matrix2d hA = h * a_osc();
    const Matrix2d T4 = Matrix2d::Identity() + hA + hA * hA / 2 + hA * hA * hA / 6 + hA * hA * hA * hA / 24;
    const VectorXd expected = T4 * x;
    CHECK_MAT_NEAR(lab::rk4_step(f_osc, 0.0, x, h), expected, 1e-14);
}

TEST_CASE("test_rk4_step_uses_time_argument") {
    // x(t) = t^4 / 4 is integrated exactly by RK4.
    const lab::StateFunction f = [](double t, const VectorXd&) { return VectorXd::Constant(1, t * t * t).eval(); };
    const VectorXd x = lab::rk4_step(f, 1.0, VectorXd::Constant(1, 0.25), 0.5);
    REQUIRE(x.size() == 1);
    CHECK(isclose(x(0), std::pow(1.5, 4) / 4, 1e-14));
}

TEST_CASE("test_simulate_euler_matches_mr_forward_dynamics_trajectory") {
    // Explicit Euler simulation = mr::forward_dynamics_trajectory (intRes = 1), with a hip torque profile and a
    // tip force.
    const lab::Robot R = robot("leg_2r");
    const json& c = cases()["simulate_euler_matches_mr_forward_dynamics_trajectory"];
    const double h = 0.005;
    const int N = 80;
    const lab::TorqueFunction tau_fn = [](double t, const VectorXd&, const VectorXd&) {
        return vec2(6.0 * std::sin(2 * PI * t), -1.0);
    };
    mr::Vector6d Ftip;
    Ftip << 0, 0, 0, 2.0, -5.0, 0;
    const lab::WrenchFunction Ftip_fn = [Ftip](double, const VectorXd&, const VectorXd&) { return Ftip; };
    const VectorXd th0 = vec2(deg2rad(-100.0), deg2rad(-10.0)), dth0 = vec2(0.5, 0.0);
    const lab::SimResult s = lab::simulate(R, th0, dth0, N * h, h, "euler", tau_fn, Ftip_fn);
    require_shape(s, N, 2);
    CHECK(isclose(s.t(N), N * h, 1e-8));
    MatrixXd taumat(N, 2), Ftipmat(N, 6);
    for (int k = 0; k < N; ++k) {
        taumat.row(k) = tau_fn(k * h, th0, dth0).transpose();
        Ftipmat.row(k) = Ftip.transpose();
    }
    const auto ref = mr::forward_dynamics_trajectory(th0, dth0, taumat, R.g, Ftipmat, R.Mlist, R.Glist, R.Slist, h, 1);
    const MatrixXd TH = s.theta.topRows(N), DTH = s.dtheta.topRows(N);
    CHECK_MAT_NEAR(TH, ref.first, 1e-10);
    CHECK_MAT_NEAR(DTH, ref.second, 1e-10);
    CHECK_MAT_NEAR(s.theta, as_mat(c["theta"]), 1e-10);
    CHECK_MAT_NEAR(s.dtheta, as_mat(c["dtheta"]), 1e-10);
}

TEST_CASE("test_observed_orders_leg_swing") {
    // Free swing of the prosthetic leg from (-60, -30) deg: Euler and semi-implicit Euler ~ order 1, RK4 ~ order 4.
    const lab::Robot R = robot("leg_2r");
    const json& c = cases()["observed_orders_leg_swing"];
    const VectorXd th0 = as_vec(c["theta0"]), ref = as_vec(c["ref"]), hs = as_vec(c["hs"]);
    const double tf = c["tf"].get<double>();
    double p[3] = {0.0, 0.0, 0.0};
    const char* methods[3] = {"euler", "semi_implicit", "rk4"};
    for (int m = 0; m < 3; ++m) {
        VectorXd errs(hs.size());
        for (Eigen::Index i = 0; i < hs.size(); ++i) {
            const lab::SimResult s = lab::simulate(R, th0, VectorXd::Zero(2), tf, hs(i), methods[m]);
            REQUIRE(s.theta.rows() >= 1);
            REQUIRE(s.theta.cols() == 2);
            const VectorXd last = s.theta.bottomRows(1).transpose();
            errs(i) = (last - ref).norm();
        }
        p[m] = lab::observed_order(hs, errs);
    }
    INFO("orders: euler " << p[0] << ", semi_implicit " << p[1] << ", rk4 " << p[2]);
    CHECK((0.8 < p[0] && p[0] < 1.3));
    CHECK((0.8 < p[1] && p[1] < 1.3));
    CHECK((3.5 < p[2] && p[2] < 4.5));
}

TEST_CASE("test_observed_order_function") {
    VectorXd hs(3);
    hs << 0.1, 0.05, 0.025;
    const VectorXd errs = (3.0 * hs.array().square()).matrix();
    CHECK(isclose(lab::observed_order(hs, errs), 2.0, 1e-12));
}

// ------------------------------------------------------------------------- checkpoint 4: UR5, energy
TEST_CASE("test_ur5_fall_rk4_vs_reference") {
    // MR Exercise 8.15(b): UR5 falls from theta = 0 under gravity; RK4 with h = 0.01 s over the first second.
    const lab::Robot R = robot("ur5");
    const sbeg::CsvTable ref = sbeg::read_csv(sbeg::lab_path("../data/ur5_fall_reference.csv"));
    REQUIRE(ref.data.cols() == 13);
    const lab::SimResult s = lab::simulate(R, VectorXd::Zero(6), VectorXd::Zero(6), 1.0, 0.01, "rk4");
    require_shape(s, 100, 6);
    int checked = 0;
    for (Eigen::Index r = 0; r < ref.data.rows(); ++r) {
        const double t = ref.data(r, 0);
        if (t > 1.0 + 1e-9) continue;
        const Eigen::Index k = static_cast<Eigen::Index>(std::llround(t / 0.01));
        INFO("t = " << t);
        const VectorXd th = s.theta.row(k).transpose(), dth = s.dtheta.row(k).transpose();
        const VectorXd th_ref = ref.data.block(r, 1, 1, 6).transpose(), dth_ref = ref.data.block(r, 7, 1, 6).transpose();
        CHECK_MAT_NEAR(th, th_ref, 1e-4);
        CHECK_MAT_NEAR(dth, dth_ref, 1e-3);
        ++checked;
    }
    CHECK(checked > 0);
}

TEST_CASE("test_ur5_energy_drift_ranking") {
    // Energy check (zero torque => E constant): explicit Euler gains energy, RK4 conserves it to < 1e-4 J.
    const lab::Robot R = robot("ur5");
    double drift[2] = {0.0, 0.0};
    const char* methods[2] = {"euler", "rk4"};
    for (int m = 0; m < 2; ++m) {
        const lab::SimResult s = lab::simulate(R, VectorXd::Zero(6), VectorXd::Zero(6), 0.5, 0.01, methods[m]);
        require_shape(s, 50, 6);
        const VectorXd E = total_energy(s, R);
        drift[m] = E(E.size() - 1) - E(0);
    }
    INFO("drift: euler " << drift[0] << " J, rk4 " << drift[1] << " J");
    CHECK(drift[0] > 0.05);
    CHECK(std::abs(drift[1]) < 1e-4);
}

TEST_CASE("test_power_balance_with_torque") {
    // dE/dt = dtheta^T tau: E(T) - E(0) equals the work done by the hip torque (trapezoid rule).
    const lab::Robot R = robot("leg_2r");
    const lab::TorqueFunction tau_fn = [](double t, const VectorXd&, const VectorXd&) {
        return vec2(8.0 * std::sin(2 * PI * t), 0.0);
    };
    const double h = 0.002;
    const lab::SimResult s = lab::simulate(R, vec2(deg2rad(-90.0), 0.0), VectorXd::Zero(2), 0.5, h, "rk4", tau_fn);
    require_shape(s, 250, 2);
    const VectorXd E = total_energy(s, R);
    VectorXd power(s.t.size());
    for (Eigen::Index k = 0; k < s.t.size(); ++k) {
        const VectorXd th = s.theta.row(k).transpose(), dth = s.dtheta.row(k).transpose();
        power(k) = dth.dot(tau_fn(s.t(k), th, dth));
    }
    double work = 0.0;
    for (Eigen::Index k = 1; k < power.size(); ++k) work += 0.5 * (power(k) + power(k - 1)) * h;
    INFO("work = " << work << " J, E(T) - E(0) = " << E(E.size() - 1) - E(0) << " J");
    CHECK(std::abs(work) > 0.1);
    CHECK(isclose(E(E.size() - 1) - E(0), work, 2e-4));
}
