// Auto-checks for SBEG 46X Lab 8 (PD vs LQR vs MPC on an exoskeleton joint with torque and range limits) --
// C++ track. Mirrors test_lab.py test by test (21 test cases, same names and order; the Python parametrized tests
// test_scalar_care and test_runs_match_reference are one test case each that loops over the parameters).
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*scalar*,*riccati*"      (run only some tests, like pytest -k)
// Ground truth (as in Python): closed-form results (scalar Riccati roots, the golden-ratio DARE of Workshop
// Problem 2, the LQR gains of a damped double integrator, the ZOH discretisation of M theta'' + b theta' = u),
// direct simulation of the discrete model written in this file, the reference runs data/ref_elbow_*.csv and
// data/expected.json, and cpp_cases.json, which holds the random inputs of test_lab.py and the numbers of the Python
// reference (SciPy's Riccati solvers solve_continuous_are / solve_discrete_are, and the joint inertia and gravity
// torque of the course models from modern_robotics MassMatrix / GravityForces), so both languages are checked
// against the same values.
// np.allclose(a, b, rtol=r, atol=x) is checked with its NumPy meaning: |a - b| <= x + r |b| for every entry
// (NumPy defaults rtol = 1e-5, atol = 1e-8).
//
// The four data / provided-code tests also call your functions (joint_state_space; the PD controller through
// simulate; mpc_qp; metrics), so with the untouched starter every test case fails with "TODO: <function>".
#include "lab.hpp"

#include <sbeg/labio.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using Eigen::Index;
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

const lab::Params& P() {
    static const lab::Params p = lab::load_params();
    return p;
}

const lab::JointParams& E() { return P().elbow; }

const json& EXP() {
    static const json e = sbeg::load_json(sbeg::lab_path("../data/expected.json"));
    return e;
}

double num(const json& v) { return sbeg::as_double(v); }

/// np.allclose(got, expected, rtol, atol): |got - expected| <= atol + rtol |expected| for every entry.
template <typename A, typename B>
sbeg::NearResult allclose(const Eigen::MatrixBase<A>& got_in, const Eigen::MatrixBase<B>& exp_in, double rtol,
                          double atol) {
    MatrixXd got = got_in.template cast<double>(), exp = exp_in.template cast<double>();
    if ((got.rows() == 1 || got.cols() == 1) && (exp.rows() == 1 || exp.cols() == 1) && got.size() == exp.size()) {
        got = Eigen::Map<VectorXd>(got.data(), got.size()).eval();
        exp = Eigen::Map<VectorXd>(exp.data(), exp.size()).eval();
    }
    sbeg::NearResult r = sbeg::compare_near(got, exp, atol);  // shape / NaN checks and the message
    if (!r.ok && got.rows() == exp.rows() && got.cols() == exp.cols() && got.allFinite()) {
        r.ok = ((got - exp).cwiseAbs().array() <= atol + rtol * exp.cwiseAbs().array()).all();
    }
    return r;
}

/// A column vector from a list of numbers.
VectorXd vec(std::initializer_list<double> v) {
    VectorXd out(static_cast<Index>(v.size()));
    Index i = 0;
    for (double x : v) out(i++) = x;
    return out;
}

/// A 1 x 1 matrix.
MatrixXd m11(double x) { return MatrixXd::Constant(1, 1, x); }

/// diag(values).
MatrixXd diag(std::initializer_list<double> v) { return MatrixXd(vec(v).asDiagonal()); }

/// The value of one metric by its Python key.
double metric(const lab::Metrics& m, const std::string& k) {
    if (k == "rom_violation") return m.rom_violation;
    if (k == "v_violation") return m.v_violation;
    if (k == "overshoot") return m.overshoot;
    if (k == "settle_time") return m.settle_time;
    if (k == "peak_tau") return m.peak_tau;
    if (k == "sat_fraction") return m.sat_fraction;
    if (k == "rms_error") return m.rms_error;
    if (k == "effort") return m.effort;
    throw std::invalid_argument("unknown metric " + k);
}

/// Closed-form elbow matrices A, B, Ad, Bd (as elbow_mats() in test_lab.py).
struct ElbowMats {
    MatrixXd A, B, Ad, Bd;
};

ElbowMats elbow_mats() {
    const double M = E().M, b = E().b, h = E().h;
    ElbowMats e;
    e.A = MatrixXd(2, 2);
    e.A << 0.0, 1.0, 0.0, -b / M;
    e.B = MatrixXd(2, 1);
    e.B << 0.0, 1.0 / M;
    const double a = b / M;
    const double ea = std::exp(-a * h);
    e.Ad = MatrixXd(2, 2);
    e.Ad << 1.0, (1 - ea) / a, 0.0, ea;
    e.Bd = MatrixXd(2, 1);
    e.Bd << (h - (1 - ea) / a) / b, (1 - ea) / b;
    return e;
}

/// The reference run data/ref_elbow_<kind>.csv.
sbeg::CsvTable load_ref(const std::string& kind) {
    return sbeg::read_csv(sbeg::lab_path("../data/ref_elbow_" + kind + ".csv"));
}

}  // namespace

#define CHECK_ALLCLOSE(got, expected, rtol, atol)                                                                \
    do {                                                                                                         \
        const ::sbeg::NearResult ac_ = allclose((got), (expected), (rtol), (atol));                              \
        CHECK_MESSAGE(ac_.ok, "np.allclose(" #got ", " #expected ", rtol=" << (rtol) << ", atol=" << (atol)       \
                                                                              << ")\n  " << ac_.msg);           \
    } while (false)

// ------------------------------------------------------------------------------ data and provided code
TEST_CASE("test_params_match_course_models") {
    const json& C = cases()["params_match_course_models"];
    for (const std::string name : {"elbow", "knee"}) {
        INFO("joint " << name);
        const lab::JointParams& j = name == "elbow" ? P().elbow : P().knee;
        const json& cm = C[name];  // course_models.py (modern_robotics MassMatrix / GravityForces)
        CHECK(std::abs(j.M - num(cm["M"])) < 1e-9);
        CHECK(std::abs(j.g_s - num(cm["g_s"])) < 1e-8);
        CHECK(std::abs(j.g_c - num(cm["g_c"])) < 1e-8);
        CHECK(j.tau_max <= num(cm["tau_rated"]));
        const double lo = num(cm["limits"][0]), hi = num(cm["limits"][1]);
        CHECK(lo <= j.rom(0));
        CHECK(j.rom(0) < j.rom(1));
        CHECK(j.rom(1) <= hi);
        double gmax = 0.0;
        for (int i = 0; i <= 200; ++i) {  // np.linspace(*rom, 201)
            const double t = j.rom(0) + (j.rom(1) - j.rom(0)) * i / 200.0;
            gmax = std::max(gmax, std::abs(lab::gravity(t, j)));
        }
        CHECK(j.mpc.g_reserve >= gmax);
        // C++ only: your error model of this joint
        const std::pair<MatrixXd, MatrixXd> AB = lab::joint_state_space(j.M, j.b);
        REQUIRE(AB.first.rows() == 2);
        REQUIRE(AB.first.cols() == 2);
        REQUIRE(AB.second.size() == 2);
        CHECK_SCALAR_NEAR(AB.first(1, 1), -j.b / j.M, 1e-12 * j.b / j.M);
        CHECK_SCALAR_NEAR(AB.second(1), 1.0 / j.M, 1e-12 / j.M);
    }
    CHECK(std::abs(E().M - 0.0778) < 1e-12);
    CHECK(std::abs(E().g_s - 2.0 * 9.81 * 0.17) < 1e-9);
}

TEST_CASE("test_gravity_matches_modern_robotics") {
    const json& C = cases()["gravity_matches_modern_robotics"];  // the random angles of test_lab.py + GravityForces
    for (const std::string name : {"elbow", "knee"}) {
        const lab::JointParams& j = name == "elbow" ? P().elbow : P().knee;
        for (const auto& c : C[name]) {
            INFO(name << ", theta = " << num(c["theta"]));
            CHECK(std::abs(lab::gravity(num(c["theta"]), j) - num(c["g"])) < 1e-8);
        }
    }
    // C++ only: the simulator adds the model gravity to your controller output, tau_cmd = g(theta) + u
    // (a short PD run: make_controller calls joint_state_space, discretize_zoh and pd_gains).
    const lab::SimResult sim = lab::simulate("pd", E(), 0.05);
    REQUIRE(sim.theta.size() == 5);
    REQUIRE(sim.u.size() == 5);
    REQUIRE(sim.tau_cmd.size() == 5);
    for (Index i = 0; i < 5; ++i)
        CHECK_SCALAR_NEAR(sim.tau_cmd(i) - sim.u(i), lab::gravity(sim.theta(i), E()), 1e-12);
}

TEST_CASE("test_solve_qp_known_problem") {
    // min (z1-1)^2 + (z2-2)^2  s.t.  z1 + z2 <= 1   ->  z = (0, 1)
    const MatrixXd H2 = 2.0 * MatrixXd::Identity(2, 2);
    const MatrixXd G1 = (MatrixXd(1, 2) << 1.0, 1.0).finished();
    const MatrixXd Aeq = (MatrixXd(1, 2) << 1.0, -1.0).finished();
    const std::pair<VectorXd, std::string> r1 = lab::solve_qp(H2, vec({-2.0, -4.0}), G1, vec({1.0}));
    CHECK(r1.second == "optimal");
    CHECK_ALLCLOSE(r1.first, vec({0.0, 1.0}), 1e-5, 1e-9);
    CHECK_MAT_NEAR(r1.first, as_vec(cases()["solve_qp_known_problem"]["z_ineq"]), 1e-12);
    const std::pair<VectorXd, std::string> r2 =
        lab::solve_qp(H2, vec({-2.0, -4.0}), MatrixXd(), VectorXd(), Aeq, vec({0.0}));
    CHECK(r2.second == "optimal");
    CHECK_ALLCLOSE(r2.first, vec({1.5, 1.5}), 1e-5, 1e-9);
    CHECK_MAT_NEAR(r2.first, as_vec(cases()["solve_qp_known_problem"]["z_eq"]), 1e-12);
    const MatrixXd G3 = (MatrixXd(2, 1) << 1.0, -1.0).finished();
    const std::pair<VectorXd, std::string> r3 = lab::solve_qp(m11(1.0), VectorXd::Zero(1), G3, vec({-1.0, -1.0}));
    CHECK(r3.second == "infeasible");
    // C++ only: one MPC step of x+ = x + u with Q = R = P = 1, x0 = 1, |u| <= 0.25 (your mpc_qp):
    // H = 2 (1 + 1) = 4, f = 2; unconstrained u = -0.5, so the bound is active: u = -0.25.
    const lab::CondensedQP qp = lab::mpc_qp(m11(1.0), m11(1.0), m11(1.0), m11(1.0), m11(1.0), 1, vec({1.0}),
                                            lab::Bounds{vec({-0.25}), vec({0.25})});
    CHECK_MAT_NEAR(qp.H, m11(4.0), 1e-12);
    CHECK_MAT_NEAR(qp.f, vec({2.0}), 1e-12);
    const std::pair<VectorXd, std::string> r4 = lab::solve_qp(qp.H, qp.f, qp.G, qp.w);
    CHECK(r4.second == "optimal");
    CHECK_MAT_NEAR(r4.first, vec({-0.25}), 1e-9);
}

TEST_CASE("test_setpoint_profile") {
    const lab::Task& task = E().task;
    CHECK(lab::setpoint(0.0, task) == 1.5);
    CHECK(lab::setpoint(1.49, task) == 1.5);
    CHECK(lab::setpoint(1.5, task) == 0.4);
    // C++ only: perfect tracking of the profile has no error, no overshoot and settles at once (your metrics).
    const Index n = 300;
    lab::SimResult sim;
    sim.t = VectorXd::LinSpaced(n, 0.0, static_cast<double>(n - 1) * E().h);
    sim.theta_d.resize(n);
    for (Index i = 0; i < n; ++i) sim.theta_d(i) = lab::setpoint(static_cast<double>(i) * E().h, task);
    sim.theta = sim.theta_d;
    sim.theta_dot = VectorXd::Zero(n);
    sim.tau = sim.theta.unaryExpr([](double t) { return lab::gravity(t, P().elbow); });
    sim.tau_cmd = sim.tau;
    sim.u = VectorXd::Zero(n);
    const lab::Metrics m = lab::metrics(sim, E());
    CHECK(m.rom_violation == 0.0);
    CHECK(m.v_violation == 0.0);
    CHECK(m.overshoot == 0.0);
    CHECK(m.settle_time == 0.0);
    CHECK(m.rms_error == 0.0);
    CHECK(m.sat_fraction == 0.0);
}

// ------------------------------------------------------------------------------ models and PD
TEST_CASE("test_joint_state_space") {
    const std::pair<MatrixXd, MatrixXd> AB = lab::joint_state_space(E().M, E().b);
    const ElbowMats e = elbow_mats();
    CHECK_ALLCLOSE(AB.first, e.A, 1e-5, 1e-8);
    CHECK_ALLCLOSE(AB.second, e.B, 1e-5, 1e-8);
}

TEST_CASE("test_discretize_zoh_closed_form") {
    const ElbowMats e = elbow_mats();
    const std::pair<MatrixXd, MatrixXd> d = lab::discretize_zoh(e.A, e.B, E().h);
    CHECK_ALLCLOSE(d.first, e.Ad, 1e-5, 1e-12);
    CHECK_ALLCLOSE(d.second, e.Bd, 1e-5, 1e-12);
    const json& C = cases()["discretize_zoh_closed_form"];  // Python reference (scipy.linalg.expm)
    CHECK_MAT_NEAR(d.first, as_mat(C["Ad"]), 1e-12);
    CHECK_MAT_NEAR(d.second, as_vec(C["Bd"]), 1e-12);
    // double integrator (b = 0): Bd = (h^2/2, h)/M
    const MatrixXd A2 = (MatrixXd(2, 2) << 0.0, 1.0, 0.0, 0.0).finished();
    const MatrixXd B2 = (MatrixXd(2, 1) << 0.0, 2.0).finished();
    const std::pair<MatrixXd, MatrixXd> d2 = lab::discretize_zoh(A2, B2, 0.1);
    CHECK_ALLCLOSE(d2.first, (MatrixXd(2, 2) << 1.0, 0.1, 0.0, 1.0).finished(), 1e-5, 1e-8);
    CHECK_ALLCLOSE(d2.second, vec({0.01, 0.2}), 1e-5, 1e-8);
}

TEST_CASE("test_pd_gains_give_wn_zeta") {
    const std::pair<double, double> g = lab::pd_gains(E().M, E().b, 0.5, 15.0);
    const double Kp = g.first, Kd = g.second;
    // roots of M s^2 + (b + Kd) s + Kp (np.roots)
    const double a2 = E().M, a1 = E().b + Kd, a0 = Kp;
    const std::complex<double> sq = std::sqrt(std::complex<double>(a1 * a1 - 4 * a2 * a0, 0.0));
    const std::complex<double> r0 = (-a1 + sq) / (2 * a2), r1 = (-a1 - sq) / (2 * a2);
    const double wn = std::sqrt(std::abs(r0 * r1));
    const double zeta = -(r0 + r1).real() / (2 * wn);
    CHECK(std::abs(wn - 15.0) < 1e-9);
    CHECK(std::abs(zeta - 0.5) < 1e-9);
    CHECK(std::abs(Kp - num(EXP()["gains"]["Kp_pd"])) < 1e-9);
    CHECK(std::abs(Kd - num(EXP()["gains"]["Kd_pd"])) < 1e-9);
}

// ------------------------------------------------------------------------------ Riccati equations
TEST_CASE("test_scalar_care") {
    // Python: parametrized over (a, b, q, r) = (-4, 2, 100, 1), (2, 1, 1, 1), (0, 1, 4, 0.25) (3 pytest items)
    for (const auto& c : cases()["scalar_care"]) {
        const double a = num(c["a"]), b = num(c["b"]), q = num(c["q"]), r = num(c["r"]);
        INFO("a, b, q, r = " << a << ", " << b << ", " << q << ", " << r);
        const std::pair<double, double> pk = lab::scalar_care(a, b, q, r);
        const double p = pk.first, k = pk.second;
        CHECK(p > 0);
        CHECK(std::abs(2 * a * p - p * p * b * b / r + q) < 1e-9 * std::max(1.0, q));
        CHECK(std::abs(p - num(c["p"])) < 1e-9 * std::max(1.0, p));  // scipy solve_continuous_are
        CHECK(std::abs((a - b * k) + std::sqrt(a * a + b * b * q / r)) < 1e-9);
    }
}

TEST_CASE("test_scalar_dare_golden_ratio") {
    const std::pair<double, double> sk = lab::scalar_dare(1.0, 0.1, 1.0, 0.01);
    CHECK(std::abs(sk.first - (1 + std::sqrt(5.0)) / 2) < 1e-12);
    CHECK(std::abs(sk.second - 6.180339887) < 1e-8);
    for (const auto& c : cases()["scalar_dare_golden_ratio"]) {
        const double a = num(c["a"]), b = num(c["b"]), q = num(c["q"]), r = num(c["r"]);
        INFO("a, b, q, r = " << a << ", " << b << ", " << q << ", " << r);
        const double s = lab::scalar_dare(a, b, q, r).first;
        CHECK(std::abs(s - num(c["s"])) < 1e-9 * std::max(1.0, s));  // scipy solve_discrete_are
    }
}

TEST_CASE("test_lqr_matches_scipy_and_closed_form") {
    const ElbowMats e = elbow_mats();
    const MatrixXd Q = diag({100.0, 1 / 2.25}), R = m11(1 / 6.25);
    const std::pair<MatrixXd, MatrixXd> KS = lab::lqr(e.A, e.B, Q, R);
    CHECK_ALLCLOSE(KS.second, as_mat(cases()["lqr_matches_scipy_and_closed_form"]["S"]), 1e-9, 1e-8);
    // damped double integrator: Kp = sqrt(q1/r), Kd = -b + sqrt(b^2 + 2 M Kp + q2/r)
    const double Kp = std::sqrt(100.0 / (1 / 6.25));
    const double Kd = -E().b + std::sqrt(E().b * E().b + 2 * E().M * Kp + (1 / 2.25) / (1 / 6.25));
    CHECK_ALLCLOSE(KS.first, vec({Kp, Kd}), 1e-9, 1e-8);
    CHECK_ALLCLOSE(KS.first, as_vec(EXP()["gains"]["K_clqr"]), 1e-9, 1e-8);
}

TEST_CASE("test_riccati_recursion_fibonacci_and_convergence") {
    const std::pair<std::vector<MatrixXd>, std::vector<MatrixXd>> KS =
        lab::riccati_recursion(m11(1.0), m11(0.1), m11(1.0), m11(0.01), m11(0.0), 4);
    REQUIRE(KS.first.size() == 4);
    REQUIRE(KS.second.size() == 5);
    VectorXd vals(5);
    for (std::size_t n = 0; n < 5; ++n) {
        REQUIRE(KS.second[n].size() == 1);
        vals(static_cast<Index>(4 - n)) = KS.second[n](0, 0);  // vals[::-1]
    }
    CHECK_ALLCLOSE(vals, vec({0.0, 1.0, 1.5, 1.6, 21.0 / 13.0}), 1e-5, 1e-12);
    REQUIRE(KS.first[3].size() == 1);
    REQUIRE(KS.first[2].size() == 1);
    CHECK(std::abs(KS.first[3](0, 0) - 0.1 * 0.0 / 0.01) < 1e-12);           // K[N-1] uses S[N] = 0
    CHECK(std::abs(KS.first[2](0, 0) - 0.1 * 1.0 / (0.01 + 0.01)) < 1e-12);  // 5.0
    const ElbowMats e = elbow_mats();
    const std::pair<MatrixXd, MatrixXd> QR = lab::bryson_weights(E().lqr.x_max, E().lqr.u_max);
    const std::pair<std::vector<MatrixXd>, std::vector<MatrixXd>> KS2 =
        lab::riccati_recursion(e.Ad, e.Bd, QR.first, QR.second, MatrixXd::Zero(2, 2), 600);
    REQUIRE(KS2.second.size() >= 1);
    CHECK_ALLCLOSE(KS2.second[0], as_mat(cases()["riccati_recursion_fibonacci_and_convergence"]["S_dare"]), 1e-8,
                   1e-8);
    CHECK(KS2.first.size() == 600);
    CHECK(KS2.second.size() == 601);
}

TEST_CASE("test_dlqr_matches_scipy") {
    const ElbowMats e = elbow_mats();
    const MatrixXd Q = diag({100.0, 1 / 2.25}), R = m11(0.16);
    const std::pair<MatrixXd, MatrixXd> KS = lab::dlqr(e.Ad, e.Bd, Q, R);
    const json& C = cases()["dlqr_matches_scipy"];  // scipy solve_discrete_are
    CHECK_ALLCLOSE(KS.second, as_mat(C["S0"]), 1e-8, 1e-8);
    CHECK_ALLCLOSE(KS.first, as_mat(C["K0"]), 1e-8, 1e-8);
    CHECK_ALLCLOSE(KS.first, as_vec(EXP()["gains"]["K_dlqr"]), 1e-8, 1e-8);
}

TEST_CASE("test_bryson_weights") {
    const std::pair<MatrixXd, MatrixXd> QR = lab::bryson_weights(vec({0.1, 1.5}), vec({2.5}));
    CHECK_ALLCLOSE(QR.first, diag({100.0, 1 / 2.25}), 1e-5, 1e-8);
    CHECK_ALLCLOSE(QR.second, m11(0.16), 1e-5, 1e-8);
    const std::pair<MatrixXd, MatrixXd> QR1 = lab::bryson_weights(vec({0.5}), vec({2.0}));  // scalars
    CHECK_ALLCLOSE(QR1.first, m11(4.0), 1e-5, 1e-8);
    CHECK_ALLCLOSE(QR1.second, m11(0.25), 1e-5, 1e-8);
}

// ------------------------------------------------------------------------------ trajectory optimisation and MPC
TEST_CASE("test_prediction_matrices_match_simulation") {
    const ElbowMats e = elbow_mats();
    const int N = 7;
    const std::pair<MatrixXd, MatrixXd> SS = lab::prediction_matrices(e.Ad, e.Bd, N);
    const MatrixXd& Sx = SS.first;
    const MatrixXd& Su = SS.second;
    REQUIRE(Sx.rows() == 14);
    REQUIRE(Sx.cols() == 2);
    REQUIRE(Su.rows() == 14);
    REQUIRE(Su.cols() == 7);
    const json& C = cases()["prediction_matrices_match_simulation"];  // RNG.normal(size=2), RNG.normal(size=7)
    const VectorXd x0 = as_vec(C["x0"]), U = as_vec(C["U"]);
    VectorXd x = x0, X(2 * N);
    for (int k = 0; k < N; ++k) {
        x = (e.Ad * x + e.Bd.col(0) * U(k)).eval();
        X.segment(2 * k, 2) = x;
    }
    CHECK_ALLCLOSE(VectorXd(Sx * x0 + Su * U), X, 1e-5, 1e-12);
    CHECK_ALLCLOSE(Su.block(0, 1, 2, 6), MatrixXd::Zero(2, 6), 1e-5, 1e-8);  // causality
}

TEST_CASE("test_mpc_qp_cost_and_constraints") {
    const ElbowMats e = elbow_mats();
    const MatrixXd Q = diag({100.0, 0.4}), R = m11(0.16), Pm = diag({300.0, 2.0});
    const int N = 5;
    const VectorXd x0 = vec({-0.4, 0.5});
    const VectorXd lo = vec({-0.5, -3.0}), hi = vec({0.1, 3.0});
    const lab::CondensedQP qp =
        lab::mpc_qp(e.Ad, e.Bd, Q, R, Pm, N, x0, lab::Bounds{vec({-2.5}), vec({2.5})}, lab::Bounds{lo, hi});
    REQUIRE(qp.H.rows() == N);
    REQUIRE(qp.H.cols() == N);
    REQUIRE(qp.G.rows() == 2 * N + 4 * N);
    REQUIRE(qp.G.cols() == N);
    REQUIRE(qp.f.size() == N);
    REQUIRE(qp.w.size() == 6 * N);
    const json& C = cases()["mpc_qp_cost_and_constraints"];
    CHECK_ALLCLOSE(qp.H, as_mat(C["H"]), 1e-9, 1e-9);  // C++ only: the Python reference H and f
    CHECK_ALLCLOSE(qp.f, as_vec(C["f"]), 1e-9, 1e-9);
    auto cost = [&](const VectorXd& U, std::vector<VectorXd>& X) {
        VectorXd x = x0;
        double J = 0.0;
        X.clear();
        for (int k = 0; k < N; ++k) {
            J += x.dot(Q * x) + U(k) * U(k) * R(0, 0);
            x = (e.Ad * x + e.Bd.col(0) * U(k)).eval();
            X.push_back(x);
        }
        return J + x.dot(Pm * x);
    };
    std::vector<VectorXd> X;
    const double J0 = cost(VectorXd::Zero(N), X);
    for (const auto& Uj : C["U"]) {  // RNG.uniform(-3, 3, N), 5 times
        const VectorXd U = as_vec(Uj);
        INFO("U = " << U.transpose());
        const double J = cost(U, X);
        CHECK(std::abs(0.5 * U.dot(qp.H * U) + qp.f.dot(U) + J0 - J) < 1e-8 * std::max(1.0, J));
        bool feasible = (U.cwiseAbs().array() <= 2.5).all();
        for (const VectorXd& xx : X) feasible = feasible && (xx.array() >= lo.array()).all() && (xx.array() <= hi.array()).all();
        CHECK(feasible == ((qp.G * U).array() <= (qp.w.array() + 1e-12)).all());
    }
}

TEST_CASE("test_transcription_equals_shooting") {
    const ElbowMats e = elbow_mats();
    const std::pair<MatrixXd, MatrixXd> QR = lab::bryson_weights(vec({0.02, 1.5}), vec({2.5}));
    const MatrixXd S = lab::dlqr(e.Ad, e.Bd, QR.first, QR.second).second;
    const int N = 12;
    const VectorXd x0 = vec({-1.2, 0.0});
    const lab::Bounds xl{vec({0.2 - 1.5, -3.0}), vec({1.6 - 1.5, 3.0})};
    const lab::Bounds ul{vec({-2.5}), vec({2.5})};
    const lab::TranscriptionQP t = lab::transcription_qp(e.Ad, e.Bd, QR.first, QR.second, S, N, x0, ul, xl);
    REQUIRE(t.H.rows() == 3 * N);
    REQUIRE(t.H.cols() == 3 * N);
    REQUIRE(t.Aeq.rows() == 2 * N);
    REQUIRE(t.Aeq.cols() == 3 * N);
    REQUIRE(t.beq.size() == 2 * N);
    const json& C = cases()["transcription_equals_shooting"];
    const VectorXd U = as_vec(C["U"]);  // RNG.normal(size=N)
    VectorXd x = x0, z(3 * N);
    for (int k = 0; k < N; ++k) {
        x = (e.Ad * x + e.Bd.col(0) * U(k)).eval();
        z.segment(2 * k, 2) = x;
    }
    z.tail(N) = U;
    CHECK_ALLCLOSE(VectorXd(t.Aeq * z), t.beq, 1e-5, 1e-12);
    const std::pair<VectorXd, std::string> zt = lab::solve_qp(t.H, t.f, t.G, t.w, t.Aeq, t.beq);
    const lab::CondensedQP c = lab::mpc_qp(e.Ad, e.Bd, QR.first, QR.second, S, N, x0, ul, xl);
    const std::pair<VectorXd, std::string> zs = lab::solve_qp(c.H, c.f, c.G, c.w);
    CHECK(zt.second == "optimal");
    CHECK(zs.second == "optimal");
    REQUIRE(zt.first.size() == 3 * N);
    CHECK_ALLCLOSE(VectorXd(zt.first.tail(N)), zs.first, 1e-5, 1e-7);
    CHECK_ALLCLOSE(VectorXd(zt.first.tail(N)), as_vec(C["z_u"]), 1e-5, 1e-7);  // Python reference
}

TEST_CASE("test_mpc_without_active_constraints_is_lqr") {
    const ElbowMats e = elbow_mats();
    const std::pair<MatrixXd, MatrixXd> QR = lab::bryson_weights(E().lqr.x_max, E().lqr.u_max);
    const MatrixXd& Q = QR.first;
    const MatrixXd& R = QR.second;
    const std::pair<MatrixXd, MatrixXd> KS = lab::dlqr(e.Ad, e.Bd, Q, R);
    const lab::Bounds big{vec({-1e6}), vec({1e6})};
    for (const VectorXd& x0 : {vec({-0.05, 0.1}), vec({0.02, -0.3})}) {
        INFO("x0 = " << x0.transpose());
        // terminal cost = DARE -> u = -K x
        const std::pair<VectorXd, std::string> r = lab::mpc_control(x0, e.Ad, e.Bd, Q, R, KS.second, 10, big);
        REQUIRE(r.first.size() == 1);
        CHECK(r.second == "optimal");
        CHECK(std::abs(r.first(0) + (KS.first * x0)(0)) < 1e-8);
        // other terminal cost -> finite-horizon LQR
        const std::vector<MatrixXd> Kf = lab::riccati_recursion(e.Ad, e.Bd, Q, R, Q, 10).first;
        REQUIRE(!Kf.empty());
        const std::pair<VectorXd, std::string> r2 = lab::mpc_control(x0, e.Ad, e.Bd, Q, R, Q, 10, big);
        REQUIRE(r2.first.size() == 1);
        CHECK(std::abs(r2.first(0) + (Kf[0] * x0)(0)) < 1e-8);
    }
}

TEST_CASE("test_mpc_control_respects_and_detects_limits") {
    const lab::Controller c = lab::make_controller("mpc", E());
    const lab::Bounds lim = lab::error_limits(1.5, E());
    const std::pair<VectorXd, std::string> r =
        lab::mpc_control(vec({-1.2, 0.0}), c.Ad, c.Bd, c.Q, c.R, c.S, 30, c.u_lim, lim);
    REQUIRE(r.first.size() == 1);
    CHECK(r.second == "optimal");
    CHECK(std::abs(r.first(0) - 2.5) < 1e-9);  // saturated at the tightened bound
    const std::pair<VectorXd, std::string> r2 =
        lab::mpc_control(vec({0.05, 2.9}), c.Ad, c.Bd, c.Q, c.R, c.S, 30, c.u_lim, lim);
    CHECK(r2.second == "infeasible");  // cannot stop before the range limit
}

TEST_CASE("test_metrics_on_synthetic_run") {
    const lab::JointParams j = E();
    const Index n = 100;  // np.arange(0, 1.0, 0.01)
    lab::SimResult sim;
    sim.t.resize(n);
    sim.theta.resize(n);
    sim.theta_dot.resize(n);
    sim.theta_d.resize(n);
    sim.tau = VectorXd::Constant(n, 2.0);
    sim.tau_cmd.resize(n);
    sim.u = VectorXd::Zero(n);
    for (Index i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) * 0.01;
        sim.t(i) = t;
        sim.theta_d(i) = t < 0.5 ? 1.0 : 0.5;
        sim.theta(i) = t < 0.5 ? 1.0 + 0.1 * std::exp(-10 * t) * std::sin(20 * t) : 0.5;
        sim.theta_dot(i) = t < 0.2 ? 3.5 : 0.0;
        sim.tau_cmd(i) = t < 0.1 ? 7.0 : 2.0;
    }
    sim.theta.head(3).setConstant(0.8);
    const lab::Metrics m = lab::metrics(sim, j);
    CHECK(m.rom_violation == 0.0);
    CHECK(std::abs(m.v_violation - 0.5) < 1e-12);
    CHECK(std::abs(m.peak_tau - 2.0) < 1e-12);
    CHECK(std::abs(m.sat_fraction - 0.1) < 1e-12);
    CHECK(std::abs(m.effort - 0.01 * 100 * 4.0) < 1e-9);
    Index imax = 0;
    sim.theta.head(50).maxCoeff(&imax);  // np.argmax(th[:50] - 1.0)
    CHECK(std::abs(m.overshoot - (sim.theta(imax) - 1.0)) < 1e-12);
    const double ref_err = std::sqrt((sim.theta - sim.theta_d).squaredNorm() / static_cast<double>(n));
    CHECK(std::abs(m.rms_error - ref_err) < 1e-12);
    Index last_out = -1;
    for (Index i = 0; i < 50; ++i)
        if (std::abs(sim.theta(i) - 1.0) > 0.02) last_out = i;
    REQUIRE(last_out >= 0);
    CHECK(std::abs(m.settle_time - static_cast<double>(last_out + 1) * 0.01) < 1e-12);
    // C++ only: every metric equals the Python reference
    for (const auto& kv : cases()["metrics_on_synthetic_run"]["metrics"].items()) {
        INFO("metric " << kv.key());
        CHECK_SCALAR_NEAR(metric(m, kv.key()), num(kv.value()), 1e-12);
    }
}

// ------------------------------------------------------------------------------ closed-loop runs
TEST_CASE("test_runs_match_reference") {
    // Python: parametrized over kind = pd, lqr, mpc (3 pytest items)
    for (const std::string kind : {"pd", "lqr", "mpc"}) {
        INFO("kind = " << kind);
        const lab::SimResult sim = lab::simulate(kind, E());
        const sbeg::CsvTable ref = load_ref(kind);
        const VectorXd th_ref = ref.data.col(ref.col("theta")), tau_ref = ref.data.col(ref.col("tau"));
        REQUIRE(sim.theta.size() == th_ref.size());
        REQUIRE(sim.tau.size() == tau_ref.size());
        CHECK_MESSAGE(allclose(sim.theta, th_ref, 1e-5, 1e-6).ok, "theta differs from the reference run");
        CHECK_MESSAGE(allclose(sim.tau, tau_ref, 1e-5, 1e-5).ok, "applied torque differs from the reference run");
        const lab::Metrics m = lab::metrics(sim, E());
        for (const auto& kv : EXP()["elbow_" + kind].items()) {
            INFO("metric " << kv.key());
            if (kv.value().is_null()) {
                CHECK(std::isnan(metric(m, kv.key())));
            } else {
                const double v = num(kv.value());
                CHECK(std::abs(metric(m, kv.key()) - v) < 2e-5 * std::max(1.0, std::abs(v)));
            }
        }
    }
}

TEST_CASE("test_mpc_run_is_safe_and_lqr_pd_are_not") {
    const lab::Metrics pd = lab::metrics(lab::simulate("pd", E()), E());
    const lab::Metrics lq = lab::metrics(lab::simulate("lqr", E()), E());
    const lab::Metrics mp = lab::metrics(lab::simulate("mpc", E()), E());
    CHECK(mp.rom_violation == 0.0);
    CHECK(mp.v_violation < 1e-6);
    CHECK(mp.sat_fraction == 0.0);
    CHECK(pd.rom_violation > 0.05);
    CHECK(lq.v_violation > 1.0);
}
