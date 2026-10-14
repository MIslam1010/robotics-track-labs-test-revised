// Auto-checks for SBEG 461 Lab 6 (computed torque vs PID with model error on the course arm; task-space control)
// -- C++ track. Mirrors test_lab.py test by test (test_starter_compiles has no C++ counterpart: building is the check).
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*gains*,*pid_torque*"      (run only some tests, like pytest -k)
// Ground truth: mrcpp (the C++ port of modern_robotics: inverse_dynamics, mass_matrix, gravity_forces,
// computed_torque, forward_dynamics, jacobian_body), finite differences for the Jacobian derivative and the
// feedforward acceleration, the closed-loop identities of MR Sec. 11.4, an independent Newton solution of the
// computed-torque steady-state error under model error, and cpp_cases.json, which holds the numbers of the Python
// reference (including the random states drawn by test_lab.py), so both languages are checked against the same values.
// The four data/infrastructure tests also call one of your functions (marked "C++ only"), so that every test case
// fails until your code is written.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using lab::Matrix6d;
using lab::Vector6d;
using sbeg::as;
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

VectorXd vec6(std::initializer_list<double> v) {
    VectorXd x(static_cast<Eigen::Index>(v.size()));
    Eigen::Index i = 0;
    for (double e : v) x(i++) = e;
    return x;
}

lab::Robot no_friction(lab::Robot r) {
    r.b = VectorXd::Zero(6);
    return r;
}

/// np.allclose(got, expected, rtol, atol): |got - expected| <= atol + rtol |expected| entry by entry.
sbeg::NearResult allclose(const MatrixXd& got_in, const MatrixXd& exp_in, double atol = 1e-8, double rtol = 1e-5) {
    sbeg::NearResult r = sbeg::compare_near(got_in, exp_in, std::numeric_limits<double>::infinity());
    if (!r.ok) return r;  // shape mismatch or NaN/inf (message already set)
    MatrixXd got = got_in, exp = exp_in;
    if (got.rows() != exp.rows()) {  // a row and a column vector of the same length
        got = Eigen::Map<VectorXd>(got.data(), got.size()).eval();
        exp = Eigen::Map<VectorXd>(exp.data(), exp.size()).eval();
    }
    if (got.size() == 0) return r;
    const MatrixXd excess = (got - exp).cwiseAbs() - rtol * exp.cwiseAbs();
    Eigen::Index i = 0, k = 0;
    const double worst = excess.maxCoeff(&i, &k);
    r.ok = worst <= atol;
    std::ostringstream s;
    s << std::setprecision(12) << "worst entry (" << i << "," << k << "): got " << got(i, k) << ", expected "
      << exp(i, k) << " (atol " << atol << ", rtol " << rtol << ")";
    r.msg = s.str();
    return r;
}

/// x == pytest.approx(expected, rel, abs)
bool approx(double got, double expected, double rel = 1e-6, double abs_tol = 1e-12) {
    return std::abs(got - expected) <= std::max(rel * std::abs(expected), abs_tol);
}

/// Sample k of an (N, n) matrix as a column vector.
VectorXd row(const MatrixXd& A, Eigen::Index k) { return A.row(k).transpose(); }

/// N copies of q as rows (np.tile(q, (N, 1))).
MatrixXd tile(const VectorXd& q, Eigen::Index N) { return q.transpose().replicate(N, 1); }

/// Python _joint_run of test_lab.py: joint move of the configuration, metrics of the run.
lab::Metrics joint_run(const std::string& c, const std::string& law, double T_end = 0.8, double ki = 0.0) {
    const json& jm = cfg()["joint_move"];
    const json& g = cfg()["gains"];
    const VectorXd t = lab::time_grid(T_end, num(cfg()["dt"]));
    const VectorXd q0 = as_vec(jm["q0"]);
    const lab::JointReference ref = lab::quintic_reference(q0, q0 + as_vec(jm["dq"]), num(jm["T"]), t);
    const auto [plant, model] = lab::make_case(c, cfg());
    const int substeps = cfg()["substeps"].get<int>();
    if (law == "computed_torque") {
        const auto [kp, kd] = lab::ct_gains(num(g["zeta"]), num(g["wn"]));
        lab::JointController ctrl(law, t, ref.qd, ref.dqd, ref.ddqd, kp, ki, kd, model);
        const lab::SimResult out =
            lab::simulate_arm(plant, ctrl, t, q0, VectorXd(), substeps, lab::tau_limits(plant, cfg()));
        return lab::run_metrics(t, ref.qd, out.q, out.tau);
    }
    const lab::PIDGains G =
        lab::decentralized_gains(model, as_vec(g["gain_reference_q"]), num(g["zeta"]), num(g["wn"]), ki);
    lab::JointController ctrl(law, t, ref.qd, ref.dqd, ref.ddqd, G.Kp, G.Ki, G.Kd, model);
    const lab::SimResult out = lab::simulate_arm(plant, ctrl, t, q0, VectorXd(), substeps, lab::tau_limits(plant, cfg()));
    return lab::run_metrics(t, ref.qd, out.q, out.tau);
}

void check_metrics_vs_python(const lab::Metrics& m, const json& ref, double tol) {
    CHECK_SCALAR_NEAR(m.max_err, num(ref["max_err"]), tol);
    CHECK_SCALAR_NEAR(m.rms_err, num(ref["rms_err"]), tol);
    CHECK_SCALAR_NEAR(m.final_err, num(ref["final_err"]), tol);
    CHECK_SCALAR_NEAR(m.effort, num(ref["effort"]), tol);
    CHECK_MAT_NEAR(m.peak_tau, as_vec(ref["peak_tau"]), tol);
}
}  // namespace

#define CHECK_ALLCLOSE(got, expected, atol)                                                          \
    do {                                                                                             \
        const ::sbeg::NearResult lab_r_ = allclose((got), (expected), (atol));                        \
        CHECK_MESSAGE(lab_r_.ok, "allclose(" #got ", " #expected ")\n  " << lab_r_.msg);             \
    } while (false)

// ============================================================================ data and infrastructure (provided code)
TEST_CASE("test_config_and_cases") {
    const json& CFG = cfg();
    CHECK(num(CFG["dt"]) == 0.002);
    CHECK(num(CFG["payload"]) == 0.25);
    const auto [plant, model] = lab::make_case("payload_unknown", CFG);
    CHECK(plant.payload == 0.25);
    CHECK(model.payload == 0.0);
    const lab::Robot m120 = lab::make_case("masses_120", CFG).second;
    CHECK_ALLCLOSE(m120.Glist[2], 1.2 * plant.Glist[2], 1e-8);
    CHECK_ALLCLOSE(lab::tau_limits(plant, CFG), 0.6 * vec6({1.8, 3.0, 1.5, 0.6, 0.6, 0.4}), 1e-8);
    const json& c = cases()["config_and_cases"];
    CHECK_MAT_NEAR(plant.Glist[2], as_mat(c["plant_G3"]), 1e-12);
    CHECK_MAT_NEAR(m120.Glist[2], as_mat(c["model120_G3"]), 1e-12);
    // C++ only: at rest, the model's inverse dynamics is its gravity torque (the model without the payload).
    const VectorXd z = VectorXd::Zero(6);
    const VectorXd g = lab::model_inverse_dynamics(model, model.q_home, z, z);
    CHECK_MAT_NEAR(g, mr::gravity_forces(model.q_home, model.g, model.Mlist, model.Glist, model.Slist), 1e-12);
    CHECK_MAT_NEAR(g, as_vec(c["g_home_payload_unknown"]), 1e-12);
}

TEST_CASE("test_plant_matches_modern_robotics_forward_dynamics") {
    const lab::Robot plant = lab::make_case("exact", cfg()).first;
    const lab::Plant P(plant);
    for (const auto& c : cases()["plant_matches_modern_robotics_forward_dynamics"]) {
        const VectorXd q = as_vec(c["q"]), dq = as_vec(c["dq"]), tau = as_vec(c["tau"]);
        const VectorXd ref = mr::forward_dynamics(q, dq, tau - plant.b.cwiseProduct(dq), plant.g, Vector6d::Zero(),
                                                  plant.Mlist, plant.Glist, plant.Slist);
        const VectorXd acc = P.accel(q, dq, tau);
        CHECK_ALLCLOSE(acc, ref, 1e-8);
        CHECK_ALLCLOSE(acc, as_vec(c["ddq"]), 1e-8);  // Python plant (Pinocchio ABA)
        // C++ only: your model_inverse_dynamics is the inverse of the plant's forward dynamics (friction included).
        CHECK_ALLCLOSE(lab::model_inverse_dynamics(plant, q, dq, acc), tau, 1e-8);
    }
}

TEST_CASE("test_references_are_consistent") {
    const VectorXd t = lab::time_grid(1.0, 0.001);
    const VectorXd q0 = VectorXd::Zero(6), q1 = VectorXd::LinSpaced(6, 0.0, 5.0);
    const lab::JointReference r = lab::quintic_reference(q0, q1, 0.8, t);
    const Eigen::Index N = t.size();
    REQUIRE(N == 1001);
    CHECK_ALLCLOSE(row(r.qd, 0), q0, 1e-8);
    CHECK_ALLCLOSE(row(r.qd, N - 1), q1, 1e-8);
    CHECK_ALLCLOSE(row(r.dqd, 0), VectorXd::Zero(6), 1e-8);
    CHECK_ALLCLOSE(row(r.dqd, N - 1), VectorXd::Zero(6), 1e-8);
    const double h = t(1) - t(0);
    for (Eigen::Index k = 5; k < N - 5; ++k)  // np.gradient (central differences inside), rows 5 .. N-6
        CHECK_ALLCLOSE((row(r.qd, k + 1) - row(r.qd, k - 1)) / (2 * h), row(r.dqd, k), 1e-4);
    const course_arm::Robot arm = course_arm::course_arm();
    const MatrixXd Blist = mr::adjoint(mr::trans_inv(arm.M)) * arm.Slist;
    const Matrix4d X0 = mr::fkin_body(arm.M, Blist, arm.q_home);
    const lab::CartesianReference cr = lab::cartesian_reference(X0, Eigen::Vector3d(0.01, -0.02, 0.03), 0.8, t);
    REQUIRE(cr.Xd.size() == static_cast<std::size_t>(N));
    const std::size_t k = 300;
    const Vector6d Vnum = mr::se3_to_vec(mr::trans_inv(cr.Xd[k]) * (cr.Xd[k + 1] - cr.Xd[k - 1]) / (2 * h));
    CHECK_ALLCLOSE(Vnum, row(cr.Vd, 300), 1e-6);
    CHECK_ALLCLOSE((row(cr.Vd, 301) - row(cr.Vd, 299)) / (2 * h), row(cr.dVd, 300), 1e-4);
    const json& c = cases()["references_are_consistent"];
    CHECK_MAT_NEAR(row(r.dqd, 300), as_vec(c["dqd_300"]), 1e-12);
    CHECK_MAT_NEAR(row(r.ddqd, 300), as_vec(c["ddqd_300"]), 1e-10);
    CHECK_MAT_NEAR(cr.Xd[k], as_mat(c["Xd_300"]), 1e-12);
    CHECK_MAT_NEAR(row(cr.Vd, 300), as_vec(c["Vd_300"]), 1e-12);
    // C++ only: a run that follows its reference exactly has zero error.
    const lab::Metrics m = lab::run_metrics(t, r.qd, r.qd, MatrixXd::Zero(N, 6));
    CHECK(m.max_err == 0.0);
    CHECK(m.effort == 0.0);
}

// ============================================================================ Part A: gains
TEST_CASE("test_ct_gains_and_ki_bound") {
    auto [kp, kd] = lab::ct_gains(1.0, 20.0);
    CHECK_APPROX(kp, 400.0);
    CHECK_APPROX(kd, 40.0);
    std::tie(kp, kd) = lab::ct_gains(0.7, 10.0);
    CHECK_APPROX(kp, 100.0);
    CHECK_APPROX(kd, 14.0);
    const double kimax = lab::ct_ki_max(400.0, 40.0);
    CHECK_APPROX(kimax, 16000.0);
    for (const std::pair<double, bool>& fs : {std::pair<double, bool>{0.9, true}, std::pair<double, bool>{1.1, false}}) {
        const double f = fs.first;
        const bool stable = fs.second;
        // np.roots([1, 40, 400, f kimax]) = eigenvalues of the companion matrix
        Eigen::Matrix3d Cm;
        Cm << -40.0, -400.0, -f * kimax, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0;
        const Eigen::EigenSolver<Eigen::Matrix3d> es(Cm);
        INFO("ki = " << f << " ki_max");
        CHECK((es.eigenvalues().real().maxCoeff() < 0) == stable);
    }
}

TEST_CASE("test_decentralized_gains") {
    const lab::Robot model = lab::make_case("exact", cfg()).second;
    const VectorXd q = as_vec(cfg()["gains"]["gain_reference_q"]);
    const auto [Kp, Ki, Kd] = lab::decentralized_gains(model, q, 1.0, 20.0, 0.1);
    const VectorXd Md = mr::mass_matrix(q, model.Mlist, model.Glist, model.Slist).diagonal();
    CHECK_ALLCLOSE(Kp, 400 * Md, 1e-8);
    CHECK_ALLCLOSE(Kd, 40 * Md - model.b, 1e-8);
    CHECK_ALLCLOSE(Ki, 0.1 * (model.b + Kd).cwiseProduct(Kp).cwiseQuotient(Md), 1e-8);
    REQUIRE(Kp.size() == 6);
    // zeta = 1 for every joint
    const VectorXd zeta = (model.b + Kd).cwiseQuotient(2 * Kp.cwiseProduct(Md).cwiseSqrt());
    CHECK_ALLCLOSE(zeta, VectorXd::Ones(6), 1e-8);
    const lab::PIDGains G0 = lab::decentralized_gains(model, q, 1.0, 20.0);
    CHECK_ALLCLOSE(G0.Kd, Kd, 1e-8);
    CHECK_ALLCLOSE(G0.Ki, VectorXd::Zero(6), 1e-8);
    const json& c = cases()["decentralized_gains"];
    CHECK_MAT_NEAR(Kp, as_vec(c["Kp"]), 1e-12);
    CHECK_MAT_NEAR(Ki, as_vec(c["Ki"]), 1e-10);
    CHECK_MAT_NEAR(Kd, as_vec(c["Kd"]), 1e-12);
}

// ============================================================================ Part B: control laws
TEST_CASE("test_pid_torque") {
    const json& c = cases()["pid_torque"];
    const VectorXd q = as_vec(c["q"]), dq = as_vec(c["dq"]), qd = as_vec(c["qd"]), dqd = as_vec(c["dqd"]),
                   eint = as_vec(c["eint"]);
    const VectorXd Kp = VectorXd::LinSpaced(6, 1.0, 6.0), Kd = VectorXd::Constant(6, 0.2);
    const VectorXd ref = Kp.cwiseProduct(qd - q) + 0.5 * eint + 0.2 * (dqd - dq);
    const VectorXd tau = lab::pid_torque(q, dq, eint, qd, dqd, Kp, 0.5, Kd);
    CHECK_ALLCLOSE(tau, ref, 1e-8);
    CHECK_MAT_NEAR(tau, as_vec(c["tau"]), 1e-12);
}

TEST_CASE("test_pd_gravity_torque_uses_model_gravity_at_q") {
    const lab::Robot model = lab::make_case("payload_unknown", cfg()).second;
    const json& c = cases()["pd_gravity_torque_uses_model_gravity_at_q"];
    const VectorXd q = as_vec(c["q"]), dq = as_vec(c["dq"]), qd = as_vec(c["qd"]), dqd = as_vec(c["dqd"]),
                   eint = as_vec(c["eint"]);
    const VectorXd g = mr::gravity_forces(q, model.g, model.Mlist, model.Glist, model.Slist);
    const VectorXd ref = 3.0 * (qd - q) + 0.1 * eint + 0.5 * (dqd - dq) + g;
    const VectorXd tau = lab::pd_gravity_torque(q, dq, eint, qd, dqd, 3.0, 0.1, 0.5, model);
    CHECK_ALLCLOSE(tau, ref, 1e-8);
    CHECK_MAT_NEAR(tau, as_vec(c["tau"]), 1e-12);
}

TEST_CASE("test_feedforward_uses_reference_state") {
    const lab::Robot model = lab::make_case("exact", cfg()).second;
    const json& c = cases()["feedforward_uses_reference_state"];
    const VectorXd q = as_vec(c["q"]), dq = as_vec(c["dq"]), qd = as_vec(c["qd"]), dqd = as_vec(c["dqd"]),
                   ddqd = as_vec(c["ddqd"]), eint = as_vec(c["eint"]);
    const VectorXd ff = mr::inverse_dynamics(qd, dqd, ddqd, model.g, Vector6d::Zero(), model.Mlist, model.Glist,
                                             model.Slist) +
                        model.b.cwiseProduct(dqd);
    const VectorXd ref = ff + 2.0 * (qd - q) + 0.3 * eint + 0.4 * (dqd - dq);
    const VectorXd tau = lab::feedforward_pd_torque(q, dq, eint, qd, dqd, ddqd, 2.0, 0.3, 0.4, model);
    CHECK_ALLCLOSE(tau, ref, 1e-8);
    CHECK_MAT_NEAR(tau, as_vec(c["tau"]), 1e-12);
}

TEST_CASE("test_computed_torque_matches_modern_robotics") {
    const lab::Robot model = lab::make_case("masses_120", cfg()).second;
    const lab::Robot m0 = no_friction(model);
    const VectorXd kv = vec6({400, 300, 200, 100, 50, 25.0});
    for (const auto& c : cases()["computed_torque_matches_modern_robotics"]) {
        const VectorXd q = as_vec(c["q"]), dq = as_vec(c["dq"]), qd = as_vec(c["qd"]), dqd = as_vec(c["dqd"]),
                       ddqd = as_vec(c["ddqd"]), eint = as_vec(c["eint"]);
        // mr::computed_torque adds the current error to the integral it is given (Ki (eint + e)), so pass eint - e
        const VectorXd ref = mr::computed_torque(q, dq, eint - (qd - q), m0.g, m0.Mlist, m0.Glist, m0.Slist, qd, dqd,
                                                 ddqd, 400.0, 50.0, 40.0);
        const VectorXd tau = lab::computed_torque(q, dq, eint, qd, dqd, ddqd, 400.0, 50.0, 40.0, m0);
        CHECK_ALLCLOSE(tau, ref, 1e-9);
        CHECK_MAT_NEAR(tau, as_vec(c["tau"]), 1e-9);
        // friction model enters h~ as b~ dq; vector gains act as diagonal matrices
        const VectorXd e = qd - q, ed = dqd - dq;
        const MatrixXd Mm = mr::mass_matrix(q, model.Mlist, model.Glist, model.Slist);
        const VectorXd h = mr::inverse_dynamics(q, dq, VectorXd::Zero(6), model.g, Vector6d::Zero(), model.Mlist,
                                                model.Glist, model.Slist) +
                           model.b.cwiseProduct(dq);
        const VectorXd ref2 =
            Mm * (ddqd + kv.cwiseProduct(e) + 10 * eint + (0.1 * kv).cwiseProduct(ed)) + h;
        const VectorXd tau2 = lab::computed_torque(q, dq, eint, qd, dqd, ddqd, kv, 10.0, 0.1 * kv, model);
        CHECK_ALLCLOSE(tau2, ref2, 1e-9);
        CHECK_MAT_NEAR(tau2, as_vec(c["tau_vector_gains"]), 1e-9);
    }
}

TEST_CASE("test_computed_torque_linearizes_exact_plant") {
    // Exact model: the plant acceleration under computed torque is qd'' + kp e + kd e' (MR Eq. 11.33, Ki = 0).
    const auto [plant, model] = lab::make_case("exact", cfg());
    const lab::Plant P(plant);
    const json& c = cases()["computed_torque_linearizes_exact_plant"];
    const VectorXd q = as_vec(c["q"]), dq = as_vec(c["dq"]), qd = as_vec(c["qd"]), dqd = as_vec(c["dqd"]),
                   ddqd = as_vec(c["ddqd"]);
    const VectorXd tau = lab::computed_torque(q, dq, VectorXd::Zero(6), qd, dqd, ddqd, 400.0, 0.0, 40.0, model);
    CHECK_ALLCLOSE(P.accel(q, dq, tau), ddqd + 400 * (qd - q) + 40 * (dqd - dq), 1e-7);
    CHECK_MAT_NEAR(tau, as_vec(c["tau"]), 1e-9);
}

TEST_CASE("test_computed_torque_with_scaled_model") {
    // M~ = a M, h~ = a (c + g) + b dq:  q'' = a (qd'' + kp e + kd e') + (a - 1) M^-1 (c + g)  (model-error identity)
    const json& CFG = cfg();
    const lab::Robot plant = lab::arm_model(0.25, 1.0, as_vec(CFG["friction_b"]));
    const lab::Robot model = lab::arm_model(0.25, 1.2, as_vec(CFG["friction_b"]));
    const lab::Plant P(plant);
    const json& c = cases()["computed_torque_with_scaled_model"];
    const VectorXd q = as_vec(c["q"]), dq = as_vec(c["dq"]), qd = as_vec(c["qd"]), dqd = as_vec(c["dqd"]),
                   ddqd = as_vec(c["ddqd"]);
    const VectorXd tau = lab::computed_torque(q, dq, VectorXd::Zero(6), qd, dqd, ddqd, 400.0, 0.0, 40.0, model);
    const MatrixXd Mm = mr::mass_matrix(q, plant.Mlist, plant.Glist, plant.Slist);
    const VectorXd cg = mr::inverse_dynamics(q, dq, VectorXd::Zero(6), plant.g, Vector6d::Zero(), plant.Mlist,
                                             plant.Glist, plant.Slist);
    const VectorXd pred = 1.2 * (ddqd + 400 * (qd - q) + 40 * (dqd - dq)) + 0.2 * Mm.partialPivLu().solve(cg);
    CHECK_ALLCLOSE(P.accel(q, dq, tau), pred, 1e-6);
    CHECK_MAT_NEAR(tau, as_vec(c["tau"]), 1e-9);
}

TEST_CASE("test_error_energy_and_its_derivative") {
    // V of MR Eq. 11.41, and dV/dt = -dq^T (Kd + B) dq under PD + exact gravity compensation (MR Eq. 11.43 with
    // viscous friction B), checked by differentiating V along the plant dynamics.
    const auto [plant, model] = lab::make_case("exact", cfg());
    const lab::Plant P(plant);
    const json& c = cases()["error_energy_and_its_derivative"];
    const VectorXd q = as_vec(c["q"]), dq = as_vec(c["dq"]), qd = as_vec(c["qd"]);
    const VectorXd Kp = vec6({4, 4, 2, 1, 0.5, 0.1}), Kd = vec6({0.4, 0.4, 0.2, 0.1, 0.05, 0.01});
    const MatrixXd Mm = mr::mass_matrix(q, model.Mlist, model.Glist, model.Slist);
    const VectorXd e = qd - q;
    const double V = lab::error_energy(q, dq, qd, Kp, model);
    CHECK_APPROX(V, 0.5 * e.dot(Kp.cwiseProduct(e)) + 0.5 * dq.dot(Mm * dq));
    CHECK_APPROX(V, num(c["V"]));
    const VectorXd z = VectorXd::Zero(6);
    const VectorXd tau = lab::pd_gravity_torque(q, dq, z, qd, z, Kp, 0.0, Kd, model);
    const VectorXd ddq = P.accel(q, dq, tau);
    const double h = 1e-6;
    const double Vp = lab::error_energy(q + h * dq, dq + h * ddq, qd, Kp, model);
    const double Vm = lab::error_energy(q - h * dq, dq - h * ddq, qd, Kp, model);
    const double dVdt = (Vp - Vm) / (2 * h);
    const double expected = -dq.dot((Kd + plant.b).cwiseProduct(dq));
    INFO("dV/dt = " << dVdt << ", expected " << expected);
    CHECK(approx(dVdt, expected, 1e-5, 1e-9));  // pytest.approx(rel=1e-5, abs=1e-9)
}

TEST_CASE("test_joint_controller_integral_and_dispatch") {
    const lab::Robot model = lab::make_case("exact", cfg()).second;
    const VectorXd t = lab::time_grid(0.01, 0.002);
    const json& c = cases()["joint_controller_integral_and_dispatch"];
    const MatrixXd qd = as_mat(c["qd"]), dqd = as_mat(c["dqd"]), ddqd = as_mat(c["ddqd"]);
    REQUIRE(qd.rows() == t.size());
    const VectorXd q = as_vec(c["q"]), dq = as_vec(c["dq"]);
    for (const std::string law : {"pid", "pd_gravity", "feedforward", "computed_torque"}) {
        INFO("law = " << law);
        lab::JointController ctrl(law, t, qd, dqd, ddqd, 2.0, 5.0, 0.3, model);
        ctrl.reset();
        VectorXd eint = VectorXd::Zero(6), tau;
        for (int k = 0; k < 3; ++k) {
            eint += (row(qd, k) - q) * 0.002;
            tau = ctrl(k, q, dq);
        }
        CHECK_ALLCLOSE(ctrl.eint, eint, 1e-8);
        const VectorXd e2 = row(qd, 2) - q, ed2 = row(dqd, 2) - dq;
        VectorXd ref;
        if (law == "pid") {
            ref = 2.0 * e2 + 5.0 * eint + 0.3 * ed2;
        } else if (law == "pd_gravity") {
            ref = 2.0 * e2 + 5.0 * eint + 0.3 * ed2 +
                  mr::gravity_forces(q, model.g, model.Mlist, model.Glist, model.Slist);
        } else if (law == "feedforward") {
            ref = 2.0 * e2 + 5.0 * eint + 0.3 * ed2 +
                  mr::inverse_dynamics(row(qd, 2), row(dqd, 2), row(ddqd, 2), model.g, Vector6d::Zero(), model.Mlist,
                                       model.Glist, model.Slist) +
                  model.b.cwiseProduct(row(dqd, 2));
        } else {
            ref = mr::computed_torque(q, dq, eint - e2, model.g, model.Mlist, model.Glist, model.Slist, row(qd, 2),
                                      row(dqd, 2), row(ddqd, 2), 2.0, 5.0, 0.3) +
                  model.b.cwiseProduct(dq);
        }
        CHECK_ALLCLOSE(tau, ref, 1e-9);
        CHECK_MAT_NEAR(tau, as_vec(c["tau"][law]), 1e-9);
        ctrl.reset();
        CHECK_ALLCLOSE(ctrl.eint, VectorXd::Zero(6), 1e-8);
    }
    // C++ only (Python raises ValueError in __init__): unknown law, or a model-based law without a model
    CHECK_THROWS_AS(lab::JointController("pd", t, qd, dqd, ddqd, 2.0, 5.0, 0.3, model), std::invalid_argument);
    CHECK_THROWS_AS(lab::JointController("computed_torque", t, qd, dqd, ddqd, 2.0, 5.0, 0.3), std::invalid_argument);
}

TEST_CASE("test_run_metrics") {
    VectorXd t(3);
    t << 0.0, 1.0, 2.0;
    const MatrixXd qd = MatrixXd::Zero(3, 2);
    MatrixXd q(3, 2), tau(3, 2);
    q << 0.0, 0.0, 3.0, 4.0, 0.0, 1.0;
    tau << 1.0, 0.0, 1.0, 1.0, 0.0, -2.0;
    const lab::Metrics m = lab::run_metrics(t, qd, q, tau);
    CHECK_APPROX(m.max_err, 5.0);
    CHECK_APPROX(m.final_err, 1.0);
    CHECK_APPROX(m.rms_err, std::sqrt((0 + 25 + 1) / 3.0));
    CHECK_APPROX(m.effort, 0.5 * (1 + 2) + 0.5 * (2 + 4));
    CHECK_ALLCLOSE(m.peak_tau, Eigen::Vector2d(1.0, 2.0), 1e-8);
}

// ============================================================================ closed-loop behaviour (short runs)
TEST_CASE("test_closed_loop_ranking_exact_model") {
    const lab::Metrics ct = joint_run("exact", "computed_torque");
    const lab::Metrics pdg = joint_run("exact", "pd_gravity");
    const lab::Metrics pid = joint_run("exact", "pid", 0.8, 0.1);
    CHECK(ct.max_err < 2e-3);
    CHECK(ct.max_err < 0.1 * pdg.max_err);
    CHECK(pdg.max_err < pid.max_err);
    // C++ only: the same runs as the Python reference (Pinocchio plant there, mr::forward_dynamics here)
    const json& c = cases()["closed_loop_ranking_exact_model"];
    check_metrics_vs_python(ct, c["ct"], 1e-6);
    check_metrics_vs_python(pdg, c["pd_gravity"], 1e-6);
    check_metrics_vs_python(pid, c["pid"], 1e-6);
}

TEST_CASE("test_ct_steady_state_error_under_model_error") {
    // Setpoint hold with M~ = 1.2 M: at rest  kp M~(q) e = g(q) - g~(q) = -0.2 g(q), q = qd - e (solved by Newton);
    // the simulated computed-torque loop must settle there.
    const std::pair<lab::Robot, lab::Robot> pm = lab::make_case("masses_120", cfg());
    const lab::Robot& plant = pm.first;  // (no structured binding: the lambda F below captures them)
    const lab::Robot& model = pm.second;
    const VectorXd qset = as_vec(cfg()["setpoint"]["q0"]);
    const VectorXd t = lab::time_grid(1.2, num(cfg()["dt"]));
    const Eigen::Index N = t.size();
    const MatrixXd Z = MatrixXd::Zero(N, 6);
    lab::JointController ctrl("computed_torque", t, tile(qset, N), Z, Z, 400.0, 0.0, 40.0, model);
    const lab::SimResult out = lab::simulate_arm(plant, ctrl, t, qset, VectorXd(), cfg()["substeps"].get<int>());
    REQUIRE(out.q.rows() == N);
    auto F = [&](const VectorXd& e_) {
        const VectorXd q = qset - e_;
        const MatrixXd Mm = mr::mass_matrix(q, model.Mlist, model.Glist, model.Slist);
        const VectorXd gp = mr::gravity_forces(q, plant.g, plant.Mlist, plant.Glist, plant.Slist);
        const VectorXd gm = mr::gravity_forces(q, model.g, model.Mlist, model.Glist, model.Slist);
        return VectorXd(400.0 * Mm * e_ - (gp - gm));
    };
    VectorXd e = VectorXd::Zero(6);
    for (int it = 0; it < 30; ++it) {
        MatrixXd Jn(6, 6);
        for (int i = 0; i < 6; ++i) {
            const VectorXd u = 1e-7 * VectorXd::Unit(6, i);
            Jn.col(i) = (F(e + u) - F(e - u)) / 2e-7;
        }
        e -= Jn.partialPivLu().solve(F(e));
    }
    CHECK_ALLCLOSE(qset - row(out.q, N - 1), e, 2e-4);
    CHECK(e.norm() > 0.02);
    const json& c = cases()["ct_steady_state_error_under_model_error"];
    CHECK_MAT_NEAR(e, as_vec(c["e_newton"]), 1e-9);
    CHECK_MAT_NEAR(row(out.q, N - 1), as_vec(c["q_final"]), 1e-6);  // Python reference run
}

TEST_CASE("test_setpoint_pd_gravity_energy_decreases") {
    const json& sp = cfg()["setpoint"];
    const json& g = cfg()["gains"];
    const auto [plant, model] = lab::make_case("exact", cfg());
    const VectorXd q0 = as_vec(sp["q0"]);
    const VectorXd qset = q0 + as_vec(sp["dq0_offset"]);
    const VectorXd t = lab::time_grid(0.8, num(cfg()["dt"]));
    const Eigen::Index N = t.size();
    const lab::PIDGains G = lab::decentralized_gains(model, as_vec(g["gain_reference_q"]), num(g["zeta"]), num(g["wn"]));
    const MatrixXd Z = MatrixXd::Zero(N, 6);
    lab::JointController ctrl("pd_gravity", t, tile(qset, N), Z, Z, G.Kp, 0.0, G.Kd, model);
    const lab::SimResult out = lab::simulate_arm(plant, ctrl, t, q0, VectorXd(), cfg()["substeps"].get<int>());
    REQUIRE(out.q.rows() == N);
    std::vector<double> V;
    for (Eigen::Index k = 0; k < N; k += 5) V.push_back(lab::error_energy(row(out.q, k), row(out.dq, k), qset, G.Kp, plant));
    bool decreasing = true;
    for (std::size_t k = 1; k < V.size(); ++k) decreasing = decreasing && (V[k] - V[k - 1] <= 1e-12);
    CHECK(decreasing);
    CHECK(V.back() < 1e-3 * V.front());
    const VectorXd Vpy = as_vec(cases()["setpoint_pd_gravity_energy_decreases"]["V"]);
    REQUIRE(Vpy.size() == static_cast<Eigen::Index>(V.size()));
    CHECK_MAT_NEAR(Eigen::Map<const VectorXd>(V.data(), Vpy.size()), Vpy, 1e-8);  // Python reference run
}

// ============================================================================ Part C: task space
TEST_CASE("test_body_jacobian_dot") {
    const lab::Robot r = lab::arm_model(0.25);
    for (const auto& c : cases()["body_jacobian_dot"]) {
        const VectorXd q = as_vec(c["q"]), dq = as_vec(c["dq"]);
        const double h = 1e-6;
        const MatrixXd numd = (mr::jacobian_body(r.Blist, q + h * dq) - mr::jacobian_body(r.Blist, q - h * dq)) / (2 * h);
        const MatrixXd Jd = lab::body_jacobian_dot(r.Blist, q, dq);
        CHECK_ALLCLOSE(Jd, numd, 1e-7);
        CHECK_MAT_NEAR(Jd, as_mat(c["Jdot"]), 1e-12);
    }
}

TEST_CASE("test_task_errors") {
    const lab::Robot r = lab::arm_model(0.25);
    const json& c = cases()["task_errors"];
    const VectorXd q1 = r.q_home, q2 = as_vec(c["q2"]);
    const Matrix4d X = mr::fkin_body(r.M, r.Blist, q1), Xd = mr::fkin_body(r.M, r.Blist, q2);
    const Vector6d V = as<Vector6d>(c["V"]), Vd = as<Vector6d>(c["Vd"]);
    const auto [Xe, Ve] = lab::task_errors(X, V, Xd, Vd);
    CHECK_ALLCLOSE(X * mr::matrix_exp6(mr::vec_to_se3(Xe)), Xd, 1e-10);
    CHECK_ALLCLOSE(Ve, mr::adjoint(mr::trans_inv(X) * Xd) * Vd - V, 1e-8);
    const auto [Xe0, Ve0] = lab::task_errors(X, V, X, V);
    CHECK_ALLCLOSE(Xe0, Vector6d::Zero(), 1e-12);
    CHECK_ALLCLOSE(Ve0, Vector6d::Zero(), 1e-12);
    CHECK_MAT_NEAR(Xe, as_vec(c["Xe"]), 1e-10);
    CHECK_MAT_NEAR(Ve, as_vec(c["Ve"]), 1e-10);
}

TEST_CASE("test_task_feedforward_accel") {
    const lab::Robot r = lab::arm_model(0.25);
    const Matrix4d X0 = mr::fkin_body(r.M, r.Blist, r.q_home);
    const json& c = cases()["task_feedforward_accel"];
    const Vector6d W1 = as<Vector6d>(c["W1"]), W2 = as<Vector6d>(c["W2"]);
    auto Xa = [&](double s) { return Matrix4d(X0 * mr::matrix_exp6(mr::vec_to_se3(W1 * (s + 0.3 * s * s)))); };
    auto Xb = [&](double s) { return Matrix4d(X0 * mr::matrix_exp6(mr::vec_to_se3(W2 * (s - 0.2 * s * s * s)))); };
    auto twist = [](const auto& f, double s, double h = 1e-6) {
        return Vector6d(mr::se3_to_vec(mr::trans_inv(f(s)) * (f(s + h) - f(s - h)) / (2 * h)));
    };
    auto W = [&](double s) { return Vector6d(mr::adjoint(mr::trans_inv(Xa(s)) * Xb(s)) * twist(Xb, s)); };
    const double s = 0.4, h = 1e-4;
    const Vector6d numd = (W(s + h) - W(s - h)) / (2 * h);
    const Vector6d dVd = (twist(Xb, s + h) - twist(Xb, s - h)) / (2 * h);
    const Vector6d got = lab::task_feedforward_accel(Xa(s), twist(Xa, s), Xb(s), twist(Xb, s), dVd);
    CHECK_ALLCLOSE(got, numd, 1e-6);
    CHECK_MAT_NEAR(got, as_vec(c["got"]), 1e-6);  // finite-difference inputs: agree to ~1e-8
}

TEST_CASE("test_model_task_space_dynamics_and_ct_equal_eq_11_47") {
    const lab::Robot model = lab::make_case("masses_120", cfg()).second;
    const json& c = cases()["model_task_space_dynamics_and_ct_equal_eq_11_47"];
    const VectorXd q = as_vec(c["q"]), dq = as_vec(c["dq"]), ddq = as_vec(c["ddq"]);
    const auto [Lam, eta] = lab::model_task_space_dynamics(model, q, dq);
    CHECK_ALLCLOSE(Lam, Lam.transpose(), 1e-10);
    const Eigen::SelfAdjointEigenSolver<Matrix6d> es(0.5 * (Lam + Lam.transpose()));
    CHECK(es.eigenvalues().minCoeff() > 0);
    const MatrixXd J = mr::jacobian_body(model.Blist, q);
    const MatrixXd Jd = lab::body_jacobian_dot(model.Blist, q, dq);
    const VectorXd tau_id = mr::inverse_dynamics(q, dq, ddq, model.g, Vector6d::Zero(), model.Mlist, model.Glist,
                                                 model.Slist) +
                            model.b.cwiseProduct(dq);
    CHECK_ALLCLOSE(J.transpose() * (Lam * (J * ddq + Jd * dq) + eta), tau_id, 1e-8);  // MR Eq. 8.89
    const Matrix4d X = mr::fkin_body(model.M, model.Blist, q);
    const Matrix4d Xd = mr::fkin_body(model.M, model.Blist, as_vec(c["q_d"]));
    const Vector6d Vd = as<Vector6d>(c["Vd"]), dVd = as<Vector6d>(c["dVd"]), Xint = as<Vector6d>(c["Xint"]);
    const Vector6d V = J * dq;
    const auto [Xe, Ve] = lab::task_errors(X, V, Xd, Vd);
    const Vector6d a = lab::task_feedforward_accel(X, V, Xd, Vd, dVd) + 400 * Xe + 50 * Xint + 40 * Ve;
    const VectorXd ref = J.transpose() * (Lam * a + eta);  // MR Eq. 11.47
    const VectorXd tau = lab::task_space_ct(q, dq, Xint, Xd, Vd, dVd, 400.0, 50.0, 40.0, model);
    CHECK_ALLCLOSE(tau, ref, 1e-8);
    CHECK_MAT_NEAR(Lam, as_mat(c["Lambda"]), 1e-8);
    CHECK_MAT_NEAR(eta, as_vec(c["eta"]), 1e-8);
    CHECK_MAT_NEAR(tau, as_vec(c["tau"]), 1e-8);
}

TEST_CASE("test_task_controller_tracks_line_with_exact_model") {
    const json& tm = cfg()["task_move"];
    const json& g = cfg()["gains"];
    const auto [plant, model] = lab::make_case("exact", cfg());
    const VectorXd t = lab::time_grid(0.6, num(cfg()["dt"]));
    const VectorXd q0 = as_vec(tm["q0"]);
    const Matrix4d X0 = mr::fkin_body(model.M, model.Blist, q0);
    const lab::CartesianReference ref = lab::cartesian_reference(X0, as<Eigen::Vector3d>(tm["dp_s"]), num(tm["T"]), t);
    const auto [kp, kd] = lab::ct_gains(num(g["zeta"]), num(g["wn"]));
    lab::TaskController ctrl(t, ref.Xd, ref.Vd, ref.dVd, kp, 0.0, kd, model);
    const lab::SimResult out = lab::simulate_arm(plant, ctrl, t, q0, VectorXd(), cfg()["substeps"].get<int>(),
                                                 lab::tau_limits(plant, cfg()));
    REQUIRE(out.q.rows() == t.size());
    double max_err = 0.0, final_err = 0.0;
    for (Eigen::Index k = 0; k < t.size(); ++k) {
        const Matrix4d Xk = mr::fkin_body(plant.M, plant.Blist, row(out.q, k));
        final_err = (Xk.topRightCorner<3, 1>() - ref.Xd[static_cast<std::size_t>(k)].topRightCorner<3, 1>()).norm();
        max_err = std::max(max_err, final_err);
    }
    CHECK(max_err < 1e-4);  // < 0.1 mm along a 50 mm line
    CHECK((ref.Xd.back().topRightCorner<3, 1>() - X0.topRightCorner<3, 1>()).norm() > 0.02);  // the reference moved
    const json& c = cases()["task_controller_tracks_line_with_exact_model"];
    CHECK_SCALAR_NEAR(max_err, num(c["max_err"]), 1e-8);  // Python reference run
    CHECK_SCALAR_NEAR(final_err, num(c["final_err"]), 1e-8);
}
