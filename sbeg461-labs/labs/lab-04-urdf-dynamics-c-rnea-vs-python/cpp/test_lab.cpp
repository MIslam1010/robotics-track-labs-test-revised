// Auto-checks for SBEG 461 Lab 4 (URDF dynamics, RNEA timing, constrained 2R) -- C++ track.
// Mirrors test_lab.py test by test (same names, same order, same tolerances).
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*urdf*"            (run only some tests, like pytest -k)
// Ground truth: mrcpp (the C++ port of modern_robotics), the numbers printed in MR Example 8.5, and
// cpp_cases.json, which holds the random inputs of test_lab.py and the Pinocchio / Python reference values,
// so both languages are checked against the same numbers.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

using Eigen::Matrix2d;
using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using lab::Matrix6d;
using lab::Vector6d;
using sbeg::as;
using sbeg::as_mat;
using sbeg::as_vec;
using sbeg::json;

namespace {
constexpr double PI = 3.14159265358979323846;

const json& cases() {  // loaded once
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

std::string urdf() { return sbeg::lab_path("../arm6.urdf"); }
std::string csv() { return sbeg::lab_path("../states_arm6.csv"); }

const Vector3d G(0.0, 0.0, -9.81);

// MR Example 8.5 state: theta = (-pi/3, 2pi/3), thetadot = (1, 0), g = 0, L = m = 1
const Vector2d TH85(-PI / 3.0, 2.0 * PI / 3.0);
const Vector2d DTH85(1.0, 0.0);

/// One of the 8 STATES of test_lab.py (rng 461, stored in cpp_cases.json).
struct State {
    VectorXd th, dth, ddth;
};
std::vector<State> states() {
    std::vector<State> out;
    for (const auto& s : cases()["states"]) out.push_back({as_vec(s["theta"]), as_vec(s["dtheta"]), as_vec(s["ddtheta"])});
    return out;
}

/// The MR 2R arm (Fig. 8.1) as Mlist/Glist/Slist with point masses (zero rotational inertia).
struct Model2R {
    lab::TransformList Mlist;
    lab::InertiaList Glist;
    MatrixXd Slist;
};
Model2R mr_2r_model(double m1 = 1.0, double m2 = 1.0, double L1 = 1.0, double L2 = 1.0) {
    Model2R m;
    m.Mlist = {mr::rp_to_trans(Matrix3d::Identity(), Vector3d(L1, 0, 0)),
               mr::rp_to_trans(Matrix3d::Identity(), Vector3d(L2, 0, 0)), Matrix4d::Identity()};
    Vector6d d1, d2;
    d1 << 0, 0, 0, m1, m1, m1;
    d2 << 0, 0, 0, m2, m2, m2;
    m.Glist = {Matrix6d(d1.asDiagonal()), Matrix6d(d2.asDiagonal())};
    m.Slist.resize(6, 2);
    m.Slist << 0, 0,  0, 0,  1, 1,  0, 0,  0, -L1,  0, 0;
    return m;
}

Vector2d head2(const VectorXd& v) { return v.head<2>(); }

std::string temp_file(const std::string& name) {
    const std::filesystem::path p = std::filesystem::temp_directory_path() / name;
#if defined(__cpp_char8_t)
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
#else
    return p.u8string();
#endif
}

/// Read back the model file written by export_for_cpp (the format main.cpp reads).
lab::MRModel read_model_file(const std::string& path) {
    std::ifstream f(sbeg::fs_path(path));
    REQUIRE(f.good());
    int n = 0;
    f >> n;
    REQUIRE(n == 6);
    lab::MRModel m;
    m.Mlist.assign(7, Matrix4d::Zero());
    m.Glist.assign(6, Matrix6d::Zero());
    m.Slist = MatrixXd::Zero(6, 6);
    for (auto& M : m.Mlist)
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) f >> M(r, c);
    for (auto& Gi : m.Glist)
        for (int r = 0; r < 6; ++r)
            for (int c = 0; c < 6; ++c) f >> Gi(r, c);
    for (int i = 0; i < 6; ++i)
        for (int r = 0; r < 6; ++r) f >> m.Slist(r, i);
    REQUIRE(f.good());
    return m;
}
}  // namespace

// ---------------------------------------------------------------- Part A: URDF -> MR parameters

TEST_CASE("test_urdf_shapes") {
    const lab::MRModel m = lab::urdf_to_mr(urdf());
    CHECK(m.Mlist.size() == 7);
    CHECK(m.Glist.size() == 6);
    CHECK(m.Slist.rows() == 6);
    CHECK(m.Slist.cols() == 6);
    for (const Matrix6d& Gi : m.Glist) {
        CHECK_MAT_NEAR(Gi, Gi.transpose(), 1e-8);
        Eigen::SelfAdjointEigenSolver<Matrix6d> es(Gi);
        CHECK(es.eigenvalues().minCoeff() > 0.0);
    }
}

TEST_CASE("test_urdf_forward_kinematics_matches_pinocchio") {
    const lab::MRModel m = lab::urdf_to_mr(urdf());
    Matrix4d M = Matrix4d::Identity();
    for (const Matrix4d& Mi : m.Mlist) M = M * Mi;
    for (const auto& c : cases()["urdf_forward_kinematics_matches_pinocchio"]) {
        const VectorXd th = as_vec(c["theta"]);
        CHECK_MAT_NEAR(mr::fkin_space(M, m.Slist, th), as<Matrix4d>(c["T_tool0"]), 1e-9);
    }
}

TEST_CASE("test_urdf_dynamics_match_pinocchio") {
    // The parsed (Mlist, Glist, Slist) must give the same torques as Pinocchio on the same URDF.
    const lab::MRModel m = lab::urdf_to_mr(urdf());
    const std::vector<State> S = states();
    const json& ref = cases()["urdf_dynamics_match_pinocchio"];
    REQUIRE(ref.size() == S.size());
    for (std::size_t k = 0; k < S.size(); ++k) {
        INFO("state " << k);
        const VectorXd tau_mr =
            mr::inverse_dynamics(S[k].th, S[k].dth, S[k].ddth, G, Vector6d::Zero(), m.Mlist, m.Glist, m.Slist);
        CHECK_MAT_NEAR(tau_mr, as_vec(ref[k]["tau"]), 1e-8);
    }
}

// ---------------------------------------------------------------- Part B: RNEA and timing

TEST_CASE("test_rnea_matches_modern_robotics") {
    const lab::MRModel m = lab::urdf_to_mr(urdf());
    const std::vector<State> S = states();
    const json& C = cases()["rnea_matches_modern_robotics"];
    REQUIRE(C.size() == S.size());
    for (std::size_t k = 0; k < S.size(); ++k) {
        INFO("state " << k);
        const Vector6d Ftip = as<Vector6d>(C[k]["Ftip"]);
        const VectorXd ref = mr::inverse_dynamics(S[k].th, S[k].dth, S[k].ddth, G, Ftip, m.Mlist, m.Glist, m.Slist);
        CHECK_MAT_NEAR(lab::rnea(S[k].th, S[k].dth, S[k].ddth, G, Ftip, m.Mlist, m.Glist, m.Slist), ref, 1e-9);
    }
}

TEST_CASE("test_rnea_2r_matches_closed_form") {
    const Model2R m = mr_2r_model(2.0, 1.5, 0.8, 0.6);
    const Vector3d g3(0.0, -9.81, 0.0);
    for (const State& s : states()) {
        const Vector2d th = head2(s.th), dth = head2(s.dth), ddth = head2(s.ddth);
        const auto [M, h] = lab::twor_dynamics(th, dth, 2.0, 1.5, 0.8, 0.6, 9.81);
        const VectorXd tau = lab::rnea(th, dth, ddth, g3, Vector6d::Zero(), m.Mlist, m.Glist, m.Slist);
        CHECK_MAT_NEAR(tau, M * ddth + h, 1e-9);
    }
}

TEST_CASE("test_time_per_call_is_positive_and_scales") {
    volatile double sink = 0.0;  // keeps the optimiser from removing the work
    volatile double xin = 1.0;
    auto slow = [&](std::size_t) {
        const double x = xin;
        double s = 0.0;
        for (int k = 0; k < 2000; ++k) s += x * k;
        sink = s;
    };
    auto fast = [&](std::size_t) { sink = xin; };
    // Python times 20 calls. A C++ call of `fast` takes about 1 ns and steady_clock ticks every 100 ns on some
    // Windows machines, so 20 calls could measure 0; with 2000 calls the time per call is measurable everywhere.
    const double t_slow = lab::time_per_call(slow, 2000);
    const double t_fast = lab::time_per_call(fast, 2000);
    CHECK(t_slow > 0.0);
    CHECK(t_fast > 0.0);
    CHECK(t_slow > 5.0 * t_fast);
}

TEST_CASE("test_cpp_rnea_agrees_with_python") {
    // Python runs the Step 3 program rnea_bench on the exported files. The C++ track checks the same chain with
    // YOUR C++ functions: urdf_to_mr -> export_for_cpp (the files rnea_bench reads) -> read back -> your rnea on
    // the first 50 states of states_arm6.csv, compared with mrcpp and with the Python reference torques.
    const lab::MRModel m = lab::urdf_to_mr(urdf());
    const json& C = cases()["cpp_rnea_agrees_with_python"];
    const auto n_states = C["n_states"].get<Eigen::Index>();
    const MatrixXd X = lab::load_states(csv()).topRows(n_states);
    const std::string mp = temp_file("sbeg461_lab04_model.txt"), sp = temp_file("sbeg461_lab04_states.txt");
    lab::export_for_cpp(mp, sp, m.Mlist, m.Glist, m.Slist, X);
    const lab::MRModel back = read_model_file(mp);
    const MatrixXd tau_ref = as_mat(C["tau"]);
    REQUIRE(tau_ref.rows() == n_states);
    for (Eigen::Index k = 0; k < n_states; ++k) {
        INFO("state " << k);
        const VectorXd th = X.row(k).segment(0, 6).transpose(), dth = X.row(k).segment(6, 6).transpose(),
                       ddth = X.row(k).segment(12, 6).transpose();
        const VectorXd tau = lab::rnea(th, dth, ddth, G, Vector6d::Zero(), back.Mlist, back.Glist, back.Slist);
        CHECK_MAT_NEAR(tau, mr::inverse_dynamics(th, dth, ddth, G, Vector6d::Zero(), m.Mlist, m.Glist, m.Slist), 1e-9);
        CHECK_MAT_NEAR(tau, tau_ref.row(k), 1e-9);
    }
    volatile double sink = 0.0;
    const double us = 1e6 * lab::time_per_call([&](std::size_t k) {
        const auto r = static_cast<Eigen::Index>(k);
        sink = lab::rnea(X.row(r).segment(0, 6).transpose(), X.row(r).segment(6, 6).transpose(),
                         X.row(r).segment(12, 6).transpose(), G, Vector6d::Zero(), m.Mlist, m.Glist, m.Slist)(0);
    }, static_cast<std::size_t>(n_states), 5);
    MESSAGE("your C++ rnea: " << us << " us per call (this build; see lab_bench for the timing table)");
    CHECK(us > 0.0);
    std::error_code ec;
    std::filesystem::remove(sbeg::fs_path(mp), ec);
    std::filesystem::remove(sbeg::fs_path(sp), ec);
}

// ---------------------------------------------------------------- Part C: task-space dynamics

TEST_CASE("test_task_space_dynamics_consistent_with_joint_space") {
    // F = Lambda Vdot + eta must equal J^-T tau for tau = M thetaddot + h (MR Eqs. 8.82-8.91).
    for (const State& s : states()) {
        const Vector2d th = head2(s.th), dth = head2(s.dth), ddth = head2(s.ddth);
        if (std::abs(std::sin(th(1))) < 0.2) continue;
        const auto [M, h] = lab::twor_dynamics(th, dth, 1.0, 1.0, 1.0, 1.0, 9.81);
        const auto [J, Jd] = lab::twor_jacobian(th, dth);
        const auto [Lam, eta] = lab::task_space_dynamics(M, h, J, Jd, dth);
        CHECK_MAT_NEAR(Lam, Lam.transpose(), 1e-12);
        const Vector2d tau = M * ddth + h;
        const Vector2d Vdot = J * ddth + Jd * dth;
        const Vector2d F = J.transpose().partialPivLu().solve(tau);
        CHECK_MAT_NEAR(Lam * Vdot + eta, F, 1e-9);
    }
}

TEST_CASE("test_task_space_mass_matrix_value") {
    // MR 2R at theta = (0, pi/2): Lambda = diag(1, 2).
    const Vector2d th(0.0, PI / 2.0), zero(0.0, 0.0);
    const auto [M, h] = lab::twor_dynamics(th, zero);
    const auto [J, Jd] = lab::twor_jacobian(th, zero);
    const auto [Lam, eta] = lab::task_space_dynamics(M, h, J, Jd, zero);
    CHECK_MAT_NEAR(Lam, Vector2d(1.0, 2.0).asDiagonal().toDenseMatrix(), 1e-12);
    CHECK_MAT_NEAR(eta, Vector2d::Zero(), 1e-12);
}

// ---------------------------------------------------------------- Part D: constrained 2R

TEST_CASE("test_twor_dynamics_matches_modern_robotics") {
    const Model2R m = mr_2r_model();
    for (const State& s : states()) {
        const Vector2d th = head2(s.th), dth = head2(s.dth);
        const auto [M, h] = lab::twor_dynamics(th, dth, 1.0, 1.0, 1.0, 1.0, 9.81);
        CHECK_MAT_NEAR(M, mr::mass_matrix(th, m.Mlist, m.Glist, m.Slist), 1e-12);
        const VectorXd h_ref = mr::vel_quadratic_forces(th, dth, m.Mlist, m.Glist, m.Slist) +
                               mr::gravity_forces(th, Vector3d(0.0, -9.81, 0.0), m.Mlist, m.Glist, m.Slist);
        CHECK_MAT_NEAR(h, h_ref, 1e-12);
    }
}

TEST_CASE("test_channel_constraint_is_gradient_of_x") {
    const double eps = 1e-6;
    auto x = [](const Vector2d& q) { return std::cos(q(0)) + std::cos(q(0) + q(1)); };
    for (const State& s : states()) {
        const Vector2d th = head2(s.th), dth = head2(s.dth);
        const auto [A, Ad] = lab::channel_constraint(th, dth);
        REQUIRE(A.rows() == 1);
        REQUIRE(A.cols() == 2);
        Vector2d grad;
        for (int i = 0; i < 2; ++i) {
            const Vector2d e = Vector2d::Unit(i);
            grad(i) = (x(th + eps * e) - x(th - eps * e)) / (2.0 * eps);
        }
        CHECK_MAT_NEAR(A.row(0), grad, 1e-8);
        const MatrixXd A_plus = lab::channel_constraint(th + eps * dth, dth).first;
        const MatrixXd A_minus = lab::channel_constraint(th - eps * dth, dth).first;
        CHECK_MAT_NEAR(Ad, (A_plus - A_minus) / (2.0 * eps), 1e-7);
    }
}

TEST_CASE("test_mr_example_8_5_forward") {
    // MR Example 8.5: lambda = 0.289 tau1 - 1.155 tau2 - 0.167, thdd1 = 0.5 tau1 + 0.289, thdd2 = -1.155.
    const auto [M, h] = lab::twor_dynamics(TH85, DTH85);
    const auto [A, Ad] = lab::channel_constraint(TH85, DTH85);
    MatrixXd A_ref(1, 2), Ad_ref(1, 2);
    A_ref << 0.0, -std::sqrt(3.0) / 2.0;
    Ad_ref << -1.0, -0.5;
    CHECK_MAT_NEAR(A, A_ref, 1e-12);
    CHECK_MAT_NEAR(Ad, Ad_ref, 1e-12);
    for (const auto& c : cases()["mr_example_8_5_forward"]) {
        const Vector2d tau = as<Vector2d>(c["tau"]);
        INFO("tau = " << tau.transpose());
        const auto [ddth, lam] = lab::constrained_forward_dynamics(M, h, A, Ad, DTH85, tau);
        const double t1 = tau(0), t2 = tau(1);
        REQUIRE(lam.size() == 1);
        REQUIRE(ddth.size() == 2);
        CHECK_MAT_NEAR(lam, Eigen::VectorXd::Constant(1, 0.2887 * t1 - 1.1547 * t2 - 0.1667), 2e-4);
        CHECK_MAT_NEAR(ddth, Vector2d(0.5 * t1 + 0.2887, -1.1547), 2e-4);
        CHECK(std::abs((A * ddth + Ad * DTH85)(0)) < 1e-12);  // MR Eq. 8.95
        CHECK_MAT_NEAR(ddth, as_vec(c["ddtheta"]), 1e-10);   // Python reference
        CHECK_MAT_NEAR(lam, as_vec(c["lam"]), 1e-10);
    }
}

TEST_CASE("test_projector_and_inverse_dynamics") {
    const auto [M, h] = lab::twor_dynamics(TH85, DTH85);
    const auto [A, Ad] = lab::channel_constraint(TH85, DTH85);
    const MatrixXd P = lab::constraint_projector(M, A);
    Matrix2d P_ref;
    P_ref << 1.0, 0.0, 0.25, 0.0;  // MR p. 304
    CHECK_MAT_NEAR(P, P_ref, 1e-12);
    CHECK_MAT_NEAR(P * P, P, 1e-12);
    const Eigen::JacobiSVD<MatrixXd> svd(P);
    int rank = 0;
    for (Eigen::Index i = 0; i < svd.singularValues().size(); ++i) rank += svd.singularValues()(i) > 1e-9 ? 1 : 0;
    CHECK(rank == 1);
    CHECK_MAT_NEAR(P * A.transpose(), MatrixXd::Zero(2, 1), 1e-12);
    // inverse dynamics with thdd = (a, -1.155), lambda = f: tau1 = 2a - 0.578, tau2 = 0.5a - 0.866 f - 0.289
    for (const auto& c : cases()["projector_and_inverse_dynamics"]["cases"]) {
        const double a = c["a"].get<double>(), f = c["f"].get<double>();
        INFO("a = " << a << ", f = " << f);
        const Vector2d ddth(a, -2.0 / std::sqrt(3.0));
        const VectorXd tau = lab::constrained_inverse_dynamics(M, h, A, ddth, VectorXd::Constant(1, f));
        CHECK_MAT_NEAR(tau, Vector2d(2.0 * a - 0.5774, 0.5 * a - 0.8660 * f - 0.2887), 2e-4);
        const auto [ddth2, lam2] = lab::constrained_forward_dynamics(M, h, A, Ad, DTH85, tau);
        CHECK_MAT_NEAR(ddth2, ddth, 1e-10);
        CHECK_MAT_NEAR(lam2, VectorXd::Constant(1, f), 1e-10);
    }
}

TEST_CASE("test_simulation_conserves_energy_and_constraint") {
    // tau = 0, g = 0, workless constraint: kinetic energy is constant and the tip stays at x = 1.
    const lab::TorqueFunction zero = [](double, const Vector2d&, const Vector2d&) { return Vector2d::Zero().eval(); };
    const lab::ChannelSim out = lab::simulate_channel_2r(TH85, DTH85, zero, 2.0, 1e-3, 1.0, 10.0, 10.0);
    REQUIRE(out.theta.rows() == 2001);
    REQUIRE(out.theta.cols() == 2);
    REQUIRE(out.dtheta.rows() == 2001);
    REQUIRE(out.dtheta.cols() == 2);
    REQUIRE(out.lam.size() == 2001);
    REQUIRE(out.e.size() == 2001);
    double ke0 = 0.0, ke_dev = 0.0, x_dev = 0.0;
    for (Eigen::Index k = 0; k < 2001; ++k) {
        const Vector2d q = out.theta.row(k).transpose(), dq = out.dtheta.row(k).transpose();
        const double ke = 0.5 * dq.dot(lab::twor_dynamics(q, dq).first * dq);
        if (k == 0) ke0 = ke;
        ke_dev = std::max(ke_dev, std::abs(ke - ke0));
        x_dev = std::max(x_dev, std::abs(std::cos(q(0)) + std::cos(q(0) + q(1)) - 1.0));
    }
    CHECK(ke_dev < 1e-4);
    CHECK(out.e.cwiseAbs().maxCoeff() < 1e-6);
    CHECK(x_dev < 1e-6);
    CHECK(std::abs(out.lam(0) - (-1.0 / 6.0)) < 1e-3);  // MR: lambda = -0.167 at tau = 0
}

TEST_CASE("test_baumgarte_removes_drift") {
    // Start with a small constraint violation; Baumgarte (Featherstone Sec. 8.3) drives it to zero,
    // the unstabilised integration keeps it.
    const Vector2d th0 = TH85 + Vector2d(0.0, 0.01);
    const lab::TorqueFunction zero = [](double, const Vector2d&, const Vector2d&) { return Vector2d::Zero().eval(); };
    const lab::ChannelSim plain = lab::simulate_channel_2r(th0, DTH85, zero, 1.0, 1e-3);
    const lab::ChannelSim stab = lab::simulate_channel_2r(th0, DTH85, zero, 1.0, 1e-3, 1.0, 20.0, 20.0);
    REQUIRE(plain.e.size() > 0);
    REQUIRE(stab.e.size() > 0);
    CHECK(std::abs(plain.e(plain.e.size() - 1)) > 0.5 * std::abs(plain.e(0)));
    CHECK(std::abs(stab.e(stab.e.size() - 1)) < 1e-3 * std::abs(stab.e(0)));
}
