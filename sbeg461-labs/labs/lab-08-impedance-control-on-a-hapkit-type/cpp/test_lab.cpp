// Auto-checks for SBEG 461 Lab 8 (impedance control on a haptic device and a simulated rehabilitation robot) --
// C++ track. Mirrors test_lab.py test by test (same names, same order, same tolerances).
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*free_decay*,*identify*"      (run only some tests, like pytest -k)
// Ground truth: mrcpp (the C++ port of modern_robotics: mass_matrix, gravity_forces, jacobian_space,
// forward_dynamics), hand formulas (parallel-axis theorem, reflected rotor inertia), the exact analytic damped
// response, the apparent masses of the 461 project scenario sheet S1 (0.70, 0.81, 0.99 kg), and cpp_cases.json,
// which holds the numbers of the Python reference (matrix exponentials, the independent loop of test_lab.py, the
// random inputs of test_lab.py), so both languages are checked against the same values.
// The three data/infrastructure tests also call one function you write each (handle_mass, task_inertia,
// spring_damper_force), so with the untouched starter every test case fails.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <vector>

using Eigen::Matrix2d;
using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using sbeg::as;
using sbeg::as_list;
using sbeg::json;

namespace {
const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

const lab::HapkitModel& HK() {
    static const lab::HapkitModel h = lab::hapkit();
    return h;
}

const lab::RehabModel& RH() {
    static const lab::RehabModel r = lab::rehab_2r();
    return r;
}

const lab::Device& DEV() {
    static const lab::Device d = lab::load_device();
    return d;
}

/// Python np.allclose(got, expected, atol=atol) (rtol 1e-5, atol 1e-8 by default): same shape, finite, and entry by
/// entry |got - expected| <= atol + rtol * |expected| (the message comes from sbeg::compare_near).
template <typename A, typename B>
sbeg::NearResult allclose(const Eigen::MatrixBase<A>& got, const Eigen::MatrixBase<B>& expected, double atol = 1e-8,
                          double rtol = 1e-5) {
    MatrixXd g = got.template cast<double>(), e = expected.template cast<double>();
    if ((g.rows() == 1 || g.cols() == 1) && (e.rows() == 1 || e.cols() == 1) && g.size() == e.size()) {
        g = VectorXd(Eigen::Map<VectorXd>(g.data(), g.size()));  // a row and a column vector compare equal
        e = VectorXd(Eigen::Map<VectorXd>(e.data(), e.size()));
    }
    if (g.rows() != e.rows() || g.cols() != e.cols() || g.size() == 0) return sbeg::compare_near(g, e, atol);
    sbeg::NearResult r = sbeg::compare_near(g, e, atol + rtol * e.cwiseAbs().maxCoeff());
    r.ok = g.allFinite() && ((g - e).array().abs() <= atol + rtol * e.array().abs()).all();
    if (!r.ok) r.msg += "\n  (np.allclose rule: |got - expected| <= atol + rtol * |expected| for every entry)";
    return r;
}

/// Column k of an N x 2 matrix as a vector, max |.| of a vector.
double max_abs(const VectorXd& v) { return v.cwiseAbs().maxCoeff(); }

Vector2d v2(double a, double b) { return Vector2d(a, b); }
}  // namespace

/// CHECK_ALLCLOSE(got, expected[, atol]) = assert np.allclose(got, expected[, atol=...]) of test_lab.py.
#define CHECK_ALLCLOSE(...)                                                     \
    do {                                                                        \
        const ::sbeg::NearResult sbeg_ac_ = allclose(__VA_ARGS__);              \
        CHECK_MESSAGE(sbeg_ac_.ok, "np.allclose(" #__VA_ARGS__ ")\n  " << sbeg_ac_.msg); \
    } while (false)

// ============================================================================ data and infrastructure
TEST_CASE("test_rehab_dynamics_matches_modern_robotics") {
    const json& C = cases()["rehab_dynamics_matches_modern_robotics"];
    // the C++ model has the numbers of course_models.rehab_2r()
    CHECK_MAT_NEAR(RH().Slist, sbeg::as_mat(C["model"]["Slist"]), 1e-15);
    CHECK_MAT_NEAR(RH().M, as<Matrix4d>(C["model"]["M"]), 1e-15);
    const auto Ml = as_list<Matrix4d>(C["model"]["Mlist"]);
    const auto Gl = sbeg::as_mats(C["model"]["Glist"]);
    REQUIRE(Ml.size() == RH().Mlist.size());
    REQUIRE(Gl.size() == RH().Glist.size());
    for (std::size_t i = 0; i < Ml.size(); ++i) CHECK_MAT_NEAR(RH().Mlist[i], Ml[i], 1e-15);
    for (std::size_t i = 0; i < Gl.size(); ++i) CHECK_MAT_NEAR(RH().Glist[i], Gl[i], 1e-15);

    for (const auto& c : C["cases"]) {  // rng 1 of test_lab.py
        const Vector2d q = as<Vector2d>(c["q"]), dq = as<Vector2d>(c["dq"]);
        INFO("q = " << q.transpose() << ", dq = " << dq.transpose());
        const auto [Mq, cq] = lab::rehab_dynamics(q, dq);
        CHECK_ALLCLOSE(Mq, mr::mass_matrix(q, RH().Mlist, RH().Glist, RH().Slist), 1e-12);
        CHECK_ALLCLOSE(cq, mr::vel_quadratic_forces(q, dq, RH().Mlist, RH().Glist, RH().Slist), 1e-12);
        const Vector2d p_mr = mr::fkin_space(RH().M, RH().Slist, q).block<2, 1>(0, 3);
        CHECK_ALLCLOSE(lab::rehab_fk(q), p_mr);
        CHECK_ALLCLOSE(lab::rehab_fk(lab::rehab_ik(lab::rehab_fk(q))), lab::rehab_fk(q));
    }
    Matrix2d M_home;
    M_home << 0.3545, 0.0628, 0.0628, 0.06569;
    CHECK_ALLCLOSE(lab::rehab_dynamics(RH().q_home, Vector2d::Zero()).first, M_home, 1e-4);
    // C++ addition (calls your code): the task-space inertia at the ready pose
    CHECK_MAT_NEAR(lab::task_inertia(RH().q_home), as<Matrix2d>(C["task_inertia_home"]), 1e-10);
}

TEST_CASE("test_device_file_and_free_decay_data") {
    const lab::Device& dev = DEV();
    CHECK_APPROX(dev.T, 1.0 / HK().params.loop_rate_hz);
    CHECK(dev.force_max == HK().params.force_max);
    for (int K : {50, 100, 150}) {
        INFO("K = " << K);
        const auto [t, x] = lab::load_free_decay(K);
        REQUIRE(t.size() == 3001);
        REQUIRE(x.size() == 3001);
        CHECK_APPROX(t(1) - t(0), dev.T);
        VectorXd xq(x.size());
        for (Eigen::Index k = 0; k < x.size(); ++k) xq(k) = std::nearbyint(x(k) / dev.x_res) * dev.x_res;
        CHECK_ALLCLOSE(xq, x, 1e-9);  // quantised
        CHECK_SCALAR_NEAR(x(0), 0.012, dev.x_res);
    }
    // the C++ hapkit model has the numbers of course_models.hapkit()
    const json& H = cases()["device_file_and_free_decay_data"]["hapkit"];
    CHECK_MAT_NEAR(HK().Slist, sbeg::as_mat(H["Slist"]), 1e-15);
    CHECK_MAT_NEAR(HK().M, as<Matrix4d>(H["M"]), 1e-15);
    CHECK_MAT_NEAR(HK().g, as<Vector3d>(H["g"]), 1e-15);
    const auto Ml = as_list<Matrix4d>(H["Mlist"]);
    const auto Gl = sbeg::as_mats(H["Glist"]);
    REQUIRE(Ml.size() == HK().Mlist.size());
    REQUIRE(Gl.size() == HK().Glist.size());
    for (std::size_t i = 0; i < Ml.size(); ++i) CHECK_MAT_NEAR(HK().Mlist[i], Ml[i], 1e-15);
    for (std::size_t i = 0; i < Gl.size(); ++i) CHECK_MAT_NEAR(HK().Glist[i], Gl[i], 1e-15);
    // C++ addition (calls your code): the m_h of the device file is the handle mass of the model
    CHECK_SCALAR_NEAR(lab::handle_mass(HK()), dev.m_h, 1e-5);
}

TEST_CASE("test_rk4_and_jacobian_dot") {
    const Vector2d q(0.7, 1.4), dq(0.3, -0.5);
    const double h = 1e-6;
    const Matrix2d Jd = (lab::rehab_jacobian(q + h * dq) - lab::rehab_jacobian(q - h * dq)) / (2 * h);
    CHECK_ALLCLOSE(lab::rehab_jacobian_dot(q, dq), Jd, 1e-6);
    const VectorXd x = lab::rk4_step([](const VectorXd& s) -> VectorXd { return -s; }, VectorXd::Constant(1, 1.0), 0.1);
    REQUIRE(x.size() == 1);
    CHECK_SCALAR_NEAR(x(0), std::exp(-0.1), 1e-6);
    // C++ addition (calls your code): one RK4 step of the handle with a rendered spring-damper (no sampling),
    // compared with the matrix exponential (scipy.linalg.expm, in cpp_cases.json)
    const json& C = cases()["rk4_and_jacobian_dot"];
    const double K = C["K"].get<double>(), B = C["B"].get<double>(), m = DEV().m_h;
    const VectorXd s1 = lab::rk4_step(
        [K, B, m](const VectorXd& s) -> VectorXd {
            return Vector2d(s(1), lab::spring_damper_force(s(0), s(1), K, B) / m);
        },
        sbeg::as_vec(C["s0"]), C["h"].get<double>());
    CHECK_MAT_NEAR(s1, sbeg::as_vec(C["exact"]), 1e-9);
}

// ============================================================================ Part A -- device model
TEST_CASE("test_handle_mass") {
    const double m_h = lab::handle_mass(HK());
    const double I_pivot = 1.5e-4 + 0.10 * 0.010 * 0.010;               // parallel-axis theorem
    const double J_refl = 1.0e-6 * (0.073 / 0.0042) * (0.073 / 0.0042);  // apparent rotor inertia G^2 J (MR 8.9.2)
    const double expected = (I_pivot + J_refl) / (0.090 * 0.090);
    CHECK_SCALAR_NEAR(m_h, expected, 1e-9 * expected);
    CHECK_SCALAR_NEAR(m_h, DEV().m_h, 1e-5);
}

TEST_CASE("test_gravity_stiffness") {
    const double kg = lab::gravity_stiffness(HK());
    const double expected = 0.10 * 9.81 * 0.010 / (0.090 * 0.090);
    CHECK_SCALAR_NEAR(kg, expected, 1e-6 * expected);
    CHECK(kg > 0);  // sector below the pivot: restoring
}

TEST_CASE("test_handle_force_to_current") {
    const VectorXd F = Vector2d(2.0, -1.0);
    const VectorXd I = lab::handle_force_to_current(F, HK(), DEV().k_t);
    const VectorXd expected = F * 0.090 / (0.073 / 0.0042) / 0.010;
    CHECK_ALLCLOSE(I, expected);
    REQUIRE(I.size() == 2);
    CHECK(std::abs(I(0)) < DEV().i_max);
}

TEST_CASE("test_fit_free_decay_on_exact_signal") {
    const double wn = 31.0, ze = 0.04;
    const Eigen::Index n = 3001;  // np.arange(0, 3.0001, 0.001)
    VectorXd t(n), x(n);
    const double wd = wn * std::sqrt(1 - ze * ze);
    for (Eigen::Index k = 0; k < n; ++k) {
        t(k) = static_cast<double>(k) * 0.001;
        x(k) = 0.01 * std::exp(-ze * wn * t(k)) * (std::cos(wd * t(k)) + ze / std::sqrt(1 - ze * ze) * std::sin(wd * t(k)));
    }
    const auto [w_fit, z_fit] = lab::fit_free_decay(t, x);
    CHECK_SCALAR_NEAR(w_fit, wn, 2e-3 * wn);
    CHECK_SCALAR_NEAR(z_fit, ze, 1e-2 * ze);
}

TEST_CASE("test_fit_free_decay_on_logged_data") {
    const double m = lab::handle_mass(HK()), kg = lab::gravity_stiffness(HK());
    for (int K : {50, 100, 150}) {
        INFO("K = " << K);
        const auto [t, x] = lab::load_free_decay(K);
        const auto [wn, ze] = lab::fit_free_decay(t, x);
        const double w_exp = std::sqrt((K + kg) / m);
        CHECK_SCALAR_NEAR(wn, w_exp, 5e-3 * w_exp);
        const double b_app = 2 * ze * wn * m;
        CHECK_SCALAR_NEAR(b_app, DEV().b_h - K * DEV().T / 2, 0.005);
    }
}

TEST_CASE("test_identify_device") {
    const VectorXd Ks = Eigen::Vector3d(50, 100, 150);
    std::vector<lab::FreeDecayFit> fits;
    for (int K : {50, 100, 150}) {
        const auto [t, x] = lab::load_free_decay(K);
        fits.push_back(lab::fit_free_decay(t, x));
    }
    const lab::Identification ident = lab::identify_device(Ks, fits);
    const double m_h = lab::handle_mass(HK());
    CHECK_SCALAR_NEAR(ident.m_h, m_h, 0.01 * m_h);
    CHECK_SCALAR_NEAR(ident.b_h, DEV().b_h, 0.03 * DEV().b_h);
    CHECK_SCALAR_NEAR(ident.tau_eff, DEV().T / 2, 0.1 * DEV().T / 2);
    REQUIRE(ident.b_app.size() == 3);
    CHECK(ident.b_app(0) > ident.b_app(2));
}

// ============================================================================ Part B -- rendering
TEST_CASE("test_spring_damper_and_wall_force") {
    CHECK_APPROX(lab::spring_damper_force(0.01, 0.1, 100.0, 2.0), -1.2);
    CHECK_APPROX(lab::spring_damper_force(0.01, 0.1, 100.0, 2.0, 0.02, 0.1), 1.0);
    CHECK(lab::wall_force(-0.001, 0.5, 1000.0, 1.0, 0.0) == 0.0);           // outside
    CHECK_APPROX(lab::wall_force(0.002, 0.01, 1000.0, 1.0, 0.0), -2.01);    // pushing in
    CHECK(lab::wall_force(0.0005, -1.0, 1000.0, 1.0, 0.0) == 0.0);          // leaving fast: no pull
    CHECK_APPROX(lab::wall_force(-0.002, 0.0, 1000.0, 1.0, 0.0, -1), 2.0);
    CHECK(lab::wall_force(0.002, 0.0, 1000.0, 1.0, 0.0, -1) == 0.0);
}

TEST_CASE("test_simulate_device_free_motion_is_exact") {
    const json& C = cases()["simulate_device_free_motion_is_exact"];
    const lab::DeviceSim sim = lab::simulate_device(
        DEV(), [](double, double, double) { return 0.0; }, 0.5, C["x0"].get<double>(), C["v0"].get<double>(),
        lab::ForceLaw(), 0, false);
    REQUIRE(sim.t.size() == 501);
    REQUIRE(sim.x.size() == 501);
    REQUIRE(sim.v.size() == 501);
    for (std::size_t i = 0; i < C["k"].size(); ++i) {
        const auto k = C["k"][i].get<Eigen::Index>();
        const VectorXd s = sbeg::as_vec(C["state"][i]);  // expm(A t_k) [x0, v0]
        INFO("k = " << k);
        CHECK_SCALAR_NEAR(sim.x(k), s(0), 1e-9);
        CHECK_SCALAR_NEAR(sim.v(k), s(1), 1e-8);
    }
}

TEST_CASE("test_simulate_device_loop_details") {
    const lab::Device dev = DEV();
    std::vector<std::tuple<double, double, double>> seen;
    const lab::ForceLaw law = [&seen](double t, double x, double v) {
        seen.emplace_back(t, x, v);
        return t >= 0.0095 ? 5.0 : 0.1;  // 5 N > force_max: must saturate
    };
    const lab::DeviceSim sim = lab::simulate_device(dev, law, 0.02, 0.00123, 0.0, lab::ForceLaw(), 3);
    const Eigen::Index n = 21;
    REQUIRE(sim.x_meas.size() == n);
    REQUIRE(sim.v_est.size() == n);
    REQUIRE(sim.F_cmd.size() == n);
    REQUIRE(sim.F_applied.size() == n);
    REQUIRE(sim.saturated.size() == static_cast<std::size_t>(n));
    REQUIRE(seen.size() == static_cast<std::size_t>(n));
    const VectorXd& xm = sim.x_meas;
    VectorXd xq(n);
    for (Eigen::Index k = 0; k < n; ++k) xq(k) = std::nearbyint(xm(k) / dev.x_res) * dev.x_res;
    CHECK_ALLCLOSE(xq, xm);
    CHECK_APPROX(xm(0), std::nearbyint(0.00123 / dev.x_res) * dev.x_res);
    CHECK(sim.v_est(0) == 0.0);
    const VectorXd dxm = (xm.tail(n - 1) - xm.head(n - 1)) / dev.T;
    CHECK_ALLCLOSE(sim.v_est.tail(n - 1), dxm);
    VectorXd sx(n), sv(n);
    for (Eigen::Index k = 0; k < n; ++k) {
        sx(k) = std::get<1>(seen[static_cast<std::size_t>(k)]);
        sv(k) = std::get<2>(seen[static_cast<std::size_t>(k)]);
    }
    CHECK_ALLCLOSE(sx, xm);
    CHECK_ALLCLOSE(sv, sim.v_est);
    CHECK(max_abs(sim.F_cmd) <= dev.force_max + 1e-12);
    CHECK(sim.saturated[10]);
    CHECK_FALSE(sim.saturated[9]);
    CHECK_ALLCLOSE(sim.F_applied.head(3), VectorXd::Zero(3));
    CHECK_ALLCLOSE(sim.F_applied.tail(n - 3), sim.F_cmd.head(n - 3));
}

TEST_CASE("test_static_push_rendered_stiffness") {
    const double kg = DEV().k_g;
    for (const std::pair<double, double>& KF : {std::pair<double, double>{100.0, 0.49}, std::pair<double, double>{300.0, -0.98}}) {
        const double K = KF.first, F = KF.second;
        INFO("K = " << K << ", F = " << F);
        const lab::PushTest r = lab::static_push_test(DEV(), K, 1.0, F);
        CHECK_SCALAR_NEAR(r.x_ss, F / (K + kg), DEV().x_res);
        CHECK_SCALAR_NEAR(r.K_rendered, K + kg, 0.02 * (K + kg));
        CHECK(std::abs(r.error_pct) < 2.0);
    }
}

TEST_CASE("test_closed_loop_matrix_reproduces_the_loop") {
    const lab::Device& D = DEV();
    for (const auto& c : cases()["closed_loop_matrix_reproduces_the_loop"]) {
        const int d = c["d"].get<int>();
        const double K = c["K"].get<double>(), B = c["B"].get<double>();
        INFO("d = " << d << ", K = " << K << ", B = " << B);
        const MatrixXd A = lab::closed_loop_matrix(D.m_h, D.b_h, D.k_g, K, B, D.T, d);
        REQUIRE(A.rows() == 3 + d);
        REQUIRE(A.cols() == 3 + d);
        VectorXd z = VectorXd::Zero(3 + d);
        z(0) = z(2) = 0.003;
        VectorXd xs(60);
        for (Eigen::Index k = 0; k < 60; ++k) {
            xs(k) = z(0);
            z = (A * z).eval();
        }
        const VectorXd own = sbeg::as_vec(c["xs"]);  // the independent loop of test_lab.py (_own_loop)
        CHECK_ALLCLOSE(xs, own, 1e-12);
        const lab::DeviceSim sim = lab::simulate_device(
            D, [K, B](double, double x, double v) { return lab::spring_damper_force(x, v, K, B); }, 0.059, 0.003,
            0.0, lab::ForceLaw(), d, false);
        CHECK_ALLCLOSE(sim.x, xs, 1e-9);
    }
}

TEST_CASE("test_max_stable_K_and_first_order_estimate") {
    const lab::Device& D = DEV();
    const auto rho = [&D](double K, double B, int d) {
        return lab::spectral_radius(lab::closed_loop_matrix(D.m_h, D.b_h, D.k_g, K, B, D.T, d));
    };
    for (int d : {0, 1, 2, 5}) {
        for (double B : {0.0, 0.5, 1.0}) {
            INFO("d = " << d << ", B = " << B);
            const double Km = lab::max_stable_K(D.m_h, D.b_h, D.k_g, B, D.T, d);
            CHECK(rho(Km - 0.05, B, d) < 1);
            CHECK(1 < rho(Km + 0.05, B, d));
            const double est = lab::first_order_K_limit(D.b_h, B, D.T, d);
            CHECK_SCALAR_NEAR(est, Km, 0.05 * Km);
        }
    }
    CHECK_SCALAR_NEAR(lab::max_stable_K(D.m_h, D.b_h, D.k_g, 0.0, D.T, 0), 300.0, 0.01 * 300.0);  // ~ b_h / (T/2)
}

TEST_CASE("test_energy_and_simulated_stability_agree_with_eigenvalues") {
    const lab::Device& D = DEV();
    const double Km = lab::max_stable_K(D.m_h, D.b_h, D.k_g, 0.5, D.T, 2);
    double growth[2] = {0.0, 0.0}, ratio[2] = {0.0, 0.0};
    const double fs[2] = {0.8, 1.2};
    for (int i = 0; i < 2; ++i) {
        const double K = fs[i] * Km;
        INFO("K = " << K);
        const lab::DeviceSim sim = lab::simulate_device(
            D, [K](double, double x, double v) { return lab::spring_damper_force(x, v, K, 0.5); }, 2.0, 0.004, 0.0,
            lab::ForceLaw(), 2);
        const VectorXd E = lab::device_energy(D, sim, K);
        REQUIRE(E.size() == sim.t.size());
        REQUIRE(E.size() > 1);
        const double E0 = 0.5 * (K + D.k_g) * 0.004 * 0.004;
        CHECK_SCALAR_NEAR(E(0), E0, 1e-9 * E0);
        growth[i] = lab::oscillation_growth(sim);
        ratio[i] = E(E.size() - 1) / E(0);
    }
    CHECK(growth[0] < 0.5);
    CHECK(ratio[0] < 0.05);
    CHECK(growth[1] > 1.2);
    CHECK(ratio[1] > 1.0);
}

// ============================================================================ Part C -- rehab robot
TEST_CASE("test_task_inertia_matches_modern_robotics") {
    for (const auto& c : cases()["task_inertia_matches_modern_robotics"]) {
        const Vector2d q = as<Vector2d>(c["q"]);
        INFO("q = " << q.transpose());
        const MatrixXd Mq = mr::mass_matrix(q, RH().Mlist, RH().Glist, RH().Slist);
        const MatrixXd Js = mr::jacobian_space(RH().Slist, q);
        const Vector3d p = mr::fkin_space(RH().M, RH().Slist, q).block<3, 1>(0, 3);
        // linear velocity of the point p: v = v_s + w_s x p  (MR Sec. 5.1.1)
        Matrix2d Jv;
        for (int i = 0; i < 2; ++i) {
            const Vector3d w = Js.block<3, 1>(0, i), v = Js.block<3, 1>(3, i);
            Jv.col(i) = (v + w.cross(p)).head<2>();
        }
        const Matrix2d Lam = (Jv * Mq.partialPivLu().solve(Jv.transpose())).inverse();
        CHECK_ALLCLOSE(lab::task_inertia(q), Lam, 1e-10);
        CHECK_MAT_NEAR(Lam, as<Matrix2d>(c["Lambda"]), 1e-10);  // same as modern_robotics in Python
    }
}

TEST_CASE("test_apparent_mass_along_the_therapy_path") {
    const double expected_tangential[3] = {0.70, 0.81, 0.99};  // 461 project scenario sheet S1
    const Vector2d P[3] = {v2(0, 0.35), v2(0, 0.45), v2(0, 0.55)};
    for (int i = 0; i < 3; ++i) {
        INFO("P = " << P[i].transpose());
        const Vector2d q = lab::rehab_ik(P[i]);
        CHECK_SCALAR_NEAR(lab::apparent_mass(q, v2(0, 1)), expected_tangential[i], 0.006);
        const Vector2d ev = Eigen::SelfAdjointEigenSolver<Matrix2d>(lab::task_inertia(q)).eigenvalues();
        for (const Vector2d& u : {v2(1, 0), v2(1, 1), v2(0.3, -1)}) {
            const double mu = lab::apparent_mass(q, u);
            CHECK(ev(0) - 1e-9 <= mu);
            CHECK(mu <= ev(1) + 1e-9);
        }
    }
    const Vector2d qm = lab::rehab_ik(v2(0, 0.45));
    CHECK_APPROX(lab::apparent_mass(qm, v2(0, 2)), lab::apparent_mass(qm, v2(0, 1)));
}

TEST_CASE("test_impedance_control_torque_and_rendered_stiffness") {
    const Vector2d q(0.7, 1.7), dq(0.2, -0.4);
    const Vector2d pd = lab::rehab_fk(q) + v2(0.01, -0.02), vd(0.05, 0.0);
    Matrix2d K;
    K << 500.0, 50.0, 50.0, 300.0;
    const Matrix2d J = lab::rehab_jacobian(q);
    const Vector2d F = K * (pd - lab::rehab_fk(q)) + 15.0 * (vd - J * dq);
    const Vector2d expected = J.transpose() * F;
    const Matrix2d B15 = 15.0 * Matrix2d::Identity();  // Python passes the scalar 15.0
    CHECK_ALLCLOSE(lab::impedance_control_torque(q, dq, pd, vd, K, B15), expected);
    const Vector2d pm(0.0, 0.45);
    const lab::RehabSim sim = lab::simulate_rehab(
        [pm](double, const Vector2d& qq, const Vector2d& dqq, const Vector2d&) {
            return lab::impedance_control_torque(qq, dqq, pm, Vector2d::Zero(), 500.0, 15.0);
        },
        lab::rehab_ik(pm), 1.5, 0.001, [](double, const Vector2d&, const Vector2d&) { return v2(5.0, 0.0); });
    REQUIRE(sim.p.rows() == 1501);
    const Vector2d d = sim.p.row(sim.p.rows() - 1).transpose() - pm;
    CHECK_SCALAR_NEAR(5.0 / d(0), 500.0, 0.01 * 500.0);
    CHECK(std::abs(d(1)) < 1e-4);
}

TEST_CASE("test_admittance_control_gives_the_commanded_hand_acceleration") {
    const Vector2d p0(0.0, 0.45);
    const Matrix2d Md = v2(1.0, 2.0).asDiagonal(), Kd = v2(500.0, 300.0).asDiagonal();
    const Matrix2d Bd = 15.0 * Matrix2d::Identity();  // Python passes the scalar 15.0
    for (const auto& c : cases()["admittance_control_gives_the_commanded_hand_acceleration"]) {  // rng 3
        const Vector2d q = as<Vector2d>(c["q"]), dq = as<Vector2d>(c["dq"]), f = as<Vector2d>(c["f"]);
        INFO("q = " << q.transpose() << ", dq = " << dq.transpose() << ", f = " << f.transpose());
        const Vector2d tau = lab::admittance_control_torque(q, dq, f, p0, Md, Bd, Kd);
        const Matrix4d T = mr::fkin_space(RH().M, RH().Slist, q);
        const Vector3d f_b = T.topLeftCorner<3, 3>().transpose() * Vector3d(f(0), f(1), 0.0);  // hand force in {b}
        mr::Vector6d Ftip;
        Ftip << 0.0, 0.0, 0.0, -f_b;  // F_tip = -f (MR Eq. 11.48)
        const VectorXd qdd = mr::forward_dynamics(q, dq, tau, RH().g, Ftip, RH().Mlist, RH().Glist, RH().Slist);
        const Vector2d a = lab::rehab_jacobian(q) * qdd + lab::rehab_jacobian_dot(q, dq) * dq;
        const Vector2d p = lab::rehab_fk(q), v = lab::rehab_jacobian(q) * dq;
        const Vector2d expected = Md.partialPivLu().solve(f - 15.0 * v - Kd * (p - p0));
        CHECK_ALLCLOSE(a, expected);
    }
}

TEST_CASE("test_admittance_renders_the_one_dof_impedance_and_stiffness_control_does_not") {
    const Vector2d pm(0.0, 0.45);
    const Vector2d q0 = lab::rehab_ik(pm);
    const lab::HandForce fp = lab::pulse_force(v2(1, 0), 10.0, 0.1, 0.12);
    for (double Md : {2.0, 0.5}) {
        INFO("M_d = " << Md);
        const lab::RehabSim sim = lab::simulate_rehab(
            [pm, Md](double, const Vector2d& q, const Vector2d& dq, const Vector2d& f) {
                return lab::admittance_control_torque(q, dq, f, pm, Md, 15.0, 500.0);
            },
            q0, 0.6, 0.001, fp);
        REQUIRE(sim.p.rows() == sim.t.size());
        const VectorXd x1 = lab::one_dof_pulse_response(Md, 15.0, 500.0, 10.0, 0.1, 0.12, sim.t);
        const MatrixXd dx = sim.p.rowwise() - pm.transpose();
        CHECK_SCALAR_NEAR(dx.col(0).maxCoeff(), x1.maxCoeff(), 0.02 * x1.maxCoeff());
        CHECK(max_abs(dx.col(1)) < 1e-5);  // decoupled
    }
    const lab::RehabSim sim = lab::simulate_rehab(
        [pm](double, const Vector2d& q, const Vector2d& dq, const Vector2d&) {
            return lab::impedance_control_torque(q, dq, pm, Vector2d::Zero(), 500.0, 15.0);
        },
        q0, 0.6, 0.001, fp);
    const MatrixXd dx = sim.p.rowwise() - pm.transpose();
    CHECK(max_abs(dx.col(1)) > 0.3 * dx.col(0).maxCoeff());  // the robot's own inertia couples x and y
}
