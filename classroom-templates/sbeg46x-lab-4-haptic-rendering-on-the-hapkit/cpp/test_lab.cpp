// Auto-checks for SBEG 46X Lab 4 (haptic rendering: virtual wall, Z-width, virtual fixtures, needle insertion) --
// C++ track.  Mirrors test_lab.py test by test (22 test cases, same names and order).
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*colgate*,*required_damping*"      (run only some tests, like pytest -k)
// Ground truth: mrcpp (the C++ port of modern_robotics: mass_matrix, gravity_forces) for the device model,
// closed-form results (Colgate's condition b > K T/2 + |B|, the exact energy K v T d of a sampled wall pushed at
// constant speed, the analytic damped response of the device), independent code written in this file, hand
// values from the Week 4 workshop, and cpp_cases.json, which holds the numbers of the Python reference, so both
// languages are checked against the same values.
//
// C++ only: the three data/infrastructure tests (which pass with the untouched Python starter) also check one of
// your functions here (colgate_K_max, threshold_from_trials, wall_force), so every C++ test case fails until
// you have written something.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <limits>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using Eigen::Vector2d;
using Eigen::VectorXd;
using sbeg::as_vec;
using sbeg::json;

namespace {
constexpr double PI = 3.14159265358979323846;
constexpr double mm = 1e-3;

const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

const lab::Device& DEV() {
    static const lab::Device D = lab::load_device();
    return D;
}

const lab::TissueParams& TIS() {
    static const lab::TissueParams P = lab::load_tissue();
    return P;
}

double num(const json& j) { return j.get<double>(); }

/// np.allclose(got, expected, rtol, atol): |got - expected| <= atol + rtol |expected| entry by entry.
sbeg::NearResult allclose(const VectorXd& got, const VectorXd& exp, double atol = 1e-8, double rtol = 1e-5) {
    sbeg::NearResult r = sbeg::compare_near(got, exp, std::numeric_limits<double>::infinity());
    if (!r.ok || got.size() == 0) return r;  // size mismatch or NaN/inf (message already set)
    const VectorXd excess = (got - exp).cwiseAbs() - rtol * exp.cwiseAbs();
    Eigen::Index i = 0;
    const double worst = excess.maxCoeff(&i);
    r.ok = worst <= atol;
    std::ostringstream s;
    s << std::setprecision(12) << "worst entry " << i << ": got " << got(i) << ", expected " << exp(i) << " (atol "
      << atol << ", rtol " << rtol << ")";
    r.msg = s.str();
    return r;
}

/// np.interp(x, xp, fp) for increasing xp (constant beyond the ends).
double interp(double x, const VectorXd& xp, const VectorXd& fp) {
    const Eigen::Index n = xp.size();
    if (x <= xp(0)) return fp(0);
    if (x >= xp(n - 1)) return fp(n - 1);
    const Eigen::Index i = static_cast<Eigen::Index>(std::upper_bound(xp.data(), xp.data() + n, x) - xp.data()) - 1;
    return fp(i) + (x - xp(i)) * (fp(i + 1) - fp(i)) / (xp(i + 1) - xp(i));
}

/// np.linspace(start, stop, num): start + i * step, the last point exactly stop (as NumPy computes it).
VectorXd np_linspace(double start, double stop, int num) {
    VectorXd y(num);
    const double step = (stop - start) / (num - 1);
    for (int i = 0; i < num; ++i) y(i) = i * step + start;
    y(num - 1) = stop;
    return y;
}

/// Number of leading entries of an increasing grid that are < limit (= the length of z[z < limit]).
Eigen::Index count_below(const VectorXd& z, double limit) {
    Eigen::Index n = 0;
    while (n < z.size() && z(n) < limit) ++n;
    return n;
}
}  // namespace

#define CHECK_ALLCLOSE(got, expected, atol)                                                          \
    do {                                                                                             \
        const ::sbeg::NearResult lab_r_ = allclose((got), (expected), (atol));                        \
        CHECK_MESSAGE(lab_r_.ok, "allclose(" #got ", " #expected ")\n  " << lab_r_.msg);             \
    } while (false)

// ============================================================================ data and infrastructure
TEST_CASE("test_device_json_matches_the_course_hapkit_model") {
    // m_h and k_g in device.json are the handle mass and gravity stiffness of the shared hapkit model.
    const lab::HapkitModel hk = lab::hapkit();
    const lab::HapkitParams& p = hk.params;
    const VectorXd q0 = VectorXd::Zero(1);
    const double I_piv = mr::mass_matrix(q0, hk.Mlist, hk.Glist, hk.Slist)(0, 0);
    const double m_h = (I_piv + p.reflected_rotor_inertia) / (p.rh * p.rh);
    const double h = 1e-6;
    const double dg = (mr::gravity_forces(VectorXd::Constant(1, h), hk.g, hk.Mlist, hk.Glist, hk.Slist)(0) -
                       mr::gravity_forces(VectorXd::Constant(1, -h), hk.g, hk.Mlist, hk.Glist, hk.Slist)(0)) /
                      (2 * h);
    const json& c = cases()["device_json_matches_the_course_hapkit_model"];
    CHECK_SCALAR_NEAR(I_piv, num(c["I_piv"]), 1e-12);
    CHECK_SCALAR_NEAR(m_h, DEV().m_h, 1e-4);
    CHECK_SCALAR_NEAR(dg / (p.rh * p.rh), DEV().k_g, 1e-3);
    CHECK(DEV().force_max == p.force_max);
    CHECK(DEV().x_range == p.x_range);
    // Coulomb friction at the handle for 0.2 mN m at the motor: G tau_f / r_h
    CHECK_SCALAR_NEAR(p.ratio * 0.2e-3 / p.rh, DEV().f_c, 1e-4);
    // C++ only: the passive (Colgate) limit of this device, 2 b_h / T = 300 N/m.
    CHECK_APPROX(lab::colgate_K_max(DEV().b_h, 0.0, DEV().T), 300.0);
}

TEST_CASE("test_data_files_load") {
    const lab::Columns tr = lab::load_csv("zwidth_trials.csv");
    REQUIRE(tr.count("K") == 1);
    REQUIRE(tr.count("e") == 1);
    const VectorXd& K = tr.at("K");
    CHECK(K.size() == 15);
    bool increasing = true;
    for (Eigen::Index i = 1; i < K.size(); ++i) increasing = increasing && K(i) > K(i - 1);
    CHECK(increasing);
    const lab::Columns log = lab::load_csv("needle_log.csv");
    std::set<std::string> keys;
    for (const auto& kv : log) keys.insert(kv.first);
    CHECK(keys == std::set<std::string>{"t", "x_meas", "f_tissue"});
    REQUIRE(log.count("t") == 1);
    CHECK(log.at("t").size() > 3000);
    CHECK(log.at("t").size() == cases()["data_files_load"]["n_log"].get<int>());
    CHECK(TIS().x_skin == -0.020);
    CHECK(TIS().v_eps > 0.0);
    // C++ only: the logged restitution crosses 1 inside the tested stiffness range.
    const double k = lab::threshold_from_trials(K, tr.at("e"));
    CHECK(k > K(0));
    CHECK(k < K(K.size() - 1));
}

TEST_CASE("test_simulator_free_motion_matches_analytic_response") {
    // Provided simulator, no rendering: m x'' + b x' + k_g x = 0 from x0 has the exact damped solution.
    const double m = DEV().m_h, b = DEV().b_h, k = DEV().k_g;
    const lab::DeviceSim sim =
        lab::simulate_device(DEV(), [](double, double, double) { return 0.0; }, 0.5, 0.01, 0.0, lab::ForceLaw(),
                             false);
    const double wn = std::sqrt(k / m), ze = b / (2 * std::sqrt(k * m));
    const double wd = wn * std::sqrt(1 - ze * ze);
    VectorXd x(sim.t.size());
    for (Eigen::Index i = 0; i < x.size(); ++i) {
        const double t = sim.t(i);
        x(i) = 0.01 * std::exp(-ze * wn * t) * (std::cos(wd * t) + ze * wn / wd * std::sin(wd * t));
    }
    CHECK_MAT_NEAR(sim.x, x, 1e-9);
    CHECK_MAT_NEAR(sim.x, as_vec(cases()["simulator_free_motion_matches_analytic_response"]["x"]), 1e-12);
    // C++ only: rendering a wall far outside the motion (x_wall = 50 mm) must not change anything: wall_force = 0.
    const lab::DeviceSim sim_w = lab::simulate_device(
        DEV(), [](double, double xx, double vv) { return lab::wall_force(xx, vv, 300.0, 0.0, 0.05); }, 0.5, 0.01, 0.0,
        lab::ForceLaw(), false);
    CHECK_MAT_NEAR(sim_w.x, x, 1e-9);
}

// ============================================================================ Part A -- passivity limits
TEST_CASE("test_colgate_K_max") {
    CHECK_APPROX(lab::colgate_K_max(0.15, 0.0, 1e-3), 300.0);
    CHECK_APPROX(lab::colgate_K_max(0.15, 0.05, 1e-3), 200.0);
    CHECK_APPROX(lab::colgate_K_max(0.15, -0.05, 1e-3), 200.0);  // |B|: negative damping is no better
    CHECK_APPROX(lab::colgate_K_max(0.15, 0.0, 0.25e-3), 1200.0);
    CHECK(lab::colgate_K_max(0.15, 0.15, 1e-3) == 0.0);
    CHECK(lab::colgate_K_max(0.15, 1.0, 1e-3) == 0.0);  // B > b: no passive wall at all
    for (const auto& c : cases()["colgate_K_max"])
        CHECK_APPROX(lab::colgate_K_max(num(c["b"]), num(c["B"]), num(c["T"])), num(c["K_max"]));
}

TEST_CASE("test_required_damping_frequency_form") {
    const VectorXd th = np_linspace(1e-3, PI, 2001);
    for (const auto& c : cases()["required_damping_frequency_form"]["cases"]) {
        const double K = num(c["K"]), B = num(c["B"]), T = num(c["T"]);
        INFO("K = " << K << ", B = " << B << ", T = " << T);
        const VectorXd got = lab::required_damping(K, B, T, th);
        REQUIRE(got.size() == th.size());
        // independent: evaluate Colgate's expression with explicit complex numbers
        VectorXd ref(th.size());
        for (Eigen::Index i = 0; i < th.size(); ++i) {
            const std::complex<double> z(std::cos(th(i)), std::sin(th(i)));
            const std::complex<double> H = K + B * (1.0 - 1.0 / z) / T;
            ref(i) = (T / 2) * ((1.0 - 1.0 / z) * H).real() / (1.0 - std::cos(th(i)));
        }
        CHECK_ALLCLOSE(got, ref, 1e-12);
        const double target = K * T / 2 + std::abs(B);
        CHECK_SCALAR_NEAR(got.maxCoeff(), target, 1e-5 * target);
        const std::vector<int> idx = c["idx"].get<std::vector<int>>();
        VectorXd sub(static_cast<Eigen::Index>(idx.size()));
        for (std::size_t i = 0; i < idx.size(); ++i) sub(static_cast<Eigen::Index>(i)) = got(idx[i]);
        CHECK_MAT_NEAR(sub, as_vec(c["b_req"]), 1e-12);
    }
    // the damper only hurts at high frequency when B > 0: at theta = pi/2 the requirement is K T / 2
    CHECK_SCALAR_NEAR(lab::required_damping(300, 0.05, 1e-3, PI / 2), 0.15, 1e-12);
}

TEST_CASE("test_quantisation_K_max") {
    CHECK_APPROX(lab::quantisation_K_max(DEV().f_c, DEV().x_res), 3860.0);
    CHECK_APPROX(lab::quantisation_K_max(0.0386, 1e-4), 772.0);
}

TEST_CASE("test_sampled_wall_work_is_KvTd") {
    const double params[3][4] = {{500, 1e-3, 0.05, 5e-3}, {300, 1e-3, 0.02, 4e-3}, {1000, 0.5e-3, 0.1, 10e-3}};
    for (const auto& q : params) {
        const double K = q[0], T = q[1], v = q[2], d = q[3];
        INFO("K = " << K << ", T = " << T << ", v = " << v << ", d = " << d);
        const double W = lab::sampled_wall_work(K, T, v, d);
        CHECK_SCALAR_NEAR(W, K * v * T * d, 1e-9 * K * v * T * d);
    }
    // a faster loop creates proportionally less energy
    const double w1 = lab::sampled_wall_work(500, 1e-3, 0.05, 5e-3), w2 = lab::sampled_wall_work(500, 0.1e-3, 0.05, 5e-3);
    CHECK_SCALAR_NEAR(w2, w1 / 10, 1e-9 * std::abs(w1 / 10));
    for (const auto& c : cases()["sampled_wall_work_is_KvTd"])
        CHECK_APPROX(lab::sampled_wall_work(num(c["K"]), num(c["T"]), num(c["v"]), num(c["d"])), num(c["W"]));
}

TEST_CASE("test_energy_balance_gives_colgate_threshold") {
    // Created energy K v T d against the energy 2 b v d dissipated by the device damping in the same push.
    const double b = DEV().b_h, T = DEV().T, v = 0.05, d = 5e-3;
    const double Ks[3] = {250, 350, 500};
    const bool active[3] = {false, true, true};
    for (int i = 0; i < 3; ++i) {
        INFO("K = " << Ks[i]);
        const double net = lab::sampled_wall_work(Ks[i], T, v, d) - 2 * b * v * d;
        CHECK((net > 0) == active[i]);
    }
    CHECK_SCALAR_NEAR(lab::sampled_wall_work(2 * b / T, T, v, d), 2 * b * v * d, 1e-9 * 2 * b * v * d);
}

// ============================================================================ Part B -- stable stiffness limit
TEST_CASE("test_wall_force_is_unilateral") {
    CHECK(lab::wall_force(-1e-3, 0.1, 300, 0.5) == 0.0);
    CHECK_APPROX(lab::wall_force(2e-3, 0.0, 300, 0.0), -0.6);
    CHECK_APPROX(lab::wall_force(2e-3, -0.1, 300, 0.5), -0.55);
    CHECK(lab::wall_force(1e-4, -0.5, 300, 0.5) == 0.0);  // damper would pull the hand in: clipped
    CHECK_APPROX(lab::wall_force(-2e-3, 0.0, 300, 0.0, 0.0, -1), 0.6);
}

TEST_CASE("test_bounce_restitution_crosses_one_near_2b_over_T") {
    CHECK(lab::bounce_restitution(DEV(), 200.0) < 0.99);
    CHECK(lab::bounce_restitution(DEV(), 450.0) > 1.0);
    CHECK(lab::bounce_restitution(DEV(), 450.0, 0.1) < 1.0);  // virtual damping helps in a slow bounce
    for (const auto& c : cases()["bounce_restitution_crosses_one_near_2b_over_T"]) {
        INFO("K = " << num(c["K"]) << ", B = " << num(c["B"]));
        CHECK_SCALAR_NEAR(lab::bounce_restitution(DEV(), num(c["K"]), num(c["B"])), num(c["e"]), 1e-9);
    }
}

TEST_CASE("test_measure_K_limit") {
    const double k0 = lab::measure_K_limit(DEV(), 0.0);
    CHECK(k0 >= 300);  // Colgate: 300 N/m (single-bounce test +4 %)
    CHECK(k0 <= 325);
    const double k1 = lab::measure_K_limit(DEV(), 0.05);
    CHECK(k1 >= 390);  // low-frequency test: ~2 (b + B)/T,
    CHECK(k1 <= 450);
    CHECK(k1 > 2 * lab::colgate_K_max(DEV().b_h, 0.05, DEV().T));  // far above the passive limit 200 N/m
}

TEST_CASE("test_threshold_from_trials") {
    const lab::Columns tr = lab::load_csv("zwidth_trials.csv");
    const double k = lab::threshold_from_trials(tr.at("K"), tr.at("e"));
    CHECK_SCALAR_NEAR(k, 349.1, 0.5);
    CHECK_APPROX(k, num(cases()["threshold_from_trials"]["log"]));
    CHECK_APPROX(lab::threshold_from_trials(Eigen::Vector3d(100, 200, 300), Eigen::Vector3d(0.9, 0.95, 1.05)), 250.0);
    CHECK_THROWS_AS(lab::threshold_from_trials(Vector2d(100, 200), Vector2d(0.9, 0.95)), std::invalid_argument);
    CHECK_THROWS_AS(lab::threshold_from_trials(Vector2d(100, 200), Vector2d(1.01, 1.05)), std::invalid_argument);
}

// ============================================================================ Part C -- virtual fixtures
TEST_CASE("test_penalty_slab_pops_through") {
    const double a = 0.0, w = 6 * mm, K = 250.0;
    CHECK(lab::penalty_slab_force(-1 * mm, a, w, K) == 0.0);
    CHECK_APPROX(lab::penalty_slab_force(1 * mm, a, w, K), -0.25);
    CHECK_APPROX(lab::penalty_slab_force(4 * mm, a, w, K), +0.5);  // past the mid-plane: pulls through
}

TEST_CASE("test_proxy_slab_never_pops_through") {
    const double a = 0.0, w = 6 * mm, K = 250.0;
    lab::ProxySlab pr(a, w, K, -5 * mm);
    VectorXd xs(282);
    xs << np_linspace(-5, 9, 141), np_linspace(9, -5, 141);  // through the slab and back
    xs *= mm;
    VectorXd F(xs.size());
    for (Eigen::Index i = 0; i < xs.size(); ++i) F(i) = pr.force(xs(i));
    bool pushes_back = true;  // always pushes back to the entry side
    for (Eigen::Index i = 0; i < xs.size(); ++i)
        if (a < xs(i) && xs(i) < a + w) pushes_back = pushes_back && F(i) <= 1e-12;
    CHECK(pushes_back);
    Eigen::Index i4 = 0, i9 = 0;
    (xs.array() - 4 * mm).abs().minCoeff(&i4);
    (xs.array() - 9 * mm).abs().minCoeff(&i9);
    CHECK_APPROX(F(i4), -K * xs(i4));
    CHECK_APPROX(F(i9), -K * 9 * mm);  // beyond the slab: still held at a
    CHECK(pr.force(-3 * mm) == 0.0);   // free again once back outside
    lab::ProxySlab far_side(a, w, K, 10 * mm);
    CHECK_APPROX(far_side.force(5 * mm), K * 1 * mm);  // entered from the far side
    const json& c = cases()["proxy_slab_never_pops_through"];
    CHECK_MAT_NEAR(xs, as_vec(c["xs"]), 1e-15);
    CHECK_MAT_NEAR(F, as_vec(c["F"]), 1e-12);
    CHECK_THROWS_AS(lab::ProxySlab(a, w, K, 2 * mm), std::invalid_argument);  // given constructor
}

TEST_CASE("test_forbidden_disc_force_workshop_values") {
    const auto r = lab::forbidden_disc_force(Vector2d(3 * mm, 2 * mm), Vector2d(0.0, 0.0), 5 * mm, 500.0);
    REQUIRE(r.second.size() == 2);
    CHECK_ALLCLOSE(r.second / mm, Vector2d(4.1603, 2.7735), 1e-4);
    CHECK_ALLCLOSE(r.first, Vector2d(0.5801, 0.3868), 1e-4);
    const auto r0 = lab::forbidden_disc_force(Vector2d(6 * mm, 0.0), Vector2d(0.0, 0.0), 5 * mm, 500.0);
    CHECK_ALLCLOSE(r0.first, Vector2d(0.0, 0.0), 1e-8);
    CHECK_ALLCLOSE(r0.second, Vector2d(6 * mm, 0.0), 1e-8);
    const auto r1 = lab::forbidden_disc_force(Vector2d(11 * mm, 13 * mm), Vector2d(10 * mm, 10 * mm), 5 * mm, 200.0);
    const double rr = std::hypot(1.0, 3.0) * mm;
    CHECK_APPROX(r1.first.norm(), 200 * (5 * mm - rr));
    for (const auto& c : cases()["forbidden_disc_force_workshop_values"]) {
        const auto g = lab::forbidden_disc_force(as_vec(c["p"]), as_vec(c["c"]), num(c["R"]), num(c["K"]));
        INFO("p = " << as_vec(c["p"]).transpose() << ", c = " << as_vec(c["c"]).transpose());
        CHECK_MAT_NEAR(g.first, as_vec(c["F"]), 1e-12);
        CHECK_MAT_NEAR(g.second, as_vec(c["q"]), 1e-12);
    }
}

TEST_CASE("test_guidance_velocity") {
    const Vector2d f(1.0, 0.5), d(0.6, 0.8);
    CHECK_ALLCLOSE(lab::guidance_velocity(f, d, 0.01, 0.2), Vector2d(0.0068, 0.0074), 1e-12);
    CHECK_ALLCLOSE(lab::guidance_velocity(f, d, 0.01, 1.0), Vector2d(0.01, 0.005), 1e-8);
    const VectorXd v0 = lab::guidance_velocity(f, Vector2d(3.0, 4.0), 0.01, 0.0);  // d need not be unit length
    CHECK_ALLCLOSE(v0, Vector2d(0.006, 0.008), 1e-8);
}

// ============================================================================ Part D -- tissue and needle models
TEST_CASE("test_kelvin_voigt_and_hunt_crossley") {
    CHECK_APPROX(lab::kelvin_voigt_force(5 * mm, 0.02, 200, 2.0), 1.04);
    CHECK_SCALAR_NEAR(lab::kelvin_voigt_force(1e-9, 0.02, 200, 2.0), 0.04, 1e-6);  // jump at touch
    CHECK_APPROX(lab::kelvin_voigt_force(0.1 * mm, -0.05, 200, 2.0), -0.08);     // pulls: non-physical
    CHECK(lab::kelvin_voigt_force(-1 * mm, 0.02, 200, 2.0) == 0.0);
    const double k = 3000.0, lam = 4500.0, n = 1.5, d = 5 * mm;
    CHECK_APPROX(lab::hunt_crossley_force(d, 0.02, k, lam, n), std::pow(d, n) * (k + lam * 0.02));
    CHECK_SCALAR_NEAR(lab::hunt_crossley_force(1e-12, 0.02, k, lam, n), 0.0, 1e-12);
    CHECK(lab::hunt_crossley_force(0.1 * mm, -0.05, k, lam, n) > 0);
    CHECK(lab::hunt_crossley_force(0.1 * mm, -1.0, k, lam, n) == 0.0);  // clipped, never pulls
}

TEST_CASE("test_puncture_depth") {
    const lab::TissueParams::Skin& s = TIS().skin;
    const double dp = lab::puncture_depth(s.a1, s.a2, s.f_p);
    CHECK_SCALAR_NEAR(dp, 4.3007e-3, 1e-7);
    CHECK_APPROX(s.a1 * dp + s.a2 * dp * dp, s.f_p);
    CHECK_APPROX(lab::puncture_depth(150.0, 0.0, 0.45), 3e-3);
    CHECK_APPROX(dp, num(cases()["puncture_depth"]["dp"]));
}

TEST_CASE("test_needle_model_values") {
    lab::NeedleTissue m(TIS());
    CHECK(m.force(-1 * mm, 0.01) == 0.0);
    CHECK_APPROX(m.force(2 * mm, 0.01), 0.28);
    CHECK_APPROX(m.force(1 * mm, -0.01), 0.12);                  // elastic before puncture
    CHECK_APPROX(m.force(4.31 * mm, 0.01), 12 * 4.31e-3 + 0.2);  // skin punctured
    REQUIRE(m.events.size() >= 1);
    CHECK(m.events[0].first == "skin");
    CHECK_APPROX(m.force(10 * mm, 0.01), 0.32);
    CHECK_SCALAR_NEAR(m.force(10 * mm, 0.0), 0.0, 1e-12);  // at rest: no friction, no cutting
    CHECK_APPROX(m.force(10 * mm, -0.01), -0.12);           // withdrawal: friction pulls back in
    CHECK_APPROX(m.force(26 * mm, 0.01), 0.65);             // tissue + pleural membrane
    CHECK_APPROX(m.force(28.01 * mm, 0.01), 0.30);          // pleura punctured: friction only
    REQUIRE(m.events.size() == 2);
    CHECK(m.events[0].first == "skin");
    CHECK(m.events[1].first == "pleura");
    CHECK_APPROX(m.events[1].second, 28.01 * mm);
    CHECK_APPROX(m.force(20 * mm, -0.01), -0.24);
    // the same sequence of calls on a fresh model, against the Python reference
    lab::NeedleTissue m2(TIS());
    for (const auto& c : cases()["needle_model_values"]["calls"]) {
        INFO("z = " << num(c["z"]) << ", v = " << num(c["v"]));
        CHECK_SCALAR_NEAR(m2.force(num(c["z"]), num(c["v"])), num(c["f"]), 1e-12);
    }
}

TEST_CASE("test_profile_slope_is_renderable") {
    const VectorXd z = np_linspace(0.0, 35 * mm, 3501);
    const VectorXd f = lab::needle_quasistatic_profile(TIS(), z);
    CHECK_APPROX(f(200), 0.28);
    const Eigen::Index n_skin = count_below(z, 4.3 * mm);  // z[z < 4.3 mm]: the grid is increasing
    const json& c = cases()["profile_slope_is_renderable"];
    CHECK(n_skin == c["n_skin"].get<int>());
    const double k_skin = lab::max_profile_slope(z.head(n_skin), f.head(n_skin));
    CHECK_SCALAR_NEAR(k_skin, 271.8, 0.5);
    CHECK(k_skin < lab::colgate_K_max(DEV().b_h, 0.0, DEV().T));
    const std::vector<int> idx = c["idx"].get<std::vector<int>>();
    VectorXd sub(static_cast<Eigen::Index>(idx.size()));
    for (std::size_t i = 0; i < idx.size(); ++i) sub(static_cast<Eigen::Index>(i)) = f(idx[i]);
    CHECK_MAT_NEAR(sub, as_vec(c["f"]), 1e-12);
}

TEST_CASE("test_detect_punctures_in_log") {
    const lab::Columns log = lab::load_csv("needle_log.csv");
    const VectorXd z = log.at("x_meas").array() - TIS().x_skin;
    const VectorXd& F = log.at("f_tissue");
    const std::vector<int> idx = lab::detect_punctures(z, F, 0.3);
    REQUIRE(idx.size() == 2);
    REQUIRE(idx[0] >= 0);
    REQUIRE(idx[1] >= 0);
    REQUIRE(idx[0] < z.size());
    REQUIRE(idx[1] < z.size());
    CHECK_ALLCLOSE(Vector2d(z(idx[0]), z(idx[1])), Vector2d(4.3 * mm, 28.0 * mm), 0.1 * mm);
    CHECK_ALLCLOSE(Vector2d(F(idx[0]), F(idx[1])), Vector2d(0.80, 0.95), 0.01);
    CHECK(idx == cases()["detect_punctures_in_log"]["idx"].get<std::vector<int>>());
    CHECK(lab::detect_punctures(z, F, 0.8).empty());
}

TEST_CASE("test_render_needle_on_device") {
    const lab::NeedleSim sim = lab::render_needle(DEV(), TIS(), TIS().x_skin, 0.01, 4.5, -0.025);
    REQUIRE(sim.events.size() == 2);
    CHECK(sim.events[0].first == "skin");
    CHECK(sim.events[1].first == "pleura");
    CHECK(std::none_of(sim.saturated.begin(), sim.saturated.end(), [](bool s) { return s; }));
    CHECK(sim.x.cwiseAbs().maxCoeff() < DEV().x_range);
    const VectorXd zq = np_linspace(0.0, 35 * mm, 3501);
    const VectorXd prof = lab::needle_quasistatic_profile(TIS(), zq);
    double sum = 0.0;
    int cnt = 0;
    for (Eigen::Index i = 0; i < sim.z.size(); ++i) {
        if (sim.z(i) > 7 * mm && sim.z(i) < 24 * mm) {
            const double e = sim.f_tissue(i) - interp(sim.z(i), zq, prof);
            sum += e * e;
            ++cnt;
        }
    }
    REQUIRE(cnt > 0);
    CHECK(std::sqrt(sum / cnt) < 0.03);
    // the measured puncture depths of the Python reference run (4.32 mm and 28.02 mm)
    const json& ev = cases()["render_needle_on_device"]["events"];
    CHECK_SCALAR_NEAR(sim.events[0].second, num(ev[0][1]), 0.1 * mm);
    CHECK_SCALAR_NEAR(sim.events[1].second, num(ev[1][1]), 0.1 * mm);
}
