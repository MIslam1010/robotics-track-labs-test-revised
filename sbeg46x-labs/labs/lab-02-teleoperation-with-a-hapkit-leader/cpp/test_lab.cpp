// Auto-checks for SBEG 46X Lab 2 (Week 2): teleoperation with a Hapkit leader -- C++ track. Mirrors test_lab.py
// test by test (the two parametrized tests C1 and D2 loop over "pp" and "pf" inside one test case).
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*A1*,*A2*"      (run only some tests, like pytest -k)
// Ground truth: closed forms (Butterworth magnitude, series stiffness, ideal hybrid matrix), the scipy.signal
// results (lfilter, butter) and the numbers of the Python reference stored in cpp_cases.json, direct solution of
// the leader/follower equations, independently built state matrices, and mrcpp (jacobian_body) for the EOD arm.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Eigen::Matrix2cd;
using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using sbeg::as_vec;
using sbeg::json;
using cd = std::complex<double>;

namespace {
constexpr double PI = 3.14159265358979323846;
constexpr double DT = 0.001;

const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

const lab::Params& params() {  // the module-scope fixture p of test_lab.py (read-only; copy to change)
    static const lab::Params P = lab::load_params();
    return P;
}

const lab::LeaderLog& leader_log() {  // the fixture log
    static const lab::LeaderLog L = lab::load_log();
    return L;
}

const lab::Environment& wall(const lab::Params& p, const std::string& name = "wall") {
    return p.environments.at(name);
}

/// np.arange(0, ..., DT) with n samples (n stored in cpp_cases.json): t_k = k DT.
VectorXd time_grid(int n) {
    VectorXd t(n);
    for (int k = 0; k < n; ++k) t(k) = k * DT;
    return t;
}

int n_of(const char* key) { return cases()[key]["n"].get<int>(); }

double ptp(const VectorXd& v) { return v.maxCoeff() - v.minCoeff(); }

/// np.allclose(a, b, rtol, atol): |a - b| <= atol + rtol |b| entry by entry (same length required).
bool allclose(const VectorXd& a, const VectorXd& b, double rtol = 1e-5, double atol = 1e-8) {
    if (a.size() != b.size() || !a.allFinite()) return false;
    for (Eigen::Index i = 0; i < a.size(); ++i)
        if (std::abs(a(i) - b(i)) > atol + rtol * std::abs(b(i))) return false;
    return true;
}

/// pytest.approx for complex values: |got - expected| <= max(rel |expected|, abs).
void check_capprox(cd got, cd expected, double rel = 1e-6, double abs_tol = 1e-12) {
    const double tol = std::max(rel * std::abs(expected), abs_tol);
    const double err = std::abs(got - expected);
    CHECK_MESSAGE((std::isfinite(err) && err <= tol), "got " << got << ", expected " << expected
                                                             << " (|difference| = " << err << ", tolerance " << tol
                                                             << ")");
}

cd as_cd(const json& j) { return {j.at(0).get<double>(), j.at(1).get<double>()}; }

Matrix2cd as_m2cd(const json& j) {
    const MatrixXd re = sbeg::as_mat(j.at("re")), im = sbeg::as_mat(j.at("im"));
    REQUIRE((re.rows() == 2 && re.cols() == 2 && im.rows() == 2 && im.cols() == 2));
    Matrix2cd H;
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 2; ++c) H(r, c) = cd(re(r, c), im(r, c));
    return H;
}

VectorXd vec(std::initializer_list<double> v) {
    VectorXd x(static_cast<Eigen::Index>(v.size()));
    Eigen::Index i = 0;
    for (double e : v) x(i++) = e;
    return x;
}

/// Solve the leader and follower equations directly for (f_h, v_e) given (v_h, f_e).
std::pair<cd, cd> solve_two_port(const std::string& arch, cd s, const lab::Params& p, cd v_h, cd f_e) {
    const cd Zm = p.m_m * s + p.b_m, Zs = p.m_s * s + p.b_s;
    const cd C = p.bc + p.kc / s;
    const double al = p.alpha, be = p.beta;
    const cd v_e = (al * C * v_h - f_e) / (Zs + C);  // follower: Zs v_e = C(al v_h - v_e) - f_e
    const cd a_s = C * (al * v_h - v_e);
    const cd a_m = arch == "pp" ? -be * a_s : -be * f_e;
    const cd f_h = Zm * v_h - a_m;  // leader: Zm v_h = f_h + a_m
    return {f_h, v_e};
}

/// The state matrix built independently (test_lab.py: _A_ref).
Matrix4d A_ref(const std::string& arch, const lab::Params& p, double ke, double be) {
    const double al = p.alpha, bt = p.beta, kc = p.kc, bc = p.bc;
    const double mL = p.m_m + p.m_h, ms = p.m_s;
    const double kh = p.k_h, bh = p.b_h + p.b_m;
    Eigen::RowVector4d r1;
    if (arch == "pp")
        r1 << -(kh + bt * kc * al), -(bh + bt * bc * al), bt * kc, bt * bc;
    else
        r1 << -kh, -bh, -bt * ke, -bt * be;
    Matrix4d A;
    A.row(0) << 0, 1, 0, 0;
    A.row(1) = r1 / mL;
    A.row(2) << 0, 0, 0, 1;
    A.row(3) << kc * al / ms, bc * al / ms, -(kc + ke) / ms, -(bc + p.b_s + be) / ms;
    return A;
}

double max_real_eig(const Matrix4d& A) {
    const Eigen::EigenSolver<Matrix4d> es(A, false);
    return es.eigenvalues().real().maxCoeff();
}
}  // namespace

// ------------------------------------------------------------------ Part A: mapping, clutch, device safety
TEST_CASE("test_A1_effective_mass") {
    const lab::HapkitModel h = lab::hapkit();
    const lab::HapkitParams& pr = h.params;
    const double expected =
        (h.I_com(2, 2) + h.mass * 0.010 * 0.010 + 1.0e-6 * (pr.rs / pr.rp) * (pr.rs / pr.rp)) / (pr.rh * pr.rh);
    const double m = lab::hapkit_effective_mass(h);
    CHECK_SCALAR_NEAR(m, expected, 1e-9 * expected);
    CHECK_SCALAR_NEAR(m, 0.05705, 5e-5);
    CHECK_SCALAR_NEAR(m, cases()["A1_effective_mass"]["m_eff"].get<double>(), 1e-12);
}

TEST_CASE("test_A2_A3_force_saturation_and_watchdog") {
    const VectorXd f = vec({-5.0, -2.0, -0.3, 0.0, 1.99, 2.5});
    CHECK(allclose(lab::saturate_force(f, 2.0), vec({-2, -2, -0.3, 0, 1.99, 2})));
    CHECK_APPROX(lab::saturate_force(3.0, 2.0), 2.0);
    const std::vector<bool> ok = lab::watchdog_ok(vec({-0.041, -0.040, 0.0, 0.039, 0.0401}), 0.040);
    CHECK(ok == std::vector<bool>{false, true, true, true, false});
}

TEST_CASE("test_A4_clutch_mapping_small_example") {
    //                      engaged ...      clutch (leader returns) ...  engaged again
    const VectorXd x_m = vec({0.00, 0.01, 0.02, 0.02, 0.00, -0.01, -0.01, 0.00, 0.01});
    const std::vector<bool> cl = {false, false, false, true, true, true, false, false, false};
    const VectorXd x_sd = lab::map_with_clutch(x_m, cl, 0.5, 0.1);
    const VectorXd expected = vec({0.100, 0.105, 0.110, 0.110, 0.110, 0.110, 0.110, 0.115, 0.120});
    CHECK(allclose(x_sd, expected, 1e-5, 1e-12));
    CHECK_MAT_NEAR(x_sd, expected, 1e-12 + 1e-5 * 0.12);  // same check, with the numbers printed on failure
}

TEST_CASE("test_A4_clutch_mapping_log_no_jump") {
    const lab::LeaderLog& log = leader_log();
    const double a = 0.25;
    const VectorXd x_sd = lab::map_with_clutch(log.x_m, log.clutch, a);
    const Eigen::Index N = log.x_m.size();
    REQUIRE(x_sd.size() == N);
    bool no_jump = true, holds = true;
    double sum_eng = 0.0;
    for (Eigen::Index k = 0; k + 1 < N; ++k) {
        const std::size_t k1 = static_cast<std::size_t>(k + 1);
        const double dx_m = log.x_m(k + 1) - log.x_m(k), dx_s = x_sd(k + 1) - x_sd(k);
        if (!(std::abs(dx_s) <= a * std::abs(dx_m) + 1e-12)) no_jump = false;  // no jump at a clutch release
        if (log.clutch[k1] && dx_s != 0.0) holds = false;                      // follower holds while clutched
        if (!log.clutch[k1] && !log.clutch[k1 - 1]) sum_eng += dx_m;
    }
    CHECK_MESSAGE(no_jump, "the follower set-point jumps (|dx_sd| > alpha |dx_m|) somewhere");
    CHECK_MESSAGE(holds, "the follower set-point moves while the clutch is pressed");
    // independent reference: sum of the scaled leader increments over the engaged steps
    CHECK_SCALAR_NEAR(x_sd(N - 1), a * sum_eng, 1e-12);
    CHECK_SCALAR_NEAR(x_sd(N - 1), 0.0378, 5e-4);
    CHECK_SCALAR_NEAR(x_sd(N - 1), cases()["A4_clutch_mapping_log_no_jump"]["x_sd_final"].get<double>(), 1e-12);
}

TEST_CASE("test_A5_rate_limit") {
    VectorXd x = VectorXd::Constant(205, 0.01);
    x.head(5).setZero();
    const VectorXd y = lab::rate_limit(x, 0.1, DT);  // 0.1 mm per sample
    REQUIRE(y.size() == x.size());
    double max_step = 0.0;
    for (Eigen::Index k = 0; k + 1 < y.size(); ++k) max_step = std::max(max_step, std::abs(y(k + 1) - y(k)));
    CHECK(max_step <= 0.1 * DT + 1e-15);
    CHECK(y(4) == 0.0);
    CHECK_APPROX(y(5), 1e-4);
    CHECK_APPROX(y(104), 0.01);
    CHECK(allclose(lab::rate_limit(x, 100.0, DT), x));
}

TEST_CASE("test_A6_clutch_strokes") {
    CHECK(lab::clutch_strokes(0.040, 0.25, 0.060) == 3);
    CHECK(lab::clutch_strokes(0.030, 0.25, 0.060) == 2);  // exact multiple: no extra stroke
    CHECK(lab::clutch_strokes(0.60, 4.0, 0.070) == 3);
}

// ------------------------------------------------------------------ Part B: tremor filtering
TEST_CASE("test_B1_lowpass1_matches_scipy") {
    const json& c = cases()["B1_lowpass1_matches_scipy"];  // x drawn with np.random.default_rng(0); ref = lfilter
    const VectorXd x = as_vec(c["x"]), ref = as_vec(c["ref"]);
    const VectorXd y = lab::lowpass1(x, c["fc"].get<double>(), DT);
    CHECK(allclose(y, ref, 1e-5, 1e-12));
}

TEST_CASE("test_B2_butter2_matches_scipy") {
    for (const auto& c : cases()["B2_butter2_matches_scipy"]) {
        const double fc = c["fc"].get<double>();
        INFO("fc = " << fc);
        const auto ba = lab::butter2(fc, DT);
        CHECK(allclose(ba.first, as_vec(c["b"]), 1e-9, 1e-15));
        CHECK(allclose(ba.second, as_vec(c["a"]), 1e-9, 1e-8));
    }
}

TEST_CASE("test_B3_iir_filter_matches_scipy") {
    const json& c = cases()["B3_iir_filter_matches_scipy"];  // x: rng 1; b, a = butter(2, 3.0); ref = lfilter
    const VectorXd y = lab::iir_filter(as_vec(c["b"]), as_vec(c["a"]), as_vec(c["x"]));
    CHECK(allclose(y, as_vec(c["ref"]), 1e-5, 1e-12));
}

TEST_CASE("test_B4_band_rms_of_a_sine_on_a_ramp") {
    const json& c = cases()["B4_band_rms_of_a_sine_on_a_ramp"];
    const VectorXd t = time_grid(n_of("B4_band_rms_of_a_sine_on_a_ramp"));
    const VectorXd x = (0.02 * t.array() + 0.0003 * (2 * PI * 9 * t.array()).sin()).matrix();  // ramp + 0.3 mm tremor
    const double r = lab::band_rms(x, DT, 8, 12);
    CHECK_SCALAR_NEAR(r, 0.0003 / std::sqrt(2.0), 0.02 * 0.0003 / std::sqrt(2.0));
    CHECK_SCALAR_NEAR(r, c["rms_8_12"].get<double>(), 1e-12);
    CHECK(lab::band_rms(x, DT, 20, 40) < 2e-6);
}

TEST_CASE("test_B4_tremor_attenuation_on_log") {
    const VectorXd& x = leader_log().x_m;
    const double r0 = lab::band_rms(x, DT, 8, 12);
    const double r1 = lab::band_rms(lab::lowpass1(x, 2.0, DT), DT, 8, 12) / r0;
    const auto ba = lab::butter2(2.0, DT);
    const double r2 = lab::band_rms(lab::iir_filter(ba.first, ba.second, x), DT, 8, 12) / r0;
    CHECK_SCALAR_NEAR(r0, 0.0003 / std::sqrt(2.0), 0.03 * 0.0003 / std::sqrt(2.0));
    CHECK_SCALAR_NEAR(r1, 1 / std::sqrt(1 + std::pow(9 / 2.0, 2)), 0.005);  // 0.217
    CHECK_SCALAR_NEAR(r2, 1 / std::sqrt(1 + std::pow(9 / 2.0, 4)), 0.003);  // 0.049
    const json& c = cases()["B4_tremor_attenuation_on_log"];
    CHECK_SCALAR_NEAR(r1, c["r1"].get<double>(), 1e-9);
    CHECK_SCALAR_NEAR(r2, c["r2"].get<double>(), 1e-9);
}

TEST_CASE("test_B5_lag_estimate") {
    const VectorXd t = time_grid(n_of("B5_lag_estimate"));
    const VectorXd x = ((2 * PI * 0.5 * t.array()).sin() + 0.3 * (2 * PI * 0.2 * t.array()).sin()).matrix();
    VectorXd y(x.size());
    y.head(37).setConstant(x(0));
    y.tail(x.size() - 37) = x.head(x.size() - 37);
    CHECK_SCALAR_NEAR(lab::lag_estimate(x, y, DT, 0.2), 0.037, 1.1e-3);
}

// ------------------------------------------------------------------ Part C: two-port models
TEST_CASE("test_C1_hybrid_matrix") {
    lab::Params q = params();
    q.alpha = 0.25;
    q.beta = 3.0;
    const cd s(0.0, 2 * PI * 2.0);
    const json& c = cases()["C1_hybrid_matrix"];
    for (const std::string arch : {"pp", "pf"}) {
        INFO("arch = " << arch);
        const Matrix2cd H = lab::hybrid_matrix(arch, s, q);
        const std::pair<double, double> inputs[] = {{1.0, 0.0}, {0.0, 1.0}, {0.3, -0.7}};
        for (const auto& in : inputs) {
            INFO("(v_h, f_e) = (" << in.first << ", " << in.second << ")");
            const auto fv = solve_two_port(arch, s, q, in.first, in.second);
            const Eigen::Vector2cd out = H * Eigen::Vector2cd(in.first, in.second);
            check_capprox(out(0), fv.first, 1e-10, 1e-12);
            check_capprox(out(1), -fv.second, 1e-10, 1e-12);
        }
        const Matrix2cd Hpy = as_m2cd(c[arch]);
        for (int r = 0; r < 2; ++r)
            for (int k = 0; k < 2; ++k) check_capprox(H(r, k), Hpy(r, k), 1e-10, 1e-12);
        const std::vector<Matrix2cd> Hs = lab::hybrid_matrix(arch, Eigen::Vector3cd(cd(0, 1.0), cd(0, 10.0), cd(0, 100.0)), q);
        CHECK(Hs.size() == 3);
    }
    CHECK_THROWS_AS(lab::hybrid_matrix("4c", s, q), std::invalid_argument);  // C++ only: unknown architecture
}

TEST_CASE("test_C2_transmitted_impedance") {
    const lab::Params& p = params();
    Matrix2cd H_ideal;
    H_ideal << cd(0, 0), cd(2.0, 0), cd(-0.25, 0), cd(0, 0);
    check_capprox(lab::transmitted_impedance(H_ideal, cd(300.0, 5.0)), 0.5 * cd(300.0, 5.0));
    // quasi-static contact with a stiff wall: both architectures feel alpha*beta*(series stiffness)
    cd s(0.0, 1e-3);
    const double ke = wall(p).k;
    const double kser = p.kc * ke / (p.kc + ke);
    const json& c = cases()["C2_transmitted_impedance"];
    for (const std::string arch : {"pp", "pf"}) {
        INFO("arch = " << arch);
        const cd Zt = lab::transmitted_impedance(lab::hybrid_matrix(arch, s, p), ke / s);
        CHECK_SCALAR_NEAR((Zt * s).real(), p.alpha * p.beta * kser, 1e-3 * p.alpha * p.beta * kser);
        check_capprox(Zt, as_cd(c["Zt_quasi_static"][arch]), 1e-9, 1e-12);
    }
    // free space (Z_e = 0): P-F feels only the leader, Z_t = h11 = Z_m
    s = cd(0.0, 20.0);
    const cd Zt = lab::transmitted_impedance(lab::hybrid_matrix("pf", s, p), 0.0);
    check_capprox(Zt, p.m_m * s + p.b_m);
}

TEST_CASE("test_C3_llewellyn") {
    const lab::Params& p = params();
    const int n = cases()["C3_llewellyn"]["n_w"].get<int>();
    Eigen::VectorXcd jw(n);
    for (int i = 0; i < n; ++i) jw(i) = cd(0.0, std::pow(10.0, -1.0 + 4.0 * i / (n - 1)));  // np.logspace(-1, 3, 400)
    lab::Params sym = p;
    sym.alpha = 1.0;
    sym.beta = 1.0;
    const std::vector<Matrix2cd> Hpp = lab::hybrid_matrix("pp", jw, sym);
    const std::vector<Matrix2cd> Hpf = lab::hybrid_matrix("pf", jw, p);
    REQUIRE(Hpp.size() == static_cast<std::size_t>(n));
    REQUIRE(Hpf.size() == static_cast<std::size_t>(n));
    double min_pp = std::numeric_limits<double>::infinity(), min_pf = min_pp;
    for (int i = 0; i < n; ++i) {
        min_pp = std::min(min_pp, lab::llewellyn(Hpp[static_cast<std::size_t>(i)]));
        min_pf = std::min(min_pf, lab::llewellyn(Hpf[static_cast<std::size_t>(i)]));
    }
    CHECK(min_pp >= -1e-9);  // symmetric P-P is a passive network
    CHECK(min_pf < -0.1);    // P-F with b_m = 0.15 N s/m is not absolutely stable
    Matrix2cd H;
    H << cd(2, 1), cd(0.5, 0), cd(-0.5, 0.2), cd(1, -1);
    const cd g = H(0, 1) * H(1, 0);
    CHECK_APPROX(lab::llewellyn(H), 2 * 2 * 1 - g.real() - std::abs(g));
}

// ------------------------------------------------------------------ Part D: bilateral control in time
TEST_CASE("test_D1_control_forces") {
    lab::Params q = params();
    q.alpha = 0.5;
    q.beta = 2.0;
    auto f = lab::control_forces("pp", 0.01, 0.1, 0.004, 0.02, 0.3, q);
    const double a_s_ref = q.kc * (0.005 - 0.004) + q.bc * (0.05 - 0.02);
    CHECK_APPROX(f.second, a_s_ref);
    CHECK_APPROX(f.first, -2.0 * a_s_ref);
    f = lab::control_forces("pf", 0.01, 0.1, 0.004, 0.02, 0.3, q);
    CHECK_APPROX(f.second, a_s_ref);
    CHECK_APPROX(f.first, -0.6);
}

TEST_CASE("test_D2_closed_loop_matrix") {
    lab::Params q = params();
    q.beta = 1.7;
    const lab::Environment& e = wall(params());
    for (const std::string arch : {"pp", "pf"}) {
        INFO("arch = " << arch);
        const Matrix4d A = lab::closed_loop_matrix(arch, q, e.k, e.b);
        const Matrix4d R = A_ref(arch, q, e.k, e.b);
        CHECK(allclose(Eigen::Map<const VectorXd>(A.data(), 16), Eigen::Map<const VectorXd>(R.data(), 16)));
        CHECK_MAT_NEAR(A, R, 1e-8 + 1e-5 * R.cwiseAbs().maxCoeff());
    }
}

TEST_CASE("test_D3_max_stable_beta") {
    const lab::Params& p = params();
    const lab::Environment& e = wall(p);
    const double bmax = lab::max_stable_beta("pf", p, e.k, e.b);
    auto lam = [&](double b) {
        lab::Params q = p;
        q.beta = b;
        return max_real_eig(A_ref("pf", q, e.k, e.b));
    };
    CHECK(lam(0.995 * bmax) < 0);
    CHECK(lam(1.005 * bmax) > 0);
    CHECK_SCALAR_NEAR(bmax, 8.155, 0.01);
    CHECK_SCALAR_NEAR(bmax, cases()["D3_max_stable_beta"]["bmax_pf"].get<double>(), 1e-6);
    const double bpp = lab::max_stable_beta("pp", p, e.k, e.b);
    CHECK((std::isinf(bpp) && bpp > 0));
}

TEST_CASE("test_D_steady_contact_force") {
    // Static balance with the stiff wall: hand spring = beta f_e, f_e = series stiffness * (alpha x_m - x_w).
    const lab::Params& p = params();
    const lab::ReachTask& rt = p.reach_task;
    const lab::Environment& e = wall(p);
    const double kser = p.kc * e.k / (p.kc + e.k);
    const double x_m = (p.k_h * rt.x_goal + p.beta * kser * e.x_wall) / (p.k_h + p.beta * kser * p.alpha);
    const double fe_ref = kser * (p.alpha * x_m - e.x_wall);
    const VectorXd t = time_grid(n_of("D_steady_contact_force"));
    for (const std::string arch : {"pp", "pf"}) {
        INFO("arch = " << arch);
        const lab::TeleopSim r = lab::simulate_teleop(arch, p, e, lab::reach_intent(p, t), 0.0, 0, true, true,
                                                      lab::control_forces);
        const Eigen::Index N = r.f_e.size();
        REQUIRE(N == t.size());
        CHECK_SCALAR_NEAR(r.f_e(N - 1), fe_ref, 1e-4 * fe_ref);  // 0.649 N
        CHECK_SCALAR_NEAR(r.x_m(N - 1), x_m, 1e-4 * x_m);
        CHECK(!r.tripped);
    }
}

TEST_CASE("test_D_free_space_transparency") {
    // In free space the P-F operator feels no motor force; P-P feels the follower's drag (small here).
    const lab::Params& p = params();
    const VectorXd t = time_grid(n_of("D_free_space_transparency"));
    const VectorXd xi = lab::reach_intent(p, t);
    const lab::TeleopSim rpf = lab::simulate_teleop("pf", p, wall(p, "free"), xi, 0.0, 0, true, true,
                                                    lab::control_forces);
    const lab::TeleopSim rpp = lab::simulate_teleop("pp", p, wall(p, "free"), xi, 0.0, 0, true, true,
                                                    lab::control_forces);
    CHECK(rpf.a_m.cwiseAbs().maxCoeff() == 0.0);
    const double m = rpp.a_m.cwiseAbs().maxCoeff();
    CHECK(1e-3 < m);
    CHECK(m < 0.05);
}

TEST_CASE("test_D_simulated_stability_trade_off") {
    const lab::Params& p = params();
    const lab::Environment& e = wall(p);
    const double bmax = lab::max_stable_beta("pf", p, e.k, e.b);
    const VectorXd t = time_grid(n_of("D_simulated_stability_trade_off"));
    const VectorXd xi = lab::minjerk(t, 0.0, 2.0, 0.0, 0.020);
    lab::Params q = p;
    q.beta = 0.5 * bmax;
    const lab::TeleopSim ok = lab::simulate_teleop("pf", q, e, xi, 0.0, 0, false, false, lab::control_forces);
    REQUIRE(ok.f_e.size() >= 1000);
    CHECK(ptp(ok.f_e.tail(500)) < 1e-6);  // settles on the wall
    q.beta = 1.2 * bmax;
    const lab::TeleopSim bad = lab::simulate_teleop("pf", q, e, xi, 0.0, 0, false, false, lab::control_forces);
    REQUIRE(bad.f_e.size() >= 1000);
    const double last = ptp(bad.f_e.tail(500)), before = ptp(bad.f_e.segment(bad.f_e.size() - 1000, 500));
    CHECK((last > 10 * std::max(before, 1e-9) || last > 1e3));  // grows without bound
    const lab::TeleopSim sat = lab::simulate_teleop("pf", q, e, xi, 0.0, 0, true, true, lab::control_forces);
    REQUIRE(sat.f_e.size() >= 500);
    CHECK(sat.a_m.cwiseAbs().maxCoeff() <= p.f_max + 1e-12);  // the 2 N limit holds
    CHECK(ptp(sat.f_e.tail(500)) > 0.01);                     // but the operator feels chatter
}

// ------------------------------------------------------------------ Part E: EOD follower speed limit
TEST_CASE("test_E1_eod_approach_rates") {
    const lab::EodModel m = lab::eod_6r_ugv();
    const VectorXd& q = m.params.q_ready;
    const MatrixXd Jb = mr::jacobian_body(m.Blist, q);
    const VectorXd qd = lab::eod_approach_rates(m, q);
    REQUIRE(qd.size() == 6);
    CHECK_MAT_NEAR(Jb * qd, vec({0, 0, 0, 1, 0, 0}), 1e-10);
    const VectorXd qd_ref = vec({0, -1.3100, 2.0195, 0, -0.7095, 0});
    CHECK_MESSAGE(allclose(qd, qd_ref, 1e-5, 1e-4), "qd = " << qd.transpose() << ", expected about "
                                                            << qd_ref.transpose());  // np.allclose(atol=1e-4)
    CHECK_MAT_NEAR(qd, as_vec(cases()["E1_eod_approach_rates"]["qd"]), 1e-10);
}

TEST_CASE("test_E2_max_approach_speed") {
    const lab::EodModel m = lab::eod_6r_ugv();
    const VectorXd& q = m.params.q_ready;
    const double v100 = lab::max_approach_speed(m, q, 1.0);
    const double v_ref = (30 * PI / 180) / 2.019465;
    CHECK_SCALAR_NEAR(v100, v_ref, 1e-5 * v_ref);
    CHECK_SCALAR_NEAR(lab::max_approach_speed(m, q, 0.5), 0.12964, 1e-5);
}
