// Auto-checks for SBEG 46X Lab 3 (delayed bilateral teleoperation: instability, wave variables, TDPA) -- C++ track.
// Mirrors test_lab.py test by test (same names, same order, same tolerances).
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*margin*,*stiffness*"      (run only some tests, like pytest -k)
// Ground truth: closed-form results (the integrator loop s + k e^{-s tau} has the delay margin pi/(2k), the series
// spring, the wave power identity, the transmission-line impedances b coth(jwT) and b tanh(jwT)), an independent
// frequency sweep for the delay margin written in this file, energy identities, the Hapkit handle mass from mrcpp
// (mass_matrix of the course Hapkit model), the reference runs in ../data/ref_*.csv (every 10th control sample), and
// cpp_cases.json, which holds the numbers of the Python reference and the random inputs drawn by test_lab.py, so both
// languages are checked against the same values.
// With the untouched starter, the three checks of the data and the given code (parameters, DelayLine, reference
// files) pass, as in Python; every other test fails with "TODO: <function>".
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

using Eigen::VectorXd;
using sbeg::json;
using cd = std::complex<double>;

namespace {
constexpr double PI = 3.14159265358979323846;

const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

const lab::Params& params() {
    static const lab::Params P = lab::load_params(sbeg::lab_path("../data/teleop.json"));
    return P;
}

const json& expected() {
    static const json E = sbeg::load_json(sbeg::lab_path("../data/expected.json"));
    return E;
}

struct LeaderModel {
    double m, b, k;
};
LeaderModel leader_model() {
    const lab::Params& P = params();
    return {P.leader.m + P.hand.m, P.leader.b + P.hand.b, P.hand.k};
}

/// Python math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol) with a readable message.
sbeg::NearResult isclose(double a, double b, double rel = 1e-9, double abs_tol = 0.0) {
    return sbeg::compare_near(a, b, std::max(rel * std::max(std::abs(a), std::abs(b)), abs_tol));
}

/// Python np.isclose(a, b) for complex numbers (rtol 1e-5, atol 1e-8 by default).
bool np_isclose(cd a, cd b, double rtol = 1e-5, double atol = 1e-8) { return std::abs(a - b) <= atol + rtol * std::abs(b); }
bool np_isclose(double a, double b, double rtol = 1e-5, double atol = 1e-8) {
    return std::abs(a - b) <= atol + rtol * std::abs(b);
}

#define CHECK_NEAR_RESULT(expr)                                    \
    do {                                                           \
        const ::sbeg::NearResult sbeg_r_ = (expr);                 \
        CHECK_MESSAGE(sbeg_r_.ok, #expr "\n  " << sbeg_r_.msg);    \
    } while (false)

/// The reference run data/<name>: column name -> values.
struct RefRun {
    sbeg::CsvTable tab;
    VectorXd col(const std::string& name) const { return tab.data.col(tab.col(name)); }
};
RefRun load_ref(const std::string& name) { return RefRun{sbeg::read_csv(sbeg::lab_path("../data/" + name))}; }

const VectorXd& sim_field(const lab::SimResult& sim, const std::string& key) {
    if (key == "x_m") return sim.x_m;
    if (key == "x_s") return sim.x_s;
    if (key == "x_d") return sim.x_d;
    if (key == "F_m") return sim.F_m;
    return sim.F_c;
}

/// Compare every 10th sample of a run with a reference file up to t_max (the Python _compare).
void compare_run(const lab::SimResult& sim, const RefRun& ref, double atol_x = 2e-6, double atol_F = 2e-3,
                 double t_max = 4.0) {
    const VectorXd t_ref = ref.col("t");
    for (const std::string key : {"x_m", "x_s", "x_d", "F_m", "F_c"}) {
        const double tol = (key[0] == 'x') ? atol_x : atol_F;
        const VectorXd& got = sim_field(sim, key);
        const VectorXd exp = ref.col(key);
        REQUIRE(got.size() == sim.t.size());
        REQUIRE((got.size() + 9) / 10 == t_ref.size());  // every 10th sample, as the reference file
        double err = 0.0;
        for (Eigen::Index i = 0; i < t_ref.size(); ++i) {
            if (t_ref(i) > t_max) continue;
            const Eigen::Index k = 10 * i;
            REQUIRE(k < got.size());
            err = std::max(err, std::abs(got(k) - exp(i)));
            if (std::isnan(got(k))) err = std::numeric_limits<double>::infinity();
        }
        CHECK_MESSAGE(err < tol, key << " differs from the reference run by " << err);
    }
}

double sign(double x) { return (x > 0) - (x < 0); }

/// Independent check: dense frequency sweep of |k - m w^2 + j b w| - kp, refined by bisection (the Python
/// _sweep_margin, same grid).
double sweep_margin(double m, double b, double k, double kp) {
    auto g = [&](double w) { return std::abs(cd(k - m * w * w, b * w)) - kp; };
    const int num = 400001;
    const double start = 1e-3, stop = 400.0, step = (stop - start) / (num - 1);
    auto wgrid = [&](int i) { return i == num - 1 ? stop : start + i * step; };
    double best = std::numeric_limits<double>::infinity();
    double g_prev = g(wgrid(0));
    for (int i = 0; i + 1 < num; ++i) {
        const double g_next = g(wgrid(i + 1));
        if (sign(g_prev) != sign(g_next)) {
            double lo = wgrid(i), hi = wgrid(i + 1);
            for (int it = 0; it < 60; ++it) {
                const double mid = 0.5 * (lo + hi);
                if (sign(g(mid)) == sign(g(lo)))
                    lo = mid;
                else
                    hi = mid;
            }
            const double wc = 0.5 * (lo + hi);
            const double phase = std::arg(-cd(k - m * wc * wc, b * wc) / kp);
            double a = std::fmod(-phase, 2 * PI);
            if (a < 0) a += 2 * PI;
            best = std::min(best, a / wc);
        }
        g_prev = g_next;
    }
    return best;
}

/// Hapkit effective handle mass at theta = 0 from the course model (course_models.hapkit / handle_mass) with mrcpp.
double hapkit_handle_mass() {
    const double rh = 0.090, rs = 0.073, rp = 0.0042;
    Eigen::MatrixXd S(6, 1);
    S << 0, 0, 1, 0, 0, 0;
    const Eigen::Matrix4d M = mr::rp_to_trans(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0, rh, 0));
    const Eigen::Matrix4d com = mr::rp_to_trans(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0, -0.010, 0));
    const mr::TransformList Mlist = {com, mr::trans_inv(com) * M};
    mr::Matrix6d G = mr::Matrix6d::Zero();
    G.diagonal() << 2.0e-4, 1.0e-4, 1.5e-4, 0.10, 0.10, 0.10;
    const mr::InertiaList Glist = {G};
    const double I_pivot = mr::mass_matrix(VectorXd::Zero(1), Mlist, Glist, S)(0, 0);
    return (I_pivot + 1.0e-6 * (rs / rp) * (rs / rp)) / (rh * rh);
}

double vmin(const VectorXd& v) { return v.size() ? v.minCoeff() : std::numeric_limits<double>::quiet_NaN(); }
}  // namespace

// ------------------------------------------------------------------------------ data and provided code (pass)
TEST_CASE("test_params_match_course_hapkit") {
    const double m = hapkit_handle_mass();
    CHECK_NEAR_RESULT(isclose(m, cases()["params_match_course_hapkit"]["handle_mass"].get<double>(), 1e-12));
    const lab::Params& P = params();
    CHECK(std::abs(P.leader.m - m) < 1e-4);
    CHECK(std::abs(P.follower.m - m) < 1e-4);
    CHECK(P.force_max == 2.0);
    CHECK(P.x_range == 0.040);
}

TEST_CASE("test_delay_line_hold_and_out_of_order") {
    const std::vector<int> d = {3, 1, 1, -1, 1, 1, 1, 1};
    lab::DelayLine line(d, 8, VectorXd::Zero(1));
    std::vector<double> got;
    for (int n = 0; n < 8; ++n) {
        line.send(n, VectorXd::Constant(1, static_cast<double>(n)));
        const VectorXd r = line.receive(n);
        REQUIRE(r.size() == 1);
        got.push_back(n > 0 ? r(0) : 0.0);
    }
    // packet 0 arrives at 3 but packet 1 (newer) already arrived at 2 -> 0 is discarded; 3 is lost (held 2)
    CHECK(got == std::vector<double>{0.0, 0.0, 1.0, 2.0, 2.0, 4.0, 5.0, 6.0});
}

TEST_CASE("test_reference_files_present") {
    for (const auto& f : cases()["reference_files_present"]["files"]) {
        const std::string name = f.get<std::string>();
        INFO("file " << name);
        std::ifstream in(sbeg::fs_path(sbeg::lab_path("../data/" + name)));
        CHECK(in.good());
    }
}

// ------------------------------------------------------------------------------ Part A: delay margin
TEST_CASE("test_contact_stiffness_series") {
    CHECK_NEAR_RESULT(isclose(lab::contact_stiffness(200.0, 300.0), 120.0));
    CHECK_NEAR_RESULT(isclose(lab::contact_stiffness(200.0, 5000.0), 200.0 * 5000.0 / 5200.0));
    for (const auto& c : cases()["contact_stiffness_series"])
        CHECK_NEAR_RESULT(isclose(lab::contact_stiffness(c["Kp"].get<double>(), c["k_e"].get<double>()),
                                  c["kp"].get<double>()));
}

TEST_CASE("test_delay_margin_integrator_closed_form") {
    // leader = pure damper (m = 0, b = 1, k = 0): s + kp e^{-s tau} -> w_c = kp, tau* = pi / (2 kp)
    for (double kp : {0.5, 2.0, 10.0}) {
        INFO("kp = " << kp);
        const auto [tau, w] = lab::delay_margin(0.0, 1.0, 0.0, kp);
        CHECK_NEAR_RESULT(isclose(tau, PI / (2 * kp)));
        CHECK_NEAR_RESULT(isclose(w, kp));
    }
}

TEST_CASE("test_delay_margin_course_teleoperator") {
    // Python: parametrized over env = tissue, wall (one C++ test case loops over both)
    const LeaderModel L = leader_model();
    for (const std::string env : {"tissue", "wall"}) {
        INFO("env = " << env);
        const double kp = lab::contact_stiffness(params().controller.Kp, params().environments.at(env).k);
        const auto [tau, w] = lab::delay_margin(L.m, L.b, L.k, kp);
        CHECK_NEAR_RESULT(isclose(tau, sweep_margin(L.m, L.b, L.k, kp), 1e-6));
        CHECK_NEAR_RESULT(isclose(tau, expected()["margin_" + env]["tau_round_trip"].get<double>(), 1e-9));
        const cd s(0.0, w);  // the characteristic equation is satisfied at s = j w_c
        CHECK(std::abs(L.m * s * s + L.b * s + L.k + kp * std::exp(-s * tau)) < 1e-8 * kp);
    }
    for (const auto& c : cases()["delay_margin_course_teleoperator"])  // the Python sweep gives the same margin
        CHECK_NEAR_RESULT(isclose(sweep_margin(L.m, L.b, L.k, c["kp"].get<double>()),
                                  c["tau_sweep"].get<double>(), 1e-9));
}

TEST_CASE("test_delay_margin_infinite_when_no_crossing") {
    const auto [tau, w] = lab::delay_margin(0.157, 50.0, 200.0, 120.0);  // heavy damping: |k - m w^2 + j b w| > kp
    CHECK(std::isinf(tau));
    CHECK(tau > 0);
    CHECK(std::isnan(w));
}

// ------------------------------------------------------------------------------ Part B: wave variables
TEST_CASE("test_wave_encode_power_identity") {
    for (const auto& c : cases()["wave_encode_power_identity"]) {  // the draws of np.random.default_rng(463)
        const double F = c["F"].get<double>(), xd = c["xdot"].get<double>(), b = c["b"].get<double>();
        const auto [u, v] = lab::wave_encode(F, xd, b);
        CHECK_NEAR_RESULT(isclose(0.5 * (u * u - v * v), F * xd, 1e-9, 1e-12));
        CHECK_NEAR_RESULT(isclose(u, (b * xd + F) / std::sqrt(2 * b)));
        CHECK_NEAR_RESULT(isclose(v, c["v"].get<double>(), 1e-9, 1e-15));
    }
}

TEST_CASE("test_wave_leader_and_follower_are_consistent") {
    for (const auto& c : cases()["wave_leader_and_follower_are_consistent"]) {
        const double b = c["b"].get<double>();
        const double xdot_m = c["xdot_m"].get<double>(), v_m = c["v_m"].get<double>();
        const auto [F_m, u_m] = lab::wave_leader(xdot_m, v_m, b);
        const auto [u1, v1] = lab::wave_encode(F_m, xdot_m, b);
        CHECK(np_isclose(u1, u_m));
        CHECK(np_isclose(v1, v_m));
        const double u_s = c["u_s"].get<double>(), F_c = c["F_c"].get<double>();
        const auto [xdot_d, v_s] = lab::wave_follower(u_s, F_c, b);
        const auto [u2, v2] = lab::wave_encode(F_c, xdot_d, b);
        CHECK(np_isclose(u2, u_s));
        CHECK(np_isclose(v2, v_s));
        CHECK_NEAR_RESULT(isclose(F_m, c["F_m"].get<double>(), 1e-9, 1e-12));
        CHECK_NEAR_RESULT(isclose(xdot_d, c["xdot_d"].get<double>(), 1e-9, 1e-12));
    }
}

TEST_CASE("test_wave_static_contact_transmits_force_exactly") {
    const double b = 5.0, F_c = 0.8;  // resting leader and follower: u = F/sqrt(2b), v = -F/sqrt(2b)
    const auto [xdot_d, v_s] = lab::wave_follower(F_c / std::sqrt(2 * b), F_c, b);
    const double F_m = lab::wave_leader(0.0, v_s, b).first;
    CHECK(std::abs(xdot_d) < 1e-12);
    CHECK_NEAR_RESULT(isclose(F_m, F_c));
}

TEST_CASE("test_wave_apparent_impedance") {
    const double b = 5.0, T = 0.1;
    for (double w : {0.3, 3.0, 30.0}) {
        INFO("omega = " << w);
        const cd x(0.0, w * T);
        const cd coth = (1.0 + std::exp(-2.0 * x)) / (1.0 - std::exp(-2.0 * x));
        CHECK(np_isclose(lab::wave_apparent_impedance(b, T, w, "rigid"), b * coth));
        CHECK(np_isclose(lab::wave_apparent_impedance(b, T, w, "free"), b / coth));
    }
    const double w = 1e-3;  // low frequency: spring b/T (rigid), mass b T (free)
    CHECK(np_isclose(lab::wave_apparent_impedance(b, T, w, "rigid") * cd(0.0, w), cd(b / T, 0.0), 1e-5));
    CHECK(np_isclose(lab::wave_apparent_impedance(b, T, w, "free") / cd(0.0, w), cd(b * T, 0.0), 1e-5));
    for (const auto& c : cases()["wave_apparent_impedance"]["cases"]) {
        const double om = c["omega"].get<double>();
        const cd rigid(c["rigid"][0].get<double>(), c["rigid"][1].get<double>());
        const cd free_(c["free"][0].get<double>(), c["free"][1].get<double>());
        CHECK(np_isclose(lab::wave_apparent_impedance(b, T, om, "rigid"), rigid, 1e-9, 0.0));
        CHECK(np_isclose(lab::wave_apparent_impedance(b, T, om, "free"), free_, 1e-9, 0.0));
    }
}

// ------------------------------------------------------------------------------ Part C: energy and TDPA
TEST_CASE("test_follower_pd") {
    CHECK_NEAR_RESULT(isclose(lab::follower_pd(0.012, 0.010, 0.05, 200.0, 2.0), 200 * 0.002 - 2 * 0.05));
}

TEST_CASE("test_port_energies") {
    const json& c = cases()["port_energies"];
    const double h = c["h"].get<double>();
    const VectorXd Pw = sbeg::as_vec(c["P"]);  // the 500 draws of np.random.default_rng(463)
    const auto [E_in, E_out] = lab::port_energies(Pw, h);
    REQUIRE(E_in.size() == Pw.size());
    REQUIRE(E_out.size() == Pw.size());
    bool monotone = true;
    for (Eigen::Index i = 1; i < Pw.size(); ++i)
        monotone = monotone && E_in(i) - E_in(i - 1) >= 0 && E_out(i) - E_out(i - 1) >= 0;
    CHECK(monotone);
    VectorXd cum(Pw.size());
    double s = 0.0, pos = 0.0;
    for (Eigen::Index i = 0; i < Pw.size(); ++i) {
        s += Pw(i);
        cum(i) = h * s;
        if (Pw(i) > 0) pos += Pw(i);
    }
    bool allclose = true;  // np.allclose(E_in - E_out, h * cumsum(P)): element by element, rtol 1e-5, atol 1e-8
    for (Eigen::Index i = 0; i < Pw.size(); ++i) allclose = allclose && np_isclose(E_in(i) - E_out(i), cum(i));
    CHECK_MESSAGE(allclose, "E_in - E_out differs from h * cumsum(P)");
    CHECK(np_isclose(E_in(E_in.size() - 1), h * pos));
    CHECK_MAT_NEAR(E_in, sbeg::as_vec(c["E_in"]), 1e-12);
    CHECK_MAT_NEAR(E_out, sbeg::as_vec(c["E_out"]), 1e-12);
}

TEST_CASE("test_passivity_ratio") {
    CHECK(lab::passivity_ratio(-1.0, 0.0) == 1.0);  // the port absorbs energy: nothing to do
    CHECK(lab::passivity_ratio(0.2, 0.5) == 1.0);   // within budget
    CHECK_NEAR_RESULT(isclose(lab::passivity_ratio(0.5, 0.2), 0.4));
    CHECK(lab::passivity_ratio(0.5, -0.1) == 0.0);  // no budget left
}

TEST_CASE("test_tdpa_leader_cases") {
    const double h = 1e-3;
    lab::TdpaLeaderResult a = lab::tdpa_leader(1.0, 0.1, h, 0.0, 0.0, 0.0);  // input power: observer only
    CHECK(a.F_m == 1.0);
    CHECK(a.r == 1.0);
    CHECK_NEAR_RESULT(isclose(a.E_in_m, 1e-4));
    CHECK(a.E_out_m == 0.0);
    a = lab::tdpa_leader(1.0, -0.1, h, 0.0, 0.0, 1.0);  // output within budget
    CHECK(a.F_m == 1.0);
    CHECK(a.r == 1.0);
    CHECK_NEAR_RESULT(isclose(a.E_out_m, 1e-4));
    a = lab::tdpa_leader(1.0, -0.1, h, 0.0, 2e-5, 6e-5);  // output beyond budget (4e-5)
    CHECK_NEAR_RESULT(isclose(a.r, 0.4));
    CHECK_NEAR_RESULT(isclose(a.F_m, 0.4));
    CHECK_NEAR_RESULT(isclose(a.E_out_m, 6e-5));
}

TEST_CASE("test_tdpa_follower_cases") {
    const double F = 0.5, dx = 2e-4;
    lab::TdpaFollowerResult a = lab::tdpa_follower(F, -dx, 0.0, 0.0, 0.0);  // energy into the channel
    CHECK(a.dx_d == -dx);
    CHECK_NEAR_RESULT(isclose(a.E_in_s, 1e-4));
    CHECK(a.r == 1.0);
    a = lab::tdpa_follower(F, dx, 0.0, 3e-5, 8e-5);  // wants 1e-4, budget 5e-5
    CHECK_NEAR_RESULT(isclose(a.r, 0.5));
    CHECK_NEAR_RESULT(isclose(a.dx_d, 1e-4));
    CHECK_NEAR_RESULT(isclose(a.E_out_s, 8e-5));
}

// ------------------------------------------------------------------------------ Part D: simulations
TEST_CASE("test_pp_short_delay_is_stable_and_transparent") {
    const lab::SimResult r = lab::simulate("pp", 0.005);
    compare_run(r, load_ref("ref_pp_tissue_T005.csv"));
    const lab::Metrics m = lab::performance_metrics(r);
    CHECK(m.x_std < 1e-5);
    CHECK(std::abs(m.force_ratio - 1) < 0.01);
    CHECK(std::abs(m.F_m_mean - 1.125) < 0.005);
}

TEST_CASE("test_pp_beyond_margin_chatters_and_channel_is_active") {
    const double T = 0.020;
    CHECK(2 * T > expected()["margin_wall"]["tau_round_trip"].get<double>());
    lab::SimOptions o;
    o.env = "wall";
    const lab::SimResult r = lab::simulate("pp", T, o);
    compare_run(r, load_ref("ref_pp_wall_T020.csv"), 2e-6, 2e-3, 1.5);
    const lab::Metrics m = lab::performance_metrics(r);
    CHECK(m.x_std > 3e-3);
    CHECK(m.E_ch_min < -0.1);
}

TEST_CASE("test_pf_beyond_margin_chatters") {
    const lab::SimResult r = lab::simulate("pf", 0.020);
    compare_run(r, load_ref("ref_pf_tissue_T020.csv"), 2e-6, 2e-3, 1.5);
    CHECK(lab::performance_metrics(r).x_std > 3e-3);
}

TEST_CASE("test_wave_long_delay_is_passive_and_stable") {
    const lab::SimResult r = lab::simulate("wave", 0.100);
    compare_run(r, load_ref("ref_wave_tissue_T100.csv"));
    CHECK(vmin(r.E_ch) > -1e-12);
    const lab::Metrics m = lab::performance_metrics(r);
    CHECK(m.x_std < 5e-4);
    CHECK(std::abs(m.force_ratio - 1) < 0.02);
}

TEST_CASE("test_tdpa_is_passive_and_stable") {
    const lab::SimResult r = lab::simulate("tdpa", 0.020);
    // the passivity controller switches on thresholds, so round-off can shift a switching instant and the trajectory
    // differs between computers: no trajectory comparison here -- the passivity and stability properties below are
    // the real checks (tdpa_leader / tdpa_follower are checked exactly above)
    CHECK(vmin(r.W_m) > -1e-12);
    CHECK(vmin(r.W_s) > -1e-12);
    CHECK(vmin(r.E_ch) > -1e-12);
    const lab::Metrics m = lab::performance_metrics(r);
    CHECK(m.x_std < 5e-4);
    CHECK(m.drift < -1e-3);  // stable, at the price of position drift
}

TEST_CASE("test_tdpa_with_jitter_and_loss_stays_passive") {
    // the packet delays drawn by the Python network_delays (seeds 3 and 4) are taken from cpp_cases.json, so this
    // run sees exactly the same packets as the Python test
    const json& c = cases()["tdpa_with_jitter_and_loss_stays_passive"];
    lab::SimOptions o;
    o.env = c["env"].get<std::string>();
    o.jitter = c["jitter"].get<double>();
    o.loss = c["loss"].get<double>();
    o.fwd_delays = c["fwd_delays"].get<std::vector<int>>();
    o.bwd_delays = c["bwd_delays"].get<std::vector<int>>();
    const lab::SimResult r = lab::simulate("tdpa", c["T"].get<double>(), o);
    CHECK(vmin(r.W_m) > -1e-12);
    CHECK(vmin(r.W_s) > -1e-12);
    CHECK(vmin(r.E_ch) > -1e-12);
    CHECK_FALSE(r.tripped);
}

TEST_CASE("test_performance_metrics_synthetic") {
    const json& c = cases()["performance_metrics_synthetic"];
    const int n = c["n"].get<int>();
    lab::SimResult sim;
    sim.t.resize(n);
    for (int i = 0; i < n; ++i) sim.t(i) = i * 1e-3;  // np.arange(0, 4, 1e-3)
    sim.F_m = VectorXd::Constant(n, 0.9);
    sim.f_e = VectorXd::Constant(n, 1.0);
    sim.x_m = (0.01 + 1e-3 * (20 * sim.t.array()).sin()).matrix();
    sim.x_s = VectorXd::Constant(n, 0.01);
    sim.x_d = VectorXd::Constant(n, 0.008);
    sim.E_ch = (-0.01 * sim.t.array().sin()).matrix();
    const lab::Metrics m = lab::performance_metrics(sim, 2.0);
    // the expectations of test_lab.py, written as loops over the samples with t >= 2
    double s_d = 0.0, s_x = 0.0, s_e2 = 0.0;
    int cnt = 0;
    for (int i = 0; i < n; ++i)
        if (sim.t(i) >= 2.0) {
            s_d += 0.008 - sim.x_m(i);
            s_x += sim.x_m(i);
            s_e2 += (0.01 - sim.x_m(i)) * (0.01 - sim.x_m(i));
            ++cnt;
        }
    REQUIRE(cnt > 0);
    double s_v = 0.0;
    for (int i = 0; i < n; ++i)
        if (sim.t(i) >= 2.0) s_v += (sim.x_m(i) - s_x / cnt) * (sim.x_m(i) - s_x / cnt);
    CHECK_NEAR_RESULT(isclose(m.force_ratio, 0.9));
    CHECK_NEAR_RESULT(isclose(m.F_m_mean, 0.9));
    CHECK_NEAR_RESULT(isclose(m.drift, s_d / cnt));
    CHECK_NEAR_RESULT(isclose(m.x_std, std::sqrt(s_v / cnt)));
    CHECK_NEAR_RESULT(isclose(m.x_err_rms, std::sqrt(s_e2 / cnt)));
    CHECK_NEAR_RESULT(isclose(m.E_ch_min, sim.E_ch.minCoeff()));
    const json& e = c["metrics"];  // the Python reference values
    CHECK_NEAR_RESULT(isclose(m.drift, e["drift"].get<double>()));
    CHECK_NEAR_RESULT(isclose(m.x_std, e["x_std"].get<double>()));
    CHECK_NEAR_RESULT(isclose(m.x_err_rms, e["x_err_rms"].get<double>()));
    CHECK_NEAR_RESULT(isclose(m.E_ch_min, e["E_ch_min"].get<double>()));
}

TEST_CASE("test_latency_budget") {
    const auto [total, margin] = lab::latency_budget({{"encode", 5.0}, {"radio", 60.0}, {"decode", 5.0}}, 100.0);
    CHECK(total == 70.0);
    CHECK(margin == 30.0);
}
