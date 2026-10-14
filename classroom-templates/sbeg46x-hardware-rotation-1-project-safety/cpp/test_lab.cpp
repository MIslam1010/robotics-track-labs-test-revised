// Auto-checks for SBEG 46X Hardware rotation 1 (Week 8) -- C++ track. Mirrors test_lab.py test by test:
// pre-flight record, dry run, contact equilibrium, free-decay identification, watchdog check, trial metrics and
// channel energy, simulation versus hardware, safe stop and the IMU at rest.
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*preflight*,*contact*,*dry_run*"      (run only some tests, like pytest -k)
// Ground truth: closed forms, the course models (course_models.hpp), synthetic signals with known answers, the
// synthetic station logs in ../data with the course team's reference results in ../data/expected.json, and
// cpp_cases.json (the numbers of the Python reference, and the random gyro signal of test_imu_rest_stats).
// test_course_models_match_task of the Python track checks only given code; its checks are at the start of
// test_contact_equilibrium_matches_linear_solve here.
#include "lab.hpp"

#include <sbeg/labio.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using sbeg::json;

namespace {
constexpr double INF = std::numeric_limits<double>::infinity();

const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

const json& EXP() {
    static const json E = sbeg::load_json(sbeg::lab_path("../data/expected.json"));
    return E;
}

/// pytest.approx(expected, rel=rel, abs=abs_tol) as an absolute tolerance.
double approx(double expected, double rel, double abs_tol = 0.0) { return std::max(rel * std::abs(expected), abs_tol); }

/// A complete haptic-station record that satisfies every rule.
json good_record() {
    json r = lab::load_record();
    r["date"] = "2026-11-04";
    r["team"] = "T07";
    r["scenario"] = 1;
    r["units"] = {{"leader", "H07"}, {"follower", "H08"}, {"imu", "IMU-03"}};
    r["members"] = json::array();
    for (const char* n : {"Salma", "Omar", "Mona"})
        r["members"].push_back({{"name", n}, {"induction_signed", true}, {"touches_hardware", true}});
    r["supervisor"] = "TA Hany";
    r["supervisor_present"] = true;
    r["estop_tested"] = true;
    r["stop_holder"] = "Omar";
    r["leader_operator"] = "Salma";
    r["hands_on_handle"] = json::array({"operator"});
    r["dry_run"] = {{"done", true}, {"peak_force_N", 0.32}, {"peak_x_m", 0.0204}};
    r["devices"] = {{"force_max_N", 2.0}, {"watchdog_enabled", true}, {"watchdog_range_m", 0.040},
                    {"loop_overrun_watchdog", true}};
    r["workspace"] = {{"cables_tidy", true}, {"loose_items", false}, {"devices_fixed", true}};
    r["imu"] = {{"powered_by", "USB"}, {"voltage_V", 5.0}, {"over_clothing", true}};
    r["logging"] = {{"csv", true}, {"session_log", true}};
    return r;
}

json good_arm() {
    return {{"first_run", true},     {"speed_fraction", 0.25}, {"soft_limit_margin_deg", 10.0},
            {"keepout_radius_m", 0.58}, {"keepout_taped", true}, {"keepout_clear", true},
            {"payload_kg", 0.10},    {"current_fraction", 0.60}, {"estop_tested", true},
            {"mat", true}};
}

/// Sorted set of the rule labels of a list of violations ("B4: ..." -> "B4").
std::vector<std::string> labels(const std::vector<std::string>& v) {
    std::set<std::string> s;
    for (const auto& e : v) s.insert(e.substr(0, e.find(':')));
    return {s.begin(), s.end()};
}

std::string show(const std::vector<std::string>& v) {
    std::string s = "[";
    for (const auto& e : v) s += (s.size() > 1 ? ", " : "") + e;
    return s + "]";
}

/// t = np.arange(0, 1.5, 1e-3) and the damped cosine _damped(t, ...) of test_lab.py.
VectorXd damped(const VectorXd& t, double A = 0.012, double zeta = 0.03, double wn = 40.0, double c = 1e-4) {
    const double wd = wn * std::sqrt(1 - zeta * zeta);
    VectorXd x(t.size());
    for (Index i = 0; i < t.size(); ++i) x(i) = A * std::exp(-zeta * wn * t(i)) * std::cos(wd * t(i)) + c;
    return x;
}
}  // namespace

// ------------------------------------------------------------------ Part A: pre-flight and dry run
TEST_CASE("test_preflight_good_record_passes") {
    std::vector<std::string> v = lab::preflight(good_record());
    CHECK_MESSAGE(v.empty(), show(v));
    json r = good_record();
    r["arm"] = good_arm();
    v = lab::preflight(r);
    CHECK_MESSAGE(v.empty(), show(v));
    r["arm"]["first_run"] = false;  // a repeated motion may run at 50 %
    r["arm"]["speed_fraction"] = 0.50;
    v = lab::preflight(r);
    CHECK_MESSAGE(v.empty(), show(v));
    r["arm"]["payload_kg"] = 0.125;  // exactly 0.125 kg is allowed (A4)
    v = lab::preflight(r);
    CHECK_MESSAGE(v.empty(), show(v));
}

TEST_CASE("test_preflight_blank_template_fails") {
    const std::vector<std::string> got = labels(lab::preflight(lab::load_record()));
    for (const char* rule : {"A5", "A7", "B1", "B2", "B3", "B4", "B5", "B6", "B7"}) {
        INFO("rule " << rule << " missing in " << show(got));
        CHECK(std::find(got.begin(), got.end(), rule) != got.end());
    }
}

TEST_CASE("test_preflight_each_rule") {
    // (JSON pointer of the field, new value, the only rule that must be flagged)
    const std::vector<std::pair<std::string, std::pair<json, std::string>>> params = {
        {"/members/1/induction_signed", {false, "A5"}},
        {"/supervisor_present", {false, "B1"}},
        {"/estop_tested", {false, "A7"}},
        {"/dry_run/done", {false, "B2"}},
        {"/dry_run/peak_force_N", {2.4, "B2"}},
        {"/devices/force_max_N", {3.0, "B3"}},
        {"/devices/watchdog_range_m", {0.050, "B3"}},
        {"/hands_on_handle", {json::array({"operator", "TA"}), "B3"}},
        {"/channel/delay_s", {0.1, "B4"}},
        {"/stop_holder", {"Salma", "B4"}},
        {"/workspace/loose_items", {true, "B5"}},
        {"/imu/voltage_V", {12.0, "B6"}},
        {"/imu/over_clothing", {false, "B6"}},
        {"/logging/csv", {false, "B7"}},
    };
    for (const auto& p : params) {
        json r = good_record();
        r[json::json_pointer(p.first)] = p.second.first;
        const std::vector<std::string> got = labels(lab::preflight(r));
        INFO(p.first << " = " << p.second.first.dump() << ": expected [" << p.second.second << "], got " << show(got));
        CHECK(got == std::vector<std::string>{p.second.second});
    }
}

TEST_CASE("test_preflight_watchdog_disabled_is_b3_and_b4") {
    json r = good_record();
    r["devices"]["watchdog_enabled"] = false;  // "emulating" a delay by disabling the watchdog
    std::vector<std::string> got = labels(lab::preflight(r));
    INFO("got " << show(got));
    CHECK(got == std::vector<std::string>{"B3", "B4"});
    r = good_record();
    r["members"].erase(1);
    r["members"].erase(1);
    r["stop_holder"] = "Salma";
    got = labels(lab::preflight(r));
    CHECK(std::find(got.begin(), got.end(), "B1") != got.end());  // never alone
}

TEST_CASE("test_preflight_arm_rules") {
    const std::vector<std::pair<std::string, std::pair<json, std::string>>> params = {
        {"soft_limit_margin_deg", {5.0, "A1"}}, {"keepout_radius_m", {0.40, "A2"}}, {"keepout_taped", {false, "A2"}},
        {"speed_fraction", {0.30, "A3"}},       {"payload_kg", {0.20, "A4"}},       {"keepout_clear", {false, "A6"}},
        {"estop_tested", {false, "A7"}},        {"mat", {false, "B5"}},             {"current_fraction", {0.80, "I"}},
    };
    for (const auto& p : params) {
        json r = good_record();
        r["arm"] = good_arm();
        r["arm"][p.first] = p.second.first;
        const std::vector<std::string> got = labels(lab::preflight(r));
        INFO("arm." << p.first << " = " << p.second.first.dump() << ": expected [" << p.second.second << "], got "
                    << show(got));
        CHECK(got == std::vector<std::string>{p.second.second});
    }
}

TEST_CASE("test_contact_equilibrium_matches_linear_solve") {
    const json task = lab::load_task();
    {  // test_course_models_match_task (given code, numbers identical)
        const lab::HandleModel h = lab::handle_model();
        CHECK_SCALAR_NEAR(h.m_h, task["device_nominal"]["m_h"].get<double>(), 5e-6);
        CHECK_SCALAR_NEAR(h.k_g, task["device_nominal"]["k_g"].get<double>(), 5e-5);
        CHECK((lab::HAPTIC_SAFETY.force_max == 2.0 && lab::HAPTIC_SAFETY.x_range == 0.040));
        const lab::CourseArmSafety& S = lab::COURSE_ARM_SAFETY;
        CHECK((S.soft_limit_margin_deg == 10.0 && S.speed_fraction_first_run == 0.25 && S.speed_fraction == 0.50 &&
               S.payload_max_kg == 0.125 && S.current_fraction == 0.60));
        CHECK_APPROX(S.reach_m + S.keepout_extra_m, 0.58);
    }
    const json &pp = task["pp"], &env = task["tissue"];
    const double kg = task["device_nominal"]["k_g"].get<double>();
    const double Kp = pp["Kp"].get<double>(), a = pp["alpha"].get<double>(), b = pp["beta"].get<double>();
    const double ke = env["k"].get<double>(), xw = env["x_wall"].get<double>();
    for (double x_m : {0.004, 0.0205, 0.030}) {
        INFO("x_m = " << x_m);
        const lab::ContactEquilibrium e = lab::contact_equilibrium(task, x_m);
        // unknowns x_s, F_c:  F_c = Kp (a x_m - x_s);  F_c = ke (x_s - xw)^+ + kg x_s
        double xs = 0.0, Fc = 0.0;
        for (bool contact : {false, true}) {
            const double k_t = kg + (contact ? ke : 0.0);
            Eigen::Matrix2d A;
            A << Kp, 1.0, -k_t, 1.0;
            const Eigen::Vector2d sol = A.partialPivLu().solve(Eigen::Vector2d(Kp * a * x_m, -(contact ? ke * xw : 0.0)));
            xs = sol(0);
            Fc = sol(1);
            if ((xs > xw) == contact) break;
        }
        CHECK_SCALAR_NEAR(e.x_s, xs, 1e-12);
        CHECK_SCALAR_NEAR(e.F_c, Fc, 1e-12);
        CHECK_SCALAR_NEAR(e.F_m, b * Fc, 1e-12);
        CHECK_SCALAR_NEAR(e.delta, std::max(0.0, xs - xw), 1e-12);
    }
    for (const auto& c : cases()["contact_equilibrium_matches_linear_solve"]) {  // the Python reference
        const lab::ContactEquilibrium e = lab::contact_equilibrium(task, c["x_m"].get<double>());
        INFO("x_m = " << c["x_m"].get<double>() << " (Python reference)");
        CHECK_SCALAR_NEAR(e.x_s, c["x_s"].get<double>(), 1e-12);
        CHECK_SCALAR_NEAR(e.F_m, c["F_m"].get<double>(), 1e-12);
    }
    // felt stiffness = slope of F_m(x_m) in contact
    const lab::ContactEquilibrium e1 = lab::contact_equilibrium(task, 0.020), e2 = lab::contact_equilibrium(task, 0.021);
    const double slope = (e2.F_m - e1.F_m) / 0.001;
    CHECK_SCALAR_NEAR(e1.k_felt, slope, approx(slope, 1e-9));
    CHECK_SCALAR_NEAR(e1.k_felt, 30.048, 1e-3);
}

TEST_CASE("test_dry_run") {
    const json task = lab::load_task();
    const lab::DryRun d = lab::dry_run(task);
    const lab::PPSim s = lab::simulate_pp(task);
    const double peak = s.F_m.cwiseAbs().maxCoeff();
    CHECK_SCALAR_NEAR(d.peak_F_m, peak, approx(peak, 1e-9));
    CHECK_APPROX(d.peak_force, EXP()["dry_run"]["peak_force"].get<double>());
    CHECK_APPROX(d.peak_x, EXP()["dry_run"]["peak_x"].get<double>());
    CHECK(d.ok == true);
    CHECK(d.sat == 0);
    json hard = task;
    hard["pp"]["alpha"] = 1.0;  // a force scale of 4 would push the leader too hard
    hard["pp"]["beta"] = 4.0;
    CHECK(lab::dry_run(hard).ok == false);
}

// ------------------------------------------------------------------ Part B: identification and the watchdog
TEST_CASE("test_log_decrement_on_a_damped_cosine") {
    const VectorXd t = lab::arange(0, 1.5, 1e-3);
    const lab::LogDecrement g = lab::log_decrement(t, damped(t, 0.012, 0.03, 40.0, 0.0), 0.0);
    CHECK_SCALAR_NEAR(g.zeta, 0.03, approx(0.03, 0.02));
    CHECK_SCALAR_NEAR(g.wn, 40.0, approx(40.0, 0.01));
    CHECK(g.n_peaks == 6);
}

TEST_CASE("test_fit_free_decay_on_the_nominal_device") {
    // No friction: m_h within 1 %; b_h within 10 % with the K T / 2 correction, clearly too low without it.
    const json task = lab::load_task();
    const json& nom = task["device_nominal"];
    lab::Device dev = lab::device_from(nom);
    const lab::DecaySim r = lab::simulate_free_decay(dev, 100.0, 0.015, 0.5, 3.0);
    const lab::DecayFit f = lab::fit_free_decay(r.t, r.x, 100.0, nom["k_g"].get<double>(), 1e-3, 0.5, 1.5);
    const double m_h = nom["m_h"].get<double>(), b_h = nom["b_h"].get<double>();
    CHECK_SCALAR_NEAR(f.m_h, m_h, approx(m_h, 0.01));
    CHECK_SCALAR_NEAR(f.b_h, b_h, approx(b_h, 0.10));
    CHECK(f.b_h - 100.0 * 1e-3 / 2 < 0.85 * b_h);
    const VectorXd t = lab::arange(0, 1.5, 1e-3);
    const lab::DecayFit f2 = lab::fit_free_decay(t, damped(t), 0.0, 40.0 * 40.0 * 0.05, 1e-3, 0.0);  // exact cosine, K = 0
    CHECK_SCALAR_NEAR(f2.m_h, 0.05, approx(0.05, 1e-4));
    CHECK_SCALAR_NEAR(f2.zeta, 0.03, approx(0.03, 1e-4));
}

TEST_CASE("test_fit_free_decay_station_units") {
    const json task = lab::load_task();
    const json& dc = task["decay"];
    for (const char* unit : {"H07", "H08"}) {
        INFO("unit " << unit);
        const lab::Log L = lab::load_log(std::string("decay_") + unit + ".csv");
        const lab::DecayFit f =
            lab::fit_free_decay(L.t, L.meas, dc["K"].get<double>(), task["device_nominal"]["k_g"].get<double>(),
                                task["dt"].get<double>(), dc["t_release"].get<double>(), dc["t_fit_end"].get<double>());
        const json& e = EXP()[std::string("decay_") + unit]["fit"];
        CHECK_SCALAR_NEAR(f.m_h, e["m_h"].get<double>(), approx(e["m_h"].get<double>(), 1e-3));
        CHECK_SCALAR_NEAR(f.b_h, e["b_h"].get<double>(), approx(e["b_h"].get<double>(), 1e-2));
        const lab::LogDecrement g =
            lab::log_decrement(L.t, L.meas, dc["t_release"].get<double>(), dc["t_fit_end"].get<double>());
        CHECK_SCALAR_NEAR(g.wn, f.wn, approx(f.wn, 0.02));  // the hand estimate agrees with the fit
    }
}

TEST_CASE("test_watchdog_check") {
    const VectorXd t = lab::arange(0, 1.0, 1e-3);
    VectorXd x(t.size()), F(t.size());
    for (Index i = 0; i < t.size(); ++i) {
        x(i) = 0.045 * std::sin(lab::PI * t(i));  // out between 0.349 s and 0.651 s
        F(i) = t(i) >= 0.353 ? 0.0 : -50 * x(i);   // firmware cut 4 ms after the crossing, latched
    }
    lab::WatchdogCheck w = lab::watchdog_check(t, x, F, 0.040);
    CHECK_SCALAR_NEAR(w.t_cross, 0.349, 1e-9);
    CHECK_SCALAR_NEAR(w.delay, 0.004, 1e-9);
    CHECK(w.latched == true);
    VectorXd F2 = F;
    for (Index i = 0; i < t.size(); ++i)
        if (t(i) > 0.8) F2(i) = -50 * x(i);  // output came back: not latched
    CHECK(lab::watchdog_check(t, x, F2, 0.040).latched == false);
    const lab::Log L = lab::load_log("watchdog_log.csv");
    w = lab::watchdog_check(L.t, L.meas, L.cur, 0.040);
    CHECK(w.delay <= 0.010);
    CHECK(w.latched == true);
    CHECK_SCALAR_NEAR(w.t_cross, EXP()["watchdog"]["t_cross"].get<double>(), 1e-9);
}

// ------------------------------------------------------------------ Part C: the teleoperation trial
TEST_CASE("test_resample") {
    VectorXd t(4), td(6);
    t << 0.0, 0.0011, 0.0019, 0.0030;
    td << -0.001, 0.0, 0.001, 0.002, 0.003, 0.004;
    const VectorXd x = (2 * t.array() + 1).matrix();
    const VectorXd y = lab::resample(t, x, td);
    REQUIRE(y.size() == 6);
    CHECK(std::isnan(y(0)));
    CHECK(std::isnan(y(5)));
    CHECK_MAT_NEAR(y.segment(1, 4), (2 * td.segment(1, 4).array() + 1).matrix(), 1e-5);  // np.allclose
}

TEST_CASE("test_channel_energy_sign_and_value") {
    const Index n = 1001;
    VectorXd t(n);
    for (Index i = 0; i < n; ++i) t(i) = static_cast<double>(i) * (1.0 / 1000.0);  // np.linspace(0, 1, 1001)
    t(n - 1) = 1.0;
    const VectorXd x_m = 0.01 * t;  // leader moves 10 mm in 1 s
    const VectorXd zero = VectorXd::Zero(n);
    const VectorXd E = lab::channel_energy(t, VectorXd::Constant(n, 0.5), x_m, zero, zero);
    REQUIRE(E.size() == n);
    CHECK(E(0) == 0.0);
    CHECK_SCALAR_NEAR(E(n - 1), 0.5 * 0.01, approx(0.005, 1e-9));  // resisting force: energy INTO the channel
    const VectorXd E2 = lab::channel_energy(t, zero, x_m, VectorXd::Constant(n, 0.5), x_m);
    REQUIRE(E2.size() == n);
    CHECK_SCALAR_NEAR(E2(n - 1), -0.005, approx(0.005, 1e-9));  // follower drives its port: energy OUT
    const VectorXd E3 = lab::channel_energy(t, VectorXd::Constant(n, 0.25), x_m, VectorXd::Constant(n, 0.5), 0.5 * x_m);
    REQUIRE(E3.size() == n);
    CHECK(E3.cwiseAbs().maxCoeff() < 1e-12);  // beta = alpha = 0.5: power-conserving channel
}

TEST_CASE("test_trial_metrics") {
    const json task = lab::load_task();
    const lab::Log L = lab::load_log("trial_log.csv");
    const lab::TrialMetrics m = lab::trial_metrics(L, task);
    const json& e = EXP()["trial"];
    const std::vector<std::pair<const char*, double>> got = {
        {"rms_free", m.rms_free}, {"x_m_hold", m.x_m_hold}, {"delta_hold", m.delta_hold}, {"F_m_hold", m.F_m_hold},
        {"F_m_pred", m.F_m_pred}, {"k_felt", m.k_felt},     {"peak_F_m", m.peak_F_m},     {"peak_x", m.peak_x},
        {"E_min", m.E_min}};
    for (const auto& kv : got) {
        INFO(kv.first);
        const double ex = e[kv.first].get<double>();
        CHECK_SCALAR_NEAR(kv.second, ex, approx(ex, 2e-3, 1e-9));
    }
    CHECK(m.n_sat == 0);
    CHECK(m.ok == true);
    CHECK(std::abs(m.k_felt / lab::contact_equilibrium(task, m.x_m_hold).k_felt - 1) < 0.03);
    CHECK(m.E_min >= task["limits"]["E_min"].get<double>());
}

TEST_CASE("test_sim_vs_hw") {
    const json task = lab::load_task();
    const lab::Log L = lab::load_log("trial_log.csv");
    const lab::SimVsHw nom = lab::sim_vs_hw(L, task, lab::device_from(task["device_nominal"]));
    const double ex = EXP()["sim_vs_hw"]["nominal"]["rms"].get<double>();
    CHECK_SCALAR_NEAR(nom.rms, ex, approx(ex, 0.02));
    CHECK(nom.ok == true);
    const json& f = EXP()["decay_H08"]["fit"];
    const json ident = {{"m_h", f["m_h"]}, {"b_h", f["b_h"]}, {"k_g", task["device_nominal"]["k_g"]}};
    const lab::SimVsHw fr = lab::sim_vs_hw(L, task, lab::device_from(ident, {{"f_c", 0.008}}));
    CHECK(fr.rms < 0.5 * nom.rms);  // most of the residual is unmodelled friction
}

// ------------------------------------------------------------------ Part D: safe stop and the IMU
TEST_CASE("test_fade_time") {
    const VectorXd t = lab::arange(0, 1.0, 1e-3);
    VectorXd F(t.size());
    for (Index i = 0; i < t.size(); ++i)
        F(i) = t(i) < 0.5 ? 0.6 : std::max(0.6 * (1 - (t(i) - 0.5) / 0.05), 0.0);
    CHECK_SCALAR_NEAR(lab::fade_time(t, F, 0.5, 0.02), 0.049, 1.1e-3);
    CHECK(lab::fade_time(t, VectorXd::Constant(t.size(), 0.01), 0.5, 0.02) == 0.0);
    CHECK(lab::fade_time(t, VectorXd::Constant(t.size(), 0.5), 0.5, 0.02) == INF);
    const lab::Log L = lab::load_log("stop_log.csv");
    MatrixXd FF(L.t.size(), 2);
    FF << L.force, L.cur;
    const double task_stop = 2.5;
    const double ft = lab::fade_time(L.t, FF, task_stop, 0.02);
    CHECK_SCALAR_NEAR(ft, EXP()["stop"]["fade_time"].get<double>(), 1e-6);
    CHECK(ft <= 0.010);
}

TEST_CASE("test_imu_rest_stats") {
    const json& C = cases()["imu_rest_stats"];
    const VectorXd t = lab::arange(0, 20, 0.005);
    const VectorXd g = sbeg::as_vec(C["g"]);  // 0.03 + 0.008 * default_rng(1).standard_normal(len(t))
    REQUIRE(g.size() == t.size());
    const double th = 10.0 * lab::DEG;
    lab::ImuRestStats r = lab::imu_rest_stats(t, g, VectorXd::Constant(t.size(), 9.81 * std::cos(th)),
                                              VectorXd::Constant(t.size(), -9.81 * std::sin(th)), 10.0);
    double sum = 0.0;
    int cnt = 0;
    for (Index i = 0; i < t.size(); ++i)
        if (t(i) < 10) {
            sum += g(i);
            ++cnt;
        }
    CHECK_SCALAR_NEAR(r.bias, sum / cnt, 1e-12);
    CHECK_SCALAR_NEAR(r.noise, 0.008, approx(0.008, 0.05));
    CHECK_SCALAR_NEAR(r.tilt_deg, 10.0, 1e-9);
    const double drift = std::abs(r.bias) * 10.0 / lab::DEG;
    CHECK_SCALAR_NEAR(r.drift_deg, drift, approx(drift, 1e-12));
    const lab::Log L = lab::load_log("imu_rest.csv");
    r = lab::imu_rest_stats(L.t, L.aux.at("gyro"), L.aux.at("ax"), L.aux.at("ay"), 10.0);
    const json& e = EXP()["imu"];
    const std::vector<std::pair<const char*, double>> got = {{"bias", r.bias},
                                                             {"noise", r.noise},
                                                             {"drift_deg", r.drift_deg},
                                                             {"tilt_deg", r.tilt_deg},
                                                             {"residual_deg", r.residual_deg}};
    for (const auto& kv : got) {
        INFO(kv.first);
        const double ex = e[kv.first].get<double>();
        CHECK_SCALAR_NEAR(kv.second, ex, approx(ex, 1e-6, 1e-9));
    }
    CHECK(std::abs(r.residual_deg) < 0.1 * r.drift_deg);
}
