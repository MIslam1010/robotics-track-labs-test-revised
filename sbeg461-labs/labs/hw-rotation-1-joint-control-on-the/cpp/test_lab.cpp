// Auto-checks for SBEG 461 Hardware rotation 1 (Week 8) -- C++ track: joint control on the course arm -- pre-flight
// and dry run, the step response of the servo's PD loop, gravity check, gravity compensation and the station loop,
// sim-vs-hardware comparison and the safe stop.  Mirrors test_lab.py test by test (18 test cases for the 19 Python
// tests: the Python test of the provided check_command, the 10 deg joint-limit margin, is checked inside
// test_preflight).
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*dry_run*,*preflight*"      (run only some tests, like pytest -k)
// Ground truth: closed forms (second-order step response, Astrom-Murray Table 6.1, MR Eqs. 11.26-11.27), mrcpp (the
// C++ port of modern_robotics: RNEA), the Lab 5 PID code, synthetic signals with known answers, the synthetic station
// logs in ../data/, and cpp_cases.json, which holds the numbers of the Python reference, so both languages are
// checked against the same values.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using sbeg::as_mat;
using sbeg::as_vec;
using sbeg::json;

namespace {

const json& cases() {  // loaded once
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

const json& task() {
    static const json T = lab::load_task();
    return T;
}

const lab::Model& model() {
    static const lab::Model M = lab::course_arm_model(task()["payload_kg"].get<double>());
    return M;
}

VectorXd pose() { return as_vec(task()["tray_pose"]); }

const lab::Servo& nominal() {
    static const lab::Servo S = lab::nominal_servo(model(), as_vec(task()["tray_pose"]), task());
    return S;
}

/// One step of the step test: joint (0-based) and (t, cmd, meas) of that joint inside its window, and the measured
/// pose of all joints at the start of the window.
struct Step {
    int joint = 0;
    VectorXd t, cmd, meas, q_meas0;
};

/// The `steps` fixture of test_lab.py.
std::vector<Step> steps() {
    const lab::Log L = lab::load_log(lab::data_path("step_log.csv"));
    std::vector<Step> out;
    const json& st = task()["step_test"];
    for (std::size_t s = 0; s < st["joints"].size(); ++s) {
        const int j = st["joints"][s].get<int>() - 1;
        const lab::Log W = lab::log_window(L, st["windows_s"][s][0].get<double>(), st["windows_s"][s][1].get<double>());
        Step x;
        x.joint = j;
        x.t = W.t;
        x.cmd = W.cmd.col(j);
        x.meas = W.meas.col(j);
        x.q_meas0 = W.meas.row(0).transpose();
        out.push_back(x);
    }
    return out;
}

lab::StaticPoses statics() { return lab::average_approaches(lab::load_log(lab::data_path("static_log.csv"))); }

std::vector<int> fit_poses() { return task()["static_fit_poses"].get<std::vector<int>>(); }
std::vector<int> check_poses() { return task()["static_check_poses"].get<std::vector<int>>(); }

/// K (6 entries, inf except joints 2-4) from YOUR gravity_check of the static log.
VectorXd K_of() {
    const lab::GravityCheck gc = lab::gravity_check(statics(), model(), {1, 2, 3}, fit_poses(), check_poses());
    REQUIRE(gc.K.size() == 3);
    VectorXd K = VectorXd::Constant(6, std::numeric_limits<double>::infinity());
    K.segment(1, 3) = gc.K;
    return K;
}

/// The identified servo (fit + static stiffness per moving joint, nominal elsewhere) and K.
std::pair<lab::Servo, VectorXd> identified() {
    const VectorXd K = K_of();
    std::map<int, lab::JointServo> fits;
    for (const Step& s : steps()) {
        const lab::ServoFit r = lab::identify_servo(s.t, s.cmd, s.meas);
        const lab::ServoPD p = lab::servo_from_fit(r.wn, r.zeta, K(s.joint));
        lab::JointServo js;
        js.kp = p.kp;
        js.kd = p.kd;
        js.M_eff = p.M_eff;
        js.b = 0.0;
        js.delay = r.delay;
        fits[s.joint] = js;
    }
    return {lab::make_servo(nominal(), fits), K};
}

/// np.arange(start, stop, step).
VectorXd arange(double start, double stop, double step) {
    const Index n = static_cast<Index>(std::ceil((stop - start) / step - 1e-12));
    VectorXd x(n);
    for (Index i = 0; i < n; ++i) x(i) = start + static_cast<double>(i) * step;
    return x;
}

/// np.linspace(start, stop, num).
VectorXd linspace(double start, double stop, Index num) {
    VectorXd x(num);
    const double step = (stop - start) / static_cast<double>(num - 1);
    for (Index i = 0; i < num; ++i) x(i) = start + static_cast<double>(i) * step;
    x(num - 1) = stop;
    return x;
}

double rms(const VectorXd& e) { return std::sqrt(e.array().square().mean()); }

/// np.allclose(got, expected, atol=atol) (rtol = 1e-5, element by element), with a readable message.
template <typename A, typename B>
sbeg::NearResult allclose(const Eigen::MatrixBase<A>& got, const Eigen::MatrixBase<B>& expected, double atol = 1e-8) {
    const MatrixXd e = expected.template cast<double>();
    sbeg::NearResult r = sbeg::compare_near(got, expected, std::numeric_limits<double>::infinity());
    if (!r.ok) return r;  // shape mismatch or NaN
    MatrixXd g = got.template cast<double>();
    if (g.rows() != e.rows()) g.transposeInPlace();  // row vs column vector of equal length
    bool ok = true;
    for (Index i = 0; i < e.rows(); ++i)
        for (Index k = 0; k < e.cols(); ++k)
            ok = ok && std::abs(g(i, k) - e(i, k)) <= atol + 1e-5 * std::abs(e(i, k));
    r = sbeg::compare_near(got, expected, atol);
    r.ok = ok;
    return r;
}

#define CHECK_ALLCLOSE(got, expected, atol)                                                    \
    do {                                                                                       \
        const ::sbeg::NearResult ac_ = allclose((got), (expected), (atol));                    \
        CHECK_MESSAGE(ac_.ok, "np.allclose(" #got ", " #expected ")\n  " << ac_.msg);          \
    } while (false)

/// math.isclose(got, expected, abs_tol=abs_tol) (rel_tol = 1e-9).
#define CHECK_ISCLOSE(got, expected, abs_tol)                                                                  \
    do {                                                                                                       \
        const double ic_g_ = static_cast<double>(got), ic_e_ = static_cast<double>(expected);                  \
        const double ic_t_ = std::max(1e-9 * std::max(std::abs(ic_g_), std::abs(ic_e_)), (abs_tol));           \
        CHECK_SCALAR_NEAR(ic_g_, ic_e_, ic_t_);                                                                \
    } while (false)

VectorXd vec(std::initializer_list<double> v) {
    VectorXd x(static_cast<Index>(v.size()));
    Index i = 0;
    for (double e : v) x(i++) = e;
    return x;
}

std::string join(const std::vector<std::string>& v) {
    std::ostringstream s;
    s << "[";
    for (std::size_t i = 0; i < v.size(); ++i) s << (i ? ", " : "") << "\"" << v[i] << "\"";
    s << "]";
    return s.str();
}

json good_record() {
    return json{{"members_at_arm", json::array({"Salma", "Omar"})}, {"inducted", json::array({"Salma", "Omar", "Youssef"})},
                {"supervisor", "TA K."}, {"estop_tested", true}, {"estop_holder", "Omar"}, {"keepout_clear", true},
                {"workspace_ready", true}, {"first_run", true}, {"limits_ok", true}, {"margin_deg", 10.0},
                {"dry_run_speed_fraction", 0.25}, {"dry_run_speed_ratio", 0.742}, {"dry_run_torque_ratio", 0.538},
                {"current_limit_fraction", 0.6}, {"payload_kg", 0.1}, {"outer_ki", 3.0}, {"outer_clamp_deg", 5.0}};
}

}  // namespace

// ------------------------------------------------------------------ Part A: pre-flight, dry run
TEST_CASE("test_servo_model_conversion") {
    const VectorXd off = as_vec(task()["servo_zero_offset_deg"]);
    const VectorXd s = lab::model_to_servo(pose(), off);
    CHECK_ALLCLOSE(s, vec({1.5, -36.3775, 80.7141, -137.5099, -90.9544, 0.0}), 1e-4);
    CHECK_MAT_NEAR(s, as_vec(cases()["servo_model_conversion"]["servo"]), 1e-9);
    CHECK_ALLCLOSE(lab::servo_to_model(s, off), pose(), 1e-12);
}

TEST_CASE("test_dry_run_step") {
    const auto sc = lab::step_command(pose(), {2, 3}, {0.5, 2.0}, 3.0, 3.5);
    const lab::DryRun d = lab::dry_run(model(), sc.first, sc.second, nominal(), {1, 2}, 0.25, 0.6);
    REQUIRE(d.peak_qd.size() == 6);
    REQUIRE(d.speed_ratio.size() == 6);
    REQUIRE(d.torque_ratio.size() == 6);
    REQUIRE(d.meas.rows() == sc.first.size());
    // peak speed of a second-order step response (zeta = 0.9, wn = 25): closed form 9.851 rad/s per rad
    const double z = 0.9, wn = 25.0, wd = wn * std::sqrt(1 - z * z);
    const double ts = std::atan(std::sqrt(1 - z * z) / z) / wd;
    const double vpk = wn / std::sqrt(1 - z * z) * std::exp(-z * wn * ts) * std::sin(wd * ts) * 3.0 * lab::DEG;
    CHECK(std::abs(d.peak_qd(1) - vpk) < 0.01);
    CHECK(std::abs(d.peak_qd(2) - vpk) < 0.01);
    CHECK_ALLCLOSE(d.speed_ratio.segment(1, 2), vec({0.7419, 0.7402}), 2e-3);
    CHECK_ALLCLOSE(d.torque_ratio.segment(1, 2), vec({0.5375, 0.4584}), 2e-3);
    CHECK(d.ok);
    CHECK_MAT_NEAR(d.torque_ratio, as_vec(cases()["dry_run_step"]["down"]["torque_ratio"]), 1e-6);
    // starts at rest on the sag: Kp (cmd - q) = g(q)
    const VectorXd q0 = d.meas.row(0).transpose();
    const VectorXd g = mr::inverse_dynamics(q0, VectorXd::Zero(6), VectorXd::Zero(6), model().g, mr::Vector6d::Zero(),
                                            model().Mlist, model().Glist, model().Slist);
    CHECK_ALLCLOSE(nominal().kp.segment(1, 2).cwiseProduct((pose() - q0).segment(1, 2)), g.segment(1, 2), 1e-9);
    // the same step UPWARDS (against gravity) needs g + Kp A: 1.95 N m on joint 2 > 60 % of 3.0 N m
    const auto up = lab::step_command(pose(), {2, 3}, {0.5, 2.0}, -3.0, 3.5);
    const lab::DryRun du = lab::dry_run(model(), up.first, up.second, nominal());
    CHECK_FALSE(du.ok);
    REQUIRE(du.torque_ratio.size() == 6);
    CHECK_ALLCLOSE(du.torque_ratio.segment(1, 2), vec({1.0817, 1.0771}), 3e-3);
}

TEST_CASE("test_preflight") {
    const std::vector<std::string> got = lab::preflight(good_record());
    CHECK_MESSAGE(got.empty(), "failed rules: " << join(got));
    const json tmpl = sbeg::load_json(lab::data_path("session_record_template.json"));
    CHECK(lab::preflight(tmpl).size() == lab::PREFLIGHT_RULES.size());  // an empty record fails all
    struct Case {
        const char* key;
        json value;
        std::size_t rule;
    };
    const std::vector<Case> cs = {{"inducted", json::array({"Salma"}), 0}, {"supervisor", " ", 1},
                                  {"estop_tested", false, 2}, {"estop_holder", "Youssef", 3},
                                  {"keepout_clear", false, 4}, {"workspace_ready", false, 5},
                                  {"limits_ok", false, 6}, {"margin_deg", 5.0, 6},
                                  {"dry_run_speed_fraction", 0.5, 7}, {"dry_run_speed_ratio", 1.05, 7},
                                  {"dry_run_torque_ratio", 1.08, 8}, {"current_limit_fraction", 0.8, 9},
                                  {"payload_kg", 0.2, 10}, {"payload_kg", nullptr, 10}, {"outer_ki", 8.0, 11},
                                  {"outer_clamp_deg", 10.0, 11}, {"outer_ki", nullptr, 11}};
    for (const Case& c : cs) {
        json r = good_record();
        r[c.key] = c.value;
        const std::vector<std::string> f = lab::preflight(r);
        INFO(c.key << " = " << c.value.dump());
        CHECK_MESSAGE(f == std::vector<std::string>{lab::PREFLIGHT_RULES[c.rule]}, "failed rules: " << join(f));
    }
    json r = good_record();
    r["first_run"] = false;  // a repeat run may be dry-run at 50 %
    r["dry_run_speed_fraction"] = 0.5;
    const std::vector<std::string> f1 = lab::preflight(r);
    CHECK_MESSAGE(f1.empty(), "failed rules: " << join(f1));
    r = good_record();
    r["payload_kg"] = 0.125;  // the A4 limit itself; no outer loop
    r["outer_ki"] = 0.0;
    r["outer_clamp_deg"] = 0.0;
    const std::vector<std::string> f2 = lab::preflight(r);
    CHECK_MESSAGE(f2.empty(), "failed rules: " << join(f2));
    // the rule texts are those of the Python version
    CHECK(cases()["preflight"]["rules"].get<std::vector<std::string>>() == lab::PREFLIGHT_RULES);
    // rule A1 (provided check_command, Python test_joint_limit_margin_is_checklist_value): >= 10 deg inside the stops
    CHECK(lab::SOFT_LIMIT_MARGIN_DEG == 10.0);
    const VectorXd soft = (model().limits.col(1) / lab::DEG).array() - lab::SOFT_LIMIT_MARGIN_DEG;
    CHECK_ALLCLOSE(soft, vec({155, 155, 155, 155, 155, 165}), 1e-9);
    const VectorXd t = arange(0, 1.02, 0.02);
    for (const auto& dc : std::vector<std::pair<double, bool>>{{150.0, true}, {157.0, false}}) {
        MatrixXd q = MatrixXd::Zero(t.size(), 6);
        q.col(0).setConstant(dc.first * lab::DEG);
        INFO("joint 1 at " << dc.first << " deg");
        CHECK(lab::check_command(t, q, model(), 0.25).within_limits == dc.second);
    }
}

// ------------------------------------------------------------------ Part B: the servo's step response
TEST_CASE("test_zeta_wn_from_step_closed_form") {
    const double zeta = 0.6, wn = 20.0;
    const double Mp = std::exp(-lab::PI * zeta / std::sqrt(1 - zeta * zeta));
    const double tp = lab::PI / (wn * std::sqrt(1 - zeta * zeta));
    const auto zw = lab::zeta_wn_from_step(Mp, tp);
    CHECK_ISCLOSE(zw.first, 0.6, 1e-12);
    CHECK_ISCLOSE(zw.second, 20.0, 1e-9);
    CHECK_ISCLOSE(lab::zeta_wn_from_step(0.16, 0.2).first, 0.5038, 1e-4);  // A-M Table 6.1: 16 % <-> 0.5
}

TEST_CASE("test_step_features_synthetic") {
    const VectorXd t = arange(0, 2.0, 0.001);
    VectorXd c(t.size());
    for (Index i = 0; i < t.size(); ++i) c(i) = t(i) >= 0.5 ? 0.05 : 0.0;
    const VectorXd y = 0.05 * lab::step_response((t.array() - 0.53).matrix(), 20.0, 0.6);
    const lab::StepFeatures f = lab::step_features(t, c, y);
    CHECK_ISCLOSE(f.t_step, 0.5, 1e-9);
    CHECK_ISCLOSE(f.A, 0.05, 0.0);
    CHECK_ISCLOSE(f.overshoot, std::exp(-0.6 * lab::PI / 0.8), 2e-4);
    CHECK_ISCLOSE(f.t_peak_abs - 0.53, lab::PI / 16, 2e-3);
    CHECK(f.t_move > 0.53);
    CHECK(f.t_move < 0.53 + 0.06);
}

TEST_CASE("test_identify_servo_noise_free") {
    const VectorXd t = arange(0, 1.5, 0.02);
    VectorXd c(t.size());
    for (Index i = 0; i < t.size(); ++i) c(i) = t(i) >= 0.4 ? 0.06 : 0.0;
    const VectorXd y = (0.01 + 0.06 * lab::step_response((t.array() - 0.44).matrix(), 18.0, 0.5).array()).matrix();
    const lab::ServoFit r = lab::identify_servo(t, c, y);
    INFO("wn = " << r.wn << ", zeta = " << r.zeta << ", delay = " << r.delay);
    CHECK(std::abs(r.wn - 18.0) < 0.05);
    CHECK(std::abs(r.zeta - 0.5) < 0.005);
    CHECK(std::abs(r.delay - 0.04) < 0.001);
    CHECK(std::abs(r.y0 - 0.01) < 1e-6);
    CHECK(std::abs(r.A_fit - 0.06) < 1e-4);
    CHECK(r.rms < 1e-5);
}

TEST_CASE("test_identify_servo_station_log") {
    const std::vector<Step> S = steps();
    REQUIRE(S.size() == 2);
    const double expected[2][3] = {{21.36, 0.669, 0.0311}, {20.90, 0.658, 0.0283}};
    for (std::size_t s = 0; s < 2; ++s) {
        const lab::ServoFit r = lab::identify_servo(S[s].t, S[s].cmd, S[s].meas);
        INFO("joint " << s + 2 << ": wn = " << r.wn << ", zeta = " << r.zeta << ", delay = " << r.delay
                      << ", rms = " << r.rms);
        CHECK(std::abs(r.wn - expected[s][0]) < 0.3);
        CHECK(std::abs(r.zeta - expected[s][1]) < 0.01);
        CHECK(std::abs(r.delay - expected[s][2]) < 0.002);
        CHECK(r.rms < 1e-3);
    }
}

// ------------------------------------------------------------------ Part C: static poses and the PD gains
TEST_CASE("test_torque_constant") {
    const VectorXd i = vec({0.2, 0.4, 0.6});
    CHECK_ISCLOSE(lab::torque_constant(i, 1.25 * i), 1.25, 0.0);
    CHECK_ISCLOSE(lab::torque_constant(vec({1.0, 2.0}), vec({1.0, 3.0})), 7.0 / 5.0, 0.0);
}

TEST_CASE("test_loop_stiffness") {
    const VectorXd tau = vec({-0.9, -0.6, -0.3});
    CHECK_ISCLOSE(lab::loop_stiffness(tau / 30.0, tau), 30.0, 0.0);
    CHECK_ISCLOSE(lab::loop_stiffness(vec({0.01, 0.03}), vec({0.1, 0.2})), 0.05 / 0.007, 0.0);
}

TEST_CASE("test_gravity_check_station_log") {
    const lab::StaticPoses st = statics();
    const lab::GravityCheck gc = lab::gravity_check(st, model(), {1, 2, 3}, fit_poses(), check_poses());
    REQUIRE(gc.g.rows() == st.meas.rows());
    REQUIRE(gc.g.cols() == 6);
    MatrixXd g(st.meas.rows(), 6);
    for (Index p = 0; p < st.meas.rows(); ++p)
        g.row(p) = mr::inverse_dynamics(st.meas.row(p).transpose(), VectorXd::Zero(6), VectorXd::Zero(6), model().g,
                                        mr::Vector6d::Zero(), model().Mlist, model().Glist, model().Slist)
                       .transpose();
    CHECK_ALLCLOSE(gc.g, g, 1e-10);
    CHECK_ALLCLOSE(gc.k, vec({1.1823, 1.1750, 0.4443}), 2e-3);
    CHECK_ALLCLOSE(gc.K, vec({14.214, 7.460, 1.976}), 0.03);
    REQUIRE(gc.nrms.size() == 3);
    CHECK(gc.nrms.maxCoeff() < 0.05);
    CHECK_MAT_NEAR(gc.K, as_vec(cases()["gravity_check_station_log"]["K"]), 1e-9);
}

TEST_CASE("test_servo_from_fit") {
    const double M = 0.035, b = 0.02, zeta = 0.6, wn = 20.0;
    const auto g = lab::pd_gains(M, b, zeta, wn);  // Lab 5 forwards ...
    const lab::ServoPD s = lab::servo_from_fit(wn, zeta, g.first, b, 0.015);  // ... and backwards
    CHECK_ISCLOSE(s.kp, g.first, 0.0);
    CHECK_ISCLOSE(s.kd, g.second, 1e-12);
    CHECK_ISCLOSE(s.M_eff, M, 1e-12);
    REQUIRE(s.armature.has_value());
    CHECK_ISCLOSE(*s.armature, 0.020, 1e-12);
    CHECK_FALSE(lab::servo_from_fit(wn, zeta, g.first).armature.has_value());
    // station log, joint 2: K = 14.21 N m/rad, wn = 21.36 rad/s -> M_eff = 0.0312 kg m^2, about twice the link M22
    const lab::ServoPD s2 = lab::servo_from_fit(21.357, 0.669, 14.214, 0.0, 0.01546);
    CHECK(std::abs(s2.M_eff - 0.03116) < 2e-4);
    CHECK(std::abs(s2.kd - 0.8905) < 3e-3);
    REQUIRE(s2.armature.has_value());
    CHECK(std::abs(*s2.armature - 0.0157) < 3e-4);
    CHECK_ISCLOSE(s2.M_eff, cases()["servo_from_fit"]["M_eff"].get<double>(), 1e-12);
}

TEST_CASE("test_predict_step") {
    // (a) noise-free: the Lab 5 single-joint prediction equals the station-loop simulator (same PD, same delay)
    const lab::Servo sv = nominal();
    const auto sc = lab::step_command(pose(), {2}, {0.3}, 3.0, 1.2);
    lab::ServoArm arm(model(), sv, {1}, lab::DT, lab::H, 0.0);
    const lab::ServoRun ar = arm.run(sc.first, sc.second);
    const lab::Plant pl = lab::joint_plant(model(), pose(), 1, sv.M_eff(1), 0.0);
    const VectorXd p = lab::predict_step(pl, sv.kp(1), sv.kd(1), sc.first, sc.second.col(1), sv.delay);
    REQUIRE(p.size() == sc.first.size());
    CHECK((p - ar.meas.col(1)).cwiseAbs().maxCoeff() < 5e-4);
    CHECK_ISCLOSE(p(0), sc.second(0, 1) - lab::pd_gravity_error(pl, sv.kp(1), sc.second(0, 1)), 1e-12);
    // (b) station log: identified PD loop (other joints locked at their MEASURED angles) vs nominal
    const VectorXd K = K_of();
    for (const Step& s : steps()) {
        const lab::ServoFit r = lab::identify_servo(s.t, s.cmd, s.meas);
        const lab::ServoPD sp = lab::servo_from_fit(r.wn, r.zeta, K(s.joint));
        const double lim = 0.6 * model().tau_max(s.joint);
        const VectorXd pi = lab::predict_step(lab::joint_plant(model(), s.q_meas0, s.joint, sp.M_eff, 0.0, lim), sp.kp,
                                              sp.kd, s.t, s.cmd, r.delay);
        const VectorXd pn = lab::predict_step(lab::joint_plant(model(), s.q_meas0, s.joint, nominal().M_eff(s.joint), 0.0, lim),
                                              nominal().kp(s.joint), nominal().kd(s.joint), s.t, s.cmd, nominal().delay);
        REQUIRE(pi.size() == s.meas.size());
        REQUIRE(pn.size() == s.meas.size());
        INFO("joint " << s.joint + 1 << ": rms identified " << rms(pi - s.meas) << ", nominal " << rms(pn - s.meas));
        CHECK(rms(pi - s.meas) < 1.2e-3);
        CHECK(rms(pn - s.meas) > 0.01);
    }
}

// ------------------------------------------------------------------ Part D: compensation, station loop, logs
TEST_CASE("test_gravity_comp_command") {
    const VectorXd K = vec({1.0, 14.214, 7.460, 1.976, 1.0, 1.0});
    const VectorXd q = lab::gravity_comp_command(model(), pose(), K);
    const VectorXd g = lab::gravity_torques(model(), pose());
    REQUIRE(q.size() == 6);
    CHECK_ALLCLOSE(q - pose(), vec({0, g(1) / K(1), g(2) / K(2), 0, 0, 0}), 1e-12);
    CHECK_ALLCLOSE(((q - pose()) / lab::DEG).segment(1, 2), vec({-3.2075, -2.7598}), 1e-3);
    // with the exact stiffness, the feedforward removes the sag of a PD servo exactly
    lab::ServoArm arm(model(), nominal(), {1, 2}, lab::DT, lab::H, 0.0);
    arm.reset(pose());
    const lab::HoldRun r = lab::station_hold(arm, pose(), 101, "ff", model(), nominal().kp);
    REQUIRE(r.meas.rows() == 101);
    CHECK((r.meas.row(100).transpose() - pose()).cwiseAbs().maxCoeff() < 1e-6);
    arm.reset(pose());
    const lab::HoldRun r0 = lab::station_hold(arm, pose(), 101, "none", model(), nominal().kp);
    REQUIRE(r0.cmd.rows() == 101);
    CHECK_ALLCLOSE(r0.cmd, pose().transpose().replicate(101, 1), 1e-12);
    CHECK((r0.meas.row(100).transpose() - pose()).cwiseAbs().maxCoeff() > 0.02);
}

TEST_CASE("test_station_hold_replays_log") {
    // your station loop, fed the logged measurements, must send exactly the logged commands
    const VectorXd K = K_of();
    const json& ht = task()["hold_test"];
    const lab::Log H = lab::load_log(lab::data_path("hold_log.csv"));
    const std::vector<std::string> modes = ht["modes"].get<std::vector<std::string>>();
    for (std::size_t mi = 0; mi < modes.size(); ++mi) {
        const lab::Log W = lab::log_window(H, 0.0, ht["duration_s"].get<double>(), static_cast<double>(mi));
        lab::ReplayIO io(W);
        const lab::HoldRun r = lab::station_hold(io, pose(), static_cast<int>(W.t.size()), modes[mi], model(), K,
                                                 ht["ki"].get<double>(), ht["clamp_deg"].get<double>() * lab::DEG,
                                                 ht["t_on_s"].get<double>(), ht["t_ramp_s"].get<double>());
        INFO("mode " << modes[mi]);
        CHECK_ALLCLOSE(r.t, W.t, 1e-9);
        CHECK_ALLCLOSE(r.cmd, W.cmd, 1e-6);
        CHECK_ALLCLOSE(r.cur, W.cur, 1e-12);
    }
    const lab::Log W = lab::log_window(H, 0.0, 1.0, 0.0);
    lab::ReplayIO io(W);
    CHECK_THROWS_AS(lab::station_hold(io, pose(), 5, "pid", model(), K), std::invalid_argument);
}

TEST_CASE("test_station_hold_modes") {
    const json& ht = task()["hold_test"];
    const lab::Log H = lab::load_log(lab::data_path("hold_log.csv"));
    const auto idk = identified();
    const std::vector<std::string> modes = ht["modes"].get<std::vector<std::string>>();
    const MatrixXd expected = as_mat(cases()["station_hold_modes"]["err_deg"]);
    const double T = ht["duration_s"].get<double>();
    for (std::size_t mi = 0; mi < modes.size(); ++mi) {
        INFO("mode " << modes[mi]);
        const lab::Log W = lab::log_window(H, 0.0, T, static_cast<double>(mi));
        VectorXd err = VectorXd::Zero(2);  // logged steady error over the last 0.5 s (deg)
        int cnt = 0;
        for (Index i = 0; i < W.t.size(); ++i)
            if (W.t(i) >= T - 0.5 - 1e-9) {
                err += ((pose() - W.meas.row(i).transpose()) / lab::DEG).segment(1, 2);
                ++cnt;
            }
        err /= cnt;
        CHECK_ALLCLOSE(err, expected.row(static_cast<Index>(mi)), 0.01);
        lab::ServoArm arm(model(), idk.first);  // simulation fallback of the same test
        arm.reset(pose());
        const lab::HoldRun r = lab::station_hold(arm, pose(), static_cast<int>(W.t.size()), modes[mi], model(), idk.second,
                                                 ht["ki"].get<double>(), ht["clamp_deg"].get<double>() * lab::DEG,
                                                 ht["t_on_s"].get<double>(), ht["t_ramp_s"].get<double>());
        REQUIRE(r.meas.rows() == W.t.size());
        const lab::RunComparison cr = lab::compare_runs(W.t, W.cmd, W.meas, r.meas);
        REQUIRE(cr.rms_sim_meas.size() == 6);
        CHECK(cr.rms_sim_meas.segment(1, 2).maxCoeff() < 2e-3);
        CHECK((r.cmd.rowwise() - pose().transpose()).cwiseAbs().maxCoeff() < ht["clamp_deg"].get<double>() * lab::DEG + 1e-9);
    }
}

TEST_CASE("test_compare_runs_synthetic") {
    const VectorXd t = linspace(0, 2, 101);
    MatrixXd c(t.size(), 2);
    c.col(0) = t.array().sin().matrix();
    c.col(1) = t.array().cos().matrix();
    const Eigen::RowVector2d dm(0.01, -0.02), ds(0.01, 0.0);
    const lab::RunComparison out = lab::compare_runs(t, c, c.rowwise() + dm, c.rowwise() + ds);
    CHECK_ALLCLOSE(out.rms_cmd_meas, vec({0.01, 0.02}), 1e-8);
    CHECK_ALLCLOSE(out.max_cmd_meas, vec({0.01, 0.02}), 1e-8);
    CHECK_ALLCLOSE(out.rms_sim_meas, vec({0.0, 0.02}), 1e-12);
    MatrixXd lagged(t.size(), 2);
    lagged.col(0) = (t.array() - 0.08).sin().matrix();
    lagged.col(1) = (t.array() - 0.08).cos().matrix();
    CHECK_ISCLOSE(lab::compare_runs(t, c, lagged, c).lag, 0.08, 1e-9);
}

TEST_CASE("test_move_log_identified_beats_nominal") {
    const lab::Log M = lab::load_log(lab::data_path("move_log.csv"));
    const auto idk = identified();
    lab::ServoArm an(model(), nominal());
    const lab::RunComparison nom = lab::compare_runs(M.t, M.cmd, M.meas, an.run(M.t, M.cmd).meas);
    lab::ServoArm ai(model(), idk.first);
    const lab::ServoRun res = ai.run(M.t, M.cmd);
    const lab::RunComparison idm = lab::compare_runs(M.t, M.cmd, M.meas, res.meas);
    REQUIRE(nom.rms_cmd_meas.size() == 6);
    REQUIRE(idm.rms_sim_meas.size() == 6);
    CHECK_ISCLOSE(nom.lag, 0.10, 1e-9);
    CHECK_ALLCLOSE(nom.rms_cmd_meas.segment(1, 2), vec({0.0551, 0.0486}), 5e-4);
    CHECK_ALLCLOSE(nom.rms_sim_meas.segment(1, 2), vec({0.0182, 0.0156}), 5e-4);
    INFO("identified rms_sim_meas = " << idm.rms_sim_meas.transpose());
    CHECK(idm.rms_sim_meas.segment(1, 2).maxCoeff() < 1.5e-3);
    CHECK((idm.rms_sim_meas.segment(1, 2).array() < nom.rms_sim_meas.segment(1, 2).array() / 10).all());
    const lab::GravityCheck gc = lab::gravity_check(statics(), model(), {1, 2, 3}, fit_poses(), check_poses());
    REQUIRE(gc.k.size() == 3);
    for (int col = 0; col < 2; ++col) {  // predicted servo torque vs k * current during the move
        const int j = col + 1;
        CHECK(lab::normalised_rms(res.tau.col(j), gc.k(col) * M.cur.col(j)) < 0.05);
    }
}

TEST_CASE("test_stop_time") {
    const VectorXd t = arange(0, 2, 0.02);
    MatrixXd q(t.size(), 1);
    for (Index i = 0; i < t.size(); ++i) q(i, 0) = t(i) < 1.0 ? 0.5 * t(i) : 0.5;  // stops dead at t = 1
    // window 0.1 s -> +-2 samples: the last centre sample that still sees motion is t = 1.02 s
    CHECK_ISCLOSE(lab::stop_time(t, q, 0.9), 0.14, 1e-9);
    const lab::Log P = lab::load_log(lab::data_path("stop_log.csv"));
    const double ts = lab::stop_time(P.t, P.meas, task()["stop_test"]["t_trigger_s"].get<double>());
    CHECK_ISCLOSE(ts, 0.18, 1e-9);
    CHECK(ts < task()["stop_test"]["pass_s"].get<double>());
    const double never = lab::stop_time(t, (0.5 * t).eval(), 0.5);
    CHECK(never == std::numeric_limits<double>::infinity());
}
