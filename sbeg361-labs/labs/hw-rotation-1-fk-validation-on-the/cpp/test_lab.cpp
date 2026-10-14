// Auto-checks for SBEG 361 Hardware rotation 1 (Week 8) -- FK validation on the course arm -- C++ track.
// Mirrors test_lab.py test by test (the 13 tests of your functions; the 4 checks of the given Python files stay in
// the Python suite).
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*check_plan*,*run_logged*"      (only the safety gate, like pytest -k)
// Ground truth: mrcpp (the C++ port of modern_robotics: fkin_space, jacobian_space), independent checks in this file,
// the emulator SimArm (arm_station.cpp), the reference session of emulated arm A5 (../data/) and cpp_cases.json,
// which holds the numbers of the Python reference, so both languages are checked against the same values.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using sbeg::as;
using sbeg::as_mat;
using sbeg::as_vec;
using sbeg::json;

namespace {
constexpr double PI = 3.14159265358979323846;

const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

std::string data(const std::string& f) { return sbeg::lab_path("../data/" + f); }

const json& plan() {
    static const json P = lab::load_plan(data("validation_plan.json"));
    return P;
}
const VectorXd& off() {
    static const VectorXd O = lab::load_offsets(data("servo_offsets_A5.json"));
    return O;
}
const lab::CourseArm& arm() {
    static const lab::CourseArm A = lab::course_arm_reference();
    return A;
}
Vector3d p_tool() { return as<Vector3d>(plan()["p_tool_m"]); }
MatrixXd soft() {
    const MatrixXd hw = as_mat(plan()["hw_limits_model_deg"]);
    MatrixXd s(hw.rows(), 2);
    s.col(0) = hw.col(0).array() + 10.0;
    s.col(1) = hw.col(1).array() - 10.0;
    return s;
}
const lab::Log& ref_log() {
    static const lab::Log L = lab::read_log(data("session_A5_log.csv"));
    return L;
}
const lab::Tips& tips() {
    static const lab::Tips T = lab::read_tips(data("session_A5_tips.csv"));
    return T;
}
VectorXd pose_servo(const std::string& name) { return as_vec(plan()["poses_model_deg"][name]) + off(); }

/// Independent tool-point FK with mrcpp (mm), one row per row of q_servo_deg.
MatrixXd fk_tool_mm(const MatrixXd& q_servo_deg, const VectorXd& offsets) {
    MatrixXd P(q_servo_deg.rows(), 3);
    const Vector3d pt = p_tool();
    for (Eigen::Index k = 0; k < q_servo_deg.rows(); ++k) {
        const VectorXd q = (q_servo_deg.row(k).transpose() - offsets) * (PI / 180.0);
        const Matrix4d T = mr::fkin_space(arm().M, arm().Slist, q);
        P.row(k) = (1e3 * (T.topLeftCorner<3, 3>() * pt + T.topRightCorner<3, 1>())).transpose();
    }
    return P;
}
MatrixXd fk_tool_mm(const MatrixXd& q_servo_deg) { return fk_tool_mm(q_servo_deg, off()); }

/// np.allclose / np.testing.assert_allclose semantics: |got - exp| <= atol + rtol * |exp| entry by entry.
sbeg::NearResult allclose(const MatrixXd& got, const MatrixXd& exp, double atol, double rtol) {
    MatrixXd g = got, e = exp;
    if ((g.rows() == 1 || g.cols() == 1) && (e.rows() == 1 || e.cols() == 1) && g.size() == e.size()) {
        g = Eigen::Map<VectorXd>(g.data(), g.size()).eval();
        e = Eigen::Map<VectorXd>(e.data(), e.size()).eval();
    }
    sbeg::NearResult r = sbeg::compare_near(g, e, 0.0);  // message with both values
    if (g.rows() != e.rows() || g.cols() != e.cols() || !g.allFinite()) return r;
    r.ok = ((g - e).array().abs() <= atol + rtol * e.array().abs()).all();
    r.msg += "\n  (np.allclose: atol " + std::to_string(atol) + ", rtol " + std::to_string(rtol) + ")";
    return r;
}
/// np.isclose(got, exp, atol) for scalars.
bool isclose(double got, double exp, double atol = 1e-8) {
    return std::isfinite(got) && std::abs(got - exp) <= atol + 1e-5 * std::abs(exp);
}
VectorXd vec(std::initializer_list<double> v) {
    VectorXd x(static_cast<Eigen::Index>(v.size()));
    Eigen::Index i = 0;
    for (double e : v) x(i++) = e;
    return x;
}
}  // namespace

// np.allclose(got, expected, atol=...) (rtol 1e-5 as NumPy)
#define CHECK_ALLCLOSE(got, expected, atol)                                                         \
    do {                                                                                            \
        const ::sbeg::NearResult sbeg_ac_ = allclose((got), (expected), (atol), 1e-5);             \
        CHECK_MESSAGE(sbeg_ac_.ok, "CHECK_ALLCLOSE(" #got ", " #expected ")\n  " << sbeg_ac_.msg); \
    } while (false)
// np.isclose(got, expected, atol=...) for scalars
#define CHECK_ISCLOSE(got, expected, atol)                                                               \
    do {                                                                                                 \
        const double sbeg_g_ = static_cast<double>(got), sbeg_e_ = static_cast<double>(expected);         \
        CHECK_MESSAGE(isclose(sbeg_g_, sbeg_e_, (atol)),                                                 \
                      "CHECK_ISCLOSE(" #got ", " #expected "): got " << sbeg_g_ << ", expected " << sbeg_e_); \
    } while (false)

// ------------------------------------------------------------------ Step 1: plan check (SAFETY GATE)
TEST_CASE("test_check_plan_margins_and_speeds") {
    const Eigen::Index N = 101;
    const VectorXd t = VectorXd::LinSpaced(N, 0.0, 2.0);
    MatrixXd q = MatrixXd::Zero(N, 6).rowwise() + off().transpose();
    for (Eigen::Index k = 0; k < N; ++k) {
        q(k, 0) = off()(0) + 60 * std::sin(PI * t(k) / 2);   // peak 30*pi = 94.2 deg/s; margin 155-60 = 95
        q(k, 3) = off()(3) - 140 + 5 * t(k);                 // margin -140 - (-155) = 15 at t = 0
    }
    const lab::PlanCheck r = lab::check_plan(t, q, off(), soft(), 160.0, 0.25, 10.0);
    CHECK_ISCLOSE(r.speed_limit, 40.0, 1e-8);
    const json& c = cases()["check_plan_margins_and_speeds"];
    CHECK_ALLCLOSE(r.margin_deg, as_vec(c["margin_deg"]), 1e-8);
    CHECK_ISCLOSE(r.margin_deg(0), 95.0, 1e-8);
    CHECK_ISCLOSE(r.margin_deg(3), 15.0, 1e-8);
    CHECK_ALLCLOSE(r.peak_speed, as_vec(c["peak_speed"]), 1e-8);   // np.gradient of the Python reference
    CHECK(r.ok_limits);
    CHECK_FALSE(r.ok_speed);
    CHECK_FALSE(r.ok);
}

TEST_CASE("test_check_plan_rejects_limit_margin") {
    const Eigen::Index N = 501;
    const VectorXd t = VectorXd::LinSpaced(N, 0.0, 10.0);
    MatrixXd q = MatrixXd::Zero(N, 6).rowwise() + off().transpose();
    for (Eigen::Index k = 0; k < N; ++k) q(k, 1) = off()(1) + 150 * t(k) / 10;   // ends 5 deg inside the limit
    const lab::PlanCheck r = lab::check_plan(t, q, off(), soft(), 160.0, 0.25, 10.0);
    REQUIRE(r.margin_deg.size() == 6);
    CHECK_ISCLOSE(r.margin_deg(1), 5.0, 1e-8);
    CHECK_FALSE(r.ok_limits);
    CHECK(r.ok_speed);
    CHECK_FALSE(r.ok);
}

TEST_CASE("test_check_plan_on_validation_plan_and_reference_log") {
    std::vector<std::string> names;
    for (const auto& k : plan()["trajectory"]) names.push_back(k.get<std::string>());
    MatrixXd via(static_cast<Eigen::Index>(names.size()), 6);
    for (std::size_t i = 0; i < names.size(); ++i) via.row(static_cast<Eigen::Index>(i)) = pose_servo(names[i]).transpose();
    const lab::Trajectory tr = lab::smooth_segments(via, plan()["T_seg_s"].get<double>(),
                                                    plan()["rate_hz"].get<double>(), plan()["t_hold_s"].get<double>());
    const json& c = cases()["check_plan_on_validation_plan_and_reference_log"];
    REQUIRE(tr.t.size() == c["n"].get<Eigen::Index>());
    const lab::PlanCheck r = lab::check_plan(tr.t, tr.q, off(), soft(), 160.0, 0.25, 10.0);
    REQUIRE(r.peak_speed.size() == 6);
    CHECK(r.ok);
    CHECK_ISCLOSE(r.peak_speed(0), 1.5 * 75 / 3.5, 0.05);   // 32.14 deg/s
    CHECK_ALLCLOSE(r.peak_speed, as_vec(c["plan_peak_speed"]), 1e-6);
    CHECK_ALLCLOSE(r.margin_deg, as_vec(c["plan_margin_deg"]), 1e-6);
    const lab::PlanCheck m = lab::check_plan(ref_log().t, ref_log().meas, off(), soft(), 160.0, 0.25, 10.0);  // HW-4, HW-5
    CHECK(m.ok);
    CHECK_ISCLOSE(m.margin_deg.minCoeff(), 35.03, 0.02);
    CHECK_ISCLOSE(m.peak_speed.maxCoeff(), 37.35, 0.02);
}

// ------------------------------------------------------------------ Step 2: logged run (SAFETY GATE)
namespace {
/// Records the calls in order; meas = last command + 0.1 deg.
class FakeArm : public lab::Arm {
public:
    explicit FakeArm(int fail_at = -1) : fail_at_(fail_at) {}
    void wait_until(double t) override { calls.push_back({"wait", std::round(t * 1e6) / 1e6}); }
    bool send(const VectorXd& q) override {
        if (stopped) throw std::runtime_error("send after stop");
        ++k_;
        if (k_ == fail_at_) throw lab::SafetyViolation("test violation");
        calls.push_back({"send", 0.0});
        q_ = q;
        return true;
    }
    VectorXd read() override {
        calls.push_back({"read", 0.0});
        return q_.array() + 0.1;
    }
    void stop() override {
        calls.push_back({"stop", 0.0});
        stopped = true;
    }
    int count(const std::string& what) const {
        int n = 0;
        for (const auto& c : calls) n += c.first == what ? 1 : 0;
        return n;
    }
    long index(const std::string& what, double t = -1.0) const {   // first position of a call (-1 if absent)
        for (std::size_t i = 0; i < calls.size(); ++i)
            if (calls[i].first == what && (t < 0 || std::abs(calls[i].second - t) < 1e-9)) return static_cast<long>(i);
        return -1;
    }
    std::vector<std::pair<std::string, double>> calls;
    bool stopped = false;

private:
    VectorXd q_ = VectorXd::Zero(6);
    int k_ = 0, fail_at_;
};

MatrixXd ramp(Eigen::Index n) {   // np.outer(np.arange(n), np.ones(6))
    MatrixXd q(n, 6);
    for (Eigen::Index k = 0; k < n; ++k) q.row(k).setConstant(static_cast<double>(k));
    return q;
}
}  // namespace

TEST_CASE("test_run_logged_order_and_stop_with_fake_arm") {
    const VectorXd t = VectorXd::LinSpaced(10, 0.0, 9.0) * 0.1;
    const MatrixXd q = ramp(10);
    FakeArm a;
    const lab::Log log = lab::run_logged(a, t, q, 0.5);
    VectorXd waits(a.count("wait"));
    Eigen::Index i = 0;
    for (const auto& c : a.calls)
        if (c.first == "wait") waits(i++) = c.second;
    CHECK_ALLCLOSE(waits, t, 1e-8);
    CHECK(a.count("stop") == 1);
    CHECK(a.count("send") == 5);
    CHECK(a.index("stop") > a.index("wait", 0.5));
    CHECK_MAT_NEAR(log.stop, vec({0, 0, 0, 0, 0, 1, 1, 1, 1, 1}), 0.0);
    REQUIRE(log.cmd.rows() == 10);
    REQUIRE(log.cmd.cols() == 6);
    CHECK_ALLCLOSE(log.cmd.topRows(5), q.topRows(5), 1e-8);
    CHECK_ALLCLOSE(log.cmd.bottomRows(5), MatrixXd(q.row(4).replicate(5, 1)), 1e-8);   // held after stop
    CHECK_ALLCLOSE(log.meas, MatrixXd(log.cmd.array() + 0.1), 1e-8);
    CHECK_FALSE(log.aborted.has_value());
    CHECK_ALLCLOSE(log.t, t, 1e-8);
}

TEST_CASE("test_run_logged_aborts_on_safety_violation") {
    const VectorXd t = VectorXd::LinSpaced(6, 0.0, 5.0) * 0.1;
    const MatrixXd q = ramp(6);
    FakeArm a(3);
    const lab::Log log = lab::run_logged(a, t, q);
    REQUIRE(log.aborted.has_value());
    CHECK(log.aborted->find("test violation") != std::string::npos);
    CHECK_MAT_NEAR(log.stop, vec({0, 0, 1, 1, 1, 1}), 0.0);
    CHECK(a.count("stop") == 1);
    REQUIRE(log.cmd.rows() == 6);
    CHECK_ALLCLOSE(log.cmd.bottomRows(4), MatrixXd(q.row(1).replicate(4, 1)), 1e-8);
}

TEST_CASE("test_run_logged_on_emulator_and_guard") {
    lab::SimArm sim("S1", off());
    sim.connect();
    const VectorXd q0 = sim.read();
    const VectorXd q1 = q0 + vec({10, -8, 6, 0, 0, 5});
    MatrixXd via(2, 6);
    via << q0.transpose(), q1.transpose();
    const lab::Trajectory tr = lab::smooth_segments(via, 1.5, 50, 1.0);
    const lab::Log log = lab::run_logged(sim, tr.t, tr.q);
    CHECK_FALSE(log.aborted.has_value());
    CHECK(log.stop.cwiseAbs().sum() == 0.0);
    REQUIRE(log.meas.rows() == tr.q.rows());
    REQUIRE(log.meas.cols() == tr.q.cols());
    CHECK_ALLCLOSE(log.meas.row(log.meas.rows() - 1), q1, 1.0);   // settles within the droop / noise
    // a command jump far above the 40 deg/s cap is refused by the guard; run_logged stops and keeps logging
    lab::SimArm sim2("S1", off());
    sim2.connect();
    const VectorXd p0 = sim2.read();
    const VectorXd t2 = VectorXd::LinSpaced(20, 0.0, 19.0) * 0.02;
    MatrixXd q2 = p0.transpose().replicate(20, 1);
    q2.col(0).tail(15).array() += 30.0;                            // 30 deg in 20 ms = 1500 deg/s
    const lab::Log log2 = lab::run_logged(sim2, t2, q2);
    REQUIRE(log2.aborted.has_value());
    CHECK(log2.aborted->find("rate") != std::string::npos);
    REQUIRE(log2.stop.size() == 20);
    CHECK(log2.stop.head(5).cwiseAbs().sum() == 0.0);
    CHECK((log2.stop.tail(15).array() == 1.0).all());
    CHECK_ALLCLOSE(log2.meas.row(19), p0, 1.0);                    // did not follow the jump (droop only)
}

// ------------------------------------------------------------------ Step 3: HW-2, HW-3, HW-6
TEST_CASE("test_joint_tracking_reference_log") {
    const lab::JointTracking r = lab::joint_tracking(ref_log());
    CHECK(r.n_samples == 800);
    CHECK_ALLCLOSE(r.rms, vec({0.934, 0.431, 0.644, 0.542, 0.056, 0.415}), 2e-3);
    CHECK_ALLCLOSE(r.max, vec({2.034, 0.861, 1.452, 1.702, 0.176, 1.216}), 2e-3);
    REQUIRE(r.mean.size() == 6);
    CHECK_ALLCLOSE(r.mean.segment(1, 2), vec({0.354, 0.334}), 2e-3);
    const json& c = cases()["joint_tracking_reference_log"];
    CHECK_ALLCLOSE(r.rms, as_vec(c["rms"]), 1e-9);
    CHECK_ALLCLOSE(r.mean, as_vec(c["mean"]), 1e-9);
}

TEST_CASE("test_task_tracking_matches_modern_robotics") {
    // the C++ copy of the course arm model equals course_arm_model.py
    CHECK_MAT_NEAR(arm().M, as<Matrix4d>(cases()["course_arm"]["M"]), 1e-15);
    CHECK_MAT_NEAR(arm().Slist, as_mat(cases()["course_arm"]["Slist"]), 1e-15);
    const lab::TaskTracking r = lab::task_tracking(ref_log(), arm().M, arm().Slist, off(), p_tool());
    const json& c = cases()["task_tracking_matches_modern_robotics"];
    const auto k0 = c["k0"].get<Eigen::Index>();
    REQUIRE(r.err_mm.size() == k0);
    const json& idx = c["idx"];                                      // rng 361 of test_lab.py, stored
    MatrixXd qm(static_cast<Eigen::Index>(idx.size()), 6), qc(static_cast<Eigen::Index>(idx.size()), 6);
    VectorXd got(static_cast<Eigen::Index>(idx.size()));
    for (std::size_t i = 0; i < idx.size(); ++i) {
        const auto k = idx[i].get<Eigen::Index>();
        const auto ii = static_cast<Eigen::Index>(i);
        qm.row(ii) = ref_log().meas.row(k);
        qc.row(ii) = ref_log().cmd.row(k);
        got(ii) = r.err_mm(k);
    }
    const VectorXd e = (fk_tool_mm(qm) - fk_tool_mm(qc)).rowwise().norm();
    CHECK_ALLCLOSE(got, e, 1e-6);
    CHECK_ALLCLOSE(got, as_vec(c["err_mm_idx"]), 1e-6);
    CHECK_ISCLOSE(r.rms_mm, 4.674, 2e-3);
    CHECK_ISCLOSE(r.max_mm, 8.544, 2e-3);
}

TEST_CASE("test_stop_check_reference_and_synthetic") {
    const lab::StopCheck r = lab::stop_check(ref_log());
    CHECK_ISCLOSE(r.t_stop, 16.0, 1e-8);
    CHECK(r.t_halt <= 0.5);
    CHECK_ISCLOSE(r.motion_deg, 0.791, 2e-3);
    const json& c = cases()["stop_check_reference_and_synthetic"];
    const VectorXd t = as_vec(c["t"]);                               // np.arange(0, 3, 0.02)
    const MatrixXd q = as_mat(c["q"]);                               // joint 3 coasts 2 deg after the stop + noise (rng 0)
    lab::Log syn;
    syn.t = t;
    syn.cmd = q;
    syn.meas = q;
    syn.stop = (t.array() >= 1.0).cast<double>();
    const lab::StopCheck s = lab::stop_check(syn, 0.3);
    CHECK(s.t_halt > 0.15);
    CHECK(s.t_halt < 0.45);
    CHECK(s.motion_deg > 1.8);
    CHECK(s.motion_deg < 2.3);
    CHECK_SCALAR_NEAR(s.t_halt, c["synthetic"]["t_halt"].get<double>(), 1e-9);
    CHECK_SCALAR_NEAR(s.motion_deg, c["synthetic"]["motion_deg"].get<double>(), 1e-9);
    lab::Log syn2 = syn;
    syn2.meas.col(4) = 5 * t;                                        // never stops
    syn2.cmd = syn2.meas;
    CHECK(std::isinf(lab::stop_check(syn2).t_halt));
}

// ------------------------------------------------------------------ Step 4: HW-1, accuracy, repeatability
TEST_CASE("test_hw_tip_errors_reference") {
    const VectorXd e = lab::hw_tip_errors_mm(tips().Q, tips().P, arm().M, arm().Slist, off(), p_tool());
    CHECK_ALLCLOSE(e, VectorXd((fk_tool_mm(tips().Q) - tips().P).rowwise().norm()), 1e-6);
    CHECK_ALLCLOSE(e, as_vec(cases()["hw_tip_errors_reference"]["e_mm"]), 1e-6);
    REQUIRE(e.size() >= 4);
    CHECK_ALLCLOSE(e.head(4), vec({4.241, 4.104, 4.502, 3.970}), 2e-3);
}

TEST_CASE("test_accuracy_repeatability_small_case_and_reference") {
    MatrixXd P(4, 3);
    P << 1, 0, 0,  -1, 0, 0,  0, 1, 0,  0, -1, 0;
    P.rowwise() += Eigen::RowVector3d(10, 20, 30);
    const lab::AccuracyRepeatability r = lab::accuracy_repeatability(P, Vector3d(10, 20, 33));
    CHECK_ALLCLOSE(r.barycentre, Vector3d(10, 20, 30), 1e-8);
    CHECK_ISCLOSE(r.AP_mm, 3.0, 1e-8);
    CHECK_ALLCLOSE(r.l_mm, VectorXd::Ones(4), 1e-8);
    CHECK_ISCLOSE(r.RP_mm, 1.0, 1e-8);                               // std of equal distances = 0
    const MatrixXd rep = tips().P.bottomRows(tips().P.rows() - 4);
    const Eigen::VectorXi ap = tips().approach.tail(tips().approach.size() - 4);
    MatrixXd plus(0, 3), minus(0, 3);
    for (Eigen::Index k = 0; k < rep.rows(); ++k) {
        MatrixXd& dst = ap(k) > 0 ? plus : minus;
        dst.conservativeResize(dst.rows() + 1, 3);
        dst.row(dst.rows() - 1) = rep.row(k);
    }
    const Vector3d pref = fk_tool_mm(pose_servo("P2").transpose()).row(0).transpose();
    const lab::AccuracyRepeatability a = lab::accuracy_repeatability(rep, pref);
    CHECK_ISCLOSE(a.AP_mm, 3.901, 2e-3);
    CHECK_ISCLOSE(a.RP_mm, 3.615, 2e-3);
    CHECK_ISCLOSE(lab::accuracy_repeatability(plus, pref).RP_mm, 1.255, 2e-3);
    CHECK_ISCLOSE(lab::backlash_split(plus, minus), 4.963, 2e-3);
    MatrixXd A(2, 3), B(1, 3);
    A << 0, 0, 0,  2, 0, 0;
    B << 0, 3, 4;
    CHECK_ISCLOSE(lab::backlash_split(A, B), std::sqrt(1.0 + 9.0 + 16.0), 1e-8);
}

// ------------------------------------------------------------------ Step 5: calibration
TEST_CASE("test_fit_offsets_recovers_known_errors") {
    const char* names[5] = {"P1", "P2", "P3", "P4", "R"};
    MatrixXd Q(5, 6);
    for (int i = 0; i < 5; ++i) Q.row(i) = pose_servo(names[i]).transpose();
    const VectorXd truth = vec({0.0, -0.7, 1.1, 0.4, 0.0, 0.0});
    const MatrixXd P = fk_tool_mm(Q, off() - truth);                 // arm whose true zero differs by `truth`
    const lab::OffsetFit f = lab::fit_offsets(Q, P, arm().M, arm().Slist, off(), p_tool(), {1, 2, 3});
    CHECK_ALLCLOSE(f.delta_deg, truth.segment(1, 3), 1e-4);
    CHECK(f.rms_after_mm < 1e-4);
    CHECK_ALLCLOSE(f.new_offsets_deg, VectorXd(off() - truth), 1e-4);
    const lab::OffsetFit g = lab::fit_offsets(tips().Q.topRows(4), tips().P.topRows(4), arm().M, arm().Slist, off(),
                                              p_tool(), {1, 2});
    CHECK_ALLCLOSE(g.delta_deg, vec({-0.609, 1.740}), 2e-3);
    CHECK_ISCLOSE(g.rms_before_mm, 4.209, 2e-3);
    CHECK_ISCLOSE(g.rms_after_mm, 1.031, 2e-3);
}

// ------------------------------------------------------------------ Step 6: error budget
TEST_CASE("test_error_budget_matches_jacobian_columns") {
    const VectorXd th = as_vec(plan()["poses_model_deg"]["P2"]) * (PI / 180.0);
    const lab::ErrorBudget r = lab::error_budget(arm().M, arm().Slist, th, p_tool());
    const Vector3d p = fk_tool_mm(pose_servo("P2").transpose()).row(0).transpose() / 1e3;
    const MatrixXd J = mr::jacobian_space(arm().Slist, th);          // Week 9: velocity of the point p
    VectorXd lever(J.cols());
    for (Eigen::Index j = 0; j < J.cols(); ++j) {
        const Vector3d w = J.col(j).head<3>(), v = J.col(j).tail<3>();
        lever(j) = (v + w.cross(p)).norm();
    }
    CHECK_ALLCLOSE(r.lever_m, lever, 1e-9);
    CHECK_ALLCLOSE(r.lever_m, vec({0.21674, 0.20812, 0.20911, 0.12958, 0.10560, 0.0}), 2e-5);
    CHECK_ISCLOSE(r.rss_mm, 0.617, 2e-3);
    CHECK_ISCLOSE(r.worst_mm, 1.333, 2e-3);
    const lab::ErrorBudget r2 = lab::error_budget(arm().M, arm().Slist, th, p_tool(), vec({0, 1, 0, 0, 0, 0}));
    REQUIRE(r2.dp_mm.size() == 6);
    CHECK_ISCLOSE(r2.dp_mm(1), 1e3 * lever(1) * PI / 180, 1e-8);
    CHECK_ISCLOSE(r2.worst_mm, r2.dp_mm(1), 1e-8);
}
