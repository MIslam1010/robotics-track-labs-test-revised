// Auto-checks for SBEG 461 Lab 9 (trajectory library, torque feasibility, time-optimal time scaling) -- C++ track.
// Mirrors test_lab.py test by test (a parametrized Python test is ONE test case here that loops over the values).
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*cubic*,*quintic*"      (run only some tests, like pytest -k)
// Ground truth: mrcpp (the C++ port of modern_robotics: cubic/quintic_time_scaling, joint_trajectory,
// inverse_dynamics, inverse_dynamics_trajectory), the numbers printed in MR Secs. 9.2-9.3, a brute-force bisection on
// the duration, the closed-form MR 2R dynamics (MR Eqs. 8.9-8.10), closed-form phase-plane results, and cpp_cases.json,
// which holds the numbers of the Python reference (including the scipy.interpolate.BSpline values and the results of
// the real toppra), so both languages are checked against the same values.
//
// Plot data: when your functions return results of the right size, the tests also write CSV files into the folder
// you run lab_tests from (lab9_cpp_scaling_cubic.csv, lab9_cpp_scaling_quintic.csv, lab9_cpp_scaling_trapezoid.csv,
// lab9_cpp_fig_9_7a.csv, lab9_cpp_bsplines.csv, lab9_cpp_tau_AB.csv, lab9_cpp_phase_2r.csv,
// lab9_cpp_phase_gantry.csv, lab9_cpp_phase_course_arm.csv); plot them with  python cpp/plot_results.py
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Eigen::Matrix2d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::VectorXd;
using sbeg::as_mat;
using sbeg::as_vec;
using sbeg::json;

namespace {
constexpr double PI = 3.14159265358979323846;
const double INF = std::numeric_limits<double>::infinity();

const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

VectorXd vec(std::initializer_list<double> v) {
    VectorXd out(static_cast<Eigen::Index>(v.size()));
    Eigen::Index i = 0;
    for (double x : v) out(i++) = x;
    return out;
}

const VectorXd TAU_2R = vec({20.0, 10.0});
const VectorXd F_GANTRY = vec({120.0, 60.0});

/// The course arm with payload, derated (Python fixture `arm`), loaded with the given helper.
const lab::Robot& arm() {
    static const lab::Robot R = lab::load_course_arm(true, true);
    return R;
}

/// The task configurations (Python fixture `tasks`), loaded with the given helper.
const lab::Tasks& tasks() {
    static const lab::Tasks T = lab::load_tasks();
    return T;
}

// The Part D fixtures of test_lab.py (rebuilt in every test case: they are cheap in C++).
struct Setup {
    lab::Robot model;
    lab::Path path;
};
Setup two_r() { return {lab::mr_2r_model(), lab::straight_line_path(vec({0.0, 0.0}), vec({PI / 2, PI / 2}))}; }
Setup gantry() { return {lab::gantry_model(), lab::semicircle_path(0.3)}; }
Setup elbow() { return {lab::elbow_exo_model(), lab::straight_line_path(vec({0.0}), vec({2 * PI / 3}))}; }

struct ArmTask {
    lab::ToppTask task;
    lab::Robot r;
    lab::Path path;
    VectorXd tl, vmax, amax;  // 60 % torque, 50 % speed, |qddot| <= 4 rad/s^2 (the hw limit set)
};
ArmTask arm_task() {
    ArmTask a;
    a.task = lab::load_topp_task();
    a.r = lab::course_arm_model(a.task.payload_kg);
    a.path = lab::waypoint_path(a.task.waypoints);
    a.tl = a.task.torque_fraction * a.r.tau_max;
    a.vmax = a.task.hw_speed_fraction * a.r.qdot_max;
    a.amax = VectorXd::Constant(6, a.task.hw_qddot_max);
    return a;
}
lab::ToppResult hw_topp(const ArmTask& a) { return lab::topp(a.r, a.path, a.tl, a.vmax, a.amax, a.task.grid_N); }

VectorXd linspace(double a, double b, int n) { return VectorXd::LinSpaced(n, a, b); }

/// Q, Qd, Qdd of the quintic time scaling of `path` with T = 1 s (input of min_time_scale), N samples.
struct Samples {
    MatrixXd Q, Qd, Qdd;
};
Samples quintic_samples(const lab::Path& path, int N = 401) {
    const lab::TrajectoryFn traj = lab::quintic_traj(path, 1.0);
    const VectorXd t = linspace(0.0, 1.0, N);
    Samples out;
    for (Eigen::Index k = 0; k < N; ++k) {
        const VectorXd q = traj(t(k), 0), qd = traj(t(k), 1), qdd = traj(t(k), 2);
        if (k == 0) {
            out.Q.resize(N, q.size());
            out.Qd.resize(N, q.size());
            out.Qdd.resize(N, q.size());
        }
        out.Q.row(k) = q.transpose();
        out.Qd.row(k) = qd.transpose();
        out.Qdd.row(k) = qdd.transpose();
    }
    return out;
}

Matrix2d M2(const Vector2d& th) {
    const double c2 = std::cos(th(1));
    Matrix2d M;
    M << 3 + 2 * c2, 1 + c2, 1 + c2, 1.0;
    return M;
}

Vector2d c2r(const Vector2d& th, const Vector2d& dth) {
    const double s2 = std::sin(th(1));
    return Vector2d(-s2 * (2 * dth(0) * dth(1) + dth(1) * dth(1)), s2 * dth(0) * dth(0));
}

/// np.interp for one value (xp increasing).
double interp1(double x, const VectorXd& xp, const VectorXd& fp) {
    if (x <= xp(0)) return fp(0);
    for (Eigen::Index i = 1; i < xp.size(); ++i)
        if (x <= xp(i)) return fp(i - 1) + (fp(i) - fp(i - 1)) * (x - xp(i - 1)) / (xp(i) - xp(i - 1));
    return fp(fp.size() - 1);
}

/// np.gradient(F, t, axis=0): second-order central differences inside, first-order differences at the ends.
MatrixXd gradient(const MatrixXd& F, const VectorXd& t) {
    const Eigen::Index n = F.rows();
    if (t.size() != n || n < 2) throw std::invalid_argument("gradient: need len(t) == rows >= 2");
    MatrixXd G(n, F.cols());
    G.row(0) = (F.row(1) - F.row(0)) / (t(1) - t(0));
    G.row(n - 1) = (F.row(n - 1) - F.row(n - 2)) / (t(n - 1) - t(n - 2));
    for (Eigen::Index i = 1; i + 1 < n; ++i) {
        const double hs = t(i) - t(i - 1), hd = t(i + 1) - t(i);
        G.row(i) = (hs * hs * F.row(i + 1) + (hd * hd - hs * hs) * F.row(i) - hd * hd * F.row(i - 1)) /
                   (hs * hd * (hd + hs));
    }
    return G;
}

/// np.allclose(got, expected, rtol, atol): |got - expected| <= atol + rtol |expected| entry by entry.
sbeg::NearResult allclose(const MatrixXd& got_in, const MatrixXd& exp_in, double atol, double rtol = 1e-5) {
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

/// np.isclose(got, expected, rtol, atol) for scalars.
sbeg::NearResult isclose(double got, double expected, double rtol = 1e-5, double atol = 1e-8) {
    return sbeg::compare_near(got, expected, atol + rtol * std::abs(expected));
}

/// Reference minimum duration by bisection (test_lab.py: _bisect_min_T), with mrcpp's inverse dynamics.
double bisect_min_T(const VectorXd& q0, const VectorXd& q1, int method, int N = 201) {
    const lab::Robot& R = arm();
    auto ok = [&](double T) {
        const lab::Trajectory tr = lab::joint_trajectory(q0, q1, T, N, method);
        const MatrixXd Tau = mr::inverse_dynamics_trajectory(tr.Q, tr.Qd, tr.Qdd, R.g, MatrixXd::Zero(N, 6), R.Mlist,
                                                             R.Glist, R.Slist);
        for (Eigen::Index k = 0; k < tr.Qd.rows(); ++k)
            for (Eigen::Index i = 0; i < 6; ++i)
                if (std::abs(tr.Qd(k, i)) > R.qdot_max(i) + 1e-12 || std::abs(Tau(k, i)) > R.tau_max(i) + 1e-12)
                    return false;
        return true;
    };
    double lo = 0.01, hi = 10.0;
    for (int it = 0; it < 40; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (ok(mid)) hi = mid;
        else lo = mid;
    }
    return hi;
}

/// Plot data for plot_results.py: t, s, sdot, sddot.
void write_scaling_csv(const std::string& name, const VectorXd& t, const lab::TimeScaling& r) {
    if (r.s.size() != t.size() || r.sd.size() != t.size() || r.sdd.size() != t.size()) return;
    MatrixXd out(t.size(), 4);
    out << t, r.s, r.sd, r.sdd;
    sbeg::write_csv(name, {"t", "s", "sd", "sdd"}, out);
}

/// Plot data (Part D figure): s, velocity limit curve, toppra's sdot(s) (interpolated on the same s).
void write_phase_csv(const std::string& name, const Setup& t, const VectorXd& tau, const VectorXd& qdot_max,
                     const lab::ToppResult& tp) {
    if (tp.s_grid.size() < 2 || tp.sdot_grid.size() != tp.s_grid.size()) return;
    const VectorXd s = linspace(0.0, 1.0, 201);
    MatrixXd out(201, 3);
    for (Eigen::Index k = 0; k < 201; ++k)
        out.row(k) << s(k), lab::velocity_limit(t.model, t.path, s(k), -tau, tau, qdot_max),
            interp1(s(k), tp.s_grid, tp.sdot_grid);
    sbeg::write_csv(name, {"s", "sdot_limit", "sdot_topp"}, out);
}

void check_command_dict(const lab::CommandCheck& got, const json& exp) {
    CHECK(got.ok == exp["ok"].get<bool>());
    CHECK(got.within_limits == exp["within_limits"].get<bool>());
    CHECK(got.speed_ok == exp["speed_ok"].get<bool>());
    CHECK(got.starts_at_rest == exp["starts_at_rest"].get<bool>());
    CHECK(got.ends_at_rest == exp["ends_at_rest"].get<bool>());
    CHECK_SCALAR_NEAR(got.peak_speed_ratio, exp["peak_speed_ratio"].get<double>(), 1e-6);
}
}  // namespace

#define CHECK_ALLCLOSE(got, expected, atol)                                                          \
    do {                                                                                             \
        const ::sbeg::NearResult lab_r_ = allclose((got), (expected), (atol));                        \
        CHECK_MESSAGE(lab_r_.ok, "allclose(" #got ", " #expected ")\n  " << lab_r_.msg);             \
    } while (false)

#define CHECK_ISCLOSE(got, expected)                                                                 \
    do {                                                                                             \
        const ::sbeg::NearResult lab_r_ = isclose((got), (expected));                                 \
        CHECK_MESSAGE(lab_r_.ok, "isclose(" #got ", " #expected ")\n  " << lab_r_.msg);              \
    } while (false)

#define CHECK_ISCLOSE_TOL(got, expected, rtol, atol)                                                 \
    do {                                                                                             \
        const ::sbeg::NearResult lab_r_ = isclose((got), (expected), (rtol), (atol));                 \
        CHECK_MESSAGE(lab_r_.ok, "isclose(" #got ", " #expected ")\n  " << lab_r_.msg);              \
    } while (false)

// ---------------------------------------------------------------- Part A: point-to-point time scalings

TEST_CASE("test_cubic_matches_modern_robotics_and_mr_fig_9_3") {
    for (const auto& c : cases()["cubic_matches_modern_robotics_and_mr_fig_9_3"]) {
        const double T = c["T"].get<double>();
        INFO("T = " << T);
        const VectorXd t = linspace(0.0, T, 41);
        const lab::TimeScaling r = lab::cubic_time_scaling(T, t);
        REQUIRE(r.s.size() == 41);
        REQUIRE(r.sd.size() == 41);
        REQUIRE(r.sdd.size() == 41);
        VectorXd s_mr(41);
        for (Eigen::Index k = 0; k < 41; ++k) s_mr(k) = mr::cubic_time_scaling(T, t(k));
        CHECK_ALLCLOSE(r.s, s_mr, 1e-12);
        CHECK_ALLCLOSE(r.s, as_vec(c["s_mr"]), 1e-12);
        CHECK_ISCLOSE_TOL(r.sd.maxCoeff(), 3.0 / (2.0 * T), 1e-5, 1e-9);  // MR Fig. 9.3: peak 3/(2T) at T/2
        CHECK_ISCLOSE(r.sdd(0), 6.0 / (T * T));
        CHECK_ISCLOSE(r.sdd(40), -6.0 / (T * T));
        CHECK_ALLCLOSE(gradient(r.s, t).middleRows(1, 39), r.sd.segment(1, 39), 0.02 / T);
        if (T == 2.0) {
            const VectorXd t1 = linspace(0.0, 1.0, 201);
            write_scaling_csv("lab9_cpp_scaling_cubic.csv", t1, lab::cubic_time_scaling(1.0, t1));
        }
    }
}

TEST_CASE("test_quintic_matches_modern_robotics_and_mr_fig_9_4") {
    for (const auto& c : cases()["quintic_matches_modern_robotics_and_mr_fig_9_4"]) {
        const double T = c["T"].get<double>();
        INFO("T = " << T);
        const VectorXd t = linspace(0.0, T, 2001);
        const lab::TimeScaling r = lab::quintic_time_scaling(T, t);
        REQUIRE(r.s.size() == 2001);
        REQUIRE(r.sd.size() == 2001);
        REQUIRE(r.sdd.size() == 2001);
        VectorXd s_mr(2001);
        for (Eigen::Index k = 0; k < 2001; ++k) s_mr(k) = mr::quintic_time_scaling(T, t(k));
        CHECK_ALLCLOSE(r.s, s_mr, 1e-12);
        const VectorXd tc = as_vec(c["t"]);
        CHECK_ALLCLOSE(lab::quintic_time_scaling(T, tc).s, as_vec(c["s_mr"]), 1e-12);
        CHECK_ISCLOSE_TOL(r.sd.maxCoeff(), 15.0 / (8.0 * T), 1e-6, 1e-8);  // MR Fig. 9.4
        CHECK_ISCLOSE_TOL(r.sdd.maxCoeff(), 10.0 / (std::sqrt(3.0) * T * T), 1e-5, 1e-8);
        CHECK(std::abs(r.sd(0)) < 1e-12);
        CHECK(std::abs(r.sdd(0)) < 1e-12);
        CHECK(std::abs(r.sdd(2000)) < 1e-9);
        CHECK_ALLCLOSE(gradient(r.sd, t).middleRows(1, 1999), r.sdd.segment(1, 1999), 0.01 / (T * T));
        if (T == 2.0) {
            const VectorXd t1 = linspace(0.0, 1.0, 201);
            write_scaling_csv("lab9_cpp_scaling_quintic.csv", t1, lab::quintic_time_scaling(1.0, t1));
        }
    }
}

TEST_CASE("test_time_scalings_accept_scalars_and_clip") {
    const json& c = cases()["time_scalings_accept_scalars_and_clip"];
    {
        INFO("cubic_time_scaling");
        const lab::TimeScalingPoint p = lab::cubic_time_scaling(2.0, 1.0);
        CHECK_ISCLOSE(p.s, 0.5);
        CHECK_ISCLOSE(lab::cubic_time_scaling(2.0, 3.0).s, 1.0);
        CHECK_ISCLOSE(lab::cubic_time_scaling(2.0, -1.0).s, 0.0);
        CHECK_ALLCLOSE(as_vec(c["cubic"]),
                       Eigen::Vector3d(p.s, lab::cubic_time_scaling(2.0, 3.0).s, lab::cubic_time_scaling(2.0, -1.0).s),
                       1e-12);
    }
    {
        INFO("quintic_time_scaling");
        const lab::TimeScalingPoint p = lab::quintic_time_scaling(2.0, 1.0);
        CHECK_ISCLOSE(p.s, 0.5);
        CHECK_ISCLOSE(lab::quintic_time_scaling(2.0, 3.0).s, 1.0);
        CHECK_ISCLOSE(lab::quintic_time_scaling(2.0, -1.0).s, 0.0);
        CHECK_ALLCLOSE(as_vec(c["quintic"]),
                       Eigen::Vector3d(p.s, lab::quintic_time_scaling(2.0, 3.0).s,
                                       lab::quintic_time_scaling(2.0, -1.0).s),
                       1e-12);
    }
}

TEST_CASE("test_trapezoid_three_stage_mr_eqs_9_16_to_9_24") {
    const double v = 0.5, a = 1.25;  // v^2/a = 0.2 <= 1: three stages
    const lab::TrapezoidParams p = lab::trapezoid_params(v, a);
    CHECK_ISCLOSE(p.T, (a + v * v) / (v * a));
    CHECK_ISCLOSE(p.T, 2.4);
    CHECK_ISCLOSE(p.t_a, 0.4);
    CHECK_ISCLOSE(p.v_peak, 0.5);
    const json& c = cases()["trapezoid_three_stage_mr_eqs_9_16_to_9_24"];
    CHECK_ALLCLOSE(Eigen::Vector3d(p.T, p.t_a, p.v_peak),
                   Eigen::Vector3d(c["T"].get<double>(), c["t_a"].get<double>(), c["v_peak"].get<double>()), 1e-12);
    const VectorXd t = linspace(0.0, p.T, 4801);
    const lab::TimeScaling r = lab::trapezoid_time_scaling(v, a, t);
    REQUIRE(r.s.size() == 4801);
    REQUIRE(r.sd.size() == 4801);
    REQUIRE(r.sdd.size() == 4801);
    CHECK_ISCLOSE(r.s(4800), 1.0);
    CHECK_ISCLOSE(r.s(0), 0.0);
    CHECK_ISCLOSE(r.sd.maxCoeff(), v);
    CHECK_ISCLOSE(r.sdd.maxCoeff(), a);
    CHECK_ISCLOSE(r.sdd.minCoeff(), -a);
    const VectorXd ds = r.s.tail(4800) - r.s.head(4800);
    CHECK_MESSAGE(ds.minCoeff() >= -1e-12, "s must not decrease; min diff(s) = " << ds.minCoeff());
    CHECK_ALLCLOSE(gradient(r.s, t), r.sd, 2e-3);
    Eigen::Index k = 0;  // np.searchsorted(t, 1.0): cruise phase, MR Eq. 9.21
    while (k < t.size() && t(k) < 1.0) ++k;
    CHECK_ISCLOSE(r.s(k), v * t(k) - v * v / (2.0 * a));
    // plot data (Part A): v = 1.5, a = 4.5 gives T = 1 s
    const VectorXd t1 = linspace(0.0, lab::trapezoid_params(1.5, 4.5).T, 201);
    write_scaling_csv("lab9_cpp_scaling_trapezoid.csv", t1, lab::trapezoid_time_scaling(1.5, 4.5, t1));
}

TEST_CASE("test_trapezoid_bang_bang_when_v2_over_a_exceeds_1") {
    const double v = 2.0, a = 1.0;  // v^2/a = 4 > 1 (MR Ex. 9.10, 9.13)
    const lab::TrapezoidParams p = lab::trapezoid_params(v, a);
    CHECK_ISCLOSE(p.T, 2.0);
    CHECK_ISCLOSE(p.t_a, 1.0);
    CHECK_ISCLOSE(p.v_peak, 1.0);
    const lab::TimeScaling r = lab::trapezoid_time_scaling(v, a, linspace(0.0, p.T, 1001));
    REQUIRE(r.s.size() == 1001);
    CHECK_ISCLOSE(r.s(1000), 1.0);
    CHECK_ISCLOSE(r.sd.maxCoeff(), 1.0);
}

TEST_CASE("test_joint_trajectory_matches_modern_robotics") {
    for (const auto& c : cases()["joint_trajectory_matches_modern_robotics"]) {
        const int method = c["method"].get<int>();
        INFO("method = " << method);
        const VectorXd q0 = as_vec(c["q0"]), q1 = as_vec(c["q1"]);
        const lab::Trajectory r = lab::joint_trajectory(q0, q1, 1.7, 35, method);
        REQUIRE(r.t.size() == 35);
        REQUIRE((r.Q.rows() == 35 && r.Q.cols() == 6));
        REQUIRE((r.Qd.rows() == 35 && r.Qd.cols() == 6));
        REQUIRE((r.Qdd.rows() == 35 && r.Qdd.cols() == 6));
        CHECK_ALLCLOSE(r.t, linspace(0.0, 1.7, 35), 1e-8);
        CHECK_ALLCLOSE(r.Q, mr::joint_trajectory(q0, q1, 1.7, 35, method), 1e-12);
        CHECK_ALLCLOSE(r.Q, as_mat(c["Q_mr"]), 1e-12);
        const double dtm = r.t(1) - r.t(0);
        const VectorXd tt = VectorXd::LinSpaced(35, 0.0, 34.0) * dtm;  // np.gradient(Q, dtm, axis=0)
        CHECK_ALLCLOSE(gradient(r.Q, tt).middleRows(2, 31), r.Qd.middleRows(2, 31), 0.05);
    }
}

TEST_CASE("test_joint_trajectory_trapezoid_respects_limits") {
    const VectorXd& q0 = tasks().A;
    const VectorXd& q1 = tasks().B;
    const VectorXd vlim = VectorXd::Constant(6, 1.3962634), alim = tasks().qddot_lim;
    const lab::TrapezoidLimits L = lab::trapezoid_from_limits(q0, q1, vlim, alim);  // provided; MR Eqs. 9.14-9.15
    CHECK_SCALAR_NEAR(L.v, 1.1636, 1e-4);
    CHECK_SCALAR_NEAR(L.a, 3.3333, 1e-4);
    CHECK_SCALAR_NEAR(L.T, 1.2085, 1e-4);
    const lab::Trajectory r = lab::joint_trajectory(q0, q1, 0.0, 501, lab::TRAP, L.v, L.a);  // T is ignored
    REQUIRE(r.t.size() == 501);
    REQUIRE((r.Q.rows() == 501 && r.Q.cols() == 6));
    REQUIRE((r.Qd.rows() == 501 && r.Qd.cols() == 6));
    REQUIRE((r.Qdd.rows() == 501 && r.Qdd.cols() == 6));
    CHECK_ISCLOSE(r.t(500), L.T);
    CHECK_ALLCLOSE(r.Q.row(500), q1, 1e-8);
    CHECK_ALLCLOSE(r.Q.row(0), q0, 1e-8);
    for (Eigen::Index i = 0; i < 6; ++i) {
        INFO("joint " << i + 1);
        CHECK(r.Qd.col(i).cwiseAbs().maxCoeff() <= vlim(i) + 1e-9);
        CHECK(r.Qdd.col(i).cwiseAbs().maxCoeff() <= alim(i) + 1e-9);
    }
    CHECK_ISCLOSE(r.Qd.col(0).cwiseAbs().maxCoeff(), vlim(0));  // joint 1 saturates (largest move)
    const json& c = cases()["joint_trajectory_trapezoid_respects_limits"];
    CHECK_ALLCLOSE(Eigen::Vector3d(L.v, L.a, L.T),
                   Eigen::Vector3d(c["v"].get<double>(), c["a"].get<double>(), c["T"].get<double>()), 1e-12);
}

// ---------------------------------------------------------------- Part B: via points and B-splines

TEST_CASE("test_via_cubic_coefficients_formula_and_continuity") {
    const Eigen::Vector4d beta(0.0, 1.0, 0.5, 2.0), bdot(0.0, 0.8, -0.3, 0.0), Tm(0.0, 1.0, 2.5, 3.0);
    const MatrixXd A = lab::via_cubic_coefficients(beta, bdot, Tm);
    REQUIRE((A.rows() == 3 && A.cols() == 4));
    for (Eigen::Index j = 0; j < 3; ++j) {  // each segment meets its four constraints
        INFO("segment j = " << j);
        const double dT = Tm(j + 1) - Tm(j);
        const double a0 = A(j, 0), a1 = A(j, 1), a2 = A(j, 2), a3 = A(j, 3);
        CHECK_ISCLOSE(a0, beta(j));
        CHECK_ISCLOSE(a1, bdot(j));
        CHECK_ISCLOSE(a0 + a1 * dT + a2 * dT * dT + a3 * dT * dT * dT, beta(j + 1));
        CHECK_ISCLOSE(a1 + 2.0 * a2 * dT + 3.0 * a3 * dT * dT, bdot(j + 1));
    }
    CHECK_ALLCLOSE(A, as_mat(cases()["via_cubic_coefficients_formula_and_continuity"]["A"]), 1e-12);
}

TEST_CASE("test_via_trajectory_mr_figure_9_7a") {
    // MR p. 337: vias (0,0), (0,1), (1,1), (1,0) at T = 0, 1, 2, 3, velocities (0,0), (1,0), (0,-1), (0,0)
    MatrixXd P(4, 2), V(4, 2);
    P << 0, 0,  0, 1,  1, 1,  1, 0;
    V << 0, 0,  1, 0,  0, -1,  0, 0;
    const Eigen::Vector4d Tm(0.0, 1.0, 2.0, 3.0);
    const lab::ViaSamples r = lab::via_trajectory(P, V, Tm, Tm);
    CHECK_ALLCLOSE(r.B, P, 1e-8);
    CHECK_ALLCLOSE(r.Bd, V, 1e-8);
    const VectorXd t = linspace(0.0, 3.0, 3001);
    const lab::ViaSamples f = lab::via_trajectory(P, V, Tm, t);
    REQUIRE((f.B.rows() == 3001 && f.B.cols() == 2));
    REQUIRE((f.Bd.rows() == 3001 && f.Bd.cols() == 2));
    CHECK_ALLCLOSE(gradient(f.B, t).middleRows(1, 2999), f.Bd.middleRows(1, 2999), 2e-3);
    CHECK(f.B.col(0).minCoeff() < -0.05);  // x dips below 0 in segment 1 (Fig. 9.7a): not in the convex hull
    CHECK_ISCLOSE(f.B.col(0).minCoeff(), cases()["via_trajectory_mr_figure_9_7a"]["x_min"].get<double>());
    MatrixXd out(3001, 3);  // plot data (Part B)
    out << t, f.B;
    sbeg::write_csv("lab9_cpp_fig_9_7a.csv", {"t", "x", "y"}, out);
}

TEST_CASE("test_uniform_cubic_bspline_matches_scipy_and_convex_hull") {
    const json& c = cases()["uniform_cubic_bspline_matches_scipy_and_convex_hull"];
    const MatrixXd P = as_mat(c["P"]);  // drawn with the generator of test_lab.py
    const Eigen::Index n = P.rows();
    for (const auto& k : c["cases"]) {
        const double u = k["u"].get<double>();
        INFO("u = " << u);
        const lab::BSplinePoint r = lab::uniform_cubic_bspline(P, u);
        CHECK_ALLCLOSE(r.p, as_vec(k["p"]), 1e-12);
        CHECK_ALLCLOSE(r.dp, as_vec(k["dp"]), 1e-10);
        CHECK_ALLCLOSE(r.ddp, as_vec(k["ddp"]), 1e-9);
        REQUIRE(r.p.size() == 2);
        const Eigen::Index i = std::min(static_cast<Eigen::Index>(std::floor(u)), n - 4);  // local convex hull
        const MatrixXd box = P.middleRows(i, 4);
        for (Eigen::Index d = 0; d < 2; ++d) {
            CHECK(r.p(d) >= box.col(d).minCoeff() - 1e-12);
            CHECK(r.p(d) <= box.col(d).maxCoeff() + 1e-12);
        }
    }
    // plot data (Part B): the four vias of Fig. 9.7a as control points, open (one segment) and clamped
    MatrixXd ctrl(4, 2), clamped(10, 2);
    ctrl << 0, 0,  0, 1,  1, 1,  1, 0;
    clamped << 0, 0,  0, 0,  0, 0,  0, 0,  0, 1,  1, 1,  1, 0,  1, 0,  1, 0,  1, 0;
    MatrixXd out(201, 5);
    for (Eigen::Index k = 0; k < 201; ++k) {
        const double rr = static_cast<double>(k) / 200.0;
        const lab::BSplinePoint a = lab::uniform_cubic_bspline(ctrl, rr);
        const lab::BSplinePoint b = lab::uniform_cubic_bspline(clamped, 7.0 * rr);
        if (a.p.size() != 2 || b.p.size() != 2) return;
        out.row(k) << rr, a.p(0), a.p(1), b.p(0), b.p(1);
    }
    sbeg::write_csv("lab9_cpp_bsplines.csv", {"r", "x_open", "y_open", "x_clamped", "y_clamped"}, out);
}

// ---------------------------------------------------------------- Part C: torque feasibility on the course arm

TEST_CASE("test_course_arm_data_and_gravity_at_A") {
    const lab::Robot& R = arm();
    CHECK((R.Slist.rows() == 6 && R.Slist.cols() == 6));
    CHECK(R.Mlist.size() == 7);
    CHECK(R.Glist.size() == 6);
    VectorXd rated(6);
    rated << 1.8, 3.0, 1.5, 0.6, 0.6, 0.4;
    CHECK_ALLCLOSE(R.tau_max, 0.6 * rated, 1e-8);
    CHECK_ALLCLOSE(R.qdot_max, VectorXd::Constant(6, 0.5 * 160.0 * PI / 180.0), 1e-8);
    const lab::Robot r = lab::course_arm_model(0.25);  // same payload rule as the course robot models
    REQUIRE(r.Glist.size() == 6);
    for (std::size_t i = 0; i < 6; ++i) CHECK_ALLCLOSE(r.Glist[i], R.Glist[i], 1e-15);
    CHECK_ALLCLOSE(r.tau_max, rated, 1e-12);
    const MatrixXd G = lab::torque_profile(R, tasks().A.transpose(), MatrixXd::Zero(1, 6), MatrixXd::Zero(1, 6));
    REQUIRE((G.rows() == 1 && G.cols() == 6));
    CHECK_ALLCLOSE(G.row(0), vec({0.0, -1.140, -0.570, -0.242, 0.0, 0.0}), 1e-3);  // checkpoint of Part C
    CHECK_ALLCLOSE(G.row(0), mr::inverse_dynamics(tasks().A, VectorXd::Zero(6), VectorXd::Zero(6), R.g,
                                                  mr::Vector6d::Zero(), R.Mlist, R.Glist, R.Slist), 1e-12);
    CHECK_ALLCLOSE(G.row(0), as_vec(cases()["course_arm_data_and_gravity_at_A"]["G_A"]), 1e-12);
}

TEST_CASE("test_torque_profile_matches_modern_robotics") {
    const lab::Robot& R = arm();
    const lab::Trajectory tr = lab::joint_trajectory(tasks().A, tasks().B, 1.8, 25, 5);
    const MatrixXd Tau = lab::torque_profile(R, tr.Q, tr.Qd, tr.Qdd);
    const MatrixXd ref =
        mr::inverse_dynamics_trajectory(tr.Q, tr.Qd, tr.Qdd, R.g, MatrixXd::Zero(25, 6), R.Mlist, R.Glist, R.Slist);
    REQUIRE((Tau.rows() == 25 && Tau.cols() == 6));
    CHECK_ALLCLOSE(Tau, ref, 1e-10);
    CHECK_ALLCLOSE(Tau, as_mat(cases()["torque_profile_matches_modern_robotics"]["Tau_mr"]), 1e-10);
}

TEST_CASE("test_min_time_scale_equals_bisection") {
    for (const auto& c : cases()["min_time_scale_equals_bisection"]) {
        const std::string goal = c["goal"].get<std::string>(), binding = c["binding"][0].get<std::string>();
        const int method = c["method"].get<int>();
        INFO("goal = " << goal << ", method = " << method << ", expected binding = " << binding);
        const lab::Trajectory tr = lab::joint_trajectory(tasks().A, tasks().config(goal), 1.0, 201, method);
        const lab::TimeScaleResult res = lab::min_time_scale(arm(), tr.Q, tr.Qd, tr.Qdd);
        CHECK_ISCLOSE(res.c, std::max(res.c_vel, res.c_tau));
        CHECK(res.c_acc == 0.0);
        CHECK(res.binding.first == binding);
        CHECK_ISCLOSE_TOL(res.c, bisect_min_T(tasks().A, tasks().config(goal), method), 1e-6, 1e-8);
        CHECK_ISCLOSE_TOL(res.c, c["c_bisect"].get<double>(), 1e-6, 1e-8);
    }
}

TEST_CASE("test_min_time_scale_checkpoint_values") {
    const lab::Trajectory ab = lab::joint_trajectory(tasks().A, tasks().B, 1.0, 401, 5);
    const lab::TimeScaleResult r = lab::min_time_scale(arm(), ab.Q, ab.Qd, ab.Qdd);
    CHECK_ISCLOSE_TOL(r.c_vel, 1.6114, 1e-5, 1e-3);
    CHECK(r.binding == std::make_pair(std::string("velocity"), 0));
    const lab::Trajectory ad = lab::joint_trajectory(tasks().A, tasks().D, 1.0, 401, 5);
    const lab::TimeScaleResult r2 = lab::min_time_scale(arm(), ad.Q, ad.Qd, ad.Qdd);
    CHECK_ISCLOSE_TOL(r2.c_tau, 0.2348, 1e-5, 1e-3);
    CHECK(r2.binding == std::make_pair(std::string("torque"), 3));  // joint 4!
    // with |qdd| <= 4 rad/s^2: c_acc = sqrt(0.1 * (10/sqrt(3)) / 4) (quintic peak sdd = 10/sqrt(3), MR Fig. 9.4)
    const lab::TimeScaleResult r3 = lab::min_time_scale(arm(), ad.Q, ad.Qd, ad.Qdd, tasks().qddot_lim);
    CHECK_ISCLOSE_TOL(r3.c_acc, std::sqrt(0.1 * 10.0 / std::sqrt(3.0) / 4.0), 1e-4, 1e-12);
    CHECK(r3.binding == std::make_pair(std::string("acceleration"), 1));
    CHECK_ISCLOSE(r3.c, r3.c_acc);
    const json& c = cases()["min_time_scale_checkpoint_values"];
    CHECK_ISCLOSE_TOL(r.c, c["AB"]["c"].get<double>(), 1e-9, 1e-12);
    CHECK_ISCLOSE_TOL(r2.c, c["AD"]["c"].get<double>(), 1e-9, 1e-12);
    CHECK_ISCLOSE_TOL(r3.c, c["AD_acc"]["c"].get<double>(), 1e-9, 1e-12);
    // plot data (Part C, Q4): tau_i / tau_max_i along the A -> B quintic at T = T_min
    if (!(r.c > 0.0) || !std::isfinite(r.c)) return;
    const lab::Trajectory tm = lab::joint_trajectory(tasks().A, tasks().B, r.c, 401, 5);
    const MatrixXd Tau = lab::torque_profile(arm(), tm.Q, tm.Qd, tm.Qdd);
    if (Tau.rows() != 401 || Tau.cols() != 6 || tm.t.size() != 401) return;
    MatrixXd out(401, 7);
    out << tm.t, Tau.array().rowwise() / arm().tau_max.transpose().array();
    sbeg::write_csv("lab9_cpp_tau_AB.csv", {"t", "r1", "r2", "r3", "r4", "r5", "r6"}, out);
}

TEST_CASE("test_min_time_scale_rejects_static_overload") {
    lab::Robot heavy = arm();
    heavy.tau_max = 0.1 * arm().tau_max;  // cannot even hold the arm against gravity
    const lab::Trajectory tr = lab::joint_trajectory(tasks().A, tasks().B, 1.0, 21, 5);
    CHECK_THROWS_AS(lab::min_time_scale(heavy, tr.Q, tr.Qd, tr.Qdd), std::invalid_argument);
}

TEST_CASE("test_via_trajectory_on_course_arm_is_feasible_and_stretch_works") {
    const lab::Robot& R = arm();
    MatrixXd P(3, 6);
    P << tasks().A.transpose(), tasks().W.transpose(), tasks().B.transpose();
    const VectorXd& Tm = tasks().via_times;
    const MatrixXd V = lab::heuristic_via_velocities(P, Tm);
    CHECK_ALLCLOSE(V, as_mat(cases()["via_trajectory_on_course_arm_is_feasible_and_stretch_works"]["V"]), 1e-12);
    const VectorXd t = linspace(Tm(0), Tm(Tm.size() - 1), 481);
    const lab::ViaSamples B = lab::via_trajectory(P, V, Tm, t);
    const lab::TimeScaleResult res = lab::min_time_scale(R, B.B, B.Bd, B.Bdd);
    CHECK(res.c < 1.0);  // the given timing is feasible
    const double c = res.c;  // stretched trajectory touches a limit
    const MatrixXd Qd = B.Bd / c, Qdd = B.Bdd / (c * c);
    const MatrixXd Tau = lab::torque_profile(R, B.B, Qd, Qdd);
    const lab::LimitCheck out = lab::check_limits(R, Qd, Tau);
    REQUIRE(out.vel_ratio.size() == 6);
    REQUIRE(out.tau_ratio.size() == 6);
    CHECK(out.ok);
    CHECK_ISCLOSE_TOL(std::max(out.vel_ratio.maxCoeff(), out.tau_ratio.maxCoeff()), 1.0, 1e-5, 1e-9);
    CHECK_ISCLOSE_TOL(c, cases()["via_trajectory_on_course_arm_is_feasible_and_stretch_works"]["c"].get<double>(),
                      1e-9, 1e-12);
}

// ---------------------------------------------------------------- Part D: time-optimal time scaling (MR Sec. 9.4)

TEST_CASE("test_path_dynamics_2r_closed_form") {
    // Straight line: theta'' = 0, so m = M(theta) Delta and c = c(theta, Delta) (MR Eqs. 9.32, 8.9-8.10).
    const Setup t = two_r();
    for (const auto& c : cases()["path_dynamics_2r_closed_form"]) {
        const double s = c["s"].get<double>();
        INFO("s = " << s);
        const lab::PathDynamics pd = lab::path_dynamics(t.model, t.path, s);
        const Vector2d D(PI / 2, PI / 2);
        const Vector2d th = s * D;
        CHECK_MAT_NEAR(pd.m, M2(th) * D, 1e-9);
        CHECK_MAT_NEAR(pd.c, c2r(th, D), 1e-9);
        CHECK_MAT_NEAR(pd.g, Vector2d::Zero(), 1e-12);
        CHECK_MAT_NEAR(pd.m, as_vec(c["m"]), 1e-10);
        CHECK_MAT_NEAR(pd.c, as_vec(c["c"]), 1e-10);
    }
}

TEST_CASE("test_path_dynamics_reproduces_rnea_on_course_arm") {
    // m sdd + c sdot^2 + g = ID(theta, theta' sdot, theta' sdd + theta'' sdot^2) for any (sdot, sdd).
    const ArmTask a = arm_task();
    const json& C = cases()["path_dynamics_reproduces_rnea_on_course_arm"];
    // the given C++ course-arm model (0.1 kg payload) and spline path are the Python ones
    REQUIRE(a.r.Mlist.size() == 7);
    REQUIRE(a.r.Glist.size() == 6);
    CHECK_MAT_NEAR(a.r.Slist, as_mat(C["Slist"]), 1e-12);
    CHECK_MAT_NEAR(a.r.M, as_mat(C["M"]), 1e-12);
    for (std::size_t i = 0; i < 7; ++i) CHECK_MAT_NEAR(a.r.Mlist[i], as_mat(C["Mlist"][i]), 1e-12);
    for (std::size_t i = 0; i < 6; ++i) CHECK_MAT_NEAR(a.r.Glist[i], as_mat(C["Glist"][i]), 1e-12);
    CHECK_MAT_NEAR(a.r.limits, as_mat(C["limits"]), 1e-12);
    CHECK_MAT_NEAR(a.r.tau_max, as_vec(C["tau_max"]), 1e-12);
    CHECK_MAT_NEAR(a.r.qdot_max, as_vec(C["qdot_max"]), 1e-12);
    for (const auto& p : C["points"]) {
        const double s = p["s"].get<double>();
        INFO("s = " << s);
        const VectorXd th = a.path(s, 0), d1 = a.path(s, 1), d2 = a.path(s, 2);
        CHECK_MAT_NEAR(th, as_vec(p["theta"]), 1e-12);
        CHECK_MAT_NEAR(d1, as_vec(p["d1"]), 1e-12);
        CHECK_MAT_NEAR(d2, as_vec(p["d2"]), 1e-12);
        const lab::PathDynamics pd = lab::path_dynamics(a.r, a.path, s);
        for (const auto& d : p["draws"]) {
            const double sd = d["sd"].get<double>(), sdd = d["sdd"].get<double>();
            const VectorXd tau = mr::inverse_dynamics(th, d1 * sd, d1 * sdd + d2 * (sd * sd), a.r.g,
                                                      mr::Vector6d::Zero(), a.r.Mlist, a.r.Glist, a.r.Slist);
            CHECK_MAT_NEAR(pd.m * sdd + pd.c * (sd * sd) + pd.g, tau, 1e-10);
            CHECK_MAT_NEAR(tau, as_vec(d["tau"]), 1e-10);
        }
        CHECK_MAT_NEAR(pd.m, as_vec(p["m"]), 1e-10);
        CHECK_MAT_NEAR(pd.c, as_vec(p["c"]), 1e-10);
        CHECK_MAT_NEAR(pd.g, as_vec(p["g"]), 1e-10);
    }
}

TEST_CASE("test_accel_limits_cone_example") {
    // Motion-cone example of Lecture B: m = (2, -0.5), c = (0.4, 0.3), g = (1, 0), |tau| <= (5, 2), sdot = 2 ->
    // L = -1.6, U = 1.2; sdot = 3 is inadmissible (sdot_lim^2 = 7.5).
    const VectorXd m = vec({2.0, -0.5}), c = vec({0.4, 0.3}), g = vec({1.0, 0.0}), t = vec({5.0, 2.0});
    auto [L, U] = lab::accel_limits(m, c, g, 2.0, -t, t);
    CHECK_APPROX(L, -1.6);
    CHECK_APPROX(U, 1.2);
    std::tie(L, U) = lab::accel_limits(m, c, g, 3.0, -t, t);
    CHECK(L > U);
    std::tie(L, U) = lab::accel_limits(m, c, g, std::sqrt(7.5) - 1e-6, -t, t);
    CHECK(L <= U);
}

TEST_CASE("test_accel_limits_zero_inertia") {
    // A zero component of m(s) gives a pure velocity constraint (MR Sec. 9.4.4).
    const VectorXd m = vec({0.0, 2.0}), c = vec({1.0, 0.0}), g = vec({0.0, 0.0}), t = vec({1.0, 1.0});
    auto [L, U] = lab::accel_limits(m, c, g, 0.9, -t, t);
    CHECK_APPROX(L, -0.5);
    CHECK_APPROX(U, 0.5);
    std::tie(L, U) = lab::accel_limits(m, c, g, 1.1, -t, t);
    CHECK(L > U);
}

TEST_CASE("test_velocity_limit_2r_closed_form") {
    // MR 2R arm at s = 0.5: the VLC is where the lower bound of joint 1 meets the upper bound of joint 2:
    // (-20 - c1 x)/m1 = (10 - c2 x)/m2, x = sdot^2.
    const Setup t = two_r();
    const lab::PathDynamics pd = lab::path_dynamics(t.model, t.path, 0.5);
    REQUIRE(pd.m.size() == 2);
    REQUIRE(pd.c.size() == 2);
    const double x = (10 / pd.m(1) + 20 / pd.m(0)) / (pd.c(1) / pd.m(1) - pd.c(0) / pd.m(0));
    const double vl = lab::velocity_limit(t.model, t.path, 0.5, -TAU_2R, TAU_2R);
    CHECK_SCALAR_NEAR(vl, std::sqrt(x), 1e-6);
    CHECK_SCALAR_NEAR(std::sqrt(x), 2.1546, 1e-4);
    CHECK_SCALAR_NEAR(vl, cases()["velocity_limit_2r_closed_form"]["vl"].get<double>(), 1e-8);
}

TEST_CASE("test_velocity_limit_gantry_closed_form") {
    // Demining gantry on the MR Fig. 9.9 semicircle: closed-form VLC values at s = 0, 0.25, 0.5.
    const Setup t = gantry();
    const double R = 0.3, Mx = 40.0, my = 15.0, fx = 120.0, fy = 60.0;
    const double a = std::sqrt(0.5);
    const double A1 = fx / (Mx * PI * R * a), A2 = fy / (my * PI * R * a);
    auto vl = [&](double s) { return lab::velocity_limit(t.model, t.path, s, -F_GANTRY, F_GANTRY); };
    CHECK_SCALAR_NEAR(vl(0.25), std::sqrt((A1 + A2) / (2 * PI)), 1e-6);
    CHECK_SCALAR_NEAR(vl(0.5), std::sqrt(fy / (my * PI * PI * R)), 1e-6);
    CHECK_SCALAR_NEAR(vl(0.0), std::sqrt(fx / (Mx * PI * PI * R)), 1e-6);
    const json& c = cases()["velocity_limit_gantry_closed_form"];
    for (std::size_t i = 0; i < c["s"].size(); ++i)
        CHECK_SCALAR_NEAR(vl(c["s"][i].get<double>()), c["vl"][i].get<double>(), 1e-8);
}

TEST_CASE("test_velocity_limit_kinematic_and_course_arm") {
    const Setup t = two_r();
    const VectorXd big = vec({1e6, 1e6});
    const double v = lab::velocity_limit(t.model, t.path, 0.3, -big, big, vec({2.0, 2.0}));
    CHECK_SCALAR_NEAR(v, 2 / (PI / 2), 1e-6);
    const ArmTask a = arm_task();  // checkpoint D1: s = 0.5, the middle waypoint
    const double vt = lab::velocity_limit(a.r, a.path, 0.5, -a.tl, a.tl);
    const double vs = lab::velocity_limit(a.r, a.path, 0.5, -a.tl, a.tl, a.vmax);
    CHECK_SCALAR_NEAR(vt, 5.59, 0.01);
    CHECK_SCALAR_NEAR(vs, 0.931, 1e-3);
    const json& c = cases()["velocity_limit_kinematic_and_course_arm"];
    CHECK_SCALAR_NEAR(v, c["vl_kinematic"].get<double>(), 1e-8);
    CHECK_SCALAR_NEAR(vt, c["vl_arm_torque"].get<double>(), 1e-8);
    CHECK_SCALAR_NEAR(vs, c["vl_arm_speed"].get<double>(), 1e-8);
}

TEST_CASE("test_topp_2r_closed_form") {
    // MR 2R arm, |tau| <= (20, 10): the single-switch (bang-bang) solution of Lecture B takes 1.368 s.
    const Setup t = two_r();
    const lab::ToppResult tp = lab::topp(t.model, t.path, TAU_2R, VectorXd(), VectorXd(), 400);
    CHECK_SCALAR_NEAR(tp.T, 1.3683, 0.01 * 1.3683);  // rel = 0.01
    REQUIRE(tp.s_grid.size() == 401);
    REQUIRE(tp.sdot_grid.size() == 401);
    CHECK_SCALAR_NEAR(tp.sdot_grid(0), 0.0, 1e-6);
    CHECK_SCALAR_NEAR(tp.sdot_grid(400), 0.0, 1e-6);
    const json& c = cases()["topp_2r_closed_form"];
    CHECK_SCALAR_NEAR(tp.T, c["T"].get<double>(), 1e-6);  // the real toppra
    CHECK_MAT_NEAR(tp.sdot_grid, as_vec(c["sdot_grid"]), 1e-6);
    write_phase_csv("lab9_cpp_phase_2r.csv", t, TAU_2R, VectorXd(), tp);
}

TEST_CASE("test_topp_elbow_closed_form") {
    // Elbow exoskeleton, |tau| <= 10 N m: closed-form bang-bang time 0.3885 s; with |qdot| <= 3 rad/s 0.7466 s.
    const Setup t = elbow();
    const double T1 = lab::topp(t.model, t.path, vec({10.0}), VectorXd(), VectorXd(), 400).T;
    const double T2 = lab::topp(t.model, t.path, vec({10.0}), vec({3.0}), VectorXd(), 400).T;
    CHECK_SCALAR_NEAR(T1, 0.388528, 2e-3 * 0.388528);  // rel = 2e-3
    CHECK_SCALAR_NEAR(T2, 0.7466, 1e-3);
    const json& c = cases()["topp_elbow_closed_form"];
    CHECK_SCALAR_NEAR(T1, c["T_torque"].get<double>(), 1e-6);
    CHECK_SCALAR_NEAR(T2, c["T_speed"].get<double>(), 1e-6);
}

TEST_CASE("test_topp_gantry") {
    const Setup t = gantry();
    const lab::ToppResult tp = lab::topp(t.model, t.path, F_GANTRY, VectorXd(), VectorXd(), 400);
    CHECK_SCALAR_NEAR(tp.T, 1.1655, 3e-3);
    REQUIRE(tp.s_grid.size() == 401);
    REQUIRE(tp.sdot_grid.size() == 401);
    const double sd = interp1(0.5, tp.s_grid, tp.sdot_grid);
    CHECK(sd <= lab::velocity_limit(t.model, t.path, 0.5, -F_GANTRY, F_GANTRY) + 1e-3);
    const json& c = cases()["topp_gantry"];
    CHECK_SCALAR_NEAR(tp.T, c["T"].get<double>(), 1e-6);
    CHECK_MAT_NEAR(tp.sdot_grid, as_vec(c["sdot_grid"]), 1e-6);
    write_phase_csv("lab9_cpp_phase_gantry.csv", t, F_GANTRY, VectorXd(), tp);
}

TEST_CASE("test_topp_torque_scaling") {
    // g = 0 and pure torque limits: scaling the limits by k scales T by 1/sqrt(k).
    const Setup t = two_r();
    const double T1 = lab::topp(t.model, t.path, TAU_2R, VectorXd(), VectorXd(), 300).T;
    const double T2 = lab::topp(t.model, t.path, 0.6 * TAU_2R, VectorXd(), VectorXd(), 300).T;
    CHECK_SCALAR_NEAR(T2, T1 / std::sqrt(0.6), 5e-3 * T1 / std::sqrt(0.6));  // rel = 5e-3
    const json& c = cases()["topp_torque_scaling"];
    CHECK_SCALAR_NEAR(T1, c["T1"].get<double>(), 1e-6);
    CHECK_SCALAR_NEAR(T2, c["T2"].get<double>(), 1e-6);
}

TEST_CASE("test_topp_course_arm_respects_limits") {
    const ArmTask a = arm_task();
    const lab::ToppResult hw = hw_topp(a);
    const lab::PeakUsage pu = lab::peak_usage(a.r, hw.traj, hw.T, a.tl, a.vmax, 2000);
    CHECK(pu.vel_ratio.maxCoeff() <= 1.01);
    CHECK(pu.tau_ratio.maxCoeff() <= 1.03);
    double qdd_peak = 0.0;
    for (int k = 0; k <= 2000; ++k) qdd_peak = std::max(qdd_peak, hw.traj(hw.T * k / 2000.0, 2).cwiseAbs().maxCoeff());
    CHECK(qdd_peak <= a.task.hw_qddot_max * 1.01);
    CHECK_SCALAR_NEAR(hw.T, 1.5656, 0.01);
    CHECK_MAT_NEAR(hw.traj(0.0, 0), a.task.waypoints.row(0), 1e-6);
    CHECK_MAT_NEAR(hw.traj(hw.T, 0), a.task.waypoints.row(a.task.waypoints.rows() - 1), 1e-6);
    const json& c = cases()["topp_course_arm_respects_limits"];
    CHECK_SCALAR_NEAR(hw.T, c["T"].get<double>(), 1e-6);
    CHECK_MAT_NEAR(hw.sdot_grid, as_vec(c["sdot_grid"]), 1e-6);
    CHECK_MAT_NEAR(pu.vel_ratio, as_vec(c["vel_ratio"]), 1e-5);
    CHECK_MAT_NEAR(pu.tau_ratio, as_vec(c["tau_ratio"]), 1e-5);
    // plot data (Part D): the hw phase plane of the course-arm task (velocity limit curve with the 50 % speed limit)
    write_phase_csv("lab9_cpp_phase_course_arm.csv", {a.r, a.path}, a.tl, a.vmax, hw);
}

TEST_CASE("test_min_time_scale_quintic_elbow_closed_form") {
    // The quintic of Part A on a Part D model: torque-limited 0.5231 s; speed-limited (3 rad/s):
    // T = (15/8) D / 3 = 1.3090 s (MR Fig. 9.4 peak speed).
    const Setup t = elbow();
    const Samples q = quintic_samples(t.path);
    lab::Robot rob = t.model;
    rob.tau_max = vec({10.0});
    rob.qdot_max = vec({INF});
    const lab::TimeScaleResult r1 = lab::min_time_scale(rob, q.Q, q.Qd, q.Qdd);
    CHECK_SCALAR_NEAR(r1.c, 0.52312, 1e-3);
    CHECK(r1.binding == std::make_pair(std::string("torque"), 0));
    rob.qdot_max = vec({3.0});
    const lab::TimeScaleResult r2 = lab::min_time_scale(rob, q.Q, q.Qd, q.Qdd);
    CHECK_SCALAR_NEAR(r2.c, 15.0 / 8 * (2 * PI / 3) / 3, 1e-3);
    CHECK(r2.binding == std::make_pair(std::string("velocity"), 0));
    const json& c = cases()["min_time_scale_quintic_elbow_closed_form"];
    CHECK_SCALAR_NEAR(r1.c, c["c_torque"].get<double>(), 1e-9);
    CHECK_SCALAR_NEAR(r2.c, c["c_speed"].get<double>(), 1e-9);
}

TEST_CASE("test_topp_beats_quintic_on_course_arm") {
    const ArmTask a = arm_task();
    const Samples q = quintic_samples(a.path);
    for (const auto& c : cases()["topp_beats_quintic_on_course_arm"]) {
        const std::string name = c["case"].get<std::string>();
        INFO("case = " << name);
        const double speed = c["speed"].get<double>(), acc = c["acc"].get<double>();
        const VectorXd vmax = speed * a.r.qdot_max;
        const VectorXd amax = acc > 0.0 ? VectorXd(VectorXd::Constant(6, acc)) : VectorXd();
        const lab::ToppResult tp = lab::topp(a.r, a.path, a.tl, vmax, amax, a.task.grid_N);
        lab::Robot rob = a.r;
        rob.tau_max = a.tl;
        rob.qdot_max = vmax;
        const lab::TimeScaleResult res = lab::min_time_scale(rob, q.Q, q.Qd, q.Qdd, amax);
        CHECK_SCALAR_NEAR(tp.T, c["T_topp"].get<double>(), 1e-6);  // the real toppra
        CHECK_SCALAR_NEAR(res.c, c["T_quintic"].get<double>(), 1e-9);
        CHECK(res.binding.first == c["binding"][0].get<std::string>());
        CHECK(1.2 < res.c / tp.T);
        CHECK(res.c / tp.T < 2.0);
    }
}

// ---------------------------------------------------------------- Part E: the command file (provided functions)

TEST_CASE("test_command_file_for_course_arm") {
    // The hw TOPP plan stretched by slow = 2, sampled at the 20 ms servo period, passes the checklist check at
    // 25 % rated speed with the 10 deg margin; the unstretched plan does not.
    const ArmTask a = arm_task();
    const lab::ToppResult hw = hw_topp(a);
    const double dt = a.task.servo_period_s;
    const auto [t, q] = lab::command_table(hw.traj, hw.T, dt, 2.0);
    REQUIRE(t.size() == 158);
    CHECK((q.rows() == 158 && q.cols() == 6));
    CHECK_SCALAR_NEAR(t(t.size() - 1), 3.131, 2e-3);
    const lab::CommandCheck ok = lab::check_command(t, q, a.r, a.task.first_run_speed_fraction, a.task.joint_margin_deg);
    CHECK(ok.ok);
    CHECK(0.9 < ok.peak_speed_ratio);
    CHECK(ok.peak_speed_ratio <= 1.01);
    const auto [t1, q1] = lab::command_table(hw.traj, hw.T, dt, 1.0);
    const lab::CommandCheck bad =
        lab::check_command(t1, q1, a.r, a.task.first_run_speed_fraction, a.task.joint_margin_deg);
    CHECK_FALSE(bad.ok);
    CHECK(bad.peak_speed_ratio > 1.8);
    const json& c = cases()["command_file_for_course_arm"];
    CHECK_SCALAR_NEAR(t(t.size() - 1), c["t_end"].get<double>(), 1e-6);
    check_command_dict(ok, c["first_run"]);
    check_command_dict(bad, c["unstretched"]);
}
