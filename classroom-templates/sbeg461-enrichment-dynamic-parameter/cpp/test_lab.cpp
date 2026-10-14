// Auto-checks for the SBEG 461 enrichment lab "Dynamic parameter identification from hardware logs" -- C++ track.
// Mirrors test_lab.py test by test (test_starter_compiles has no C++ counterpart: building is the check).
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*inertial*,*wrench*,*regressor_matches*"      (run only some tests, like pytest -k)
// Ground truth: mrcpp (the C++ port of modern_robotics: inverse_dynamics with general spatial inertias, MR Eq. 8.43;
// ad; gravity_forces), the rigid-body equation G dV - [ad_V]^T G V (MR Eq. 8.40), finite differences, synthetic data
// with known parameters, the weighed mass of the EOD training package, and cpp_cases.json, which holds the numbers of
// the Python reference (including the random inputs drawn by test_lab.py and the SciPy filter output for one log),
// so both languages are checked against the same values.  The hidden "true" parameters of the simulated robots are
// NOT used: on the logs your model is judged, as on real hardware, by its prediction of held-out data.
// The two data/infrastructure tests also call one of your functions (marked "C++ only"), so that every test case
// fails until your code is written.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
using lab::Matrix6d;
using lab::Vector6d;
using sbeg::as;
using sbeg::as_list;
using sbeg::as_mat;
using sbeg::as_vec;
using sbeg::json;

namespace {
constexpr double PI = 3.14159265358979323846;

const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

/// META = lab.load_meta() (given helper).
const json& meta() {
    static const json M = lab::load_meta();
    return M;
}

double num(const json& j) { return j.get<double>(); }

VectorXd vec(std::initializer_list<double> v) {
    VectorXd x(static_cast<Eigen::Index>(v.size()));
    Eigen::Index i = 0;
    for (double e : v) x(i++) = e;
    return x;
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

/// np.isclose(got, expected, rtol, atol) for two scalars.
bool isclose(double got, double expected, double rtol = 1e-5, double atol = 1e-8) {
    return std::isfinite(got) && std::abs(got - expected) <= atol + rtol * std::abs(expected);
}

#define CHECK_ALLCLOSE(got, expected, atol)                                                          \
    do {                                                                                             \
        const ::sbeg::NearResult lab_r_ = allclose((got), (expected), (atol));                        \
        CHECK_MESSAGE(lab_r_.ok, "allclose(" #got ", " #expected ")\n  " << lab_r_.msg);             \
    } while (false)

/// np.linalg.matrix_rank (SVD, tolerance s_max max(rows, cols) eps).
int matrix_rank(const MatrixXd& A) {
    const Eigen::JacobiSVD<MatrixXd> svd(A);
    const VectorXd& s = svd.singularValues();
    const double tol = s.maxCoeff() * static_cast<double>(std::max(A.rows(), A.cols())) *
                       std::numeric_limits<double>::epsilon();
    return static_cast<int>((s.array() > tol).count());
}

/// Columns idx of A (A[:, idx]); every index is checked first.
MatrixXd take_cols(const MatrixXd& A, const std::vector<int>& idx) {
    MatrixXd out(A.rows(), static_cast<Eigen::Index>(idx.size()));
    for (std::size_t c = 0; c < idx.size(); ++c) {
        REQUIRE_MESSAGE((idx[c] >= 0 && idx[c] < A.cols()), "index " << idx[c] << " out of range");
        out.col(static_cast<Eigen::Index>(c)) = A.col(idx[c]);
    }
    return out;
}

/// Position of a name in a list (Python list.index); fails the test if absent.
std::size_t index_of(const std::vector<std::string>& names, const std::string& name) {
    const auto it = std::find(names.begin(), names.end(), name);
    REQUIRE_MESSAGE(it != names.end(), "'" << name << "' is not in the list of base parameters");
    return static_cast<std::size_t>(it - names.begin());
}

/// np.arange(start, stop, step) for doubles (numpy's length and fill rule).
VectorXd arange(double start, double stop, double step) {
    const auto len = static_cast<Eigen::Index>(std::ceil((stop - start) / step));
    const double delta = (start + step) - start;
    VectorXd t(len);
    for (Eigen::Index i = 0; i < len; ++i) t(i) = start + static_cast<double>(i) * delta;
    return t;
}

/// np.linspace(start, stop, num).
VectorXd linspace(double start, double stop, Eigen::Index num_pts) {
    const double step = (stop - start) / static_cast<double>(num_pts - 1);
    VectorXd t(num_pts);
    for (Eigen::Index i = 0; i < num_pts; ++i) t(i) = static_cast<double>(i) * step + start;
    t(num_pts - 1) = stop;
    return t;
}

/// load(name) of test_lab.py: process_log(load_log(data/name), torque constant of its robot); computed once per log.
const lab::ProcessedLog& load(const std::string& name) {
    static std::map<std::string, lab::ProcessedLog> cache;
    const auto it = cache.find(name);
    if (it != cache.end()) return it->second;
    const std::string key = meta()["logs"][name]["robot"].get<std::string>();
    lab::ProcessedLog d =
        lab::process_log(lab::load_log(lab::data_path(name)), as_vec(meta()["robots"][key]["torque_constant"]));
    return cache.emplace(name, std::move(d)).first->second;
}

struct Identified {
    lab::Robot robot;
    lab::BaseParams base;
    lab::WlsResult fit;
    lab::Validation val;
    lab::Validation nominal;
};

/// The module-scope fixture `identified` of test_lab.py (computed once; a failure is reported by every user).
const std::map<std::string, Identified>& identified() {
    static const std::map<std::string, Identified> out = [] {
        std::map<std::string, Identified> m;
        for (const std::string key : {"eod_3r", "course_arm"}) {
            Identified d;
            d.robot = lab::get_robot(key);
            const int n = d.robot.n;
            d.base = lab::base_parameters(d.robot);
            const lab::LinearSystem S = lab::build_system(d.robot, load(key + "_excite.csv"), d.base.idx_base);
            const lab::LinearSystem Sv = lab::build_system(d.robot, load(key + "_validate.csv"), d.base.idx_base);
            d.fit = lab::wls(S.Y, S.tau, S.joint, n);
            d.val = lab::validate(Sv.Y, d.fit.beta, Sv.tau, Sv.joint, n);
            d.nominal = lab::validate(Sv.Y, d.base.E * lab::nominal_parameters(d.robot), Sv.tau, Sv.joint, n);
            m.emplace(key, std::move(d));
        }
        return m;
    }();
    return out;
}
}  // namespace

// ============================================================================ data and infrastructure (provided code)
TEST_CASE("test_meta_and_logs") {
    const json& M = meta();
    std::set<std::string> keys;
    for (auto it = M["robots"].begin(); it != M["robots"].end(); ++it) keys.insert(it.key());
    CHECK(keys == std::set<std::string>{"course_arm", "eod_3r"});
    for (auto it = M["logs"].begin(); it != M["logs"].end(); ++it) {
        INFO("log " << it.key());
        const lab::HwLog log = lab::load_log(lab::data_path(it.key()));
        const int n = lab::get_robot(it.value()["robot"].get<std::string>()).n;
        CHECK(log.cmd.cols() == n);
        CHECK(log.cur.cols() == n);
        REQUIRE(log.t.size() > 2);
        VectorXd dt = log.t.tail(log.t.size() - 1) - log.t.head(log.t.size() - 1);
        CHECK((dt.array() > 0.0).all());  // sorted, duplicates dropped
        std::sort(dt.data(), dt.data() + dt.size());
        const Eigen::Index m = dt.size();
        const double median = m % 2 ? dt(m / 2) : 0.5 * (dt(m / 2 - 1) + dt(m / 2));
        CHECK(400.0 < 1.0 / median);  // about 500 Hz
        CHECK(1.0 / median < 600.0);
    }
    CHECK(num(M["package_mass_weighed"]) == 1.20);
    // C++ only: your process_log keeps the expected number of samples of the EOD excitation log.
    CHECK(load("eod_3r_excite.csv").t.size() == cases()["meta_and_logs"]["eod_excite_samples"].get<Eigen::Index>());
}

TEST_CASE("test_robot_models_are_the_course_models") {
    const lab::Robot ca = lab::course_arm(), eod = lab::eod_3r();
    double mass = 0.0;
    for (const auto& G : ca.Glist) mass += G(3, 3);
    CHECK(isclose(mass, 0.67));
    const VectorXd g0 = mr::gravity_forces(VectorXd::Zero(3), eod.g, eod.Mlist, eod.Glist, eod.Slist);
    CHECK_ALLCLOSE(g0, vec({106.44, 32.86, 3.43}), 0.01);  // value quoted in the course robot models
    CHECK_ALLCLOSE(eod.tau_max, vec({155, 70, 15}), 1e-8);
    // C++ only: the masses in your parameter vector (nominal_parameters calls inertial_parameters).
    const VectorXd nom = lab::nominal_parameters(ca);
    REQUIRE(nom.size() == 78);
    double m_pi = 0.0;
    for (int i = 0; i < 6; ++i) m_pi += nom(10 * i);
    CHECK(isclose(m_pi, 0.67));
}

// ============================================================================ Part A: linear-in-parameters form
TEST_CASE("test_inertial_parameters_round_trip") {
    const json& c = cases()["inertial_parameters_round_trip"];
    std::vector<Matrix6d> Gs = lab::course_arm().Glist;
    for (const auto& G : as_list<Matrix6d>(c["random_G"])) Gs.push_back(G);
    const std::vector<VectorXd> p_ref = as_list<VectorXd>(c["p"]);
    for (std::size_t k = 0; k < Gs.size(); ++k) {
        INFO("G number " << k);
        const VectorXd p = lab::inertial_parameters(Gs[k]);
        REQUIRE(p.size() == 10);
        CHECK_ALLCLOSE(lab::spatial_inertia(p), Gs[k], 1e-14);
        if (k >= 6) CHECK_MAT_NEAR(p, p_ref[k - 6], 1e-14);
    }
    const VectorXd p = lab::inertial_parameters(as<Matrix6d>(c["G_m2"]));
    REQUIRE(p.size() == 10);
    CHECK(isclose(p(0), 2.0));
}

TEST_CASE("test_wrench_regressor_is_the_rigid_body_equation") {
    for (const auto& c : cases()["wrench_regressor_is_the_rigid_body_equation"]) {
        const Matrix6d G = as<Matrix6d>(c["G"]);
        const Vector6d V = as<Vector6d>(c["V"]), dV = as<Vector6d>(c["dV"]);
        const MatrixXd Phi = lab::wrench_regressor(V, dV);
        REQUIRE(Phi.rows() == 6);
        REQUIRE(Phi.cols() == 10);
        const VectorXd rhs = G * dV - mr::ad(V).transpose() * G * V;
        CHECK_ALLCLOSE(Phi * lab::inertial_parameters(G), rhs, 1e-12);
        CHECK_MAT_NEAR(rhs, as_vec(c["rhs"]), 1e-12);
    }
}

TEST_CASE("test_regressor_matches_modern_robotics") {
    for (const std::string key : {"course_arm", "eod_3r"}) {
        INFO("robot " << key);
        const json& c = cases()["regressor_matches_modern_robotics"][key];
        const lab::Robot r = lab::get_robot(key);
        const int n = r.n;
        const std::vector<Matrix6d> Gl = as_list<Matrix6d>(c["Glist"]);
        const VectorXd Ia = as_vec(c["Ia"]), bv = as_vec(c["bv"]), bc = as_vec(c["bc"]);
        VectorXd pi(13 * n);
        for (int i = 0; i < n; ++i) {
            const VectorXd p = lab::inertial_parameters(Gl[static_cast<std::size_t>(i)]);
            REQUIRE(p.size() == 10);
            pi.segment(10 * i, 10) = p;
        }
        pi.tail(3 * n) << Ia, bv, bc;
        for (const auto& s : c["states"]) {
            const VectorXd q = as_vec(s["q"]), dq = as_vec(s["dq"]), ddq = as_vec(s["ddq"]);
            const MatrixXd Y = lab::regressor(r, q, dq, ddq);
            REQUIRE(Y.rows() == n);
            REQUIRE(Y.cols() == 13 * n);
            VectorXd ref = mr::inverse_dynamics(q, dq, ddq, r.g, Vector6d::Zero(), r.Mlist, Gl, r.Slist);
            ref += Ia.cwiseProduct(ddq) + bv.cwiseProduct(dq) + bc.cwiseProduct(lab::sign(dq));
            CHECK_ALLCLOSE(Y * pi, ref, 1e-10);
            CHECK_ALLCLOSE(ref, as_vec(s["ref"]), 1e-10);  // Python modern_robotics
        }
    }
}

// ============================================================================ Part B: base parameters
TEST_CASE("test_base_parameters") {
    for (const std::string key : {"eod_3r", "course_arm"}) {
        INFO("robot " << key);
        const json& c = cases()["base_parameters"][key];
        const int rank = c["rank"].get<int>();
        const lab::Robot r = lab::get_robot(key);
        const int n = r.n;
        const lab::BaseParams b = lab::base_parameters(r);
        CHECK(b.rank == rank);
        REQUIRE(b.E.rows() == rank);
        REQUIRE(b.E.cols() == 13 * n);
        REQUIRE(static_cast<int>(b.idx_base.size()) == rank);
        CHECK(static_cast<int>(b.idx_dep.size()) == 13 * n - rank);
        const VectorXd pi = as_vec(c["pi"]);
        // states NOT used to find the base parameters (Python: random_states(r, 20, seed=99), stored)
        const MatrixXd W = lab::stack_regressor(r, as_mat(c["Q"]), as_mat(c["dQ"]), as_mat(c["ddQ"]));
        const MatrixXd Wb = take_cols(W, b.idx_base);
        CHECK_ALLCLOSE(W * pi, Wb * (b.E * pi), 1e-9);
        CHECK(matrix_rank(Wb) == rank);
        const std::vector<std::string> names = lab::param_names(n);
        // joint-1 rotor inertia adds to a link-1 inertia: its column equals another one, and only ONE of the
        // identical columns can be a base parameter (which one, if any, depends on the pivoting of the QR)
        const int k_ia = static_cast<int>(std::find(names.begin(), names.end(), "Ia1") - names.begin());
        std::set<int> group{k_ia};
        for (int k = 0; k < W.cols(); ++k)
            if (k != k_ia && allclose(W.col(k), W.col(k_ia), 1e-9).ok) group.insert(k);
        CHECK(group.size() > 1);
        int in_base = 0;
        for (int k : b.idx_base) in_base += static_cast<int>(group.count(k));
        CHECK(in_base <= 1);
        for (int j = 1; j <= n; ++j) {  // friction always identifiable
            const std::string bc = "bc" + std::to_string(j);
            CHECK_MESSAGE(std::find(b.names.begin(), b.names.end(), bc) != b.names.end(), bc << " is not a base parameter");
        }
    }
}

// ============================================================================ Part C: logs
TEST_CASE("test_process_log_on_a_clean_sine") {
    const double fs = 500.0;
    const VectorXd k = vec({2.0, 0.5});
    const VectorXd t = arange(0.0, 12.0, 1.0 / fs);
    const double w = 2 * PI * 0.4;
    const Eigen::Index N = t.size();
    lab::HwLog log;
    log.t = t;
    log.meas.resize(N, 2);
    log.cur.resize(N, 2);
    for (Eigen::Index i = 0; i < N; ++i) {
        log.meas.row(i) << 0.5 * std::sin(w * t(i)), 0.3 * std::cos(w * t(i));
        log.cur.row(i) << std::cos(w * t(i)), std::sin(w * t(i));
    }
    log.cmd = log.meas;
    const lab::ProcessedLog d = lab::process_log(log, k, fs, 4.0, 4, 5, 1.0);
    const Eigen::Index M = d.t.size();
    REQUIRE(M > 2);
    REQUIRE(d.q.rows() == M);
    REQUIRE(d.q.cols() == 2);
    REQUIRE(d.dq.rows() == M);
    REQUIRE(d.ddq.rows() == M);
    REQUIRE(d.tau.rows() == M);
    CHECK(d.t(0) >= 1.0 - 1e-9);
    CHECK(d.t(M - 1) <= t(N - 1) - 1.0 + 1e-9);
    const VectorXd dtt = d.t.tail(M - 1) - d.t.head(M - 1);
    CHECK_ALLCLOSE(dtt, VectorXd::Constant(M - 1, 5 / fs), 1e-8);
    const VectorXd& tt = d.t;
    const VectorXd s = (w * tt).array().sin(), co = (w * tt).array().cos();
    CHECK_ALLCLOSE(d.q.col(0), 0.5 * s, 1e-3);
    CHECK_ALLCLOSE(d.dq.col(0), 0.5 * w * co, 1e-2);
    CHECK_ALLCLOSE(d.ddq.col(1), -0.3 * w * w * co, 2e-2);
    MatrixXd tau_ref(M, 2);
    tau_ref << k(0) * co, k(1) * s;
    CHECK_ALLCLOSE(d.tau, tau_ref, 1e-2);

    // C++ only: the given butter / filtfilt are SciPy's, and your process_log gives the numbers of the Python
    // reference on a real log (rows of the processed eod_3r_excite.csv).
    const json& c = cases()["process_log_on_a_clean_sine"];
    const auto ba = lab::butter(4, 4.0 / 250.0);
    CHECK(allclose(ba.first, as_vec(c["butter_b"]), 0.0, 1e-10).ok);
    CHECK(allclose(ba.second, as_vec(c["butter_a"]), 0.0, 1e-10).ok);
    const json& e = c["eod_excite"];
    const lab::ProcessedLog& p = load("eod_3r_excite.csv");
    MatrixXd q(0, 0), dq, ddq, tau;
    VectorXd tp;
    const std::vector<int> rows = e["rows"].get<std::vector<int>>();
    tp.resize(static_cast<Eigen::Index>(rows.size()));
    q.resize(tp.size(), 3);
    dq.resize(tp.size(), 3);
    ddq.resize(tp.size(), 3);
    tau.resize(tp.size(), 3);
    REQUIRE(p.t.size() > rows.back());
    REQUIRE(p.q.rows() == p.t.size());
    REQUIRE(p.q.cols() == 3);
    REQUIRE(p.tau.cols() == 3);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto ii = static_cast<Eigen::Index>(i);
        tp(ii) = p.t(rows[i]);
        q.row(ii) = p.q.row(rows[i]);
        dq.row(ii) = p.dq.row(rows[i]);
        ddq.row(ii) = p.ddq.row(rows[i]);
        tau.row(ii) = p.tau.row(rows[i]);
    }
    CHECK_MAT_NEAR(tp, as_vec(e["t"]), 1e-12);
    CHECK_MAT_NEAR(q, as_mat(e["q"]), 1e-9);
    CHECK_MAT_NEAR(dq, as_mat(e["dq"]), 1e-7);
    CHECK_MAT_NEAR(ddq, as_mat(e["ddq"]), 1e-5);
    const sbeg::NearResult tau_ok = allclose(tau, as_mat(e["tau"]), 1e-9, 1e-8);  // N m, up to 90: relative
    CHECK_MESSAGE(tau_ok.ok, "allclose(tau, Python tau, atol 1e-9, rtol 1e-8)\n  " << tau_ok.msg);
}

TEST_CASE("test_moving_rows") {
    MatrixXd dq(2, 2);
    dq << 0.0, 0.5, -0.2, 0.05;
    CHECK(lab::moving_rows(dq, 0.1) == std::vector<bool>{false, true, true, false});
}

// ============================================================================ Part D: estimation
TEST_CASE("test_ols_on_synthetic_data") {
    const json& c = cases()["ols_on_synthetic_data"];
    const MatrixXd Y = as_mat(c["Y"]);
    const VectorXd tau = as_vec(c["tau"]);  // Y beta + noise, beta = [1, -2, 0.5, 3]
    const lab::OlsResult out = lab::ols(Y, tau);
    const VectorXd ref = as_vec(c["ref"]);   // np.linalg.lstsq
    CHECK_MAT_NEAR(ref, Y.colPivHouseholderQr().solve(tau), 1e-12);
    CHECK_ALLCLOSE(out.beta, ref, 1e-8);
    const double s2 = num(c["s2"]);
    CHECK(isclose(out.sigma2, s2));
    CHECK_ALLCLOSE(out.cov, s2 * (Y.transpose() * Y).inverse(), 1e-8);
    REQUIRE(out.cov.rows() == 4);
    CHECK_ALLCLOSE(out.rel_std, 100.0 * out.cov.diagonal().cwiseSqrt().cwiseQuotient(ref.cwiseAbs()), 1e-8);
    CHECK(isclose(out.cond, num(c["cond"])));
}

TEST_CASE("test_wls_finds_the_noise_levels") {
    const json& c = cases()["wls_finds_the_noise_levels"];
    const int N = c["N"].get<int>(), n = c["n"].get<int>();
    const MatrixXd Y = as_mat(c["Y"]);
    const VectorXd tau = as_vec(c["tau"]);  // Y [1, 2, 3] + sd[joint] noise, sd = [0.01, 0.5]
    const VectorXi joint = lab::joint_index(N, n);
    const lab::WlsResult out = lab::wls(Y, tau, joint, n);
    REQUIRE(out.sigma.size() == 2);
    REQUIRE(out.beta.size() == 3);
    REQUIRE(out.cov.rows() == 3);
    CHECK(isclose(out.sigma(1), 0.5, 0.05));
    CHECK(out.sigma(0) < 0.05);  // the OLS residual of the quiet joint also contains the OLS estimation error
    // caused by the noisy joint: why weighting is needed
    const VectorXd sd = out.cov.diagonal().cwiseSqrt();
    CHECK(((out.beta - vec({1, 2, 3})).cwiseAbs().array() < 4.0 * sd.array()).all());
    CHECK(sd(0) < std::sqrt(lab::ols(Y, tau).cov(0, 0)));  // weighting helps
}

TEST_CASE("test_validate") {
    const MatrixXd Y = MatrixXd::Identity(4, 4);
    VectorXi joint(4);
    joint << 0, 1, 0, 1;
    const lab::Validation v = lab::validate(Y, vec({1.0, 2.0, 3.0, 4.0}), vec({1.0, 2.0, 2.0, 4.0}), joint, 2);
    CHECK_ALLCLOSE(v.rms_err, vec({std::sqrt(0.5), 0.0}), 1e-8);
    CHECK_ALLCLOSE(v.nrms, vec({std::sqrt(0.5) / std::sqrt(2.5), 0.0}), 1e-8);
}

TEST_CASE("test_identified_eod_arm_predicts_held_out_torque") {
    const Identified& d = identified().at("eod_3r");
    REQUIRE(d.val.nrms.size() == 3);
    REQUIRE(d.nominal.nrms.size() == 3);
    INFO("held-out NRMS per joint: " << d.val.nrms.transpose() << "   CAD model: " << d.nominal.nrms.transpose());
    CHECK((d.val.nrms.array() < 0.05).all());                           // held-out data, within 5 % per joint
    CHECK((d.val.nrms.array() < 0.5 * d.nominal.nrms.array()).all());   // much better than the CAD model
    const std::vector<std::string>& names = d.base.names;
    REQUIRE(d.fit.beta.size() == static_cast<Eigen::Index>(names.size()));
    REQUIRE(d.fit.rel_std.size() == d.fit.beta.size());
    for (int j = 1; j <= 3; ++j) {
        const auto ic = static_cast<Eigen::Index>(index_of(names, "bc" + std::to_string(j)));
        const auto iv = static_cast<Eigen::Index>(index_of(names, "bv" + std::to_string(j)));
        CHECK(d.fit.beta(ic) > 0.0);
        CHECK(d.fit.beta(iv) > 0.0);
        CHECK(d.fit.rel_std(ic) < 2.0);  // well identified
    }
}

TEST_CASE("test_identified_course_arm_predicts_held_out_torque") {
    const Identified& d = identified().at("course_arm");
    REQUIRE(d.val.nrms.size() == 6);
    REQUIRE(d.nominal.nrms.size() == 6);
    INFO("held-out NRMS per joint: " << d.val.nrms.transpose() << "   CAD model: " << d.nominal.nrms.transpose()
                                     << "   cond: " << d.fit.cond);
    CHECK(d.val.nrms(1) < 0.05);  // shoulder and elbow
    CHECK(d.val.nrms(2) < 0.05);
    CHECK((d.val.nrms.array() < 0.2).all());  // every joint, incl. the noisy wrist
    CHECK((d.val.nrms.array() < 0.5 * d.nominal.nrms.array()).all());
    CHECK(200.0 < d.fit.cond);  // weighted, base regressor
    CHECK(d.fit.cond < 5000.0);
}

// ============================================================================ Part E: excitation
TEST_CASE("test_fourier_trajectory_derivatives_and_period") {
    const json& c = cases()["fourier_trajectory_derivatives_and_period"];
    const double wf = 2 * PI * 0.1;
    const MatrixXd a = as_mat(c["a"]), b = as_mat(c["b"]);
    const VectorXd t = linspace(0.0, 10.0, 2001);
    const VectorXd q0 = vec({0.1, -0.2});
    const lab::Trajectory tr = lab::fourier_trajectory(t, q0, a, b, wf);
    REQUIRE(tr.q.rows() == 2001);
    REQUIRE(tr.q.cols() == 2);
    REQUIRE(tr.dq.rows() == 2001);
    REQUIRE(tr.dq.cols() == 2);
    REQUIRE(tr.ddq.rows() == 2001);
    REQUIRE(tr.ddq.cols() == 2);
    const double h = t(1) - t(0);
    const Eigen::Index m = 2001 - 4;  // rows [2:-2]
    CHECK_ALLCLOSE(lab::gradient(tr.q, h).middleRows(2, m), tr.dq.middleRows(2, m), 1e-4);
    CHECK_ALLCLOSE(lab::gradient(tr.dq, h).middleRows(2, m), tr.ddq.middleRows(2, m), 1e-4);
    CHECK_ALLCLOSE(tr.q.row(0), tr.q.row(2000), 1e-8);  // one period = 10 s
    CHECK_ALLCLOSE(tr.dq.row(0), tr.dq.row(2000), 1e-8);
    CHECK_ALLCLOSE(tr.q.colwise().mean(), q0, 0.02);  // centred on q0
    CHECK_ALLCLOSE(tr.dq.row(0), a.rowwise().sum(), 1e-8);
}

TEST_CASE("test_optimised_excitation_beats_a_single_sine") {
    const lab::Robot r = lab::eod_3r();
    const lab::BaseParams base = lab::base_parameters(r);
    const json& ex = meta()["robots"]["eod_3r"]["excitation"];
    const double f_f = num(ex["f_f"]);
    const double wf = 2 * PI * f_f;
    const VectorXd t = linspace(0.0, 1.0 / f_f, 201).head(200);
    const VectorXd q0 = as_vec(ex["q0"]);
    const lab::Trajectory opt = lab::fourier_trajectory(t, q0, as_mat(ex["a"]), as_mat(ex["b"]), wf);
    const double c_opt = lab::excitation_cost(r, base, opt.q, opt.dq, opt.ddq, 2);
    MatrixXd a1 = MatrixXd::Zero(3, 5);
    a1.col(0).setConstant(0.4);  // every joint the same single sine
    const lab::Trajectory sine = lab::fourier_trajectory(t, q0, a1, MatrixXd::Zero(3, 5), wf);
    const double c_sine = lab::excitation_cost(r, base, sine.q, sine.dq, sine.ddq, 2);
    INFO("cond optimised = " << c_opt << ", single sine = " << c_sine);
    CHECK(c_opt < 20.0);
    CHECK(c_sine > 10.0 * c_opt);
}

// ============================================================================ Part F: physical consistency
TEST_CASE("test_pseudo_inertia_and_consistency") {
    const Eigen::Vector3d p(0.1, 0.0, 0.0);
    const double m = 2.0;  // point mass
    const Eigen::Matrix3d S = m * (p.dot(p) * Eigen::Matrix3d::Identity() - p * p.transpose());
    VectorXd pi(10);
    pi << m, m * p(0), m * p(1), m * p(2), S(0, 0), S(0, 1), S(0, 2), S(1, 1), S(1, 2), S(2, 2);
    const Eigen::Matrix4d J = lab::pseudo_inertia(pi);
    const Eigen::Vector4d p1(p(0), p(1), p(2), 1.0);
    CHECK_ALLCLOSE(J, m * p1 * p1.transpose(), 1e-8);  // rank 1: a point is the limit case
    const lab::Robot r = lab::course_arm();
    const VectorXd nom = lab::nominal_parameters(r, VectorXd::Constant(6, 1e-3), VectorXd::Constant(6, 0.01),
                                                 VectorXd::Constant(6, 0.02));
    const std::vector<bool> ok = lab::is_consistent(nom, 6);
    REQUIRE(ok.size() == 6);
    CHECK(std::all_of(ok.begin(), ok.end(), [](bool x) { return x; }));
    VectorXd bad = nom;
    bad.segment(4, 6) << 0.001, 0, 0, 0.001, 0, 0.003;  // Ixx + Iyy < Izz: no such body
    bad.segment(1, 3).setZero();
    CHECK(lab::is_consistent(bad, 6) == std::vector<bool>{false, true, true, true, true, true});
    VectorXd neg = nom;
    neg(13 * 6 - 1) = -0.01;  // negative Coulomb friction on joint 6
    const std::vector<bool> okn = lab::is_consistent(neg, 6);
    REQUIRE(okn.size() == 6);
    CHECK_FALSE(okn[5]);
}

TEST_CASE("test_log_cholesky_parameterisation") {
    for (const auto& c : cases()["log_cholesky_parameterisation"]) {
        const VectorXd pi = lab::log_cholesky_to_pi(as_vec(c["x"]));
        REQUIRE(pi.size() == 10);
        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> es(lab::pseudo_inertia(pi), Eigen::EigenvaluesOnly);
        CHECK(es.eigenvalues().minCoeff() > 0.0);  // always consistent
        CHECK_ALLCLOSE(lab::log_cholesky_to_pi(lab::pi_to_log_cholesky(pi)), pi, 1e-8);
        CHECK_ALLCLOSE(pi, as_vec(c["pi"]), 1e-12);  // Python reference
    }
}

TEST_CASE("test_consistent_fit_eod") {
    const Identified& d = identified().at("eod_3r");
    const lab::Robot& r = d.robot;
    const int n = 3;
    const lab::LinearSystem S = lab::build_system(r, load("eod_3r_excite.csv"));
    const VectorXd prior = lab::nominal_parameters(r, vec({1.0, 0.5, 0.05}), vec({3, 2, 0.3}), vec({3, 2, 0.3}));
    const lab::ConsistentFit cf = lab::consistent_fit(S.Y, S.tau, S.joint, n, prior, 1e-3, d.fit.sigma);
    REQUIRE(cf.pi.size() == 39);
    const std::vector<bool> ok = lab::is_consistent(cf.pi, n);
    REQUIRE(ok.size() == 3);
    CHECK(std::all_of(ok.begin(), ok.end(), [](bool x) { return x; }));
    const lab::LinearSystem Sv = lab::build_system(r, load("eod_3r_validate.csv"));
    const lab::Validation v = lab::validate(Sv.Y, cf.pi, Sv.tau, Sv.joint, n);
    REQUIRE(v.nrms.size() == 3);
    INFO("consistent fit NRMS: " << v.nrms.transpose() << "   WLS: " << d.val.nrms.transpose());
    CHECK((v.nrms.array() < d.val.nrms.array() * 1.2 + 0.005).all());  // consistency costs (almost) nothing here
}

// ============================================================================ Part G: payload
TEST_CASE("test_payload_mass_of_the_training_package") {
    const Identified& d = identified().at("eod_3r");
    const MatrixXd& E = d.base.E;
    const VectorXd model = mr::pinv(E) * d.fit.beta;  // any pi with E pi = beta predicts the same torques
    const lab::Payload est = lab::payload_estimate(d.robot, load("eod_3r_package.csv"), model);
    const double weighed = num(meta()["package_mass_weighed"]);
    INFO("mass = " << est.mass << " kg, com = " << est.com.transpose());
    CHECK(std::abs(est.mass - weighed) < 0.05 * weighed);
    CHECK(0.0 < est.com(0));  // package CoM ahead of the link-3 frame
    CHECK(est.com(0) < 0.08);
}
