// Auto-checks for the SBEG 461 enrichment lab "Safety-critical control with control barrier functions" --
// C++ track. Mirrors test_lab.py test by test (same names, same order, same tolerances).
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*barrier*,*cbf*"      (run only some tests, like pytest -k)
// Ground truth: mrcpp (the C++ port of modern_robotics: mass_matrix, vel_quadratic_forces, gravity_forces,
// fkin_space, jacobian_space, forward_dynamics, inverse_dynamics), finite differences of the barrier along exact
// double-integrator motion, an independent active-set enumeration of the small QP, the Week 13 numbers (apparent
// mass 1.063 kg at the home pose, 0.204 m/s and 0.300 m/s speed limits), closed-loop properties that any correct
// implementation must have (forward invariance, speed cap), and cpp_cases.json, which holds the numbers of the
// Python reference and the random inputs of test_lab.py, so both languages are checked against the same values.
// The three model/data tests also call one function you write each (computed_torque, nominal_accel), so with the
// untouched starter every test case fails.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Eigen::Matrix2d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::VectorXd;
using sbeg::as;
using sbeg::json;

namespace {
const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

const json& DATA() {
    static const json D = sbeg::load_json(sbeg::lab_path("../data/scenario.json"));
    return D;
}

const lab::RehabModel& RH() {
    static const lab::RehabModel r = lab::rehab_2r();
    return r;
}

Vector2d v2(const json& j) { return as<Vector2d>(j); }

/// np.allclose(got, expected, rtol=1e-5, atol=atol), element by element: |got - expected| <= atol + 1e-5 |expected|
/// for every entry (a row and a column vector of equal length compare equal; NaN, inf or a shape mismatch fail).
sbeg::NearResult allclose(const MatrixXd& got, const MatrixXd& expected, double atol) {
    sbeg::NearResult r = sbeg::compare_near(got, expected, std::numeric_limits<double>::infinity());
    if (!r.ok) return r;  // shape mismatch or NaN/inf
    const Eigen::Map<const VectorXd> g(got.data(), got.size()), e(expected.data(), expected.size());
    r.ok = ((g - e).array().abs() <= atol + 1e-5 * e.array().abs()).all();
    std::ostringstream s;
    s << "np.allclose (rtol 1e-5, atol " << atol << ") fails; " << r.msg;
    r.msg = s.str();
    return r;
}
#define CHECK_ALLCLOSE(got, expected, atol)                                                           \
    do {                                                                                              \
        const ::sbeg::NearResult allclose_ = allclose((got), (expected), (atol));                     \
        CHECK_MESSAGE(allclose_.ok, "CHECK_ALLCLOSE(" #got ", " #expected ")\n  " << allclose_.msg); \
    } while (false)

/// pytest.approx(expected, rel=rel) (abs 1e-12).
double rel_tol(double expected, double rel) { return std::max(rel * std::abs(expected), 1e-12); }

/// Python run(name, filter_on, alpha1, v_max, model_scale): a scenario (changed as asked) and its simulation.
struct Run {
    lab::Scenario scn;
    lab::SimLog log;
};

constexpr double KEEP = std::numeric_limits<double>::quiet_NaN();  // Python None: keep the scenario value

Run run(const std::string& name, bool filter_on = true, double alpha1 = KEEP, double v_max = KEEP,
        double model_scale = 1.0) {
    Run r;
    r.scn = lab::load_scenario(name);
    if (!std::isnan(alpha1)) r.scn.alpha1 = alpha1;
    if (!std::isnan(v_max)) r.scn.v_max = v_max;
    r.log = lab::simulate(r.scn, filter_on, model_scale);
    return r;
}

/// Every array of the log has N = round(t_end / dt) + 1 rows (checked before any indexing).
void require_log_shape(const lab::SimLog& log, const lab::Scenario& scn) {
    const Eigen::Index N = static_cast<Eigen::Index>(std::llround(scn.t_end / scn.dt)) + 1;
    REQUIRE(log.t.size() == N);
    for (const MatrixXd* M : {&log.q, &log.qd, &log.p, &log.v, &log.p_o, &log.v_o, &log.u_nom, &log.u, &log.tau}) {
        REQUIRE(M->rows() == N);
        REQUIRE(M->cols() == 2);
    }
    REQUIRE(log.d.size() == N);
    REQUIRE(log.approach.size() == N);
    REQUIRE(static_cast<Eigen::Index>(log.active.size()) == N);
}

bool within_limits(const Vector2d& q) {
    return (q.array() >= RH().limits.col(0).array()).all() && (q.array() <= RH().limits.col(1).array()).all();
}

/// Independent solution of min |u - u_nom|^2 s.t. A u >= b (2-D, few rows) by enumerating active sets
/// (= _qp_enumerate of test_lab.py).
Vector2d qp_enumerate(const Vector2d& u_nom, const MatrixXd& A, const VectorXd& b, bool& found) {
    Vector2d best = u_nom;
    double best_cost = std::numeric_limits<double>::infinity();
    found = false;
    const int m = static_cast<int>(b.size());
    std::vector<std::vector<int>> sets = {{}};
    for (int i = 0; i < m; ++i) sets.push_back({i});
    for (int i = 0; i < m; ++i)
        for (int k = i + 1; k < m; ++k) sets.push_back({i, k});
    for (const auto& S : sets) {
        Vector2d u = u_nom;
        if (!S.empty()) {
            const Eigen::Index k = static_cast<Eigen::Index>(S.size());
            MatrixXd AS(k, 2);
            VectorXd bS(k);
            for (Eigen::Index r = 0; r < k; ++r) {
                AS.row(r) = A.row(S[static_cast<std::size_t>(r)]);
                bS(r) = b(S[static_cast<std::size_t>(r)]);
            }
            const MatrixXd G = AS * AS.transpose();
            if (std::abs(G.determinant()) < 1e-12) continue;
            const VectorXd lam = G.partialPivLu().solve(bS - AS * u_nom);
            if ((lam.array() < -1e-12).any()) continue;
            u = u_nom + AS.transpose() * lam;
        }
        const double cost = (u - u_nom).squaredNorm();
        if (((A * u).array() >= (b.array() - 1e-10)).all() && cost < best_cost) {
            best = u;
            best_cost = cost;
            found = true;
        }
    }
    return best;
}
}  // namespace

// ============================================================================ model and data (provided code)
TEST_CASE("test_model_matches_modern_robotics") {
    const auto& R = RH();
    for (const auto& c : cases()["model_matches_modern_robotics"]) {
        const Vector2d q = v2(c["q"]), qd = v2(c["qd"]);
        INFO("q = " << q.transpose() << ", qd = " << qd.transpose());
        const auto [Mq, cv] = lab::dynamics(q, qd);
        const MatrixXd Mmr = mr::mass_matrix(q, R.Mlist, R.Glist, R.Slist);
        const VectorXd cmr = mr::vel_quadratic_forces(q, qd, R.Mlist, R.Glist, R.Slist);
        CHECK_ALLCLOSE(Mq, Mmr, 1e-12);
        CHECK_ALLCLOSE(cv, cmr, 1e-12);
        CHECK_MAT_NEAR(mr::gravity_forces(q, R.g, R.Mlist, R.Glist, R.Slist), Vector2d::Zero(), 1e-12);
        const Eigen::Matrix4d T = mr::fkin_space(R.M, R.Slist, q);
        CHECK_ALLCLOSE(lab::fk(q), (T.block<2, 1>(0, 3)), 1e-12);
        const MatrixXd Js = mr::jacobian_space(R.Slist, q);  // p' = v_s + w_s x p
        const MatrixXd Jp = Js.bottomRows(3) - mr::vec_to_so3(T.block<3, 1>(0, 3)) * Js.topRows(3);
        CHECK_ALLCLOSE(lab::jacobian(q), Jp.topRows(2), 1e-12);
        const double h = 1e-6;  // J' q' by a central difference along q'
        const Matrix2d Jdot = (lab::jacobian(q + h * qd) - lab::jacobian(q - h * qd)) / (2 * h);
        CHECK_ALLCLOSE(lab::jdot_qdot(q, qd), Jdot * qd, 1e-7);
        // C++ only: computed torque with the provided model equals inverse dynamics of modern_robotics
        const Vector2d u(0.3, -0.2);
        const Vector2d qdd = lab::jacobian(q).partialPivLu().solve(u - lab::jdot_qdot(q, qd));
        const VectorXd tau_mr = mr::inverse_dynamics(q, qd, qdd, R.g, sbeg::Vector6d::Zero(), R.Mlist, R.Glist, R.Slist);
        CHECK_ALLCLOSE(lab::computed_torque(q, qd, u), tau_mr, 1e-12);
    }
    // Week 13 WE12 numbers at the home pose
    Matrix2d M_we12;
    M_we12 << 0.3545, 0.0628, 0.0628, 0.0657;
    CHECK_ALLCLOSE(lab::dynamics(R.q_home, Vector2d::Zero()).first, M_we12, 1e-4);
}

TEST_CASE("test_scenario_file_and_start_pose") {
    for (const std::string name : {"static", "approaching"}) {
        INFO("scenario " << name);
        const lab::Scenario scn = lab::load_scenario(name);
        // C++: the struct has every key; check that each one holds the value of data/scenario.json
        json m = DATA()["common"];
        for (const auto& item : DATA()["scenarios"][name].items()) m[item.key()] = item.value();
        const std::vector<std::pair<std::string, double>> scalars = {
            {"dt", scn.dt}, {"T_move", scn.T_move}, {"t_end", scn.t_end}, {"Kp", scn.Kp}, {"Kd", scn.Kd},
            {"r_keepout", scn.r_keepout}, {"alpha1", scn.alpha1}, {"alpha2", scn.alpha2}, {"gamma", scn.gamma},
            {"v_max", scn.v_max}, {"m_H", scn.m_H}, {"k_contact", scn.k_contact}, {"F_max", scn.F_max},
            {"person_t0", scn.person_t0}, {"person_t1", scn.person_t1}};
        for (const auto& kv : scalars) {
            INFO(kv.first);
            CHECK(kv.second == m.at(kv.first).get<double>());
        }
        CHECK_MAT_NEAR(scn.p_A, v2(m.at("p_A")), 0.0);
        CHECK_MAT_NEAR(scn.p_B, v2(m.at("p_B")), 0.0);
        CHECK_MAT_NEAR(scn.person_p0, v2(m.at("person_p0")), 0.0);
        CHECK_MAT_NEAR(scn.person_p1, v2(m.at("person_p1")), 0.0);
        CHECK_APPROX(scn.Kd * scn.Kd, 4 * scn.Kp);  // critically damped error dynamics
        for (const Vector2d& p : {scn.p_A, scn.p_B}) {
            const Vector2d q = lab::ik(p);
            CHECK_ALLCLOSE(lab::fk(q), p, 1e-12);
            CHECK(within_limits(q));
        }
        // C++ only: at t = 0 the reference is at rest at p_A, so the nominal command at rest at p_A is zero
        CHECK_MAT_NEAR(lab::nominal_accel(0.0, scn.p_A, Vector2d::Zero(), scn), Vector2d::Zero(), 1e-12);
    }
    CHECK(DATA()["common"]["v_max"].get<double>() == 0.20);  // the Week 13 value (H3) to be re-checked
}

TEST_CASE("test_person_and_reference_profiles") {
    const lab::Scenario scn = lab::load_scenario("approaching");
    const lab::Profile s0 = lab::person_state(0.0, scn);
    const lab::Profile s1 = lab::person_state(5.0, scn);
    CHECK_ALLCLOSE(s0.p, scn.person_p0, 1e-8);
    CHECK_ALLCLOSE(s1.p, scn.person_p1, 1e-8);
    CHECK_MAT_NEAR(s0.v, Vector2d::Zero(), 1e-8);
    CHECK_MAT_NEAR(s1.v, Vector2d::Zero(), 1e-8);
    CHECK_MAT_NEAR(s0.a, Vector2d::Zero(), 1e-8);
    const double t = 0.83, h = 1e-5;
    const Vector2d dp = (lab::person_state(t + h, scn).p - lab::person_state(t - h, scn).p) / (2 * h);
    CHECK_ALLCLOSE(dp, lab::person_state(t, scn).v, 1e-6);
    const Vector2d dv = (lab::reference(t + h, scn).v - lab::reference(t - h, scn).v) / (2 * h);
    CHECK_ALLCLOSE(dv, lab::reference(t, scn).a, 1e-5);
    // C++ only: on the reference the nominal command is the feed-forward acceleration
    const lab::Profile r = lab::reference(t, scn);
    CHECK_ALLCLOSE(lab::nominal_accel(t, r.p, r.v, scn), r.a, 1e-8);
}

// ============================================================================ Part A -- barrier and filter
TEST_CASE("test_barrier_terms_values_and_second_derivative") {
    const Vector2d p(0.10, 0.50), v(-0.20, 0.05);
    const Vector2d p_o(0.02, 0.47), v_o(0.03, -0.01);
    const lab::BarrierTerms B = lab::barrier_terms(p, v, p_o, v_o, 0.06);
    const Vector2d e = p - p_o, w = v - v_o;
    CHECK_APPROX(B.d, e.norm() - 0.06);
    CHECK_ALLCLOSE(B.n, e / e.norm(), 1e-8);
    CHECK_APPROX(B.d_dot, B.n.dot(w));
    CHECK(B.kappa >= 0);
    // exact motion with constant accelerations u (handle) and a_o (person): d'' by finite differences
    const Vector2d u(0.7, -0.4), a_o(0.1, 0.2);
    const double h = 1e-4;
    auto dist = [&](double s) {
        const Vector2d rel = p + v * s + 0.5 * u * s * s - (p_o + v_o * s + 0.5 * a_o * s * s);
        return rel.norm() - 0.06;
    };
    const double ddd_fd = (dist(h) - 2 * dist(0.0) + dist(-h)) / (h * h);
    const double ddd = B.n.dot(u - a_o) + B.kappa;
    CHECK_SCALAR_NEAR(ddd_fd, ddd, rel_tol(ddd, 1e-5));
    CHECK_SCALAR_NEAR((dist(h) - dist(-h)) / (2 * h), B.d_dot, rel_tol(B.d_dot, 1e-7));
    // the numbers of the Python reference
    const json& c = cases()["barrier_terms_values_and_second_derivative"];
    CHECK_SCALAR_NEAR(B.d, c["d"].get<double>(), 1e-12);
    CHECK_SCALAR_NEAR(B.d_dot, c["d_dot"].get<double>(), 1e-12);
    CHECK_SCALAR_NEAR(B.kappa, c["kappa"].get<double>(), 1e-12);
    CHECK_MAT_NEAR(B.n, v2(c["n"]), 1e-12);
}

TEST_CASE("test_cbf_bound_definitions") {
    const lab::Scenario scn = lab::load_scenario("static");
    const Vector2d p(0.12, 0.52), v(-0.25, -0.02);
    const Vector2d p_o(0.01, 0.48), v_o = Vector2d::Zero(), a_o = Vector2d::Zero();
    const lab::CbfBound cb = lab::cbf_bound(p, v, p_o, v_o, a_o, scn);
    const lab::BarrierTerms B = lab::barrier_terms(p, v, p_o, v_o, scn.r_keepout);
    const double a1 = scn.alpha1, a2 = scn.alpha2, g = scn.gamma;
    CHECK_ALLCLOSE(cb.n, B.n, 1e-8);
    CHECK_APPROX(cb.b1, -B.kappa - (a1 + a2) * B.d_dot - a1 * a2 * B.d);
    CHECK_APPROX(cb.b2, -B.kappa - g * (scn.v_max + B.d_dot));
    CHECK_APPROX(cb.b, std::max(cb.b1, cb.b2));
    // on the boundary n.u = b1 the second-order condition holds with equality: psi1' + a2 psi1 = 0
    const Vector2d u = cb.b1 * cb.n;
    const double ddd = cb.n.dot(u) + B.kappa;
    CHECK_SCALAR_NEAR((ddd + a1 * B.d_dot) + a2 * (B.d_dot + a1 * B.d), 0.0, 1e-12);
    // a moving person adds n.a_o to both bounds
    const Vector2d a_o2(0.3, -0.5);
    const lab::CbfBound cm = lab::cbf_bound(p, v, p_o, v_o, a_o2, scn);
    CHECK_APPROX(cm.b1 - cb.b1, cb.n.dot(a_o2));
    CHECK_APPROX(cm.b2 - cb.b2, cb.n.dot(a_o2));
    // the numbers of the Python reference
    const json& c = cases()["cbf_bound_definitions"];
    CHECK_SCALAR_NEAR(cb.b1, c["b1"].get<double>(), 1e-12);
    CHECK_SCALAR_NEAR(cb.b2, c["b2"].get<double>(), 1e-12);
    CHECK_SCALAR_NEAR(cb.b, c["b"].get<double>(), 1e-12);
}

TEST_CASE("test_safety_filter_closed_form") {
    const Vector2d a(0.6, 0.8);
    const Vector2d u_nom(1.0, -2.0);
    auto [u0, lam0] = lab::safety_filter(u_nom, a, -5.0);  // a.u_nom = -1 >= -5: inactive
    CHECK(lam0 == 0.0);
    CHECK_ALLCLOSE(u0, u_nom, 1e-8);
    const auto [u, lam] = lab::safety_filter(u_nom, a, 0.5);  // active: projection onto a.u = 0.5
    CHECK_APPROX(a.dot(u), 0.5);
    CHECK_APPROX(lam, 1.5);
    const Vector2d du = u - u_nom;
    CHECK(std::abs(du(0) * a(1) - du(1) * a(0)) < 1e-12);  // correction along a (KKT)
    for (const auto& xj : cases()["safety_filter_closed_form"]["x"]) {  // rng 7 of test_lab.py, 200 points
        const Vector2d x = v2(xj);
        if (a.dot(x) >= 0.5) CHECK((x - u_nom).norm() >= (u - u_nom).norm() - 1e-12);
    }
    const Vector2d u2 = lab::safety_filter(u_nom, 2 * a, 1.0).first;  // a non-unit normal gives the same u
    const Vector2d u1 = lab::safety_filter(u_nom, a, 0.5).first;
    CHECK_ALLCLOSE(u2, u1, 1e-8);
    CHECK_MAT_NEAR(u, v2(cases()["safety_filter_closed_form"]["u"]), 1e-12);
}

TEST_CASE("test_qp_with_scipy") {
    const json& C = cases()["qp_with_scipy"];
    for (const auto& c : C["single"]) {  // one constraint: equals the closed form
        const Vector2d a = v2(c["a"]), u_nom = v2(c["u_nom"]);
        const double b = c["b"].get<double>();
        const MatrixXd A = a.transpose();
        const Vector2d u = lab::safety_filter_qp(u_nom, A, VectorXd::Constant(1, b));
        const Vector2d uc = lab::safety_filter(u_nom, a, b).first;
        CHECK_ALLCLOSE(u, uc, 1e-6);
        CHECK_ALLCLOSE(u, v2(c["u"]), 1e-6);
    }
    for (const auto& c : C["multi"]) {  // two or three constraints: active-set enumeration
        const MatrixXd A = sbeg::as_mat(c["A"]);
        const Vector2d u_nom = v2(c["u_nom"]);
        const VectorXd b = sbeg::as_vec(c["b"]);
        INFO("A = " << A << ", b = " << b.transpose());
        bool found = false;
        const Vector2d ue = qp_enumerate(u_nom, A, b, found);
        REQUIRE(found);
        const Vector2d u = lab::safety_filter_qp(u_nom, A, b);
        CHECK_ALLCLOSE(u, ue, 1e-6);
        CHECK_ALLCLOSE(u, v2(c["u"]), 1e-6);
    }
}

// ============================================================================ Part B -- the controller
TEST_CASE("test_nominal_accel") {
    const lab::Scenario scn = lab::load_scenario("static");
    const double t = 0.7;
    const lab::Profile r = lab::reference(t, scn);
    CHECK_ALLCLOSE(lab::nominal_accel(t, r.p, r.v, scn), r.a, 1e-8);
    const Vector2d dp(0.01, -0.02), dv(0.05, 0.03);
    const Vector2d expect = r.a + scn.Kd * (-dv) + scn.Kp * (-dp);
    CHECK_ALLCLOSE(lab::nominal_accel(t, r.p + dp, r.v + dv, scn), expect, 1e-8);
}

TEST_CASE("test_computed_torque_gives_the_commanded_handle_acceleration") {
    const auto& R = RH();
    for (const auto& c : cases()["computed_torque_gives_the_commanded_handle_acceleration"]) {  // rng 5
        const Vector2d q = v2(c["q"]), qd = v2(c["qd"]), u = v2(c["u"]);
        INFO("q = " << q.transpose() << ", qd = " << qd.transpose() << ", u = " << u.transpose());
        const Vector2d tau = lab::computed_torque(q, qd, u);
        const VectorXd qdd = mr::forward_dynamics(q, qd, tau, R.g, sbeg::Vector6d::Zero(), R.Mlist, R.Glist, R.Slist);
        REQUIRE(qdd.size() == 2);
        CHECK_ALLCLOSE(lab::jacobian(q) * qdd + lab::jdot_qdot(q, qd), u, 1e-9);
        const Vector2d tau12 = 1.2 * tau;
        CHECK_ALLCLOSE(lab::computed_torque(q, qd, u, 1.2), tau12, 1e-8);
        CHECK_ALLCLOSE(tau, v2(c["tau"]), 1e-8);
    }
}

TEST_CASE("test_nominal_run_tracks_but_enters_the_keep_out_disc") {
    for (const std::string name : {"static", "approaching"}) {
        INFO("scenario " << name);
        const Run r = run(name, false);
        require_log_shape(r.log, r.scn);
        const lab::Metrics m = lab::metrics(r.log, r.scn);
        CHECK(m.max_dev < 5e-5);  // exact model: p'' = u_nom (only the 1 ms hold)
        CHECK(m.final_error < 5e-5);
        CHECK(m.min_d < -0.01);  // the unfiltered handle enters the disc
        CHECK(m.max_approach > 0.3);
        CHECK(std::none_of(r.log.active.begin(), r.log.active.end(), [](bool x) { return x; }));
    }
    const Run r = run("static", false);
    CHECK_SCALAR_NEAR(lab::metrics(r.log, r.scn).min_d, -0.0285, 5e-4);  // path passes 31.5 mm from the centre
}

TEST_CASE("test_filter_keeps_out_and_caps_the_approach_speed") {
    for (const std::string name : {"static", "approaching"}) {
        INFO("scenario " << name);
        const Run r = run(name);
        require_log_shape(r.log, r.scn);
        const lab::Metrics m = lab::metrics(r.log, r.scn);
        CHECK(m.min_d >= 0.0);                         // never inside the keep-out disc
        CHECK(m.max_approach <= r.scn.v_max + 1e-3);   // speed cap (1 kHz sampling tolerance)
        CHECK(m.active_fraction > 0.1);                // the filter acts only near the hand
        CHECK(m.active_fraction < 0.5);
        CHECK(m.final_error < 2e-3);                   // still completes the movement
        CHECK(m.max_tau_ratio < 1.0);                  // within the rated torques
        bool all_inside = true;
        for (Eigen::Index k = 0; k < r.log.q.rows(); ++k) all_inside = all_inside && within_limits(r.log.q.row(k).transpose());
        CHECK(all_inside);
    }
}

TEST_CASE("test_forward_invariance_of_the_barriers") {
    for (const std::string name : {"static", "approaching"}) {
        INFO("scenario " << name);
        const Run r = run(name);
        require_log_shape(r.log, r.scn);
        const VectorXd psi1 = -r.log.approach + r.scn.alpha1 * r.log.d;  // psi1 = d' + a1 d
        const VectorXd h_v = r.scn.v_max - r.log.approach.array();
        CHECK(psi1.minCoeff() >= -1e-3);
        CHECK(h_v.minCoeff() >= -1e-3);
        // where the filter is inactive the command is the nominal one; where active it differs
        int mismatches = 0;
        for (Eigen::Index k = 0; k < r.log.u.rows(); ++k) {
            const bool same = ((r.log.u.row(k) - r.log.u_nom.row(k)).array().abs() <=
                               1e-8 + 1e-5 * r.log.u_nom.row(k).array().abs())
                                  .all();  // np.isclose
            if (same == r.log.active[static_cast<std::size_t>(k)]) ++mismatches;
        }
        CHECK(mismatches == 0);
    }
}

TEST_CASE("test_keep_out_only_reaches_the_boundary_without_crossing") {
    const Run r = run("static", true, 10.0, 1e3);
    const lab::Metrics m = lab::metrics(r.log, r.scn);
    CHECK(m.min_d >= 0.0);
    CHECK(m.min_d < 0.006);
    CHECK(m.max_approach > 0.25);  // no speed cap in this run
}

TEST_CASE("test_qp_and_closed_form_runs_agree") {
    lab::Scenario scn = lab::load_scenario("static");
    scn.t_end = 1.6;
    const lab::SimLog a = lab::simulate(scn);
    const lab::SimLog b = lab::simulate(scn, true, 1.0, true);
    require_log_shape(a, scn);
    require_log_shape(b, scn);
    CHECK_ALLCLOSE(a.p, b.p, 1e-6);
}

TEST_CASE("test_model_error_keeps_the_handle_out_of_the_disc") {
    for (const double s : {0.7, 1.3}) {
        INFO("model_scale = " << s);
        const Run r = run("static", true, KEEP, KEEP, s);
        CHECK(lab::metrics(r.log, r.scn).min_d > 0.0);
    }
}

// ============================================================================ Part C -- force limits near a person
TEST_CASE("test_apparent_mass_and_pfl_speed") {
    const auto& R = RH();
    CHECK_SCALAR_NEAR(lab::apparent_mass(R.q_home, Vector2d(1.0, 0.0)), 1.063, 5e-4);  // Week 13 WE12
    const Vector2d q(0.7, 1.3), n(0.6, -0.8);
    const Matrix2d J = lab::jacobian(q);
    const Matrix2d Mmr = mr::mass_matrix(q, R.Mlist, R.Glist, R.Slist);
    const Matrix2d Lam = (J * Mmr.inverse() * J.transpose()).inverse();
    const Eigen::SelfAdjointEigenSolver<Matrix2d> es(Lam);
    const double lo = es.eigenvalues()(0), hi = es.eigenvalues()(1);
    const double m = lab::apparent_mass(q, n);
    CHECK(lo - 1e-12 <= m);
    CHECK(m <= hi + 1e-12);
    CHECK_APPROX(m, 1.0 / n.dot(Lam.inverse() * n));
    CHECK_SCALAR_NEAR(lab::pfl_speed_limit(6.0, 10.0, 50000.0, 130.0), 0.300, 5e-4);  // Week 13 WE6
    CHECK_SCALAR_NEAR(lab::pfl_speed_limit(1.0633, 2.0, 5000.0, 12.0), 0.204, 5e-4);  // Week 13 WE12
    const json& c = cases()["apparent_mass_and_pfl_speed"];
    CHECK_SCALAR_NEAR(m, c["m"].get<double>(), 1e-12);
}

TEST_CASE("test_design_speed_limit") {
    const lab::SpeedDesign s = lab::design_speed_limit(lab::load_scenario("static"));
    CHECK_SCALAR_NEAR(s.m_R_max, 2.476, 2e-3);
    CHECK_SCALAR_NEAR(s.v_exact, 0.1613, 2e-4);
    CHECK_APPROX(s.v_design, 0.16);
    const json& c = cases()["design_speed_limit"];
    CHECK_SCALAR_NEAR(s.m_R_max, c["m_R_max"].get<double>(), 1e-9);
}

TEST_CASE("test_week13_speed_is_not_conservative_but_the_designed_one_is") {
    {
        const Run r = run("approaching");  // v_max = 0.20 from the home pose along x
        require_log_shape(r.log, r.scn);
        CHECK(lab::peak_contact_force(r.log, r.scn).maxCoeff() > r.scn.F_max);
    }
    const double v = lab::design_speed_limit(lab::load_scenario("approaching")).v_design;
    for (const std::string name : {"static", "approaching"}) {
        INFO("scenario " << name << ", v_max = " << v);
        const Run r = run(name, true, KEEP, v);
        require_log_shape(r.log, r.scn);
        CHECK(lab::metrics(r.log, r.scn).min_d >= 0.0);
        CHECK(lab::peak_contact_force(r.log, r.scn).maxCoeff() <= r.scn.F_max);
    }
}
