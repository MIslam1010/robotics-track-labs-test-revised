// Auto-checks for SBEG 361 Lab 8 (statics, manipulability maps, workspace design) -- C++ track.
// Mirrors test_lab.py test by test (same names, same order, same tolerances).
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*torque*,*wrench*"      (run only some tests, like pytest -k)
// Ground truth: mrcpp (the C++ port of modern_robotics: jacobian_space, jacobian_body, fkin_space, adjoint),
// independent moment-arm formulas, the textbook numbers of MR Ch. 5, and cpp_cases.json, which holds the numbers
// of the Python reference and the random inputs drawn by test_lab.py, so both languages are checked against the
// same values.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using sbeg::as;
using sbeg::as_mat;
using sbeg::as_vec;
using sbeg::json;
using sbeg::Vector6d;

namespace {
constexpr double PI = 3.14159265358979323846;

const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

double d2r(double deg) { return deg * (PI / 180.0); }

VectorXd d2r(const VectorXd& deg) { return deg * (PI / 180.0); }

/// np.allclose(got, expected, rtol=1e-5, atol=atol), entry by entry: |got - expected| <= atol + rtol |expected|.
/// Shape check and NaN check as CHECK_MAT_NEAR (a row and a column vector of equal length compare equal).
sbeg::NearResult allclose(const MatrixXd& got_in, const MatrixXd& exp_in, double atol, double rtol = 1e-5) {
    sbeg::NearResult r = sbeg::compare_near(got_in, exp_in, std::numeric_limits<double>::infinity());
    if (!r.ok) return r;  // shape mismatch, or NaN/inf in got
    MatrixXd got = got_in, want = exp_in;
    if (got.rows() != want.rows()) {  // equal-length row and column vectors
        got.resize(got.size(), 1);
        want.resize(want.size(), 1);
    }
    double worst = -std::numeric_limits<double>::infinity();  // largest |got - expected| - rtol |expected|
    Eigen::Index wi = 0, wk = 0;
    for (Eigen::Index i = 0; i < got.rows(); ++i) {
        for (Eigen::Index k = 0; k < got.cols(); ++k) {
            const double e = std::abs(got(i, k) - want(i, k)) - rtol * std::abs(want(i, k));
            if (e > worst) {
                worst = e;
                wi = i;
                wk = k;
            }
        }
    }
    r.ok = got.size() == 0 || worst <= atol;
    std::ostringstream s;
    s << std::setprecision(3) << "|got - expected| - rtol |expected| = " << worst << " at (" << wi << "," << wk
      << "), atol " << atol << ", rtol " << rtol << "\n  got:      " << sbeg::mat_to_string(got_in)
      << "\n  expected: " << sbeg::mat_to_string(exp_in);
    r.msg = s.str();
    return r;
}

/// Absolute tolerance of np.isclose(got, expected, atol=atol) (rtol = 1e-5).
double ic(double expected, double atol = 1e-8) { return atol + 1e-5 * std::abs(expected); }

struct Model {
    MatrixXd S;
    Matrix4d M;
    MatrixXd B;
    lab::Robot info;
};

/// (S, M, B, info) as model() in test_lab.py; B from mrcpp.
Model model(const std::string& name) {
    const lab::Robot r = lab::load_robot(name);
    return Model{r.Slist, r.M, mr::adjoint(mr::trans_inv(r.M)) * r.Slist, r};
}

VectorXd vec1(double v) { return VectorXd::Constant(1, v); }
}  // namespace

// np.allclose(got, expected, atol=atol) with the default rtol = 1e-5, checked entry by entry.
#define CHECK_ALLCLOSE(got, expected, atol)                                                         \
    do {                                                                                            \
        const ::sbeg::NearResult sbeg_ac_ = allclose((got), (expected), (atol));                    \
        CHECK_MESSAGE(sbeg_ac_.ok, "CHECK_ALLCLOSE(" #got ", " #expected ")\n  " << sbeg_ac_.msg); \
    } while (false)

// ---------------------------------------------------------------- Checkpoint 1: statics
TEST_CASE("test_joint_torques_is_JT_F") {
    const json& c = cases()["joint_torques_is_JT_F"];
    const MatrixXd J = as_mat(c["J"]);
    const VectorXd F = as_vec(c["F"]);
    const VectorXd expected = J.transpose() * F;
    CHECK_ALLCLOSE(lab::joint_torques(J, F), expected, 1e-8);
}

TEST_CASE("test_space_and_body_torques_agree") {
    for (const auto& c : cases()["space_and_body_torques_agree"]) {
        const std::string name = c["name"].get<std::string>();
        INFO("robot = " << name);
        const Model m = model(name);
        const VectorXd q = as_vec(c["q"]);
        const Matrix4d T = mr::fkin_space(m.M, m.S, q);
        const Vector6d Fb = as<Vector6d>(c["Fb"]);
        const Vector6d Fs = lab::body_to_space_wrench(Fb, T);
        const Vector6d Fs_ref = mr::adjoint(mr::trans_inv(T)).transpose() * Fb;
        CHECK_ALLCLOSE(Fs, Fs_ref, 1e-10);
        const VectorXd tb = lab::joint_torques(mr::jacobian_body(m.B, q), Fb);
        const VectorXd ts = lab::joint_torques(mr::jacobian_space(m.S, q), Fs);
        CHECK_ALLCLOSE(tb, ts, 1e-10);
        CHECK_MAT_NEAR(tb, as_vec(c["tau"]), 1e-9);
    }
}

TEST_CASE("test_power_balance") {
    // tau . thetadot = F . V for every thetadot (MR p. 188).
    const Model m = model("course_arm");
    const json& c = cases()["power_balance"];
    const VectorXd q = as_vec(c["q"]);
    const MatrixXd Jb = mr::jacobian_body(m.B, q);
    const VectorXd F = as_vec(c["F"]);
    for (const auto& qdj : c["qd"]) {
        const VectorXd qd = as_vec(qdj);
        const double expected = F.dot(Jb * qd);
        CHECK_SCALAR_NEAR(lab::joint_torques(Jb, F).dot(qd), expected, ic(expected));
    }
}

TEST_CASE("test_holding_wrench_moment_arms") {
    // tau_i = w_i . ((p - q_i) x f) for revolute joint i with axis (w_i, q_i) at the current pose.
    const Model md = model("course_arm");
    const VectorXd q = md.info.q_home;
    const double m = 0.25;
    const Vector3d g(0.0, 0.0, -9.81);
    const Matrix4d T = mr::fkin_space(md.M, md.S, q);
    const Vector3d p = T.topRightCorner<3, 1>();
    const Vector6d F = lab::holding_wrench_space(m, p, g);
    const Vector3d f = F.tail<3>();
    const Vector3d f_ref(0.0, 0.0, m * 9.81);
    CHECK_ALLCLOSE(f, f_ref, 1e-8);
    const Vector3d mom_ref = p.cross(f);
    CHECK_ALLCLOSE(F.head<3>(), mom_ref, 1e-8);
    const MatrixXd Js = mr::jacobian_space(md.S, q);
    const VectorXd tau = lab::joint_torques(Js, F);
    REQUIRE(tau.size() == 6);
    for (Eigen::Index i = 0; i < 6; ++i) {
        const Vector3d w = Js.block<3, 1>(0, i), v = Js.block<3, 1>(3, i);
        const Vector3d qi = w.cross(v);  // point on the axis (|w| = 1)
        const double expected = w.dot((p - qi).cross(f));
        INFO("joint " << i + 1);
        CHECK_SCALAR_NEAR(tau(i), expected, ic(expected, 1e-12));
    }
}

TEST_CASE("test_exoskeleton_load_torques") {
    // Workshop P1: arm exoskeleton (0.30, 0.35) m holding 2 kg at the hand, gravity -y.
    const Model m = model("arm_exo_2r");
    const Vector3d g(0.0, -9.81, 0.0);
    for (const auto& c : cases()["exoskeleton_load_torques"]) {
        const VectorXd th = d2r(as_vec(c["q_deg"]));
        const VectorXd expected = as_vec(c["expected"]);
        INFO("q (deg) = " << c["q_deg"].dump());
        const Vector3d p = mr::fkin_space(m.M, m.S, th).topRightCorner<3, 1>();
        const VectorXd tau = lab::joint_torques(mr::jacobian_space(m.S, th), lab::holding_wrench_space(2.0, p, g));
        CHECK_ALLCLOSE(tau, expected, 1e-3);
    }
}

TEST_CASE("test_wrench_from_torques") {
    const MatrixXd J = lab::jacobian_2r(Vector2d(d2r(0.0), d2r(90.0)), 0.30, 0.35);
    const Vector2d f_ref(-5.7143, 10.0);
    CHECK_ALLCLOSE(lab::wrench_from_torques(J, Vector2d(5.0, 2.0)), f_ref, 1e-4);
    const Model m = model("course_arm");
    const json& c = cases()["wrench_from_torques"];
    const MatrixXd Jb = mr::jacobian_body(m.B, as_vec(c["q"]));
    const VectorXd F = as_vec(c["F"]);
    CHECK_ALLCLOSE(lab::wrench_from_torques(Jb, Jb.transpose() * F), F, 1e-8);
    CHECK_THROWS_AS(lab::wrench_from_torques(MatrixXd::Ones(6, 5), VectorXd::Ones(5)),
                    std::invalid_argument);  // not square (n != 6)
    CHECK_THROWS_AS(lab::wrench_from_torques(lab::jacobian_2r(Vector2d(0.4, 0.0), 0.35, 0.35), Vector2d(1.0, 1.0)),
                    std::invalid_argument);  // singular
}

TEST_CASE("test_wrench_null_space") {
    const Model h = model("hapkit");
    const MatrixXd N = lab::wrench_null_space(h.B);
    REQUIRE(N.rows() == 6);
    REQUIRE(N.cols() == 5);  // MR p. 189: 1-DOF door resists a 5-D wrench space
    CHECK_MAT_NEAR(h.B.transpose() * N, MatrixXd::Zero(1, 5), 1e-12);
    CHECK_ALLCLOSE(N.transpose() * N, MatrixXd::Identity(5, 5), 1e-10);
    const Model m = model("course_arm");
    VectorXd q = m.info.q_home;
    q(2) = 0.0;  // elbow straight
    const MatrixXd Jb = mr::jacobian_body(m.B, q);
    const MatrixXd N2 = lab::wrench_null_space(Jb);
    REQUIRE(N2.rows() == 6);
    REQUIRE(N2.cols() == 1);
    CHECK_MAT_NEAR(Jb.transpose() * N2, MatrixXd::Zero(6, 1), 1e-9);
}

// ---------------------------------------------------------------- Checkpoint 2: ellipsoids and measures
TEST_CASE("test_ellipsoid_mr_2r") {
    // MR p. 171 (Figs. 5.3, 5.5): 2R, L1 = L2 = 1.
    const Vector2d a1(2.0731, 0.3411), a2(1.0731, 0.6589);
    const VectorXd s1 = lab::manipulability_ellipsoid(lab::jacobian_2r(Vector2d(0.0, PI / 4), 1, 1)).semi_axes;
    CHECK_ALLCLOSE(s1, a1, 1e-4);
    const VectorXd s2 = lab::manipulability_ellipsoid(lab::jacobian_2r(Vector2d(0.0, 3 * PI / 4), 1, 1)).semi_axes;
    CHECK_ALLCLOSE(s2, a2, 1e-4);
}

TEST_CASE("test_ellipsoid_reconstructs_A") {
    for (const auto& c : cases()["ellipsoid_reconstructs_A"]) {
        INFO("seed = " << c["seed"].get<int>());
        const MatrixXd J = as_mat(c["J"]);  // 3 x 6, np.random.default_rng(seed).normal(size=(3, 6))
        const lab::Ellipsoid e = lab::manipulability_ellipsoid(J);
        const VectorXd& a = e.semi_axes;
        const MatrixXd& V = e.directions;
        REQUIRE(a.size() == 3);
        REQUIRE(V.rows() == 3);
        REQUIRE(V.cols() == 3);
        for (Eigen::Index k = 0; k + 1 < a.size(); ++k) CHECK(a(k + 1) - a(k) <= 1e-12);  // longest first
        CHECK_MAT_NEAR(a, as_vec(c["semi_axes"]), 1e-10);
        CHECK_ALLCLOSE(V.transpose() * V, MatrixXd::Identity(3, 3), 1e-10);
        const MatrixXd A = J * J.transpose();
        const MatrixXd VAV = V * a.cwiseAbs2().asDiagonal() * V.transpose();
        CHECK_ALLCLOSE(VAV, A, 1e-10);
        const lab::Ellipsoid fe = lab::force_ellipsoid(J);
        REQUIRE(fe.semi_axes.size() == 3);
        CHECK_ALLCLOSE(fe.directions, V, 1e-8);  // same axes ...
        CHECK_ALLCLOSE(fe.semi_axes.cwiseProduct(a), VectorXd::Ones(3), 1e-8);  // ... reciprocal lengths
    }
}

TEST_CASE("test_force_ellipsoid_singular_is_infinite") {
    const VectorXd f = lab::force_ellipsoid(lab::jacobian_2r(Vector2d(0.3, 0.0), 0.35, 0.35)).semi_axes;
    REQUIRE(f.size() == 2);
    CHECK(std::isinf(f(1)));
    CHECK(std::isfinite(f(0)));
}

TEST_CASE("test_measures_values") {
    const MatrixXd J = lab::jacobian_2r(Vector2d(d2r(0.0), d2r(90.0)), 0.35, 0.35);  // Workshop P2
    const lab::Measures m = lab::manipulability_measures(J);
    CHECK_SCALAR_NEAR(m.mu1, 2.6180, ic(2.6180, 1e-4));
    CHECK_SCALAR_NEAR(m.mu2, 6.8541, ic(6.8541, 1e-4));
    CHECK_SCALAR_NEAR(m.mu3, 0.1225, ic(0.1225, 1e-6));
    for (const auto& qj : cases()["measures_values"]["qs"]) {
        const Vector2d q = as<Vector2d>(qj);
        INFO("q = " << qj.dump());
        const MatrixXd Jq = lab::jacobian_2r(q, 0.35, 0.35);
        const lab::Measures mq = lab::manipulability_measures(Jq);
        const double det = std::abs(Jq.determinant());
        CHECK_SCALAR_NEAR(mq.mu3, det, ic(det));  // square J: sqrt(det JJ^T) = |det J|
        CHECK_SCALAR_NEAR(mq.mu2, mq.mu1 * mq.mu1, ic(mq.mu1 * mq.mu1));
        CHECK(mq.mu1 >= 1 - 1e-12);
    }
}

TEST_CASE("test_measures_singular") {
    const lab::Measures m = lab::manipulability_measures(lab::jacobian_2r(Vector2d(0.2, 0.0), 1, 1));
    CHECK(std::isinf(m.mu1));
    CHECK(std::isinf(m.mu2));
    CHECK_SCALAR_NEAR(m.mu3, 0.0, 1e-12);
}

TEST_CASE("test_volume_product_constant") {
    // mu3(A) mu3(B) = 1 with B = A^-1 (MR p. 197).
    for (const auto& qj : cases()["volume_product_constant"]["qs"]) {
        INFO("q = " << qj.dump());
        const MatrixXd J = lab::jacobian_2r(as<Vector2d>(qj), 0.35, 0.35);
        const VectorXd a = lab::manipulability_ellipsoid(J).semi_axes;
        const VectorXd f = lab::force_ellipsoid(J).semi_axes;
        CHECK_SCALAR_NEAR(a.prod() * f.prod(), 1.0, ic(1.0));
    }
}

TEST_CASE("test_angular_measures_frame_independent_linear_not") {
    // J_omega of J_s and J_b differ by a rotation (same ellipsoid shape); J_v does not (MR p. 196).
    const Model m = model("course_arm");
    VectorXd q(6);
    q << 0.3, -0.7, 1.2, -1.9, -1.2, 0.5;
    const MatrixXd Js = mr::jacobian_space(m.S, q), Jb = mr::jacobian_body(m.B, q);
    const lab::Measures ms = lab::manipulability_measures(lab::select_rows(Js, "angular"));
    const lab::Measures mb = lab::manipulability_measures(lab::select_rows(Jb, "angular"));
    CHECK_SCALAR_NEAR(ms.mu1, mb.mu1, ic(mb.mu1));
    CHECK_SCALAR_NEAR(ms.mu3, mb.mu3, ic(mb.mu3));
    const lab::Measures ls = lab::manipulability_measures(lab::select_rows(Js, "linear"));
    const lab::Measures lb = lab::manipulability_measures(lab::select_rows(Jb, "linear"));
    INFO("linear mu3 of J_s = " << ls.mu3 << ", of J_b = " << lb.mu3);
    CHECK(std::abs(ls.mu3 - lb.mu3) > 1e-8 + 1e-3 * std::abs(lb.mu3));  // not np.isclose(..., rtol=1e-3)
}

// ---------------------------------------------------------------- Checkpoint 3: singularities
TEST_CASE("test_rank_space_equals_body") {
    for (const auto& c : cases()["rank_space_equals_body"]) {
        const std::string name = c["name"].get<std::string>();
        const Model m = model(name);
        const json& qs = c["qs"];
        const json& ranks = c["ranks"];
        REQUIRE(qs.size() == ranks.size());
        for (std::size_t k = 0; k < qs.size(); ++k) {
            const VectorXd q = as_vec(qs[k]);
            INFO("robot = " << name << ", q = " << qs[k].dump());
            const int rs = lab::singularity_report(mr::jacobian_space(m.S, q)).rank;
            const int rb = lab::singularity_report(mr::jacobian_body(m.B, q)).rank;
            CHECK(rs == rb);
            CHECK(rs == ranks[k].get<int>());  // np.linalg.matrix_rank(J_s, tol=1e-6)
        }
    }
}

TEST_CASE("test_eod_singularity_cases") {
    // Workshop P3: stretched (Cases I + II), wrist (Case I), regular, shoulder (Case III).
    const Model m = model("eod_6r_ugv");
    const double elbow_x = 0.10 + 0.80 * std::cos(d2r(45.0));  // theta2 = -45 deg (inside [-60, 110] deg)
    const double th3 = 45.0 - std::acos(-elbow_x / 0.75) * (180.0 / PI);  // -107.57 deg: wrist centre on axis 1
    const double qd[4][6] = {{0, 0, 0, 0, 0, 0}, {0, -30, 60, 0, 0, 0}, {0, -30, 60, 0, 45, 0},
                             {0, -45, th3, 0, 45, 0}};
    const int rank[4] = {4, 5, 6, 5};
    for (int k = 0; k < 4; ++k) {
        VectorXd q(6);
        for (int i = 0; i < 6; ++i) q(i) = d2r(qd[k][i]);
        INFO("q (deg) = " << q.transpose() * (180.0 / PI));
        CHECK(lab::singularity_report(mr::jacobian_space(m.S, q)).rank == rank[k]);
    }
    VectorXd q(6);
    q << 0.0, d2r(-30.0), d2r(60.0), 0.0, d2r(45.0), 0.0;
    const lab::SingularityReport rep = lab::singularity_report(mr::jacobian_space(m.S, q));
    CHECK_SCALAR_NEAR(rep.sigma_min, 0.29938, ic(0.29938, 1e-4));
}

TEST_CASE("test_lost_twists_and_null_motions") {
    const Model m = model("eod_6r_ugv");
    VectorXd q(6);
    q << 0.0, d2r(-30.0), d2r(60.0), 0.0, 0.0, 0.0;
    const MatrixXd J = mr::jacobian_space(m.S, q);
    const lab::SingularityReport rep = lab::singularity_report(J);
    const MatrixXd& U = rep.lost_twists;
    const MatrixXd& N = rep.null_motions;
    REQUIRE(U.rows() == 6);
    REQUIRE(U.cols() == 1);
    REQUIRE(N.rows() == 6);
    REQUIRE(N.cols() == 1);
    CHECK_MAT_NEAR(J.transpose() * U, MatrixXd::Zero(6, 1), 1e-9);
    CHECK_MAT_NEAR(J * N, MatrixXd::Zero(6, 1), 1e-9);
    const VectorXd n = N.col(0) / N(3, 0);
    VectorXd n_ref(6);
    n_ref << 0, 0, 0, 1, 0, -1;  // forearm roll against gripper roll
    CHECK_ALLCLOSE(n, n_ref, 1e-9);
}

TEST_CASE("test_singularity_independent_of_tool_frame") {
    // MR Fig. 5.10(b): moving {b} (needle guide) does not move the singularities.
    const Model a = model("course_arm");
    const Model d = model("desktop_6r");
    for (double q3 : {0.0, 0.3}) {
        VectorXd q = a.info.q_home;
        q(2) = q3;
        INFO("theta3 = " << q3);
        const int ra = lab::singularity_report(mr::jacobian_space(a.S, q)).rank;
        const int rd = lab::singularity_report(mr::jacobian_space(d.S, q)).rank;
        CHECK(ra == rd);
        CHECK(ra == (q3 == 0.0 ? 5 : 6));
    }
}

// ---------------------------------------------------------------- Checkpoint 4: maps and design (rehab_2r)
TEST_CASE("test_map_2r_formula_and_nan") {
    const json& c = cases()["map_2r_formula_and_nan"];
    const VectorXd xs = as_vec(c["xs"]), ys = as_vec(c["ys"]);  // np.linspace(-0.8, 0.8, 9)
    const MatrixXd mp = lab::manipulability_map_2r(0.35, 0.35, xs, ys, "mu3");
    REQUIRE(mp.rows() == 9);
    REQUIRE(mp.cols() == 9);
    for (Eigen::Index a = 0; a < ys.size(); ++a) {
        for (Eigen::Index b = 0; b < xs.size(); ++b) {
            const double x = xs(b), y = ys(a);
            const double r2 = x * x + y * y;
            INFO("(x, y) = (" << x << ", " << y << ")");
            if (r2 > 0.7 * 0.7) {
                CHECK(std::isnan(mp(a, b)));
            } else {
                const double c2 = (r2 - 2 * 0.35 * 0.35) / (2 * 0.35 * 0.35);
                const double expected = 0.35 * 0.35 * std::sqrt(std::max(0.0, 1 - c2 * c2));
                CHECK_SCALAR_NEAR(mp(a, b), expected, ic(expected, 1e-12));
            }
        }
    }
}

TEST_CASE("test_map_respects_limits_and_base") {
    const Model m = model("rehab_2r");
    const MatrixXd& lim = m.info.limits;
    const MatrixXd mp = lab::manipulability_map_2r(0.35, 0.35, Vector2d(0.0, 0.0), Vector2d(-0.3, 0.3), "mu3", 1, lim);
    REQUIRE(mp.rows() == 2);
    REQUIRE(mp.cols() == 2);
    CHECK(std::isnan(mp(0, 0)));  // theta1 in [0, pi]: no targets behind the base
    CHECK(!std::isnan(mp(1, 0)));
    const MatrixXd mp0 = lab::manipulability_map_2r(0.35, 0.35, vec1(0.1), vec1(0.4), "mu1");
    const MatrixXd mp1 =
        lab::manipulability_map_2r(0.35, 0.35, vec1(0.1), vec1(0.3), "mu1", 1, MatrixXd(), Vector2d(0.0, -0.1));
    REQUIRE(mp0.size() == 1);
    REQUIRE(mp1.size() == 1);
    CHECK_SCALAR_NEAR(mp0(0, 0), mp1(0, 0), ic(mp1(0, 0)));  // same point relative to the base
}

TEST_CASE("test_region_score_values") {
    const Model m = model("rehab_2r");
    REQUIRE(m.info.L.size() == 2);
    const double L1 = m.info.L(0), L2 = m.info.L(1);
    const MatrixXd& lim = m.info.limits;
    const MatrixXd pts = lab::load_targets();
    REQUIRE(pts.rows() == 25);
    REQUIRE(pts.cols() == 2);
    std::pair<double, double> wf = lab::region_score(L1, L2, pts, "mu3", 1, lim, Vector2d(0.0, -0.10));
    CHECK_SCALAR_NEAR(wf.first, 0.106088, ic(0.106088, 1e-5));
    CHECK(wf.second == 1.0);
    wf = lab::region_score(L1, L2, pts, "mu3", 1, lim, Vector2d(0.0, 0.0));
    CHECK_SCALAR_NEAR(wf.second, 0.92, ic(0.92));
    wf = lab::region_score(L1, L2, pts, "mu1", 1, lim, Vector2d(0.0, -0.10));
    CHECK_SCALAR_NEAR(wf.first, 3.87994, ic(3.87994, 1e-4));
}

TEST_CASE("test_design_search") {
    const Model m = model("rehab_2r");
    const MatrixXd& lim = m.info.limits;
    const MatrixXd pts = lab::load_targets();
    const json& c = cases()["design_search"];
    const VectorXd ratios = as_vec(c["ratios"]);    // np.round(np.arange(0.30, 0.71, 0.05), 2)
    const VectorXd offsets = as_vec(c["offsets"]);  // 0, 0.05, ..., 0.20
    lab::DesignSearchResult r = lab::design_search(0.70, ratios, offsets, pts, lim);
    CHECK(r.rows.size() == 45);
    REQUIRE(r.best.has_value());
    CHECK_SCALAR_NEAR(r.best->ratio, 0.5, ic(0.5));
    CHECK_SCALAR_NEAR(r.best->b, 0.10, ic(0.10));
    CHECK_SCALAR_NEAR(r.best->worst, 0.106088, ic(0.106088, 1e-5));
    r = lab::design_search(0.70, ratios, offsets, pts, lim, 1, "mu1");
    REQUIRE(r.best.has_value());
    CHECK_SCALAR_NEAR(r.best->ratio, 0.6, ic(0.6));
    CHECK_SCALAR_NEAR(r.best->b, 0.0, ic(0.0));
    CHECK_SCALAR_NEAR(r.best->worst, 2.29716, ic(2.29716, 1e-4));
    r = lab::design_search(0.40, vec1(0.5), vec1(0.0), pts, lim);
    CHECK(!r.best.has_value());  // too short to reach every target
}

// ---------------------------------------------------------------- Checkpoint 5: spatial arms
TEST_CASE("test_feeding_path_measures") {
    const Model md = model("srs_7r_feeding");
    const std::pair<VectorXd, MatrixXd> tq = lab::load_feeding_csv();
    const MatrixXd& Q = tq.second;
    REQUIRE(Q.rows() == 100);
    REQUIRE(Q.cols() == 7);
    const lab::PathMeasures m = lab::measures_along_path(md.S, md.M, Q, "linear");
    REQUIRE(m.mu3.size() == 100);
    REQUIRE(m.mu1.size() == 100);
    CHECK_SCALAR_NEAR(m.mu3(0), 0.135794, ic(0.135794, 1e-5));
    CHECK_SCALAR_NEAR(m.mu3(99), 0.014930, ic(0.014930, 1e-5));
    Eigen::Index imin = -1;
    m.mu3.minCoeff(&imin);
    CHECK(imin == 99);
    CHECK_SCALAR_NEAR(m.mu1(99), 57.617, ic(57.617, 1e-2));
    MatrixXd Q2(2, Q.cols());
    Q2.row(0) = Q.row(0);
    Q2.row(1) = Q.row(Q.rows() - 1);
    const lab::PathMeasures ma = lab::measures_along_path(md.S, md.M, Q2, "angular");
    REQUIRE(ma.mu1.size() == 2);
    CHECK_SCALAR_NEAR(ma.mu1(1), 1.3132, ic(1.3132, 1e-4));  // orientation stays well conditioned
}

TEST_CASE("test_joint_space_map_course_arm") {
    const Model m = model("course_arm");
    const VectorXd q0 = m.info.q_home;
    const VectorXd g3 = d2r(Eigen::Vector3d(-30.0, 0.0, 30.0));
    const VectorXd g5 = d2r(Eigen::Vector3d(-90.0, 0.0, 45.0));
    const MatrixXd mp = lab::joint_space_map(m.S, m.M, q0, 2, 4, g3, g5, "sigma_min", "all");
    REQUIRE(mp.rows() == 3);
    REQUIRE(mp.cols() == 3);
    CHECK_MAT_NEAR(mp.col(1), MatrixXd::Zero(3, 1), 1e-9);  // elbow straight (theta3 = 0)
    CHECK_MAT_NEAR(mp.row(1), MatrixXd::Zero(1, 3), 1e-9);  // wrist singular (theta5 = 0)
    CHECK_MAT_NEAR(mp, as_mat(cases()["joint_space_map_course_arm"]["map"]), 1e-9);
    const MatrixXd one = lab::joint_space_map(m.S, m.M, q0, 2, 4, vec1(q0(2)), vec1(q0(4)), "mu3", "all");
    REQUIRE(one.size() == 1);
    const double det = std::abs(mr::jacobian_body(m.B, q0).determinant());
    CHECK_SCALAR_NEAR(one(0, 0), det, ic(det));
    CHECK_THROWS_AS(lab::select_rows(MatrixXd::Identity(6, 6), "upper"), std::invalid_argument);
}
