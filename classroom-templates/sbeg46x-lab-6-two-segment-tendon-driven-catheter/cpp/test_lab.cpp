// Auto-checks for SBEG 46X Lab 6 (constant-curvature kinematics of a two-segment tendon-driven catheter) --
// C++ track. Mirrors test_lab.py test by test (25 test cases, same names and order).
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*arc*,*bend_arc*,*segment_twist*"      (run only some tests, like pytest -k)
// Ground truth: mrcpp (the C++ port of modern_robotics: matrix_exp6, vec_to_se3, trans_inv, adjoint), the
// independent closed form ref_segment and the central finite differences written in this file, the hand values of
// the Week 6 workshop, and cpp_cases.json (numbers of the Python reference, including the random inputs that
// test_lab.py draws), so both languages are checked against the same values.
//
// Plot files: when the functions they use work, the tests write lab6_cpp_segment_backbones.csv (E2),
// lab6_cpp_goto.csv (E7), lab6_cpp_ring.csv and lab6_cpp_ring_backbones.csv (E8) into the folder you run
// lab_tests from; plot them with  python cpp/plot_results.py
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <cmath>
#include <string>
#include <vector>

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using lab::Matrix35d;
using lab::Matrix65d;
using lab::Vector5d;
using lab::Vector6d;
using sbeg::as;
using sbeg::json;

namespace {
constexpr double PI = 3.14159265358979323846;

const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

const lab::Catheter& CAT() {
    static const lab::Catheter c = lab::load_catheter();
    return c;
}

Vector5d qv(double a, double b, double c, double d, double e) {
    Vector5d q;
    q << a, b, c, d, e;
    return q;
}

const std::vector<Vector5d>& Q_TEST() {
    static const std::vector<Vector5d> Q = {qv(0.010, 0.4, -0.7, 1.1, 0.3), qv(0.000, -1.2, 0.2, 0.0, 0.0),
                                            qv(0.025, 0.05, 0.02, -0.6, 1.5)};
    return Q;
}

/// Tendon angles 0, 120, 240 deg (rad).
VectorXd sig120() {
    VectorXd s(3);
    s << 0.0, 2.0 * PI / 3.0, 4.0 * PI / 3.0;
    return s;
}

/// n x 2 matrix of the tendon directions s_i = (cos sigma_i, sin sigma_i).
MatrixXd dirs(const VectorXd& sigma) {
    MatrixXd s(sigma.size(), 2);
    for (Eigen::Index i = 0; i < sigma.size(); ++i) {
        s(i, 0) = std::cos(sigma(i));
        s(i, 1) = std::sin(sigma(i));
    }
    return s;
}

Matrix3d ry(double a) {
    const double c = std::cos(a), s = std::sin(a);
    Matrix3d R;
    R << c, 0, s, 0, 1, 0, -s, 0, c;
    return R;
}

/// Independent closed form of one CC segment written with numerically stable sinc-type factors
/// (Rodrigues' formula MR Eq. 3.51 for R; the arc for p), valid also at and near a straight segment.
Matrix4d ref_segment(const Vector2d& theta_v, double ell) {
    const double tx = theta_v(0), ty = theta_v(1);
    const double th = std::hypot(tx, ty);
    const double a = th < 1e-4 ? 1.0 - th * th / 6 : std::sin(th) / th;
    const double b = th < 1e-4 ? 0.5 - th * th / 24 : 2 * std::pow(std::sin(th / 2), 2) / (th * th);
    const Matrix3d W = mr::vec_to_so3(Vector3d(-ty, tx, 0.0));
    const Matrix3d R = Matrix3d::Identity() + a * W + b * W * W;
    return mr::rp_to_trans(R, ell * Vector3d(b * tx, b * ty, a));
}

Matrix4d ref_fk(const Vector5d& q) {
    Matrix4d Tz = Matrix4d::Identity();
    Tz(2, 3) = q(0);
    return CAT().T_base * Tz * ref_segment(q.segment<2>(1), CAT().ell(0)) * ref_segment(q.segment<2>(3), CAT().ell(1));
}

Matrix65d fd_space_jacobian(const Vector5d& q, double h = 1e-6) {
    const Matrix4d Ti = mr::trans_inv(ref_fk(q));
    Matrix65d J;
    for (int i = 0; i < 5; ++i) {
        Vector5d e = Vector5d::Zero();
        e(i) = h;
        const Matrix4d dT = (ref_fk(q + e) - ref_fk(q - e)) / (2 * h);
        J.col(i) = mr::se3_to_vec(dT * Ti);
    }
    return J;
}

Vector3d pos(const Matrix4d& T) { return T.block<3, 1>(0, 3); }

/// Tip path of data/ablation_ring.csv: t (801) and P (801 x 3).
void load_ring(VectorXd& t, MatrixXd& P) {
    const sbeg::CsvTable ring = lab::load_csv("ablation_ring.csv");
    t = ring.data.col(ring.col("t"));
    P.resize(ring.data.rows(), 3);
    P.col(0) = ring.data.col(ring.col("x"));
    P.col(1) = ring.data.col(ring.col("y"));
    P.col(2) = ring.data.col(ring.col("z"));
}
}  // namespace

// ============================================================================ data
TEST_CASE("test_catheter_json") {
    const lab::Catheter& c = CAT();
    CHECK((c.ell == Vector2d(0.040, 0.025)));
    CHECK((c.d == Vector2d(0.0010, 0.0010)));
    CHECK((c.EI == Vector2d(1.0e-4, 0.6e-4)));
    for (int j = 0; j < 2; ++j)
        for (int i = 0; i < 2; ++i) CHECK_SCALAR_NEAR(c.sigma_deg(j, i + 1) - c.sigma_deg(j, i), 120.0, 1e-8);
    const Matrix3d R = c.T_base.topLeftCorner<3, 3>();
    CHECK_MAT_NEAR(R.transpose() * R, Matrix3d::Identity(), 1e-8);
    CHECK_APPROX(R.determinant(), 1.0);
    // C++ addition (so that this test needs your code): a straight catheter's tendons are as long as its segments.
    CHECK_MAT_NEAR(lab::catheter_tendon_lengths(Vector5d::Zero(), c),
                   (Vector6d() << 0.040, 0.040, 0.040, 0.065, 0.065, 0.065).finished(), 1e-15);
}

TEST_CASE("test_data_files_load") {
    const sbeg::CsvTable ring = lab::load_csv("ablation_ring.csv");
    CHECK((ring.header == std::vector<std::string>{"t", "x", "y", "z"}));
    CHECK(ring.data.rows() == 801);
    const sbeg::CsvTable log = lab::load_csv("tendon_log.csv");
    CHECK(log.data.rows() == 5);
    CHECK_NOTHROW((void)log.col("pose"));
    CHECK_NOTHROW((void)log.col("dl6"));
    // C++ addition (so that this test needs your code): pose 2 of the log bends only segment 2, by 1.2 rad.
    REQUIRE(log.data.rows() == 5);
    Vector6d dl;
    for (int i = 0; i < 6; ++i) dl(i) = log.data(1, log.col("dl" + std::to_string(i + 1)));
    CHECK_MAT_NEAR(lab::bend_from_catheter_tendons(dl, CAT()), Eigen::Vector4d(0.0, 0.0, 0.0, 1.2), 0.03);
}

// ============================================================================ Part A -- one segment
TEST_CASE("test_arc_transform_hand_values") {
    // Workshop P1: kappa = 10 1/m, ell = pi/20 m -> a quarter circle of radius 0.1 m.
    Matrix4d T = lab::arc_transform(10.0, 0.0, PI / 20);
    CHECK_MAT_NEAR(pos(T), Vector3d(0.1, 0.0, 0.1), 1e-12);
    CHECK_MAT_NEAR((T.topLeftCorner<3, 3>()), ry(PI / 2), 1e-12);
    T = lab::arc_transform(10.0, PI / 2, PI / 20);
    CHECK_MAT_NEAR(pos(T), Vector3d(0.0, 0.1, 0.1), 1e-12);
    T = lab::arc_transform(0.0, 1.3, 0.03);  // straight
    Matrix4d T_s = Matrix4d::Identity();
    T_s(2, 3) = 0.03;
    CHECK_MAT_NEAR(T, T_s, 1e-12);
    const json& c = cases()["arc_transform_hand_values"];
    CHECK_MAT_NEAR(lab::arc_transform(10.0, 0.0, PI / 20), as<Matrix4d>(c["T_phi0"]), 1e-12);
    CHECK_MAT_NEAR(lab::arc_transform(10.0, PI / 2, PI / 20), as<Matrix4d>(c["T_phi90"]), 1e-12);
    CHECK_MAT_NEAR(lab::arc_transform(0.0, 1.3, 0.03), as<Matrix4d>(c["T_straight"]), 1e-12);

    // E2 plot file: backbones of one 40 mm segment for theta = 0, 30, ..., 150 deg in the plane phi = 0.
    const double ell = 0.04;
    MatrixXd out(6 * 41, 4);
    for (int a = 0; a < 6; ++a) {
        const double th = a * 30.0 * PI / 180.0;
        for (int k = 0; k <= 40; ++k) {
            const double s = ell * k / 40.0;
            const Vector3d p = pos(lab::arc_transform(th / ell, 0.0, s));
            out.row(a * 41 + k) << a * 30.0, s, p(0), p(2);
        }
    }
    sbeg::write_csv("lab6_cpp_segment_backbones.csv", {"theta_deg", "s", "x", "z"}, out);
}

TEST_CASE("test_arc_transform_matches_mr_exponential") {
    for (const auto& c : cases()["arc_transform_matches_mr_exponential"]) {
        const double k = c["kappa"].get<double>(), phi = c["phi"].get<double>(), ell = c["ell"].get<double>();
        INFO("kappa = " << k << ", phi = " << phi << ", ell = " << ell);
        Vector6d S;
        S << -k * std::sin(phi), k * std::cos(phi), 0, 0, 0, 1.0;  // twist per unit arc length
        const Matrix4d T_ref = mr::matrix_exp6(mr::vec_to_se3(S * ell));
        CHECK_MAT_NEAR(lab::arc_transform(k, phi, ell), T_ref, 1e-12);
        CHECK_MAT_NEAR(lab::arc_transform(k, phi, ell), as<Matrix4d>(c["T_mr"]), 1e-12);
    }
}

TEST_CASE("test_arc_tip_z_axis_is_the_backbone_tangent") {
    const double k = 18.0, phi = 0.7, ell = 0.05, h = 1e-7;
    const Matrix4d T = lab::arc_transform(k, phi, ell);
    const Vector3d tangent = (pos(lab::arc_transform(k, phi, ell + h)) - pos(lab::arc_transform(k, phi, ell - h))) / (2 * h);
    CHECK_MAT_NEAR(tangent, (T.block<3, 1>(0, 2)), 1e-7);
    const Matrix3d R = T.topLeftCorner<3, 3>();
    CHECK_MAT_NEAR(R.transpose() * R, Matrix3d::Identity(), 1e-12);
}

TEST_CASE("test_bend_arc_roundtrip") {
    lab::ArcParams a = lab::bend_to_arc(Vector2d(0.0, 0.0), 0.04);
    CHECK(a.kappa == 0.0);
    CHECK(a.theta == 0.0);
    CHECK(a.phi == 0.0);
    a = lab::bend_to_arc(Vector2d(-0.3, 0.4), 0.04);
    CHECK_APPROX(a.theta, 0.5);
    CHECK_APPROX(a.kappa, 12.5);
    CHECK_APPROX(a.phi, std::atan2(0.4, -0.3));
    CHECK_MAT_NEAR(lab::arc_to_bend(a.kappa, a.phi, 0.04), Vector2d(-0.3, 0.4), 1e-8);
    const json& c = cases()["bend_arc_roundtrip"];
    CHECK_APPROX(a.phi, c["phi"].get<double>());
}

TEST_CASE("test_segment_twist_and_exponential") {
    const Vector2d theta_v(0.6, -0.8);
    const double ell = 0.03;
    const Vector6d V = lab::segment_twist(theta_v, ell);
    CHECK_MAT_NEAR(V, (Vector6d() << 0.8, 0.6, 0, 0, 0, 0.03).finished(), 1e-8);
    const lab::ArcParams a = lab::bend_to_arc(theta_v, ell);
    CHECK_MAT_NEAR(lab::segment_transform_exp(theta_v, ell), lab::arc_transform(a.kappa, a.phi, ell), 1e-12);
    CHECK_MAT_NEAR(lab::segment_transform_exp(theta_v, ell), ref_segment(theta_v, ell), 1e-12);
    CHECK_MAT_NEAR(lab::segment_transform_exp(theta_v, ell), as<Matrix4d>(cases()["segment_twist_and_exponential"]["T"]),
                   1e-12);
}

TEST_CASE("test_tendon_lengths_hand_values") {
    // Workshop P2: d = 1 mm, ell = 40 mm, bent 90 deg towards phi = 0.
    const VectorXd sig = sig120();
    const VectorXd l = lab::tendon_lengths(Vector2d(PI / 2, 0.0), 0.040, 0.001, sig);
    REQUIRE(l.size() == 3);
    const VectorXd expected = (VectorXd::Constant(3, 0.040) - 0.001 * PI / 2 * sig.array().cos().matrix()).eval();
    CHECK_MAT_NEAR(l, expected, 1e-15);
    CHECK_APPROX(l(0), 0.040 - 0.001 * PI / 2);
    CHECK_SCALAR_NEAR((l.array() - 0.040).sum(), 0.0, 1e-15);  // 3 equally spaced tendons: zero sum
    CHECK_MAT_NEAR(l, as<VectorXd>(cases()["tendon_lengths_hand_values"]["l"]), 1e-15);
}

TEST_CASE("test_bend_from_tendons_roundtrip") {
    const VectorXd sig = sig120();
    const MatrixXd s = dirs(sig);
    for (const auto& c : cases()["bend_from_tendons_roundtrip"]) {
        const Vector2d theta_v = as<Vector2d>(c["theta_v"]);
        INFO("theta_v = " << theta_v.transpose());
        const VectorXd dl = (lab::tendon_lengths(theta_v, 0.04, 0.001, sig).array() - 0.04).matrix();
        CHECK_MAT_NEAR(dl, as<VectorXd>(c["dl"]), 1e-15);
        CHECK_MAT_NEAR(lab::bend_from_tendons(dl, 0.001, sig), theta_v, 1e-12);
        CHECK_MAT_NEAR((-(2 / (3 * 0.001)) * s.transpose() * dl).eval(), theta_v, 1e-12);  // closed form
    }
}

TEST_CASE("test_tension_allocation") {
    const VectorXd sig = sig120();
    const MatrixXd s = dirs(sig);
    for (const auto& c : cases()["tension_allocation"]) {
        const Vector2d m = as<Vector2d>(c["m"]);
        INFO("m = " << m.transpose());
        const VectorXd t = lab::tension_allocation(m, 0.001, sig);
        REQUIRE(t.size() == 3);
        CHECK_SCALAR_NEAR(t.minCoeff(), 0.0, 1e-12);
        CHECK((t.array() >= -1e-12).all());
        CHECK_MAT_NEAR((0.001 * s.transpose() * t).eval(), m, 1e-14);
        CHECK_MAT_NEAR(t, as<VectorXd>(c["t"]), 1e-12);
    }
}

TEST_CASE("test_segment_tensions_single_tendon_hand_value") {
    // Bending towards a tendon needs only that tendon: t = EI theta / (ell d).
    const VectorXd t = lab::segment_tensions(Vector2d(0.5, 0.0), 0.04, 0.001, sig120(), 1.0e-4);
    CHECK_MAT_NEAR(t, Vector3d(1.0e-4 * 0.5 / (0.04 * 0.001), 0, 0), 1e-12);  // 1.25 N
}

// ============================================================================ Part B -- two-segment catheter
TEST_CASE("test_catheter_fk_matches_independent_product") {
    for (const auto& c : cases()["catheter_fk_matches_independent_product"]) {
        const Vector5d q = as<Vector5d>(c["q"]);
        INFO("q = " << q.transpose());
        CHECK_MAT_NEAR(lab::catheter_fk(q, CAT()), ref_fk(q), 1e-12);
        CHECK_MAT_NEAR(lab::catheter_fk(q, CAT()), as<Matrix4d>(c["T"]), 1e-12);
    }
}

TEST_CASE("test_catheter_fk_straight") {
    const Matrix4d T = lab::catheter_fk(qv(0.012, 0, 0, 0, 0), CAT());
    const Vector3d z = CAT().T_base.block<3, 1>(0, 2);
    CHECK_MAT_NEAR(pos(T), (pos(CAT().T_base) + (0.012 + 0.065) * z).eval(), 1e-12);
    CHECK_MAT_NEAR((T.topLeftCorner<3, 3>()), (CAT().T_base.topLeftCorner<3, 3>()), 1e-12);
}

TEST_CASE("test_catheter_tendon_lengths") {
    const Vector6d l0 = lab::catheter_tendon_lengths(Vector5d::Zero(), CAT());
    CHECK_MAT_NEAR(l0, (Vector6d() << 0.040, 0.040, 0.040, 0.065, 0.065, 0.065).finished(), 1e-8);
    const Vector5d q = Q_TEST()[0];
    Vector5d q2 = q;
    q2(0) = 0.0;
    CHECK_MAT_NEAR(lab::catheter_tendon_lengths(q, CAT()), lab::catheter_tendon_lengths(q2, CAT()), 1e-8);
    const MatrixXd s2 = dirs(CAT().sigma.row(1).transpose());
    const Vector3d dl2 = (lab::catheter_tendon_lengths(q, CAT()).tail<3>().array() - 0.065).matrix();
    CHECK_MAT_NEAR(dl2, (-0.001 * s2 * (q.segment<2>(1) + q.segment<2>(3))).eval(), 1e-15);
    CHECK_MAT_NEAR(lab::catheter_tendon_lengths(q, CAT()), as<Vector6d>(cases()["catheter_tendon_lengths"]["l_q0"]),
                   1e-15);
}

TEST_CASE("test_catheter_tendon_jacobian") {
    const Matrix65d A = lab::catheter_tendon_jacobian(CAT());
    CHECK_MAT_NEAR(A.col(0), Vector6d::Zero(), 1e-8);
    const Vector5d q = Q_TEST()[2];
    const double h = 1e-6;
    for (int i = 0; i < 5; ++i) {
        Vector5d e = Vector5d::Zero();
        e(i) = h;
        const Vector6d col = (lab::catheter_tendon_lengths(q + e, CAT()) - lab::catheter_tendon_lengths(q - e, CAT())) / (2 * h);
        INFO("column " << i);
        CHECK_MAT_NEAR(A.col(i), col, 1e-9);
    }
    CHECK_MAT_NEAR(A, as<Matrix65d>(cases()["catheter_tendon_jacobian"]["A"]), 1e-15);
}

TEST_CASE("test_bend_from_catheter_tendons_and_log") {
    const Vector6d l0 = lab::catheter_tendon_lengths(Vector5d::Zero(), CAT());
    for (const Vector5d& q : Q_TEST()) {
        const Vector6d dl = lab::catheter_tendon_lengths(q, CAT()) - l0;
        CHECK_MAT_NEAR(lab::bend_from_catheter_tendons(dl, CAT()), q.tail<4>(), 1e-12);
    }
    const sbeg::CsvTable log = lab::load_csv("tendon_log.csv");
    REQUIRE(log.data.rows() == 5);
    MatrixXd truth(5, 4);
    truth << 0.5, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 1.2,
             0.4, -0.6, 0.8, 0.3,
             -1.0, 0.5, 1.5, -0.4,
             1.6, 1.0, -0.9, 0.9;
    for (int k = 0; k < 5; ++k) {
        Vector6d dl;
        for (int i = 0; i < 6; ++i) dl(i) = log.data(k, log.col("dl" + std::to_string(i + 1)));
        INFO("pose " << k + 1);
        CHECK_MAT_NEAR(lab::bend_from_catheter_tendons(dl, CAT()), truth.row(k), 0.03);  // 5 um encoder noise
    }
}

TEST_CASE("test_catheter_tensions_equilibrium") {
    CHECK_MAT_NEAR(lab::catheter_tensions(Vector5d::Zero(), CAT()), Vector6d::Zero(), 1e-8);
    const MatrixXd s0 = dirs(CAT().sigma.row(0).transpose()), s1 = dirs(CAT().sigma.row(1).transpose());
    for (const Vector5d& q : Q_TEST()) {
        INFO("q = " << q.transpose());
        const Vector6d t = lab::catheter_tensions(q, CAT());
        CHECK((t.array() >= -1e-12).all());
        CHECK_SCALAR_NEAR(t.tail<3>().minCoeff(), 0.0, 1e-12);
        const Vector2d m2 = 0.001 * s1.transpose() * t.tail<3>();
        const Vector2d m1 = 0.001 * s0.transpose() * t.head<3>() + m2;
        CHECK_MAT_NEAR(m2, (CAT().EI(1) / CAT().ell(1) * q.segment<2>(3)).eval(), 1e-12);
        CHECK_MAT_NEAR(m1, (CAT().EI(0) / CAT().ell(0) * q.segment<2>(1)).eval(), 1e-12);
    }
}

TEST_CASE("test_dexp_left_matches_finite_difference") {
    Vector6d V, Vd;
    V << 0.3, -0.9, 0.2, 0.01, 0.02, 0.04;
    Vd << 0.5, 0.1, -0.2, 0.03, -0.01, 0.02;
    const double h = 1e-6;
    const Matrix4d dT = (mr::matrix_exp6(mr::vec_to_se3(V + h * Vd)) - mr::matrix_exp6(mr::vec_to_se3(V - h * Vd))) / (2 * h);
    const Vector6d ref = mr::se3_to_vec(dT * mr::trans_inv(mr::matrix_exp6(mr::vec_to_se3(V))));
    CHECK_MAT_NEAR((lab::dexp_left(V) * Vd).eval(), ref, 1e-8);
    CHECK_MAT_NEAR((lab::dexp_left(V) * Vd).eval(), as<Vector6d>(cases()["dexp_left_matches_finite_difference"]["ref"]),
                   1e-8);
    CHECK_MAT_NEAR(lab::dexp_left(Vector6d::Zero()), lab::Matrix6d::Identity(), 1e-8);
}

TEST_CASE("test_space_jacobian_matches_finite_difference") {
    const json& cs = cases()["space_jacobian_matches_finite_difference"];
    for (std::size_t k = 0; k < Q_TEST().size(); ++k) {
        const Vector5d& q = Q_TEST()[k];
        INFO("q = " << q.transpose());
        const Matrix65d J = lab::catheter_jacobian_space(q, CAT());
        CHECK_MAT_NEAR(J, fd_space_jacobian(q), 1e-7);
        CHECK_MAT_NEAR(J, as<Matrix65d>(cs[k]["J_fd"]), 1e-7);
    }
}

TEST_CASE("test_space_jacobian_insertion_column") {
    const Matrix65d J = lab::catheter_jacobian_space(Q_TEST()[0], CAT());
    Vector6d ez;
    ez << 0, 0, 0, 0, 0, 1;
    CHECK_MAT_NEAR(J.col(0), (mr::adjoint(CAT().T_base) * ez).eval(), 1e-8);
}

TEST_CASE("test_tip_position_jacobian") {
    for (const Vector5d& q : Q_TEST()) {
        INFO("q = " << q.transpose());
        const Matrix35d Jp = lab::tip_position_jacobian(q, CAT());
        const double h = 1e-6;
        for (int i = 0; i < 5; ++i) {
            Vector5d e = Vector5d::Zero();
            e(i) = h;
            const Vector3d col = (pos(ref_fk(q + e)) - pos(ref_fk(q - e))) / (2 * h);
            INFO("column " << i);
            CHECK_MAT_NEAR(Jp.col(i), col, 1e-8);
        }
        // same thing from the body Jacobian: pdot = R_sb J_b,v qdot
        const Matrix4d T = ref_fk(q);
        const Matrix65d Jb = mr::adjoint(mr::trans_inv(T)) * lab::catheter_jacobian_space(q, CAT());
        CHECK_MAT_NEAR(Jp, (T.topLeftCorner<3, 3>() * Jb.bottomRows<3>()).eval(), 1e-12);
    }
}

// ============================================================================ Part C -- resolved-rate control
TEST_CASE("test_clamp_q") {
    const Vector5d q = lab::clamp_q(qv(0.05, 3.0, 4.0, 0.1, 0.0), CAT());
    CHECK_APPROX(q(0), 0.030);
    CHECK_APPROX(q.segment<2>(1).norm(), CAT().theta_max);
    CHECK_APPROX(q(1) / q(2), 0.75);
    CHECK_MAT_NEAR(q.tail<2>(), Vector2d(0.1, 0.0), 1e-8);
    CHECK(lab::clamp_q(qv(-0.01, 0, 0, 0, 0), CAT())(0) == 0.0);
}

TEST_CASE("test_dls_inverse") {
    const json& c = cases()["dls_inverse"];
    const MatrixXd J = sbeg::as_mat(c["J"]);
    CHECK_MAT_NEAR(lab::dls_inverse(J, 0.0), sbeg::as_mat(c["pinv"]), 1e-12);
    CHECK_MAT_NEAR(lab::dls_inverse(J, 0.0), mr::pinv(J), 1e-12);
    const double lam = 0.1;
    const MatrixXd ref = (J.transpose() * J + lam * lam * MatrixXd::Identity(5, 5)).partialPivLu().solve(J.transpose());
    CHECK_MAT_NEAR(lab::dls_inverse(J, lam), ref, 1e-12);
    CHECK_MAT_NEAR(lab::dls_inverse(J, lam), sbeg::as_mat(c["dls"]), 1e-12);
}

TEST_CASE("test_resolved_rate_reaches_a_target") {
    const Vector5d q_goal = qv(0.018, 0.9, -0.4, -0.5, 0.8);
    const Vector3d p_goal = pos(ref_fk(q_goal));
    const lab::GoToResult out = lab::go_to(qv(0.010, 0.2, 0.0, 0.2, 0.0), p_goal, CAT(), 0.01, 5.0, 600);
    REQUIRE(out.err.size() == 601);
    REQUIRE(out.q.rows() == 601);
    REQUIRE(out.q.cols() == 5);
    CHECK(out.err(600) < 1e-6);
    bool monotone = true;
    for (int k = 0; k + 1 < 200; ++k) monotone = monotone && (out.err(k + 1) - out.err(k) < 1e-9);
    CHECK((monotone || out.err(200) < 0.5 * out.err(0)));
    for (Eigen::Index k = 0; k < out.q.rows(); ++k) CHECK(out.q.row(k).segment<2>(1).norm() <= CAT().theta_max + 1e-12);
}

TEST_CASE("test_track_ablation_ring") {
    VectorXd t;
    MatrixXd P;
    load_ring(t, P);
    REQUIRE(P.rows() == 801);
    const lab::GoToResult start =
        lab::go_to(qv(0.010, 0.0, 0.3, 0.0, 0.3), P.row(0).transpose(), CAT(), 0.01, 5.0, 600);
    REQUIRE(start.err.size() == 601);
    REQUIRE(start.q.rows() == 601);
    CHECK(start.err(600) < 1e-6);
    {  // E7 plot file
        MatrixXd out(601, 2);
        for (int k = 0; k <= 600; ++k) out.row(k) << k, start.err(k);
        sbeg::write_csv("lab6_cpp_goto.csv", {"step", "err"}, out);
    }
    const Vector5d q0 = start.q.row(600).transpose();
    const lab::TrackResult res = lab::track_path(q0, t, P, CAT(), 20.0);
    REQUIRE(res.q.rows() == 801);
    REQUIRE(res.q.cols() == 5);
    REQUIRE(res.tendon_dl.rows() == 801);
    REQUIRE(res.tendon_dl.cols() == 6);
    REQUIRE(res.err.size() == 801);
    REQUIRE(res.p.rows() == 801);
    REQUIRE(res.p.cols() == 3);
    CHECK(res.err.maxCoeff() < 0.3e-3);  // 0.3 mm
    for (Eigen::Index k = 0; k < 801; k += 50) {
        const Vector5d qq = res.q.row(k).transpose();
        CHECK_MAT_NEAR(lab::catheter_fk(qq, CAT()), ref_fk(qq), 1e-12);
    }
    CHECK((res.q.col(0).array() >= 0.0).all());
    CHECK((res.q.col(0).array() <= 0.030).all());
    CHECK_MAT_NEAR(res.tendon_dl.leftCols<3>().rowwise().sum(), VectorXd::Zero(801), 1e-12);

    // E8 plot files: commanded and achieved ring, q and tendon displacements; the catheter at four instants.
    MatrixXd out(801, 1 + 3 + 3 + 1 + 5 + 6);
    out << t, P, res.p, res.err, res.q, res.tendon_dl;
    sbeg::write_csv("lab6_cpp_ring.csv",
                    {"t", "x_des", "y_des", "z_des", "x", "y", "z", "err", "z_ins", "th1x", "th1y", "th2x", "th2y",
                     "dl1", "dl2", "dl3", "dl4", "dl5", "dl6"},
                    out);
    MatrixXd bb(4 * 42, 4);
    const int instants[4] = {0, 200, 400, 600};
    for (int i = 0; i < 4; ++i) {
        const MatrixXd pts = lab::catheter_backbone(res.q.row(instants[i]).transpose(), CAT());
        REQUIRE(pts.rows() == 42);
        bb.block(i * 42, 0, 42, 1).setConstant(t(instants[i]));
        bb.block(i * 42, 1, 42, 3) = pts;
    }
    sbeg::write_csv("lab6_cpp_ring_backbones.csv", {"t", "x", "y", "z"}, bb);
}
