// Auto-checks for SBEG 46X Lab 5 (navigation pipeline: pivot calibration, paired-point registration, FLE/FRE/TRE)
// -- C++ track. Mirrors test_lab.py test by test (20 test cases, same names and order).
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*A1*,*A2*"      (run only some tests, like pytest -k)
// Ground truth (as in Python): mrcpp (the C++ port of modern_robotics: fkin_space, trans_inv), an independent
// standard-DH forward kinematics of the course arm, Horn's unit-quaternion registration (a different algorithm from
// the SVD), the eigenvalue form of the Fitzpatrick-West-Maurer TRE formula, brute-force searches written in this file,
// the hand values of the Week 5 workshop, and cpp_cases.json, which holds the random inputs of test_lab.py and the
// numbers of the Python reference, so both languages are checked against the same values.
// np.allclose(a, b, atol=x) is checked with its NumPy meaning: |a - b| <= x + 1e-5 |b| for every entry.
//
// The three data tests also call one of your functions (apply_T, flange_pose_mm), so with the untouched starter
// every test case fails with "TODO: <function>".
//
// Plot data: when test_D2 passes its checks it also writes lab5_cpp_mc.csv (E7), lab5_cpp_tre_map.csv (E8) and
// lab5_cpp_points.csv (E10) into the folder you run lab_tests from; plot them with  python cpp/plot_results.py
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using sbeg::as;
using sbeg::as_list;
using sbeg::as_mat;
using sbeg::as_vec;
using sbeg::json;

namespace {
constexpr double PI = 3.14159265358979323846;

const json& cases() {  // loaded once
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

const lab::Phantom& PH() {  // the phantom, read once with the given helper (Python: PH = lab.load_phantom())
    static const lab::Phantom P = lab::load_phantom();
    return P;
}

/// np.allclose(got, expected, atol=atol) with NumPy's default rtol = 1e-5: |got - expected| <= atol + rtol |expected|.
template <typename A, typename B>
sbeg::NearResult allclose(const Eigen::MatrixBase<A>& got_in, const Eigen::MatrixBase<B>& exp_in, double atol) {
    MatrixXd got = got_in.template cast<double>(), exp = exp_in.template cast<double>();
    if ((got.rows() == 1 || got.cols() == 1) && (exp.rows() == 1 || exp.cols() == 1) && got.size() == exp.size()) {
        got = Eigen::Map<VectorXd>(got.data(), got.size()).eval();
        exp = Eigen::Map<VectorXd>(exp.data(), exp.size()).eval();
    }
    sbeg::NearResult r = sbeg::compare_near(got, exp, atol);  // shape / NaN checks and the message
    if (!r.ok && got.rows() == exp.rows() && got.cols() == exp.cols() && got.allFinite()) {
        r.ok = ((got - exp).cwiseAbs().array() <= atol + 1e-5 * exp.cwiseAbs().array()).all();
    }
    return r;
}
}  // namespace

#define CHECK_ALLCLOSE(got, expected, atol)                                                                \
    do {                                                                                                   \
        const ::sbeg::NearResult ac_ = allclose((got), (expected), (atol));                                \
        CHECK_MESSAGE(ac_.ok, "np.allclose(" #got ", " #expected ", atol=" << (atol) << ")\n  " << ac_.msg); \
    } while (false)

namespace {
/// pytest.approx(expected, rel=rel): tolerance max(rel |expected|, 1e-12).
double rel_tol(double expected, double rel) { return std::max(rel * std::abs(expected), 1e-12); }

/// Standard DH FK of the course arm flange (table in Course resources), base = I, tool = Rot(z, pi); mm.
Matrix4d dh_fk_mm(const VectorXd& q) {
    const double d[6] = {cm::H1, 0.0, 0.0, cm::W1, cm::H2, cm::W2};
    const double a[6] = {0.0, cm::L1, cm::L2, 0.0, 0.0, 0.0};
    const double al[6] = {-PI / 2, 0.0, 0.0, -PI / 2, PI / 2, 0.0};
    Matrix4d T = Matrix4d::Identity();
    for (int i = 0; i < 6; ++i) {
        const double ct = std::cos(q(i)), st = std::sin(q(i)), ca = std::cos(al[i]), sa = std::sin(al[i]);
        Matrix4d A;
        A << ct, -st * ca, st * sa, a[i] * ct,
             st, ct * ca, -ct * sa, a[i] * st,
             0.0, sa, ca, d[i],
             0.0, 0.0, 0.0, 1.0;
        T = (T * A).eval();
    }
    T = (T * Eigen::Vector4d(-1.0, -1.0, 1.0, 1.0).asDiagonal()).eval();
    T.block<3, 1>(0, 3) *= 1000.0;
    return T;
}

/// Horn (1987) closed-form registration with unit quaternions: R, t with Y ~ R X + t.
std::pair<Matrix3d, Vector3d> horn(const MatrixXd& X, const MatrixXd& Y) {
    const Eigen::RowVector3d xb = X.colwise().mean(), yb = Y.colwise().mean();
    const MatrixXd Xc = X.rowwise() - xb, Yc = Y.rowwise() - yb;
    const Matrix3d S = Xc.transpose() * Yc;
    const double Sxx = S(0, 0), Sxy = S(0, 1), Sxz = S(0, 2), Syx = S(1, 0), Syy = S(1, 1), Syz = S(1, 2),
                 Szx = S(2, 0), Szy = S(2, 1), Szz = S(2, 2);
    Eigen::Matrix4d N;
    N << Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx,
         Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz,
         Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy,
         Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz;
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> es(N);
    const Eigen::Vector4d v = es.eigenvectors().col(3);  // largest eigenvalue (ascending order, as eigh)
    const double q0 = v(0), qx = v(1), qy = v(2), qz = v(3);
    Matrix3d R;
    R << q0 * q0 + qx * qx - qy * qy - qz * qz, 2 * (qx * qy - q0 * qz), 2 * (qx * qz + q0 * qy),
         2 * (qy * qx + q0 * qz), q0 * q0 - qx * qx + qy * qy - qz * qz, 2 * (qy * qz - q0 * qx),
         2 * (qz * qx - q0 * qy), 2 * (qz * qy + q0 * qx), q0 * q0 - qx * qx - qy * qy + qz * qz;
    return {R, yb.transpose() - R * xb.transpose()};
}

/// Fitzpatrick-West-Maurer with the eigenvalues L_k of sum x~ x~^T:  f_k^2 = (L_i + L_j)/N (i, j != k).
double tre_eig(const MatrixXd& X, const Vector3d& r, double fle) {
    const double N = static_cast<double>(X.rows());
    const Eigen::RowVector3d xb = X.colwise().mean();
    const MatrixXd Xc = X.rowwise() - xb;
    const Eigen::SelfAdjointEigenSolver<Matrix3d> es(Xc.transpose() * Xc);
    const Vector3d L = es.eigenvalues();
    const Vector3d rr = es.eigenvectors().transpose() * (r - xb.transpose());
    double tot = 0.0;
    for (int k = 0; k < 3; ++k) {
        const int i = (k + 1) % 3, j = (k + 2) % 3;
        tot += (rr(i) * rr(i) + rr(j) * rr(j)) / ((L(i) + L(j)) / N);
    }
    return std::sqrt(fle * fle / N * (1.0 + tot / 3.0));
}

/// Least squares (np.linalg.lstsq) for the independent references.
VectorXd lstsq(const MatrixXd& A, const VectorXd& b) {
    return A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
}

/// k-subsets of {0..n-1} in itertools.combinations order (independent of the given lab::combinations).
void combos(int n, int k, int start, std::vector<int>& cur, std::vector<std::vector<int>>& out) {
    if (static_cast<int>(cur.size()) == k) {
        out.push_back(cur);
        return;
    }
    for (int i = start; i < n; ++i) {
        cur.push_back(i);
        combos(n, k, i + 1, cur, out);
        cur.pop_back();
    }
}

MatrixXd rows_of(const MatrixXd& X, const std::vector<int>& idx) {
    MatrixXd out(static_cast<Eigen::Index>(idx.size()), X.cols());
    for (std::size_t m = 0; m < idx.size(); ++m) out.row(static_cast<Eigen::Index>(m)) = X.row(idx[m]);
    return out;
}

/// Points as rows (N x 3): Y = X R^T + t.
MatrixXd move_rows(const MatrixXd& X, const Matrix3d& R, const Vector3d& t) {
    MatrixXd Y = X * R.transpose();
    Y.rowwise() += t.transpose();
    return Y;
}

Matrix3d rot_of(const Matrix4d& T) { return T.topLeftCorner<3, 3>(); }
Vector3d pos_of(const Matrix4d& T) { return T.block<3, 1>(0, 3); }

Vector3d hom_apply(const Matrix4d& T, const Vector3d& p) { return (T * p.homogeneous()).head<3>(); }

double rms(const VectorXd& v) { return std::sqrt(v.squaredNorm() / static_cast<double>(v.size())); }
}  // namespace

// ============================================================================ data and infrastructure
TEST_CASE("test_data_files_load") {
    const MatrixXd& X_FID = PH().fiducials_image_mm;
    CHECK((X_FID.rows() == 6 && X_FID.cols() == 3 && PH().check_points_image_mm.rows() == 2 &&
           PH().check_points_image_mm.cols() == 3));
    std::set<std::string> names;
    for (const auto& kv : PH().targets_image_mm) names.insert(kv.first);
    const std::set<std::string> expected_names = {"T1_central", "T2_lateral", "T3_deep"};
    CHECK(names == expected_names);
    const std::vector<std::pair<std::string, int>> logs = {
        {"pivot_joints.csv", 10}, {"fiducial_joints.csv", 6}, {"check_joints.csv", 2}};
    for (const auto& log : logs) {
        const std::string& name = log.first;
        const int n = log.second;
        INFO("file " << name);
        const auto [labels, Q] = lab::load_joints(name);
        CHECK(Q.cols() == 6);
        CHECK(static_cast<Eigen::Index>(labels.size()) == Q.rows());
        CHECK(Q.rows() >= n);
    }
    // C++ track: the fiducial array also passes through your apply_T unchanged with T = I
    CHECK_ALLCLOSE(lab::apply_T(Matrix4d::Identity(), X_FID), X_FID, 1e-12);
}

TEST_CASE("test_course_model_matches_dh_table") {
    // course_models (PoE) and the DH table of the course robot models give the same flange pose.
    const cm::CourseArm k = cm::course_arm_kinematics();
    const json& c = cases()["course_model_matches_dh_table"];
    const MatrixXd qs = as_mat(c["q"]);
    const auto T_ref = as_list<Matrix4d>(c["T_mm"]);
    REQUIRE(T_ref.size() == 3);
    CHECK_ALLCLOSE(qs.row(1), cm::q_home(), 0.0);
    for (Eigen::Index i = 0; i < qs.rows(); ++i) {
        const VectorXd q = qs.row(i).transpose();
        INFO("q = " << q.transpose());
        Matrix4d T = mr::fkin_space(k.M, k.Slist, q);
        T.block<3, 1>(0, 3) *= 1000.0;
        CHECK_ALLCLOSE(T, dh_fk_mm(q), 1e-9);
        CHECK_ALLCLOSE(T, T_ref[static_cast<std::size_t>(i)], 1e-9);
        // C++ track: your flange_pose_mm is the same model in mm
        CHECK_ALLCLOSE(lab::flange_pose_mm(q), dh_fk_mm(q), 1e-8);
    }
}

TEST_CASE("test_logged_angles_are_encoder_counts_inside_soft_limits") {
    const cm::CourseArm k = cm::course_arm_kinematics();
    for (const std::string name : {"pivot_joints.csv", "fiducial_joints.csv", "check_joints.csv"}) {
        INFO("file " << name);
        const MatrixXd Q = lab::load_joints(name).second;
        const MatrixXd counts = Q / cm::ENC_RES;
        CHECK_ALLCLOSE(counts, counts.array().round().matrix(), 2e-3);
        for (Eigen::Index i = 0; i < Q.rows(); ++i)
            for (Eigen::Index j = 0; j < 6; ++j) CHECK(std::abs(Q(i, j)) <= k.limits(j, 1) + 1e-9);
        // C++ track: every logged pose gives a valid flange pose with your flange_pose_mm
        for (Eigen::Index i = 0; i < Q.rows(); ++i) {
            const Matrix4d T = lab::flange_pose_mm(Q.row(i).transpose());
            CHECK(mr::test_if_so3(rot_of(T)));
        }
    }
}

// ============================================================================ Part A
TEST_CASE("test_A1_flange_pose_mm_matches_dh") {
    const json& c = cases()["A1_flange_pose_mm_matches_dh"];
    const MatrixXd qs = as_mat(c["q"]);
    const auto T_ref = as_list<Matrix4d>(c["T"]);
    REQUIRE(T_ref.size() == static_cast<std::size_t>(qs.rows()));
    for (Eigen::Index i = 0; i < qs.rows(); ++i) {
        const VectorXd q = qs.row(i).transpose();
        INFO("q = " << q.transpose());
        CHECK_ALLCLOSE(lab::flange_pose_mm(q), dh_fk_mm(q), 1e-8);
        CHECK_ALLCLOSE(lab::flange_pose_mm(q), T_ref[static_cast<std::size_t>(i)], 1e-8);
    }
    const Matrix4d T0 = lab::flange_pose_mm(VectorXd::Zero(6));
    CHECK_ALLCLOSE(pos_of(T0), Vector3d(206.0, 109.0, 56.1), 1e-9);
}

TEST_CASE("test_A2_pivot_calibration_exact_and_workshop_values") {
    const json& c = cases()["A2_pivot_calibration_exact_and_workshop_values"];
    const Vector3d p_tip(2.0, -1.0, 95.0), p_piv(30.0, 40.0, 500.0);
    const auto Rs = as_list<Matrix3d>(c["Rs"]);
    std::vector<Vector3d> ps;
    for (const Matrix3d& R : Rs) ps.push_back(p_piv - R * p_tip);
    const lab::PivotResult r = lab::pivot_calibration(Rs, ps);
    CHECK_ALLCLOSE(r.p_tip, p_tip, 1e-9);
    CHECK_ALLCLOSE(r.p_piv, p_piv, 1e-9);
    CHECK(r.rms < 1e-9);
    // Workshop Problem 2 (three poses: I, Ry(90), Rx(-90))
    Matrix3d Ry, Rx;
    Ry << 0, 0, 1, 0, 1, 0, -1, 0, 0;
    Rx << 1, 0, 0, 0, 0, 1, 0, -1, 0;
    const std::vector<Vector3d> pw = {Vector3d(100, -52, 1050), Vector3d(-50, -52, 1200), Vector3d(100, -200, 1202)};
    const lab::PivotResult w = lab::pivot_calibration({Matrix3d::Identity(), Ry, Rx}, pw);
    CHECK_ALLCLOSE(w.p_tip, Vector3d(0, 2, 150), 1e-9);
    CHECK_ALLCLOSE(w.p_piv, Vector3d(100, -50, 1200), 1e-9);
    // C++ track: unequal list lengths are rejected
    CHECK_THROWS_AS(lab::pivot_calibration({Matrix3d::Identity(), Ry}, pw), std::invalid_argument);
}

TEST_CASE("test_A2_pivot_calibration_least_squares_with_noise") {
    const json& c = cases()["A2_pivot_calibration_least_squares_with_noise"];
    const auto Rs = as_list<Matrix3d>(c["Rs"]);
    const auto ps = as_list<Vector3d>(c["ps"]);  // p_piv - R p_tip + noise(0.3 mm), drawn by test_lab.py
    const lab::PivotResult r = lab::pivot_calibration(Rs, ps);
    // independent: eliminate p_piv with differences to the mean pose, then solve for p_tip
    const std::size_t K = Rs.size();
    Matrix3d Rm = Matrix3d::Zero();
    Vector3d pm = Vector3d::Zero();
    for (std::size_t i = 0; i < K; ++i) {
        Rm += Rs[i] / static_cast<double>(K);
        pm += ps[i] / static_cast<double>(K);
    }
    MatrixXd A(3 * static_cast<Eigen::Index>(K), 3);
    VectorXd y(3 * static_cast<Eigen::Index>(K));
    for (std::size_t i = 0; i < K; ++i) {
        A.block<3, 3>(3 * static_cast<Eigen::Index>(i), 0) = Rs[i] - Rm;
        y.segment<3>(3 * static_cast<Eigen::Index>(i)) = pm - ps[i];
    }
    const Vector3d tip = lstsq(A, y);
    CHECK_ALLCLOSE(r.p_tip, tip, 1e-8);
    CHECK_ALLCLOSE(r.p_piv, Rm * tip + pm, 1e-8);
    double ss = 0.0;
    for (std::size_t i = 0; i < K; ++i) ss += (Rs[i] * tip + ps[i] - (Rm * tip + pm)).squaredNorm();
    CHECK_SCALAR_NEAR(r.rms, std::sqrt(ss / static_cast<double>(K)), 1e-9);
    CHECK_SCALAR_NEAR(r.rms, c["rms"].get<double>(), 1e-9);
}

TEST_CASE("test_A3_pivot_from_joint_log") {
    const MatrixXd Q = lab::load_joints("pivot_joints.csv").second;
    const lab::PivotResult r = lab::pivot_from_joint_log(Q);
    MatrixXd A(3 * Q.rows(), 6);
    VectorXd b(3 * Q.rows());
    for (Eigen::Index i = 0; i < Q.rows(); ++i) {
        const Matrix4d T = dh_fk_mm(Q.row(i).transpose());
        A.block<3, 3>(3 * i, 0) = T.topLeftCorner<3, 3>();
        A.block<3, 3>(3 * i, 3) = -Matrix3d::Identity();
        b.segment<3>(3 * i) = -T.block<3, 1>(0, 3);
    }
    const VectorXd x = lstsq(A, b);
    CHECK_ALLCLOSE(r.p_tip, x.head<3>(), 1e-8);
    CHECK_ALLCLOSE(r.p_piv, x.tail<3>(), 1e-8);
    CHECK(std::abs(r.p_tip.norm() - PH().pointer_nominal_length_mm) < 2.0);
    CHECK(r.rms > 0.0);
    CHECK(r.rms < 0.5);
    const json& c = cases()["A3_pivot_from_joint_log"];
    CHECK_ALLCLOSE(r.p_tip, as<Vector3d>(c["p_tip"]), 1e-8);
    CHECK_ALLCLOSE(r.p_piv, as<Vector3d>(c["p_piv"]), 1e-8);
}

TEST_CASE("test_A4_tip_position") {
    const json& c = cases()["A4_tip_position"];
    const VectorXd q = as_vec(c["q"]);
    const Vector3d p_tip(1.0, -2.0, 70.0);
    const Matrix4d T = dh_fk_mm(q);
    CHECK_ALLCLOSE(lab::tip_position_mm(q, p_tip), rot_of(T) * p_tip + pos_of(T), 1e-8);
    CHECK_ALLCLOSE(lab::tip_position_mm(q, p_tip), as<Vector3d>(c["p"]), 1e-8);
}

// ============================================================================ Part B
TEST_CASE("test_B1_register_points_recovers_exact_transform") {
    const json& c = cases()["B1_register_points_recovers_exact_transform"];
    const Matrix3d R = as<Matrix3d>(c["R"]);
    const Vector3d t = as<Vector3d>(c["t"]);
    const MatrixXd X = as_mat(c["X"]);
    const Matrix4d T = lab::register_points(X, move_rows(X, R, t));
    CHECK_ALLCLOSE(rot_of(T), R, 1e-9);
    CHECK_ALLCLOSE(pos_of(T), t, 1e-8);
    CHECK_ALLCLOSE(T.row(3), Eigen::RowVector4d(0, 0, 0, 1), 1e-8);
    // C++ track: wrong shapes are rejected
    CHECK_THROWS_AS(lab::register_points(X.topRows(2), move_rows(X, R, t).topRows(2)), std::invalid_argument);
}

TEST_CASE("test_B1_register_points_matches_horn_with_noise") {
    for (const json& c : cases()["B1_register_points_matches_horn_with_noise"]) {
        const MatrixXd X = as_mat(c["X"]), Y = as_mat(c["Y"]);
        INFO("N = " << X.rows());
        const Matrix4d T = lab::register_points(X, Y);
        const auto [Rh, th] = horn(X, Y);
        CHECK_ALLCLOSE(rot_of(T), Rh, 1e-7);
        CHECK_ALLCLOSE(pos_of(T), th, 1e-6);
        CHECK_ALLCLOSE(T, as<Matrix4d>(c["T"]), 1e-7);
    }
}

TEST_CASE("test_B1_no_reflection_for_coplanar_points") {
    // Coplanar fiducials: the plain SVD formula V U^T may be a reflection; the fix must return det R = +1.
    const json& c = cases()["B1_no_reflection_for_coplanar_points"];
    MatrixXd X(4, 3);
    X << 0, 0, 0, 60, 0, 0, 0, 40, 0, 60, 40, 0;
    const MatrixXd Y = as_mat(c["Y"]);  // mirror image (z -> -z) plus noise 0.01 mm, drawn by test_lab.py
    const Matrix4d T = lab::register_points(X, Y);
    const Matrix3d R = T.topLeftCorner<3, 3>();
    CHECK_SCALAR_NEAR(R.determinant(), 1.0, 1e-9);
    CHECK_ALLCLOSE(R * R.transpose(), Matrix3d::Identity(), 1e-9);
}

TEST_CASE("test_B1_workshop_problem3_values") {
    MatrixXd X(3, 3), Y(3, 3);
    X << 0, 0, 0, 60, 0, 0, 0, 90, 0;
    Y << 100, 50, 20, 136, 98, 20, 28, 104, 20;
    const Matrix4d T = lab::register_points(X, Y);
    Matrix3d R;
    R << 0.6, -0.8, 0, 0.8, 0.6, 0, 0, 0, 1;
    CHECK_ALLCLOSE(rot_of(T), R, 1e-12);
    CHECK_ALLCLOSE(pos_of(T), Vector3d(100, 50, 20), 1e-9);
}

TEST_CASE("test_B2_apply_T_shapes") {
    const json& c = cases()["B2_apply_T_shapes"];
    const Matrix4d T = as<Matrix4d>(c["T"]);
    const Vector3d p = as<Vector3d>(c["p"]);
    const MatrixXd P = as_mat(c["P"]);
    const Vector3d Tp = lab::apply_T(T, p);  // shape (3,) by type
    const MatrixXd TP = lab::apply_T(T, P);
    REQUIRE(TP.rows() == 4);
    REQUIRE(TP.cols() == 3);
    CHECK_ALLCLOSE(TP.row(2), hom_apply(T, P.row(2).transpose()), 1e-8);
    CHECK_ALLCLOSE(Tp, hom_apply(T, p), 1e-8);
    CHECK_ALLCLOSE(TP, as_mat(c["TP"]), 1e-12);
    // C++ track: P must be N x 3
    CHECK_THROWS_AS(lab::apply_T(T, MatrixXd(P.transpose())), std::invalid_argument);
}

TEST_CASE("test_B3_fre") {
    MatrixXd X(3, 3), E(3, 3);
    X << 0, 0, 0, 10, 0, 0, 0, 10, 0;
    E << 0.3, 0, 0, 0, 0.4, 0, 0, 0, 0;
    const MatrixXd Y = X + E;
    CHECK_SCALAR_NEAR(lab::fre(X, Y, Matrix4d::Identity()), std::sqrt((0.09 + 0.16) / 3), 1e-12);
}

// ============================================================================ Part C
TEST_CASE("test_C1_C2_fre_formulae") {
    CHECK_SCALAR_NEAR(lab::fre_expected(4, 0.4), 0.4 * std::sqrt(0.5), 1e-12);
    CHECK_SCALAR_NEAR(lab::fre_expected(3, 1.0), 1 / std::sqrt(3.0), 1e-12);
    CHECK_SCALAR_NEAR(lab::fle_from_fre(lab::fre_expected(6, 0.7), 6), 0.7, 1e-12);
}

TEST_CASE("test_C3_tre_predict_matches_eigen_form_and_workshop") {
    MatrixXd X(4, 3);
    X << 60, 0, 0, -60, 0, 0, 0, 40, 0, 0, -40, 0;
    CHECK_SCALAR_NEAR(lab::tre_predict(X, Vector3d(0, 0, -80), 0.4),
                      std::sqrt(0.04 * (1 + (8 + 6400.0 / 1800) / 3)), 1e-12);
    CHECK_SCALAR_NEAR(lab::tre_predict(X, Vector3d(0, 0, 0), 0.4), 0.2, 1e-12);
    const json& c = cases()["C3_tre_predict_matches_eigen_form_and_workshop"];
    for (const json& rc : c["random"]) {
        const MatrixXd Xr = as_mat(rc["X"]);
        const Vector3d r = as<Vector3d>(rc["r"]);
        const double ref = tre_eig(Xr, r, 0.6);
        CHECK_SCALAR_NEAR(lab::tre_predict(Xr, r, 0.6), ref, rel_tol(ref, 1e-10));
        CHECK_SCALAR_NEAR(lab::tre_predict(Xr, r, 0.6), rc["tre"].get<double>(), rel_tol(ref, 1e-10));
    }
    // invariant under a rigid motion of fiducials and target together
    const Matrix3d R = as<Matrix3d>(c["R"]);
    const Vector3d t = as<Vector3d>(c["t"]);
    const Vector3d r(10.0, 20.0, -30.0);
    const double ref = lab::tre_predict(X, r, 0.5);
    CHECK_SCALAR_NEAR(lab::tre_predict(move_rows(X, R, t), R * r + t, 0.5), ref, rel_tol(ref, 1e-10));
}

TEST_CASE("test_C4_monte_carlo_agrees_with_prediction") {
    // Statistical test: the C++ generator lab::Rng differs from NumPy's, so the samples differ from Python's,
    // but the RMS values must agree with the prediction to the same tolerances.
    MatrixXd X(4, 3), tg(2, 3);
    X << 70, 0, 0, -70, 0, 0, 0, 60, 40, 0, -60, 40;
    tg << 10, 20, -30, 0, 7.5, 15;
    const lab::MonteCarloResult mc = lab::monte_carlo(X, tg, 0.5, 4000, 1);
    const VectorXd& f = mc.fre_samples;
    const MatrixXd& e = mc.tre_samples;
    REQUIRE(f.size() == 4000);
    REQUIRE(e.rows() == 4000);
    REQUIRE(e.cols() == 2);
    const double fre_ref = lab::fre_expected(4, 0.5);
    CHECK_SCALAR_NEAR(rms(f), fre_ref, rel_tol(fre_ref, 0.04));
    for (Eigen::Index j = 0; j < 2; ++j) {
        const double ref = tre_eig(X, tg.row(j).transpose(), 0.5);
        INFO("target " << j);
        CHECK_SCALAR_NEAR(rms(e.col(j)), ref, rel_tol(ref, 0.05));
    }
    const lab::MonteCarloResult mc2 = lab::monte_carlo(X, tg, 0.5, 50, 1);  // same seed, same first samples
    REQUIRE(mc2.fre_samples.size() == 50);
    CHECK_ALLCLOSE(mc2.fre_samples, f.head(50), 1e-8);
    CHECK_ALLCLOSE(mc2.tre_samples, e.topRows(50), 1e-8);
    // FRE says (almost) nothing about TRE: |corrcoef(f, e[:, 0])| < 0.1
    const VectorXd fc = f.array() - f.mean();
    const VectorXd ec = e.col(0).array() - e.col(0).mean();
    const double corr = fc.dot(ec) / (fc.norm() * ec.norm());
    CHECK(std::abs(corr) < 0.1);
}

TEST_CASE("test_C5_best_subset_brute_force") {
    const Vector3d target = PH().targets_image_mm.at("T2_lateral");
    const MatrixXd& X_FID = PH().fiducials_image_mm;
    const json& stored = cases()["C5_best_subset_brute_force"];
    std::size_t s = 0;
    for (int k : {3, 4, 6}) {
        INFO("k = " << k);
        const auto [idx, val] = lab::best_subset(X_FID, target, k, 0.4);
        std::vector<std::vector<int>> all;
        std::vector<int> cur;
        combos(6, k, 0, cur, all);
        std::vector<int> best;
        double best_val = 0.0;
        for (const auto& cmb : all) {  // min(vals, key=vals.get): the first minimum in combinations order
            const double v = tre_eig(rows_of(X_FID, cmb), target, 0.4);
            if (best.empty() || v < best_val) {
                best = cmb;
                best_val = v;
            }
        }
        CHECK(idx == best);
        CHECK_SCALAR_NEAR(val, best_val, rel_tol(best_val, 1e-10));
        CHECK(idx == stored[s]["indices"].get<std::vector<int>>());
        ++s;
    }
}

// ============================================================================ Part D
TEST_CASE("test_D1_chain_robot_from_image") {
    const json& c = cases()["D1_chain_robot_from_image"];
    const Matrix4d T_CR = as<Matrix4d>(c["T_CR"]), T_CP = as<Matrix4d>(c["T_CP"]), T_PI = as<Matrix4d>(c["T_PI"]);
    const Matrix4d T = lab::chain_robot_from_image(T_CR, T_CP, T_PI);
    const Vector3d p_I = as<Vector3d>(c["p_I"]);
    const Vector3d p_C = hom_apply(T_CP * T_PI, p_I);
    CHECK_ALLCLOSE(hom_apply(T, p_I), hom_apply(mr::trans_inv(T_CR), p_C), 1e-8);
}

TEST_CASE("test_D2_navigation_pipeline") {
    const lab::PipelineResult out = lab::navigation_pipeline(1500, 0);
    const Vector3d a = lab::pivot_from_joint_log(lab::load_joints("pivot_joints.csv").second).p_tip;
    CHECK_ALLCLOSE(out.p_tip, a, 1e-9);
    const MatrixXd Qf = lab::load_joints("fiducial_joints.csv").second;
    MatrixXd Y(Qf.rows(), 3);
    for (Eigen::Index i = 0; i < Qf.rows(); ++i) {
        const Matrix4d T = dh_fk_mm(Qf.row(i).transpose());
        Y.row(i) = (T.topLeftCorner<3, 3>() * a + T.block<3, 1>(0, 3)).transpose();
    }
    CHECK_ALLCLOSE(out.Y_fid, Y, 1e-8);
    const MatrixXd& X_FID = PH().fiducials_image_mm;
    const auto [Rh, th] = horn(X_FID, Y);
    CHECK_ALLCLOSE(rot_of(out.T_sI), Rh, 1e-7);
    CHECK_ALLCLOSE(pos_of(out.T_sI), th, 1e-5);
    const double fre_ref = std::sqrt((Y - move_rows(X_FID, Rh, th)).rowwise().squaredNorm().mean());
    CHECK_SCALAR_NEAR(out.fre, fre_ref, 1e-6);
    CHECK_SCALAR_NEAR(out.fle_est, out.fre / std::sqrt(1 - 2.0 / 6), 1e-9);
    for (const auto& kv : PH().targets_image_mm) {
        const std::string& k = kv.first;
        const Vector3d& r = kv.second;
        INFO("target " << k);
        const double pred_ref = tre_eig(X_FID, r, out.fle_est);
        CHECK_SCALAR_NEAR(out.tre_pred.at(k), pred_ref, rel_tol(pred_ref, 1e-9));
        CHECK_SCALAR_NEAR(out.tre_mc.at(k), out.tre_pred.at(k), rel_tol(out.tre_pred.at(k), 0.08));
        CHECK_ALLCLOSE(out.targets_s.at(k), Rh * r + th, 1e-5);
    }
    REQUIRE(out.check_err.size() == 2);
    CHECK((out.check_err.array() < 1.5).all());
    // the same numbers as the Python reference
    const json& c = cases()["D2_navigation_pipeline"];
    CHECK_ALLCLOSE(out.T_sI, as<Matrix4d>(c["T_sI"]), 1e-7);
    CHECK_ALLCLOSE(out.check_err, as_vec(c["check_err"]), 1e-7);
    CHECK_SCALAR_NEAR(out.pivot_rms, c["pivot_rms"].get<double>(), 1e-9);

    // ---- plot data for E7, E8, E10 (python cpp/plot_results.py); written only when the sizes are right
    if (out.Y_fid.rows() == X_FID.rows() && out.Y_fid.cols() == 3 && out.targets_s.size() == 3) {
        MatrixXd tg(3, 3);
        Eigen::Index j = 0;
        for (const auto& kv : PH().targets_image_mm) tg.row(j++) = kv.second.transpose();
        const lab::MonteCarloResult mc = lab::monte_carlo(X_FID, tg, out.fle_est, 5000, 0);  // E7
        if (mc.fre_samples.size() == 5000 && mc.tre_samples.rows() == 5000 && mc.tre_samples.cols() == 3) {
            MatrixXd M(5000, 4);
            M << mc.fre_samples, mc.tre_samples;
            sbeg::write_csv("lab5_cpp_mc.csv", {"fre", "tre_T1_central", "tre_T2_lateral", "tre_T3_deep"}, M);
        }
        MatrixXd G(121 * 101, 3);  // E8: TRE map in the plane z_I = 10 mm
        Eigen::Index row = 0;
        for (int iy = 0; iy <= 100; ++iy)
            for (int ix = 0; ix <= 120; ++ix) {
                const Vector3d r(-60.0 + 2.0 * ix, -60.0 + 2.0 * iy, 10.0);
                G.row(row++) << r(0), r(1), lab::tre_predict(X_FID, r, out.fle_est);
            }
        sbeg::write_csv("lab5_cpp_tre_map.csv", {"x_I", "y_I", "tre"}, G);
        const MatrixXd Xs = lab::apply_T(out.T_sI, X_FID);  // E10: kind 0 digitised, 1 registered CT, 2 target
        MatrixXd Pts(out.Y_fid.rows() + Xs.rows() + 3, 4);
        row = 0;
        for (Eigen::Index i = 0; i < out.Y_fid.rows(); ++i) Pts.row(row++) << 0.0, out.Y_fid.row(i);
        for (Eigen::Index i = 0; i < Xs.rows(); ++i) Pts.row(row++) << 1.0, Xs.row(i);
        for (const auto& kv : out.targets_s) Pts.row(row++) << 2.0, kv.second.transpose();
        sbeg::write_csv("lab5_cpp_points.csv", {"kind", "x_s", "y_s", "z_s"}, Pts);
    }
}
