// Auto-checks for SBEG 361 Lab 3 (exponential coordinates and logarithms on SO(3) and SE(3), quaternions, IMU
// orientation, tracked screw insertion) -- C++ track.
// Mirrors test_lab.py test by test (30 test cases, same names, same order, same tolerances).
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*vec_to_so3*,*skew*,*axis_ang3*"      (run only some tests, like pytest -k)
// Results are compared with mrcpp (the C++ port of modern_robotics), with Eigen's own quaternion class (as the
// Python tests use scipy.spatial.transform) and with cpp_cases.json, which holds the numbers of the Python
// reference, the noise-free data model and the random inputs drawn by test_lab.py, so both languages are checked
// against the same values.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;
using sbeg::as;
using sbeg::json;
using sbeg::Vector6d;

namespace {
constexpr double PI = 3.14159265358979323846;
constexpr double DT = 0.01;
const double S2 = 1.0 / std::sqrt(2.0);

const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

/// The IMU data file imu_knee.csv (same file as the Python tests), loaded once.
const sbeg::CsvTable& imu() {
    static const sbeg::CsvTable T = sbeg::read_csv(sbeg::lab_path("../imu_knee.csv"));
    return T;
}

/// Path of the tracker log screw_insertion.csv (same file as the Python tests).
std::string track_file() { return sbeg::lab_path("../screw_insertion.csv"); }

/// Reference rotation from mrcpp: exp([axis] ang).
Matrix3d rot(const Vector3d& axis, double ang) { return mr::matrix_exp3(mr::vec_to_so3(axis * ang)); }

Vector6d v6(double a, double b, double c, double d, double e, double f) {
    Vector6d V;
    V << a, b, c, d, e, f;
    return V;
}

Matrix4d mr_rp(const Matrix3d& R, const Vector3d& p) { return mr::rp_to_trans(R, p); }

/// np.allclose(a, b, atol=atol) with NumPy's default rtol = 1e-5: every entry |a - b| <= atol + rtol * |b|.
bool allclose(const Vector4d& a, const Vector4d& b, double atol) {
    return a.allFinite() && ((a - b).cwiseAbs() - 1e-5 * b.cwiseAbs()).maxCoeff() <= atol;
}

/// q and -q represent the same rotation (MR App. B.3). Same test as Python's same_quat (np.allclose).
bool same_quat(const Vector4d& q, const Vector4d& p, double tol = 1e-9) {
    return allclose(q, p, tol) || allclose(q, Vector4d(-p), tol);
}

/// Quaternion columns <prefix>_q0..q3 as an N x 4 matrix (row k = sample k).
MatrixXd quats(const std::string& prefix) {
    const sbeg::CsvTable& T = imu();
    MatrixXd Q(T.data.rows(), 4);
    for (int i = 0; i < 4; ++i) Q.col(i) = T.data.col(T.col(prefix + "_q" + std::to_string(i)));
    return Q;
}

/// Gyro columns <prefix>_wx..wz (or _nwx..nwz) without the last (padding) row, as in Python's [:-1].
MatrixXd gyro(const std::string& prefix, bool noisy = false) {
    const sbeg::CsvTable& T = imu();
    const std::string n = noisy ? "n" : "";
    MatrixXd W(T.data.rows() - 1, 3);
    const char* comps[3] = {"x", "y", "z"};
    for (int c = 0; c < 3; ++c) W.col(c) = T.data.col(T.col(prefix + "_" + n + "w" + comps[c])).head(W.rows());
    return W;
}

/// np.testing.assert_allclose(got, expected, atol=atol) with NumPy's default rtol = 1e-7:
/// every entry must satisfy |got - expected| <= atol + rtol * |expected|.
#define CHECK_ALLCLOSE(got, expected, atol)                                                                  \
    do {                                                                                                     \
        const Matrix3d g_ = (got), e_ = (expected);                                                          \
        const bool ok_ = g_.allFinite() && ((g_ - e_).cwiseAbs() - 1e-7 * e_.cwiseAbs()).maxCoeff() <= (atol); \
        CHECK_MESSAGE(ok_, "CHECK_ALLCLOSE(" #got ", " #expected ") (rtol 1e-7, atol " << (atol) \
                                             << ")\n  " << ::sbeg::compare_near(g_, e_, (atol)).msg);          \
    } while (false)

Vector4d qrow(const MatrixXd& Q, Eigen::Index k) { return Q.row(k).transpose(); }

/// Ground-truth rotation of a scalar-first quaternion with Eigen's quaternion class (independent of the lab code;
/// the Python test uses scipy Rotation.from_quat).
Matrix3d truth_rot(const Vector4d& q) { return Eigen::Quaterniond(q(0), q(1), q(2), q(3)).normalized().toRotationMatrix(); }

/// Least-squares slope of y against x (np.polyfit(x, y, 1)[0]).
double fit_slope(const VectorXd& x, const VectorXd& y) {
    const double mx = x.mean(), my = y.mean();
    return ((x.array() - mx) * (y.array() - my)).sum() / ((x.array() - mx).square().sum());
}
}  // namespace

// ------------------------------------------------------------- Checkpoint 1: skew matrices
TEST_CASE("test_vec_to_so3_matches_mr") {
    for (const auto& c : cases()["vec_to_so3_matches_mr"]) {
        const Vector3d w = as<Vector3d>(c["w"]);
        CHECK_MAT_NEAR(lab::vec_to_so3(w), mr::vec_to_so3(w), 1e-12);
        CHECK_MAT_NEAR(lab::vec_to_so3(w), as<Matrix3d>(c["so3"]), 1e-12);
    }
}

TEST_CASE("test_skew_properties") {
    const json& c = cases()["skew_properties"];
    const Vector3d a = as<Vector3d>(c["a"]), b = as<Vector3d>(c["b"]);
    const Matrix3d A = lab::vec_to_so3(a);
    CHECK_MAT_NEAR(A, Matrix3d(-A.transpose()), 1e-12);                            // [a] = -[a]^T
    CHECK_MAT_NEAR(Vector3d(A * b), a.cross(b), 1e-12);                           // [a]b = a x b
    const Matrix3d R = as<Matrix3d>(c["R"]);
    CHECK_MAT_NEAR(Matrix3d(R * A * R.transpose()), lab::vec_to_so3(R * a), 1e-12);  // MR Eq. 3.31
    CHECK_MAT_NEAR(lab::so3_to_vec(A), a, 1e-12);
}

TEST_CASE("test_axis_ang3") {
    const Vector3d expc3(1.0, 2.0, 2.0);
    const auto [w, th] = lab::axis_ang3(expc3);
    const auto [ref_w, ref_th] = mr::axis_ang3(expc3);
    CHECK_MAT_NEAR(w, ref_w, 1e-12);
    CHECK_APPROX(th, ref_th);
    CHECK_THROWS_AS(lab::axis_ang3(Vector3d::Zero()), std::invalid_argument);
}

// ------------------------------------------------------------- Checkpoint 2: exponentials, SO(3) and SE(3)
TEST_CASE("test_matrix_exp3_mr_example_3_12") {
    // MR Example 3.12: w1 = (0, 0.866, 0.5), theta1 = 30 deg.
    const Vector3d w(0.0, std::sqrt(3.0) / 2, 0.5);
    const Matrix3d R = lab::matrix_exp3(lab::vec_to_so3(w * PI / 6));
    Matrix3d expected;
    expected << 0.866, -0.250, 0.433, 0.250, 0.967, 0.058, -0.433, 0.058, 0.899;
    CHECK_MAT_NEAR(R, expected, 1.5e-3);
}

TEST_CASE("test_matrix_exp3_matches_mr_and_is_rotation") {
    for (const auto& c : cases()["matrix_exp3_matches_mr_and_is_rotation"]) {
        const Matrix3d so3 = mr::vec_to_so3(as<Vector3d>(c["omega"]));
        const Matrix3d R = lab::matrix_exp3(so3);
        CHECK_MAT_NEAR(R, mr::matrix_exp3(so3), 1e-12);
        CHECK_MAT_NEAR(R, as<Matrix3d>(c["R"]), 1e-12);
        CHECK_MAT_NEAR(Matrix3d(R.transpose() * R), Matrix3d::Identity(), 1e-12);
        CHECK_APPROX(R.determinant(), 1.0);
    }
    CHECK_ALLCLOSE(lab::matrix_exp3(Matrix3d::Zero()), Matrix3d::Identity(), 0.0);  // Python default: atol 0
}

TEST_CASE("test_g_matrix_and_inverse") {
    // G(theta) of MR Eq. 3.87 = its power series (Eq. 3.86); G^-1 of Eq. 3.92 is its inverse;
    // G(theta) v is the translation of e^{[S]theta} (Eq. 3.88).
    for (const auto& c : cases()["g_matrix_and_inverse"]) {
        const Vector3d w = as<Vector3d>(c["w"]), v = as<Vector3d>(c["v"]);
        const double th = c["theta"].get<double>();
        INFO("theta = " << th);
        const Matrix3d G = lab::g_matrix(w, th);
        const Matrix3d W = mr::vec_to_so3(w);
        Matrix3d series = Matrix3d::Zero(), Wk = Matrix3d::Identity();
        double coef = th;  // theta^(k+1) / (k+1)!
        for (int k = 0; k < 40; ++k) {
            series += Wk * coef;
            Wk = (Wk * W).eval();
            coef *= th / (k + 2);
        }
        CHECK_MAT_NEAR(G, series, 1e-10);
        CHECK_MAT_NEAR(Matrix3d(G * lab::g_inv_matrix(w, th)), Matrix3d::Identity(), 1e-10);
        Vector6d S;
        S << w, v;
        const Matrix4d T = mr::matrix_exp6(mr::vec_to_se3(S) * th);
        CHECK_MAT_NEAR(Vector3d(G * v), Vector3d(T.topRightCorner<3, 1>()), 1e-10);
        CHECK_MAT_NEAR(G, as<Matrix3d>(c["G"]), 1e-10);
    }
}

TEST_CASE("test_matrix_exp6_matches_mr") {
    for (const auto& c : cases()["matrix_exp6_matches_mr"]) {
        const Matrix4d se3 = as<Matrix4d>(c["se3"]);
        CHECK_MAT_NEAR(lab::matrix_exp6(se3), mr::matrix_exp6(se3), 1e-9);
        CHECK_MAT_NEAR(lab::matrix_exp6(se3), as<Matrix4d>(c["T"]), 1e-9);
    }
    // pure translation, MR Eq. 3.89
    const Matrix4d se3 = mr::vec_to_se3(v6(0, 0, 0, 0.03, 0.04, 0));
    CHECK_MAT_NEAR(lab::matrix_exp6(se3), mr_rp(Matrix3d::Identity(), Vector3d(0.03, 0.04, 0)), 1e-12);
}

TEST_CASE("test_exp6_screw_through_a_point") {
    // Axis z through q = (1, 0, 0), theta = 90 deg, pitch 0 and 0.1 m/rad.
    for (const double h : {0.0, 0.1}) {
        INFO("pitch h = " << h);
        const double pz = h * PI / 2;
        const Vector3d w(0, 0, 1);
        Vector6d S;
        S << w, -w.cross(Vector3d(1, 0, 0)) + h * w;
        const Matrix4d T = lab::matrix_exp6(mr::vec_to_se3(S) * (PI / 2));
        Matrix4d T_exp;
        T_exp << 0, -1, 0, 1, 1, 0, 0, -1, 0, 0, 1, pz, 0, 0, 0, 1;
        CHECK_MAT_NEAR(T, T_exp, 1e-12);
    }
}

// ------------------------------------------------------------- Checkpoint 3: logarithms, SO(3) and SE(3)
TEST_CASE("test_matrix_log3_matches_mr_random") {
    for (const auto& c : cases()["matrix_log3_matches_mr_random"]) {
        const Matrix3d R = as<Matrix3d>(c["R"]);
        CHECK_MAT_NEAR(lab::matrix_log3(R), mr::matrix_log3(R), 1e-9);
        CHECK_MAT_NEAR(lab::matrix_log3(R), as<Matrix3d>(c["log"]), 1e-9);
    }
}

TEST_CASE("test_matrix_log3_identity_and_pi_cases") {
    CHECK_ALLCLOSE(lab::matrix_log3(Matrix3d::Identity()), Matrix3d::Zero(), 0.0);  // Python default: exact zero
    for (const auto& c : cases()["matrix_log3_identity_and_pi_cases"]) {
        const Vector3d w = as<Vector3d>(c["w"]);
        INFO("axis w = " << w.transpose());
        const Matrix3d R = rot(w, PI);
        CHECK_MAT_NEAR(R, as<Matrix3d>(c["R"]), 1e-12);
        const Matrix3d L = lab::matrix_log3(R);
        const Vector3d v = lab::so3_to_vec(L);
        CHECK_APPROX(v.norm(), PI);
        CHECK_MAT_NEAR(mr::matrix_exp3(L), R, 1e-9);  // either +w or -w is fine
    }
}

TEST_CASE("test_exp_log_round_trip_small_and_large_angles") {
    for (const auto& c : cases()["exp_log_round_trip_small_and_large_angles"]) {
        const double th = c["theta"].get<double>();
        const Vector3d w = as<Vector3d>(c["w"]);
        INFO("theta = " << th);
        const Matrix3d R = mr::matrix_exp3(mr::vec_to_so3(w * th));
        const Vector3d v = lab::so3_to_vec(lab::matrix_log3(R));
        CHECK_MAT_NEAR(v, Vector3d(w * th), 1e-6);
    }
}

TEST_CASE("test_matrix_log6_matches_mr_and_inverts_exp") {
    for (const auto& c : cases()["matrix_log6_matches_mr_and_inverts_exp"]) {
        const Matrix4d T = as<Matrix4d>(c["T"]);
        const Matrix4d L = lab::matrix_log6(T);
        CHECK_MAT_NEAR(L.row(3), Eigen::RowVector4d::Zero(), 1e-8);  // np.allclose(L[3], 0)
        CHECK_MAT_NEAR(L, mr::matrix_log6(T), 1e-8);
        CHECK_MAT_NEAR(L, as<Matrix4d>(c["L"]), 1e-8);
        CHECK_MAT_NEAR(mr::matrix_exp6(L), T, 1e-9);
    }
}

TEST_CASE("test_matrix_log6_special_cases") {
    CHECK_MAT_NEAR(lab::matrix_log6(Matrix4d::Identity()), Matrix4d::Zero(), 1e-12);
    const Matrix4d T = mr_rp(Matrix3d::Identity(), Vector3d(0.4, 0, 0.3));  // case (a): R = I
    CHECK_MAT_NEAR(lab::se3_to_vec(lab::matrix_log6(T)), v6(0, 0, 0, 0.4, 0, 0.3), 1e-12);
    const Vector3d diags[3] = {Vector3d(1, -1, -1), Vector3d(-1, 1, -1), Vector3d(-1, -1, 1)};  // theta = pi
    for (const Vector3d& dg : diags) {
        INFO("R = diag(" << dg.transpose() << ")");
        const Matrix3d R = dg.asDiagonal();
        const Matrix4d Tp = mr_rp(R, Vector3d(0.1, 0.2, -0.3));
        const Matrix4d L = lab::matrix_log6(Tp);
        const Matrix3d so3 = L.topLeftCorner<3, 3>();
        CHECK_SCALAR_NEAR(lab::so3_to_vec(so3).norm(), PI, 1e-9);
        CHECK_MAT_NEAR(mr::matrix_exp6(L), Tp, 1e-9);
    }
}

TEST_CASE("test_log6_mr_example_3_26") {
    // MR Example 3.26 (pp. 104-106): S = (0, 0, 1, 3.37, -3.37, 0), theta = pi/6.
    const double c30 = std::cos(PI / 6), s30 = 0.5, c60 = 0.5, s60 = std::sin(PI / 3);
    Matrix4d Tsb, Tsc;
    Tsb << c30, -s30, 0, 1, s30, c30, 0, 2, 0, 0, 1, 0, 0, 0, 0, 1;
    Tsc << c60, -s60, 0, 2, s60, c60, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1;
    const Matrix4d D = Tsc * mr::trans_inv(Tsb);
    const Matrix4d L = lab::matrix_log6(D);
    CHECK_MAT_NEAR(L, mr::matrix_log6(D), 1e-9);
    const Vector6d St = lab::se3_to_vec(L);
    CHECK_SCALAR_NEAR(St.head<3>().norm(), PI / 6, 1e-9);
    CHECK_MAT_NEAR(Vector6d(St / (PI / 6)), v6(0, 0, 1, 3.3660, -3.3660, 0), 1e-4);
    CHECK_MAT_NEAR(L, as<Matrix4d>(cases()["log6_mr_example_3_26"]["L"]), 1e-9);
}

// ------------------------------------------------------------- Checkpoint 4: screw motions
TEST_CASE("test_screw_between_space_and_body") {
    const json& C = cases()["screw_between_space_and_body"];
    for (const auto& c : C["cases"]) {
        const Matrix4d T1 = as<Matrix4d>(c["T1"]), T2 = as<Matrix4d>(c["T2"]);
        const auto [Ss, ts] = lab::screw_between(T1, T2, "space");
        const auto [Sb, tb] = lab::screw_between(T1, T2, "body");
        CHECK_SCALAR_NEAR(Ss.head<3>().norm(), 1.0, 1e-9);
        CHECK_SCALAR_NEAR(ts, tb, 1e-9);
        CHECK_MAT_NEAR(Matrix4d(mr::matrix_exp6(mr::vec_to_se3(Ss) * ts) * T1), T2, 1e-9);
        CHECK_MAT_NEAR(Matrix4d(T1 * mr::matrix_exp6(mr::vec_to_se3(Sb) * tb)), T2, 1e-9);
        CHECK_MAT_NEAR(Sb, Vector6d(mr::adjoint(mr::trans_inv(T1)) * Ss), 1e-9);  // S_b = [Ad_{T1^-1}] S_s
        CHECK_MAT_NEAR(Ss, as<Vector6d>(c["S_s"]), 1e-9);
        CHECK_SCALAR_NEAR(ts, c["theta_s"].get<double>(), 1e-9);
    }
    const auto [S, th] = lab::screw_between(Matrix4d::Identity(), mr_rp(Matrix3d::Identity(), Vector3d(0, 0.3, 0.4)));
    CHECK_MAT_NEAR(S, v6(0, 0, 0, 0, 0.6, 0.8), 1e-12);
    CHECK_SCALAR_NEAR(th, 0.5, 1e-12);
    const Matrix4d T_same = as<Matrix4d>(C["T_same"]);
    const auto [S0, th0] = lab::screw_between(T_same, T_same);
    CHECK_MAT_NEAR(S0, Vector6d::Zero(), 1e-12);
    CHECK(th0 == 0.0);
    CHECK_THROWS_AS(lab::screw_between(Matrix4d::Identity(), Matrix4d::Identity(), "world"), std::invalid_argument);
}

TEST_CASE("test_screw_path_needle_guide") {
    // Needle guide: half-way along the screw from T1 to T2.
    const Matrix4d T1 = mr_rp(Matrix3d::Identity(), Vector3d(0.3, 0, 0.1));
    const Matrix4d T2 = mr_rp(mr::matrix_exp3(mr::vec_to_so3(Vector3d(0, 0, PI / 2))), Vector3d(0.2, 0.3, 0.15));
    CHECK_MAT_NEAR(lab::screw_path(T1, T2, 0.0), T1, 1e-12);
    CHECK_MAT_NEAR(lab::screw_path(T1, T2, 1.0), T2, 1e-12);
    const Matrix4d Tm = lab::screw_path(T1, T2, 0.5);
    CHECK_MAT_NEAR(Vector3d(Tm.topRightCorner<3, 1>()), Vector3d(0.1 + 0.3 * S2, 0.1 + 0.1 * S2, 0.125), 1e-9);
    CHECK_MAT_NEAR(Tm, as<Matrix4d>(cases()["screw_path_needle_guide"]["T_half"]), 1e-9);
}

// ------------------------------------------------------------- Checkpoint 5: quaternions
TEST_CASE("test_rot_to_quat_matches_scipy") {
    for (const auto& c : cases()["rot_to_quat_matches_scipy"]) {
        const Matrix3d R = as<Matrix3d>(c["R"]);
        const Vector4d q = lab::rot_to_quat(R);
        INFO("R =\n" << R << "\nq = " << q.transpose() << ", reference " << as<Vector4d>(c["q"]).transpose());
        CHECK_APPROX(q.norm(), 1.0);
        CHECK(q(0) >= -1e-12);
        CHECK(same_quat(q, as<Vector4d>(c["q"]), 1e-7));
    }
}

TEST_CASE("test_quat_to_rot_round_trip") {
    for (const auto& c : cases()["quat_to_rot_round_trip"]) {
        const Matrix3d R = as<Matrix3d>(c["R"]);
        // Python: assert_allclose(..., atol=1e-12) also allows rtol = 1e-7 (needed near theta = pi)
        CHECK_ALLCLOSE(lab::quat_to_rot(lab::rot_to_quat(R)), R, 1e-12);
        CHECK_ALLCLOSE(lab::quat_to_rot(Vector4d(-lab::rot_to_quat(R))), R, 1e-12);
    }
}

TEST_CASE("test_quat_mul_is_matrix_product") {
    for (const auto& c : cases()["quat_mul_is_matrix_product"]) {
        const Matrix3d Ra = as<Matrix3d>(c["Ra"]), Rb = as<Matrix3d>(c["Rb"]);
        const Vector4d q = lab::quat_mul(lab::rot_to_quat(Ra), lab::rot_to_quat(Rb));
        INFO("q = " << q.transpose() << ", reference " << as<Vector4d>(c["q"]).transpose());
        CHECK(same_quat(q, as<Vector4d>(c["q"]), 1e-9));
        const Vector4d qc = lab::quat_conj(lab::rot_to_quat(Ra));
        CHECK_ALLCLOSE(lab::quat_to_rot(qc), Matrix3d(Ra.transpose()), 1e-12);  // atol 1e-12 + rtol 1e-7, as Python
    }
}

TEST_CASE("test_workshop_quaternion_example") {
    // Rot(x, 90 deg) Rot(y, 90 deg) -> q = (1, 1, 1, 1)/2, a 120 deg rotation about (1,1,1)/sqrt 3.
    const Vector4d qx(std::cos(PI / 4), std::sin(PI / 4), 0.0, 0.0);
    const Vector4d qy(std::cos(PI / 4), 0.0, std::sin(PI / 4), 0.0);
    CHECK_MAT_NEAR(lab::quat_mul(qx, qy), Vector4d(0.5 * Vector4d::Ones()), 1e-12);
}

// ------------------------------------------------------------- Checkpoint 6: IMU data
TEST_CASE("test_gyro_exponential_integration_reproduces_truth") {
    for (const std::string seg : {"thigh", "shank"}) {
        INFO("segment " << seg);
        const MatrixXd q = quats(seg);
        const Matrix3d R0 = lab::quat_to_rot(qrow(q, 0));
        const std::vector<Matrix3d> Rs = lab::integrate_gyro_exp(R0, gyro(seg), DT);
        REQUIRE(Rs.size() == static_cast<std::size_t>(q.rows()));
        double err = 0.0;
        for (Eigen::Index k = 0; k < q.rows(); ++k) {
            const double a = lab::rotation_angle(Rs[static_cast<std::size_t>(k)], truth_rot(qrow(q, k)));
            if (!(a <= err)) err = a;  // like std::max, but a NaN angle is kept (and fails the check)
        }
        CHECK(err < 1e-8);
        const json& c = cases()["gyro_exponential_integration_reproduces_truth"][seg];
        CHECK_MAT_NEAR(Rs.back(), as<Matrix3d>(c["R_last"]), 1e-9);
    }
}

TEST_CASE("test_euler_integration_drifts_off_SO3") {
    const MatrixXd q = quats("thigh");
    const std::vector<Matrix3d> Rs = lab::integrate_gyro_euler(lab::quat_to_rot(qrow(q, 0)), gyro("thigh"), DT);
    REQUIRE(Rs.size() == static_cast<std::size_t>(q.rows()));
    // forward Euler does not preserve orthonormality: det grows above 1
    CHECK(Rs.back().determinant() > 1.01);
    CHECK_MAT_NEAR(Rs.back(), as<Matrix3d>(cases()["euler_integration_drifts_off_SO3"]["R_last"]), 1e-9);
    const std::vector<Matrix3d> Re = lab::integrate_gyro_exp(lab::quat_to_rot(qrow(q, 0)), gyro("thigh"), DT);
    REQUIRE(Re.size() == static_cast<std::size_t>(q.rows()));
    CHECK_SCALAR_NEAR(Re.back().determinant(), 1.0, 1e-12);  // Python: pytest.approx(1.0, abs=1e-12)
}

TEST_CASE("test_knee_flexion_from_quaternions") {
    const MatrixXd qth = quats("thigh"), qsh = quats("shank");
    VectorXd kap(qth.rows());
    for (Eigen::Index k = 0; k < qth.rows(); ++k)
        kap(k) = lab::knee_flexion(lab::quat_to_rot(qrow(qth, k)), lab::quat_to_rot(qrow(qsh, k)));
    CHECK(kap.size() == cases()["knee_flexion_from_quaternions"]["n"].get<int>());
    const VectorXd kappa = imu().data.col(imu().col("kappa"));
    CHECK_MAT_NEAR(kap, kappa, 1e-8);
}

TEST_CASE("test_noisy_gyro_drift_is_bounded_but_nonzero") {
    // Bias makes integrated orientation drift (reflection question); check the order of magnitude.
    const MatrixXd q = quats("shank");
    const std::vector<Matrix3d> Rs =
        lab::integrate_gyro_exp(lab::quat_to_rot(qrow(q, 0)), gyro("shank", true), DT);
    REQUIRE(!Rs.empty());
    const double drift = lab::rotation_angle(Rs.back(), lab::quat_to_rot(qrow(q, q.rows() - 1)));
    INFO("drift = " << drift << " rad");
    CHECK(drift > 0.01);
    CHECK(drift < 0.2);
}

// ------------------------------------------------------------- Checkpoint 7: tracked screw insertion
TEST_CASE("test_insertion_axis_from_two_poses") {
    // Biomedical: the log of T_210 T_200^-1 (0.2 s at full speed) gives the screw axis (direction, a point, pitch) of the insertion.
    const json& c = cases()["insertion_axis_from_two_poses"];
    const lab::PoseTrack tr = lab::load_pose_track(track_file());
    REQUIRE(tr.Ts.size() == c["n"].get<std::size_t>());
    REQUIRE(tr.Ts.size() > 210);
    const auto [S, th] = lab::screw_between(tr.Ts[200], tr.Ts[210], "space");
    CHECK_SCALAR_NEAR(th, c["theta_step"].get<double>(), 1e-9);
    const lab::ScrewParams sc = lab::twist_to_screw(S);
    const Vector3d s_hat = as<Vector3d>(c["s_hat"]), q_axis = as<Vector3d>(c["q_axis"]);
    const double h = c["h"].get<double>();
    CHECK_MAT_NEAR(sc.s_hat, s_hat, 1e-9);
    CHECK_SCALAR_NEAR(sc.h, h, 1e-9);
    const Vector3d dq = sc.q - q_axis;
    CHECK((dq - dq.dot(s_hat) * s_hat).norm() < 1e-9);  // q lies on the true axis
    const auto [Sb, tb] = lab::screw_between(tr.Ts[200], tr.Ts[210], "body");  // tip frame: z along the shaft
    CHECK_MAT_NEAR(Sb, v6(0, 0, 1, 0, 0, h), 1e-9);
    CHECK_SCALAR_NEAR(tb, th, 1e-9);
}

TEST_CASE("test_screw_motion_profile_ideal_data") {
    const json& c = cases()["screw_motion_profile_ideal_data"];
    const lab::PoseTrack tr = lab::load_pose_track(track_file());
    const auto [theta, d] = lab::screw_motion_profile(tr.Ts);
    REQUIRE(theta.size() == static_cast<Eigen::Index>(tr.Ts.size()));
    REQUIRE(d.size() == static_cast<Eigen::Index>(tr.Ts.size()));
    const double h = c["h"].get<double>();
    CHECK_MAT_NEAR(theta, sbeg::as_vec(c["theta"]), 1e-8);
    CHECK_MAT_NEAR(d, VectorXd(h * theta), 1e-10);
    CHECK_SCALAR_NEAR(theta(theta.size() - 1), c["theta_end"].get<double>(), 1e-8);
    CHECK_SCALAR_NEAR(d(d.size() - 1), c["d_end"].get<double>(), 1e-10);
    // one log of the whole motion cannot see the 5 full turns: R_N = R_0, so it is a pure translation
    const auto [S, th] = lab::screw_between(tr.Ts.front(), tr.Ts.back(), "space");
    Vector6d S_exp;
    S_exp << Vector3d::Zero(), as<Vector3d>(c["S_hat"]);
    CHECK_MAT_NEAR(S, S_exp, 1e-6);
    CHECK_SCALAR_NEAR(th, 0.0125, 1e-9);
}

TEST_CASE("test_screw_motion_profile_handles_reversal") {
    // The surgeon backs the screw out: the angle (and the advance) must decrease.
    const json& c = cases()["screw_motion_profile_handles_reversal"];
    const Vector6d S = as<Vector6d>(c["S"]);
    const Matrix4d T0 = as<Matrix4d>(c["T0"]);
    const VectorXd ths = sbeg::as_vec(c["ths"]);
    std::vector<Matrix4d> Ts;
    for (Eigen::Index k = 0; k < ths.size(); ++k) Ts.push_back(mr::matrix_exp6(mr::vec_to_se3(S) * ths(k)) * T0);
    const auto [theta, d] = lab::screw_motion_profile(Ts);
    CHECK_MAT_NEAR(theta, ths, 1e-9);
    CHECK_MAT_NEAR(d, VectorXd(c["h"].get<double>() * ths), 1e-12);
    const std::vector<Matrix4d> shift = {Matrix4d::Identity(), mr_rp(Matrix3d::Identity(), Vector3d(0, 0, 0.01))};
    CHECK_THROWS_AS(lab::screw_motion_profile(shift), std::invalid_argument);
}

TEST_CASE("test_screw_motion_profile_noisy_data") {
    // Tracker noise: the fitted lead is still within 5 % of 2.5 mm per turn.
    const json& c = cases()["screw_motion_profile_noisy_data"];
    const lab::PoseTrack tr = lab::load_pose_track(track_file(), true);
    const auto [theta, d] = lab::screw_motion_profile(tr.Ts);
    REQUIRE(theta.size() == static_cast<Eigen::Index>(tr.Ts.size()));
    REQUIRE(d.size() == theta.size());
    const double lead = c["lead"].get<double>(), theta_end = c["theta_end"].get<double>();
    const double lead_fit = 2 * PI * fit_slope(theta, d);
    INFO("fitted lead = " << lead_fit * 1e3 << " mm");
    CHECK(std::abs(lead_fit - lead) < 0.05 * lead);
    CHECK(std::abs(theta(theta.size() - 1) - theta_end) < 0.01 * theta_end);
}

// ------------------------------------------------------------- Checkpoint 8: EOD gripper errors
TEST_CASE("test_orientation_error_eod_gripper") {
    // Defence/EOD: gripper wrist must turn from R_cur to R_des; error in the gripper frame.
    const double deg = PI / 180.0;
    const Matrix3d R_cur = rot(Vector3d::UnitZ(), 40 * deg) * rot(Vector3d::UnitY(), -20 * deg);
    const Matrix3d R_des = R_cur * rot(Vector3d::UnitX(), 35 * deg);
    const Vector3d e = lab::orientation_error(R_cur, R_des);
    CHECK_MAT_NEAR(e, Vector3d(35 * deg, 0.0, 0.0), 1e-12);
    for (const auto& c : cases()["orientation_error_eod_gripper"]["pairs"]) {
        const Matrix3d Ra = as<Matrix3d>(c["Ra"]), Rb = as<Matrix3d>(c["Rb"]);
        const Vector3d ref = mr::so3_to_vec(mr::matrix_log3(Matrix3d(Ra.transpose() * Rb)));
        CHECK_MAT_NEAR(lab::orientation_error(Ra, Rb), ref, 1e-9);
        CHECK_MAT_NEAR(lab::orientation_error(Ra, Rb), as<Vector3d>(c["e"]), 1e-9);
        const double ang = lab::rotation_angle(Ra, Rb);
        CHECK(ang >= 0.0);
        CHECK(ang <= PI + 1e-12);
    }
}

TEST_CASE("test_eod_gripper_screw_motion") {
    // Defence/EOD: full pose error of the gripper as a screw motion in the gripper frame. Desired pose = 35 deg about
    // an axis parallel to the gripper x-axis through (0, 0, 0.05) m (gripper frame), with 0.02 m/rad advance along it.
    const double deg = PI / 180.0;
    const Matrix3d R_cur = rot(Vector3d::UnitZ(), 40 * deg) * rot(Vector3d::UnitY(), -20 * deg);
    const Matrix4d T_cur = mr_rp(R_cur, Vector3d(0.60, 0.10, 0.40));
    Vector6d Sb_true;
    Sb_true << Vector3d::UnitX(), -Vector3d::UnitX().cross(Vector3d(0, 0, 0.05)) + 0.02 * Vector3d::UnitX();
    const Matrix4d T_des = T_cur * mr::matrix_exp6(mr::vec_to_se3(Sb_true) * (35 * deg));
    const auto [Sb, th] = lab::screw_between(T_cur, T_des, "body");
    CHECK_MAT_NEAR(Sb, v6(1, 0, 0, 0.02, 0.05, 0), 1e-9);
    CHECK_SCALAR_NEAR(th, 35 * deg, 1e-9);
    const lab::ScrewParams sc = lab::twist_to_screw(Sb);
    CHECK_MAT_NEAR(sc.q, Vector3d(0, 0, 0.05), 1e-9);
    CHECK_SCALAR_NEAR(sc.h, 0.02, 1e-9);
    const auto [Ss, ts] = lab::screw_between(T_cur, T_des, "space");
    CHECK_MAT_NEAR(Ss, Vector6d(mr::adjoint(T_cur) * Sb), 1e-9);
    CHECK_MAT_NEAR(lab::screw_path(T_cur, T_des, 0.5),
                   Matrix4d(T_cur * mr::matrix_exp6(mr::vec_to_se3(Sb_true) * (17.5 * deg))), 1e-9);
    CHECK_MAT_NEAR(lab::screw_path(T_cur, T_des, 0.5), as<Matrix4d>(cases()["eod_gripper_screw_motion"]["T_half"]), 1e-9);
}
