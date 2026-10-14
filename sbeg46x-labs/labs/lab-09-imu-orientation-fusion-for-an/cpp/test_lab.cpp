// Auto-checks for SBEG 46X Lab 9 (IMU orientation fusion for an exoskeleton thigh segment: complementary filter vs
// EKF) -- C++ track.  Mirrors test_lab.py test by test (25 test cases, same names and order).
//
// Build and run from this folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
//     ./build/lab_tests -tc="*A4*,*cf_alpha*"      (run only some tests, like pytest -k)
// Ground truth: the thigh-IMU pose and ideal accelerometer of the course model gait_exo_2r (mrcpp fkin_space via
// course_models.cpp), rotation matrices from mrcpp matrix_exp3, closed-form responses of first-order filters, the
// information form of the Kalman update, the Astrom-Murray predictor form (Eq. 7.22), finite-difference Jacobians,
// the hip-encoder reference of the recording, the reference results in data/expected.json, and cpp_cases.json,
// which holds the random inputs of test_lab.py and the numbers of the Python reference, so both languages are
// checked against the same values.
//
// C++ only: the four data/provided-code tests (which pass with the untouched Python starter) also check one of
// your functions here (bias_at_rest, quat_to_rot, accel_tilt), so every C++ test case fails until you have
// written something.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <doctest/doctest.h>

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using Eigen::Matrix2d;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;
using sbeg::as;
using sbeg::as_mat;
using sbeg::as_vec;
using sbeg::json;

namespace {
constexpr double PI = 3.14159265358979323846;
constexpr double G = 9.81;

const json& cases() {
    static const json C = sbeg::load_json(sbeg::lab_path("cpp_cases.json"));
    return C;
}

const json& EXP() {
    static const json E = sbeg::load_json(sbeg::lab_path("../data/expected.json"));
    return E;
}

const lab::ImuMeta& META() {
    static const lab::ImuMeta M = lab::load_meta();
    return M;
}

const lab::Recording& REC() {
    static const lab::Recording R = lab::load_recording();
    return R;
}

double num(const json& j) { return j.get<double>(); }
double deg(double rad) { return rad * 180.0 / PI; }

/// np.allclose(got, expected, rtol, atol): |got - expected| <= atol + rtol |expected| entry by entry.
sbeg::NearResult allclose(const MatrixXd& got, const MatrixXd& exp, double atol = 1e-8, double rtol = 1e-5) {
    sbeg::NearResult r = sbeg::compare_near(got, exp, std::numeric_limits<double>::infinity());
    if (!r.ok || got.size() == 0) return r;  // shape mismatch or NaN/inf (message already set)
    MatrixXd g = got, e = exp;
    if (g.rows() != e.rows()) g.transposeInPlace();  // a row and a column vector compare equal
    const MatrixXd excess = (g - e).cwiseAbs() - rtol * e.cwiseAbs();
    Eigen::Index i = 0, k = 0;
    const double worst = excess.maxCoeff(&i, &k);
    r.ok = worst <= atol;
    std::ostringstream s;
    s << std::setprecision(12) << "worst entry (" << i << "," << k << "): got " << g(i, k) << ", expected "
      << e(i, k) << " (atol " << atol << ", rtol " << rtol << ")";
    r.msg = s.str();
    return r;
}

#define CHECK_ALLCLOSE(got, expected, atol)                                                 \
    do {                                                                                    \
        const ::sbeg::NearResult ac_ = allclose((got), (expected), (atol));                 \
        CHECK_MESSAGE(ac_.ok, "np.allclose(" #got ", " #expected ")\n  " << ac_.msg);       \
    } while (false)

Matrix3d rot_from_axis_angle(const Vector3d& w_hat, double phi) {
    return mr::matrix_exp3(mr::vec_to_so3(w_hat * phi));
}

Vector4d quat_from_axis_angle(const Vector3d& w, double phi) {  // MR Eq. B.9
    const Vector3d w_hat = w / w.norm();
    Vector4d q;
    q << std::cos(phi / 2), w_hat * std::sin(phi / 2);
    return q;
}

/// t_k = k h, k = 0..n-1 (np.arange(0, n h, h)).
VectorXd time_axis(int n, double h) {
    VectorXd t(n);
    for (int k = 0; k < n; ++k) t(k) = k * h;
    return t;
}

std::vector<bool> mask_before(const VectorXd& t, double t_end) {
    std::vector<bool> m(static_cast<std::size_t>(t.size()));
    for (Eigen::Index k = 0; k < t.size(); ++k) m[static_cast<std::size_t>(k)] = t(k) < t_end;
    return m;
}
}  // namespace

// ------------------------------------------------------------------------------ data and provided code
TEST_CASE("test_recording_format") {
    const json& c = cases()["recording_format"];
    const sbeg::CsvTable tab = sbeg::read_csv(sbeg::lab_path("../data/thigh_imu.csv"));
    std::vector<std::string> cols;
    for (const auto& s : c["columns"]) cols.push_back(s.get<std::string>());
    CHECK(tab.header == cols);
    const lab::Recording& R = REC();
    const Eigen::Index N = EXP()["N"].get<Eigen::Index>();
    CHECK(R.t.size() == N);
    for (const VectorXd* v : {&R.gx, &R.gy, &R.gz, &R.ax, &R.ay, &R.az, &R.theta_enc}) CHECK(v->size() == N);
    const VectorXd dt = R.t.tail(N - 1) - R.t.head(N - 1);
    CHECK_ALLCLOSE(dt, VectorXd::Constant(N - 1, META().h), 1e-9);
    CHECK(META().r_imu == lab::R_IMU);
    // C++ only: your bias_at_rest on the rest phase reproduces the reference value of the recording
    CHECK_SCALAR_NEAR(lab::bias_at_rest(R.t, R.gz, 8.0), num(c["bias_at_rest"]), 1e-12);
}

TEST_CASE("test_course_model_imu_pose") {
    // T_sb from the PoE of gait_exo_2r equals Rot(z, theta) at p = r (sin theta, -cos theta, 0).
    for (const auto& c : cases()["course_model_imu_pose"]) {
        const double th = num(c["theta"]);
        INFO("theta = " << th);
        const auto [R, p] = mr::trans_to_rp(lab::imu_pose(th));
        const double co = std::cos(th), si = std::sin(th);
        Matrix3d Rz;
        Rz << co, -si, 0,
              si, co, 0,
              0, 0, 1;
        CHECK_ALLCLOSE(R, Rz, 1e-12);
        CHECK_ALLCLOSE(p, (lab::R_IMU * Vector3d(si, -co, 0.0)), 1e-12);
        CHECK_MAT_NEAR(R, as<Matrix3d>(c["R"]), 1e-12);  // Python course model (modern_robotics)
        // C++ only: your quat_to_rot gives the same R_sb = Rot(z, theta) from q = (cos th/2, 0, 0, sin th/2)
        CHECK_MAT_NEAR(lab::quat_to_rot(Vector4d(std::cos(th / 2), 0.0, 0.0, std::sin(th / 2))), Rz, 1e-12);
    }
}

TEST_CASE("test_recording_at_rest_matches_model") {
    // While the leg hangs (t < 8 s) the accelerometer mean equals the ideal reading of the course model.
    const lab::Recording& R = REC();
    const std::vector<bool> rest = mask_before(R.t, 8.0);
    Vector3d s = Vector3d::Zero();
    int n = 0;
    for (Eigen::Index k = 0; k < R.t.size(); ++k) {
        if (!rest[static_cast<std::size_t>(k)]) continue;
        s += Vector3d(R.ax(k), R.ay(k), R.az(k));
        ++n;
    }
    REQUIRE(n > 0);
    const Vector3d f_meas = s / n;
    const Vector3d f_model = lab::imu_specific_force(0.0, 0.0, 0.0);
    CHECK_ALLCLOSE(f_meas, f_model, 0.02);
    CHECK_ALLCLOSE(f_model, Vector3d(0.0, G, 0.0), 1e-12);
    // C++ only: your accel_tilt of the mean rest reading is (almost) zero, as the Python reference
    const double tilt = lab::accel_tilt(f_meas(0), f_meas(1));
    CHECK(std::abs(tilt) < 0.002);
    CHECK_SCALAR_NEAR(tilt, num(cases()["recording_at_rest_matches_model"]["tilt"]), 1e-12);
}

TEST_CASE("test_provided_quat_from_accel") {
    const Vector4d q = lab::quat_from_accel(Vector3d(G * std::sin(0.3), G * std::cos(0.3), 0.0));
    const Vector3d up(0.0, 1.0, 0.0), expect(std::sin(0.3), std::cos(0.3), 0.0);
    const Matrix3d R = rot_from_axis_angle(q.tail<3>() / q.tail<3>().norm(), 2 * std::acos(q(0)));
    CHECK_ALLCLOSE((R.transpose() * up), expect, 1e-12);
    CHECK_MAT_NEAR(q, as_vec(cases()["provided_quat_from_accel"]["q"]), 1e-12);
    // C++ only: the same check with your quat_to_rot
    CHECK_ALLCLOSE((lab::quat_to_rot(q).transpose() * up), expect, 1e-12);
}

// ------------------------------------------------------------------------------ Part A
TEST_CASE("test_A1_accel_tilt_inverts_the_course_model") {
    const json& c = cases()["A1_accel_tilt_inverts_the_course_model"];
    const VectorXd th = as_vec(c["theta"]);
    VectorXd fx(th.size()), fy(th.size());
    for (Eigen::Index i = 0; i < th.size(); ++i) {
        const Vector3d f = lab::imu_specific_force(th(i), 0.0, 0.0);
        fx(i) = f(0);
        fy(i) = f(1);
    }
    CHECK_MAT_NEAR(fx, as_vec(c["fx"]), 1e-12);  // course model = Python course model
    CHECK_ALLCLOSE(lab::accel_tilt(fx, fy), th, 1e-12);
    CHECK_ALLCLOSE(lab::accel_tilt(VectorXd(3 * fx), VectorXd(3 * fy)), th, 1e-12);  // scale-free
}

TEST_CASE("test_A1_accel_tilt_error_during_swing") {
    // Dynamic acceleration (r theta_dot^2, r theta_ddot) biases the accelerometer tilt: Workshop Problem 3b.
    const Vector3d f = lab::imu_specific_force(20.0 * PI / 180.0, 3.0, 20.0);
    CHECK(std::abs(deg(lab::accel_tilt(f(0), f(1))) - 33.72) < 0.01);
}

TEST_CASE("test_A2_gyro_integrate_forward_euler") {
    const json& c = cases()["A2_gyro_integrate_forward_euler"];
    const VectorXd t = as_vec(c["t"]), w = as_vec(c["w"]);
    const VectorXd out = lab::gyro_integrate(t, w, 0.2, 0.1);
    VectorXd ref(t.size());
    ref(0) = 0.2;
    for (Eigen::Index k = 1; k < t.size(); ++k) ref(k) = ref(k - 1) + 0.01 * (w(k) - 0.1);
    CHECK_ALLCLOSE(out, ref, 1e-12);
    CHECK_ALLCLOSE(out, as_vec(c["ref"]), 1e-12);
    // constant rate: exact ramp
    CHECK_ALLCLOSE(lab::gyro_integrate(t, VectorXd::Constant(t.size(), 0.5)), VectorXd(0.5 * t), 1e-12);
}

TEST_CASE("test_A3_cf_alpha") {
    CHECK(std::abs(lab::cf_alpha(0.5, 0.01) - 0.5 / 0.51) < 1e-15);
    CHECK(std::abs(lab::cf_alpha(0.0, 0.01)) < 1e-15);
}

TEST_CASE("test_A4_complementary_step_and_bias_responses") {
    const json& c = cases()["A4_complementary_step_and_bias_responses"];
    const double h = 0.01, tau = 0.5;
    const int n = c["n"].get<int>();
    const VectorXd t = time_axis(n, h);
    const double a = tau / (tau + h);
    // accelerometer step, gyro silent: theta_k = 1 - a^k  (first-order low-pass of the accelerometer)
    VectorXd th = lab::complementary_filter(t, VectorXd::Zero(n), VectorXd::Ones(n), tau, 0.0);
    VectorXd expect(n);
    for (int k = 0; k < n; ++k) expect(k) = 1 - std::pow(a, k);
    CHECK_ALLCLOSE(th, expect, 1e-12);
    // constant gyro bias b, segment still: steady error a h b / (1 - a) = tau b
    const double b = 0.02;
    th = lab::complementary_filter(t, VectorXd::Constant(n, b), VectorXd::Zero(n), tau, 0.0);
    REQUIRE(th.size() == n);
    CHECK(std::abs(th(n - 1) - tau * b) < 1e-6);
    // and the bias argument removes it
    th = lab::complementary_filter(t, VectorXd::Constant(n, b), VectorXd::Zero(n), tau, 0.0, b);
    CHECK_ALLCLOSE(th, VectorXd::Zero(n), 1e-15);
}

TEST_CASE("test_A4_complementary_on_recording") {
    const lab::Recording& R = REC();
    const VectorXd th_acc = lab::accel_tilt(R.ax, R.ay);
    const VectorXd th = lab::complementary_filter(R.t, R.gz, th_acc, 0.5);
    const double rms = lab::rms_deg(th, R.theta_enc);
    CHECK(std::abs(rms - num(EXP()["rms_deg"]["cf_tau05"])) < 1e-6);
    CHECK(rms < 0.3 * lab::rms_deg(th_acc, R.theta_enc));
}

TEST_CASE("test_A5_bias_at_rest") {
    const lab::Recording& R = REC();
    const double b = lab::bias_at_rest(R.t, R.gz, 8.0);
    double s = 0.0;
    int n = 0;
    for (Eigen::Index k = 0; k < R.t.size(); ++k) {
        if (R.t(k) < 8.0) {
            s += R.gz(k);
            ++n;
        }
    }
    CHECK(std::abs(b - s / n) < 1e-12);
    CHECK(std::abs(b - num(EXP()["true_gyro_bias_z_start"])) < 0.002);
}

// ------------------------------------------------------------------------------ Part B
TEST_CASE("test_B1_kf_predict") {
    const json& c = cases()["B1_kf_predict"];
    const MatrixXd A = as_mat(c["A"]), B = as_mat(c["B"]), P = as_mat(c["P"]), Q = as_mat(c["Q"]);
    const VectorXd x = as_vec(c["x"]);
    const double u = num(c["u"]);
    const auto [xp, Pp] = lab::kf_predict(x, P, A, Q, B, VectorXd::Constant(1, u));
    CHECK_ALLCLOSE(xp, VectorXd(A * x + B.col(0) * u), 1e-12);
    CHECK_ALLCLOSE(Pp, MatrixXd(A * P * A.transpose() + Q), 1e-12);
    CHECK_MAT_NEAR(xp, as_vec(c["xp"]), 1e-12);
    const auto [xp0, Pp0] = lab::kf_predict(x, P, A, Q);
    CHECK_ALLCLOSE(xp0, VectorXd(A * x), 1e-12);
}

TEST_CASE("test_B2_kf_update_information_form") {
    // Covariance form == information form: P+ = (P^-1 + C^T R^-1 C)^-1, x+ = P+ (P^-1 x + C^T R^-1 y).
    const json& c = cases()["B2_kf_update_information_form"];
    const MatrixXd P = as_mat(c["P"]), C = as_mat(c["C"]), R = as_mat(c["R"]);
    const VectorXd x = as_vec(c["x"]), y = as_vec(c["y"]);
    const lab::KfUpdate up = lab::kf_update(x, P, y, C, R);
    const MatrixXd Pi = (P.inverse() + C.transpose() * R.inverse() * C).inverse();
    CHECK_ALLCLOSE(up.P, Pi, 1e-10);
    CHECK_ALLCLOSE(up.x, VectorXd(Pi * (P.partialPivLu().solve(x) + C.transpose() * R.partialPivLu().solve(y))),
                   1e-10);
    CHECK_ALLCLOSE(up.nu, VectorXd(y - C * x), 1e-12);
    CHECK_ALLCLOSE(up.S, MatrixXd(C * P * C.transpose() + R), 1e-12);
    CHECK_MAT_NEAR(up.x, as_vec(c["xn"]), 1e-10);
}

TEST_CASE("test_B2_kf_matches_astrom_murray_predictor_form") {
    // update then predict == Astrom & Murray Eq. 7.22: L = A P C^T (R + C P C^T)^-1,
    // P[k+1] = (A - L C) P (A - L C)^T + Q + L R L^T  (their R_v = Q, R_w = R).
    MatrixXd A(2, 2), C(1, 2), Q(2, 2), R(1, 1), P(2, 2);
    A << 1.0, -0.01,
         0.0, 1.0;
    C << 1.0, 0.0;
    Q << 1e-4, 0.0,
         0.0, 1e-6;
    R << 4e-4;
    P << 0.01, 0.0,
         0.0, 0.002;
    const VectorXd x = Vector2d(0.1, 0.0), y = VectorXd::Constant(1, 0.12);
    const lab::KfUpdate up = lab::kf_update(x, P, y, C, R);
    const auto [xp, Pp] = lab::kf_predict(up.x, up.P, A, Q);
    const MatrixXd Lg = A * P * C.transpose() * (R + C * P * C.transpose()).inverse();
    const MatrixXd ALC = A - Lg * C;
    const MatrixXd P_am = ALC * P * ALC.transpose() + Q + Lg * R * Lg.transpose();
    CHECK_ALLCLOSE(Pp, P_am, 1e-14);
    CHECK_ALLCLOSE(xp, VectorXd(A * x + Lg * (y - C * x)), 1e-14);
}

TEST_CASE("test_B2_scalar_workshop_problem_1") {
    VectorXd x = VectorXd::Zero(1);
    MatrixXd P = MatrixXd::Constant(1, 1, 16.0);
    const MatrixXd one = MatrixXd::Ones(1, 1);
    for (double y : {3.0, 1.0, 2.5}) {
        auto pred = lab::kf_predict(x, P, one, MatrixXd::Constant(1, 1, 2.0));
        const lab::KfUpdate up =
            lab::kf_update(pred.first, pred.second, VectorXd::Constant(1, y), one, MatrixXd::Constant(1, 1, 4.0));
        x = up.x;
        P = up.P;
    }
    REQUIRE(x.size() == 1);
    REQUIRE(P.size() == 1);
    CHECK(std::abs(x(0) - 2.0782) < 1e-4);
    CHECK(std::abs(P(0, 0) - 2.0664) < 1e-4);
}

TEST_CASE("test_B3_accel_model_and_jacobian") {
    for (const auto& c : cases()["B3_accel_model_and_jacobian"]) {
        const double th = num(c["theta"]);
        INFO("theta = " << th);
        const auto [hx, H] = lab::accel_model(Vector2d(th, 0.03));
        CHECK_ALLCLOSE(hx, (lab::imu_specific_force(th, 0.0, 0.0).head<2>()), 1e-12);
        const double eps = 1e-6;
        Matrix2d Hfd = Matrix2d::Zero();
        Hfd.col(0) = (lab::accel_model(Vector2d(th + eps, 0.03)).first -
                      lab::accel_model(Vector2d(th - eps, 0.03)).first) / (2 * eps);
        CHECK_ALLCLOSE(H, Hfd, 1e-6);
        CHECK_MAT_NEAR(H, as<Matrix2d>(c["H"]), 1e-12);
    }
}

TEST_CASE("test_B4_ekf_converges_on_static_data") {
    // Segment still at 25 deg, gyro reads only its bias: the EKF learns both the angle and the bias.
    const int n = cases()["B4_ekf_converges_on_static_data"]["n"].get<int>();
    const double h = 0.01, th0 = 25.0 * PI / 180.0, b = 0.03;
    const VectorXd t = time_axis(n, h);
    Matrix2d P0;
    P0 << 0.5, 0.0,
          0.0, 0.05 * 0.05;
    const lab::EkfResult out =
        lab::ekf_tilt(t, VectorXd::Constant(n, b), VectorXd::Constant(n, G * std::sin(th0)),
                      VectorXd::Constant(n, G * std::cos(th0)), 0.006, 0.02, 0.5, Vector2d(0.0, 0.0), P0);
    REQUIRE(out.theta.size() == n);
    REQUIRE(out.bias.size() == n);
    REQUIRE(out.P.size() == static_cast<std::size_t>(n));
    CHECK(std::abs(out.theta(n - 1) - th0) < 1e-3);
    CHECK(std::abs(out.bias(n - 1) - b) < 1e-3);
    const Eigen::SelfAdjointEigenSolver<Matrix2d> es(out.P.back());
    CHECK((es.eigenvalues().array() > 0).all());
}

TEST_CASE("test_B4_ekf_gate_zero_is_dead_reckoning") {
    const lab::Recording& R = REC();
    const int n = 500;
    const lab::EkfResult out = lab::ekf_tilt(R.t.head(n), R.gz.head(n), R.ax.head(n), R.ay.head(n), 0.006, 0.02,
                                             1.5, std::nullopt, std::nullopt, 0.0);
    CHECK(out.rejected == 499);
    const double th0 = std::atan2(R.ax(0), R.ay(0));
    CHECK_ALLCLOSE(out.theta, lab::gyro_integrate(R.t.head(n), R.gz.head(n), th0), 1e-12);
}

TEST_CASE("test_B4_ekf_on_recording") {
    const lab::Recording& R = REC();
    const Eigen::Index N = R.t.size();
    const lab::EkfResult out = lab::ekf_tilt(R.t, R.gz, R.ax, R.ay, 0.006, 0.02, 1.5);
    REQUIRE(out.theta.size() == N);
    REQUIRE(out.bias.size() == N);
    REQUIRE(out.nis.size() == N);
    const double rms = lab::rms_deg(out.theta, R.theta_enc);
    CHECK(std::abs(rms - num(EXP()["rms_deg"]["ekf_tuned"])) < 1e-6);
    CHECK(rms < 1.2);
    CHECK(std::abs(out.bias(N - 1) - num(EXP()["true_gyro_bias_z_end"])) < 0.005);
    CHECK(std::isnan(out.nis(0)));
    CHECK(out.nis.tail(N - 1).allFinite());
    const lab::EkfResult naive = lab::ekf_tilt(R.t, R.gz, R.ax, R.ay, 0.006, 0.02, 0.06);
    CHECK(lab::rms_deg(naive.theta, R.theta_enc) > 3 * rms);  // R too small: trusts swing
}

// ------------------------------------------------------------------------------ Part C
TEST_CASE("test_C2_quat_to_rot_matches_matrix_exp") {
    for (const auto& c : cases()["C2_quat_to_rot_matches_matrix_exp"]) {
        const Vector3d w = as<Vector3d>(c["w"]);
        const double phi = num(c["phi"]);
        INFO("w = " << w.transpose() << ", phi = " << phi);
        CHECK_ALLCLOSE(lab::quat_to_rot(quat_from_axis_angle(w, phi)), rot_from_axis_angle(w, phi), 1e-12);
    }
}

TEST_CASE("test_C1_quat_mult_is_rotation_product") {
    for (const auto& c : cases()["C1_quat_mult_is_rotation_product"]) {
        const Vector3d w1 = as<Vector3d>(c["w1"]), w2 = as<Vector3d>(c["w2"]);
        const double a1 = num(c["a1"]), a2 = num(c["a2"]);
        INFO("a1 = " << a1 << ", a2 = " << a2);
        const Vector4d n = lab::quat_mult(quat_from_axis_angle(w1, a1), quat_from_axis_angle(w2, a2));
        CHECK(std::abs(n.norm() - 1) < 1e-12);
        CHECK_ALLCLOSE(lab::quat_to_rot(n), MatrixXd(rot_from_axis_angle(w1, a1) * rot_from_axis_angle(w2, a2)),
                       1e-12);
        CHECK_MAT_NEAR(n, as_vec(c["n"]), 1e-12);
    }
}

TEST_CASE("test_C3_quat_exp") {
    for (const auto& c : cases()["C3_quat_exp"]) {
        const Vector3d w = as<Vector3d>(c["w"]);
        INFO("w = " << w.transpose());
        CHECK_ALLCLOSE(lab::quat_to_rot(lab::quat_exp(w, 0.01)), mr::matrix_exp3(mr::vec_to_so3(w * 0.01)), 1e-12);
    }
    CHECK_ALLCLOSE(lab::quat_exp(Vector3d::Zero(), 0.01), Vector4d(1, 0, 0, 0), 1e-8);
}

TEST_CASE("test_C5_thigh_angle_from_quat_ignores_heading") {
    for (const auto& c : cases()["C5_thigh_angle_from_quat_ignores_heading"]) {
        const double th = num(c["theta"]);
        INFO("theta = " << th);
        const Vector4d q = quat_from_axis_angle(Vector3d(0, 0, 1), th);  // R_sb = Rot(z, theta)
        CHECK(std::abs(lab::thigh_angle_from_quat(q) - th) < 1e-12);
        const Vector4d q_turned = lab::quat_mult(quat_from_axis_angle(Vector3d(0, 1, 0), 0.7), q);  // about y_s
        CHECK(std::abs(lab::thigh_angle_from_quat(q_turned) - th) < 1e-12);
    }
}

TEST_CASE("test_C4_mahony_static_learns_tilt_and_observable_bias") {
    const int n = cases()["C4_mahony_static_learns_tilt_and_observable_bias"]["n"].get<int>();
    const double h = 0.01, th0 = 30.0 * PI / 180.0;
    const VectorXd t = time_axis(n, h);
    const Eigen::RowVector3d b(0.0, 0.0, 0.02), a(G * std::sin(th0), G * std::cos(th0), 0.0);
    const MatrixXd gyro3 = b.replicate(n, 1), acc3 = a.replicate(n, 1);
    const lab::MahonyResult out = lab::mahony_filter(t, gyro3, acc3, 1.0, 0.1, Vector4d(1.0, 0.0, 0.0, 0.0));
    REQUIRE(out.q.rows() == n);
    REQUIRE(out.q.cols() == 4);
    REQUIRE(out.bias.rows() == n);
    REQUIRE(out.bias.cols() == 3);
    const Vector4d q_end = out.q.row(n - 1).transpose();
    CHECK(std::abs(lab::thigh_angle_from_quat(q_end) - th0) < 1e-3);
    CHECK(std::abs(out.bias(n - 1, 2) - 0.02) < 1e-3);
}

TEST_CASE("test_C4_mahony_on_recording") {
    const lab::Recording& R = REC();
    const Eigen::Index N = R.t.size();
    MatrixXd gyro3(N, 3), acc3(N, 3);
    gyro3 << R.gx, R.gy, R.gz;
    acc3 << R.ax, R.ay, R.az;
    const lab::MahonyResult out = lab::mahony_filter(R.t, gyro3, acc3, 1.0, 0.1);
    REQUIRE(out.q.rows() == N);
    REQUIRE(out.q.cols() == 4);
    REQUIRE(out.bias.rows() == N);
    REQUIRE(out.bias.cols() == 3);
    VectorXd th(N);
    for (Eigen::Index k = 0; k < N; ++k) th(k) = lab::thigh_angle_from_quat(out.q.row(k).transpose());
    const double rms = lab::rms_deg(th, R.theta_enc);
    CHECK(std::abs(rms - num(EXP()["rms_deg"]["mahony"])) < 1e-6);
    CHECK(rms < 2.5);
    CHECK(std::abs(out.bias(N - 1, 2) - num(EXP()["true_gyro_bias_z_end"])) < 0.01);
    // heading (about the vertical) is not observable from the accelerometer: it drifts
    CHECK(std::abs(deg(lab::heading_from_quat(out.q.row(N - 1).transpose()))) > 10.0);
}
