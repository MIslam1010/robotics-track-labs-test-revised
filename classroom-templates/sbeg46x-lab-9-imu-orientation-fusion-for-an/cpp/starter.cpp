// SBEG 46X Lab 9 -- C++ track: STARTER CODE.  Fill in every function marked TODO (in the order A1 ... C5).
// The declarations and their documentation are in lab.hpp; the provided functions (load_meta, load_recording,
// rms_deg, quat_from_accel, heading_from_quat, accel_tilt for one sample) are in helpers.cpp and the course model
// (gait_exo_2r, imu_pose, imu_specific_force) in course_models.cpp.  Keep the names and signatures.
//
// Build and run the auto-checks from your Lab code folder:   python run_lab.py 9 --cpp
//   or:  cmake -S cpp -B build -G Ninja && cmake --build build && ./build/lab_tests
#include "lab.hpp"

#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

// Need pi?  Write constexpr double PI = 3.14159265358979323846; (never M_PI, which MSVC does not define).
namespace lab {

// ==========================================================================================
// Part A -- tilt from the accelerometer, gyro integration and the complementary filter
// ==========================================================================================
Eigen::VectorXd accel_tilt(const Eigen::VectorXd& ax, const Eigen::VectorXd& ay) {
    // TODO A1: write this function (see lab.hpp and the lab handout).  One std::atan2 per sample.
    throw std::logic_error("TODO: accel_tilt");
}

Eigen::VectorXd gyro_integrate(const Eigen::VectorXd& t, const Eigen::VectorXd& gyro, double theta0, double bias) {
    // TODO A2: write this function (see lab.hpp and the lab handout).  Use gyro(k), not gyro(k - 1).
    throw std::logic_error("TODO: gyro_integrate");
}

double cf_alpha(double tau, double h) {
    // TODO A3: write this function (see lab.hpp and the lab handout).
    throw std::logic_error("TODO: cf_alpha");
}

Eigen::VectorXd complementary_filter(const Eigen::VectorXd& t, const Eigen::VectorXd& gyro,
                                     const Eigen::VectorXd& theta_acc, double tau, std::optional<double> theta0,
                                     double bias) {
    // TODO A4: write this function (see lab.hpp and the lab handout).
    // theta0 is a std::optional<double>: `if (theta0)` is true when it was given, its value is *theta0.
    throw std::logic_error("TODO: complementary_filter");
}

double bias_at_rest(const Eigen::VectorXd& t, const Eigen::VectorXd& gyro, double t_end) {
    // TODO A5: write this function (see lab.hpp and the lab handout).
    throw std::logic_error("TODO: bias_at_rest");
}

// ==========================================================================================
// Part B -- linear Kalman filter and the EKF with gyro-bias state
// ==========================================================================================
std::pair<Eigen::VectorXd, Eigen::MatrixXd> kf_predict(const Eigen::VectorXd& x, const Eigen::MatrixXd& P,
                                                       const Eigen::MatrixXd& A, const Eigen::MatrixXd& Q,
                                                       const Eigen::MatrixXd& B, const Eigen::VectorXd& u) {
    // TODO B1: write this function (see lab.hpp and the lab handout).
    // B and u are "None" when empty: add B u only if B.size() > 0 && u.size() > 0.  Return {xp, Pp}.
    throw std::logic_error("TODO: kf_predict");
}

KfUpdate kf_update(const Eigen::VectorXd& x, const Eigen::MatrixXd& P, const Eigen::VectorXd& y,
                   const Eigen::MatrixXd& C, const Eigen::MatrixXd& R) {
    // TODO B2: write this function (see lab.hpp and the lab handout).
    // Fill a KfUpdate r (r.x, r.P, r.nu, r.S); symmetrise P.  Eigen::MatrixXd::Identity(n, n) with n = x.size().
    throw std::logic_error("TODO: kf_update");
}

std::pair<Eigen::Vector2d, Eigen::Matrix2d> accel_model(const Eigen::Vector2d& x, double g) {
    // TODO B3: write this function (see lab.hpp and the lab handout).  Return {hx, H}.
    throw std::logic_error("TODO: accel_model");
}

EkfResult ekf_tilt(const Eigen::VectorXd& t, const Eigen::VectorXd& gyro, const Eigen::VectorXd& ax,
                   const Eigen::VectorXd& ay, double sigma_gyro, double sigma_bias, double sigma_acc,
                   const std::optional<Eigen::Vector2d>& x0, const std::optional<Eigen::Matrix2d>& P0,
                   std::optional<double> gate, double g) {
    // TODO B4: write this function (see lab.hpp and the lab handout).
    // Size the outputs first (out.theta.resize(N); out.P.resize(N); ...); out.nis starts as NaN:
    // Eigen::VectorXd::Constant(N, std::numeric_limits<double>::quiet_NaN()).  Use kf_predict with
    // B = Eigen::Vector2d(h, 0.0) and u = Eigen::VectorXd::Constant(1, gyro(k)), and accel_model.
    throw std::logic_error("TODO: ekf_tilt");
}

// ==========================================================================================
// Part C -- quaternions and a 3-D explicit complementary filter (Mahony, Hamel & Pflimlin 2008)
// ==========================================================================================
Eigen::Vector4d quat_mult(const Eigen::Vector4d& q, const Eigen::Vector4d& p) {
    // TODO C1: write this function (see lab.hpp and the lab handout).
    throw std::logic_error("TODO: quat_mult");
}

Eigen::Matrix3d quat_to_rot(const Eigen::Vector4d& q) {
    // TODO C2: write this function (see lab.hpp and the lab handout).
    throw std::logic_error("TODO: quat_to_rot");
}

Eigen::Vector4d quat_exp(const Eigen::Vector3d& omega, double h) {
    // TODO C3: write this function (see lab.hpp and the lab handout).
    // Fill a Vector4d with the comma initialiser: q << c, v;  (c a double, v a Vector3d).
    throw std::logic_error("TODO: quat_exp");
}

MahonyResult mahony_filter(const Eigen::VectorXd& t, const Eigen::MatrixXd& gyro3, const Eigen::MatrixXd& acc3,
                           double kp, double ki, const std::optional<Eigen::Vector4d>& q0,
                           const Eigen::Vector3d& up) {
    // TODO C4: write this function (see lab.hpp and the lab handout).
    // Row k of an N x 3 matrix as a Vector3d: Eigen::Vector3d a = acc3.row(k).transpose();
    // store q in row k of out.q with out.q.row(k) = q.transpose();
    throw std::logic_error("TODO: mahony_filter");
}

double thigh_angle_from_quat(const Eigen::Vector4d& q) {
    // TODO C5: write this function (see lab.hpp and the lab handout).  std::copysign, std::hypot, std::atan2.
    throw std::logic_error("TODO: thigh_angle_from_quat");
}

}  // namespace lab
