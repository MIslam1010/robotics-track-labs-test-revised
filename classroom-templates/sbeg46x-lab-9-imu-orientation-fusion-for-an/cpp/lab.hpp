// SBEG 46X Lab 9 -- IMU orientation fusion for an exoskeleton thigh segment: complementary filter vs EKF
// (C++ track): declarations.
//
// The same functions as starter.py (module `estimation` of your course library sbeg46x_medrob), in namespace `lab`,
// with Eigen types instead of NumPy arrays.  Write your code in starter.cpp.  Do not change these declarations:
// test_lab.cpp calls them.
//
// Segment and sensor (course model gait_exo_2r, thigh on the test stand; see course_models.cpp):
//     theta = hip flexion angle (rad), 0 with the leg hanging, positive = flexion (thigh forward).
//     IMU on the thigh cuff, r_imu = 0.20 m below the hip; sensor frame {b} with R_sb = Rot(z, theta):
//     x_b forward (perpendicular to the thigh), y_b up the thigh toward the hip, z_b = hip axis.
//     Gyro:          omega_z = theta_dot + b + noise                    (rad/s)
//     Accelerometer: ax = g sin(theta) + r theta_ddot + noise,  ay = g cos(theta) + r theta_dot^2 + noise   (m/s^2)
//     (specific force f_b = R_sb^T (a - g) in sensor axes: at rest the accelerometer reads +g "up", so
//     theta = atan2(ax, ay)).
// Discrete time: samples k = 0..N-1, period h; the gyro sample k propagates the state from k-1 to k and the
// accelerometer sample k corrects it (as in the lecture notes).
// Kalman filter notation (Astrom & Murray Sec. 7.4, written in predict/update form; Corke Sec. 6.1):
//     x_k = A x_{k-1} + B u_{k-1} + v,  v ~ N(0, Q);     y_k = C x_k + w,  w ~ N(0, R)
//     predict  x- = A x + B u,  P- = A P A^T + Q
//     update   S = C P- C^T + R,  L = P- C^T S^-1,  x = x- + L (y - C x-),  P = (I - L C) P-
//     (gain L as in Astrom & Murray and the lecture notes; many texts write K)
// Quaternions: q = (q0, q1, q2, q3) = (cos(phi/2), w_hat sin(phi/2)) (MR Eq. B.9), product MR Eq. B.15
// (q*p <-> R_q R_p), rotation matrix MR Eq. B.12.  Body-frame gyro rates: R_k = R_{k-1} exp([omega] h)  <->
// q_k = q_{k-1} * exp_q(omega h).
//
// Python -> C++: a dict -> a small struct with the same field names; a tuple -> std::pair (auto [a, b] = ...) or a
// struct (4 values); an argument that defaults to None -> std::optional (std::nullopt = None) or an empty Eigen
// object; a signal of N samples -> Eigen::VectorXd; an (N, 3) or (N, 4) array -> Eigen::MatrixXd with one ROW per
// sample; an (N, 2, 2) array -> std::vector<Eigen::Matrix2d>; a quaternion -> Eigen::Vector4d (q0 first).
// Units: m, rad, s.  All values are illustrative.
#ifndef SBEG46X_LAB9_HPP
#define SBEG46X_LAB9_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lab {

/// Gravity (m/s^2), as G in starter.py.
constexpr double G = 9.81;

// ===========================================================================================================
// Provided (complete): course model (course_models.cpp = course_models.py)
// ===========================================================================================================

/// IMU on the thigh cuff (illustrative): r_imu below the hip axis (m); sensor frame {b} parallel to {s} when the
/// leg hangs (q1 = 0), so R_sb = Rot(z, theta) and at rest f_b = (g sin(theta), g cos(theta), 0).
constexpr double R_IMU = 0.20;

/// Course robot model (Python: the dict returned by gait_exo_2r()).
struct ExoModel {
    std::string key, name;
    Eigen::MatrixXd Slist;                                   ///< 6 x 2 space screw axes, COLUMNS S_i = (w_i, v_i)
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();         ///< home pose of the foot frame
    std::vector<Eigen::Matrix4d> Mlist;                      ///< M_{i-1,i} (MR Ch. 8 form)
    std::vector<Eigen::Matrix<double, 6, 6>> Glist;          ///< spatial inertias G_i (MR Eq. 8.32)
    Eigen::Vector3d g = Eigen::Vector3d::Zero();             ///< gravity in {s} (m/s^2)
    Eigen::MatrixXd limits;                                  ///< 2 x 2 joint limits (rad), row = joint
    Eigen::VectorXd qdot_max, tau_max;                       ///< speed (rad/s) and torque (N m) limits
    double L1 = 0.0, L2 = 0.0, gear_ratio = 0.0, rotor_inertia = 0.0;   ///< Python: params
};

/// SBEG 461 project S2: hip-knee exoskeleton + leg (swing leg on a test stand), leg along -y at q = 0; hip flexion
/// about +z, knee flexion about -z.  (Provided, course_models.cpp.)
ExoModel gait_exo_2r();

/// M_imu: pose of the thigh IMU frame {b} in {s} at q = 0 (on the thigh, r_imu below the hip).  (Provided.)
Eigen::Matrix4d imu_home(double r_imu = R_IMU);

/// T_sb(theta) = e^{[S1] theta} M_imu (MR Eq. 4.14 with the hip joint only).  (Provided.)
Eigen::Matrix4d imu_pose(double theta_hip, double r_imu = R_IMU);

/// Ideal accelerometer reading f_b = R_sb^T (a_s - g_s) of the thigh IMU (no noise).  T_sb comes from mrcpp
/// (fkin_space); for a pure rotation about the fixed hip axis through the {s} origin the IMU acceleration is
/// a_s = alpha x p + w x (w x p) with w = w_1 dtheta, alpha = w_1 ddtheta.  (Provided.)
Eigen::Vector3d imu_specific_force(double theta, double dtheta, double ddtheta, double r_imu = R_IMU);

// ===========================================================================================================
// Provided (complete, helpers.cpp): data loading, error metrics, initial quaternion, heading
// ===========================================================================================================

/// One phase of the recording.
struct Phase {
    std::string name;
    double t0 = 0.0, t1 = 0.0;
};

/// Recording metadata (data/imu_meta.json): sample period h, r_imu, sensor noise specifications, phases.
/// (Python: the dict returned by load_meta(); meta["imu_spec"]["gyro_noise_rad_s"] -> meta.gyro_noise_rad_s.)
struct ImuMeta {
    double h = 0.0, g = 0.0, r_imu = 0.0;
    double gyro_noise_rad_s = 0.0, acc_noise_m_s2 = 0.0, gyro_bias_class_rad_s = 0.0;
    std::vector<Phase> phases;
};

/// Load data/imu_meta.json (empty path) or `path`.  (Provided.)
ImuMeta load_meta(const std::string& path = "");

/// Thigh IMU recording (data/thigh_imu.csv): t (s), gx, gy, gz (rad/s), ax, ay, az (m/s^2) and theta_enc (rad), the
/// hip-joint encoder of the test stand (reference only; NOT an input of the filters).
/// (Python: the dict of arrays returned by load_recording(); rec["gz"] -> rec.gz.)
struct Recording {
    Eigen::VectorXd t, gx, gy, gz, ax, ay, az, theta_enc;
};

/// Load data/thigh_imu.csv (empty path) or `path`.  (Provided.)
Recording load_recording(const std::string& path = "");

/// RMS of (est - ref) in degrees.  (Provided.)
double rms_deg(const Eigen::VectorXd& est, const Eigen::VectorXd& ref);

/// RMS of (est - ref) in degrees over the samples where mask is true.  (Provided.)
double rms_deg(const Eigen::VectorXd& est, const Eigen::VectorXd& ref, const std::vector<bool>& mask);

/// Initial attitude from one accelerometer sample at rest: the rotation that takes the world 'up' axis
/// y_s = (0, 1, 0) onto the measured up direction in the sensor frame (heading set to zero).  Returns q_sb
/// (sensor frame {b} seen from {s}).  (Provided.)
Eigen::Vector4d quat_from_accel(const Eigen::Vector3d& acc);

/// Heading (rotation about the vertical y_s) of the thigh's flexion axis z_i, in rad: 0 when the hip axis is along
/// z_s, positive about +y_s (vertical).  Shows the drift that the accelerometer cannot correct.  (Provided; it
/// calls your quat_to_rot.)
double heading_from_quat(const Eigen::Vector4d& q);

/// accel_tilt for one sample (C++ only; provided: it calls your vector version).
double accel_tilt(double ax, double ay);

// ===========================================================================================================
// Part A -- tilt from the accelerometer, gyro integration and the complementary filter
// ===========================================================================================================

/// A1. Thigh angle from the accelerometer alone, theta = atan2(ax, ay) (rad), sample by sample; exact only at rest.
Eigen::VectorXd accel_tilt(const Eigen::VectorXd& ax, const Eigen::VectorXd& ay);

/// A2. Dead reckoning: theta_0 = theta0, theta_k = theta_{k-1} + (t_k - t_{k-1}) (gyro_k - bias).
Eigen::VectorXd gyro_integrate(const Eigen::VectorXd& t, const Eigen::VectorXd& gyro, double theta0 = 0.0,
                               double bias = 0.0);

/// A3. Blending factor of the discrete first-order complementary filter, alpha = tau / (tau + h).
double cf_alpha(double tau, double h);

/// A4. theta_k = alpha (theta_{k-1} + h (gyro_k - bias)) + (1 - alpha) theta_acc_k, alpha = tau/(tau + h),
/// h = t_k - t_{k-1}.  theta_0 = theta_acc_0 unless theta0 is given.
Eigen::VectorXd complementary_filter(const Eigen::VectorXd& t, const Eigen::VectorXd& gyro,
                                     const Eigen::VectorXd& theta_acc, double tau = 0.5,
                                     std::optional<double> theta0 = std::nullopt, double bias = 0.0);

/// A5. Gyro bias estimated as the mean gyro reading while the segment is still (t < t_end).
double bias_at_rest(const Eigen::VectorXd& t, const Eigen::VectorXd& gyro, double t_end);

// ===========================================================================================================
// Part B -- linear Kalman filter and the EKF with gyro-bias state
// ===========================================================================================================

/// B1. Kalman predict: x- = A x + B u, P- = A P A^T + Q.  Returns {x-, P-}.
/// B (n x m) and u (m) are optional: empty (the defaults) = no input, as B=None / u=None in Python.
std::pair<Eigen::VectorXd, Eigen::MatrixXd> kf_predict(const Eigen::VectorXd& x, const Eigen::MatrixXd& P,
                                                       const Eigen::MatrixXd& A, const Eigen::MatrixXd& Q,
                                                       const Eigen::MatrixXd& B = Eigen::MatrixXd(),
                                                       const Eigen::VectorXd& u = Eigen::VectorXd());

/// Result of kf_update (Python: the tuple (x, P, nu, S)).
struct KfUpdate {
    Eigen::VectorXd x;   ///< updated state
    Eigen::MatrixXd P;   ///< updated covariance (symmetrised: 0.5 (P + P^T))
    Eigen::VectorXd nu;  ///< innovation nu = y - C x-
    Eigen::MatrixXd S;   ///< innovation covariance S = C P- C^T + R
};

/// B2. Kalman update with measurement y = C x + w, w ~ N(0, R).
/// Returns {x, P, nu, S}: updated state and covariance, innovation nu = y - C x- and its covariance S.
KfUpdate kf_update(const Eigen::VectorXd& x, const Eigen::MatrixXd& P, const Eigen::VectorXd& y,
                   const Eigen::MatrixXd& C, const Eigen::MatrixXd& R);

/// B3. EKF measurement model of the accelerometer for the state x = (theta, b):
/// h(x) = (g sin theta, g cos theta) and its Jacobian H = dh/dx (2 x 2).  Returns {h(x), H}.
std::pair<Eigen::Vector2d, Eigen::Matrix2d> accel_model(const Eigen::Vector2d& x, double g = G);

/// Result of ekf_tilt (Python: dict(theta, bias, P, nis, rejected)).
struct EkfResult {
    Eigen::VectorXd theta;            ///< N, estimated angle (rad)
    Eigen::VectorXd bias;             ///< N, estimated gyro bias (rad/s)
    std::vector<Eigen::Matrix2d> P;   ///< N covariances (Python: N x 2 x 2)
    Eigen::VectorXd nis;              ///< N, NIS nu^T S^-1 nu (NaN at k = 0)
    int rejected = 0;                 ///< number of gated (skipped) updates
};

/// B4. EKF for x = (theta, b) with the gyro as input and the 2-axis accelerometer as measurement.
///    predict: theta- = theta + h (gyro_k - b), b- = b;  A = [[1, -h], [0, 1]], B = [h, 0]^T,
///             Q = diag((h sigma_gyro)^2, (h sigma_bias)^2)    (sigma_bias: rate of the slow bias walk, rad/s^2;
///             as in the lecture notes Sec. 10.2)
///    update:  y = (ax, ay), linearised with accel_model at x-;  R = sigma_acc^2 I.
///             If gate is given, the update is skipped when the NIS nu^T S^-1 nu exceeds gate.
/// x0 defaults to (accel tilt of sample 0, 0) and P0 to diag((5 deg)^2, (0.05 rad/s)^2).
/// Returns {theta, bias, P (N matrices 2 x 2), nis (N, NaN at k = 0), rejected (number of gated updates)}.
EkfResult ekf_tilt(const Eigen::VectorXd& t, const Eigen::VectorXd& gyro, const Eigen::VectorXd& ax,
                   const Eigen::VectorXd& ay, double sigma_gyro, double sigma_bias, double sigma_acc,
                   const std::optional<Eigen::Vector2d>& x0 = std::nullopt,
                   const std::optional<Eigen::Matrix2d>& P0 = std::nullopt,
                   std::optional<double> gate = std::nullopt, double g = G);

// ===========================================================================================================
// Part C -- quaternions and a 3-D explicit complementary filter (Mahony, Hamel & Pflimlin 2008)
// ===========================================================================================================

/// C1. Quaternion product n = q * p of MR Eq. B.15 (the quaternion of R_q R_p).
Eigen::Vector4d quat_mult(const Eigen::Vector4d& q, const Eigen::Vector4d& p);

/// C2. Rotation matrix of a unit quaternion, MR Eq. B.12 (q is normalised first).
Eigen::Matrix3d quat_to_rot(const Eigen::Vector4d& q);

/// C3. Quaternion of the rotation exp([omega] h): (cos(|omega| h / 2), omega_hat sin(|omega| h / 2)) (MR Eq. B.9).
/// For |omega| h < 1e-12 return (1, 0.5 h omega).
Eigen::Vector4d quat_exp(const Eigen::Vector3d& omega, double h);

/// Result of mahony_filter (Python: dict(q, bias)).
struct MahonyResult {
    Eigen::MatrixXd q;     ///< N x 4, q_sb (one row per sample)
    Eigen::MatrixXd bias;  ///< N x 3, gyro-bias estimate (rad/s)
};

/// C4. Explicit complementary filter on SO(3) with gyro-bias estimation (Mahony, Hamel & Pflimlin 2008):
///     v_hat = R_hat^T up             (predicted 'up' direction in the sensor frame)
///     v     = acc / |acc|            (measured 'up' direction: the accelerometer reads +g up at rest)
///     e     = v x v_hat              (correction rate, rad/s)
///     q_k   = q_{k-1} * exp_q((gyro_k - b_hat + kp e) h),   b_hat <- b_hat - ki e h
/// with v from the accelerometer sample k and v_hat from q_{k-1}; q is renormalised after every step.
/// gyro3 and acc3 are N x 3 (one row per sample).  q0 defaults to quat_from_accel(acc3 row 0).
/// Returns {q (N x 4, q_sb), bias (N x 3)}.
MahonyResult mahony_filter(const Eigen::VectorXd& t, const Eigen::MatrixXd& gyro3, const Eigen::MatrixXd& acc3,
                           double kp = 1.0, double ki = 0.1,
                           const std::optional<Eigen::Vector4d>& q0 = std::nullopt,
                           const Eigen::Vector3d& up = Eigen::Vector3d(0.0, 1.0, 0.0));

/// C5. Signed thigh inclination from the estimated attitude q_sb: the thigh axis pointing to the knee is
/// d = -R_sb y_b (second column of R, negated); theta = atan2(sign(d_x) sqrt(d_x^2 + d_z^2), -d_y).
/// Uses only the tilt of the thigh, so it is not affected by the (unobservable) heading about the vertical.
double thigh_angle_from_quat(const Eigen::Vector4d& q);

}  // namespace lab

#endif  // SBEG46X_LAB9_HPP
