// SBEG 46X Lab 11 -- image-based visual servoing of an endoscope-holding arm with a simulated pinhole camera
// (formal report 3) -- C++ track: declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp. Do not change these declarations: test_lab.cpp calls them. The provided helpers
// (scene, poses, data, camera-matrix decomposition, endoscope simulator, metrics) are given in helpers.cpp.
//
// Models (all values illustrative, data/scene.json):
//   Pinhole camera (Corke Sec. 11.1, Eqs. 11.2 and 11.7-11.10, pp. 252-257): camera frame {c}, optical axis z_c,
//   x_c to the right and y_c down in the image; K = [[f/rho_w, 0, u0], [0, f/rho_h, v0], [0, 0, 1]];
//   a point P_c = (X, Y, Z) in {c} (Z > 0) has pixels (u, v) = (u0 + f/rho_w X/Z, v0 + f/rho_h Y/Z).
//   Camera poses are T_wc (camera in the world, as T_sb in MR); P_c = T_wc^-1 P_w.
//   Camera matrix (Corke Eq. 11.9): C = K [R_cw | p_cw] (3 x 4); DLT calibration (Corke Sec. 11.2.1,
//   Eqs. 11.15-11.16, pp. 262-263), solved by SVD and scaled so that ||C[2, :3]|| = 1.
//   Camera twist: the BODY twist of {c}, V_c = (omega, v), MR order (MR Eq. 3.70, p. 96). Corke writes (v, omega):
//   the interaction matrix here is Corke's Eq. 15.6 (p. 461) with its two 2 x 3 blocks swapped. With
//   x = (u - u0)/f_u, y = (v - v0)/f_v:
//       L = [ f_u (x y)       -f_u (1 + x^2)   f_u y    -f_u/Z      0       f_u x/Z ]
//           [ f_v (1 + y^2)   -f_v x y          -f_v x    0       -f_v/Z    f_v y/Z ]
//   IBVS (Corke Sec. 15.2.2, Eqs. 15.8-15.11): e = p - p* (u1, v1, u2, v2, ...), V_c = -lambda L^+ e,
//   T_wc <- T_wc exp([V_c] h) (MR Eq. 3.88, p. 103).
//   PBVS (Corke Sec. 15.1): T_ct from the image (homography, Corke Sec. 14.2.4, Eq. 14.12, then H ~ K [r1 r2 t]);
//   V_c = lambda log(T_ct T_c*t^-1) (MR Sec. 3.3.3.2, p. 104).
//   Endoscope through a port: port at r = (0, 0, -ell) in {c}; allowed twists V_c = W(ell) xi,
//   xi = (omega_x, omega_y, omega_z, d_dot), v = (ell omega_y, -ell omega_x, d_dot); joint rates by MR Eq. 6.7 with
//   the body Jacobian (camera {c} = end-effector {b} at the scope tip).
//
// Python -> C++: (N, 2) / (N, 3) arrays -> Eigen::MatrixXd with one ROW per point; a tuple of results -> std::pair;
// the run dict of run_ibvs / run_pbvs -> lab::ServoRun; None -> std::nullopt (std::optional); a scalar depth Z_hat
// -> Eigen::VectorXd::Constant(N, Z_hat); the scene dict -> lab::Scene with the same field names.
#ifndef SBEG46X_LAB11_HPP
#define SBEG46X_LAB11_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include "course_models.hpp"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lab {

using Vector6d = sbeg::Vector6d;
using Matrix34d = Eigen::Matrix<double, 3, 4>;
using Matrix64d = Eigen::Matrix<double, 6, 4>;

// ==========================================================================================
// Provided (given in helpers.cpp, do not edit)
// ==========================================================================================

/// One camera entry of scene.json.
struct Camera {
    double f = 0, rho_w = 0, rho_h = 0, u0 = 0, v0 = 0;
    int W = 0, H = 0;
};

/// Lab parameters (data/scene.json, illustrative values), field names as the JSON keys; poses are 4 x 4 matrices.
struct Scene {
    Camera inspection_camera, endoscope_camera;
    double target_side = 0;
    Eigen::Matrix4d T_wt = Eigen::Matrix4d::Identity();
    struct Ibvs {
        double lam = 0, h = 0, tol = 0, Z_star = 0;
        int n_max = 0;
        Eigen::Matrix4d T_wc_star = Eigen::Matrix4d::Identity();
    } ibvs;
    struct StartPoses {
        Eigen::Matrix4d corke = Eigen::Matrix4d::Identity(), fov = Eigen::Matrix4d::Identity();
        std::vector<double> roll_angles;
    } start_poses;
    struct Calib {
        double sigma_px = 0;
        int seed = 0;
        Eigen::Matrix4d T_wc = Eigen::Matrix4d::Identity();
    } calib;
    struct Endoscope {
        Eigen::VectorXd theta0;
        double ell0 = 0, Z0 = 0, Z_des = 0, a = 0, dz = 0, f_hz = 0, t_still = 0, T_end = 0, h = 0, lam = 0;
        int delay = 0;
        double k_z = 0, k_rcm = 0, ell_min = 0;
    } endoscope;
};

/// Load data/scene.json (default path: the lab's data folder).
Scene load_scene(const std::string& path = "");
/// Pure translation T = (I, p).
Eigen::Matrix4d trans(const Eigen::Vector3d& p);
/// Rotation about z by a (4 x 4).
Eigen::Matrix4d rotz(double a);
/// Intrinsic matrix of a camera entry (calls your intrinsic_matrix).
Eigen::Matrix3d camera_K(const Camera& cam);

/// Square planar target of side s in the plane z_t = 0 of {t}: P_w (4 x 3) world points and xy_t (4 x 2)
/// target-plane coordinates, order (-s/2, -s/2), (-s/2, s/2), (s/2, s/2), (s/2, -s/2).
struct Target {
    Eigen::MatrixXd P_w, xy_t;
};
Target target_points(const Scene& scene);

/// Calibration data (data/calib_points.csv): first = P (N x 3) points of the calibration object, second = p (N x 2)
/// measured pixels (noisy).
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> load_calib(const std::string& path = "");

/// Split C = K [R | p] (scaled as camera_matrix_dlt returns it) into first = K (K(2, 2) = 1) and second = T_cw,
/// by Gram-Schmidt from the third row (an RQ decomposition, Corke Sec. 11.2.2).
std::pair<Eigen::Matrix3d, Eigen::Matrix4d> decompose_camera_matrix(const Matrix34d& C);

/// interaction_matrix with the same depth Z for all points (calls your function).
Eigen::MatrixXd interaction_matrix(const Eigen::MatrixXd& p, double Z, const Eigen::Matrix3d& K);

/// Result of run_ibvs / run_pbvs (Python: the returned dict). n = n_iter velocity commands.
struct ServoRun {
    std::vector<Eigen::Matrix4d> T;   ///< n + 1 camera poses T_wc
    std::vector<Eigen::MatrixXd> p;   ///< n + 1 feature sets (N x 2)
    Eigen::MatrixXd Z;                ///< (n + 1) x N depths
    Eigen::VectorXd err;              ///< n + 1 rms image errors (px)
    Eigen::MatrixXd V;                ///< n x 6 camera twists (one per ROW)
    Eigen::VectorXd cond;             ///< n condition numbers (run_ibvs only; empty for run_pbvs)
    bool converged = false;
    int n_iter = 0;
};

/// servo_metrics of a run: n_iter, converged, final_err (px), path_len (m), retreat (m), min_margin (px),
/// straightness (1 = straight image paths).
struct ServoMetrics {
    int n_iter = 0;
    bool converged = false;
    double final_err = 0, path_len = 0, retreat = 0, min_margin = 0, straightness = 0;
};
ServoMetrics servo_metrics(const ServoRun& run, const Camera& cam);

/// Start of the endoscope experiments (Python: endoscope_setup).
struct EndoSetup {
    cm::Robot robot;
    Eigen::VectorXd theta0;
    Eigen::Matrix4d T0;
    Eigen::Vector3d port, centre;
    Eigen::Matrix3d K;
};
EndoSetup endoscope_setup(const Scene& scene);

/// World position of the instrument tip at time t.
Eigen::Vector3d instrument_tip(const Scene& scene, const EndoSetup& setup, double t);

/// Options of simulate_endoscope (Python keyword arguments; unset = value of scene.endoscope).
struct EndoOptions {
    std::optional<double> lam;
    std::optional<int> delay;
    std::optional<double> Z_hat, k_z, k_rcm, T_end;
};

/// Result of simulate_endoscope (rows = samples; theta_dot, xi, V, sat have one row fewer than t).
struct EndoSim {
    Eigen::VectorXd t, Z, err, ell, e_lat;
    Eigen::MatrixXd p, theta, theta_dot, xi, V;
    std::vector<bool> sat, in_view;
};
/// Closed loop of the endoscope holder (calls your project_points, rcm_errors, rcm_ibvs_velocity, joint_rates).
EndoSim simulate_endoscope(const Scene& scene, const EndoOptions& opt = EndoOptions());

struct EndoMetrics {
    double rms_err = 0, max_err = 0, final_still_err = 0, max_e_lat = 0, ell_min = 0, ell_max = 0, peak_qdot = 0,
           sat_frac = 0, view_frac = 0;
};
EndoMetrics endoscope_metrics(const EndoSim& sim, double t_from = 5.0);

// ==========================================================================================
// Part A -- pinhole camera and calibration (YOUR CODE, starter.cpp)
// ==========================================================================================

/// A1. Intrinsic matrix K = [[f/rho_w, 0, u0], [0, f/rho_h, v0], [0, 0, 1]] (Corke Eq. 11.7, f folded in).
Eigen::Matrix3d intrinsic_matrix(double f, double rho_w, double rho_h, double u0, double v0);

/// A2. Project world points P_w (N x 3, one point per ROW) into the camera at pose T_wc. Returns first = p (N x 2)
/// pixels (u, v), second = Z (N) depths in {c}. Throw std::invalid_argument if any Z <= 0.
std::pair<Eigen::MatrixXd, Eigen::VectorXd> project_points(const Eigen::Matrix3d& K, const Eigen::Matrix4d& T_wc,
                                                           const Eigen::MatrixXd& P_w);

/// A3. DLT camera matrix C (3 x 4) with p ~ C [P; 1] from N >= 6 non-coplanar points (Corke Eqs. 11.15-11.16):
/// two rows per point, right singular vector of the smallest singular value, scaled so that ||C[2, :3]|| = 1 and
/// the mean of C[2] . [P; 1] is positive. Throw std::invalid_argument if N < 6.
Matrix34d camera_matrix_dlt(const Eigen::MatrixXd& P_w, const Eigen::MatrixXd& p);

/// A4. Root-mean-square reprojection error (pixels) of C on the points P_w with measured pixels p.
double reprojection_rms(const Matrix34d& C, const Eigen::MatrixXd& P_w, const Eigen::MatrixXd& p);

// ==========================================================================================
// Part B -- interaction matrix
// ==========================================================================================

/// B1. Stacked interaction matrix (2N x 6), rows u1, v1, u2, v2, ..., columns in MR twist order
/// (omega_x, omega_y, omega_z, v_x, v_y, v_z). p: N x 2 pixels; Z: N depths.
Eigen::MatrixXd interaction_matrix(const Eigen::MatrixXd& p, const Eigen::VectorXd& Z, const Eigen::Matrix3d& K);

// ==========================================================================================
// Part C -- image-based visual servoing (free-flying camera)
// ==========================================================================================

/// C1. IBVS law V_c = -lam L^+ (p - p*), L = interaction_matrix(p, Z_hat, K), error stacked (u1 - u1*, v1 - v1*, ...).
Vector6d ibvs_velocity(const Eigen::MatrixXd& p, const Eigen::MatrixXd& p_star, const Eigen::VectorXd& Z_hat,
                       const Eigen::Matrix3d& K, double lam);

/// C2. T_wc exp([V_c] h) (MR Eq. 3.88).
Eigen::Matrix4d camera_step(const Eigen::Matrix4d& T_wc, const Vector6d& V_c, double h);

/// C3. IBVS loop: for k = 0, 1, ...: project; err_k = sqrt(mean_i ||p_ik - p_i*||^2); stop if err_k <= tol
/// (converged) or k == n_max; otherwise V_k = ibvs_velocity(p_k, p*, Z_used, K_model, lam) with Z_used = Z_k (true
/// depths) if Z_hat is unset, else *Z_hat; K_model = K_hat if set, else K; cond_k = 2-norm condition number of the
/// interaction matrix used (ratio of its largest to smallest singular value); T_{k+1} = camera_step(T_k, V_k, h).
ServoRun run_ibvs(const Eigen::Matrix4d& T_wc0, const Eigen::MatrixXd& P_w, const Eigen::MatrixXd& p_star,
                  const Eigen::Matrix3d& K, double lam, double h,
                  const std::optional<Eigen::VectorXd>& Z_hat = std::nullopt, int n_max = 400, double tol = 0.5,
                  const std::optional<Eigen::Matrix3d>& K_hat = std::nullopt);

// ==========================================================================================
// Part D -- pose from the image and position-based visual servoing
// ==========================================================================================

/// D1. Homography with p ~ H [x; y; 1] from N >= 4 target-plane points xy_t (N x 2) and pixels p (N x 2): DLT, SVD
/// null vector, scaled so that H(2, 2) = 1. Throw std::invalid_argument if N < 4.
Eigen::Matrix3d homography_dlt(const Eigen::MatrixXd& xy_t, const Eigen::MatrixXd& p);

/// D2. Pose T_ct of the target plane from H ~ K [r1 r2 t]: [m1 m2 m3] = K^-1 H, s = 2 / (||m1|| + ||m2||) with the
/// sign that makes s m3_z > 0; r1 = s m1, r2 = s m2, r3 = r1 x r2; R = closest rotation (SVD: U diag(1, 1,
/// det(U V^T)) V^T); t = s m3.
Eigen::Matrix4d pose_from_homography(const Eigen::Matrix3d& H, const Eigen::Matrix3d& K);

/// D3. PBVS law V_c = lam * se3_to_vec(matrix_log6(T_ct T_c*t^-1)).
Vector6d pbvs_velocity(const Eigen::Matrix4d& T_ct, const Eigen::Matrix4d& T_cstar_t, double lam);

/// D4. PBVS loop with the structure, stopping rule and result of run_ibvs (cond left empty): T_ct =
/// pose_from_homography(homography_dlt(xy_t, p_k), K_model), V_k = pbvs_velocity(T_ct, T_cstar_t, lam).
ServoRun run_pbvs(const Eigen::Matrix4d& T_wc0, const Eigen::MatrixXd& P_w, const Eigen::MatrixXd& xy_t,
                  const Eigen::Matrix4d& T_cstar_t, const Eigen::MatrixXd& p_star, const Eigen::Matrix3d& K,
                  double lam, double h, int n_max = 400, double tol = 0.5,
                  const std::optional<Eigen::Matrix3d>& K_hat = std::nullopt);

// ==========================================================================================
// Part E -- the endoscope through a port
// ==========================================================================================

/// E1. W(ell) (6 x 4): V_c = W xi, xi = (omega_x, omega_y, omega_z, d_dot), v = (ell omega_y, -ell omega_x, d_dot).
Matrix64d rcm_twist_basis(double ell);

/// E2. Pan/tilt IBVS through the port: A = interaction_matrix(p, Z_hat, K) W(ell); omega_z = 0; d_dot given;
/// (omega_x, omega_y) = A[:, 0:2]^+ (-lam e - A[:, 3] d_dot). Returns first = xi (4), second = V_c = W xi.
std::pair<Eigen::Vector4d, Vector6d> rcm_ibvs_velocity(const Eigen::MatrixXd& p, const Eigen::MatrixXd& p_star,
                                                       const Eigen::VectorXd& Z_hat, const Eigen::Matrix3d& K,
                                                       double lam, double ell, double d_dot = 0.0);

/// E3. theta_dot = J_b(theta)^+ V_c (MR Eq. 6.7; mr::jacobian_body, mr::pinv); if qdot_max is set and the largest
/// |theta_dot_i| exceeds it, scale the whole vector by one factor so that the largest equals qdot_max.
Eigen::VectorXd joint_rates(const Eigen::MatrixXd& Blist, const Eigen::VectorXd& theta, const Vector6d& V_c,
                            std::optional<double> qdot_max = std::nullopt);

/// E4. p_c = T_sc^-1 p_port; returns first = ell = -p_c(2) (insertion depth beyond the port, m), second = e =
/// (p_c(0), p_c(1)) (lateral offset of the port from the scope axis, m).
std::pair<double, Eigen::Vector2d> rcm_errors(const Eigen::Matrix4d& T_sc, const Eigen::Vector3d& p_port);

}  // namespace lab

#endif  // SBEG46X_LAB11_HPP
