// SBEG 46X Lab 5 -- Navigation pipeline: pivot calibration of a digitising pointer on the course arm, paired-point
// rigid registration of a phantom (SVD), and navigation accuracy (FLE, FRE, TRE): prediction vs Monte Carlo vs
// measurement (C++ track): declarations.
//
// The same functions as starter.py (module `registration` of your course library sbeg46x_medrob), in namespace
// `lab`, with Eigen types instead of NumPy arrays. Write your code in starter.cpp (A1 ... D2). Do not change these
// declarations: test_lab.cpp calls them. The provided helpers are in helpers.cpp, the course arm in course_models.hpp.
//
// Notation and units
//   * Positions in MILLIMETRES, angles in radians, rotation matrices R in SO(3), homogeneous transforms T in SE(3)
//     (MR Sec. 3.3.1, pp. 87-95).  T_ab is the pose of frame {b} in frame {a}; p_a = T_ab p_b.
//   * Frames: {s} robot base (course arm), {b} flange, {I} image (CT) of the phantom, {C} optical-tracker camera,
//     {P} patient reference marker.
//   * Pivot calibration: the pointer tip is fixed at the pivot while the flange takes poses (R_i, p_i):
//         R_i p_tip + p_i = p_piv           (p_tip in {b}, p_piv in {s})
//   * Registration: fiducials x_k in {I} and the same fiducials y_k in {s};  y_k ~ R x_k + t,  T_sI = (R, t).
//   * FRE = RMS residual over the fiducials; FLE = fiducial localisation error (RMS magnitude);
//     TRE = error at a target that was NOT used in the registration.
//
// Python -> C++: an N x 3 point array -> Eigen::MatrixXd with one ROW per point; a tuple -> a small struct (fields
// named as in Python) or std::pair; the result dict of navigation_pipeline -> lab::PipelineResult;
// np.random.default_rng(seed) -> lab::Rng (helpers.cpp); itertools.combinations -> lab::combinations (helpers.cpp).
#ifndef SBEG46X_LAB5_HPP
#define SBEG46X_LAB5_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include "course_models.hpp"

#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace lab {

// ==========================================================================================
// Provided (given in helpers.cpp, do not edit): data files and small helpers
// ==========================================================================================

/// Phantom description from data/phantom.json (illustrative values).
struct Phantom {
    std::string description;
    Eigen::Vector3d ct_voxel_mm = Eigen::Vector3d::Zero();
    Eigen::MatrixXd fiducials_image_mm;                       ///< 6 x 3 fiducial divots in {I} (one row each)
    Eigen::MatrixXd check_points_image_mm;                    ///< 2 x 3 check divots in {I}
    std::map<std::string, Eigen::Vector3d> targets_image_mm;  ///< "T1_central", "T2_lateral", "T3_deep" in {I}
    std::string pivot_divot;
    double pointer_nominal_length_mm = 0.0;
};

/// The phantom description of data/phantom.json.
Phantom load_phantom();

/// A joint log of data/ (columns: label, q1 ... q6 in rad). Returns (labels, N x 6 matrix).
std::pair<std::vector<std::string>, Eigen::MatrixXd> load_joints(const std::string& name);

/// 4 x 4 homogeneous transform from R (3 x 3) and t (3,)  (MR Eq. 3.62).
Eigen::Matrix4d rot_to_T(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);

/// Random numbers for the Monte-Carlo simulation (C++ stand-in for np.random.default_rng(seed)):
/// the same seed always gives the same numbers (on another compiler at most the last digits may differ, as
/// std::log/std::sin/std::cos may round differently). The numbers differ from NumPy's.
class Rng {
public:
    explicit Rng(std::uint64_t seed = 0);
    /// One Gaussian sample with mean `loc` and standard deviation `scale`  (rng.normal(loc, scale)).
    double normal(double loc = 0.0, double scale = 1.0);
    /// rows x cols Gaussian samples, filled row by row  (rng.normal(loc, scale, (rows, cols))).
    Eigen::MatrixXd normal(double loc, double scale, Eigen::Index rows, Eigen::Index cols);

private:
    std::mt19937_64 gen_;
    bool has_spare_ = false;
    double spare_ = 0.0;
};

/// All k-element subsets of {0, ..., n-1} as sorted index lists, in itertools.combinations order
/// (lexicographic: {0,1,2}, {0,1,3}, ...).
std::vector<std::vector<int>> combinations(int n, int k);

// ==========================================================================================
// Part A -- tool calibration with the course arm
// ==========================================================================================

/// A1. Pose T_sb of the flange frame {b} in the base frame {s} for joint angles q (rad, 6 values), with the
/// position in MILLIMETRES. Use the space-form PoE of the course arm (MR Eq. 4.14) from course_models.hpp.
Eigen::Matrix4d flange_pose_mm(const Eigen::VectorXd& q);

/// Result of a pivot calibration (Python: the tuple (p_tip, p_piv, rms)).
struct PivotResult {
    Eigen::Vector3d p_tip = Eigen::Vector3d::Zero();  ///< tip offset in the flange frame {b} (mm)
    Eigen::Vector3d p_piv = Eigen::Vector3d::Zero();  ///< pivot point in the base frame {s} (mm)
    double rms = 0.0;       ///< RMS length of the 3-D residuals (mm)
};

/// A2. Pivot calibration by linear least squares.
///
/// Rs: K rotation matrices R_i of the tool (flange) frame; ps: K positions p_i (mm).
/// Every pose satisfies R_i p_tip + p_i = p_piv; stack them as [R_i  -I] [p_tip; p_piv] = -p_i.
/// Returns {p_tip, p_piv, rms} where rms is the RMS length of the 3-D residuals (mm).
/// Throws std::invalid_argument if Rs and ps have different lengths.
PivotResult pivot_calibration(const std::vector<Eigen::Matrix3d>& Rs, const std::vector<Eigen::Vector3d>& ps);

/// A3. Pivot calibration from a joint log Q (K x 6, rad) recorded while the pointer tip stays in the pivot divot.
/// Returns the same result as pivot_calibration.
PivotResult pivot_from_joint_log(const Eigen::MatrixXd& Q);

/// A4. Position (mm) of the pointer tip in {s} for joint angles q and the calibrated offset p_tip (mm, in {b}).
Eigen::Vector3d tip_position_mm(const Eigen::VectorXd& q, const Eigen::Vector3d& p_tip);

// ==========================================================================================
// Part B -- paired-point rigid registration
// ==========================================================================================

/// B1. Least-squares rigid registration (Arun, Huang & Blostein 1987, with the reflection fix).
///
/// X, Y: N x 3 corresponding points (one per row; N >= 3, not collinear). Finds R in SO(3) and t minimising
/// sum_k |y_k - (R x_k + t)|^2 and returns T = (R, t) as a 4 x 4 matrix (so T maps X-coordinates to Y-coordinates).
/// Steps: centroids, H = sum x~_k y~_k^T, H = U S V^T, R = V diag(1, 1, det(V U^T)) U^T, t = y_bar - R x_bar.
/// Throws std::invalid_argument if X and Y are not both N x 3 with N >= 3.
Eigen::Matrix4d register_points(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y);

/// B2. Apply a homogeneous transform to one point p (3,); returns T p (3,).
Eigen::Vector3d apply_T(const Eigen::Matrix4d& T, const Eigen::Vector3d& p);

/// B2. Apply a homogeneous transform to N points (N x 3, one per row); returns N x 3.
/// Throws std::invalid_argument if P does not have 3 columns.
Eigen::MatrixXd apply_T(const Eigen::Matrix4d& T, const Eigen::MatrixXd& P);

/// B3. Fiducial registration error: RMS over the N fiducials of |y_k - T x_k| (mm).
double fre(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, const Eigen::Matrix4d& T);

// ==========================================================================================
// Part C -- error prediction (Fitzpatrick, West & Maurer 1998) and Monte Carlo
// ==========================================================================================

/// C1. Expected RMS FRE for N fiducials with isotropic FLE of RMS magnitude fle_rms:
/// <FRE^2> = (1 - 2/N) <FLE^2>.
double fre_expected(int N, double fle_rms);

/// C2. Estimate the RMS FLE from one observed FRE by inverting C1 (a rough estimate from a single registration).
double fle_from_fre(double fre_value, int N);

/// C3. Predicted RMS TRE at `target` (Fitzpatrick, West & Maurer 1998, isotropic FLE):
///
///     <TRE^2(r)> = <FLE^2>/N * (1 + 1/3 * sum_{k=1..3} d_k^2 / f_k^2)
///
/// with the principal axes of the fiducial configuration (eigenvectors of sum x~ x~^T, x~ = x - x_bar),
/// d_k = distance of the target from principal axis k, f_k = RMS distance of the fiducials from axis k.
double tre_predict(const Eigen::MatrixXd& X, const Eigen::Vector3d& target, double fle_rms);

/// Result of monte_carlo (Python: the tuple (fre_samples, tre_samples)).
struct MonteCarloResult {
    Eigen::VectorXd fre_samples;  ///< n FRE values
    Eigen::MatrixXd tre_samples;  ///< n x M TRE values (column j = target j)
};

/// C4. Monte-Carlo simulation of registration error.
///
/// In each of n trials, add isotropic Gaussian noise of RMS magnitude fle_rms (std fle_rms/sqrt(3) per axis) to
/// the measured copy: Y = X + noise; register X -> Y with register_points; record the FRE and, for every
/// target r, the TRE |T r - r| (the true transform is the identity, so any motion of r is error).
/// targets: M x 3 (one target per row; one target = a 1 x 3 matrix). Returns {fre_samples (n), tre_samples (n x M)}.
/// Use lab::Rng rng(seed) (C++ stand-in for np.random.default_rng(seed)).
MonteCarloResult monte_carlo(const Eigen::MatrixXd& X, const Eigen::MatrixXd& targets, double fle_rms, int n = 2000,
                             std::uint64_t seed = 0);

/// C5. Fiducial configuration: among all subsets of k fiducials (k >= 3) of X, find the one with the smallest
/// predicted TRE at `target`. Returns (indices as a sorted list, predicted TRE). Ties: the first in
/// combinations order (lab::combinations = itertools.combinations).
std::pair<std::vector<int>, double> best_subset(const Eigen::MatrixXd& X, const Eigen::Vector3d& target, int k,
                                                double fle_rms);

// ==========================================================================================
// Part D -- frame chains and the full pipeline
// ==========================================================================================

/// D1. Optical-tracker chain: camera {C} sees the robot-base marker (T_CR = pose of {s} in {C}) and the patient
/// reference (T_CP); registration gives T_PI (image in patient-reference coordinates).
/// Return T_RI = T_CR^-1 T_CP T_PI, the image frame in robot-base coordinates.
Eigen::Matrix4d chain_robot_from_image(const Eigen::Matrix4d& T_CR, const Eigen::Matrix4d& T_CP,
                                       const Eigen::Matrix4d& T_PI);

/// Result of navigation_pipeline (Python: the result dict; same keys as fields).
struct PipelineResult {
    Eigen::Vector3d p_tip = Eigen::Vector3d::Zero();   ///< A3 on data/pivot_joints.csv
    Eigen::Vector3d p_piv = Eigen::Vector3d::Zero();
    double pivot_rms = 0.0;
    Eigen::MatrixXd Y_fid;                             ///< N x 3 digitised fiducials in {s} (A4, fiducial_joints.csv)
    Eigen::Matrix4d T_sI = Eigen::Matrix4d::Zero();    ///< B1: image fiducials -> digitised fiducials
    double fre = 0.0;                                  ///< B3
    double fle_est = 0.0;                              ///< C2
    std::map<std::string, double> tre_pred;            ///< C3 at every target of phantom.json, with fle_est
    std::map<std::string, double> tre_mc;              ///< RMS TRE of C4 (fle_est, n_mc trials, seed)
    double fre_mc_rms = 0.0;                           ///< RMS of the C4 FRE samples
    std::map<std::string, Eigen::Vector3d> targets_s;  ///< target positions in {s}
    Eigen::VectorXd check_err;                         ///< M distances |T_sI c_I - c_s| at the check points
};

/// D2. Run the whole pipeline on the data files and return a PipelineResult with
///     p_tip, p_piv, pivot_rms                     (A3 on data/pivot_joints.csv)
///     Y_fid                                       (N x 3 digitised fiducials in {s}, A4 on data/fiducial_joints.csv)
///     T_sI, fre                                   (B1, B3: image fiducials -> digitised fiducials)
///     fle_est                                     (C2)
///     tre_pred   {name: predicted TRE}            (C3 at every target of phantom.json, with fle_est)
///     tre_mc     {name: RMS TRE of C4}            (C4 with fle_est, n_mc trials, seed; targets in name order)
///     fre_mc_rms                                  (RMS of the C4 FRE samples)
///     targets_s  {name: target position in {s}}
///     check_err  (M) distances |T_sI c_I - c_s| at the check points (digitised with data/check_joints.csv)
PipelineResult navigation_pipeline(int n_mc = 2000, std::uint64_t seed = 0);

}  // namespace lab

#endif  // SBEG46X_LAB5_HPP
