// SBEG 46X Lab 5 -- Navigation pipeline: pivot calibration, paired-point registration, FLE/FRE/TRE (C++ STARTER).
//
// Fill in every function marked  // TODO  (in the order A1 ... D2; replace the `throw std::logic_error("TODO: ...")`
// line). Do not change function names or signatures (they are declared in lab.hpp). Build and test from the cpp folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
// or from your Lab code folder:   python run_lab.py 5 --cpp
//
// These functions become the `registration` module of your cumulative library sbeg46x_medrob.
// Notation and units: see lab.hpp (positions in MILLIMETRES; T_ab = pose of {b} in {a}).
// Given: helpers.cpp (load_phantom, load_joints, rot_to_T, lab::Rng, lab::combinations) and course_models.hpp
// (cm::course_arm_kinematics(), metres). You may use mrcpp (the C++ port of modern_robotics: mr::fkin_space,
// mr::trans_inv, mr::rp_to_trans, ...) where starter.py may use modern_robotics.
//
// Eigen reminders:  points are ROWS of an N x 3 Eigen::MatrixXd;  X.colwise().mean() (1 x 3 centroid),
//   X.rowwise() - xb (subtract a row from every row), X.row(i).transpose() (a point as a column vector),
//   T.topLeftCorner<3, 3>() (R), T.block<3, 1>(0, 3) (p), A.completeOrthogonalDecomposition().solve(b) (lstsq),
//   Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV) -> svd.matrixU(),
//   svd.matrixV() (V itself, not V^T), Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(S) -> es.eigenvectors()
//   (as np.linalg.eigh), M.rowwise().norm() / .squaredNorm() (one value per row), v.mean().
#include "lab.hpp"

#include <sbeg/mrcpp.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace lab {

// ==========================================================================================
// Part A -- tool calibration with the course arm
// ==========================================================================================

Eigen::Matrix4d flange_pose_mm(const Eigen::VectorXd& q) {
    // TODO A1: cm::course_arm_kinematics() gives Slist and M (metres); mr::fkin_space(M, Slist, q);
    //          then scale ONLY the position column to millimetres.
    throw std::logic_error("TODO: flange_pose_mm");
}

PivotResult pivot_calibration(const std::vector<Eigen::Matrix3d>& Rs, const std::vector<Eigen::Vector3d>& ps) {
    // TODO A2: throw std::invalid_argument if Rs.size() != ps.size().
    //          Stack A (3K x 6) = [R_i  -I] and b (3K) = -p_i, solve in the least-squares sense,
    //          rms = sqrt(mean over the K poses of |3-D residual|^2).
    throw std::logic_error("TODO: pivot_calibration");
}

PivotResult pivot_from_joint_log(const Eigen::MatrixXd& Q) {
    // TODO A3: one flange pose per row of Q (Q.row(i).transpose()), then pivot_calibration.
    throw std::logic_error("TODO: pivot_from_joint_log");
}

Eigen::Vector3d tip_position_mm(const Eigen::VectorXd& q, const Eigen::Vector3d& p_tip) {
    // TODO A4
    throw std::logic_error("TODO: tip_position_mm");
}

// ==========================================================================================
// Part B -- paired-point rigid registration
// ==========================================================================================

Eigen::Matrix4d register_points(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) {
    // TODO B1: throw std::invalid_argument unless X and Y are both N x 3 with N >= 3.
    //          Centroids, H = sum x~ y~^T (3 x 3), SVD, R = V diag(1, 1, sign(det(V U^T))) U^T, t = y_bar - R x_bar;
    //          return rot_to_T(R, t). (np.sign(0) == 0: keep that case.)
    throw std::logic_error("TODO: register_points");
}

Eigen::Vector3d apply_T(const Eigen::Matrix4d& T, const Eigen::Vector3d& p) {
    // TODO B2 (one point)
    throw std::logic_error("TODO: apply_T");
}

Eigen::MatrixXd apply_T(const Eigen::Matrix4d& T, const Eigen::MatrixXd& P) {
    // TODO B2 (N x 3 points, one per row): throw std::invalid_argument if P.cols() != 3.
    //          Tip: P * R^T, then add t^T to every row.
    throw std::logic_error("TODO: apply_T");
}

double fre(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, const Eigen::Matrix4d& T) {
    // TODO B3
    throw std::logic_error("TODO: fre");
}

// ==========================================================================================
// Part C -- error prediction (Fitzpatrick, West & Maurer 1998) and Monte Carlo
// ==========================================================================================

double fre_expected(int N, double fle_rms) {
    // TODO C1 (watch out: 2 / N is integer division in C++)
    throw std::logic_error("TODO: fre_expected");
}

double fle_from_fre(double fre_value, int N) {
    // TODO C2
    throw std::logic_error("TODO: fle_from_fre");
}

double tre_predict(const Eigen::MatrixXd& X, const Eigen::Vector3d& target, double fle_rms) {
    // TODO C3: principal axes = eigenvectors V of Xc^T Xc; coordinates Xp = Xc V and rp = V^T (target - x_bar);
    //          d_k^2 = |rp|^2 - rp_k^2, f_k^2 = mean over the fiducials of (|Xp_i|^2 - Xp_ik^2).
    throw std::logic_error("TODO: tre_predict");
}

MonteCarloResult monte_carlo(const Eigen::MatrixXd& X, const Eigen::MatrixXd& targets, double fle_rms, int n,
                             std::uint64_t seed) {
    // TODO C4: Rng rng(seed); in each trial Y = X + rng.normal(0.0, fle_rms / sqrt(3), X.rows(), 3);
    //          register_points, fre, and for each target row r the distance |T r - r|.
    //          Size the results first: fre_samples (n), tre_samples (n x targets.rows()).
    throw std::logic_error("TODO: monte_carlo");
}

std::pair<std::vector<int>, double> best_subset(const Eigen::MatrixXd& X, const Eigen::Vector3d& target, int k,
                                                double fle_rms) {
    // TODO C5: loop over combinations(X.rows(), k); copy the chosen rows into a k x 3 matrix; keep a new best only
    //          if it is smaller by more than 1e-12 (ties: the first one wins).
    throw std::logic_error("TODO: best_subset");
}

// ==========================================================================================
// Part D -- frame chains and the full pipeline
// ==========================================================================================

Eigen::Matrix4d chain_robot_from_image(const Eigen::Matrix4d& T_CR, const Eigen::Matrix4d& T_CP,
                                       const Eigen::Matrix4d& T_PI) {
    // TODO D1
    throw std::logic_error("TODO: chain_robot_from_image");
}

PipelineResult navigation_pipeline(int n_mc, std::uint64_t seed) {
    // TODO D2: load_phantom(), load_joints("pivot_joints.csv").second, ... (see lab.hpp for every field).
    //          Iterate over ph.targets_image_mm (a std::map: names in sorted order) to build the M x 3 target
    //          matrix for monte_carlo; tre_mc[name] = RMS of column j of tre_samples.
    throw std::logic_error("TODO: navigation_pipeline");
}

}  // namespace lab
