// SBEG 46X Lab 11 -- image-based visual servoing of an endoscope-holding arm (formal report 3) -- C++ track.
// STARTER CODE: you edit this file. Write every function marked TODO (keep the signatures of lab.hpp); the provided
// helpers are in helpers.cpp. Build and run the checks from your Lab code folder:
//     cmake -S cpp -B build -G Ninja && cmake --build build && ./build/lab_tests
// As in Python, you may use mrcpp (mr::trans_inv, mr::matrix_exp6, mr::vec_to_se3, mr::matrix_log6, mr::se3_to_vec,
// mr::jacobian_body, mr::pinv) where the TODO hints say so.
#include "lab.hpp"

#include <sbeg/mrcpp.hpp>

#include <cmath>
#include <stdexcept>

namespace lab {

// ------------------------------------------------------------------------------------------ Part A
Eigen::Matrix3d intrinsic_matrix(double f, double rho_w, double rho_h, double u0, double v0) {
    // TODO: 3 x 3 matrix; f/rho_w and f/rho_h on the diagonal, (u0, v0) in the last column.
    throw std::logic_error("TODO: intrinsic_matrix");
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> project_points(const Eigen::Matrix3d& K, const Eigen::Matrix4d& T_wc,
                                                           const Eigen::MatrixXd& P_w) {
    // TODO: T_cw = mr::trans_inv(T_wc); for every ROW i of P_w: P_c = R_cw P + p_cw; throw std::invalid_argument
    //       if P_c(2) <= 0; (u, v) = first two entries of K * (P_c / P_c(2)); Z(i) = P_c(2).
    //       Return {p (N x 2), Z (N)}.
    throw std::logic_error("TODO: project_points");
}

Matrix34d camera_matrix_dlt(const Eigen::MatrixXd& P_w, const Eigen::MatrixXd& p) {
    // TODO: throw std::invalid_argument for fewer than 6 points. A (2N x 12) with the rows
    //       [X, Y, Z, 1, 0, 0, 0, 0, -uX, -uY, -uZ, -u] and [0, 0, 0, 0, X, Y, Z, 1, -vX, -vY, -vZ, -v];
    //       Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV); c = svd.matrixV().col(11) (V, not V^T);
    //       C(r, k) = c(4 r + k); divide by the norm of C.block<1, 3>(2, 0); flip the sign if the mean (or sum) of
    //       C[2] . [P; 1] over the points is negative.
    throw std::logic_error("TODO: camera_matrix_dlt");
}

double reprojection_rms(const Matrix34d& C, const Eigen::MatrixXd& P_w, const Eigen::MatrixXd& p) {
    // TODO: q = C [P; 1] for every point, (u, v) = (q(0) / q(2), q(1) / q(2)); sqrt(mean squared distance to p).
    throw std::logic_error("TODO: reprojection_rms");
}

// ------------------------------------------------------------------------------------------ Part B
Eigen::MatrixXd interaction_matrix(const Eigen::MatrixXd& p, const Eigen::VectorXd& Z, const Eigen::Matrix3d& K) {
    // TODO: f_u = K(0, 0), f_v = K(1, 1), u0 = K(0, 2), v0 = K(1, 2); for point i: x = (u - u0)/f_u,
    //       y = (v - v0)/f_v; rows 2i and 2i+1 as in lab.hpp (MR order: the three angular columns first).
    throw std::logic_error("TODO: interaction_matrix");
}

// ------------------------------------------------------------------------------------------ Part C
Vector6d ibvs_velocity(const Eigen::MatrixXd& p, const Eigen::MatrixXd& p_star, const Eigen::VectorXd& Z_hat,
                       const Eigen::Matrix3d& K, double lam) {
    // TODO: stack e = p - p_star as (u1, v1, u2, v2, ...) (Eigen is column-major: fill it explicitly);
    //       return -lam * mr::pinv(interaction_matrix(p, Z_hat, K)) * e.
    throw std::logic_error("TODO: ibvs_velocity");
}

Eigen::Matrix4d camera_step(const Eigen::Matrix4d& T_wc, const Vector6d& V_c, double h) {
    // TODO: T_wc * mr::matrix_exp6(mr::vec_to_se3(V_c * h))
    throw std::logic_error("TODO: camera_step");
}

ServoRun run_ibvs(const Eigen::Matrix4d& T_wc0, const Eigen::MatrixXd& P_w, const Eigen::MatrixXd& p_star,
                  const Eigen::Matrix3d& K, double lam, double h, const std::optional<Eigen::VectorXd>& Z_hat,
                  int n_max, double tol, const std::optional<Eigen::Matrix3d>& K_hat) {
    // TODO: loop k = 0 .. n_max as described in lab.hpp; push_back T and p, collect Z and err; break when
    //       err <= tol (converged = true) or k == n_max; otherwise compute V (and cond = largest / smallest singular
    //       value of the same interaction matrix, Eigen::JacobiSVD) and step the camera. Fill r.Z ((n+1) x N),
    //       r.err, r.V (n x 6), r.cond (n) and r.n_iter = n at the end.
    throw std::logic_error("TODO: run_ibvs");
}

// ------------------------------------------------------------------------------------------ Part D
Eigen::Matrix3d homography_dlt(const Eigen::MatrixXd& xy_t, const Eigen::MatrixXd& p) {
    // TODO: throw std::invalid_argument for fewer than 4 points; A (2N x 9) with the rows
    //       [-x, -y, -1, 0, 0, 0, u x, u y, u] and [0, 0, 0, -x, -y, -1, v x, v y, v]; JacobiSVD with ComputeFullV;
    //       h = V.col(8); H row by row from h; return H / H(2, 2).
    throw std::logic_error("TODO: homography_dlt");
}

Eigen::Matrix4d pose_from_homography(const Eigen::Matrix3d& H, const Eigen::Matrix3d& K) {
    // TODO: Mx = K.partialPivLu().solve(H); s = 2 / (|Mx.col(0)| + |Mx.col(1)|); s = -s if s * Mx(2, 2) < 0;
    //       r1 = s Mx.col(0), r2 = s Mx.col(1); R0 = [r1 r2 r1 x r2]; SVD of R0 (ComputeFullU | ComputeFullV):
    //       R = U diag(1, 1, det(U V^T)) V^T; t = s Mx.col(2).
    throw std::logic_error("TODO: pose_from_homography");
}

Vector6d pbvs_velocity(const Eigen::Matrix4d& T_ct, const Eigen::Matrix4d& T_cstar_t, double lam) {
    // TODO: lam * mr::se3_to_vec(mr::matrix_log6(T_ct * mr::trans_inv(T_cstar_t)))
    throw std::logic_error("TODO: pbvs_velocity");
}

ServoRun run_pbvs(const Eigen::Matrix4d& T_wc0, const Eigen::MatrixXd& P_w, const Eigen::MatrixXd& xy_t,
                  const Eigen::Matrix4d& T_cstar_t, const Eigen::MatrixXd& p_star, const Eigen::Matrix3d& K,
                  double lam, double h, int n_max, double tol, const std::optional<Eigen::Matrix3d>& K_hat) {
    // TODO: the loop of run_ibvs (no cond); V = pbvs_velocity(pose_from_homography(homography_dlt(xy_t, p),
    //       K_model), T_cstar_t, lam).
    throw std::logic_error("TODO: run_pbvs");
}

// ------------------------------------------------------------------------------------------ Part E
Matrix64d rcm_twist_basis(double ell) {
    // TODO: 6 x 4 zero matrix; identity in the top-left 3 x 3; W(3, 1) = ell, W(4, 0) = -ell, W(5, 3) = 1.
    throw std::logic_error("TODO: rcm_twist_basis");
}

std::pair<Eigen::Vector4d, Vector6d> rcm_ibvs_velocity(const Eigen::MatrixXd& p, const Eigen::MatrixXd& p_star,
                                                       const Eigen::VectorXd& Z_hat, const Eigen::Matrix3d& K,
                                                       double lam, double ell, double d_dot) {
    // TODO: e stacked as in ibvs_velocity; W = rcm_twist_basis(ell); A = interaction_matrix(p, Z_hat, K) * W;
    //       w = mr::pinv(A.leftCols(2)) * (-lam * e - A.col(3) * d_dot); xi = (w(0), w(1), 0, d_dot);
    //       return {xi, W * xi}.
    throw std::logic_error("TODO: rcm_ibvs_velocity");
}

Eigen::VectorXd joint_rates(const Eigen::MatrixXd& Blist, const Eigen::VectorXd& theta, const Vector6d& V_c,
                            std::optional<double> qdot_max) {
    // TODO: qd = mr::pinv(mr::jacobian_body(Blist, theta)) * V_c; if qdot_max is set and the largest |qd_i|
    //       exceeds *qdot_max, multiply qd by *qdot_max / (largest |qd_i|).
    throw std::logic_error("TODO: joint_rates");
}

std::pair<double, Eigen::Vector2d> rcm_errors(const Eigen::Matrix4d& T_sc, const Eigen::Vector3d& p_port) {
    // TODO: p_c = mr::trans_inv(T_sc) * (p_port, 1); return {-p_c(2), (p_c(0), p_c(1))}.
    throw std::logic_error("TODO: rcm_errors");
}

}  // namespace lab
