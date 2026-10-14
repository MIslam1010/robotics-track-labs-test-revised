// SBEG 46X Lab 6 -- Constant-curvature kinematics of a two-segment tendon-driven catheter (C++ STARTER).
//
// Fill in every function marked  // TODO  (replace the `throw std::logic_error("TODO: ...")` line), in the order
// A1 ... C4. Do not change function names or signatures (they are declared in lab.hpp). Build and test from the
// cpp folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
// or from your Lab code folder:   python run_lab.py 6 --cpp
//
// Conventions and notation: see lab.hpp (the same text as the docstring of starter.py).
// As in the Python lab you may use mrcpp, the C++ port of modern_robotics (mr::rp_to_trans, mr::matrix_exp6,
// mr::vec_to_se3, mr::vec_to_so3, mr::adjoint, mr::ad, mr::pinv, ...). The provided helpers are in helpers.cpp:
// load_catheter, load_csv, vee, trans_z, split_q, catheter_backbone and gradient_rows (= np.gradient(P, t, axis=0)).
//
// Eigen reminders:  Eigen::Matrix3d R;  R << a, b, c,  d, e, f,  g, h, i;   (comma initialiser, row by row)
//   A * B, A.transpose(), v.norm(), v.dot(w), Eigen::VectorXd::Constant(n, x), M.minCoeff(),
//   q.segment<2>(1) (= q[1:3]), T.block<3, 1>(0, 3) (= T[:3, 3]), J.block<6, 2>(0, 1) (= J[:, 1:3]),
//   M.rightCols<4>() (= M[:, 1:]), M.row(k), M.col(i), v.cwiseSqrt(), v.cwiseProduct(w), v.asDiagonal(),
//   least squares (np.linalg.lstsq): A.colPivHouseholderQr().solve(b);  std::hypot, std::atan2.
//   Write the type of every Eigen result (Eigen::Matrix3d C = A * B;), never `auto`.
#include "lab.hpp"

#include <sbeg/mrcpp.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lab {

// ---------------------------------------------------------------------------
// Part A -- one constant-curvature segment
// ---------------------------------------------------------------------------

Eigen::Matrix4d arc_transform(double kappa, double phi, double ell) {
    // TODO A1: R = Rot(z, phi) Rot(y, theta) Rot(z, -phi) with theta = kappa ell; p from the arc (lab.hpp);
    //          kappa = 0 (|kappa| < 1e-12): p = (0, 0, ell). Return mr::rp_to_trans(R, p).
    throw std::logic_error("TODO: arc_transform");
}

ArcParams bend_to_arc(const Eigen::Vector2d& theta_v, double ell) {
    // TODO A2: theta = |theta_v|, phi = atan2(theta_y, theta_x) (0 when theta_v = 0), kappa = theta / ell.
    //          Return ArcParams{kappa, phi, theta}.
    throw std::logic_error("TODO: bend_to_arc");
}

Eigen::Vector2d arc_to_bend(double kappa, double phi, double ell) {
    // TODO A2
    throw std::logic_error("TODO: arc_to_bend");
}

Vector6d segment_twist(const Eigen::Vector2d& theta_v, double ell) {
    // TODO A3
    throw std::logic_error("TODO: segment_twist");
}

Eigen::Matrix4d segment_transform_exp(const Eigen::Vector2d& theta_v, double ell) {
    // TODO A3: mr::matrix_exp6(mr::vec_to_se3(V))
    throw std::logic_error("TODO: segment_transform_exp");
}

Eigen::VectorXd tendon_lengths(const Eigen::Vector2d& theta_v, double ell, double d, const Eigen::VectorXd& sigma) {
    // TODO A4: one entry per tendon, l_i = ell - d (theta_v . s_i), s_i = (cos sigma_i, sin sigma_i).
    throw std::logic_error("TODO: tendon_lengths");
}

Eigen::Vector2d bend_from_tendons(const Eigen::VectorXd& dl, double d, const Eigen::VectorXd& sigma) {
    // TODO A5: least squares of (-d S) theta_v = dl, S = n x 2 matrix with rows s_i.
    throw std::logic_error("TODO: bend_from_tendons");
}

Eigen::VectorXd tension_allocation(const Eigen::Vector2d& m, double d, const Eigen::VectorXd& sigma) {
    // TODO A6: minimum-norm t = pinv(d S^T) m (mr::pinv), then subtract min(t) from every entry.
    throw std::logic_error("TODO: tension_allocation");
}

Eigen::VectorXd segment_tensions(const Eigen::Vector2d& theta_v, double ell, double d, const Eigen::VectorXd& sigma,
                                 double EI) {
    // TODO A6
    throw std::logic_error("TODO: segment_tensions");
}

// ---------------------------------------------------------------------------
// Part B -- the two-segment catheter
// ---------------------------------------------------------------------------

Eigen::Matrix4d catheter_fk(const Vector5d& q, const Catheter& cat) {
    // TODO B1: use split_q, trans_z and segment_transform_exp. Segment j: cat.ell(j); tendon angles of
    //          segment j: cat.sigma.row(j) (radians).
    throw std::logic_error("TODO: catheter_fk");
}

Vector6d catheter_tendon_lengths(const Vector5d& q, const Catheter& cat) {
    // TODO B2: tendons 1-3 (segment 1), then tendons 4-6 (they run through segment 1 AND segment 2).
    //          Eigen::VectorXd sig2 = cat.sigma.row(1).transpose();   l.head<3>() = ...;  l.tail<3>() = ...;
    throw std::logic_error("TODO: catheter_tendon_lengths");
}

Matrix65d catheter_tendon_jacobian(const Catheter& cat) {
    // TODO B2: start from Matrix65d::Zero(); column 0 (insertion) stays zero.
    throw std::logic_error("TODO: catheter_tendon_jacobian");
}

Eigen::Vector4d bend_from_catheter_tendons(const Vector6d& dl, const Catheter& cat) {
    // TODO B2: least squares with the bending columns of catheter_tendon_jacobian (6 x 4).
    throw std::logic_error("TODO: bend_from_catheter_tendons");
}

Vector6d catheter_tensions(const Vector5d& q, const Catheter& cat) {
    // TODO B3: segment 2 first (tension_allocation), then segment 1 with the moment of the segment-2 tendons
    //          subtracted. Return (t1, t2):  Vector6d t;  t << t1, t2;
    throw std::logic_error("TODO: catheter_tensions");
}

Matrix6d dexp_left(const Vector6d& V, int terms) {
    // TODO B4: sum over k = 0 .. terms-1 of ad_V^k / (k+1)!  (mr::ad). Keep (k+1)! as a double.
    throw std::logic_error("TODO: dexp_left");
}

Matrix65d catheter_jacobian_space(const Vector5d& q, const Catheter& cat) {
    // TODO B4: column 0 = Ad_{T_base} (0,0,0,0,0,1); columns 1-2 and 3-4 = Ad_{T_(j-1)} dexp_left(V_j) dV_j/dtheta_v,
    //          T_(j-1) = pose of the BASE of segment j. dV/dtheta_v is the 6 x 2 matrix of segment_twist.
    throw std::logic_error("TODO: catheter_jacobian_space");
}

Matrix35d tip_position_jacobian(const Vector5d& q, const Catheter& cat) {
    // TODO B5: pdot = [-[p]  I] V_s  ->  -[p] J_s(top 3 rows) + J_s(bottom 3 rows)
    throw std::logic_error("TODO: tip_position_jacobian");
}

// ---------------------------------------------------------------------------
// Part C -- resolved-rate tip control
// ---------------------------------------------------------------------------

Vector5d clamp_q(const Vector5d& q, const Catheter& cat) {
    // TODO C1: work on a copy of q; std::min(std::max(z, lo), hi); scale q.segment<2>(1) and q.segment<2>(3).
    throw std::logic_error("TODO: clamp_q");
}

Eigen::MatrixXd dls_inverse(const Eigen::MatrixXd& J, double lam) {
    // TODO C1: J^T (J J^T + lam^2 I)^-1   (Eigen::MatrixXd::Identity(J.rows(), J.rows()), .inverse())
    throw std::logic_error("TODO: dls_inverse");
}

Vector5d resolved_rate_step(const Vector5d& q, const Eigen::Vector3d& p_des, const Eigen::Vector3d& pdot_des,
                            const Catheter& cat, double dt, double K, double lam, const Vector5d& W) {
    // TODO C2: with s = sqrt(W) (cwiseSqrt): Jw = J s.asDiagonal(),  qdot = s .* (dls_inverse(Jw, lam) e).
    throw std::logic_error("TODO: resolved_rate_step");
}

GoToResult go_to(const Vector5d& q0, const Eigen::Vector3d& p_goal, const Catheter& cat, double dt, double K,
                 int steps, double lam, const Vector5d& W) {
    // TODO C3: out.q is (steps+1) x 5 (row k = q after k steps), out.err has steps+1 entries (row 0 = q0).
    throw std::logic_error("TODO: go_to");
}

TrackResult track_path(const Vector5d& q0, const Eigen::VectorXd& t, const Eigen::MatrixXd& P, const Catheter& cat,
                       double K, double lam, const Vector5d& W) {
    // TODO C4: Pdot = gradient_rows(P, t); for k = 0 .. N-1 store q, p, err, tendon_dl in row k, then (k < N-1)
    //          step towards P.row(k + 1) with feed-forward Pdot.row(k + 1) and dt = t(k+1) - t(k).
    throw std::logic_error("TODO: track_path");
}

}  // namespace lab
