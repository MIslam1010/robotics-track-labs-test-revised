// SBEG 461 Lab 4 -- URDF dynamics, RNEA timing and the constrained 2R arm (C++ STARTER).
//
// Fill in every function marked  // TODO  (replace the `throw std::logic_error("TODO: ...")` line).
// Do not change function names or signatures (they are declared in lab.hpp, with the full docstrings).
// Build and test from the cpp folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
// or from your Lab code folder:   python run_lab.py 4 --cpp
//
// Rules (as in starter.py): do NOT call mrcpp (mr::...) in this file; it is the reference the tests compare you
// with. You may use the provided helpers of helpers.cpp (declared in lab.hpp): skew, rpy_to_R, rp_to_trans,
// trans_inv, adjoint, ad, matrix_exp6, read_urdf_chain, load_states, export_for_cpp, benchmark.
//
// Eigen reminders:  Eigen::Matrix4d T = A * B;  T.topLeftCorner<3, 3>() (rotation), T.block<3, 1>(0, 3)
//   (translation), Slist.col(i), w.cross(q), F.dot(A), M.inverse(), A.transpose(),
//   X.partialPivLu().solve(b), Vector6d S; S << w, v;  (stack two 3-vectors), Matrix6d::Zero().
//   Never store an Eigen expression in `auto` (write the type, e.g. `Eigen::Matrix4d T = A * B;`).
#include "lab.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace lab {

// ===========================================================================
// Part A -- URDF -> MR dynamic parameters (MR Sec. 8.8)
// ===========================================================================

MRModel urdf_to_mr(const std::string& path) {
    // TODO: implement (see lab.hpp and lab_handout.md, Step 1).
    //   chain = read_urdf_chain(path); walk it accumulating T_link = T_link * j.T_origin;
    //   for a moving joint: w = R_link * j.axis, q = p_link, S_i = (w, -w x q); T_com = T_link * j.T_com;
    //   Mlist gets trans_inv(T_prev) * T_com; Glist gets diag(I, m I3); T_prev = T_com.
    //   After the loop: Mlist gets trans_inv(T_prev) * T_link (the tool frame). Slist is 6 x n (columns S_i).
    throw std::logic_error("TODO: urdf_to_mr");
}

// ===========================================================================
// Part B -- RNEA (MR Eqs. 8.50-8.54) and timing
// ===========================================================================

Eigen::VectorXd rnea(const Eigen::VectorXd& thetalist, const Eigen::VectorXd& dthetalist,
                     const Eigen::VectorXd& ddthetalist, const Eigen::Vector3d& g, const Vector6d& Ftip,
                     const TransformList& Mlist, const InertiaList& Glist, const Eigen::MatrixXd& Slist) {
    // TODO: implement (see lab.hpp and lab_handout.md, Step 2).
    //   A_i = [Ad_{M_i^-1}] S_i with M_i = M01 M12 ... M_{i-1,i} (MR p. 289); forward iterations (8.50)-(8.52)
    //   with V_0 = 0, Vdot_0 = (0, -g); then backward (8.53)-(8.54) with F_{n+1} = Ftip and
    //   AdT_{n+1,n} = [Ad_{M_{n,n+1}^-1}]. std::vector<Vector6d> / std::vector<Matrix6d> are handy work arrays.
    throw std::logic_error("TODO: rnea");
}

double time_per_call(const std::function<void(std::size_t)>& fn, std::size_t n_calls, int repeats) {
    // TODO: implement (see lab.hpp and lab_handout.md, Step 2).
    //   for each of `repeats` runs: t0 = std::chrono::steady_clock::now(); call fn(k) for k = 0 .. n_calls - 1;
    //   store (elapsed seconds) / n_calls. Return the median of the runs (std::sort). Throw
    //   std::invalid_argument if n_calls == 0 or repeats < 1.
    throw std::logic_error("TODO: time_per_call");
}

// ===========================================================================
// Part C -- Task-space dynamics (MR Sec. 8.6)
// ===========================================================================

std::pair<Eigen::MatrixXd, Eigen::VectorXd> task_space_dynamics(const Eigen::MatrixXd& M, const Eigen::VectorXd& h,
                                                                const Eigen::MatrixXd& J, const Eigen::MatrixXd& Jdot,
                                                                const Eigen::VectorXd& dthetalist) {
    // TODO: implement (see lab.hpp and lab_handout.md, Step 4): Lambda (8.90), eta (8.91); return {Lam, eta}.
    throw std::logic_error("TODO: task_space_dynamics");
}

// ===========================================================================
// Part D -- The MR 2R arm (Fig. 8.1) and constrained dynamics (Sec. 8.7)
// ===========================================================================

std::pair<Eigen::Matrix2d, Eigen::Vector2d> twor_dynamics(const Eigen::Vector2d& theta, const Eigen::Vector2d& dtheta,
                                                          double m1, double m2, double L1, double L2, double g) {
    // TODO: implement (see lab.hpp and lab_handout.md, Step 4): MR Eqs. 8.9-8.10; return {M, c + g_vector}.
    throw std::logic_error("TODO: twor_dynamics");
}

std::pair<Eigen::Matrix2d, Eigen::Matrix2d> twor_jacobian(const Eigen::Vector2d& theta, const Eigen::Vector2d& dtheta,
                                                          double L1, double L2) {
    // TODO: implement (see lab.hpp and lab_handout.md, Step 4): tip J and Jdot (MR Example 8.5); return {J, Jdot}.
    throw std::logic_error("TODO: twor_jacobian");
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> channel_constraint(const Eigen::Vector2d& theta,
                                                               const Eigen::Vector2d& dtheta, double L1, double L2) {
    // TODO: implement (see lab.hpp and lab_handout.md, Step 5): A and Adot as 1 x 2 matrices
    //   (e.g. Eigen::MatrixXd A(1, 2); A << a1, a2;).
    throw std::logic_error("TODO: channel_constraint");
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> constrained_forward_dynamics(const Eigen::MatrixXd& M,
                                                                         const Eigen::VectorXd& h,
                                                                         const Eigen::MatrixXd& A,
                                                                         const Eigen::MatrixXd& Adot,
                                                                         const Eigen::VectorXd& dtheta,
                                                                         const Eigen::VectorXd& tau) {
    // TODO: implement (see lab.hpp and lab_handout.md, Step 5): lambda from (8.98), thetaddot from (8.96);
    //   return {ddtheta, lam}.
    throw std::logic_error("TODO: constrained_forward_dynamics");
}

Eigen::VectorXd constrained_inverse_dynamics(const Eigen::MatrixXd& M, const Eigen::VectorXd& h,
                                             const Eigen::MatrixXd& A, const Eigen::VectorXd& ddtheta,
                                             const Eigen::VectorXd& lam) {
    // TODO: implement (see lab.hpp and lab_handout.md, Step 5): MR Eq. 8.93.
    throw std::logic_error("TODO: constrained_inverse_dynamics");
}

Eigen::MatrixXd constraint_projector(const Eigen::MatrixXd& M, const Eigen::MatrixXd& A) {
    // TODO: implement (see lab.hpp and lab_handout.md, Step 5): MR Eq. 8.100.
    throw std::logic_error("TODO: constraint_projector");
}

ChannelSim simulate_channel_2r(const Eigen::Vector2d& theta0, const Eigen::Vector2d& dtheta0,
                               const TorqueFunction& tau_fun, double T, double dt, double x_c, double alpha,
                               double beta, double m1, double m2, double L1, double L2, double g) {
    // TODO: implement (see lab.hpp and lab_handout.md, Step 6).
    //   N = std::llround(T / dt); resize t (N+1), theta and dtheta ((N+1) x 2), lam and e (N+1).
    //   A small lambda accel(t, th, dth) -> (thetaddot, lambda, e) with the Baumgarte terms; then RK4 on
    //   (theta, thetadot) with fixed step dt (MR p. 298). Store lam(k), e(k) at every sample, including the last.
    throw std::logic_error("TODO: simulate_channel_2r");
}

}  // namespace lab
