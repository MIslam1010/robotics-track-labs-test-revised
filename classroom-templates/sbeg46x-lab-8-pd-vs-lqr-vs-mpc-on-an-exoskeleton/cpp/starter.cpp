// SBEG 46X Lab 8 -- PD vs LQR vs MPC on an exoskeleton joint with torque and range limits (C++ STARTER).
//
// STARTER CODE -- you edit this file. Its functions join the module `optctrl` of your library sbeg46x_medrob.
// Write every function marked TODO (replace the `throw std::logic_error("TODO: ...")` line); keep the names and
// signatures (they are declared in lab.hpp). The provided functions (load_params, setpoint, gravity, solve_qp,
// error_limits, make_controller, simulate, limit_rows, and, C++ only, expm, solve_continuous_are, write_sim_csv)
// are in helpers.cpp (given, do not change them). Build and test from the cpp folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
// or from your Lab code folder:   python run_lab.py 8 --cpp
//
// Model, notation and conventions: see lab.hpp (the same text as the docstring of starter.py).
//
// Eigen reminders:  Eigen::MatrixXd A(2, 2);  A << a, b,  c, d;   (comma initialiser, row by row)
//   A * B, A.transpose(), MatrixXd::Identity(n, n), MatrixXd::Zero(r, c), A.rows(), B.cols(),
//   blocks: Z.topLeftCorner(n, n) (= Z[:n, :n]), Z.topRightCorner(n, m) (= Z[:n, n:]),
//           M.block(r0, c0, rows, cols) (= M[r0:r0+rows, c0:c0+cols]), v.head(m) (= v[:m]);
//   stacking: G.resize(r1 + r2, c); G << G1, G2;  (rows of G1, then rows of G2);
//   solve instead of inverse: K = (R + B^T S B)^{-1} B^T S A  ->  (R + Bt*S*B).partialPivLu().solve(Bt*S*A);
//   v.asDiagonal(), v.array().square().inverse(), A.cwiseAbs().maxCoeff(), std::sqrt, std::max.
//   Optional bounds: `if (x_lim) { ... x_lim->first ... x_lim->second ... }`.
//   Write the type of every Eigen result (Eigen::MatrixXd C = A * B;), never `auto`.
#include "lab.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lab {

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;

std::pair<MatrixXd, MatrixXd> joint_state_space(double M, double b) {
    // TODO: write A and B from M theta_ddot + b theta_dot = u with x = (theta - theta_d, theta_dot).
    throw std::logic_error("TODO: joint_state_space");
}

std::pair<MatrixXd, MatrixXd> discretize_zoh(const MatrixXd& A, const MatrixXd& B, double h) {
    // TODO: build Z = [[A, B], [0, 0]] (size n+m), E = lab::expm(Z * h); Ad = E[:n, :n], Bd = E[:n, n:].
    throw std::logic_error("TODO: discretize_zoh");
}

std::pair<double, double> pd_gains(double M, double b, double zeta, double wn) {
    // TODO: match M s^2 + (b + Kd) s + Kp to M (s^2 + 2 zeta wn s + wn^2).
    throw std::logic_error("TODO: pd_gains");
}

std::pair<double, double> scalar_care(double a, double b, double q, double r) {
    // TODO: solve the quadratic in p and keep the POSITIVE root (Workshop Problem 1).
    throw std::logic_error("TODO: scalar_care");
}

std::pair<double, double> scalar_dare(double a, double b, double q, double r) {
    // TODO: solve the quadratic in s and keep the POSITIVE root (Workshop Problem 2).
    throw std::logic_error("TODO: scalar_dare");
}

std::pair<MatrixXd, MatrixXd> lqr(const MatrixXd& A, const MatrixXd& B, const MatrixXd& Q, const MatrixXd& R) {
    // TODO: S from lab::solve_continuous_are(A, B, Q, R); K = R^{-1} B^T S (R.partialPivLu().solve(...), not inverse).
    throw std::logic_error("TODO: lqr");
}

std::pair<std::vector<MatrixXd>, std::vector<MatrixXd>> riccati_recursion(const MatrixXd& A, const MatrixXd& B,
                                                                          const MatrixXd& Q, const MatrixXd& R,
                                                                          const MatrixXd& Qf, int N) {
    // TODO: S has N + 1 entries, K has N; S[N] = Qf; loop n = N-1 down to 0; keep S symmetric (0.5 (S + S^T)).
    throw std::logic_error("TODO: riccati_recursion");
}

std::pair<MatrixXd, MatrixXd> dlqr(const MatrixXd& A, const MatrixXd& B, const MatrixXd& Q, const MatrixXd& R,
                                   double tol, int max_iter) {
    // TODO: iterate the same update as riccati_recursion until it stops changing; throw std::runtime_error if it
    //       does not.
    throw std::logic_error("TODO: dlqr");
}

std::pair<MatrixXd, MatrixXd> bryson_weights(const VectorXd& x_max, const VectorXd& u_max) {
    // TODO: two diagonal matrices (a scalar limit is a vector of length 1).
    throw std::logic_error("TODO: bryson_weights");
}

std::pair<MatrixXd, MatrixXd> prediction_matrices(const MatrixXd& A, const MatrixXd& B, int N) {
    // TODO: x_k = A^k x0 + sum_{i<k} A^{k-1-i} B u_i; fill the blocks with a double loop.
    throw std::logic_error("TODO: prediction_matrices");
}

CondensedQP mpc_qp(const MatrixXd& A, const MatrixXd& B, const MatrixXd& Q, const MatrixXd& R, const MatrixXd& P,
                   int N, const VectorXd& x0, const Bounds& u_lim, const std::optional<Bounds>& x_lim) {
    // TODO: use prediction_matrices and the provided limit_rows; remember the terminal weight P on x_N.
    throw std::logic_error("TODO: mpc_qp");
}

TranscriptionQP transcription_qp(const MatrixXd& A, const MatrixXd& B, const MatrixXd& Q, const MatrixXd& R,
                                 const MatrixXd& P, int N, const VectorXd& x0, const Bounds& u_lim,
                                 const std::optional<Bounds>& x_lim) {
    // TODO: one block row of Aeq per step k: x_{k+1} - A x_k - B u_k = 0, with A x0 moved to beq for k = 0.
    throw std::logic_error("TODO: transcription_qp");
}

std::pair<VectorXd, std::string> mpc_control(const VectorXd& x0, const MatrixXd& A, const MatrixXd& B,
                                             const MatrixXd& Q, const MatrixXd& R, const MatrixXd& P, int N,
                                             const Bounds& u_lim, const std::optional<Bounds>& x_lim) {
    // TODO: mpc_qp -> solve_qp -> return the first input (U.head(m), m = B.cols()) and the status.
    throw std::logic_error("TODO: mpc_control");
}

Metrics metrics(const SimResult& sim, const JointParams& j, double band) {
    // TODO: follow the definitions in lab.hpp exactly (the tests compare with reference values).
    //       np.sign(0) is 0 (no overshoot for a segment without a step); a settling time that never happens is
    //       std::numeric_limits<double>::quiet_NaN().
    throw std::logic_error("TODO: metrics");
}

}  // namespace lab
