// SBEG 46X Lab 8 -- PD vs LQR vs MPC on an exoskeleton joint with torque and range limits (C++ track):
// declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp (the 14 functions marked TODO). Do not change these declarations: test_lab.cpp
// calls them. The provided functions (load_params, setpoint, gravity, solve_qp, error_limits, make_controller,
// simulate, limit_rows, and, C++ only, expm, solve_continuous_are, write_sim_csv) are in helpers.cpp (given code,
// do not edit). Your functions become the `optctrl` module of your cumulative library sbeg46x_medrob.
//
// Plant (MR Sec. 11.4.1, Eq. 11.21, angle measured from the hanging position):
//     M theta_ddot + b theta_dot + g(theta) = sat(tau),   g(theta) = g_s sin(theta) + g_c cos(theta),
//     sat(tau) = clip(tau, -tau_max, tau_max)   (the actuator torque limit),
// with the elbow of the shared course model arm_exo_2r (upper arm hanging): M = 0.0778 kg m^2, g_s = 3.3354 N m.
// Every controller adds the model gravity torque and designs the remaining torque u (PD + gravity compensation,
// MR Eq. 11.38 with K_i = 0):
//     tau = g(theta) + u,   so that   M theta_ddot + b theta_dot = u   (while tau is not saturated).
// Error state  x = (theta - theta_d, theta_dot);  continuous model  x_dot = A x + B u,  A = [[0, 1], [0, -b/M]],
// B = [0, 1/M]^T;  discrete (zero-order hold, period h)  x[n+1] = Ad x[n] + Bd u[n].
// Units: SI (rad, rad/s, N m, s).  All values are illustrative.
// References: Tedrake, Underactuated Robotics, Ch. 8 (Sec. 8.1 continuous LQR, Sec. 8.3.1 discrete Riccati
// equations, Sec. 8.3.2 constraints -> MPC) and Ch. 10 (Sec. 10.2.1 direct transcription, Sec. 10.2.2 direct
// shooting, Sec. 10.5.2 MPC); Astrom & Murray, Feedback Systems, Sec. 6.3 (Eq. 6.26-6.27) and Exercise 6.15
// (Bryson's rule); MR Sec. 10.7 (Eqs. 10.6-10.12) and Sec. 11.4.1.
//
// Python -> C++: the parameter dict j -> lab::JointParams (same field names: j.M, j.lqr.x_max, j.task.steps);
// the controller dict c -> lab::Controller; the simulation dict -> lab::SimResult; the metrics dict ->
// lab::Metrics; a tuple of 2 -> std::pair (.first, .second); the QP tuples -> lab::CondensedQP /
// lab::TranscriptionQP (fields named as in the docstrings); a bound pair (lo, hi) -> lab::Bounds; x_lim=None ->
// std::nullopt (the default argument); matrices are Eigen::MatrixXd (also 1 x 1 for a scalar system), vectors
// Eigen::VectorXd; ValueError -> std::invalid_argument, RuntimeError -> std::runtime_error.
#ifndef SBEG46X_LAB8_HPP
#define SBEG46X_LAB8_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lab {

/// A pair of bounds (lo, hi), one entry per input or per state; +-infinity entries are skipped by limit_rows.
using Bounds = std::pair<Eigen::VectorXd, Eigen::VectorXd>;

/// Task of one run: initial angle, duration and the piecewise-constant set-point steps [[t_0, theta_0], ...].
struct Task {
    double theta0 = 0.0;                            ///< theta(0) (rad); theta_dot(0) = 0
    double T = 0.0;                                 ///< run length (s)
    std::vector<std::pair<double, double>> steps;   ///< (t_k, theta_k): theta_d = theta_k from t_k on
};

/// Parameters of one joint (data/exo_joint.json, key "elbow" or "knee") -- the `j` dict of the Python lab.
struct JointParams {
    std::string model;               ///< e.g. "arm_exo_2r (elbow)"
    double M = 0.0;                  ///< inertia about the joint (kg m^2)
    double b = 0.0;                  ///< viscous damping (N m s/rad)
    double g_s = 0.0, g_c = 0.0;     ///< gravity torque g(theta) = g_s sin(theta) + g_c cos(theta) (N m)
    double tau_rated = 0.0;          ///< rated torque of the course model (N m)
    Eigen::Vector2d mech_limits;     ///< mechanical joint limits (rad)
    double tau_max = 0.0;            ///< derated motor limit (N m)
    Eigen::Vector2d rom;             ///< prescribed range of motion [lo, hi] (rad)
    double v_max = 0.0;              ///< speed limit (rad/s)
    double h = 0.0;                  ///< control period (s)
    struct {
        double wn = 0.0, zeta = 0.0;
    } pd;                            ///< hand-tuned PD
    struct {
        Eigen::VectorXd x_max, u_max;
    } lqr;                           ///< Bryson limits
    struct {
        int N = 0;
        double g_reserve = 0.0;
    } mpc;                           ///< MPC horizon and gravity reserve
    Task task;
};

/// The whole parameter file: note, elbow (main lab) and knee (optional Experiment E2).
struct Params {
    std::string note;
    JointParams elbow, knee;
};

/// Controller data built once by make_controller (the `c` dict). Unused entries stay empty / 0.
struct Controller {
    std::string kind;                         ///< "pd", "lqr" or "mpc"
    Eigen::MatrixXd A, B, Ad, Bd;             ///< continuous and ZOH error models
    double Kp = 0.0, Kd = 0.0;                ///< PD gains (kind "pd")
    Eigen::MatrixXd Q, R, K, S;               ///< Bryson weights and discrete LQR (kinds "lqr", "mpc")
    int N = 0;                                ///< MPC horizon
    Bounds u_lim;                             ///< MPC input bounds (tightened by the gravity reserve)
};

/// Result of simulate (the dict of arrays sampled at the control instants), plus status and controller.
struct SimResult {
    Eigen::VectorXd t, theta, theta_dot, theta_d, u, tau_cmd, tau;
    std::vector<std::string> status;          ///< MPC QP status per sample ("-" for pd / lqr)
    Controller ctrl;
};

/// The performance and safety numbers of one run (the dict returned by metrics; settle_time may be NaN).
struct Metrics {
    double rom_violation = 0.0, v_violation = 0.0, overshoot = 0.0, settle_time = 0.0, peak_tau = 0.0,
           sat_fraction = 0.0, rms_error = 0.0, effort = 0.0;
};

/// (H, f, G, w) of the condensed (direct-shooting) MPC QP.
struct CondensedQP {
    Eigen::MatrixXd H;
    Eigen::VectorXd f;
    Eigen::MatrixXd G;
    Eigen::VectorXd w;
};

/// (H, f, G, w, Aeq, beq) of the direct-transcription MPC QP.
struct TranscriptionQP {
    Eigen::MatrixXd H;
    Eigen::VectorXd f;
    Eigen::MatrixXd G;
    Eigen::VectorXd w;
    Eigen::MatrixXd Aeq;
    Eigen::VectorXd beq;
};

// =====================================================================================================
// Provided (helpers.cpp, do not edit): parameters, set-points, gravity, QP solver, simulator
// =====================================================================================================

/// Read the lab parameters (data/exo_joint.json, illustrative values). Keys: elbow (main lab) and knee
/// (optional Experiment E), each with the joint model, limits, task and controller settings.
/// An empty path reads the lab's data/exo_joint.json.
Params load_params(const std::string& path = "");

/// Piecewise-constant set-point theta_d(t): task.steps = [(t_0, theta_0), (t_1, theta_1), ...].
double setpoint(double t, const Task& task);

/// g(theta) = scale (g_s sin theta + g_c cos theta); scale != 1 models a heavier/lighter limb than the model.
double gravity(double theta, const JointParams& j, double scale = 1.0);

/// Solve  min 1/2 z^T H z + f^T z  s.t.  G z <= w,  Aeq z = beq  (H symmetric positive definite).
/// Course fallback solver: equality constraints are eliminated with a null-space basis, the remaining problem is
/// turned into a least-distance problem  min |y|  s.t.  M y <= d  (y = R z + R^{-T} f, H = R^T R) and solved exactly
/// with non-negative least squares (Lawson & Hanson's LDP method; NNLS, an active-set method, written in helpers.cpp
/// as scipy.optimize.nnls). Empty G / Aeq (the defaults) = no inequality / equality constraints.
/// Returns (z, status) with status "optimal" or "infeasible".
std::pair<Eigen::VectorXd, std::string> solve_qp(const Eigen::MatrixXd& H, const Eigen::VectorXd& f,
                                                 const Eigen::MatrixXd& G = Eigen::MatrixXd(),
                                                 const Eigen::VectorXd& w = Eigen::VectorXd(),
                                                 const Eigen::MatrixXd& Aeq = Eigen::MatrixXd(),
                                                 const Eigen::VectorXd& beq = Eigen::VectorXd(), double tol = 1e-9);

/// State limits in error coordinates x = (theta - theta_d, theta_dot) for the range of motion and speed.
Bounds error_limits(double theta_d, const JointParams& j);

/// Build the controller data once (gains, Riccati solutions) for kind "pd", "lqr" or "mpc". Uses YOUR functions.
Controller make_controller(const std::string& kind, const JointParams& j);

/// Simulate the exoskeleton joint under controller kind = "pd", "lqr" or "mpc".
/// The controller runs every h seconds (zero-order hold); the plant M theta_ddot = sat(tau) - b theta_dot -
/// g_scale g(theta) is integrated with RK4 (`substeps` per control period). The controller always uses the
/// MODEL gravity g(theta) (g_scale = 1); g_scale != 1 tests a limb heavier or lighter than the model.
/// T and task default to j.task (std::nullopt); ctrl defaults to make_controller(kind, j).
/// Returns the arrays sampled at the control instants (t, theta, theta_dot, theta_d, u, tau_cmd, tau) plus status
/// (the MPC QP status per sample) and the controller data ctrl. An unknown kind throws std::invalid_argument.
SimResult simulate(const std::string& kind, const JointParams& j, std::optional<double> T = std::nullopt,
                   double g_scale = 1.0, const std::optional<Task>& task = std::nullopt, int substeps = 10,
                   const std::optional<Controller>& ctrl = std::nullopt);

/// PROVIDED HELPER (Python: _limit_rows): rows of  [I; -I] X <= [hi; -lo]  for the finite entries of the per-step
/// bounds lim_lo, lim_hi (length n each) repeated over N steps. Returns (G, w).
std::pair<Eigen::MatrixXd, Eigen::VectorXd> limit_rows(int N, int n, const Eigen::VectorXd& lim_lo,
                                                       const Eigen::VectorXd& lim_hi);

/// C++ only: matrix exponential e^A (= scipy.linalg.expm; scaling and squaring with a Pade(13) approximant).
Eigen::MatrixXd expm(const Eigen::MatrixXd& A);

/// C++ only: stabilising solution S of the continuous algebraic Riccati equation
/// A^T S + S A - S B R^{-1} B^T S + Q = 0 (= scipy.linalg.solve_continuous_are; matrix-sign-function method).
Eigen::MatrixXd solve_continuous_are(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& Q,
                                     const Eigen::MatrixXd& R);

/// C++ only: write a run to a CSV file with the columns of data/ref_elbow_*.csv
/// (t, theta, theta_dot, theta_d, u, tau_cmd, tau), for plotting with cpp/plot_results.py.
void write_sim_csv(const std::string& path, const SimResult& sim);

// =====================================================================================================
// Your functions (starter.cpp; module optctrl)
// =====================================================================================================

/// Continuous error model of the gravity-compensated joint  M theta_ddot + b theta_dot = u:
/// x = (theta - theta_d, theta_dot), x_dot = A x + B u. Returns (A (2x2), B (2x1)).
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> joint_state_space(double M, double b);

/// Exact zero-order-hold discretisation: Ad = e^{A h}, Bd = int_0^h e^{A s} ds B.
/// Use the matrix exponential of the block matrix [[A, B], [0, 0]] h (lab::expm). Returns (Ad, Bd).
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> discretize_zoh(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                                                           double h);

/// PD gains (with gravity compensation) that give M e'' + (b + Kd) e' + Kp e = 0 the natural frequency wn
/// and damping ratio zeta (MR Eqs. 11.25-11.26; SBEG 461 Week 5). Returns (Kp, Kd).
std::pair<double, double> pd_gains(double M, double b, double zeta, double wn);

/// Scalar continuous LQR: x_dot = a x + b u, J = int (q x^2 + r u^2) dt. The Riccati equation
/// 2 a p - p^2 b^2 / r + q = 0 has the positive root p; u = -k x with k = b p / r. Returns (p, k).
std::pair<double, double> scalar_care(double a, double b, double q, double r);

/// Scalar discrete LQR: x[n+1] = a x[n] + b u[n], J = sum (q x^2 + r u^2). The fixed point of
/// s = q + a^2 s - a^2 b^2 s^2 / (r + b^2 s) is the positive root of
/// b^2 s^2 + (r (1 - a^2) - q b^2) s - q r = 0;  u = -k x with k = a b s / (r + b^2 s). Returns (s, k).
std::pair<double, double> scalar_dare(double a, double b, double q, double r);

/// Continuous infinite-horizon LQR (Tedrake Sec. 8.1; Astrom & Murray Eq. 6.27):
/// S A + A^T S - S B R^{-1} B^T S + Q = 0,  K = R^{-1} B^T S. Returns (K, S).
/// (lab::solve_continuous_are may be used.)
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> lqr(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                                                const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R);

/// Finite-horizon discrete LQR by the Riccati difference equation (Tedrake Sec. 8.3.1), backwards from
/// S[N] = Qf:   K[n] = (R + B^T S[n+1] B)^{-1} B^T S[n+1] A,
///              S[n] = Q + A^T S[n+1] A - A^T S[n+1] B K[n].
/// Returns (K_list, S_list) with K_list[n] for n = 0..N-1 and S_list[n] for n = 0..N.
/// (Index convention: the gain applied at step n uses S[n+1].)
std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> riccati_recursion(
    const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
    const Eigen::MatrixXd& Qf, int N);

/// Infinite-horizon discrete LQR: iterate the Riccati difference equation from S = Q until
/// max|S_new - S| < tol * max(1, max|S|) (the DARE fixed point). Returns (K, S), u = -K x.
/// Throws std::runtime_error if it does not converge in max_iter iterations.
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> dlqr(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                                                 const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                                                 double tol = 1e-12, int max_iter = 100000);

/// Bryson's rule (Astrom & Murray Exercise 6.15): Q = diag(1/x_max_i^2), R = diag(1/u_max_j^2). Returns (Q, R).
/// (A scalar limit is a vector of length 1: Eigen::VectorXd::Constant(1, 0.5).)
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> bryson_weights(const Eigen::VectorXd& x_max, const Eigen::VectorXd& u_max);

/// Direct-shooting (condensed) prediction X = Sx x0 + Su U with X = [x1; ...; xN] (N n x 1) and
/// U = [u0; ...; u_{N-1}] (N m x 1):  block row k (k = 1..N) of Sx is A^k, block (k, i) of Su is A^{k-1-i} B
/// for i < k (zero otherwise). Returns (Sx, Su).
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> prediction_matrices(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                                                                int N);

/// Condensed (direct-shooting) MPC QP:  min_U  sum_{k=0}^{N-1} (x_k^T Q x_k + u_k^T R u_k) + x_N^T P x_N
/// written as  1/2 U^T H U + f^T U + const  with  H = 2 (Su^T Qb Su + Rb),  f = 2 Su^T Qb Sx x0,
/// Qb = blkdiag(Q, ..., Q, P) over x1..xN, Rb = blkdiag(R, ..., R)  (the x0 term is constant).
/// Constraints G U <= w:  u_lim = (u_lo, u_hi) on every u_k;  x_lim = (x_lo, x_hi) on x1..xN (std::nullopt = no
/// state limits; +-inf entries are skipped). Returns {H, f, G, w}.
CondensedQP mpc_qp(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& Q,
                   const Eigen::MatrixXd& R, const Eigen::MatrixXd& P, int N, const Eigen::VectorXd& x0,
                   const Bounds& u_lim, const std::optional<Bounds>& x_lim = std::nullopt);

/// Direct-transcription MPC QP (Tedrake Sec. 10.2.1): decision z = [x1; ...; xN; u0; ...; u_{N-1}],
/// cost 1/2 z^T H z + f^T z with H = 2 blkdiag(Qb, Rb), f = 0 (x0^T Q x0 is constant),
/// dynamics as equality constraints  x_{k+1} - A x_k - B u_k = 0  (k = 0..N-1, x_0 given): Aeq z = beq,
/// bounds as G z <= w. Returns {H, f, G, w, Aeq, beq}.
TranscriptionQP transcription_qp(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& Q,
                                 const Eigen::MatrixXd& R, const Eigen::MatrixXd& P, int N,
                                 const Eigen::VectorXd& x0, const Bounds& u_lim,
                                 const std::optional<Bounds>& x_lim = std::nullopt);

/// Receding-horizon control law (Tedrake Sec. 10.5.2): build the condensed QP with mpc_qp, solve it with
/// solve_qp, and apply only the FIRST input. Returns (u0, status): u0 = the first m entries of U (length m = number
/// of inputs; for the joint, m = 1 and the torque is u0(0)).
std::pair<Eigen::VectorXd, std::string> mpc_control(const Eigen::VectorXd& x0, const Eigen::MatrixXd& A,
                                                    const Eigen::MatrixXd& B, const Eigen::MatrixXd& Q,
                                                    const Eigen::MatrixXd& R, const Eigen::MatrixXd& P, int N,
                                                    const Bounds& u_lim,
                                                    const std::optional<Bounds>& x_lim = std::nullopt);

/// Performance and safety numbers of one run:
///   rom_violation   max over time of max(theta - rom_hi, rom_lo - theta, 0)            [rad]
///   v_violation     max over time of max(|theta_dot| - v_max, 0)                        [rad/s]
///   overshoot       max over each set-point segment of the motion past the new set-point,
///                   in the direction of the step (0 if none); the largest over all steps [rad]
///   settle_time     for each step, the time from the step until |theta - theta_d| stays <= band to the end of
///                   the segment (NaN if never); the LARGEST over all steps                [s]
///   peak_tau        max |tau| applied                                                   [N m]
///   sat_fraction    fraction of samples with |tau_cmd| > tau_max                        [-]
///   rms_error       sqrt(mean((theta - theta_d)^2)) over the whole run                  [rad]
///   effort          h * sum(tau^2)                                                      [N^2 m^2 s]
/// Returns a Metrics struct with these fields.
Metrics metrics(const SimResult& sim, const JointParams& j, double band = 0.02);

}  // namespace lab

#endif  // SBEG46X_LAB8_HPP
