// SBEG 461 Lab 6 -- Computed torque vs PID with model error on the course arm; task-space control (C++ track):
// declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp. Do not change these declarations: test_lab.cpp calls them.
// The provided Python helpers (load_config, arm_model, make_case, tau_limits, time_grid, quintic_reference,
// cartesian_reference, rk4_step, Plant, simulate_arm, _diag) are given in helpers.cpp, the course arm in
// course_arm.hpp / course_arm.cpp (do not edit them).
//
// Notation follows Modern Robotics (MR) Ch. 11: theta_e = theta_d - theta, error integral eint = int theta_e dt,
// dynamics tau = M(theta) theta'' + h(theta, theta')  (MR Eq. 11.36), h = c + g + friction.  "model" is the
// CONTROLLER's model {M~, h~}; "plant" is the simulated robot.  In code q = theta.  Units: SI (s, rad, N m).
//
// Python -> C++: ValueError -> std::invalid_argument; a model dict -> lab::Robot; a gain that is "a scalar or a
// vector" -> lab::Gain (write 400.0 or an Eigen::VectorXd, both convert automatically); a tuple -> std::pair or a
// small struct; an (N, n) array -> Eigen::MatrixXd (row k = sample k); a list of 4x4 -> mr::TransformList;
// a result dict -> a struct with the same field names; a class with reset() and __call__ -> a class with reset()
// and operator()(k, q, dq).
#ifndef SBEG461_LAB06_HPP
#define SBEG461_LAB06_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings
#include <sbeg/labio.hpp>  // sbeg::json (the configuration file)
#include <sbeg/mrcpp.hpp>

#include "course_arm.hpp"

#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lab {

using Vector6d = mr::Vector6d;
using Matrix6d = mr::Matrix6d;

// ==========================================================================================
// Provided (do not edit; in helpers.cpp): configuration, robot models, references, plant simulator
// ==========================================================================================

/// The Python model dict: the course arm (course_arm::Robot: name, n, Slist, M, Mlist, Glist, g, tau_max,
/// q_home, ...) plus Blist, the viscous friction vector b, mass_scale and payload.
struct Robot : course_arm::Robot {
    Eigen::MatrixXd Blist;  ///< 6 x n body screw axes [Ad_{M^-1}] Slist, one per COLUMN
    Eigen::VectorXd b;      ///< viscous friction b_j (N m s/rad), zeros if no friction
    double mass_scale = 1.0;
    double payload = 0.0;
};

/// A gain given as a scalar (k I) or as a vector (diagonal matrix), as in Python.  Converts automatically:
/// pass 400.0 or an Eigen vector (e.g. Kp, 0.1 * kv).  Use diag_gain(K, n) to get the length-n vector.
class Gain {
public:
    Gain(double k) : scalar_(k), is_scalar_(true) {}  // NOLINT: implicit on purpose (Python duck typing)
    Gain(const Eigen::VectorXd& K) : vec_(K), is_scalar_(false) {}  // NOLINT
    template <typename Derived>
    Gain(const Eigen::MatrixBase<Derived>& K) : vec_(K), is_scalar_(false) {}  // NOLINT
    bool is_scalar() const { return is_scalar_; }
    double scalar() const { return scalar_; }
    const Eigen::VectorXd& vector() const { return vec_; }

private:
    double scalar_ = 0.0;
    Eigen::VectorXd vec_;
    bool is_scalar_ = true;
};

/// data/lab06_config.json (illustrative values; see make_data.py), as JSON: cfg["dt"].get<double>(),
/// sbeg::as_vec(cfg["joint_move"]["q0"]), cfg["cases"]["exact"]["model_payload"], ...
sbeg::json load_config();

/// The course arm (course_arm.cpp, identical to the shared model) as a Robot with Slist, Blist, M, Mlist, Glist,
/// g, tau_max, q_home and b (viscous friction vector, zeros if `friction` is empty).  `payload` (kg) is a point mass
/// at the tool point; `mass_scale` multiplies every spatial inertia G_i (masses and inertias, CoMs unchanged).
Robot arm_model(double payload = 0.0, double mass_scale = 1.0, const Eigen::VectorXd& friction = Eigen::VectorXd());

/// (plant, model) for a model-error case of the configuration: the plant is always the course arm with the
/// 0.25 kg payload and the configured friction; the model follows cfg["cases"][name].
std::pair<Robot, Robot> make_case(const std::string& name, const sbeg::json& cfg);
std::pair<Robot, Robot> make_case(const std::string& name);  ///< loads the configuration

/// Derated torque limits: tau_limit_fraction (0.6) x rated torque (hardware rule in Course resources).
Eigen::VectorXd tau_limits(const Robot& robot, const sbeg::json& cfg);
Eigen::VectorXd tau_limits(const Robot& robot);  ///< loads the configuration

/// Sample times 0, dt, 2 dt, ..., n dt with n = round(T / dt).
Eigen::VectorXd time_grid(double T, double dt);

/// qd, dqd, ddqd as (len(t), n) matrices (row k = sample k).
struct JointReference {
    Eigen::MatrixXd qd, dqd, ddqd;
};

/// Rest-to-rest quintic time scaling s(t) = 10 tau^3 - 15 tau^4 + 6 tau^5, tau = t/T (MR Eq. 9.12 type),
/// q = q0 + s (q1 - q0), held at q1 for t > T.
JointReference quintic_reference(const Eigen::VectorXd& q0, const Eigen::VectorXd& q1, double T,
                                 const Eigen::VectorXd& t);

/// Xd: list of 4x4; Vd, dVd: (N, 6) body twists of X_d and their time derivatives (row k = sample k).
struct CartesianReference {
    mr::TransformList Xd;
    Eigen::MatrixXd Vd, dVd;
};

/// Straight-line tool motion with fixed orientation: X_d(t) = (R0, p0 + s(t) dp_s), quintic s(t), held after T.
/// Body twist of X_d = (R, p): omega_b = 0, v_b = R^T p'  (MR Eq. 3.83);  dv_b/dt = R^T p''.
CartesianReference cartesian_reference(const Eigen::Matrix4d& X0, const Eigen::Vector3d& dp_s, double T,
                                       const Eigen::VectorXd& t);

/// f(x) for rk4_step.
using StateFunction = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;

/// One classical Runge-Kutta step of x' = f(x) with step h.
Eigen::VectorXd rk4_step(const StateFunction& f, const Eigen::VectorXd& x, double h);

/// Gain as a length-n vector: a scalar k gives (k, ..., k), a vector is returned as it is (Python `_diag`).
Eigen::VectorXd diag_gain(const Gain& K, Eigen::Index n);

/// Forward dynamics of the simulated robot:  q'' = M(q)^-1 (tau - h(q, q')), h including viscous friction b q'.
/// (Python uses Pinocchio's ABA when available; the C++ track uses mr::forward_dynamics, identical to 1e-12.)
class Plant {
public:
    explicit Plant(const Robot& robot);
    Eigen::VectorXd accel(const Eigen::VectorXd& q, const Eigen::VectorXd& dq, const Eigen::VectorXd& tau) const;
    Robot r;
    Eigen::VectorXd b;
};

/// A sampled controller for simulate_arm: reset(), then tau = controller(k, q, dq) at every sample k.
class Controller {
public:
    virtual ~Controller() = default;
    virtual void reset() = 0;
    virtual Eigen::VectorXd operator()(int k, const Eigen::VectorXd& q, const Eigen::VectorXd& dq) = 0;
};

/// The Python result dict(t, q, dq, tau): q, dq, tau are (len(t), n).
struct SimResult {
    Eigen::VectorXd t;
    Eigen::MatrixXd q, dq, tau;
};

/// Closed-loop simulation (the Lab 3/Lab 5 loop for n joints).  controller.reset() first; at every sample k the
/// controller sees the exact state and returns tau_k = controller(k, q_k, dq_k); the actuators apply tau_k clipped
/// to +-tau_limit (if not empty), held constant until t_{k+1} (zero-order hold); the plant is integrated with
/// `substeps` RK4 steps.  An empty dq0 means zeros.
SimResult simulate_arm(const Plant& plant, Controller& controller, const Eigen::VectorXd& t, const Eigen::VectorXd& q0,
                       const Eigen::VectorXd& dq0 = Eigen::VectorXd(), int substeps = 2,
                       const Eigen::VectorXd& tau_limit = Eigen::VectorXd());
SimResult simulate_arm(const Robot& plant, Controller& controller, const Eigen::VectorXd& t, const Eigen::VectorXd& q0,
                       const Eigen::VectorXd& dq0 = Eigen::VectorXd(), int substeps = 2,
                       const Eigen::VectorXd& tau_limit = Eigen::VectorXd());

// ==========================================================================================
// Part A -- gains
// ==========================================================================================

/// Computed-torque PD gains (scalars, K = k I) giving each joint the error dynamics
/// e'' + kd e' + kp e = 0 (MR Eq. 11.33 with Ki = 0) with the target zeta and wn:  kp = wn^2, kd = 2 zeta wn.
/// Units s^-2 and s^-1 (acceleration gains: the model M~ turns them into torques).  Returns (kp, kd).
std::pair<double, double> ct_gains(double zeta, double wn);

/// Stability bound for computed torque with integral action: the error dynamics (MR Eq. 11.33) has the
/// characteristic polynomial s^3 + kd s^2 + kp s + ki, stable iff kp, kd > 0 and 0 < ki < kd kp.
double ct_ki_max(double kp, double kd);

/// (Kp, Ki, Kd) as vectors of length n.
struct PIDGains {
    Eigen::VectorXd Kp, Ki, Kd;
};

/// Independent-joint (decentralized, MR Sec. 11.4.2.1) PID gains: joint j is treated as the Lab 5 single joint
/// with inertia M_jj = M~(q_ref)[j, j] and friction b_j = model.b[j]:
///     Kp_j = M_jj wn^2,  Kd_j = 2 zeta wn M_jj - b_j  (MR Eq. 11.26),  Ki_j = ki_fraction (b_j + Kd_j) Kp_j / M_jj
/// (MR Eq. 11.30 bound).
PIDGains decentralized_gains(const Robot& model, const Eigen::VectorXd& q_ref, double zeta, double wn,
                             double ki_fraction = 0.0);

// ==========================================================================================
// Part B -- joint-space control laws
// ==========================================================================================

/// M~(q) ddq + h~(q, dq) of the controller model with ONE RNEA call (MR Sec. 8.3), h~ = c~ + g~ + b~ dq.
Eigen::VectorXd model_inverse_dynamics(const Robot& model, const Eigen::VectorXd& q, const Eigen::VectorXd& dq,
                                       const Eigen::VectorXd& ddq);

/// Decentralized PID (MR Eq. 11.23 at every joint, no model):  tau = Kp e + Ki eint + Kd e',
/// e = qd - q, e' = dqd - dq.  Gains: scalars or length-n vectors (diagonal matrices).
Eigen::VectorXd pid_torque(const Eigen::VectorXd& q, const Eigen::VectorXd& dq, const Eigen::VectorXd& eint,
                           const Eigen::VectorXd& qd, const Eigen::VectorXd& dqd, const Gain& Kp, const Gain& Ki,
                           const Gain& Kd);

/// PID control with gravity compensation (MR Eq. 11.38):  tau = Kp e + Ki eint + Kd e' + g~(q).
Eigen::VectorXd pd_gravity_torque(const Eigen::VectorXd& q, const Eigen::VectorXd& dq, const Eigen::VectorXd& eint,
                                  const Eigen::VectorXd& qd, const Eigen::VectorXd& dqd, const Gain& Kp,
                                  const Gain& Ki, const Gain& Kd, const Robot& model);

/// Feedforward plus decentralized PID (MR Eq. 11.32 plus feedback; Corke Sec. 9.4.3.1):
/// tau = M~(qd) ddqd + h~(qd, dqd) + Kp e + Ki eint + Kd e'.  The model is evaluated on the REFERENCE.
Eigen::VectorXd feedforward_pd_torque(const Eigen::VectorXd& q, const Eigen::VectorXd& dq,
                                      const Eigen::VectorXd& eint, const Eigen::VectorXd& qd,
                                      const Eigen::VectorXd& dqd, const Eigen::VectorXd& ddqd, const Gain& Kp,
                                      const Gain& Ki, const Gain& Kd, const Robot& model);

/// Computed torque control (MR Eq. 11.37, pseudocode MR Fig. 11.20):
///     tau = M~(q) (ddqd + Kp e + Ki eint + Kd e') + h~(q, dq),
/// with gains scalars (k I, MR p. 434) or vectors (diagonal).  Hint: M~(q) a + h~(q, dq) is the inverse dynamics of
/// the model at (q, dq, a) -- one RNEA call, no mass matrix needed.
Eigen::VectorXd computed_torque(const Eigen::VectorXd& q, const Eigen::VectorXd& dq, const Eigen::VectorXd& eint,
                                const Eigen::VectorXd& qd, const Eigen::VectorXd& dqd, const Eigen::VectorXd& ddqd,
                                const Gain& kp, const Gain& ki, const Gain& kd, const Robot& model);

/// Virtual error energy for setpoint control (MR Eq. 11.41, qd' = 0):
/// V = 1/2 e^T Kp e + 1/2 dq^T M~(q) dq,  e = qd - q,  Kp scalar or vector (diagonal).
double error_energy(const Eigen::VectorXd& q, const Eigen::VectorXd& dq, const Eigen::VectorXd& qd, const Gain& Kp,
                    const Robot& model);

/// Sampled joint-space controller for simulate_arm.
///
///     lab::JointController ctrl(law, t, qd, dqd, ddqd, Kp, Ki, Kd, model);   // model optional for "pid"
///     ctrl.reset();  tau = ctrl(k, q, dq);
///
/// law is "pid" (pid_torque), "pd_gravity" (pd_gravity_torque), "feedforward" (feedforward_pd_torque) or
/// "computed_torque" (computed_torque; then Kp, Ki, Kd are the acceleration gains kp, ki, kd).  t, qd, dqd, ddqd
/// are the reference samples (matrices of len(t) rows).  At every call, as in MR Fig. 11.20:
///     e = qd[k] - q ;  eint = eint + e dt (dt = t[1] - t[0]) ;  tau = law(q, dq, eint, qd[k], dqd[k], [ddqd[k],] ...)
/// reset() sets eint = 0.  Member eint holds the current error integral.
/// The constructor (given) throws std::invalid_argument for an unknown law, or a law other than "pid" without a
/// model, and calls reset().
class JointController : public Controller {
public:
    JointController(const std::string& law, const Eigen::VectorXd& t, const Eigen::MatrixXd& qd,
                    const Eigen::MatrixXd& dqd, const Eigen::MatrixXd& ddqd, const Gain& Kp, const Gain& Ki,
                    const Gain& Kd, const std::optional<Robot>& model = std::nullopt);
    void reset() override;
    Eigen::VectorXd operator()(int k, const Eigen::VectorXd& q, const Eigen::VectorXd& dq) override;

    std::string law;
    Eigen::VectorXd t;
    double dt = 0.0;
    Eigen::MatrixXd qd, dqd, ddqd;
    Gain Kp, Ki, Kd;
    std::optional<Robot> model;
    Eigen::VectorXd eint;
};

/// Metrics of a run (MR Fig. 11.19):  with e(t) = qd(t) - q(t) (rows) and |.| the Euclidean norm,
///     max_err   = max_t |e(t)|            (rad)
///     rms_err   = sqrt(mean_t |e(t)|^2)   (rad)
///     final_err = |e(t_end)|              (rad)
///     effort    = int tau^T tau dt        (N^2 m^2 s, trapezoidal rule)
///     peak_tau  = max_t |tau_j(t)| per joint (vector, N m).
struct Metrics {
    double max_err = 0.0, rms_err = 0.0, final_err = 0.0, effort = 0.0;
    Eigen::VectorXd peak_tau;
};
Metrics run_metrics(const Eigen::VectorXd& t, const Eigen::MatrixXd& qd, const Eigen::MatrixXd& q,
                    const Eigen::MatrixXd& tau);

// ==========================================================================================
// Part C -- task-space computed torque (MR Sec. 11.4.3)
// ==========================================================================================

/// Time derivative of the body Jacobian (MR Exercise 8.6b):  column i,
///     d/dt J_bi = sum_{j > i} [ad_{J_bi}] J_bj dq_j,  with J_b = mr::jacobian_body(Blist, q).
Eigen::MatrixXd body_jacobian_dot(const Eigen::MatrixXd& Blist, const Eigen::VectorXd& q, const Eigen::VectorXd& dq);

/// Configuration and velocity errors of MR Sec. 11.4.3 (p. 436), both in the end-effector frame at X:
///     [X_e] = log(X^-1 X_d),   V_e = [Ad_{X^-1 X_d}] V_d - V.   Returns (Xe, Ve) as 6-vectors (omega first).
std::pair<Vector6d, Vector6d> task_errors(const Eigen::Matrix4d& X, const Vector6d& V, const Eigen::Matrix4d& Xd,
                                          const Vector6d& Vd);

/// The feedforward term d/dt([Ad_{X^-1 X_d}] V_d) of MR Eq. 11.47 (body twists V of X and V_d of X_d).
/// With A = X^-1 X_d one finds  d/dt(Ad_A V_d) = Ad_A dV_d - [ad_V] Ad_A V_d ;  at X = X_d, V = V_d it is dV_d.
Vector6d task_feedforward_accel(const Eigen::Matrix4d& X, const Vector6d& V, const Eigen::Matrix4d& Xd,
                                const Vector6d& Vd, const Vector6d& dVd);

/// Task-space dynamics of the model in the end-effector (body) frame (MR Eqs. 8.90-8.91), square J_b:
///     Lambda = J^-T M J^-1,   eta = J^-T h(q, dq) - Lambda Jdot dq   (V = J dq).   Returns (Lambda, eta).
std::pair<Matrix6d, Vector6d> model_task_space_dynamics(const Robot& model, const Eigen::VectorXd& q,
                                                        const Eigen::VectorXd& dq);

/// Task-space computed torque (MR Eq. 11.47):
///     tau = J_b^T ( Lambda~ (a) + eta~ ),  a = d/dt([Ad_{X^-1 X_d}] V_d) + Kp X_e + Ki Xint + Kd V_e,
/// with X = mr::fkin_body(M, Blist, q), V = J_b dq, (X_e, V_e) = task_errors(...), Xint the integral of X_e, gains
/// scalars or 6-vectors.  For a square, non-singular J_b this equals
///     tau = M~(q) J_b^-1 (a - Jdot_b dq) + h~(q, dq)    (substitute MR Eqs. 8.90-8.91),
/// i.e. the inverse dynamics of the model at the joint acceleration J_b^-1 (a - Jdot_b dq): one RNEA call.
Eigen::VectorXd task_space_ct(const Eigen::VectorXd& q, const Eigen::VectorXd& dq, const Vector6d& Xint,
                              const Eigen::Matrix4d& Xd, const Vector6d& Vd, const Vector6d& dVd, const Gain& kp,
                              const Gain& ki, const Gain& kd, const Robot& model);

/// Sampled task-space computed-torque controller for simulate_arm.
///
///     lab::TaskController ctrl(t, Xd, Vd, dVd, kp, ki, kd, model);  ctrl.reset();  tau = ctrl(k, q, dq);
///
/// At every call: X = mr::fkin_body(model M, Blist, q), V = J_b dq, Xe from task_errors, Xint = Xint + Xe dt, then
/// tau = task_space_ct(q, dq, Xint, Xd[k], Vd.row(k), dVd.row(k), kp, ki, kd, model).  reset() sets Xint = 0.
/// The constructor is given and calls reset().
class TaskController : public Controller {
public:
    TaskController(const Eigen::VectorXd& t, const mr::TransformList& Xd, const Eigen::MatrixXd& Vd,
                   const Eigen::MatrixXd& dVd, const Gain& kp, const Gain& ki, const Gain& kd, const Robot& model);
    void reset() override;
    Eigen::VectorXd operator()(int k, const Eigen::VectorXd& q, const Eigen::VectorXd& dq) override;

    Eigen::VectorXd t;
    double dt = 0.0;
    mr::TransformList Xd;
    Eigen::MatrixXd Vd, dVd;
    Gain kp, ki, kd;
    Robot model;
    Vector6d Xint = Vector6d::Zero();
};

}  // namespace lab

#endif  // SBEG461_LAB06_HPP
