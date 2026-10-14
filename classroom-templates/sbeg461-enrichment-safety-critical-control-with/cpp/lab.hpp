// SBEG 461 enrichment lab -- a control-barrier-function (CBF) safety filter for the rehabilitation robot
// (C++ track): declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp. Do not change these declarations: test_lab.cpp calls them.
// The provided functions (model, kinematics, profiles, metrics, the small QP solver, ...) are in helpers.cpp
// (given code, do not edit).
//
// The handle of the planar rehabilitation robot (shared model rehab_2r, horizontal plane) follows a reaching
// movement with task-space computed torque (MR Sec. 11.4.3).  A therapist's hand rests on the table near the
// path.  A CBF-QP safety filter changes the commanded handle acceleration as little as possible so that
//   (1) the handle never enters a keep-out disc of radius r around the hand (distance barrier, relative degree 2,
//       handled by a second-order / exponential CBF), and
//   (2) the speed at which the handle approaches the hand never exceeds v_max, a power-and-force-limiting speed
//       from the reduced-mass contact model of Week 13 (approach-speed barrier, relative degree 1).
//
// Model and notation (MR Ch. 8 and 11; SI units)
//   joint dynamics          M(q) q'' + c(q, q') = tau                       (horizontal plane: g(q) = 0)
//   handle kinematics       p = f(q),  p' = J(q) q',  p'' = J q'' + J' q'
//   computed torque         tau = M~ J^-1 (u - J' q') + c~   gives  p'' = u  with an exact model (M~ = M, c~ = c)
//   nominal command         u_nom = p_d'' + Kd (p_d' - p') + Kp (p_d - p)
//   person (hand) centre    p_o(t), velocity v_o, acceleration a_o (known, e.g. from a tracker)
//   relative position       e = p - p_o,  rho = |e|,  n = e / rho,  w = p' - v_o
//   distance barrier        d = rho - r,   d' = n.w,   d'' = n.(u - a_o) + kappa,   kappa = (|w|^2 - (n.w)^2) / rho
//   CBF conditions (both are linear in u and have the same normal n):
//       keep-out     psi1 = d' + a1 d ,       psi1' + a2 psi1 >= 0   <=>  n.u >= b1
//       speed cap    h_v  = v_max + d' ,      h_v'  + g  h_v  >= 0   <=>  n.u >= b2
//       b1 = n.a_o - kappa - (a1 + a2) d' - a1 a2 d,      b2 = n.a_o - kappa - g (v_max + d')
//   CBF-QP                  u* = argmin |u - u_nom|^2  s.t.  n.u >= b = max(b1, b2)
//                           closed form  u* = u_nom + max(0, b - n.u_nom) n            (|n| = 1)
//
// Parts: A. barrier and filter (barrier_terms, cbf_bound, safety_filter, safety_filter_qp)
//        B. the controller (nominal_accel, computed_torque, simulate)
//        C. force limits near a person (apparent_mass, pfl_speed_limit, design_speed_limit)
// ILLUSTRATIVE VALUES throughout.
//
// Python -> C++: the scenario dict -> lab::Scenario (copy it and change a field, e.g. `auto s = scn; s.v_max =
// 0.16;`, instead of dict(scn, v_max=0.16)); the model dict -> lab::RehabModel; tuples of 3 or 4 values ->
// small structs with the Python names (lab::Profile, lab::BarrierTerms, lab::CbfBound, lab::SpeedDesign);
// (u, lam) and (M, c) -> std::pair; the log dict -> lab::SimLog; the metrics dict -> lab::Metrics;
// scipy.optimize.minimize (SLSQP) -> the given lab::solve_qp (exact active-set solver for small QPs);
// RuntimeError -> std::runtime_error; KeyError / ValueError -> std::invalid_argument.
#ifndef SBEG461_ENRICH_CBF_HPP
#define SBEG461_ENRICH_CBF_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings
#include <sbeg/mrcpp.hpp>  // mr::TransformList, mr::InertiaList (types only)

#include <string>
#include <utility>
#include <vector>

namespace lab {

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Geometry and masses of rehab_2r (Python: rehab_2r()["params"]).
struct RehabParams {
    double L1 = 0.0, L2 = 0.0;  ///< link lengths [m]
    double m1 = 0.0, m2 = 0.0;  ///< link masses [kg]
};

/// The planar 2R rehabilitation robot in MR list form (Python: the dict returned by course_models.rehab_2r()).
struct RehabModel {
    Eigen::MatrixXd Slist;    ///< 6 x 2 screw axes (one per COLUMN)
    Eigen::Matrix4d M;        ///< home configuration of the handle frame
    mr::TransformList Mlist;  ///< 3 frames M_{i-1,i}
    mr::InertiaList Glist;    ///< 2 spatial inertias
    Eigen::Vector3d g;        ///< gravity in {s} (along -z: horizontal plane, g(q) = 0)
    Eigen::Matrix2d limits;   ///< joint limits, one ROW per joint: (min, max) [rad]
    Eigen::Vector2d qdot_max, tau_max, q_home;
    RehabParams params;
};

/// Scenario parameters (Python: the dict returned by load_scenario; common values merged with the scenario).
struct Scenario {
    std::string name;
    double dt = 0.0;         ///< control period [s]
    double T_move = 0.0;     ///< duration of the reaching movement [s]
    double t_end = 0.0;      ///< end of the simulation [s]
    Eigen::Vector2d p_A, p_B;  ///< start and end of the reaching movement [m]
    double Kp = 0.0, Kd = 0.0;  ///< task-space PD gains [1/s^2], [1/s]
    double r_keepout = 0.0;  ///< radius of the keep-out disc [m]
    double alpha1 = 0.0, alpha2 = 0.0;  ///< gains of the second-order (keep-out) barrier [1/s]
    double gamma = 0.0;      ///< gain of the approach-speed barrier [1/s]
    double v_max = 0.0;      ///< approach-speed limit [m/s]
    double m_H = 0.0;        ///< effective mass of the body region (hand) [kg]
    double k_contact = 0.0;  ///< contact stiffness [N/m]
    double F_max = 0.0;      ///< force limit [N]
    Eigen::Vector2d person_p0, person_p1;  ///< start and end of the hand movement [m]
    double person_t0 = 0.0, person_t1 = 0.0;  ///< start and end time of the hand movement [s]
};

/// Quintic time scaling (Python: the tuple (s, s', s'')).
struct Quintic {
    double s = 0.0, sd = 0.0, sdd = 0.0;
};

/// A position profile (Python: the tuples (p, v, a) of person_state and (p_d, p_d', p_d'') of reference).
struct Profile {
    Eigen::Vector2d p, v, a;
};

/// Result of barrier_terms (Python: the tuple (d, d_dot, n, kappa)).
struct BarrierTerms {
    double d = 0.0, d_dot = 0.0;
    Eigen::Vector2d n;
    double kappa = 0.0;
};

/// Result of cbf_bound (Python: the tuple (n, b, b1, b2)).
struct CbfBound {
    Eigen::Vector2d n;
    double b = 0.0, b1 = 0.0, b2 = 0.0;
};

/// Log of simulate (Python: the dict of numpy arrays).  N = round(t_end / dt) + 1 samples; the N x 2 matrices
/// have one ROW per sample.  approach = -d'; active[k] = (filter correction lam > 1e-9).
struct SimLog {
    Eigen::VectorXd t;
    Eigen::MatrixXd q, qd, p, v, p_o, v_o, u_nom, u, tau;
    Eigen::VectorXd d, approach;
    std::vector<bool> active;
};

/// Summary of a run (Python: the dict returned by metrics).
struct Metrics {
    double min_d = 0.0;            ///< smallest d [m]
    double max_approach = 0.0;     ///< largest approach speed [m/s]
    double max_tau_ratio = 0.0;    ///< max |tau_i| / tau_max_i
    double active_fraction = 0.0;  ///< fraction of samples with the filter active
    double final_error = 0.0;      ///< |p(t_end) - p_B| [m]
    double max_dev = 0.0;          ///< largest |p - p_d| [m]
};

/// Result of design_speed_limit (Python: the tuple (v_design, m_R_max, v_exact)).
struct SpeedDesign {
    double v_design = 0.0, m_R_max = 0.0, v_exact = 0.0;
};

/// Result information of solve_qp.
struct QPInfo {
    bool success = false;
    double fun = 0.0;  ///< optimal cost 1/2 z'Hz + f'z
    int nit = 0;       ///< number of active sets examined
};

// ---------------------------------------------------------------------------
// Provided (given in helpers.cpp; do not edit)
// ---------------------------------------------------------------------------

/// Shared course model rehab_2r (course_models.rehab_2r, numbers identical).  Planar end-effector
/// rehabilitation robot in a horizontal plane (gravity along -z, so g(theta) = 0).  L1 = L2 = 0.35 m,
/// m1 = 2.4 kg, m2 = 1.6 kg (uniform rods, radius 0.03 m); joint limits theta1 in [0, pi], theta2 in [0.2, 2.8]
/// rad; rated torques (20, 12) N m; rated speeds 3 rad/s; home pose (0.9, 1.6) rad.  ILLUSTRATIVE VALUES.
RehabModel rehab_2r();

/// The robot used by the functions below (Python: the module constant ROBOT = cm.rehab_2r()).
const RehabModel& robot();

/// Scenario parameters from data/scenario.json (common values merged with the named scenario, "static" or
/// "approaching"); std::invalid_argument for an unknown name.
Scenario load_scenario(const std::string& name = "static");

/// Handle position p = (x, y) in the space frame {s} (MR Eqs. 4.1-4.2 with two links).
Eigen::Vector2d fk(const Eigen::Vector2d& q);

/// 2x2 Jacobian of the handle position, p' = J(q) q'.
Eigen::Matrix2d jacobian(const Eigen::Vector2d& q);

/// The velocity-product term J'(q, q') q' of the handle acceleration p'' = J q'' + J' q'.
Eigen::Vector2d jdot_qdot(const Eigen::Vector2d& q, const Eigen::Vector2d& qd);

/// Closed-form M(q) and c(q, q') of rehab_2r (uniform rods); equal to mr::mass_matrix and
/// mr::vel_quadratic_forces (checked by the tests).  Gravity is along -z, so g(q) = 0 in this horizontal plane.
std::pair<Eigen::Matrix2d, Eigen::Vector2d> dynamics(const Eigen::Vector2d& q, const Eigen::Vector2d& qd);

/// Inverse kinematics of the handle; elbow = +1 gives theta2 > 0 (inside the joint limits of rehab_2r).
Eigen::Vector2d ik(const Eigen::Vector2d& p, double elbow = 1.0);

/// Quintic time scaling s(t), s'(t), s''(t) on [0, T] (MR Sec. 9.2.2.1, Fig. 9.4), held at the ends.
Quintic quintic(double t, double T);

/// Centre p_o, velocity v_o and acceleration a_o of the person's hand: a straight quintic move from
/// person_p0 to person_p1 during [person_t0, person_t1] (a resting hand if p0 = p1).
Profile person_state(double t, const Scenario& scn);

/// Nominal straight-line reaching movement p_d(t), p_d'(t), p_d''(t) from p_A to p_B in T_move seconds.
Profile reference(double t, const Scenario& scn);

/// Transient contact force F(t) = v_app(t) sqrt(mu(t) k) that a contact at time t would cause, using the
/// apparent mass of the handle along -n (towards the hand) at the logged pose; zero when moving away.
/// (Calls your apparent_mass.)
Eigen::VectorXd peak_contact_force(const SimLog& log, const Scenario& scn);

/// Summary of a run: min_d (m), max_approach (m/s), max_tau_ratio (max |tau_i| / tau_max_i),
/// active_fraction, final_error (m, |p(t_end) - p_B|), max_dev (m, largest |p - p_d|).
Metrics metrics(const SimLog& log, const Scenario& scn);

// ---- C++ only (the Python lab uses scipy.optimize.minimize for this) ----

/// min 1/2 z'Hz + f'z  s.t.  A_ub z <= b_ub  (H symmetric positive definite; an empty A_ub = no constraint).
/// Exact active-set enumeration of the KKT conditions for the small problems of this lab (at most 8 variables
/// and 20 constraints, else std::invalid_argument).  Returns (z, info); info.success = false if no feasible
/// point was found (infeasible constraints).
std::pair<Eigen::VectorXd, QPInfo> solve_qp(const Eigen::MatrixXd& H, const Eigen::VectorXd& f,
                                            const Eigen::MatrixXd& A_ub = Eigen::MatrixXd(),
                                            const Eigen::VectorXd& b_ub = Eigen::VectorXd());

// ---------------------------------------------------------------------------
// Part A -- barrier and filter
// ---------------------------------------------------------------------------

/// Distance barrier of the keep-out disc.
/// Returns {d, d_dot, n, kappa}: d = |p - p_o| - r, d_dot = n.(v - v_o), n = (p - p_o)/|p - p_o| and the
/// curvature term kappa = (|w|^2 - (n.w)^2)/|p - p_o| with w = v - v_o, so that d'' = n.(u - a_o) + kappa.
BarrierTerms barrier_terms(const Eigen::Vector2d& p, const Eigen::Vector2d& v, const Eigen::Vector2d& p_o,
                           const Eigen::Vector2d& v_o, double r);

/// The CBF constraint n.u >= b on the commanded handle acceleration u.
/// Returns {n, b, b1, b2}: b1 from the keep-out barrier (second-order CBF with gains alpha1, alpha2),
/// b2 from the approach-speed barrier h_v = v_max + d' (gain gamma), b = max(b1, b2).
CbfBound cbf_bound(const Eigen::Vector2d& p, const Eigen::Vector2d& v, const Eigen::Vector2d& p_o,
                   const Eigen::Vector2d& v_o, const Eigen::Vector2d& a_o, const Scenario& scn);

/// Closed-form solution of the one-constraint CBF-QP  min |u - u_nom|^2  s.t.  a.u >= b.
/// Returns (u, lam): lam >= 0 is the size of the correction, u = u_nom + lam a / |a|^2 (lam = 0: inactive).
std::pair<Eigen::Vector2d, double> safety_filter(const Eigen::Vector2d& u_nom, const Eigen::Vector2d& a, double b);

/// General small CBF-QP:  min |u - u_nom|^2  s.t.  A u >= b  (A: m x 2, one row per constraint; b: m values).
/// Use the given lab::solve_qp; throw std::runtime_error if it does not succeed.  Returns the optimal u.
Eigen::Vector2d safety_filter_qp(const Eigen::Vector2d& u_nom, const Eigen::MatrixXd& A, const Eigen::VectorXd& b);

// ---------------------------------------------------------------------------
// Part B -- the controller
// ---------------------------------------------------------------------------

/// Nominal task-space command u_nom = p_d'' + Kd (p_d' - p') + Kp (p_d - p)  (MR Eq. 11.47 in
/// minimal coordinates, with Ki = 0).
Eigen::Vector2d nominal_accel(double t, const Eigen::Vector2d& p, const Eigen::Vector2d& v, const Scenario& scn);

/// Task-space computed torque tau = M~ J^-1 (u - J' q') + c~ with the controller's model
/// M~ = model_scale * M, c~ = model_scale * c (all link masses and inertias scaled by model_scale).
Eigen::Vector2d computed_torque(const Eigen::Vector2d& q, const Eigen::Vector2d& qd, const Eigen::Vector2d& u,
                                double model_scale = 1.0);

/// Closed-loop simulation.  The controller runs every dt seconds (zero-order hold); the robot
/// M(q) q'' + c = tau is integrated with one RK4 step per control period.
/// Returns the log (lab::SimLog): t, q, qd, p, v, p_o, v_o, u_nom, u, tau, d, approach (= -d'),
/// active (filter correction > 0).  use_qp = true uses safety_filter_qp instead of the closed form.
SimLog simulate(const Scenario& scn, bool filter_on = true, double model_scale = 1.0, bool use_qp = false);

// ---------------------------------------------------------------------------
// Part C -- force limits near a person
// ---------------------------------------------------------------------------

/// Apparent (effective) mass of the handle along the unit direction n: m_R = 1 / (n^T J M^-1 J^T n)
/// (MR Eq. 8.90: Lambda^-1 = J M^-1 J^T for this 2x2 Jacobian).
double apparent_mass(const Eigen::Vector2d& q, const Eigen::Vector2d& n);

/// Largest relative contact speed whose transient peak force stays below F_max (Week 13, Sec. 4.3):
/// mu = (1/m_R + 1/m_H)^-1,  F = v sqrt(mu k)  =>  v_max = F_max / sqrt(mu k).
double pfl_speed_limit(double m_R, double m_H, double k, double F_max);

/// Conservative approach-speed limit for the whole reaching movement.
/// For the poses q = ik(p_d(t)) of the nominal path (n_samples instants on [0, T_move], equally spaced
/// including both ends) take the largest apparent mass in ANY direction, m_R,max = max lambda_max(Lambda(q))
/// with Lambda = (J M^-1 J^T)^-1, and return {v_design, m_R_max, v_exact}:
/// v_exact = pfl_speed_limit(m_R_max, m_H, k_contact, F_max) and v_design = v_exact rounded DOWN to 0.01 m/s.
SpeedDesign design_speed_limit(const Scenario& scn, int n_samples = 201);

}  // namespace lab

#endif  // SBEG461_ENRICH_CBF_HPP
