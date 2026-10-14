// SBEG 461 Lab 9 -- C++ track: GIVEN code (do not edit).  "toppralite": a small C++ stand-in for the part of the
// Python library toppra (0.6, TOPP-RA) that this lab uses.
//
// toppra is a Python library; the course does not ship it for the C++ track.  toppralite implements the SAME
// algorithm with the SAME conventions, so your C++ `topp` is the same thin wrapper as the Python one:
//   * constraints in toppra's canonical linear form  a(s) sdd + b(s) sdot^2 + c(s) in [lower, upper] at each grid
//     point (joint torque: collocation; joint acceleration: interpolation, toppra's defaults), and a velocity
//     constraint as an upper bound on x = sdot^2;
//   * TOPP-RA (Pham & Pham, IEEE T-RO 2018): a backward pass computes the controllable sets K_i of x = sdot^2 on the
//     grid (a 2-variable linear program per grid point, solved exactly here by eliminating sdd), then a greedy
//     forward pass picks the largest admissible x at every grid point;
//   * ParametrizeConstAccel: constant sdd between grid points, t_{i+1} = t_i + 2 ds / (sdot_i + sdot_{i+1}).
// The auto-checks also compare your results with the numbers of the real toppra (cpp_cases.json); toppralite
// reproduces them to about 1e-8.
//
// Python                                                    C++
//   import toppra as ta, toppra.constraint as constraint      namespace ta = toppralite;
//   constraint.JointTorqueConstraint(inv_dyn, lim, fs)        ta::joint_torque_constraint(inv_dyn, lim, fs)
//   constraint.JointVelocityConstraint(lim)                   ta::joint_velocity_constraint(lim)
//   constraint.JointAccelerationConstraint(lim)               ta::joint_acceleration_constraint(lim)
//   inst = algo.TOPPRA(cons, path, gridpoints=grid,           ta::TOPPRA inst(cons, path, grid);
//                      parametrizer="ParametrizeConstAccel")
//   traj = inst.compute_trajectory(0.0, 0.0)  (None: fail)    auto traj = inst.compute_trajectory(0.0, 0.0);  (nullptr: fail)
//   traj.duration, traj(t, order)                             traj->duration(), (*traj)(t, order)
//   inst.problem_data.sd_vec                                  inst.problem_data().sd_vec
// `lim` is an n x 2 matrix [lower, upper] per joint, as in toppra.
#ifndef SBEG461_LAB9_TOPPRALITE_HPP
#define SBEG461_LAB9_TOPPRALITE_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include <functional>
#include <memory>
#include <vector>

namespace toppralite {

/// A geometric path called like a toppra path: path(s, 0) = theta(s), path(s, 1) = theta'(s), path(s, 2) = theta''(s).
using Path = std::function<Eigen::VectorXd(double s, int order)>;
/// A time-parameterised trajectory: traj(t, 0) = q(t), traj(t, 1) = qdot(t), traj(t, 2) = qddot(t).
using Trajectory = std::function<Eigen::VectorXd(double t, int order)>;
/// inv_dyn(q, qd, qdd) -> joint torques (with gravity), as toppra's JointTorqueConstraint expects.
using InverseDynamics = std::function<Eigen::VectorXd(const Eigen::VectorXd& q, const Eigen::VectorXd& qd,
                                                      const Eigen::VectorXd& qdd)>;

enum class Discretization { Collocation, Interpolation };

/// One constraint (toppra.constraint.*).  Build it with the three functions below.
struct Constraint {
    enum class Kind { Torque, Velocity, Acceleration } kind = Kind::Torque;
    InverseDynamics inv_dyn;          ///< torque only
    Eigen::MatrixXd lim;              ///< n x 2: [lower, upper] per joint
    Eigen::VectorXd fs_coef;          ///< torque only: dry-friction coefficients (zeros in this lab)
    Discretization discretization = Discretization::Collocation;
};

/// constraint.JointTorqueConstraint(inv_dyn, tau_lim, fs_coef): tau_lim is n x 2 [tau_min, tau_max].
Constraint joint_torque_constraint(InverseDynamics inv_dyn, const Eigen::MatrixXd& tau_lim,
                                   const Eigen::VectorXd& fs_coef,
                                   Discretization discretization = Discretization::Collocation);
/// constraint.JointVelocityConstraint(vlim): vlim is n x 2 [-qdot_max, qdot_max].
Constraint joint_velocity_constraint(const Eigen::MatrixXd& vlim);
/// constraint.JointAccelerationConstraint(alim): alim is n x 2 [-qddot_max, qddot_max].
Constraint joint_acceleration_constraint(const Eigen::MatrixXd& alim,
                                         Discretization discretization = Discretization::Interpolation);

/// toppra.parametrizer.ParametrizeConstAccel: theta(s(t)) with constant sdd on every grid interval.
class ParametrizeConstAccel {
public:
    ParametrizeConstAccel(Path path, const Eigen::VectorXd& gridpoints, const Eigen::VectorXd& velocities);
    /// q (order 0), qdot (1) or qddot (2) at time t.
    Eigen::VectorXd operator()(double t, int order = 0) const;
    double duration() const { return ts_(ts_.size() - 1) - ts_(0); }
    const Eigen::VectorXd& ts() const { return ts_; }   ///< time of every grid point

private:
    Path path_;
    Eigen::VectorXd ss_, vs_, ts_, us_;
};

/// inst.problem_data
struct ProblemData {
    Eigen::VectorXd gridpoints;
    Eigen::VectorXd sd_vec;   ///< optimal sdot at every grid point (empty if the problem failed)
    Eigen::VectorXd sdd_vec;  ///< sdd on every interval
    Eigen::MatrixXd K;        ///< (N+1) x 2 controllable sets of x = sdot^2
    bool ok = false;
};

/// toppra.algorithm.TOPPRA(constraint_list, path, gridpoints=grid, parametrizer="ParametrizeConstAccel")
class TOPPRA {
public:
    TOPPRA(std::vector<Constraint> constraint_list, Path path, const Eigen::VectorXd& gridpoints);
    /// Solve and return the trajectory, or nullptr if no feasible time scaling exists (Python: None).
    std::shared_ptr<ParametrizeConstAccel> compute_trajectory(double sd_start = 0.0, double sd_end = 0.0);
    const ProblemData& problem_data() const { return data_; }

private:
    std::vector<Constraint> constraints_;
    Path path_;
    ProblemData data_;
};

}  // namespace toppralite

#endif  // SBEG461_LAB9_TOPPRALITE_HPP
