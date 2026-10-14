// sbeg/mrcpp.hpp -- C++17 port of the `modern_robotics` Python library (version 1.1.1, core.py)
// for the SBEG 361 / 461 / 46X C++ lab track.
//
// Every public function of modern_robotics.core is here, in namespace `mr`, with the Python name written
// in snake_case (MatrixExp6 -> mr::matrix_exp6, VecToso3 -> mr::vec_to_so3, so3ToVec -> mr::so3_to_vec,
// FKinSpace -> mr::fkin_space, IKinBody -> mr::ikin_body, ...). The algorithms, the order of operations
// and the tolerances are the same as in Python (near_zero: |z| < 1e-6, test_if_so3/se3: distance < 1e-3,
// IK: at most 20 Newton-Raphson iterations with the Moore-Penrose pseudoinverse), so results agree with
// the Python library to round-off (checked by cpp_support/tests).
//
// Conventions (as in Modern Robotics):
//   * twists V = (omega, v), wrenches F = (m, f): angular part first;
//   * screw-axis lists Slist / Blist are 6 x n matrices whose COLUMNS are the axes (Python: Slist[:, i]);
//   * Mlist is a list of n+1 link frames M_{i-1,i} (4x4), Glist a list of n spatial inertias G_i (6x6);
//   * trajectory matrices (thetamat, taumat, Ftipmat, ...) are N x n: ROW k is time step k (as in Python).
// Functions that return a tuple in Python return a std::pair (or a small struct) here.
//
// Students: use these functions exactly where the Python labs allow `modern_robotics`, never to replace a
// function the lab asks you to write.
#ifndef SBEG_MRCPP_HPP
#define SBEG_MRCPP_HPP

#include <sbeg/eigen.hpp>

#include <utility>
#include <vector>

namespace mr {

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

/// A list of 4x4 homogeneous transforms (Mlist, trajectories). std::vector is safe with Eigen fixed sizes in C++17.
using TransformList = std::vector<Matrix4d>;
/// A list of 6x6 spatial inertia matrices (Glist).
using InertiaList = std::vector<Matrix6d>;

// ----------------------------------------------------------------- basic helpers
/// True if |z| < 1e-6 (NearZero).
bool near_zero(double z);
/// V / ||V|| (Normalize). No check for a zero vector, as in Python (the result is then NaN).
VectorXd normalize(const VectorXd& V);
/// Moore-Penrose pseudoinverse with numpy's default cut-off (singular values <= 1e-15 * max are treated as 0).
/// (Helper used by the IK functions; numpy.linalg.pinv in Python.)
MatrixXd pinv(const MatrixXd& A);

// ----------------------------------------------------------------- Chapter 3: rigid-body motions
/// Inverse of a rotation matrix = transpose (RotInv).
Matrix3d rot_inv(const Matrix3d& R);
/// 3-vector -> 3x3 skew-symmetric matrix [omg] (VecToso3).
Matrix3d vec_to_so3(const Vector3d& omg);
/// 3x3 skew-symmetric matrix -> 3-vector (so3ToVec).
Vector3d so3_to_vec(const Matrix3d& so3mat);
/// Exponential coordinates omg*theta -> (unit axis omghat, theta) (AxisAng3).
std::pair<Vector3d, double> axis_ang3(const Vector3d& expc3);
/// exp([omg]theta) in SO(3) by Rodrigues' formula (MatrixExp3).
Matrix3d matrix_exp3(const Matrix3d& so3mat);
/// Matrix logarithm of a rotation matrix (MatrixLog3); returns [omg]theta.
Matrix3d matrix_log3(const Matrix3d& R);
/// (R, p) -> T (RpToTrans).
Matrix4d rp_to_trans(const Matrix3d& R, const Vector3d& p);
/// T -> (R, p) (TransToRp).
std::pair<Matrix3d, Vector3d> trans_to_rp(const Matrix4d& T);
/// Inverse of a homogeneous transform using R^T (TransInv).
Matrix4d trans_inv(const Matrix4d& T);
/// Twist (omega, v) -> 4x4 se(3) matrix [V] (VecTose3).
Matrix4d vec_to_se3(const Vector6d& V);
/// 4x4 se(3) matrix -> twist (omega, v) (se3ToVec).
Vector6d se3_to_vec(const Matrix4d& se3mat);
/// 6x6 adjoint representation [Ad_T] (Adjoint).
Matrix6d adjoint(const Matrix4d& T);
/// Normalized screw axis from a point q on the axis, unit direction s and pitch h (ScrewToAxis).
Vector6d screw_to_axis(const Vector3d& q, const Vector3d& s, double h);
/// Exponential coordinates S*theta -> (normalized screw axis S, theta) (AxisAng6).
std::pair<Vector6d, double> axis_ang6(const Vector6d& expc6);
/// exp([S]theta) in SE(3) (MatrixExp6).
Matrix4d matrix_exp6(const Matrix4d& se3mat);
/// Matrix logarithm of a homogeneous transform (MatrixLog6); returns [S]theta.
Matrix4d matrix_log6(const Matrix4d& T);
/// Closest rotation matrix (SVD; MR library rule: if det(U V^T) < 0 the third column is negated) (ProjectToSO3).
Matrix3d project_to_so3(const Matrix3d& mat);
/// Closest homogeneous transform (ProjectToSE3).
Matrix4d project_to_se3(const Matrix4d& mat);
/// ||R^T R - I||_F if det > 0, else 1e9 (DistanceToSO3).
double distance_to_so3(const Matrix3d& mat);
/// Distance of a 4x4 matrix from SE(3), 1e9 if det(R) <= 0 (DistanceToSE3).
double distance_to_se3(const Matrix4d& mat);
/// distance_to_so3 < 1e-3 (TestIfSO3).
bool test_if_so3(const Matrix3d& mat);
/// distance_to_se3 < 1e-3 (TestIfSE3).
bool test_if_se3(const Matrix4d& mat);

// ----------------------------------------------------------------- Chapter 4: forward kinematics
/// Body-frame PoE forward kinematics T = M e^[B1]th1 ... e^[Bn]thn (FKinBody). Blist is 6 x n.
Matrix4d fkin_body(const Matrix4d& M, const MatrixXd& Blist, const VectorXd& thetalist);
/// Space-frame PoE forward kinematics T = e^[S1]th1 ... e^[Sn]thn M (FKinSpace). Slist is 6 x n.
Matrix4d fkin_space(const Matrix4d& M, const MatrixXd& Slist, const VectorXd& thetalist);

// ----------------------------------------------------------------- Chapter 5: velocity kinematics
/// Body Jacobian J_b (6 x n) (JacobianBody).
MatrixXd jacobian_body(const MatrixXd& Blist, const VectorXd& thetalist);
/// Space Jacobian J_s (6 x n) (JacobianSpace).
MatrixXd jacobian_space(const MatrixXd& Slist, const VectorXd& thetalist);

// ----------------------------------------------------------------- Chapter 6: inverse kinematics
/// Result of the numerical IK: Python returns the tuple (thetalist, success).
struct IKResult {
    VectorXd thetalist;
    bool success = false;
};
/// Newton-Raphson IK in the body frame, at most 20 iterations (IKinBody).
IKResult ikin_body(const MatrixXd& Blist, const Matrix4d& M, const Matrix4d& T, const VectorXd& thetalist0,
                   double eomg, double ev);
/// Newton-Raphson IK in the space frame, at most 20 iterations (IKinSpace).
IKResult ikin_space(const MatrixXd& Slist, const Matrix4d& M, const Matrix4d& T, const VectorXd& thetalist0,
                    double eomg, double ev);

// ----------------------------------------------------------------- Chapter 8: dynamics of open chains
/// 6x6 matrix [ad_V] (ad).
Matrix6d ad(const Vector6d& V);
/// Recursive Newton-Euler inverse dynamics (InverseDynamics). Mlist has n+1 entries, Glist n.
VectorXd inverse_dynamics(const VectorXd& thetalist, const VectorXd& dthetalist, const VectorXd& ddthetalist,
                          const Vector3d& g, const Vector6d& Ftip, const TransformList& Mlist,
                          const InertiaList& Glist, const MatrixXd& Slist);
/// Mass matrix M(theta) by n calls of inverse_dynamics (MassMatrix).
MatrixXd mass_matrix(const VectorXd& thetalist, const TransformList& Mlist, const InertiaList& Glist,
                     const MatrixXd& Slist);
/// Coriolis and centripetal torques c(theta, dtheta) (VelQuadraticForces).
VectorXd vel_quadratic_forces(const VectorXd& thetalist, const VectorXd& dthetalist, const TransformList& Mlist,
                              const InertiaList& Glist, const MatrixXd& Slist);
/// Gravity torques g(theta) (GravityForces).
VectorXd gravity_forces(const VectorXd& thetalist, const Vector3d& g, const TransformList& Mlist,
                        const InertiaList& Glist, const MatrixXd& Slist);
/// Joint torques J^T(theta) Ftip that balance the tip wrench (EndEffectorForces).
VectorXd end_effector_forces(const VectorXd& thetalist, const Vector6d& Ftip, const TransformList& Mlist,
                             const InertiaList& Glist, const MatrixXd& Slist);
/// Joint accelerations ddtheta = M^{-1}(tau - c - g - J^T Ftip) (ForwardDynamics).
VectorXd forward_dynamics(const VectorXd& thetalist, const VectorXd& dthetalist, const VectorXd& taulist,
                          const Vector3d& g, const Vector6d& Ftip, const TransformList& Mlist,
                          const InertiaList& Glist, const MatrixXd& Slist);
/// One explicit Euler step; returns (thetalist_next, dthetalist_next) (EulerStep).
std::pair<VectorXd, VectorXd> euler_step(const VectorXd& thetalist, const VectorXd& dthetalist,
                                         const VectorXd& ddthetalist, double dt);
/// Inverse dynamics along a trajectory: rows are time steps; returns taumat (N x n) (InverseDynamicsTrajectory).
MatrixXd inverse_dynamics_trajectory(const MatrixXd& thetamat, const MatrixXd& dthetamat, const MatrixXd& ddthetamat,
                                     const Vector3d& g, const MatrixXd& Ftipmat, const TransformList& Mlist,
                                     const InertiaList& Glist, const MatrixXd& Slist);
/// Forward simulation with intRes Euler steps per row of taumat; returns (thetamat, dthetamat), N x n
/// (ForwardDynamicsTrajectory).
std::pair<MatrixXd, MatrixXd> forward_dynamics_trajectory(const VectorXd& thetalist, const VectorXd& dthetalist,
                                                          const MatrixXd& taumat, const Vector3d& g,
                                                          const MatrixXd& Ftipmat, const TransformList& Mlist,
                                                          const InertiaList& Glist, const MatrixXd& Slist, double dt,
                                                          int intRes);

// ----------------------------------------------------------------- Chapter 9: trajectory generation
/// s(t) = 3(t/Tf)^2 - 2(t/Tf)^3 (CubicTimeScaling).
double cubic_time_scaling(double Tf, double t);
/// s(t) = 10(t/Tf)^3 - 15(t/Tf)^4 + 6(t/Tf)^5 (QuinticTimeScaling).
double quintic_time_scaling(double Tf, double t);
/// Straight-line joint trajectory, N x n (row k = time step k). method 3 = cubic, otherwise quintic (JointTrajectory).
MatrixXd joint_trajectory(const VectorXd& thetastart, const VectorXd& thetaend, double Tf, int N, int method);
/// N transforms along a constant screw motion (ScrewTrajectory).
TransformList screw_trajectory(const Matrix4d& Xstart, const Matrix4d& Xend, double Tf, int N, int method);
/// N transforms: straight-line origin, constant-axis rotation (CartesianTrajectory).
TransformList cartesian_trajectory(const Matrix4d& Xstart, const Matrix4d& Xend, double Tf, int N, int method);

// ----------------------------------------------------------------- Chapter 11: robot control
/// Computed-torque control law with scalar gains Kp, Ki, Kd (ComputedTorque).
VectorXd computed_torque(const VectorXd& thetalist, const VectorXd& dthetalist, const VectorXd& eint,
                         const Vector3d& g, const TransformList& Mlist, const InertiaList& Glist,
                         const MatrixXd& Slist, const VectorXd& thetalistd, const VectorXd& dthetalistd,
                         const VectorXd& ddthetalistd, double Kp, double Ki, double Kd);
/// Closed-loop simulation with computed-torque control; returns (taumat, thetamat), N x n (SimulateControl).
/// Unlike Python, nothing is plotted: write the matrices to CSV (sbeg::write_csv) and plot in Python.
std::pair<MatrixXd, MatrixXd> simulate_control(const VectorXd& thetalist, const VectorXd& dthetalist,
                                               const Vector3d& g, const MatrixXd& Ftipmat,
                                               const TransformList& Mlist, const InertiaList& Glist,
                                               const MatrixXd& Slist, const MatrixXd& thetamatd,
                                               const MatrixXd& dthetamatd, const MatrixXd& ddthetamatd,
                                               const Vector3d& gtilde, const TransformList& Mtildelist,
                                               const InertiaList& Gtildelist, double Kp, double Ki, double Kd,
                                               double dt, int intRes);

}  // namespace mr

#endif  // SBEG_MRCPP_HPP
