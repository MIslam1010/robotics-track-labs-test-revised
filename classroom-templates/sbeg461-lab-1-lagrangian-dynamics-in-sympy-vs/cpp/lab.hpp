// SBEG 461 Lab 1 -- Lagrangian dynamics, checked against modern_robotics (C++ track): declarations.
//
// The same steps as starter.py / test_lab.py, in namespace `lab`, with Eigen types instead of NumPy arrays.
// Write your code in starter.cpp. Do not change these declarations: test_lab.cpp calls them.
//
// C++ has no SymPy. In the C++ track the model is derived NUMERICALLY at the requested state instead of as formulas:
//   * a "Lagrangian", "K" or any f(theta, dtheta) is a function object lab::StateFn (any lambda
//     [&](const Eigen::VectorXd& q, const Eigen::VectorXd& dq) { return ...; });  P(theta) is a lab::ConfigFn;
//   * partial derivatives come from the given finite-difference helpers partial / partial_q / partial_dq /
//     partial_matrix (5-point central differences, error about 1e-11 for the smooth functions of this lab);
//   * M(theta) is read off K exactly, because K = 1/2 dtheta^T M dtheta is a quadratic form in dtheta.
// The tests compare your numbers with the SymPy model of the Python reference (cpp_cases.json) and with mrcpp
// (the C++ port of modern_robotics).
//
// Robot model (see data/arms.json): a planar n-R chain in the vertical x-y plane, gravity g along -y, joint axes
// along z, zero configuration with all links along +x.  Link i has length L[i], mass m[i], centre of mass (CoM) at
// distance r[i] from joint i along the link, and moment of inertia I[i] about its CoM (about z).
//
// Notation follows Modern Robotics (MR) Section 8.1:
//     L(theta, dtheta) = K - P                                   (MR Eq. 8.13)
//     K = 1/2 dtheta^T M(theta) dtheta                           (MR Eq. 8.14)
//     tau_i = d/dt dL/d(dtheta_i) - dL/d(theta_i)                (MR Eq. 8.15)
//     Gamma_ijk = 1/2 (dm_ij/dth_k + dm_ik/dth_j - dm_jk/dth_i)  (MR Eq. 8.17)
//     c_ij = sum_k Gamma_ijk dtheta_k                            (MR Eq. 8.19)
//     tau = M(theta) ddtheta + C(theta, dtheta) dtheta + g(theta)
// Twists in modern_robotics / mrcpp are (omega, v), angular first.
//
// Python -> C++: a SymPy expression in (th, dth) -> lab::StateFn; a SymPy Matrix -> Eigen::MatrixXd evaluated at
// theta; nested list G[i][j][k] -> std::vector<Eigen::MatrixXd> G with G[i](j, k); the dict of numeric_model ->
// struct lab::NumericModel; the params dict -> struct lab::ChainParams; tuples -> small structs (field names as in
// Python); the dict of compare -> std::map<std::string, double>.
#ifndef SBEG461_LAB1_HPP
#define SBEG461_LAB1_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings
#include <sbeg/mrcpp.hpp>  // mr::TransformList, mr::InertiaList (types of mr_chain)

#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace lab {

/// f(theta, dtheta) -> number, e.g. a Lagrangian or the kinetic energy K.
using StateFn = std::function<double(const Eigen::VectorXd& q, const Eigen::VectorXd& dq)>;
/// f(theta) -> number, e.g. the potential energy P.
using ConfigFn = std::function<double(const Eigen::VectorXd& q)>;
/// F(theta) -> matrix, e.g. the mass matrix M(theta).
using MatrixFn = std::function<Eigen::MatrixXd(const Eigen::VectorXd& q)>;

/// Parameters of one robot of data/arms.json (Python: the dict with keys L, m, r, I, g).
struct ChainParams {
    Eigen::VectorXd L;  ///< link lengths [m]
    Eigen::VectorXd m;  ///< link masses [kg]
    Eigen::VectorXd r;  ///< CoM distance from its own joint along the link [m]
    Eigen::VectorXd I;  ///< link inertia about its CoM, about z [kg m^2]
    double g = 9.81;    ///< gravitational acceleration [m/s^2] (acts along -y)
    /// Number of joints n.
    int n() const { return static_cast<int>(L.size()); }
};

/// Return the parameters of robot `name` from data/arms.json.  (Given.)
ChainParams load_params(const std::string& name);

// ---------------------------------------------------------------------------
// Finite-difference helpers (given)
// ---------------------------------------------------------------------------

/// Step of the 5-point central differences used by the helpers below.
constexpr double FD_STEP = 1e-3;

/// df/dx_i at x (5-point central difference).  (Given.)
double partial(const ConfigFn& f, const Eigen::VectorXd& x, int i);
/// df/dq_i of f(q, dq) at (q, dq), dq held fixed.  (Given.)
double partial_q(const StateFn& f, const Eigen::VectorXd& q, const Eigen::VectorXd& dq, int i);
/// df/d(dq_i) of f(q, dq) at (q, dq), q held fixed.  (Given.)
double partial_dq(const StateFn& f, const Eigen::VectorXd& q, const Eigen::VectorXd& dq, int i);
/// Entry-by-entry dF/dx_k of a matrix-valued function at x.  (Given.)
Eigen::MatrixXd partial_matrix(const MatrixFn& F, const Eigen::VectorXd& x, int k);

// ---------------------------------------------------------------------------
// Checkpoint 1 -- Euler-Lagrange machinery
// ---------------------------------------------------------------------------

/// Total time derivative of f(q, dq) at the state (q, dq, ddq) by the chain rule:
///     sum_i (df/dq_i) dq_i + (df/d(dq_i)) ddq_i.
double time_derivative(const StateFn& f, const Eigen::VectorXd& q, const Eigen::VectorXd& dq,
                       const Eigen::VectorXd& ddq);

/// Euler-Lagrange equations (MR Eq. 8.3 / 8.15) at the state (q, dq, ddq):
/// return the vector f with f_i = d/dt dL/d(dq_i) - dL/dq_i  (use time_derivative).
Eigen::VectorXd euler_lagrange(const StateFn& Lag, const Eigen::VectorXd& q, const Eigen::VectorXd& dq,
                               const Eigen::VectorXd& ddq);

// ---------------------------------------------------------------------------
// Checkpoint 2 -- energies, mass matrix, gravity, Christoffel symbols, Coriolis matrix
// ---------------------------------------------------------------------------

/// Kinetic and potential energy (Python: the pair K, P of planar_chain_energy).
struct Energy {
    double K = 0.0;  ///< kinetic energy [J]
    double P = 0.0;  ///< potential energy [J]
};

/// Kinetic and potential energy of the planar n-R chain at (theta, dtheta):
/// K = sum_i 1/2 m_i |v_ci|^2 + 1/2 I_i (dth_1 + ... + dth_i)^2,  P = sum_i m_i g y_ci.
Energy planar_chain_energy(const ChainParams& params, const Eigen::VectorXd& theta, const Eigen::VectorXd& dtheta);

/// Mass matrix at theta from K = 1/2 dtheta^T M dtheta (MR Eq. 8.14): M = Hessian of K with respect to dtheta
/// (n = theta.size()).
Eigen::MatrixXd mass_matrix(const StateFn& K, const Eigen::VectorXd& theta);

/// g(theta) = dP/dtheta (length n).
Eigen::VectorXd gravity_vector(const ConfigFn& P, const Eigen::VectorXd& theta);

/// Christoffel symbols of the first kind at theta (MR Eq. 8.17).  Return G (n matrices of size n x n) with
/// G[i](j, k) = Gamma_ijk.
std::vector<Eigen::MatrixXd> christoffel(const MatrixFn& M, const Eigen::VectorXd& theta);

/// Coriolis matrix at (theta, dtheta) (MR Eq. 8.19): C_ij = sum_k Gamma_ijk dtheta_k.
Eigen::MatrixXd coriolis_matrix(const MatrixFn& M, const Eigen::VectorXd& theta, const Eigen::VectorXd& dtheta);

// ---------------------------------------------------------------------------
// Checkpoint 3 -- numeric model
// ---------------------------------------------------------------------------

/// The numeric model of a chain (Python: the dict returned by numeric_model). Inputs are vectors of length n.
struct NumericModel {
    int n = 0;                                                                  ///< number of joints
    MatrixFn M;                                                                 ///< M(theta), n x n
    std::function<Eigen::MatrixXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> C;     ///< C(theta, dtheta)
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> c;     ///< c = C dtheta
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> g;                              ///< g(theta)
    std::function<Eigen::MatrixXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> Mdot;  ///< dM/dt
    /// tau(theta, dtheta, ddtheta) from the Euler-Lagrange equations of L = K - P
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&)> tau;
    StateFn K;                                                                  ///< K(theta, dtheta)
    ConfigFn P;                                                                 ///< P(theta)
};

/// Build the numeric model of the chain: fill every member of NumericModel with the functions above.
NumericModel numeric_model(const ChainParams& params);

/// Return max |N + N^T| for N = Mdot - 2C (MR Proposition 8.1: should be ~0).
double skew_residual(const NumericModel& model, const Eigen::VectorXd& theta, const Eigen::VectorXd& dtheta);

/// Energy check: |d/dt (K + P) - dtheta^T tau| with tau from the Euler-Lagrange equations.
/// d/dt(K + P) = dtheta^T M ddtheta + 1/2 dtheta^T Mdot dtheta + dtheta^T g.
double power_residual(const NumericModel& model, const Eigen::VectorXd& theta, const Eigen::VectorXd& dtheta,
                      const Eigen::VectorXd& ddtheta);

// ---------------------------------------------------------------------------
// Checkpoint 4 -- the same robots in modern_robotics (mrcpp)
// ---------------------------------------------------------------------------

/// The modern_robotics description of a chain (Python: the tuple Mlist, Glist, Slist, gvec).
struct MRChain {
    mr::TransformList Mlist;  ///< n+1 transforms: Mlist[i] = M_{i-1,i} = CoM frame {i} in CoM frame {i-1} at theta = 0
    mr::InertiaList Glist;    ///< n spatial inertias diag(I, I, I_i, m_i, m_i, m_i) in the CoM frames
    Eigen::MatrixXd Slist;    ///< 6 x n screw axes in the space frame (columns), S_i = ((0,0,1), -omega x q_i)
    Eigen::Vector3d gvec;     ///< gravity vector [0, -g, 0]
};

/// Build the modern_robotics description of the same chain ({0} = space frame, the last entry of Mlist is the tip
/// frame at the end of link n; only Izz of Glist matters for planar motion).
MRChain mr_chain(const ChainParams& params);

/// M, c and g from modern_robotics (Python: the tuple returned by mr_terms).
struct MRTerms {
    Eigen::MatrixXd M;  ///< mr::mass_matrix
    Eigen::VectorXd c;  ///< mr::vel_quadratic_forces
    Eigen::VectorXd g;  ///< mr::gravity_forces
};

/// Return (M, c, g) from mrcpp: mass_matrix, vel_quadratic_forces, gravity_forces.
MRTerms mr_terms(const ChainParams& params, const Eigen::VectorXd& theta, const Eigen::VectorXd& dtheta);

/// Kinematics backend (given): T = e^[S1]th1 ... e^[Sn]thn M (MR Eq. 4.14).  In starter.cpp this calls
/// mr::fkin_space; you may replace its body by your own SBEG 361 C++ function.
Eigen::Matrix4d fk_space(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist, const Eigen::VectorXd& theta);
/// Kinematics backend (given): space Jacobian J_s(theta), 6 x n (MR Eq. 5.11).  In starter.cpp this calls
/// mr::jacobian_space; you may replace its body by your own SBEG 361 C++ function.
Eigen::MatrixXd jacobian_space(const Eigen::MatrixXd& Slist, const Eigen::VectorXd& theta);

/// Bridge to 361: M(theta) = sum_i ( m_i Jv_i^T Jv_i + I_i Jw_i^T Jw_i ), built with fk_space and jacobian_space.
/// For CoM i, with p_ci = fk_space(M_ci, first i+1 columns of Slist, first i+1 angles) and M_ci the home pose
/// Trans(x_joint_i + r_i, 0, 0): column j <= i of Jw_i is omega_j of J_s and column j of Jv_i is
/// v_j + omega_j x p_ci (point velocity p_dot = v_s + omega_s x p); columns j > i are zero.
/// Planar chain: omega is along z, so I_i |omega|^2 = I_zz,i omega_z^2.
Eigen::MatrixXd mass_matrix_from_jacobians(const ChainParams& params, const Eigen::VectorXd& theta);

// ---------------------------------------------------------------------------
// Checkpoint 5 -- numeric Lagrangian model vs modern_robotics
// ---------------------------------------------------------------------------

/// A list of states (theta, dtheta).
using StateList = std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>;

/// Max absolute differences between the numeric Lagrangian model and modern_robotics over the states.
/// Returns {"M": ..., "c": ..., "g": ..., "M_jac": ...} (M_jac compares mass_matrix_from_jacobians with the
/// numeric model).
std::map<std::string, double> compare(const ChainParams& params, const StateList& states);

}  // namespace lab

#endif  // SBEG461_LAB1_HPP
