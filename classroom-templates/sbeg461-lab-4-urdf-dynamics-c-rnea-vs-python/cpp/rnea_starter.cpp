// SBEG 461 Lab 4 -- YOUR C++ RNEA. Translate your Python rnea() (MR Eqs. 8.50-8.54) into Eigen.
//
// Hints
//  * Use fixed-size types (se3::Mat4, se3::Mat6, se3::Vec6) inside the loops: no heap allocation.
//  * Precompute nothing across calls (a fair comparison with Python); but do reuse local arrays.
//  * A_i = [Ad_{M_i^{-1}}] S_i with M_i = M_{01} M_{12} ... M_{i-1,i}   (MR p. 289)
//  * T_{i,i-1} = e^{-[A_i] theta_i} M_{i,i-1},  M_{i,i-1} = M_{i-1,i}^{-1}   (8.50)
//  * Vdot_0 = (0, -g); F_{n+1} = Ftip; AdT_{n+1,n} = [Ad_{M_{n,n+1}^{-1}}]
#include <stdexcept>

#include "rnea.hpp"

Eigen::VectorXd inverse_dynamics(const Model& model, const Eigen::VectorXd& theta,
                                 const Eigen::VectorXd& dtheta, const Eigen::VectorXd& ddtheta,
                                 const se3::Vec3& g, const se3::Vec6& Ftip) {
  // TODO: forward iterations (8.50)-(8.52), then backward iterations (8.53)-(8.54).
  (void)model; (void)theta; (void)dtheta; (void)ddtheta; (void)g; (void)Ftip;
  throw std::logic_error("TODO: implement inverse_dynamics in rnea_starter.cpp");
}
