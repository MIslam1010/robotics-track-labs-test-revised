// SBEG 461 Lab 4 -- RNEA in C++ with Eigen. Interface (PROVIDED, do not edit).
#pragma once
#include <string>
#include <vector>

#include "se3.hpp"

struct Model {
  int n = 0;
  std::vector<se3::Mat4> Mlist;  // n + 1 matrices M_{i-1,i} (MR notation, CoM frames at theta = 0)
  std::vector<se3::Mat6> Glist;  // n spatial inertias G_i (MR Eq. 8.32)
  std::vector<se3::Vec6> Slist;  // n space-frame screw axes S_i
};

// Reads the text file written by export_for_cpp() in starter.py.
Model load_model(const std::string& path);

// Joint torques by the recursive Newton-Euler algorithm, MR Eqs. (8.50)-(8.54).
// Same meaning as modern_robotics.InverseDynamics(theta, dtheta, ddtheta, g, Ftip, Mlist, Glist, Slist).
Eigen::VectorXd inverse_dynamics(const Model& model, const Eigen::VectorXd& theta,
                                 const Eigen::VectorXd& dtheta, const Eigen::VectorXd& ddtheta,
                                 const se3::Vec3& g, const se3::Vec6& Ftip);
