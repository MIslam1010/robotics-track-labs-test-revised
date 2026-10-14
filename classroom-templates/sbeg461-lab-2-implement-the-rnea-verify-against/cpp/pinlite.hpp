// SBEG 461 Lab 2 -- C++ track: GIVEN code (do not edit).  "pinlite": a minimal C++ stand-in for the part of
// Pinocchio that this lab uses (pin.Model with revolute/prismatic unaligned joints, pin.rnea with external forces).
//
// Pinocchio is a large C++ library; the course does not ship it for the C++ track. pinlite reproduces Pinocchio's
// CONVENTIONS exactly, so Checkpoint 5 trains the same conversion as in Python:
//   * spatial vectors are LINEAR FIRST:  motion (v, omega), force (f, n)   (MR uses angular first);
//   * every joint has its own joint frame J_j; twists, accelerations and forces of body j are expressed in J_j;
//   * fext[j] is the wrench applied ON body j by the environment, expressed in J_j, linear first;
//   * index 0 is the fixed base ("universe"), joints are 1..n, so model.njoints = n + 1 (as Pinocchio).
// The test program also compares your results with numbers computed by the real Pinocchio (cpp_cases.json).
#ifndef SBEG461_LAB2_PINLITE_HPP
#define SBEG461_LAB2_PINLITE_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include <utility>
#include <vector>

namespace lab {

struct Robot;  // lab.hpp

/// One joint of a pinlite model (pin.JointModelRevoluteUnaligned / PrismaticUnaligned + its body inertia).
struct PinJoint {
    int parent = 0;                                    ///< parent joint index (0 = universe)
    Eigen::Matrix4d placement = Eigen::Matrix4d::Identity();  ///< J_j seen from J_parent at q = 0 (pin jointPlacement)
    bool revolute = true;                              ///< true: revolute, false: prismatic
    Eigen::Vector3d axis = Eigen::Vector3d::UnitZ();   ///< unit joint axis in J_j
    double mass = 0.0;                                 ///< body mass (pin.Inertia mass)
    Eigen::Vector3d lever = Eigen::Vector3d::Zero();   ///< centre of mass in J_j (pin.Inertia lever)
    Eigen::Matrix3d inertia = Eigen::Matrix3d::Zero(); ///< rotational inertia about the CoM, axes of J_j
};

/// pin.Model: joints[0] is the universe (unused), joints[1..n] the robot joints.
struct PinModel {
    int njoints = 1;                                   ///< n + 1 (as Pinocchio: includes the universe)
    std::vector<PinJoint> joints = std::vector<PinJoint>(1);
    Eigen::Vector3d gravity = Eigen::Vector3d::Zero(); ///< model.gravity.linear
};

/// The `info` dict of the Python pinocchio_model.
struct PinInfo {
    Eigen::Matrix4d T_jn_ee = Eigen::Matrix4d::Identity();  ///< end-effector frame {n+1} seen from the last joint frame J_n
};

/// Build a pinlite model from MR lists.  (Provided -- read it; you will need info.T_jn_ee.)
/// For joint i we create a joint frame J_i with the ORIENTATION of link frame {i} at home and its ORIGIN on the joint
/// axis (revolute) or at the link CoM (prismatic).  The CoM of link i is then at R_i^T (p_i - q_i) in J_i and the
/// rotational inertia block of G_i can be used unchanged.  Returns (model, info).  Zero-pitch joints only.
std::pair<PinModel, PinInfo> pinocchio_model(const Robot& robot);

/// pin.rnea(model, data, q, v, a): joint torques, gravity = model.gravity, no external forces.
Eigen::VectorXd pin_rnea(const PinModel& model, const Eigen::VectorXd& q, const Eigen::VectorXd& v,
                         const Eigen::VectorXd& a);

/// pin.rnea(model, data, q, v, a, fext): fext has model.njoints entries (fext[0] is ignored); fext[j] is the wrench
/// applied ON body j, expressed in joint frame J_j, LINEAR FIRST (f, n).  Throws std::invalid_argument on a wrong size.
Eigen::VectorXd pin_rnea(const PinModel& model, const Eigen::VectorXd& q, const Eigen::VectorXd& v,
                         const Eigen::VectorXd& a, const std::vector<sbeg::Vector6d>& fext);

}  // namespace lab

#endif  // SBEG461_LAB2_PINLITE_HPP
