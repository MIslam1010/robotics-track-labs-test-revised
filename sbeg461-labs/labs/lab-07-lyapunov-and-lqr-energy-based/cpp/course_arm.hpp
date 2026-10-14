// SBEG 461 Lab 7 -- C++ track: GIVEN code (do not edit).  The C++ copy of the local model file course_arm.py:
// the SBEG course arm (desktop UR-type 6R, myCobot-280 class) with its illustrative dynamics.  Numbers are IDENTICAL
// to course_arm.py and to the shared course robot models (Course resources); the tests check the model built from
// it against the numbers of the Python model.
//
// Kinematics: MR Example 4.5 layout (pp. 145-146) scaled down, W1 = 0.0634, W2 = 0.0456, L1 = 0.110,
// L2 = 0.096, H1 = 0.1312, H2 = 0.0751 m.  Rated speed, torques, masses and inertias are ILLUSTRATIVE VALUES;
// replace them with the commissioned / identified values published on the course site.
#ifndef SBEG461_LAB07_COURSE_ARM_HPP
#define SBEG461_LAB07_COURSE_ARM_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings
#include <sbeg/mrcpp.hpp>

#include <string>

namespace course_arm {

constexpr double G0 = 9.81;
constexpr double DEG = 3.14159265358979323846 / 180.0;

constexpr double COURSE_ARM_RATED_SPEED_DEG_S = 160.0;  ///< illustrative; re-measure at commissioning
constexpr double COURSE_ARM_PAYLOAD_MAX = 0.25;         ///< kg, illustrative

/// A robot model: the Python model dict.  Every model fills name, n, Mlist, Glist, Slist and g; the course arm
/// also fills M, limits (n x 2, rad), qdot_max (rad/s), tau_max (N m), q_home, payload_max and the derating values.
struct Robot {
    std::string name;
    int n = 0;
    Eigen::MatrixXd Slist;                         ///< 6 x n, one screw axis per COLUMN
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    Eigen::MatrixXd limits;                        ///< n x 2: [lower, upper] joint limits (rad)
    Eigen::VectorXd qdot_max;                      ///< rated joint speeds (rad/s)
    Eigen::VectorXd tau_max;                       ///< rated joint torques (N m)
    Eigen::VectorXd q_home;
    mr::TransformList Mlist;                       ///< n + 1 frames M_{i-1,i}
    mr::InertiaList Glist;                         ///< n spatial inertias
    Eigen::Vector3d g = Eigen::Vector3d::Zero();   ///< gravity vector in {s}
    double payload_max = 0.0;
    double derated_speed_fraction = 0.0;
    double derated_current_fraction = 0.0;
};

/// G = diag(I_b, m I3) in a CoM frame (MR Eq. 8.32).
mr::Matrix6d body_inertia(double m, const Eigen::Vector3d& Ib_diag);
/// Spatial inertia of a solid box of mass m and edges a, b, c.
mr::Matrix6d box_inertia(double m, double a, double b, double c);
/// G_a = [Ad_Tba]^T G_b [Ad_Tba] (MR Eq. 8.42).
mr::Matrix6d transform_inertia(const mr::Matrix6d& G_b, const Eigen::Matrix4d& T_ba);

/// THE course arm.  `payload` (kg) adds a point mass at the tool point (same rule as the shared model);
/// `tool` (m) moves the tool point along the flange z axis.
Robot course_arm(double tool = 0.0, double payload = 0.0);

}  // namespace course_arm

#endif  // SBEG461_LAB07_COURSE_ARM_HPP
