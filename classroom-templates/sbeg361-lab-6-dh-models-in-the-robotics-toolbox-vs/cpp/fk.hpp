// SBEG 361 Lab 6 -- C++/Eigen forward kinematics of the course arm (PoE and standard DH).
// Numbers identical to the course robot models (Course resources): course_arm (illustrative rated values not needed here).
#pragma once

#include <Eigen/Dense>
#include <array>

namespace sbeg361 {

using Mat4 = Eigen::Matrix4d;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat6 = Eigen::Matrix<double, 6, 6>;

// Course arm geometry (m)
constexpr double W1 = 0.0634, W2 = 0.0456, L1 = 0.110, L2 = 0.096, H1 = 0.1312, H2 = 0.0751;
constexpr double PI = 3.14159265358979323846;

// Space screw axes as columns S_i = (w_i, v_i) (MR Example 4.5 layout) and the home pose M.
Mat6 courseArmSlist();
Mat4 courseArmM();

// One standard DH row (Corke Eq. 7.1): theta (offset), d, a, alpha.
struct DHRow { double offset, d, a, alpha; };
std::array<DHRow, 6> courseArmDH();   // the standard table of Step 3 (provided in robot_data.cpp)
Mat4 courseArmDHTool();               // constant tool transform Rot(z, pi)

// --- functions you implement (fk_starter.cpp) -------------------------------------------------
Mat4 dhStd(double theta, double d, double a, double alpha);        // Corke Eq. 7.2
Mat4 matrixExp6(const Vec6& S, double theta);                      // MR Eq. 3.88 (||w|| = 1 or w = 0)
Mat4 fkPoE(const Mat6& Slist, const Mat4& M, const Vec6& q);       // MR Eq. 4.14
Mat4 fkDH(const std::array<DHRow, 6>& table, const Mat4& tool, const Vec6& q);   // Corke Eq. 7.4

}  // namespace sbeg361
