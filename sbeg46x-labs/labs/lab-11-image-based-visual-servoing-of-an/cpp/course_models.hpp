// Robot model used in SBEG 46X Lab 11 (C++ track) -- a self-contained copy of the shared course model (numbers
// identical to course_models.py and to the course robot models in Course resources). Given code: do not edit.
//
//   ur5_laparoscope()  UR5 (MR Example 4.5 kinematics with the full-precision values of the MR software:
//                      W1 = 0.10915, W2 = 0.0823, L1 = 0.425, L2 = 0.39225, H1 = 0.089159, H2 = 0.09465 m) holding a
//                      0.30 m laparoscope along z_b of the flange. {b} is at the scope TIP; in Lab 11 the endoscope
//                      camera frame {c} is {b} (optical axis = z_b, a 0-degree scope). Kinematics only.
// ILLUSTRATIVE VALUES for the tool: teaching values, not the data of a real product.
// Notation: Modern Robotics (MR) -- screw axes S = (w, v) (MR Sec. 3.3.2), home pose M, body screws B = [Ad_{M^-1}] S.
#ifndef SBEG46X_LAB11_COURSE_MODELS_HPP
#define SBEG46X_LAB11_COURSE_MODELS_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

namespace cm {

/// The UR5 + laparoscope (Python: the dict of ur5_laparoscope()).
struct Robot {
    Eigen::MatrixXd Slist;    ///< 6 x 6 space screw axes, one per COLUMN
    Eigen::MatrixXd Blist;    ///< 6 x 6 body screw axes, one per COLUMN: [Ad_{M^-1}] Slist
    Eigen::Matrix4d M;        ///< home pose of the scope-tip frame {b} in the base frame {s}
    Eigen::VectorXd q_home;   ///< start pose of Lab 11 (scope pointing down, z_b ~ -z_s)
    double qdot_max = 1.0;    ///< joint-speed limit (rad/s), all joints
    double L_tool = 0.30;     ///< scope length beyond the flange (m)
};

/// UR5 + 0.30 m laparoscope ({b} at the scope tip).
Robot ur5_laparoscope();

}  // namespace cm

#endif  // SBEG46X_LAB11_COURSE_MODELS_HPP
