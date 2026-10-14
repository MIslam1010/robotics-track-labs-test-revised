// Robot model used in SBEG 46X Lab 5 (C++ track) -- a self-contained copy of the course arm (numbers identical to
// course_models.py and to the course robot models in Course resources). Given code: do not edit.
//
// The course arm is the low-cost desktop UR-type 6R arm of the myCobot-280 class (layout of the UR5 in Modern
// Robotics (MR) Example 4.5, pp. 145-146, scaled down). In this lab it carries a straight digitising pointer on its
// flange; the pointer tip offset p_tip in the flange frame {b} is NOT given: you find it by pivot calibration.
// ILLUSTRATIVE VALUES: the pointer and its nominal length are teaching values, not data of a real product.
// Notation: screw axes S = (w, v), angular part first; space-form PoE T = e^[S1]th1 ... e^[S6]th6 M (MR Eq. 4.14).
// Units in this file: metres and radians (the lab code converts positions to millimetres).
#ifndef SBEG46X_LAB5_COURSE_MODELS_HPP
#define SBEG46X_LAB5_COURSE_MODELS_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

namespace cm {

// ---------------------------------------------------------------- course arm geometry (m)
inline constexpr double W1 = 0.0634, W2 = 0.0456, L1 = 0.110, L2 = 0.096, H1 = 0.1312, H2 = 0.0751;

inline constexpr double SOFT_LIMIT_MARGIN_DEG = 10.0;     // safety checklist rule 1
inline constexpr double RATED_SPEED_DEG_S = 160.0;        // illustrative, all joints
inline constexpr double FIRST_RUN_SPEED_FRACTION = 0.25;  // safety checklist rule 3
inline constexpr int ENCODER_BITS = 12;                   // 4096 counts per turn (0.088 deg)
inline constexpr double ENC_RES = 2.0 * 3.14159265358979323846 / 4096.0;  // rad per count = 2 pi / 2^ENCODER_BITS
inline constexpr double POINTER_NOMINAL_LENGTH = 0.070;   // m, nominal pointer length (illustrative)

/// Hardware joint limits (deg), 6 x 2: (-165, 165) for joints 1-5, (-175, 175) for joint 6.
Eigen::MatrixXd limits_deg();

/// "Ready" configuration (rad): (0, -0.6, 1.4, -2.4, -1.57, 0)  (Python: course_models.Q_HOME).
Eigen::VectorXd q_home();

/// Course arm PoE model of the FLANGE frame {b} (tool = 0)  (Python: the dict of course_arm_kinematics()).
struct CourseArm {
    Eigen::MatrixXd Slist;     ///< 6 x 6 space screw axes, one per COLUMN: S_i = (w_i, v_i), metres
    Eigen::Matrix4d M;         ///< home pose of the flange frame {b} in the base frame {s} (metres)
    Eigen::MatrixXd limits;    ///< 6 x 2 software joint limits (rad) = hardware limits minus the 10 deg margin
    Eigen::VectorXd qdot_max;  ///< 6 joint-speed limits (rad/s) = 25 % of the rated 160 deg/s (first-run rule)
};

/// The course arm (MR Example 4.5 form).
CourseArm course_arm_kinematics();

}  // namespace cm

#endif  // SBEG46X_LAB5_COURSE_MODELS_HPP
