// SBEG 46X Lab 1 -- robot model (C++ track).  GIVEN code: do not edit.
//
// C++ copy of course_models.py: the course arm (kinematics and rated speed only), numbers identical.
// The course arm is the low-cost desktop UR-type 6R arm of the myCobot-280 class that you used in SBEG 361 and
// SBEG 461 (layout of the UR5 in MR Example 4.5, pp. 145-146, scaled down).  In this lab it holds a straight
// laparoscope of length TOOL_LENGTH along the flange axis z_b, so {b} sits at the laparoscope tip.
// ILLUSTRATIVE VALUES: the rated speed and the 0.12 m teaching laparoscope are teaching values, not data of a real
// product.  Robot data and the hardware safety checklist: "Course resources: Course robot models and hardware safety
// checklist".
#include "lab.hpp"

namespace lab {
namespace {
constexpr double PI = 3.14159265358979323846;
constexpr double W1 = 0.0634, W2 = 0.0456, L1 = 0.110, L2 = 0.096, H1 = 0.1312, H2 = 0.0751;  // m
constexpr double SOFT_LIMIT_MARGIN_DEG = 10.0;    // safety checklist rule 1
constexpr double RATED_SPEED_DEG_S = 160.0;       // illustrative, all joints
constexpr double FIRST_RUN_SPEED_FRACTION = 0.25; // safety checklist rule 3
}  // namespace

Robot course_arm_kinematics(double tool) {
    Robot r;
    Eigen::Matrix<double, 6, 6> S;  // one ROW per joint here (as written in Python), transposed below
    S << 0, 0, 1, 0, 0, 0,
         0, 1, 0, -H1, 0, 0,
         0, 1, 0, -H1, 0, L1,
         0, 1, 0, -H1, 0, L1 + L2,
         0, 0, -1, -W1, L1 + L2, 0,
         0, 1, 0, H2 - H1, 0, L1 + L2;
    r.Slist = S.transpose();
    r.M << -1, 0, 0, L1 + L2,
           0, 0, 1, W1 + W2,
           0, 1, 0, H1 - H2,
           0, 0, 0, 1;
    r.M.block<3, 1>(0, 3) += tool * r.M.block<3, 1>(0, 2);  // M @ Trans(0, 0, tool)
    r.limits.resize(6, 2);
    for (int i = 0; i < 6; ++i) {
        const double hw = (i < 5 ? 165.0 : 175.0);  // hardware joint limits (deg)
        r.limits(i, 0) = (-hw + SOFT_LIMIT_MARGIN_DEG) * PI / 180.0;
        r.limits(i, 1) = (hw - SOFT_LIMIT_MARGIN_DEG) * PI / 180.0;
    }
    r.qdot_max = Eigen::VectorXd::Constant(6, FIRST_RUN_SPEED_FRACTION * RATED_SPEED_DEG_S * PI / 180.0);
    r.tool = tool;
    return r;
}

}  // namespace lab
