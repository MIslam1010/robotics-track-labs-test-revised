// Robot model used in SBEG 46X Lab 5 (C++ track): the course arm. Given code: do not edit (see course_models.hpp).
#include "course_models.hpp"

namespace cm {

Eigen::MatrixXd limits_deg() {
    Eigen::MatrixXd L(6, 2);
    L << -165.0, 165.0,
         -165.0, 165.0,
         -165.0, 165.0,
         -165.0, 165.0,
         -165.0, 165.0,
         -175.0, 175.0;
    return L;
}

Eigen::VectorXd q_home() {
    Eigen::VectorXd q(6);
    q << 0.0, -0.6, 1.4, -2.4, -1.57, 0.0;
    return q;
}

CourseArm course_arm_kinematics() {
    constexpr double DEG = 3.14159265358979323846 / 180.0;
    CourseArm k;
    Eigen::MatrixXd S(6, 6);  // one ROW per joint here, transposed below (as the .T in Python)
    S << 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, -H1, 0.0, 0.0,
         0.0, 1.0, 0.0, -H1, 0.0, L1,
         0.0, 1.0, 0.0, -H1, 0.0, L1 + L2,
         0.0, 0.0, -1.0, -W1, L1 + L2, 0.0,
         0.0, 1.0, 0.0, H2 - H1, 0.0, L1 + L2;
    k.Slist = S.transpose();
    k.M << -1.0, 0.0, 0.0, L1 + L2,
           0.0, 0.0, 1.0, W1 + W2,
           0.0, 1.0, 0.0, H1 - H2,
           0.0, 0.0, 0.0, 1.0;
    const Eigen::MatrixXd lim = limits_deg();
    k.limits.resize(6, 2);
    k.limits.col(0) = (lim.col(0).array() + SOFT_LIMIT_MARGIN_DEG) * DEG;
    k.limits.col(1) = (lim.col(1).array() - SOFT_LIMIT_MARGIN_DEG) * DEG;
    k.qdot_max = Eigen::VectorXd::Constant(6, FIRST_RUN_SPEED_FRACTION * RATED_SPEED_DEG_S * DEG);
    return k;
}

}  // namespace cm
