// SBEG 46X Lab 11 -- course robot model (C++ track). Given code: do not edit. Numbers identical to course_models.py.
#include "course_models.hpp"

#include <sbeg/mrcpp.hpp>

namespace cm {

Robot ur5_laparoscope() {
    const double W1 = 0.10915, W2 = 0.0823, L1 = 0.425, L2 = 0.39225, H1 = 0.089159, H2 = 0.09465;
    Robot r;
    r.L_tool = 0.30;
    Eigen::Matrix<double, 6, 6> S;  // rows = joints (as the Python array before .T)
    S << 0, 0, 1, 0, 0, 0,
         0, 1, 0, -H1, 0, 0,
         0, 1, 0, -H1, 0, L1,
         0, 1, 0, -H1, 0, L1 + L2,
         0, 0, -1, -W1, L1 + L2, 0,
         0, 1, 0, H2 - H1, 0, L1 + L2;
    r.Slist = S.transpose();
    Eigen::Matrix4d M;
    M << -1, 0, 0, L1 + L2,
         0, 0, 1, W1 + W2,
         0, 1, 0, H1 - H2,
         0, 0, 0, 1;
    Eigen::Matrix4d Tt = Eigen::Matrix4d::Identity();
    Tt(2, 3) = r.L_tool;
    r.M = M * Tt;
    r.Blist = mr::adjoint(mr::trans_inv(r.M)) * r.Slist;
    r.q_home.resize(6);
    r.q_home << 2.6, -2.5, 2.3, -1.4, -1.57, 0.0;
    r.qdot_max = 1.0;
    return r;
}

}  // namespace cm
