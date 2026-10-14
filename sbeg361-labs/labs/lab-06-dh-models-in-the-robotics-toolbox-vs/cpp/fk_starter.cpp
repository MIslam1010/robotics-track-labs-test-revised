// SBEG 361 Lab 6 -- C++/Eigen FK (STARTER).  Replace every TODO; keep the signatures of fk.hpp.
#include <cmath>
#include <stdexcept>

#include "fk.hpp"

namespace sbeg361 {

Mat4 dhStd(double theta, double d, double a, double alpha) {
    // TODO: the standard DH link transform of Corke Eq. 7.2 (same as dh_std in Python).
    //       Fill a Mat4 with the comma initialiser:  Mat4 A;  A << ..., ..., ..., ...;
    (void)theta; (void)d; (void)a; (void)alpha;
    throw std::logic_error("dhStd not implemented");
}

Mat4 matrixExp6(const Vec6& S, double theta) {
    // TODO: e^{[S]theta}.  w = S.head<3>(), v = S.tail<3>().
    //       Prismatic (w = 0): rotation I, translation v*theta.
    //       Revolute: R = I + sin(theta)[w] + (1 - cos(theta))[w]^2           (MR Eq. 3.51)
    //                 p = (I theta + (1 - cos)[w] + (theta - sin)[w]^2) v     (MR Eq. 3.88)
    //       Write a small skew() helper; use T.block<3,3>(0,0) and T.block<3,1>(0,3).
    (void)S; (void)theta;
    throw std::logic_error("matrixExp6 not implemented");
}

Mat4 fkPoE(const Mat6& Slist, const Mat4& M, const Vec6& q) {
    // TODO: T = e^{[S1]q1} ... e^{[S6]q6} M  (MR Eq. 4.14).  Slist.col(i) is S_i.
    (void)Slist; (void)M; (void)q;
    throw std::logic_error("fkPoE not implemented");
}

Mat4 fkDH(const std::array<DHRow, 6>& table, const Mat4& tool, const Vec6& q) {
    // TODO: T = A_1(offset_1 + q_1) ... A_6(offset_6 + q_6) * tool  (base = I).
    (void)table; (void)tool; (void)q;
    throw std::logic_error("fkDH not implemented");
}

}  // namespace sbeg361
