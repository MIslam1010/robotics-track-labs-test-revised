// SBEG 361 Lab 6 -- course-arm data (provided).
// from the course robot models (Course resources): course_arm  (numbers identical)
#include "fk.hpp"

namespace sbeg361 {

Mat6 courseArmSlist() {
    Mat6 S;
    // each row below is one screw axis (w, v); the matrix is transposed so that columns are S_i
    S << 0, 0, 1, 0, 0, 0,
         0, 1, 0, -H1, 0, 0,
         0, 1, 0, -H1, 0, L1,
         0, 1, 0, -H1, 0, L1 + L2,
         0, 0, -1, -W1, L1 + L2, 0,
         0, 1, 0, H2 - H1, 0, L1 + L2;
    return S.transpose();
}

Mat4 courseArmM() {
    Mat4 M;
    M << -1, 0, 0, L1 + L2,
          0, 0, 1, W1 + W2,
          0, 1, 0, H1 - H2,
          0, 0, 0, 1;
    return M;
}

std::array<DHRow, 6> courseArmDH() {
    // standard DH: offset, d, a, alpha  (same zero pose and joint signs as the PoE model)
    return {{{0, H1, 0, -PI / 2},
             {0, 0, L1, 0},
             {0, 0, L2, 0},
             {0, W1, 0, -PI / 2},
             {0, H2, 0, PI / 2},
             {0, W2, 0, 0}}};
}

Mat4 courseArmDHTool() {
    Mat4 T = Mat4::Identity();
    T(0, 0) = -1;
    T(1, 1) = -1;   // Rot(z, pi)
    return T;
}

}  // namespace sbeg361
