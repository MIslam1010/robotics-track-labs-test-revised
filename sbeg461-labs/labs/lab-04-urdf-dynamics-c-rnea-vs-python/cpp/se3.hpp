// SBEG 461 Lab 4 -- small SE(3) toolkit in Eigen (PROVIDED, do not edit).
// Notation follows Modern Robotics (MR): twists V = (omega, v), wrenches F = (m, f).
#pragma once
#include <Eigen/Dense>
#include <cmath>

namespace se3 {
using Mat3 = Eigen::Matrix3d;
using Mat4 = Eigen::Matrix4d;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using Vec3 = Eigen::Vector3d;
using Vec6 = Eigen::Matrix<double, 6, 1>;

// [w] (MR Eq. 3.30)
inline Mat3 skew(const Vec3& w) {
  Mat3 W;
  W << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;
  return W;
}

inline Mat4 trans_inv(const Mat4& T) {
  Mat4 Ti = Mat4::Identity();
  const Mat3 Rt = T.topLeftCorner<3, 3>().transpose();
  Ti.topLeftCorner<3, 3>() = Rt;
  Ti.topRightCorner<3, 1>() = -Rt * T.topRightCorner<3, 1>();
  return Ti;
}

// [Ad_T] (MR Definition 3.20)
inline Mat6 adjoint(const Mat4& T) {
  const Mat3 R = T.topLeftCorner<3, 3>();
  const Vec3 p = T.topRightCorner<3, 1>();
  Mat6 A = Mat6::Zero();
  A.topLeftCorner<3, 3>() = R;
  A.bottomLeftCorner<3, 3>() = skew(p) * R;
  A.bottomRightCorner<3, 3>() = R;
  return A;
}

// [ad_V] (MR Eq. 8.38)
inline Mat6 ad(const Vec6& V) {
  Mat6 A = Mat6::Zero();
  const Mat3 W = skew(V.head<3>());
  A.topLeftCorner<3, 3>() = W;
  A.bottomLeftCorner<3, 3>() = skew(V.tail<3>());
  A.bottomRightCorner<3, 3>() = W;
  return A;
}

// e^{[S] theta} for ||omega|| = 1, or omega = 0 and ||v|| = 1 (MR Prop. 3.25)
inline Mat4 exp6(const Vec6& S, double theta) {
  Mat4 T = Mat4::Identity();
  const Vec3 w = S.head<3>(), v = S.tail<3>();
  if (w.norm() < 1e-12) {
    T.topRightCorner<3, 1>() = v * theta;
    return T;
  }
  const Mat3 W = skew(w), W2 = W * W;
  const double s = std::sin(theta), c = std::cos(theta);
  T.topLeftCorner<3, 3>() = Mat3::Identity() + s * W + (1 - c) * W2;
  T.topRightCorner<3, 1>() = (Mat3::Identity() * theta + (1 - c) * W + (theta - s) * W2) * v;
  return T;
}
}  // namespace se3
