// SBEG 361 Lab 2 -- Angular velocities and twists (C++ STARTER).
//
// Fill in every function marked  // TODO  (replace the `throw std::logic_error("TODO: ...")` line).
// Do not change function names or signatures (they are declared in lab.hpp). Build and test from the cpp folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
// or from your Lab code folder:   python run_lab.py 2 --cpp
//
// You may NOT use mrcpp (the C++ port of modern_robotics) or Eigen's general .inverse() inside these functions --
// they are what the tests compare against. Everything in the GIVEN section (your Lab 1 SE(3) functions and
// transform tree, the loaders and the analysis helpers) is complete and may be used freely.
// Notation follows Lynch & Park, *Modern Robotics* (MR), Sections 3.2.2 and 3.3.2 (see lab.hpp).
//
// Eigen reminders:  T.topLeftCorner<3, 3>() (= T[:3, :3]),  T.topRightCorner<3, 1>() (= T[:3, 3]),
//   R.transpose(), V.head<3>() (= V[:3]), V.tail<3>() (= V[3:]), a.cross(b), a.dot(b), v.norm(),
//   Eigen::Matrix4d::Zero(), Matrix6d::Zero(), A.block<3, 3>(3, 0) (= A[3:, :3]), X.row(i), X.rows(),
//   Vector6d V;  V << w, v;  (stack two 3-vectors).  Write the type of every Eigen result
//   (Eigen::Matrix4d M = A * B;), never `auto M = A * B;`.
#include "lab.hpp"

#include <sbeg/labio.hpp>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>

namespace lab {

// ================================================================ GIVEN (from Lab 1)
Eigen::Matrix3d vec_to_so3(const Eigen::Vector3d& omega) {
    Eigen::Matrix3d m;
    m << 0.0, -omega(2), omega(1),
         omega(2), 0.0, -omega(0),
         -omega(1), omega(0), 0.0;
    return m;
}

Eigen::Vector3d so3_to_vec(const Eigen::Matrix3d& so3mat) {
    return Eigen::Vector3d(so3mat(2, 1), so3mat(0, 2), so3mat(1, 0));
}

Eigen::Matrix4d rp_to_trans(const Eigen::Matrix3d& R, const Eigen::Vector3d& p) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.topLeftCorner<3, 3>() = R;
    T.topRightCorner<3, 1>() = p;
    return T;
}

std::pair<Eigen::Matrix3d, Eigen::Vector3d> trans_to_rp(const Eigen::Matrix4d& T) {
    return {T.topLeftCorner<3, 3>(), T.topRightCorner<3, 1>()};
}

Eigen::Matrix4d trans_inv(const Eigen::Matrix4d& T) {
    const auto [R, p] = trans_to_rp(T);
    const Eigen::Matrix3d Rt = R.transpose();
    return rp_to_trans(Rt, -Rt * p);
}

namespace {
/// np.allclose(a, b, atol=tol) (NumPy's default rtol = 1e-5 is kept).
bool allclose(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b, double tol) {
    return ((a - b).cwiseAbs().array() <= tol + 1e-5 * b.cwiseAbs().array()).all();
}
}  // namespace

bool is_se3(const Eigen::MatrixXd& T, double tol) {
    if (T.rows() != 4 || T.cols() != 4) return false;
    const Eigen::Matrix3d R = T.topLeftCorner(3, 3);
    return allclose(R.transpose() * R, Eigen::Matrix3d::Identity(), tol) && std::abs(R.determinant() - 1.0) < tol &&
           allclose(T.row(3), Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0), tol);
}

Eigen::Vector3d transform_point(const Eigen::Matrix4d& T, const Eigen::Vector3d& x) {
    const auto [R, p] = trans_to_rp(T);
    return R * x + p;
}

Eigen::MatrixXd transform_points(const Eigen::Matrix4d& T, const Eigen::MatrixXd& X) {
    const auto [R, p] = trans_to_rp(T);
    Eigen::MatrixXd Y = X * R.transpose();
    Y.rowwise() += p.transpose();
    return Y;
}

Eigen::Matrix3d quat_xyzw_to_rot(const Eigen::Vector4d& q_xyzw) {
    const Eigen::Vector4d q = q_xyzw / q_xyzw.norm();
    const double x = q(0), y = q(1), z = q(2), w = q(3);
    Eigen::Matrix3d R;
    R << w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (w * y + x * z),
         2 * (w * z + x * y), w * w - x * x + y * y - z * z, 2 * (y * z - w * x),
         2 * (x * z - w * y), 2 * (w * x + y * z), w * w - x * x - y * y + z * z;
    return R;
}

std::set<std::string> TransformTree::frames() const {
    std::set<std::string> f;
    for (const auto& [c, p] : parent) {
        f.insert(c);
        f.insert(p);
    }
    return f;
}

void TransformTree::add(const std::string& parent_frame, const std::string& child,
                        const Eigen::Matrix4d& T_parent_child) {
    if (parent_frame == child) throw std::invalid_argument("a frame cannot be its own parent");
    if (!is_se3(T_parent_child))
        throw std::invalid_argument("transform " + parent_frame + "->" + child + " is not in SE(3)");
    const auto it = parent.find(child);
    if (it != parent.end() && it->second != parent_frame)
        throw std::invalid_argument("frame '" + child + "' already has parent '" + it->second + "'");
    const std::vector<std::string> up = _ancestors(parent_frame);
    if (std::find(up.begin(), up.end(), child) != up.end())
        throw std::invalid_argument("edge " + parent_frame + "->" + child + " would create a cycle");
    parent[child] = parent_frame;
    T[child] = T_parent_child;
}

std::vector<std::string> TransformTree::_ancestors(const std::string& frame) const {
    std::vector<std::string> chain{frame};
    for (auto it = parent.find(chain.back()); it != parent.end(); it = parent.find(chain.back()))
        chain.push_back(it->second);
    return chain;
}

Eigen::Matrix4d TransformTree::_to_ancestor(const std::string& frame, const std::string& ancestor) const {
    Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
    std::string f = frame;
    while (f != ancestor) {
        result = T.at(f) * result;  // T_parent_f T_f_frame = T_parent_frame
        f = parent.at(f);
    }
    return result;
}

std::string TransformTree::_common(const std::string& target, const std::string& source) const {
    const std::set<std::string> all = frames();
    for (const std::string& f : {target, source})
        if (all.count(f) == 0) throw KeyError("unknown frame '" + f + "'");
    const std::vector<std::string> up_s = _ancestors(source);
    for (const std::string& f : _ancestors(target))
        if (std::find(up_s.begin(), up_s.end(), f) != up_s.end()) return f;
    throw LookupError("frames '" + target + "' and '" + source + "' are not connected");
}

std::vector<std::string> TransformTree::path(const std::string& target, const std::string& source) const {
    const std::string common = _common(target, source);
    const std::vector<std::string> up_t = _ancestors(target), up_s = _ancestors(source);
    std::vector<std::string> out;
    for (const std::string& f : up_t) {
        out.push_back(f);
        if (f == common) break;
    }
    const auto it = std::find(up_s.begin(), up_s.end(), common);
    for (auto r = std::make_reverse_iterator(it); r != up_s.rend(); ++r) out.push_back(*r);
    return out;
}

Eigen::Matrix4d TransformTree::lookup(const std::string& target, const std::string& source) const {
    const std::string common = _common(target, source);
    return trans_inv(_to_ancestor(target, common)) * _to_ancestor(source, common);
}

Eigen::Matrix4d transform_from_tf(const Eigen::Vector3d& translation, const Eigen::Vector4d& rotation_xyzw) {
    return rp_to_trans(quat_xyzw_to_rot(rotation_xyzw), translation);
}

TransformTree load_tree_json(const std::string& path) {
    const sbeg::json data = sbeg::load_json(path);
    TransformTree tree;
    for (const auto& e : data.at("transforms"))
        tree.add(e.at("parent").get<std::string>(), e.at("child").get<std::string>(),
                 transform_from_tf(sbeg::as<Eigen::Vector3d>(e.at("translation")),
                                   sbeg::as<Eigen::Vector4d>(e.at("rotation_xyzw"))));
    return tree;
}

std::pair<Eigen::VectorXd, std::vector<Eigen::Matrix4d>> load_track_csv(const std::string& path) {
    const sbeg::CsvTable d = sbeg::read_csv(path);  // skips the header line
    const Eigen::VectorXd t = d.data.col(0);
    std::vector<Eigen::Matrix4d> Ts;
    for (Eigen::Index k = 0; k < d.data.rows(); ++k) {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) T(i, j) = d.data(k, 1 + 3 * i + j);  // r11..r33, row-major
            T(i, 3) = d.data(k, 10 + i);                                     // px, py, pz
        }
        Ts.push_back(T);
    }
    return {t, Ts};
}

double axis_angle_between(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    const double c = a.dot(b) / (a.norm() * b.norm());
    return std::acos(std::max(-1.0, std::min(1.0, c)));
}

double point_line_distance(const Eigen::Vector3d& x, const Eigen::Vector3d& q, const Eigen::Vector3d& s_hat) {
    const Eigen::Vector3d s = s_hat / s_hat.norm();
    const Eigen::Vector3d d = x - q;
    return (d - d.dot(s) * s).norm();
}

// ================================================================ Step 1: angular velocity
std::pair<Eigen::Vector3d, Eigen::Vector3d> angular_velocities(const Eigen::Matrix3d& R, const Eigen::Matrix3d& Rdot) {
    // TODO: [omega_s] = Rdot R^T and [omega_b] = R^T Rdot (MR Eqs. 3.37-3.38); read the vectors with so3_to_vec
    throw std::logic_error("TODO: angular_velocities");
}

// ================================================================ Step 2: se(3) and the adjoint
Eigen::Matrix4d vec_to_se3(const Vector6d& V) {
    // TODO: top-left block [omega] (vec_to_so3), top-right column v, bottom row zeros
    throw std::logic_error("TODO: vec_to_se3");
}

Vector6d se3_to_vec(const Eigen::Matrix4d& se3mat) {
    // TODO: inverse of vec_to_se3
    throw std::logic_error("TODO: se3_to_vec");
}

Matrix6d adjoint(const Eigen::Matrix4d& T) {
    // TODO: MR Def. 3.20: blocks R, 0, [p]R, R
    throw std::logic_error("TODO: adjoint");
}

// ================================================================ Step 3: twists and point velocities
std::pair<Vector6d, Vector6d> twists_from_T_Tdot(const Eigen::Matrix4d& T, const Eigen::Matrix4d& Tdot) {
    // TODO: [V_s] = Tdot T^-1, [V_b] = T^-1 Tdot (MR Eqs. 3.81-3.82); use trans_inv and se3_to_vec
    throw std::logic_error("TODO: twists_from_T_Tdot");
}

Eigen::Vector3d point_velocity(const Vector6d& V, const Eigen::Vector3d& x) {
    // TODO: omega x x + v with omega = V.head<3>(), v = V.tail<3>()
    throw std::logic_error("TODO: point_velocity");
}

Eigen::MatrixXd point_velocities(const Vector6d& V, const Eigen::MatrixXd& X) {
    // TODO: an N x 3 result; row i = point_velocity(V, row i of X) (throw std::invalid_argument if X.cols() != 3)
    throw std::logic_error("TODO: point_velocities");
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> twists_from_samples(const std::vector<Eigen::Matrix4d>& Ts, double dt) {
    // TODO: central difference Tdot_k = (T_{k+1}-T_{k-1})/(2 dt); read omega from the skew part (M - M^T)/2;
    //       return two (N-2) x 6 matrices, row k-1 = twist at sample k
    throw std::logic_error("TODO: twists_from_samples");
}

// ================================================================ Step 4: screws
Vector6d screw_to_twist(const Eigen::Vector3d& q, const Eigen::Vector3d& s_hat, double h, double thetadot) {
    // TODO: normalise s_hat; h = INF (std::isinf) -> (0, s_hat*thetadot); else (s_hat, -s_hat x q + h s_hat)*thetadot
    throw std::logic_error("TODO: screw_to_twist");
}

Screw twist_to_screw(const Vector6d& V) {
    // TODO: case |omega| > TOL: thetadot = |omega|, h = omega.v/|omega|^2, q = omega x v/|omega|^2;
    //       case omega = 0: h = INF, q empty (sc.q.reset()); zero twist -> throw std::invalid_argument
    throw std::logic_error("TODO: twist_to_screw");
}

}  // namespace lab
