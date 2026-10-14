// SBEG 361 Lab 1 -- Rotations and transformations: SO(3)/SE(3) property tests and transform trees (C++ STARTER).
//
// Fill in every function marked  // TODO  (replace the `throw std::logic_error("TODO: ...")` line).
// Do not change function names or signatures (they are declared in lab.hpp). Build and test from the cpp folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
// or from your Lab code folder:   python run_lab.py 1 --cpp
//
// Notation follows Lynch & Park, *Modern Robotics* (MR), Sections 3.1-3.3.1 (see lab.hpp).
// Use Eigen only: do NOT use mrcpp (the C++ port of modern_robotics) or Eigen's general .inverse() inside this
// file -- they are what the tests compare your answers with. The functions marked "Given" (rot_x, quat_xyzw_to_rot,
// frames, _ancestors, _common, path, transform_from_tf and the loaders) are complete and may be used freely.
//
// Eigen reminders:  Eigen::Matrix3d R;  R << a, b, c,  d, e, f,  g, h, i;   (comma initialiser, row by row)
//   R.transpose(), R.determinant(), R.norm() (Frobenius norm), Eigen::Matrix3d::Identity(), A * B (matrix product),
//   v.norm(), R.col(0), R(i, j), T.topLeftCorner<3, 3>() (= T[:3, :3]), T.topRightCorner<3, 1>() (= T[:3, 3]),
//   T.row(3).  Write the type of every Eigen result (Eigen::Matrix4d M = A * B;), never `auto M = A * B;`.
#include "lab.hpp"

#include <sbeg/labio.hpp>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>

namespace lab {

// ---------------------------------------------------------------------------
// Part A -- building R and T
// ---------------------------------------------------------------------------

Eigen::Matrix3d rot_x(double theta) {
    // Given as an example.
    const double c = std::cos(theta), s = std::sin(theta);
    Eigen::Matrix3d R;
    R << 1.0, 0.0, 0.0,
         0.0, c, -s,
         0.0, s, c;
    return R;
}

Eigen::Matrix3d rot_y(double theta) {
    // TODO: build the matrix; watch the sign of the sin terms (it differs from rot_x and rot_z)
    throw std::logic_error("TODO: rot_y");
}

Eigen::Matrix3d rot_z(double theta) {
    // TODO
    throw std::logic_error("TODO: rot_z");
}

Eigen::Matrix3d rot_axis_angle(const Eigen::Vector3d& omega_hat, double theta) {
    // TODO: throw std::invalid_argument("omega_hat must be a unit vector") if |omega_hat.norm() - 1| > 1e-6.
    // Hint: write c = cos(theta), s = sin(theta), v = 1 - c and fill the 9 entries.
    throw std::logic_error("TODO: rot_axis_angle");
}

Eigen::Matrix4d rp_to_trans(const Eigen::Matrix3d& R, const Eigen::Vector3d& p) {
    // TODO: start from Eigen::Matrix4d::Identity() and fill the R block and the p column
    throw std::logic_error("TODO: rp_to_trans");
}

std::pair<Eigen::Matrix3d, Eigen::Vector3d> trans_to_rp(const Eigen::Matrix4d& T) {
    // TODO: return {top-left 3x3 block, top-right 3x1 column}
    throw std::logic_error("TODO: trans_to_rp");
}

// ---------------------------------------------------------------------------
// Part B -- group property tests: SO(3) and SE(3) side by side
// ---------------------------------------------------------------------------

std::pair<double, double> so3_errors(const Eigen::MatrixXd& R) {
    // TODO: throw std::invalid_argument if R is not 3x3 (R.rows(), R.cols()).
    // orth_err = Frobenius norm of R^T R - I   (MR Eq. 3.19)   -> .norm()
    // det      = det(R)                         (MR Eq. 3.21 requires +1)
    // return {orth_err, det};
    throw std::logic_error("TODO: so3_errors");
}

bool is_rotation(const Eigen::MatrixXd& R, double tol) {
    // TODO: return false (do not throw) for a wrong size; otherwise use so3_errors.
    throw std::logic_error("TODO: is_rotation");
}

Eigen::Matrix3d rot_inv(const Eigen::Matrix3d& R) {
    // TODO: do NOT call .inverse()
    throw std::logic_error("TODO: rot_inv");
}

Eigen::Matrix3d project_to_so3(const Eigen::Matrix3d& M) {
    // TODO: Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    //       svd.matrixU() is U and svd.matrixV() is V -- NOT V^T as in NumPy: transpose it yourself.
    //       R = U diag(1, 1, d) V^T with d = sign(det(U V^T))   (Eigen::Vector3d(1, 1, d).asDiagonal())
    throw std::logic_error("TODO: project_to_so3");
}

bool is_se3(const Eigen::MatrixXd& T, double tol) {
    // TODO: return false for a size other than 4x4 (T.rows(), T.cols()); then check R^T R = I,
    //       det R = +1 and the last row (0, 0, 0, 1) within tol (Python: np.allclose with atol=tol)
    throw std::logic_error("TODO: is_se3");
}

Eigen::Matrix4d trans_inv(const Eigen::Matrix4d& T) {
    // TODO: use MR Eq. 3.64 (and your trans_to_rp / rp_to_trans) -- do NOT call .inverse()
    throw std::logic_error("TODO: trans_inv");
}

// ---------------------------------------------------------------------------
// Part C -- the three uses of R and T (MR Sections 3.2.1, 3.3.1)
// ---------------------------------------------------------------------------

Eigen::Vector3d change_frame(const Eigen::Matrix3d& R_ab, const Eigen::Vector3d& p_b) {
    // TODO
    throw std::logic_error("TODO: change_frame");
}

Eigen::Vector3d transform_point(const Eigen::Matrix4d& T, const Eigen::Vector3d& x) {
    // TODO: R x + p
    throw std::logic_error("TODO: transform_point");
}

Eigen::MatrixXd transform_points(const Eigen::Matrix4d& T, const Eigen::MatrixXd& X) {
    // TODO: one point per ROW of X: X R^T + p on every row (Python: x @ R.T + p);
    //       Y.rowwise() += p.transpose() adds p to every row
    throw std::logic_error("TODO: transform_points");
}

Eigen::Matrix3d rotate_in_fixed(const Eigen::Matrix3d& R_sb, const Eigen::Matrix3d& R) {
    // TODO
    throw std::logic_error("TODO: rotate_in_fixed");
}

Eigen::Matrix3d rotate_in_body(const Eigen::Matrix3d& R_sb, const Eigen::Matrix3d& R) {
    // TODO
    throw std::logic_error("TODO: rotate_in_body");
}

Eigen::Matrix4d transform_in_fixed(const Eigen::Matrix4d& T_sb, const Eigen::Matrix4d& T) {
    // TODO: the SE(3) twin of rotate_in_fixed
    throw std::logic_error("TODO: transform_in_fixed");
}

Eigen::Matrix4d transform_in_body(const Eigen::Matrix4d& T_sb, const Eigen::Matrix4d& T) {
    // TODO: the SE(3) twin of rotate_in_body
    throw std::logic_error("TODO: transform_in_body");
}

Eigen::Matrix3d planar_to_se2(const Eigen::Matrix2d& P, const Eigen::Vector2d& p) {
    // TODO: start from Eigen::Matrix3d::Identity(), as in rp_to_trans (P.topLeftCorner<2, 2>(), topRightCorner<2, 1>())
    throw std::logic_error("TODO: planar_to_se2");
}

std::pair<Eigen::Matrix2d, Eigen::Vector2d> planar_compose(const Eigen::Matrix2d& P, const Eigen::Vector2d& p,
                                                           const Eigen::Matrix2d& Q, const Eigen::Vector2d& q) {
    // TODO: return {R, r};
    throw std::logic_error("TODO: planar_compose");
}

// ---------------------------------------------------------------------------
// Part D -- the transform tree (tf2-like)
// ---------------------------------------------------------------------------

Eigen::Matrix3d quat_xyzw_to_rot(const Eigen::Vector4d& q_xyzw) {
    // Given.
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
    // TODO: validate (parent_frame != child, is_se3, single parent, no cycle via _ancestors(parent_frame)) and
    //       throw std::invalid_argument on failure; then store parent[child] = parent_frame; T[child] = T_parent_child;
    throw std::logic_error("TODO: TransformTree::add");
}

std::vector<std::string> TransformTree::_ancestors(const std::string& frame) const {
    std::vector<std::string> chain{frame};
    for (auto it = parent.find(chain.back()); it != parent.end(); it = parent.find(chain.back()))
        chain.push_back(it->second);
    return chain;
}

Eigen::Matrix4d TransformTree::_to_ancestor(const std::string& frame, const std::string& ancestor) const {
    // TODO: walk up from frame to ancestor, left-multiplying by T.at(f) (subscript cancellation); f = parent.at(f)
    throw std::logic_error("TODO: TransformTree::_to_ancestor");
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
    // TODO: common = _common(target, source); combine two _to_ancestor results (one inverted with trans_inv)
    throw std::logic_error("TODO: TransformTree::lookup");
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

// ---------------------------------------------------------------------------
// Part E -- applications: tracker data cleaning and surgical navigation
// ---------------------------------------------------------------------------

std::vector<Eigen::Matrix3d> load_tracker_csv(const std::string& path) {
    // Given.
    const sbeg::CsvTable t = sbeg::read_csv(path);  // skips the header line
    std::vector<Eigen::Matrix3d> Rs;
    for (Eigen::Index k = 0; k < t.data.rows(); ++k) {
        Eigen::Matrix3d R;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) R(i, j) = t.data(k, 1 + 3 * i + j);  // column 0 is the sample id
        Rs.push_back(R);
    }
    return Rs;
}

std::vector<bool> classify_samples(const std::vector<Eigen::Matrix3d>& Rs, double tol) {
    // TODO: one flag per sample (use is_rotation)
    throw std::logic_error("TODO: classify_samples");
}

std::vector<Eigen::Matrix3d> tool_orientations(const std::vector<Eigen::Matrix3d>& R_sm_list,
                                               const Eigen::Matrix3d& R_mt) {
    // TODO: which matrix goes first?  Use subscript cancellation.
    throw std::logic_error("TODO: tool_orientations");
}

std::pair<Eigen::VectorXd, std::vector<Eigen::Matrix4d>> load_track_csv(const std::string& path) {
    // Given.
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

}  // namespace lab
