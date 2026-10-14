// SBEG 361 Lab 0 -- Environment, degrees of freedom, C-space, first contact with mrcpp (C++ STARTER).
//
// Fill in every function marked  // TODO  (replace the `throw std::logic_error("TODO: ...")` line).
// Do not change function names or signatures (they are declared in lab.hpp). Build and test from the cpp folder:
//     cmake -S . -B build -G Ninja && cmake --build build && ./build/lab_tests
// or from your Lab code folder:   python run_lab.py 0 --cpp
//
// Topics: environment check, degrees of freedom (MR Sec. 2.1-2.2, Gruebler's formula MR Eq. 2.4),
// C-space topology and representation (MR Sec. 2.3), first contact with mrcpp (the C++ port of modern_robotics,
// used only in fk_2r_mr, as the Python lab uses modern_robotics only there).
//
// C++ reminders:  std::map<std::string, int> m = {{"R", 1}};  m.find(key) == m.end()  (key missing),  m.count(key);
//   for (const auto& [key, value] : m) { ... }  (loop over a map);  std::to_string(42);  s1 + " x " + s2;
//   Eigen: v.norm(), v.size(), v(i), Eigen::Vector3d(x, y, z).
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lab {

// For pi, write a local constant (M_PI is not standard C++), e.g. inside a namespace { ... } block here:
//     constexpr double PI = 3.14159265358979323846;

// Freedoms f of common joints, MR Table 2.1 (printed p. 17), plus three course extras (see lab.hpp).
// TODO: complete the table (H, C, U, S, RP, E, F are missing).
const std::map<std::string, int> JOINT_FREEDOMS = {{"R", 1}, {"P", 1}};
const std::set<std::string> PLANAR_JOINTS = {"R", "P", "RP", "F"};

// ---------------------------------------------------------------------------
// Checkpoint 1 -- environment
// ---------------------------------------------------------------------------

long cpp_standard_value() {
    // Given.
#if defined(_MSVC_LANG)
    return static_cast<long>(_MSVC_LANG);  // MSVC keeps __cplusplus at 199711 unless /Zc:__cplusplus is given
#else
    return static_cast<long>(__cplusplus);
#endif
}

std::string compiler_name() {
    // Given.
#if defined(__clang__)
    return "clang " + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__) + "." +
           std::to_string(__clang_patchlevel__);
#elif defined(__GNUC__)
    return "gcc " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." +
           std::to_string(__GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
    return "msvc " + std::to_string(_MSC_VER);
#else
    return "unknown compiler";
#endif
}

std::map<std::string, std::string> env_report() {
    // TODO: return {{"cpp_standard", ...}, {"compiler", ...}, {"eigen", ...}};
    // Hint: std::to_string(cpp_standard_value()), compiler_name(), and the Eigen version from the macros
    //       EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION  ->  "3.4.0".
    throw std::logic_error("TODO: env_report");
}

// ---------------------------------------------------------------------------
// Checkpoint 2 -- joints and Gruebler's formula
// ---------------------------------------------------------------------------

int joint_freedoms(const std::string& jtype) {
    // TODO: look jtype up in JOINT_FREEDOMS; throw std::invalid_argument("unknown joint type") if it is missing.
    throw std::logic_error("TODO: joint_freedoms");
}

int joint_constraints(const std::string& jtype, int m) {
    // TODO: throw std::invalid_argument for m not in {3, 6}, and for m == 3 and a joint not in PLANAR_JOINTS.
    throw std::logic_error("TODO: joint_constraints");
}

int grubler(int m, int N, const std::vector<int>& freedoms) {
    // TODO: dof = m (N - 1 - J) + sum_i f_i  with J = number of joints; throw std::invalid_argument for m not in {3, 6}.
    // Hint: write this version first; the two others can convert their input to a list of f_i and call it.
    throw std::logic_error("TODO: grubler (list of f_i)");
}

int grubler(int m, int N, const std::vector<std::string>& joints) {
    // TODO: f_i = joint_freedoms(type) for every joint
    throw std::logic_error("TODO: grubler (list of joint types)");
}

int grubler(int m, int N, const std::map<std::string, int>& counts) {
    // TODO: {type: count} -> each type repeated `count` times
    throw std::logic_error("TODO: grubler ({type: count})");
}

// ---------------------------------------------------------------------------
// Checkpoint 3 -- mechanism table
// ---------------------------------------------------------------------------

std::map<std::string, int> dof_table(const std::string& path) {
    // TODO: const sbeg::json data = sbeg::load_json(path);   then loop:  for (const auto& e : data.at("mechanisms"))
    // Hint: e.at("name").get<std::string>(), e.at("m").get<int>(), e.at("N").get<int>(),
    //       e.at("joints").get<std::map<std::string, int>>()   (the {type: count} object)
    // Use .at("key"), not ["key"]: on a const json, ["key"] does not check that the key exists (it can crash).
    throw std::logic_error("TODO: dof_table");
}

// ---------------------------------------------------------------------------
// Checkpoint 4 -- C-space topology
// ---------------------------------------------------------------------------

double wrap_to_pi(double theta) {
    // TODO: the interval is half-open, [-pi, pi): wrap_to_pi(PI) must give -PI.
    // Careful: std::fmod(x, y) keeps the sign of x, whereas Python's x % y has the sign of y.
    throw std::logic_error("TODO: wrap_to_pi");
}

Eigen::VectorXd wrap_to_pi(const Eigen::VectorXd& theta) {
    // TODO: same as above, entry by entry (result has the same size as theta)
    throw std::logic_error("TODO: wrap_to_pi (vector)");
}

double torus_distance(const Eigen::VectorXd& q1, const Eigen::VectorXd& q2) {
    // TODO: wrap q2 - q1, then the Euclidean norm
    throw std::logic_error("TODO: torus_distance");
}

std::string cspace_topology(const std::vector<std::string>& joints) {
    // TODO: count P and R joints; throw std::invalid_argument for other types or an empty chain.
    throw std::logic_error("TODO: cspace_topology");
}

// ---------------------------------------------------------------------------
// Checkpoint 5 -- a singular representation
// ---------------------------------------------------------------------------

Eigen::Vector3d latlong_to_xyz(double lat, double lon) {
    // TODO
    throw std::logic_error("TODO: latlong_to_xyz");
}

std::pair<double, double> xyz_to_latlong(const Eigen::Vector3d& p) {
    // TODO: normalise p first; std::asin, std::atan2(y, x), std::clamp(z, -1.0, 1.0); return {lat, lon};
    throw std::logic_error("TODO: xyz_to_latlong");
}

double longitude_rate(double lat, double speed) {
    // TODO: the formula you derived in the pre-lab
    throw std::logic_error("TODO: longitude_rate");
}

// ---------------------------------------------------------------------------
// Checkpoint 6 -- first contact with mrcpp: planar 2R arm
// ---------------------------------------------------------------------------

Eigen::Vector2d fk_2r_trig(const Eigen::Vector2d& theta, double L1, double L2) {
    // TODO
    throw std::logic_error("TODO: fk_2r_trig");
}

Eigen::Matrix4d fk_2r_mr(const Eigen::Vector2d& theta, double L1, double L2) {
    // TODO: Eigen::MatrixXd Slist(6, 2);  Slist.col(0) << 0, 0, 1, 0, 0, 0;  Slist.col(1) << ...;
    //       Eigen::Matrix4d M = Eigen::Matrix4d::Identity();  M(0, 3) = ...;
    //       return mr::fkin_space(M, Slist, theta);
    throw std::logic_error("TODO: fk_2r_mr");
}

}  // namespace lab
