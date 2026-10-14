// SBEG 361 Lab 0 -- Environment, degrees of freedom, C-space and first contact with modern_robotics
// (C++ track): declarations.
//
// The same functions as starter.py / test_lab.py, in namespace `lab`, with Eigen and standard-library types instead
// of Python dicts, lists and NumPy arrays. Write your code in starter.cpp. Do not change these declarations:
// test_lab.cpp calls them.
//
// Topics: environment check, degrees of freedom (MR Sec. 2.1-2.2, Gruebler's formula MR Eq. 2.4),
// C-space topology and representation (MR Sec. 2.3), first contact with mrcpp (the C++ port of modern_robotics).
//
// Python -> C++: ValueError -> std::invalid_argument; dict -> std::map; list of strings -> std::vector<std::string>;
// a tuple -> std::pair; a NumPy vector -> Eigen::VectorXd / Eigen::Vector2d / Eigen::Vector3d.
#ifndef SBEG361_LAB0_HPP
#define SBEG361_LAB0_HPP

#include <sbeg/eigen.hpp>  // always first: includes Eigen/Dense with the course settings

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace lab {

// Freedoms f of common joints, MR Table 2.1 (printed p. 17), plus three course extras:
//   RP = revolute-prismatic pair treated as one 2-dof joint (MR Ex 2.3 (ii), Ex 2.5 (ii))
//   E  = planar contact joint (a body sliding on a plane: x, y and rotation about the normal)
//   F  = rigid attachment ("weld", rigid grasp), f = 0
// Defined in starter.cpp (you complete it there).
extern const std::map<std::string, int> JOINT_FREEDOMS;
/// Joint types that are meaningful in a planar mechanism (given).
extern const std::set<std::string> PLANAR_JOINTS;

// ---------------------------------------------------------------------------
// Checkpoint 1 -- environment
// ---------------------------------------------------------------------------

/// The C++ standard the compiler uses, e.g. 201703 for C++17 (given: reads __cplusplus, or _MSVC_LANG on MSVC).
long cpp_standard_value();

/// Name and version of the compiler, e.g. "clang 15.0.0" (given).
std::string compiler_name();

/// Return the versions of the C++ standard, the compiler and Eigen as non-empty strings.
/// Keys: "cpp_standard" (e.g. "201703"), "compiler", "eigen" (e.g. "3.4.0").
std::map<std::string, std::string> env_report();

// ---------------------------------------------------------------------------
// Checkpoint 2 -- joints and Gruebler's formula
// ---------------------------------------------------------------------------

/// Number of freedoms f provided by a joint of type `jtype` (MR Table 2.1).
/// Throws std::invalid_argument for an unknown type.
int joint_freedoms(const std::string& jtype);

/// Number of constraints c = m - f provided by a joint (MR Table 2.1).
/// m = 3 for planar, 6 for spatial mechanisms. For planar mechanisms only R, P, RP and F are meaningful
/// (H, C, U, S are "N/A" in MR Table 2.1): throw std::invalid_argument for the others, and for m not in {3, 6}.
int joint_constraints(const std::string& jtype, int m);

/// Gruebler's formula, MR Eq. 2.4:  dof = m (N - 1 - J) + sum_i f_i.
///   m      : 3 (planar) or 6 (spatial); throw std::invalid_argument otherwise
///   N      : number of links, ground INCLUDED
///   joints : one of three forms (three overloads, as the three forms Python accepts):
///            a list of joint-type strings  {"R", "R", "RP"},
///            a list of integers f_i        {3, 3, 1},
///            a map {type: count}           {{"R", 4}}.
/// Write the variables with their type, e.g. std::vector<std::string> js = {"R", "R"}; lab::grubler(3, 3, js);
int grubler(int m, int N, const std::vector<std::string>& joints);
int grubler(int m, int N, const std::vector<int>& freedoms);
int grubler(int m, int N, const std::map<std::string, int>& counts);

// ---------------------------------------------------------------------------
// Checkpoint 3 -- mechanism table
// ---------------------------------------------------------------------------

/// Read a mechanisms JSON file (see data/mechanisms.json) and return {name: dof}.
std::map<std::string, int> dof_table(const std::string& path);

// ---------------------------------------------------------------------------
// Checkpoint 4 -- C-space topology
// ---------------------------------------------------------------------------

/// Wrap an angle to the interval [-pi, pi).
double wrap_to_pi(double theta);

/// Wrap every entry of a vector of angles to [-pi, pi).
Eigen::VectorXd wrap_to_pi(const Eigen::VectorXd& theta);

/// Shortest distance between two configurations of an nR robot on T^n (no joint limits).
/// Each joint difference is wrapped to [-pi, pi) before taking the Euclidean norm, because
/// theta and theta + 2 pi are the same point of S^1 (MR Sec. 2.3.1).
double torus_distance(const Eigen::VectorXd& q1, const Eigen::VectorXd& q2);

/// Topology of the C-space of an open chain of R and P joints, ignoring joint limits.
/// Each P joint contributes R^1, each R joint contributes S^1, and S^1 x ... x S^1 (n copies) = T^n
/// (MR Sec. 2.3.1). Format: "R^k x T^n", omitting a factor that does not appear; one R joint gives
/// "S^1" (not "T^1"). Examples: {"P", "R"} -> "R^1 x S^1"; {"R", "R"} -> "T^2"; {"P"} -> "R^1".
/// Throws std::invalid_argument for other joint types or an empty chain.
std::string cspace_topology(const std::vector<std::string>& joints);

// ---------------------------------------------------------------------------
// Checkpoint 5 -- a singular representation
// ---------------------------------------------------------------------------

/// Point on the unit sphere S^2 from latitude/longitude in radians (explicit parametrisation).
Eigen::Vector3d latlong_to_xyz(double lat, double lon);

/// Latitude in [-pi/2, pi/2] and longitude in [-pi, pi) of a (not necessarily unit) vector p,
/// returned as the pair (lat, lon).
std::pair<double, double> xyz_to_latlong(const Eigen::Vector3d& p);

/// Longitude rate d(lon)/dt (rad/s) of a point moving due east on the unit sphere at angular
/// speed `speed` (rad/s) at latitude `lat`. Derive the formula in the pre-lab.
double longitude_rate(double lat, double speed);

// ---------------------------------------------------------------------------
// Checkpoint 6 -- first contact with mrcpp: planar 2R arm (same as MR Studio "Planar 2R arm")
// ---------------------------------------------------------------------------

/// Tip position (x, y) of a planar 2R arm by trigonometry.
Eigen::Vector2d fk_2r_trig(const Eigen::Vector2d& theta, double L1 = 0.6, double L2 = 0.4);

/// 4x4 tip pose T_sb of the planar 2R arm computed with mr::fkin_space (= modern_robotics.FKinSpace).
/// Screw axes and home pose (MR Ch. 4, used here as a black box):
///   S1 = (0,0,1, 0,0,0),  S2 = (0,0,1, 0,-L1,0),  M = [[I, (L1+L2, 0, 0)], [0, 1]].
/// Put S1 and S2 as the COLUMNS of Slist (a 6 x 2 Eigen::MatrixXd), then call mr::fkin_space(M, Slist, theta).
Eigen::Matrix4d fk_2r_mr(const Eigen::Vector2d& theta, double L1 = 0.6, double L2 = 0.4);

}  // namespace lab

#endif  // SBEG361_LAB0_HPP
