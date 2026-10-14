// SBEG 46X Lab 5 -- provided helpers (C++ track). Given code: do not edit; use these functions freely.
//
//   load_phantom, load_joints, rot_to_T     the "Provided" section of starter.py, translated to C++
//   Rng                                     Gaussian random numbers (stand-in for np.random.default_rng(seed))
//   combinations                            k-subsets in itertools.combinations order
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lab {

// ---------------------------------------------------------------------------
// Data files
// ---------------------------------------------------------------------------

Phantom load_phantom() {
    const sbeg::json d = sbeg::load_json(sbeg::lab_path("../data/phantom.json"));
    Phantom ph;
    ph.description = d.at("description").get<std::string>();
    ph.ct_voxel_mm = sbeg::as<Eigen::Vector3d>(d.at("ct_voxel_mm"));
    ph.fiducials_image_mm = sbeg::as_mat(d.at("fiducials_image_mm"));
    ph.check_points_image_mm = sbeg::as_mat(d.at("check_points_image_mm"));
    for (const auto& item : d.at("targets_image_mm").items())
        ph.targets_image_mm[item.key()] = sbeg::as<Eigen::Vector3d>(item.value());
    ph.pivot_divot = d.at("pivot_divot").get<std::string>();
    ph.pointer_nominal_length_mm = d.at("pointer_nominal_length_mm").get<double>();
    return ph;
}

std::pair<std::vector<std::string>, Eigen::MatrixXd> load_joints(const std::string& name) {
    const std::string path = sbeg::lab_path("../data/" + name);
    std::ifstream in(sbeg::fs_path(path), std::ios::binary);
    if (!in) throw std::runtime_error("cannot open file: " + path);
    std::string line;
    std::vector<std::string> labels;
    std::vector<std::vector<double>> rows;
    bool header = true;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (header) {  // label,q1,...,q6
            header = false;
            continue;
        }
        std::istringstream ss(line);
        std::string cell;
        std::getline(ss, cell, ',');
        labels.push_back(cell);
        std::vector<double> q;
        while (std::getline(ss, cell, ',')) q.push_back(std::stod(cell));
        if (q.size() != 6) throw std::runtime_error(path + ": expected label + 6 joint angles per line");
        rows.push_back(q);
    }
    Eigen::MatrixXd Q(static_cast<Eigen::Index>(rows.size()), 6);
    for (std::size_t i = 0; i < rows.size(); ++i)
        for (Eigen::Index k = 0; k < 6; ++k) Q(static_cast<Eigen::Index>(i), k) = rows[i][static_cast<std::size_t>(k)];
    return {labels, Q};
}

Eigen::Matrix4d rot_to_T(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) { return mr::rp_to_trans(R, t); }

// ---------------------------------------------------------------------------
// Random numbers: Box-Muller on the 64-bit Mersenne Twister. std::mt19937_64 gives the same integers on every
// compiler, but std::normal_distribution does not, so the Gaussian transform is written here.
// ---------------------------------------------------------------------------

Rng::Rng(std::uint64_t seed) : gen_(seed) {}

double Rng::normal(double loc, double scale) {
    if (has_spare_) {
        has_spare_ = false;
        return loc + scale * spare_;
    }
    constexpr double TWO_PI = 2.0 * 3.14159265358979323846;
    constexpr double INV_2_53 = 1.0 / 9007199254740992.0;  // 2^-53
    const double u1 = (static_cast<double>(gen_() >> 11) + 0.5) * INV_2_53;  // in (0, 1)
    const double u2 = (static_cast<double>(gen_() >> 11) + 0.5) * INV_2_53;
    const double r = std::sqrt(-2.0 * std::log(u1));
    spare_ = r * std::sin(TWO_PI * u2);
    has_spare_ = true;
    return loc + scale * r * std::cos(TWO_PI * u2);
}

Eigen::MatrixXd Rng::normal(double loc, double scale, Eigen::Index rows, Eigen::Index cols) {
    Eigen::MatrixXd out(rows, cols);
    for (Eigen::Index i = 0; i < rows; ++i)
        for (Eigen::Index k = 0; k < cols; ++k) out(i, k) = normal(loc, scale);
    return out;
}

// ---------------------------------------------------------------------------
// itertools.combinations(range(n), k)
// ---------------------------------------------------------------------------

std::vector<std::vector<int>> combinations(int n, int k) {
    std::vector<std::vector<int>> out;
    if (k < 0 || k > n) return out;
    std::vector<int> idx(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i) idx[static_cast<std::size_t>(i)] = i;
    while (true) {
        out.push_back(idx);
        int i = k - 1;  // rightmost index that can still move right
        while (i >= 0 && idx[static_cast<std::size_t>(i)] == n - k + i) --i;
        if (i < 0) break;
        ++idx[static_cast<std::size_t>(i)];
        for (int m = i + 1; m < k; ++m) idx[static_cast<std::size_t>(m)] = idx[static_cast<std::size_t>(m - 1)] + 1;
    }
    return out;
}

}  // namespace lab
