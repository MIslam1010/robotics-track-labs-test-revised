// SBEG 461 enrichment lab "parameter identification" -- hardware-log helpers (C++ track).  GIVEN code: do not edit.
//
// C++ copy of hwlog.py (load_log, resample, rms, normalised_rms of the project package).  Log format (one row per
// sample, header required, SI units: s, rad, A):
//     t, cmd_1, ..., cmd_n, meas_1, ..., meas_n [, cur_1, ..., cur_n] [, force]
// The device clock does not tick at exactly the nominal rate (jitter, an occasional repeated time stamp), so logs
// are always resampled onto a uniform grid before they are filtered or differentiated.
#include "lab.hpp"

#include <sbeg/labio.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lab {
namespace {

/// Columns "<prefix>_1", "<prefix>_2", ... sorted by their number (empty if there is none).
std::vector<Eigen::Index> group(const sbeg::CsvTable& tab, const std::string& prefix) {
    std::vector<std::pair<int, Eigen::Index>> found;
    for (std::size_t c = 0; c < tab.header.size(); ++c) {
        const std::string& h = tab.header[c];
        if (h.size() > prefix.size() + 1 && h.compare(0, prefix.size() + 1, prefix + "_") == 0)
            found.emplace_back(std::stoi(h.substr(prefix.size() + 1)), static_cast<Eigen::Index>(c));
    }
    std::sort(found.begin(), found.end());
    std::vector<Eigen::Index> out;
    for (const auto& f : found) out.push_back(f.second);
    return out;
}

Eigen::MatrixXd take_cols(const Eigen::MatrixXd& data, const std::vector<Eigen::Index>& cols) {
    Eigen::MatrixXd out(data.rows(), static_cast<Eigen::Index>(cols.size()));
    for (std::size_t k = 0; k < cols.size(); ++k) out.col(static_cast<Eigen::Index>(k)) = data.col(cols[k]);
    return out;
}

}  // namespace

HwLog load_log(const std::string& path) {
    const sbeg::CsvTable tab = sbeg::read_csv(path);
    const Eigen::Index ct = tab.col("t");  // throws if the column is missing
    const Eigen::Index N = tab.data.rows();
    // stable sort by time, then keep the first row of every time stamp (np.unique(..., return_index=True))
    std::vector<Eigen::Index> order(static_cast<std::size_t>(N));
    std::iota(order.begin(), order.end(), Eigen::Index(0));
    std::stable_sort(order.begin(), order.end(),
                     [&](Eigen::Index a, Eigen::Index b) { return tab.data(a, ct) < tab.data(b, ct); });
    std::vector<Eigen::Index> keep;
    for (std::size_t k = 0; k < order.size(); ++k)
        if (k == 0 || tab.data(order[k], ct) != tab.data(order[k - 1], ct)) keep.push_back(order[k]);
    Eigen::MatrixXd data(static_cast<Eigen::Index>(keep.size()), tab.data.cols());
    for (std::size_t k = 0; k < keep.size(); ++k) data.row(static_cast<Eigen::Index>(k)) = tab.data.row(keep[k]);

    HwLog out;
    out.t = data.col(ct);
    const auto cmd = group(tab, "cmd"), meas = group(tab, "meas"), cur = group(tab, "cur");
    if (cmd.empty() || meas.empty() || cmd.size() != meas.size())
        throw std::runtime_error(path + ": need matching cmd_j and meas_j columns");
    out.cmd = take_cols(data, cmd);
    out.meas = take_cols(data, meas);
    if (!cur.empty()) out.cur = take_cols(data, cur);
    if (std::find(tab.header.begin(), tab.header.end(), "force") != tab.header.end())
        out.force = data.col(tab.col("force"));
    return out;
}

Eigen::MatrixXd resample(const Eigen::VectorXd& t_src, const Eigen::MatrixXd& x_src, const Eigen::VectorXd& t_dst) {
    if (x_src.rows() != t_src.size() || t_src.size() == 0)
        throw std::invalid_argument("resample: x_src must have one row per entry of t_src");
    const Eigen::Index N = t_src.size();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    Eigen::MatrixXd y(t_dst.size(), x_src.cols());
    for (Eigen::Index i = 0; i < t_dst.size(); ++i) {
        const double x = t_dst(i);
        if (x < t_src(0) || x > t_src(N - 1)) {
            y.row(i).setConstant(nan);
            continue;
        }
        if (x == t_src(N - 1)) {
            y.row(i) = x_src.row(N - 1);
            continue;
        }
        // j with t_src(j) <= x < t_src(j + 1)
        const double* first = t_src.data();
        const Eigen::Index j = static_cast<Eigen::Index>(std::upper_bound(first, first + N, x) - first) - 1;
        for (Eigen::Index c = 0; c < x_src.cols(); ++c) {
            const double slope = (x_src(j + 1, c) - x_src(j, c)) / (t_src(j + 1) - t_src(j));
            y(i, c) = slope * (x - t_src(j)) + x_src(j, c);
        }
    }
    return y;
}

double rms(const Eigen::VectorXd& x) { return std::sqrt(x.squaredNorm() / static_cast<double>(x.size())); }

double normalised_rms(const Eigen::VectorXd& predicted, const Eigen::VectorXd& measured) {
    return rms(predicted - measured) / rms(measured);
}

}  // namespace lab
