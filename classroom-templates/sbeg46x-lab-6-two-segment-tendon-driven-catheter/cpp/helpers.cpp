// SBEG 46X Lab 6 -- provided helpers (C++ track). Given code: do not edit; use these functions freely.
//
//   load_catheter, load_csv        data/catheter.json, data/<name>.csv
//   vee, trans_z, split_q          small helpers of starter.py
//   catheter_backbone              backbone points for plotting (calls your segment_transform_exp)
//   gradient_rows                  np.gradient(P, t, axis=0), used by track_path
//
// These are the "Provided" functions of starter.py, translated to C++ (gradient_rows replaces np.gradient).
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <stdexcept>
#include <string>

namespace lab {

namespace {
constexpr double PI = 3.14159265358979323846;
}  // namespace

Catheter load_catheter() {
    const sbeg::json c = sbeg::load_json(sbeg::lab_path("../data/catheter.json"));
    Catheter cat;
    cat.description = c.value("description", std::string());
    cat.ell = sbeg::as<Eigen::Vector2d>(c["ell"]);
    cat.d = sbeg::as<Eigen::Vector2d>(c["d"]);
    cat.sigma_deg = sbeg::as<Matrix23d>(c["sigma_deg"]);
    cat.sigma = cat.sigma_deg * (PI / 180.0);
    cat.EI = sbeg::as<Eigen::Vector2d>(c["EI"]);
    cat.theta_max = sbeg::as_double(c["theta_max"]);
    cat.z_range = sbeg::as<Eigen::Vector2d>(c["z_range"]);
    cat.T_base = sbeg::as<Eigen::Matrix4d>(c["T_base"]);
    return cat;
}

sbeg::CsvTable load_csv(const std::string& name) { return sbeg::read_csv(sbeg::lab_path("../data/" + name)); }

Vector6d vee(const Eigen::Matrix4d& se3mat) { return mr::se3_to_vec(se3mat); }

Eigen::Matrix4d trans_z(double z) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T(2, 3) = z;
    return T;
}

SplitQ split_q(const Vector5d& q) { return SplitQ{q(0), q.segment<2>(1), q.segment<2>(3)}; }

Eigen::MatrixXd catheter_backbone(const Vector5d& q, const Catheter& cat, int n) {
    const SplitQ s = split_q(q);
    Eigen::MatrixXd pts(2 * n + 2, 3);
    pts.row(0) = cat.T_base.block<3, 1>(0, 3).transpose();
    Eigen::Matrix4d T0 = cat.T_base * trans_z(s.z);
    pts.row(1) = T0.block<3, 1>(0, 3).transpose();
    Eigen::Index row = 2;
    for (int j = 0; j < 2; ++j) {
        const Eigen::Vector2d theta_v = j == 0 ? s.theta_v1 : s.theta_v2;
        const double ell = cat.ell(j);
        for (int k = 1; k <= n; ++k) {
            const double f = static_cast<double>(k) / n;
            const Eigen::Matrix4d Tk = T0 * segment_transform_exp(theta_v * f, ell * f);
            pts.row(row++) = Tk.block<3, 1>(0, 3).transpose();
        }
        T0 = T0 * segment_transform_exp(theta_v, ell);
    }
    return pts;
}

Eigen::MatrixXd gradient_rows(const Eigen::MatrixXd& P, const Eigen::VectorXd& t) {
    const Eigen::Index N = P.rows();
    if (t.size() != N || N < 2) throw std::invalid_argument("gradient_rows: need t.size() == P.rows() >= 2");
    Eigen::MatrixXd D(N, P.cols());
    D.row(0) = (P.row(1) - P.row(0)) / (t(1) - t(0));
    D.row(N - 1) = (P.row(N - 1) - P.row(N - 2)) / (t(N - 1) - t(N - 2));
    for (Eigen::Index k = 1; k + 1 < N; ++k) {
        const double hs = t(k) - t(k - 1), hd = t(k + 1) - t(k);
        D.row(k) = (hs * hs * P.row(k + 1) + (hd * hd - hs * hs) * P.row(k) - hd * hd * P.row(k - 1)) /
                   (hs * hd * (hd + hs));
    }
    return D;
}

}  // namespace lab
