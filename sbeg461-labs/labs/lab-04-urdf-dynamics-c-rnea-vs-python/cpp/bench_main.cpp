// SBEG 461 Lab 4 -- C++ track program for Step 3 (timing) and Step 6 (simulation files). PROVIDED, do not edit.
//
// Usage:  lab_bench [output folder]          (default: the current folder)
//
// Step 3 -- uses YOUR urdf_to_mr, rnea and time_per_call on arm6.urdf and the first 200 states of states_arm6.csv:
//   * writes model.txt and states.txt (export_for_cpp), the input files of the Step 3 program:
//         rnea_bench model.txt states.txt tau_cpp.txt 2000
//   * prints the microseconds per call of your lab::rnea and of mr::inverse_dynamics (mrcpp).
// Step 6 -- uses YOUR simulate_channel_2r and twor_dynamics (tau = 0, g = 0, dt = 1 ms) and writes CSV files with
//   the columns  t, theta1, theta2, dtheta1, dtheta2, lam, e, KE  for plot_results.py:
//     step6_1_energy.csv        Example 8.5 state, 2 s, alpha = beta = 0                         (Step 6.1)
//     step6_2_plain.csv         theta(0) = (-60 deg, 120 deg + 0.57 deg), 1 s, alpha = beta = 0  (Step 6.2)
//     step6_2_baumgarte.csv     same start, alpha = beta = 20 1/s                                (Step 6.2)
//     step6_3_stiff.csv         same start, alpha = beta = 2000 1/s                              (Step 6.3)
// Each part reports "error: TODO: ..." and is skipped until the functions it needs are written.
#include "lab.hpp"

#include <sbeg/labio.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

namespace {
constexpr double PI = 3.14159265358979323846;

void write_sim(const std::string& path, const lab::ChannelSim& s) {
    const Eigen::Index N = s.t.size();
    Eigen::MatrixXd D(N, 8);
    for (Eigen::Index k = 0; k < N; ++k) {
        const Eigen::Vector2d q = s.theta.row(k).transpose(), dq = s.dtheta.row(k).transpose();
        const double ke = 0.5 * dq.dot(lab::twor_dynamics(q, dq).first * dq);
        D.row(k) << s.t(k), q(0), q(1), dq(0), dq(1), s.lam(k), s.e(k), ke;
    }
    sbeg::write_csv(path, {"t", "theta1", "theta2", "dtheta1", "dtheta2", "lam", "e", "KE"}, D);
    std::cout << "wrote " << path << "  (max |e| = " << s.e.cwiseAbs().maxCoeff() << ")\n";
}
}  // namespace

int main(int argc, char** argv) {
    const std::string out_dir = argc > 1 ? std::string(argv[1]) : std::string(".");
    int status = 0;
    try {  // ---- Step 3
        const std::string urdf = sbeg::lab_path("../arm6.urdf");
        const std::string csv = sbeg::lab_path("../states_arm6.csv");
        const lab::MRModel m = lab::urdf_to_mr(urdf);
        const Eigen::MatrixXd all = lab::load_states(csv);
        const Eigen::MatrixXd X = all.topRows(std::min<Eigen::Index>(200, all.rows()));
        lab::export_for_cpp(out_dir + "/model.txt", out_dir + "/states.txt", m.Mlist, m.Glist, m.Slist, X);
        std::cout << "Step 3: wrote " << out_dir << "/model.txt and " << out_dir << "/states.txt (" << X.rows()
                  << " states)\n";
        const auto old_flags = std::cout.flags();
        std::cout << std::fixed << std::setprecision(3);
        for (const auto& [name, us] : lab::benchmark(urdf, csv, 200))
            std::cout << "  " << std::left << std::setw(26) << name << " " << us << " us per call\n";
        std::cout.flags(old_flags);
    } catch (const std::exception& e) {
        std::cerr << "Step 3 skipped -- error: " << e.what() << "\n";
        status = 1;
    }
    try {  // ---- Step 6
        const lab::TorqueFunction zero = [](double, const Eigen::Vector2d&, const Eigen::Vector2d&) {
            return Eigen::Vector2d::Zero().eval();
        };
        const Eigen::Vector2d th85(-PI / 3.0, 2.0 * PI / 3.0), dth85(1.0, 0.0);
        const Eigen::Vector2d th0 = th85 + Eigen::Vector2d(0.0, 0.01);  // 0.57 deg violation
        write_sim(out_dir + "/step6_1_energy.csv", lab::simulate_channel_2r(th85, dth85, zero, 2.0, 1e-3));
        write_sim(out_dir + "/step6_2_plain.csv", lab::simulate_channel_2r(th0, dth85, zero, 1.0, 1e-3));
        write_sim(out_dir + "/step6_2_baumgarte.csv",
                  lab::simulate_channel_2r(th0, dth85, zero, 1.0, 1e-3, 1.0, 20.0, 20.0));
        write_sim(out_dir + "/step6_3_stiff.csv",
                  lab::simulate_channel_2r(th0, dth85, zero, 1.0, 1e-3, 1.0, 2000.0, 2000.0));
        std::cout << "Step 6: plot with  python cpp/plot_results.py " << out_dir << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Step 6 skipped -- error: " << e.what() << "\n";
        status = 1;
    }
    return status;
}
