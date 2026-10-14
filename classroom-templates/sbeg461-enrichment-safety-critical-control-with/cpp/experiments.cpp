// SBEG 461 enrichment lab (CBF safety filter) -- the experiments program (C++ track). Given code: do not edit.
//
// Same runs as `python starter.py`: for each scenario (static, approaching) the four experiments
//   nominal (no filter), keep-out only (alpha1 = 10, no speed cap), filter with the scenario v_max (0.20 m/s),
//   filter with your conservative design speed (design_speed_limit).
// Prints the metrics and the largest contact force of each run, and writes one CSV log per run for plotting:
//   cbf_<scenario>_<k>.csv   (k = 1..4)   columns  t, x, y, x_o, y_o, x_d, y_d, q1, q2, d, approach, F, active
// Usage (after every test passes):   ./build/lab_experiments [output folder]      (default: the current folder)
// Then plot with:                    python cpp/plot_results.py [output folder]
#include "lab.hpp"

#include <sbeg/labio.hpp>

#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
void write_log(const std::string& path, const lab::SimLog& log, const lab::Scenario& scn) {
    const Eigen::Index N = log.t.size();
    const Eigen::VectorXd F = lab::peak_contact_force(log, scn);
    Eigen::MatrixXd D(N, 13);
    for (Eigen::Index k = 0; k < N; ++k) {
        const Eigen::Vector2d pd = lab::reference(log.t(k), scn).p;
        D.row(k) << log.t(k), log.p(k, 0), log.p(k, 1), log.p_o(k, 0), log.p_o(k, 1), pd(0), pd(1), log.q(k, 0),
            log.q(k, 1), log.d(k), log.approach(k), F(k), log.active[static_cast<std::size_t>(k)] ? 1.0 : 0.0;
    }
    sbeg::write_csv(path, {"t", "x", "y", "x_o", "y_o", "x_d", "y_d", "q1", "q2", "d", "approach", "F", "active"}, D);
}
}  // namespace

int main(int argc, char** argv) {
    const std::string out = argc > 1 ? argv[1] : ".";
    try {
        std::filesystem::create_directories(sbeg::fs_path(out));
        std::cout << std::fixed << std::setprecision(4);
        for (const std::string name : {"static", "approaching"}) {
            const lab::Scenario scn = lab::load_scenario(name);
            const double v_design = lab::design_speed_limit(scn).v_design;
            lab::Scenario keep = scn, design = scn;
            keep.alpha1 = 10.0;
            keep.v_max = 1e3;
            design.v_max = v_design;
            std::ostringstream l3, l4;
            l3 << std::fixed << std::setprecision(2) << "filter, v_max " << scn.v_max;
            l4 << std::fixed << std::setprecision(2) << "filter, v_max " << v_design;
            struct Exp {
                std::string label;
                lab::Scenario scn;
                bool on;
            };
            const std::vector<Exp> runs = {
                {"nominal (no filter)", scn, false}, {"keep-out only", keep, true}, {l3.str(), scn, true},
                {l4.str(), design, true}};
            int k = 0;
            for (const Exp& e : runs) {
                ++k;
                const lab::SimLog log = lab::simulate(e.scn, e.on);
                const lab::Metrics m = lab::metrics(log, e.scn);
                const double F_peak = lab::peak_contact_force(log, e.scn).maxCoeff();
                std::cout << std::left << std::setw(12) << name << " " << std::setw(20) << e.label << std::right
                          << " min_d " << m.min_d << "  max_approach " << m.max_approach << "  max_tau_ratio "
                          << m.max_tau_ratio << "  active_fraction " << m.active_fraction << "  final_error "
                          << m.final_error << "  max_dev " << m.max_dev << "  F_peak " << F_peak << "\n";
                const std::string path = out + "/cbf_" + name + "_" + std::to_string(k) + ".csv";
                write_log(path, log, e.scn);
            }
        }
        std::cout << "CSV logs written to " << out << "  (plot: python cpp/plot_results.py " << out << ")\n";
    } catch (const std::exception& ex) {
        std::cerr << "lab_experiments stopped: " << ex.what() << "\n(make every test of ./build/lab_tests pass first)\n";
        return 1;
    }
    return 0;
}
