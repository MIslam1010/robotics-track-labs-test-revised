// SBEG 461 Lab 2 -- C++ track: GIVEN program (do not edit; built together with lab_tests).
//
// Checkpoint 6 and the optional timing with YOUR functions (starter.cpp):
//     ./build/torque_profiles            peak torques and their times for the EOD lift and the exoskeleton reach, and
//                                        CSV files for the figure: eod_lift_torques.csv, exo_reach_torques.csv
//                                        (columns t, tau1..n, grav1..n; written to the folder you run it from)
//     ./build/torque_profiles --time     also: mean time per inverse-dynamics call of your rnea, mrcpp and pinlite
// (Windows: build\torque_profiles.exe).  Plot the CSV files with Python (matplotlib) or any plotting tool.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <chrono>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {
using Eigen::MatrixXd;
using Eigen::VectorXd;

std::string fmt(const VectorXd& v, int prec) {
    std::ostringstream s;
    s << std::fixed << std::setprecision(prec) << "(";
    for (Eigen::Index i = 0; i < v.size(); ++i) s << (i ? ", " : "") << v(i);
    s << ")";
    return s.str();
}

void profile(const std::string& name, const std::string& fname, const std::string& out_csv) {
    const lab::Robot r = lab::load_robot(name);
    const lab::Trajectory tr = lab::load_trajectory(fname);
    const MatrixXd tau = lab::torque_profile(r, tr.theta, tr.dtheta, tr.ddtheta);
    const VectorXd peak = lab::peak_torques(tau);
    VectorXd t_peak(r.n);
    for (Eigen::Index j = 0; j < r.n; ++j) {
        Eigen::Index k = 0;
        tau.col(j).cwiseAbs().maxCoeff(&k);
        t_peak(j) = tr.t(k);
    }
    MatrixXd table(tr.t.size(), 1 + 2 * r.n);
    table.col(0) = tr.t;
    table.middleCols(1, r.n) = tau;
    for (Eigen::Index k = 0; k < tr.t.size(); ++k) {
        const VectorXd th = tr.theta.row(k).transpose();
        const lab::DynamicsTerms d = lab::dynamics_terms(th, VectorXd::Zero(r.n), r.g, r.Mlist, r.Glist, r.Slist);
        table.block(k, 1 + r.n, 1, r.n) = d.grav.transpose();
    }
    std::vector<std::string> header = {"t"};
    for (int j = 1; j <= r.n; ++j) header.push_back("tau" + std::to_string(j));
    for (int j = 1; j <= r.n; ++j) header.push_back("grav" + std::to_string(j));
    sbeg::write_csv(out_csv, header, table);
    std::cout << std::left << std::setw(12) << name << " peak |tau| = " << fmt(peak, 2) << " N m at t = "
              << fmt(t_peak, 2) << " s   -> " << out_csv << "\n";
}

double time_us(const std::function<void()>& f, int n_calls) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < n_calls; ++k) f();
    const std::chrono::duration<double, std::micro> dt = std::chrono::steady_clock::now() - t0;
    return dt.count() / n_calls;
}

void timing() {
    std::mt19937 gen(0);
    std::normal_distribution<double> nd(0.0, 1.0);
    const int n_calls = 20000;
    for (const std::string name : {"mr_2r_point", "arm_exo_2r", "eod_3r", "ur5"}) {
        const lab::Robot r = lab::load_robot(name);
        VectorXd q(r.n), v(r.n), a(r.n);
        for (int i = 0; i < r.n; ++i) {
            q(i) = nd(gen);
            v(i) = nd(gen);
            a(i) = nd(gen);
        }
        const sbeg::Vector6d F0 = sbeg::Vector6d::Zero();
        const auto model_info = lab::pinocchio_model(r);
        const lab::PinModel& model = model_info.first;
        volatile double sink = 0.0;
        const double yours = time_us([&] { sink = sink + lab::rnea(q, v, a, r.g, F0, r.Mlist, r.Glist, r.Slist)(0); },
                                     n_calls);
        const double mrc = time_us(
            [&] { sink = sink + mr::inverse_dynamics(q, v, a, r.g, F0, r.Mlist, r.Glist, r.Slist)(0); }, n_calls);
        const double pl = time_us([&] { sink = sink + lab::pin_rnea(model, q, v, a)(0); }, n_calls);
        std::cout << std::left << std::setw(12) << name << std::fixed << std::setprecision(2) << " yours " << yours
                  << " us   mrcpp " << mrc << " us   pinlite " << pl << " us\n";
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        profile("eod_3r", "eod_lift.csv", "eod_lift_torques.csv");
        profile("arm_exo_2r", "exo_reach.csv", "exo_reach_torques.csv");
        if (argc > 1 && std::string(argv[1]) == "--time") timing();
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
