// SBEG 461 Lab 4 -- benchmark driver (PROVIDED, do not edit).
// Usage: rnea_bench model.txt states.txt tau_out.txt [repeats]
// Reads the files written by export_for_cpp(), computes tau for every state with g = (0, 0, -9.81) and
// Ftip = 0, writes one row of n torques per state, and prints "us_per_call <microseconds>".
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "rnea.hpp"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: " << argv[0] << " model.txt states.txt tau_out.txt [repeats]\n";
    return 2;
  }
  try {
    const Model model = load_model(argv[1]);
    const int n = model.n;
    const int repeats = argc > 4 ? std::stoi(argv[4]) : 1000;
    std::ifstream fs(argv[2]);
    if (!fs) throw std::runtime_error("cannot open states file");
    std::vector<Eigen::VectorXd> th, dth, ddth;
    while (true) {
      Eigen::VectorXd a(n), b(n), c(n);
      for (int i = 0; i < n; ++i) fs >> a(i);
      for (int i = 0; i < n; ++i) fs >> b(i);
      for (int i = 0; i < n; ++i) fs >> c(i);
      if (!fs) break;
      th.push_back(a);
      dth.push_back(b);
      ddth.push_back(c);
    }
    if (th.empty()) throw std::runtime_error("no states read");
    const se3::Vec3 g(0.0, 0.0, -9.81);
    const se3::Vec6 Ftip = se3::Vec6::Zero();

    std::ofstream fo(argv[3]);
    fo << std::setprecision(17);
    for (std::size_t k = 0; k < th.size(); ++k) fo << inverse_dynamics(model, th[k], dth[k], ddth[k], g, Ftip).transpose() << "\n";

    double sink = 0.0;  // keeps the optimiser from removing the timed loop
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; ++r)
      for (std::size_t k = 0; k < th.size(); ++k) sink += inverse_dynamics(model, th[k], dth[k], ddth[k], g, Ftip)(0);
    const auto t1 = std::chrono::steady_clock::now();
    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / (double(repeats) * static_cast<double>(th.size()));
    std::cout << "states " << th.size() << "\nus_per_call " << std::setprecision(6) << us << "\nchecksum " << sink << "\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
