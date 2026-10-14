// SBEG 361 Lab 6 -- driver (provided): FK of the course arm by PoE and by DH for every joint sample,
// pose files for the Python comparison, and a timing benchmark.
//
// Usage:  fk_compare <q_samples.csv> <output_dir> [repeats]
// Writes: <output_dir>/poses_poe_cpp.csv, <output_dir>/poses_dh_cpp.csv  (r11..r33, px, py, pz per row)
//         <output_dir>/timing_cpp.csv  (method, ns_per_call)
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "fk.hpp"

using namespace sbeg361;

static std::vector<Vec6> readSamples(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<Vec6> qs;
    std::string line;
    std::getline(in, line);                      // header
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        Vec6 q;
        for (int i = 0; i < 6; ++i) {
            std::string cell;
            std::getline(ss, cell, ',');
            q(i) = std::stod(cell);
        }
        qs.push_back(q);
    }
    return qs;
}

static void writePoses(const std::filesystem::path& path, const std::vector<Mat4>& Ts) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path.string());
    out << "r11,r12,r13,r21,r22,r23,r31,r32,r33,px,py,pz\n" << std::setprecision(17);
    for (const Mat4& T : Ts) {
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) out << T(r, c) << ",";
        out << T(0, 3) << "," << T(1, 3) << "," << T(2, 3) << "\n";
    }
}

template <class F>
static double nsPerCall(F&& fk, const std::vector<Vec6>& qs, int repeats) {
    double best = 1e300, sink = 0.0;
    for (int r = 0; r < repeats; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        for (const Vec6& q : qs) sink += fk(q)(0, 3);
        const auto t1 = std::chrono::steady_clock::now();
        const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / static_cast<double>(qs.size());
        if (ns < best) best = ns;
    }
    volatile double keep = sink;                 // stops the compiler from removing the loop
    (void)keep;
    return best;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: fk_compare <q_samples.csv> <output_dir> [repeats]\n";
        return 2;
    }
    try {
        const std::vector<Vec6> qs = readSamples(argv[1]);
        const std::filesystem::path outdir = argv[2];
        const int repeats = argc > 3 ? std::stoi(argv[3]) : 20;
        const Mat6 S = courseArmSlist();
        const Mat4 M = courseArmM();
        const auto table = courseArmDH();
        const Mat4 tool = courseArmDHTool();
        auto poe = [&](const Vec6& q) { return fkPoE(S, M, q); };
        auto dh = [&](const Vec6& q) { return fkDH(table, tool, q); };

        std::vector<Mat4> Tp, Td;
        double maxdiff = 0.0;
        for (const Vec6& q : qs) {
            Tp.push_back(poe(q));
            Td.push_back(dh(q));
            maxdiff = std::max(maxdiff, (Tp.back() - Td.back()).cwiseAbs().maxCoeff());
        }
        writePoses(outdir / "poses_poe_cpp.csv", Tp);
        writePoses(outdir / "poses_dh_cpp.csv", Td);

        const double tp = nsPerCall(poe, qs, repeats), td = nsPerCall(dh, qs, repeats);
        std::ofstream t(outdir / "timing_cpp.csv");
        t << "method,ns_per_call\npoe," << tp << "\ndh," << td << "\n";
        std::cout << qs.size() << " samples; max |T_PoE - T_DH| entry = " << maxdiff << "\n"
                  << "C++ PoE FK: " << tp << " ns/call\nC++ DH  FK: " << td << " ns/call\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
