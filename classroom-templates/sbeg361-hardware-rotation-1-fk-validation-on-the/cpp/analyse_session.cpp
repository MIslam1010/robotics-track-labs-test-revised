// SBEG 361 Hardware rotation 1 (Week 8) -- C++ track: GIVEN analysis program (do not edit; built with lab_tests).
//
// Runs YOUR functions (starter.cpp) on one session and prints the results table of Part C (HW-1 to HW-6, AP, RP,
// backlash split, fitted offsets, error budget).  Use it for the reference data of arm A5 and for your own Station H
// and Station S files written by run_session.py:
//     ./build/analyse_session                                        (A5 reference data in ../data)
//     ./build/analyse_session <log.csv> <tips.csv> [offsets.json]    (your files; default offsets: A5 file)
// (Windows: build\analyse_session.exe ...).  Paths are relative to the folder you run it from.
#include "lab.hpp"

#include <sbeg/labio.hpp>

#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr double PI = 3.14159265358979323846;

std::string fmt(const Eigen::VectorXd& v, int prec = 3) {
    std::ostringstream s;
    s << std::fixed << std::setprecision(prec) << "(";
    for (Eigen::Index i = 0; i < v.size(); ++i) s << (i ? ", " : "") << v(i);
    s << ")";
    return s.str();
}
}  // namespace

int main(int argc, char** argv) {
    const std::string log_path = argc > 1 ? argv[1] : sbeg::lab_path("../data/session_A5_log.csv");
    const std::string tips_path = argc > 2 ? argv[2] : sbeg::lab_path("../data/session_A5_tips.csv");
    const std::string off_path = argc > 3 ? argv[3] : sbeg::lab_path("../data/servo_offsets_A5.json");
    try {
        const sbeg::json plan = lab::load_plan(sbeg::lab_path("../data/validation_plan.json"));
        const Eigen::VectorXd off = lab::load_offsets(off_path);
        const lab::CourseArm arm = lab::course_arm_reference();
        const Eigen::Vector3d pt = sbeg::as<Eigen::Vector3d>(plan["p_tool_m"]);
        const Eigen::MatrixXd hw = sbeg::as_mat(plan["hw_limits_model_deg"]);
        const double m = plan["soft_limit_margin_deg"].get<double>();
        Eigen::MatrixXd soft(hw.rows(), 2);
        soft.col(0) = hw.col(0).array() + m;
        soft.col(1) = hw.col(1).array() - m;
        const lab::Log log = lab::read_log(log_path);
        const lab::Tips tips = lab::read_tips(tips_path);
        std::cout << std::fixed << std::setprecision(3) << "log:  " << log_path << "\ntips: " << tips_path
                  << "\noffsets " << fmt(off, 2) << " deg\n\n";

        const lab::JointTracking jt = lab::joint_tracking(log);
        std::cout << "HW-2  joint error before the stop (" << jt.n_samples << " samples)\n      rms  " << fmt(jt.rms)
                  << " deg\n      max  " << fmt(jt.max) << " deg\n      mean " << fmt(jt.mean) << " deg\n";
        const lab::TaskTracking tt = lab::task_tracking(log, arm.M, arm.Slist, off, pt);
        std::cout << "HW-3  tool-point error rms " << tt.rms_mm << " mm, max " << tt.max_mm << " mm\n";
        const lab::PlanCheck pc = lab::check_plan(log.t, log.meas, off, soft, plan["rated_speed_deg_s"].get<double>(),
                                                  plan["speed_fraction"].get<double>(),
                                                  plan["hw4_margin_deg"].get<double>());
        std::cout << "HW-4  smallest measured margin " << pc.margin_deg.minCoeff() << " deg (ok " << pc.ok_limits
                  << ")\nHW-5  peak measured speed " << pc.peak_speed.maxCoeff() << " deg/s (limit " << pc.speed_limit
                  << ", ok " << pc.ok_speed << ")\n";
        if (lab::before_stop(log) < log.t.size()) {
            const lab::StopCheck sc = lab::stop_check(log);
            std::cout << "HW-6  stop at " << sc.t_stop << " s, halt time " << sc.t_halt << " s, motion after stop "
                      << sc.motion_deg << " deg\n";
        } else {
            std::cout << "HW-6  no stop command in this log\n";
        }

        const auto n_acc = static_cast<Eigen::Index>(plan["accuracy_block"].size());
        if (tips.Q.rows() < n_acc + 2) throw std::runtime_error("the tips file needs the accuracy and repeat blocks");
        const Eigen::VectorXd e = lab::hw_tip_errors_mm(tips.Q.topRows(n_acc), tips.P.topRows(n_acc), arm.M,
                                                        arm.Slist, off, pt);
        std::cout << "\nHW-1  tip errors at the accuracy poses " << fmt(e) << " mm, rms "
                  << std::sqrt(e.array().square().mean()) << " mm\n";

        const Eigen::MatrixXd rep = tips.P.bottomRows(tips.P.rows() - n_acc);
        const Eigen::VectorXi ap = tips.approach.tail(tips.approach.size() - n_acc);
        Eigen::MatrixXd plus(0, 3), minus(0, 3);
        for (Eigen::Index k = 0; k < rep.rows(); ++k) {
            Eigen::MatrixXd& dst = ap(k) > 0 ? plus : minus;
            dst.conservativeResize(dst.rows() + 1, 3);
            dst.row(dst.rows() - 1) = rep.row(k);
        }
        const std::string rp_name = plan["repeat_pose"].get<std::string>();
        const Eigen::VectorXd q_rep = sbeg::as_vec(plan["poses_model_deg"][rp_name]) * (PI / 180.0);
        const Eigen::Vector3d pref = 1e3 * lab::tool_points(arm.M, arm.Slist, q_rep.transpose(), pt).row(0).transpose();
        const lab::AccuracyRepeatability all = lab::accuracy_repeatability(rep, pref);
        std::cout << "Repeat pose " << rp_name << " (" << rep.rows() << " visits): AP " << all.AP_mm << " mm, RP "
                  << all.RP_mm << " mm\n";
        if (plus.rows() >= 2 && minus.rows() >= 2) {
            const lab::AccuracyRepeatability a_p = lab::accuracy_repeatability(plus, pref);
            const lab::AccuracyRepeatability a_m = lab::accuracy_repeatability(minus, pref);
            std::cout << "      + side: AP " << a_p.AP_mm << " mm, RP " << a_p.RP_mm << " mm;  - side: AP "
                      << a_m.AP_mm << " mm, RP " << a_m.RP_mm << " mm\n      backlash split "
                      << lab::backlash_split(plus, minus) << " mm\n";
        }

        std::cout << "\nOffset fits on the accuracy poses (joints 1-based; RMS before "
                  << lab::fit_offsets(tips.Q.topRows(n_acc), tips.P.topRows(n_acc), arm.M, arm.Slist, off, pt, {1})
                         .rms_before_mm
                  << " mm)\n";
        const std::vector<std::vector<int>> models = {{1}, {2}, {1, 2}, {1, 2, 3}};
        for (const auto& js : models) {
            const lab::OffsetFit f =
                lab::fit_offsets(tips.Q.topRows(n_acc), tips.P.topRows(n_acc), arm.M, arm.Slist, off, pt, js);
            std::cout << "      joints";
            for (int j : js) std::cout << " " << j + 1;
            std::cout << ": delta " << fmt(f.delta_deg, 2) << " deg, RMS after " << f.rms_after_mm << " mm\n";
        }

        const lab::ErrorBudget b = lab::error_budget(arm.M, arm.Slist, q_rep, pt);
        std::cout << "\nError budget at " << rp_name << " (12-bit quantisation): lever " << fmt(b.lever_m, 4)
                  << " m, worst " << b.worst_mm << " mm, RSS " << b.rss_mm << " mm\n";
    } catch (const std::exception& ex) {
        std::cerr << "analyse_session: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
