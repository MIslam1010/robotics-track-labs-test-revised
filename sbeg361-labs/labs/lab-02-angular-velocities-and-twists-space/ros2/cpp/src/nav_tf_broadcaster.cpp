// SBEG 361 Lab 2, Part B (optional C++ version, ROS 2 Jazzy) -- publish the navigation transform tree to tf2.
// Same behaviour as ../nav_tf_broadcaster.py:
//   * every entry of nav_tree.json      -> /tf_static  (tf2_ros::StaticTransformBroadcaster)
//   * tool_track.csv (T_tracker_tool_marker(t)), replayed in a loop at 50 Hz -> /tf  (tf2_ros::TransformBroadcaster)
//
// Build and run (inside the course ROS 2 Jazzy Docker image or an Ubuntu 24.04 install with ROS 2 Jazzy):
//     source /opt/ros/jazzy/setup.bash
//     mkdir -p ~/ros2_ws/src && ln -s "<your Lab code folder>/ros2/cpp" ~/ros2_ws/src/sbeg361_nav_tf
//     cd ~/ros2_ws && colcon build --packages-select sbeg361_nav_tf && source install/setup.bash
//     ros2 run sbeg361_nav_tf nav_tf_broadcaster --ros-args -p lab_dir:="<your Lab code folder>"
// then use rviz2 / tf2_echo / view_frames exactly as in the handout (Part B).
//
// tf2 conventions used here (they match lab.hpp and the lab handout):
//     TransformStamped.header.frame_id = parent, .child_frame_id = child,
//     .transform = pose of the child expressed in the parent = T_parent_child;
//     quaternions are (x, y, z, w), scalar LAST (MR App. B.3 writes the scalar first).
// This file is not auto-tested (no ROS on the grading server).
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include <nlohmann/json.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef SBEG_LAB_DIR
#define SBEG_LAB_DIR "."
#endif

namespace {

using Mat3 = std::array<std::array<double, 3>, 3>;
using Quat = std::array<double, 4>;  // (x, y, z, w)

/// Unit quaternion (x, y, z, w) of a rotation matrix (Shepperd's method; robust near pi).
Quat rot_to_quat_xyzw(const Mat3& R) {
    const double tr = R[0][0] + R[1][1] + R[2][2];
    double w, x, y, z;
    if (tr > 0) {
        const double s = 2.0 * std::sqrt(tr + 1.0);
        w = 0.25 * s; x = (R[2][1] - R[1][2]) / s; y = (R[0][2] - R[2][0]) / s; z = (R[1][0] - R[0][1]) / s;
    } else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) {
        const double s = 2.0 * std::sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]);
        w = (R[2][1] - R[1][2]) / s; x = 0.25 * s; y = (R[0][1] + R[1][0]) / s; z = (R[0][2] + R[2][0]) / s;
    } else if (R[1][1] > R[2][2]) {
        const double s = 2.0 * std::sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]);
        w = (R[0][2] - R[2][0]) / s; x = (R[0][1] + R[1][0]) / s; y = 0.25 * s; z = (R[1][2] + R[2][1]) / s;
    } else {
        const double s = 2.0 * std::sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]);
        w = (R[1][0] - R[0][1]) / s; x = (R[0][2] + R[2][0]) / s; y = (R[1][2] + R[2][1]) / s; z = 0.25 * s;
    }
    const double n = std::sqrt(x * x + y * y + z * z + w * w);
    return {x / n, y / n, z / n, w / n};
}

geometry_msgs::msg::TransformStamped make_msg(const builtin_interfaces::msg::Time& stamp, const std::string& parent,
                                              const std::string& child, const std::array<double, 3>& t,
                                              const Quat& q) {
    geometry_msgs::msg::TransformStamped m;
    m.header.stamp = stamp;
    m.header.frame_id = parent;
    m.child_frame_id = child;
    m.transform.translation.x = t[0];
    m.transform.translation.y = t[1];
    m.transform.translation.z = t[2];
    m.transform.rotation.x = q[0];
    m.transform.rotation.y = q[1];
    m.transform.rotation.z = q[2];
    m.transform.rotation.w = q[3];
    return m;
}

struct Pose {
    Mat3 R;
    std::array<double, 3> p;
};

/// tool_track.csv: header line, then t, r11..r33 (row-major), px, py, pz.
std::vector<Pose> read_track(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::string line;
    std::getline(in, line);  // header
    std::vector<Pose> track;
    while (std::getline(in, line)) {
        if (line.empty() || line == "\r") continue;
        std::vector<double> v;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) v.push_back(std::stod(cell));
        if (v.size() != 13) throw std::runtime_error("bad line in " + path + ": " + line);
        Pose P;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) P.R[i][j] = v[1 + 3 * i + j];
            P.p[i] = v[10 + i];
        }
        track.push_back(P);
    }
    return track;
}

class NavTfBroadcaster : public rclcpp::Node {
public:
    NavTfBroadcaster() : rclcpp::Node("sbeg361_nav_tf") {
        const std::string lab_dir = declare_parameter<std::string>("lab_dir", SBEG_LAB_DIR);
        static_br_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
        br_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        std::ifstream f(lab_dir + "/nav_tree.json");
        if (!f) throw std::runtime_error("cannot open " + lab_dir + "/nav_tree.json (set -p lab_dir:=...)");
        const nlohmann::json tree = nlohmann::json::parse(f);
        const auto now = get_clock()->now();
        std::vector<geometry_msgs::msg::TransformStamped> edges;
        for (const auto& e : tree.at("transforms"))
            edges.push_back(make_msg(now, e.at("parent").get<std::string>(), e.at("child").get<std::string>(),
                                     e.at("translation").get<std::array<double, 3>>(),
                                     e.at("rotation_xyzw").get<Quat>()));
        static_br_->sendTransform(edges);

        track_ = read_track(lab_dir + "/tool_track.csv");
        timer_ = create_wall_timer(std::chrono::milliseconds(20), [this] { tick(); });  // 50 Hz, as the tracker
        RCLCPP_INFO(get_logger(), "static edges: %zu, tool samples: %zu", edges.size(), track_.size());
    }

private:
    void tick() {
        const Pose& P = track_[k_];
        // EXERCISE B3: which frame is the parent here, and why must it be 'tracker'?
        br_->sendTransform(make_msg(get_clock()->now(), "tracker", "tool_marker", P.p, rot_to_quat_xyzw(P.R)));
        k_ = (k_ + 1) % track_.size();
    }

    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_br_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> br_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::vector<Pose> track_;
    std::size_t k_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<NavTfBroadcaster>());
    rclcpp::shutdown();
    return 0;
}
