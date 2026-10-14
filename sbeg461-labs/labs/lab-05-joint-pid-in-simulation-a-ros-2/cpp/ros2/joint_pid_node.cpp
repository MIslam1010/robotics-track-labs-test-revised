// SBEG 461 Lab 5, Part D (C++ track, guided, ROS 2 Jazzy) -- a PID joint controller node built on YOUR JointPID.
//
// Given (optional exercise, not auto-checked). The C++ twin of ros2/joint_pid_node.py: same topics, parameters and
// behaviour. Its ROS-free part (node_logic::PidNodeLogic in node_logic.hpp) is tested by
// test_ros_logic_matches_offline_simulation; this file only connects it to ROS.
//
//   subscribes  /joint_states                  sensor_msgs/JointState   (the joint named by joint_name)
//   publishes   /effort_controller/commands    std_msgs/Float64MultiArray  data[cmd_index] = tau (N m)
//               /pid_debug                     std_msgs/Float64MultiArray  [t, qd, q, e, tau, P, I, D, saturated]
//
// The controller runs once per received joint state; controller time is k * dt (k = states received).
// Gains can be changed while it runs:   ros2 param set /joint_pid kp 12.0
// Give every numeric parameter with a decimal point (kp:=12.0, not 12): they are declared as doubles.
//
// Build (from your Lab code folder, in a terminal where ROS 2 Jazzy is sourced; links YOUR cpp/starter.cpp):
//     source /opt/ros/jazzy/setup.bash
//     cmake -S cpp/ros2 -B build_ros && cmake --build build_ros
// Run (the simulator is the Python node ros2/joint_sim_node.py, which uses only provided functions):
//     ./build_ros/joint_pid_node --ros-args -p plant:=arm_j2 -p wn:=20.0 -p reference:=step -p q_target:=-0.3 \
//         -p t_step:=1.0 -p dt:=0.005
//     python3 ros2/joint_sim_node.py --ros-args -p plant:=arm_j2 -p dt:=0.005        # second terminal
// Parameters: plant, zeta, wn, kp, ki, kd (negative = default from lab::pd_gains; ki defaults to 0 = PD),
// tau_max (negative = the plant's), anti_windup, dt, reference (hold | step | knee_swing), q_target, t_step,
// velocity (measured | difference), joint_name, cmd_size, cmd_index -- as in joint_pid_node.py.
#include "node_logic.hpp"

#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class JointPidNode : public rclcpp::Node {
public:
    JointPidNode() : rclcpp::Node("joint_pid") {
        const std::string plant = declare_parameter<std::string>("plant", "arm_j2");
        const double zeta = declare_parameter<double>("zeta", 1.0);
        const double wn = declare_parameter<double>("wn", 20.0);
        const double dt = declare_parameter<double>("dt", 0.005);
        const double kp = declare_parameter<double>("kp", -1.0);
        const double ki = declare_parameter<double>("ki", 0.0);
        const double kd = declare_parameter<double>("kd", -1.0);
        const double tau_max = declare_parameter<double>("tau_max", -1.0);
        const bool anti_windup = declare_parameter<bool>("anti_windup", true);
        lab::JointPID ctrl = node_logic::default_controller(plant, zeta, wn, dt, kp, ki, kd, tau_max, anti_windup);
        const std::string joint_name = declare_parameter<std::string>("joint_name", "joint");
        const std::string reference = declare_parameter<std::string>("reference", "hold");
        const double q_target = declare_parameter<double>("q_target", 0.0);
        const double t_step = declare_parameter<double>("t_step", 1.0);
        const std::string velocity = declare_parameter<std::string>("velocity", "measured");
        logic_ = std::make_unique<node_logic::PidNodeLogic>(ctrl, joint_name, reference, q_target, t_step, velocity);
        cmd_size_ = static_cast<std::size_t>(declare_parameter<std::int64_t>("cmd_size", 1));
        cmd_index_ = static_cast<std::size_t>(declare_parameter<std::int64_t>("cmd_index", 0));

        pub_cmd_ = create_publisher<std_msgs::msg::Float64MultiArray>("/effort_controller/commands", 10);
        pub_dbg_ = create_publisher<std_msgs::msg::Float64MultiArray>("/pid_debug", 10);
        sub_ = create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10, [this](const sensor_msgs::msg::JointState& msg) { on_state(msg); });
        param_cb_ = add_on_set_parameters_callback(
            [this](const std::vector<rclcpp::Parameter>& params) { return on_params(params); });
        const lab::JointPID& c = logic_->ctrl;
        const std::string lim = c.tau_max ? std::to_string(*c.tau_max) : std::string("None");
        RCLCPP_INFO(get_logger(), "PID on '%s': kp=%.4g ki=%.4g kd=%.4g tau_max=%s dt=%g", plant.c_str(), c.kp, c.ki,
                    c.kd, lim.c_str(), c.dt);
    }

private:
    rcl_interfaces::msg::SetParametersResult on_params(const std::vector<rclcpp::Parameter>& params) {
        for (const rclcpp::Parameter& p : params) {
            const std::string& n = p.get_name();
            if ((n == "kp" || n == "ki" || n == "kd") && p.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE &&
                p.as_double() >= 0) {
                double& gain = n == "kp" ? logic_->ctrl.kp : (n == "ki" ? logic_->ctrl.ki : logic_->ctrl.kd);
                gain = p.as_double();
                RCLCPP_INFO(get_logger(), "%s -> %g", n.c_str(), gain);
            }
        }
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        return result;
    }

    void on_state(const sensor_msgs::msg::JointState& msg) {
        const auto [tau, dbg] = logic_->on_joint_state(msg.name, msg.position, msg.velocity);
        std_msgs::msg::Float64MultiArray cmd;
        cmd.data.assign(cmd_size_, 0.0);
        if (cmd_index_ < cmd.data.size()) cmd.data[cmd_index_] = tau;
        pub_cmd_->publish(cmd);
        std_msgs::msg::Float64MultiArray d;
        d.data.assign(dbg.begin(), dbg.end());
        pub_dbg_->publish(d);
    }

    std::unique_ptr<node_logic::PidNodeLogic> logic_;
    std::size_t cmd_size_ = 1, cmd_index_ = 0;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_cmd_, pub_dbg_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<JointPidNode>());
    rclcpp::shutdown();
    return 0;
}
