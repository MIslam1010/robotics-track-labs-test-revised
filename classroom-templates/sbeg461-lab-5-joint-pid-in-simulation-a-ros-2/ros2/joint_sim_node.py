#!/usr/bin/env python3
"""SBEG 461 Lab 5, Part D (guided, ROS 2 Jazzy) -- a one-joint plant simulator node.

Simulates  tau = M q'' + b q' + g(q)  (MR Eq. 11.21) for one plant of ../data/joint_plants.json and behaves like
the ros2_control pair "joint_state_broadcaster + effort controller" of a real or Gazebo robot:

  publishes   /joint_states                  sensor_msgs/JointState   (name, position, velocity, effort) every dt
  subscribes  /effort_controller/commands    std_msgs/Float64MultiArray  data[0] = joint torque (N m)

The torque received last is held until the next sample (zero-order hold) and clipped to the plant's tau_max.

Run (course ROS 2 Jazzy Docker image or Ubuntu 24.04), from the lab/ folder:
    source /opt/ros/jazzy/setup.bash
    python3 ros2/joint_sim_node.py --ros-args -p plant:=arm_j2 -p dt:=0.005
Parameters: plant (arm_j1 | arm_j2 | knee | eod_wrist | mr_link), dt (s), q0 (rad; default per plant),
substeps (RK4 steps per sample), joint_name.

rclpy is imported only inside main(), so the plant logic (class JointSimLogic) is importable and auto-tested
without ROS (test_lab.py::test_ros_logic_matches_offline_simulation).  Uses only the provided functions of the
lab module, so it works with the untouched starter too.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB))
lab = importlib.import_module(os.environ.get("LAB_IMPL", "starter"))

DEFAULT_Q0 = dict(arm_j1=0.0, arm_j2=-0.6, knee=float(np.deg2rad(5.0)), eod_wrist=0.0, mr_link=-np.pi / 2)


class JointSimLogic:
    """Plant state and integration, free of ROS.  apply(tau) stores the command; step() advances one sample."""

    def __init__(self, plant: dict, dt: float, q0: float, dq0: float = 0.0, substeps: int = 4):
        self.plant, self.dt, self.substeps = plant, float(dt), int(substeps)
        self.x = np.array([q0, dq0], float)
        self.tau = 0.0
        self.k = 0

    def apply(self, tau: float) -> None:
        lim = self.plant.get("tau_max")
        self.tau = float(tau) if lim is None else float(np.clip(tau, -lim, lim))

    def state(self):
        return float(self.x[0]), float(self.x[1]), self.tau

    def step(self):
        h = self.dt / self.substeps
        f = lambda s: lab.plant_rhs(self.plant, s, self.tau)  # noqa: E731
        for _ in range(self.substeps):
            self.x = lab.rk4_step(f, self.x, h)
        self.k += 1
        return self.state()


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray

    class JointSimNode(Node):
        def __init__(self):
            super().__init__("joint_sim")
            name = self.declare_parameter("plant", "arm_j2").value
            dt = float(self.declare_parameter("dt", 0.005).value)
            q0 = float(self.declare_parameter("q0", float(DEFAULT_Q0.get(name, 0.0))).value)
            substeps = int(self.declare_parameter("substeps", 10).value)
            self.joint = self.declare_parameter("joint_name", "joint").value
            self.logic = JointSimLogic(lab.load_plant(name), dt, q0, 0.0, substeps)
            self.pub = self.create_publisher(JointState, "/joint_states", 10)
            self.create_subscription(Float64MultiArray, "/effort_controller/commands", self.on_cmd, 10)
            self.publish(*self.logic.state())                   # sample k = 0
            self.create_timer(dt, self.tick)
            self.get_logger().info(f"simulating plant '{name}' at {1 / dt:.0f} Hz from q0 = {q0:.3f} rad")

        def on_cmd(self, msg):
            if msg.data:
                self.logic.apply(msg.data[0])

        def tick(self):
            self.publish(*self.logic.step())

        def publish(self, q, dq, tau):
            m = JointState()
            m.header.stamp = self.get_clock().now().to_msg()
            m.name = [self.joint]
            m.position, m.velocity, m.effort = [q], [dq], [tau]
            self.pub.publish(m)

    rclpy.init(args=args)
    node = JointSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
