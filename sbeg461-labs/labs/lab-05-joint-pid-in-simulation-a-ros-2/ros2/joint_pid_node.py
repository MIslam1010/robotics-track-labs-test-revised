#!/usr/bin/env python3
"""SBEG 461 Lab 5, Part D (guided, ROS 2 Jazzy) -- a PID joint controller node built on YOUR JointPID.

  subscribes  /joint_states                  sensor_msgs/JointState   (the joint named by joint_name)
  publishes   /effort_controller/commands    std_msgs/Float64MultiArray  data[cmd_index] = tau (N m)
              /pid_debug                     std_msgs/Float64MultiArray  [t, qd, q, e, tau, P, I, D, saturated]

The controller runs once per received joint state (event-driven: the simulator, or ros2_control's
joint_state_broadcaster, sets the rate).  Controller time is k * dt, k = number of states received, so a run is
repeatable and identical to lab.simulate_joint (tested).  Gains can be changed while it runs:
    ros2 param set /joint_pid kp 12.0

Run, from the lab/ folder, after joint_sim_node.py:
    source /opt/ros/jazzy/setup.bash
    LAB_IMPL=starter python3 ros2/joint_pid_node.py --ros-args -p plant:=arm_j2 -p reference:=hold
    ros2 topic echo /pid_debug --csv > run1.csv           # in a third terminal; Ctrl-C to stop
Parameters
    plant        plant used for the default gains and torque limit (as in joint_sim_node.py)
    zeta, wn     target damping ratio and natural frequency; default kp, kd = lab.pd_gains(M, b, zeta, wn)
    kp, ki, kd   override the gains (negative = use the default); ki defaults to 0 (PD)
    tau_max      torque limit (negative = the plant's tau_max); anti_windup (bool)
    dt           sample time (s), must equal the simulator's dt
    reference    hold (keep the first measured angle) | step (to q_target at t_step) | knee_swing (data file)
    q_target, t_step, velocity (measured | difference), joint_name, cmd_size, cmd_index

To drive a Gazebo / ros2_control robot instead of joint_sim_node.py, configure an
effort_controllers/JointGroupEffortController named effort_controller for the joint(s) and set cmd_size and
cmd_index so that data[cmd_index] is this joint's torque.  The ROS-free part (class PidNodeLogic) is auto-tested.
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


def make_reference(kind: str, q_start: float, q_target: float = 0.0, t_step: float = 1.0):
    """Return ref(t) -> (qd, qd_dot) for the reference types of the node."""
    if kind == "hold":
        return lambda t: (q_start, 0.0)
    if kind == "step":
        return lambda t: ((q_target if t >= t_step else q_start), 0.0)
    if kind == "knee_swing":
        ts, qd, vd, _ = lab.load_knee_swing()

        def ref(t):
            i = min(int(round(t / (ts[1] - ts[0]))), len(ts) - 1)
            return float(qd[i]), float(vd[i])
        return ref
    raise ValueError(f"unknown reference '{kind}'")


class PidNodeLogic:
    """ROS-free controller logic: on_joint_state(names, positions, velocities) -> (tau, debug list)."""

    def __init__(self, ctrl, joint_name: str, reference: str = "hold", q_target: float = 0.0,
                 t_step: float = 1.0, velocity: str = "measured"):
        self.ctrl, self.joint = ctrl, joint_name
        self.kind, self.q_target, self.t_step, self.velocity = reference, q_target, t_step, velocity
        self.ref = None
        self.k = 0

    def on_joint_state(self, names, positions, velocities):
        i = list(names).index(self.joint) if self.joint in names else 0
        q = float(positions[i])
        dq = float(velocities[i]) if (self.velocity == "measured" and len(velocities) > i) else None
        if self.ref is None:                                   # first sample: start the reference here
            self.ref = make_reference(self.kind, q, self.q_target, self.t_step)
            self.ctrl.reset(q)
        t = self.k * self.ctrl.dt
        qd, vd = self.ref(t)
        tau = self.ctrl.update(qd, vd, q, dq)
        self.k += 1
        T = self.ctrl.terms
        return tau, [t, qd, q, qd - q, tau, T["P"], T["I"], T["D"], float(self.ctrl.saturated)]


def default_controller(plant_name: str, zeta: float, wn: float, dt: float, kp=-1.0, ki=0.0, kd=-1.0,
                       tau_max=-1.0, anti_windup=True):
    """JointPID with kp, kd from lab.pd_gains for the plant (unless overridden) and the plant's torque limit."""
    p = lab.load_plant(plant_name)
    kp0, kd0 = lab.pd_gains(p["M"], p["b"], zeta, wn)
    lim = p.get("tau_max") if tau_max < 0 else tau_max
    return lab.JointPID(kp0 if kp < 0 else kp, max(ki, 0.0), kd0 if kd < 0 else kd, dt, lim, anti_windup)


def main(args=None):
    import rclpy
    from rcl_interfaces.msg import SetParametersResult
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray

    class JointPidNode(Node):
        def __init__(self):
            super().__init__("joint_pid")
            P = lambda n, v: self.declare_parameter(n, v).value  # noqa: E731
            plant = P("plant", "arm_j2")
            ctrl = default_controller(plant, float(P("zeta", 1.0)), float(P("wn", 20.0)), float(P("dt", 0.005)),
                                      float(P("kp", -1.0)), float(P("ki", 0.0)), float(P("kd", -1.0)),
                                      float(P("tau_max", -1.0)), bool(P("anti_windup", True)))
            self.logic = PidNodeLogic(ctrl, P("joint_name", "joint"), P("reference", "hold"),
                                      float(P("q_target", 0.0)), float(P("t_step", 1.0)), P("velocity", "measured"))
            self.cmd_size, self.cmd_index = int(P("cmd_size", 1)), int(P("cmd_index", 0))
            self.pub_cmd = self.create_publisher(Float64MultiArray, "/effort_controller/commands", 10)
            self.pub_dbg = self.create_publisher(Float64MultiArray, "/pid_debug", 10)
            self.create_subscription(JointState, "/joint_states", self.on_state, 10)
            self.add_on_set_parameters_callback(self.on_params)
            self.get_logger().info(f"PID on '{plant}': kp={ctrl.kp:.4g} ki={ctrl.ki:.4g} kd={ctrl.kd:.4g} "
                                   f"tau_max={ctrl.tau_max} dt={ctrl.dt}")

        def on_params(self, params):
            for p in params:
                if p.name in ("kp", "ki", "kd") and p.value >= 0:
                    setattr(self.logic.ctrl, p.name, float(p.value))
                    self.get_logger().info(f"{p.name} -> {p.value}")
            return SetParametersResult(successful=True)

        def on_state(self, msg):
            tau, dbg = self.logic.on_joint_state(msg.name, msg.position, msg.velocity)
            cmd = Float64MultiArray()
            data = [0.0] * self.cmd_size
            data[self.cmd_index] = tau
            cmd.data = data
            self.pub_cmd.publish(cmd)
            d = Float64MultiArray()
            d.data = [float(v) for v in dbg]
            self.pub_dbg.publish(d)

    rclpy.init(args=args)
    node = JointPidNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
