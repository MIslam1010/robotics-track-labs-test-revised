#!/usr/bin/env python3
"""SBEG 361 Lab 2, Part B (guided, ROS 2 Jazzy) -- publish the navigation transform tree to tf2.

  * every entry of ../nav_tree.json      -> /tf_static  (tf2_ros.StaticTransformBroadcaster)
  * ../tool_track.csv (T_tracker_tool_marker(t)), replayed in a loop at 50 Hz -> /tf
    (tf2_ros.TransformBroadcaster)

Run (inside the course ROS 2 Jazzy Docker image or an Ubuntu 24.04 install):
    source /opt/ros/jazzy/setup.bash
    python3 nav_tf_broadcaster.py
then, in other terminals:
    rviz2                                              # Fixed Frame = theatre, Add -> TF
    ros2 run tf2_ros tf2_echo ct_image tool_tip        # prints T_ct_image_tool_tip
    ros2 run tf2_tools view_frames                     # writes frames_<date>.pdf of the tree

tf2 conventions used here (they match the frame conventions of starter.py and the lab handout):
    TransformStamped.header.frame_id = parent, .child_frame_id = child,
    .transform = pose of the child expressed in the parent = T_parent_child;
    quaternions are (x, y, z, w), scalar LAST (MR App. B.3 writes the scalar first).
This file is not auto-tested (no ROS on the grading server); it only needs numpy and rclpy.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

LAB = Path(__file__).resolve().parent.parent


def rot_to_quat_xyzw(R):
    """Unit quaternion (x, y, z, w) of a rotation matrix (Shepperd's method; robust near pi)."""
    R = np.asarray(R, float)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * math.sqrt(tr + 1.0)
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def make_msg(stamp, parent, child, translation, quat_xyzw):
    m = TransformStamped()
    m.header.stamp = stamp
    m.header.frame_id = parent
    m.child_frame_id = child
    m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = map(float, translation)
    (m.transform.rotation.x, m.transform.rotation.y,
     m.transform.rotation.z, m.transform.rotation.w) = map(float, quat_xyzw)
    return m


class NavTfBroadcaster(Node):
    def __init__(self):
        super().__init__("sbeg361_nav_tf")
        self.static_br = StaticTransformBroadcaster(self)
        self.br = TransformBroadcaster(self)
        tree = json.loads((LAB / "nav_tree.json").read_text())
        now = self.get_clock().now().to_msg()
        self.static_br.sendTransform([make_msg(now, e["parent"], e["child"], e["translation"], e["rotation_xyzw"])
                                      for e in tree["transforms"]])
        with open(LAB / "tool_track.csv") as fh:
            rows = list(csv.reader(fh))[1:]
        self.track = [(np.array(r[1:10], float).reshape(3, 3), np.array(r[10:13], float)) for r in rows]
        self.k = 0
        self.timer = self.create_timer(0.02, self.tick)          # 50 Hz, same as the tracker
        self.get_logger().info(f"static edges: {len(tree['transforms'])}, tool samples: {len(self.track)}")

    def tick(self):
        R, p = self.track[self.k]
        # EXERCISE B3: which frame is the parent here, and why must it be 'tracker'?
        self.br.sendTransform(make_msg(self.get_clock().now().to_msg(), "tracker", "tool_marker",
                                       p, rot_to_quat_xyzw(R)))
        self.k = (self.k + 1) % len(self.track)


def main():
    rclpy.init()
    node = NavTfBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
