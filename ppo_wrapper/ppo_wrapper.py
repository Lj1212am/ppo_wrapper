#!/usr/bin/env python3
"""
ROS 2 node that wraps a PPO driving policy and publishes /drive commands
from a plain torch.save() checkpoint.

Assumptions
-----------
• Observation dim = 7   (pos_x, pos_y, cte, heading_err, roll, yaw, lin_vel)
• Action dim       = 2   (raw_steering, speed)
"""
import math
from typing import Tuple, Union

import numpy as np
import rclpy
import torch
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from ppo_wrapper.actor_critic import ActorCritic, strip_prefix


# ───────────────────────── helpers ───────────────────────────────────────────
def quaternion_to_euler(q) -> Tuple[float, float, float]:
    sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    pitch = np.pi / 2 * np.sign(sinp) if abs(sinp) >= 1 else np.arcsin(sinp)

    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


# ───────────────────────── main node ─────────────────────────────────────────
class PPOModelWrapperNode(Node):
    def __init__(self):
        super().__init__("ppo_model_wrapper")
        self.get_logger().info("🚗 PPO Wrapper starting (torch.load)…")

        ckpt_path = "/home/lee/ros2_ws/src/ppo_wrapper/model/best_agent.pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build ActorCritic and load weights
        self.model: ActorCritic = ActorCritic(input_dim=7, output_dim=2).to(self.device)
        state_dict = self._extract_state_dict(torch.load(ckpt_path, map_location=self.device))
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            self.get_logger().warn(f"Missing keys: {missing}")
        if unexpected:
            self.get_logger().warn(f"Unexpected keys: {unexpected}")
        self.model.eval()
        self.get_logger().info(f"✓ Loaded checkpoint from {ckpt_path}")

        # Waypoints for CTE calculation
        csv = (
            "/home/lee/Hybrid_DRL_Deployments/source/"
            "isaaclab_tasks/isaaclab_tasks/direct/hunter_hybrid/levine_transformed.csv"
        )
        self.track_points = self._load_track(csv)

        # ROS 2 I/O
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.create_subscription(Odometry, "/ego_racecar/odom", self.odom_cb, 10)
        self.create_subscription(PointStamped, "/clicked_point", self.click_cb, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self.goal_cb, 10)
        self.create_subscription(LaserScan, "/scan", self.scan_cb, 10)

        self.goal_pose: Union[PoseStamped, None] = None

    # ───────────── load helpers ─────────────
    def _extract_state_dict(self, ckpt: dict) -> dict:
        """Return the actual state‑dict inside an arbitrary checkpoint dict."""
        if not isinstance(ckpt, dict):
            raise RuntimeError("Checkpoint is not a dict; cannot extract state‑dict.")

        for k in (
            "policy",              # SKRL
            "actor", "actor_state_dict",
            "model_state_dict", "state_dict", "model", "network",
        ):
            if k in ckpt and isinstance(ckpt[k], dict):
                sd = ckpt[k]
                break
        else:
            if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                sd = ckpt
            else:
                raise RuntimeError("No suitable state‑dict key found.")

        sd = strip_prefix(sd, "a2c_network.")
        sd = strip_prefix(sd, "module.")
        return sd

    def _load_track(self, path: str) -> np.ndarray:
        try:
            pts = np.genfromtxt(path, delimiter=",")[::10, :2]
            self.get_logger().info(f"Loaded {len(pts)} waypoints.")
            return pts
        except Exception as e:
            self.get_logger().error(f"Track CSV failed: {e} – using fallback.")
            return np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float32)

    # ───────────── callbacks ─────────────
    def odom_cb(self, msg: Odometry) -> None:
        pos_x, pos_y = msg.pose.pose.position.x, msg.pose.pose.position.y
        lin_vel = msg.twist.twist.linear.x
        roll, _, yaw = quaternion_to_euler(msg.pose.pose.orientation)

        # CTE + heading error
        dists = np.linalg.norm(self.track_points - np.array([pos_x, pos_y]), axis=1)
        i_min = int(np.argmin(dists))
        cte = dists[i_min]

        nxt = (
            self.track_points[i_min + 1]
            if i_min < len(self.track_points) - 1
            else self.track_points[i_min]
        )
        desired_yaw = math.atan2(nxt[1] - self.track_points[i_min][1],
                                 nxt[0] - self.track_points[i_min][0])
        heading_err = (desired_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        if heading_err <= 0.0:
            cte = -cte

        obs = np.array(
            [pos_x, pos_y, cte, heading_err, roll, yaw, lin_vel], dtype=np.float32
        )
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_tensor = self.model(obs_tensor)[0]   # shape (1, 2) or (2,)
        # action_np = action_tensor.detach().cpu().numpy().flatten()
        # if action_np.shape[0] != 2:
        #     self.get_logger().error(f"Unexpected action shape: {action_np.shape}")
        #     return
        # raw_delta, speed = float(action_np[0]), float(action_np[1])

        raw = action_tensor.detach().cpu().numpy().flatten()    # [-1, 1]
        raw_delta = float(np.clip(raw[0] * 0.1,  -0.75, 0.75))  # steering rad at joint
        speed      = float(np.clip(raw[1] * 10.0, -50.0, 50.0)) # m/s (positive == forward)
        
        # Ackermann steering conversion
        delta_out = math.atan(0.608 * math.tan(raw_delta) /
                              (0.608 + 0.5 * 0.554 * math.tan(raw_delta)))
        delta_in = math.atan(0.608 * math.tan(raw_delta) /
                             (0.608 - 0.5 * 0.554 * math.tan(raw_delta)))
        steering = delta_in if raw_delta <= 0 else delta_out

        msg_out = AckermannDriveStamped()
        msg_out.drive.steering_angle = steering
        msg_out.drive.speed = speed
        self.cmd_pub.publish(msg_out)
        self.get_logger().info(
            f"/drive → steering={steering:+.2f} rad  speed={speed:+.2f} m/s"
        )

    def click_cb(self, msg: PointStamped) -> None:
        self.get_logger().info(
            f"Clicked point: ({msg.point.x:+.2f}, {msg.point.y:+.2f})"
        )

    def goal_cb(self, msg: PoseStamped) -> None:
        self.goal_pose = msg
        self.get_logger().info("Goal pose updated.")

    def scan_cb(self, msg: LaserScan) -> None:
        pass


# ───────────────────────── entry point ───────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = PPOModelWrapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt – shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
