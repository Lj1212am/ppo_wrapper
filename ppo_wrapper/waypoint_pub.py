#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
import numpy as np
import sys
import os

# Add necessary path for NMPC imports
sys.path.append('/home/nvidia/ros_ws/src/f1planning_ros_wrapper/f1tenth_planning')

# NMPC Imports
from f1tenth_gym.envs.track import Track

# ROS2 imports for visualization markers
from visualization_msgs.msg import MarkerArray, Marker

class WaypointPublisher(Node):
    def __init__(self):
        super().__init__('nmpc_planner_node')
        
        # Update the config path to point to your CSV file's directory
        self.config_path = "/home/lee/Hybrid_DRL_Deployments/source/isaaclab_tasks/isaaclab_tasks/direct/hunter_hybrid"
        self.csv = "levine_transformed.csv"  # Use your CSV file name here
        self.map_name = os.path.join(self.config_path, self.csv)
        
        # Load CSV: assuming the file is comma-separated with at least 2 columns (x and y)
        coordinates = np.genfromtxt(self.map_name, delimiter=',')
        # Subsample every 10th row as before
        x_coords = coordinates[::10, 0]
        y_coords = coordinates[::10, 1]
        # Assume a constant velocity if not provided (default to 3.0)
        v = np.ones_like(x_coords) * 3.0
        
        # Create a Track object from the reference line
        self.track = Track.from_refline(x_coords, y_coords, v)
        
        # Optional: store initial positions if needed
        self.initial_x = None
        self.initial_y = None
        
        # Extract raceline waypoints and subsample further if needed.
        WAYPOINTS_SUBSAMPLE_STEP = 10
        waypointx = self.track.raceline.xs
        waypointy = self.track.raceline.ys
        waypointyaw = self.track.raceline.yaws
        self.waypoints = np.column_stack((waypointx, waypointy, waypointyaw))
        self.waypoints = self.waypoints[::WAYPOINTS_SUBSAMPLE_STEP, :]

        # Create MarkerArray for publishing waypoints as markers in RViz.
        self.marker_array = MarkerArray()
        
        # Create markers for each waypoint.
        ref = True  # Flag to choose marker color scheme.
        for i, waypoint in enumerate(self.waypoints):
            marker = Marker()
            marker.header.frame_id = "map"  # Set your frame ID as needed.
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.id = i + 1000  # Unique ID for each marker.
            marker.scale.x = 1.0  # Arrow length.
            marker.scale.y = 0.2  # Arrow width.
            marker.scale.z = 0.2  # Arrow height.
            
            # Set marker position (x, y, and a fixed z for visualization).
            marker.pose.position.x = float(waypoint[0])
            marker.pose.position.y = float(waypoint[1])
            marker.pose.position.z = 0.2

            # Convert yaw to quaternion for orientation.
            yaw = float(waypoint[2])
            q = self.yaw_to_quaternion(yaw)
            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]

            # Set marker color: green for reference.
            if ref:
                marker.color.a = 1.0
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            else:
                marker.color.a = 1.0
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 1.0

            self.marker_array.markers.append(marker)
        
        self.get_logger().info("Waypoint Node Initialized")

        # Set up a timer to publish waypoints as markers.
        self.timer = self.create_timer(0.1, self.publish_waypoints_as_markers)

    def publish_waypoints_as_markers(self):
        """Publish waypoints as visualization markers in RViz."""
        self.marker_pub.publish(self.marker_array)

    def yaw_to_quaternion(self, yaw):
        """Convert a yaw angle to a quaternion (x, y, z, w)."""
        return [0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]
    
def main(args=None):
    rclpy.init(args=args)
    node = WaypointPublisher()
    
    # Create publisher for markers.
    node.marker_pub = node.create_publisher(MarkerArray, 'waypoints_markers', 10)
    
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
