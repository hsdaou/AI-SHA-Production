#!/usr/bin/env python3
"""
Waypoint Resolver Node — bridges brain_node NAV intents to Nav2 goals.

Subscribes to /nav_goal (String) from brain_node, resolves the natural-
language destination to map coordinates via nav_locations.json, and sends
a NavigateToPose action goal to the Nav2 stack.

Architecture:
  brain_node  --/nav_goal-->  waypoint_resolver  --NavigateToPose-->  nav2
                                                                      |
                                                                  /cmd_vel
                                                                      |
                                                              mecanum_driver (Pi 4b)

Prerequisites:
  - A SLAM-generated map loaded by nav2's map_server
  - nav2_bringup running (planner_server, controller_server, bt_navigator)
  - nav_locations.json calibrated with real map coordinates

nav_locations.json format:
  {
    "locations": {
      "admin office":  {"x": 1.2, "y": 3.4, "oz": 0.0, "ow": 1.0},
      "cafeteria":     {"x": 5.6, "y": 7.8, "oz": 0.707, "ow": 0.707},
      "main entrance": {"x": 0.0, "y": 0.0, "oz": 0.0, "ow": 1.0}
    },
    "aliases": {
      "office": "admin office",
      "front door": "main entrance",
      "food": "cafeteria"
    }
  }

  oz/ow are the Z and W components of the orientation quaternion from
  amcl_pose.  Legacy "yaw" (radians) is also supported but quaternion
  is preferred — paste directly from 'ros2 topic echo /amcl_pose'.
"""

import math
import json
import os

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from std_msgs.msg import String

# Nav2 may not be installed yet — graceful degradation
_NAV2_AVAILABLE = False
try:
    from nav2_msgs.action import NavigateToPose
    from rclpy.action import ActionClient
    from geometry_msgs.msg import PoseStamped
    _NAV2_AVAILABLE = True
except ImportError:
    pass


class WaypointResolverNode(Node):
    def __init__(self):
        super().__init__('waypoint_resolver')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('nav_locations_file', '',
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Path to nav_locations.json (empty = use default)'))
        self.declare_parameter('nav2_timeout_sec', 5.0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Seconds to wait for Nav2 action server'))

        locations_file = self.get_parameter(
            'nav_locations_file').get_parameter_value().string_value
        self.nav2_timeout = self.get_parameter(
            'nav2_timeout_sec').get_parameter_value().double_value

        # ── Load location map ─────────────────────────────────────────────
        self.locations: dict = {}
        self.aliases: dict = {}

        if not locations_file:
            # Default: look for nav_locations.json next to this file
            locations_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..', 'config', 'nav_locations.json'
            )

        self._load_locations(locations_file)

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(String, '/nav_goal', self._on_nav_goal, 10)

        # ── Speech feedback (reuse brain_node's speech bus) ───────────────
        self.speech_pub = self.create_publisher(String, '/robot_speech', 10)

        # ── Nav2 action client ────────────────────────────────────────────
        self.nav2_client = None
        if _NAV2_AVAILABLE:
            self.nav2_client = ActionClient(
                self, NavigateToPose, 'navigate_to_pose')
            self.get_logger().info('Nav2 ActionClient created')
        else:
            self.get_logger().warning(
                'nav2_msgs not installed — waypoint_resolver will resolve '
                'locations but cannot send goals. Install nav2_msgs to enable.')

        self.get_logger().info(
            f'Waypoint resolver ready: {len(self.locations)} locations, '
            f'{len(self.aliases)} aliases, '
            f'nav2={"connected" if self.nav2_client else "unavailable"}')

    # ══════════════════════════════════════════════════════════════════════
    # Location loading
    # ══════════════════════════════════════════════════════════════════════

    def _load_locations(self, path: str):
        """Load nav_locations.json with locations and optional aliases."""
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            self.get_logger().warning(
                f'nav_locations.json not found at {path} — '
                f'NAV goals will fail until file is created')
            return

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            if 'locations' in data:
                self.locations = data['locations']
            else:
                # Flat format: entire file is the location dict
                self.locations = {
                    k: v for k, v in data.items() if k != 'aliases'
                }

            self.aliases = data.get('aliases', {})
            self.get_logger().info(f'Loaded {len(self.locations)} locations from {path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load {path}: {e}')

    # ══════════════════════════════════════════════════════════════════════
    # Goal resolution
    # ══════════════════════════════════════════════════════════════════════

    def _resolve_location(self, text: str):
        """
        Fuzzy-match a natural-language destination to a known location.

        Tries exact match first, then alias lookup, then substring search.
        Returns (name, {x, y, yaw}) or (None, None).
        """
        query = text.strip().lower()

        # 1. Exact match
        for name, coords in self.locations.items():
            if name.lower() == query:
                return name, coords

        # 2. Alias lookup
        for alias, target in self.aliases.items():
            if alias.lower() == query:
                if target in self.locations:
                    return target, self.locations[target]

        # 3. Substring match (e.g. "take me to the cafeteria" → "cafeteria")
        for name, coords in self.locations.items():
            if name.lower() in query:
                return name, coords

        # 4. Alias substring match
        for alias, target in self.aliases.items():
            if alias.lower() in query:
                if target in self.locations:
                    return target, self.locations[target]

        return None, None

    # ══════════════════════════════════════════════════════════════════════
    # Nav goal handler
    # ══════════════════════════════════════════════════════════════════════

    def _on_nav_goal(self, msg: String):
        """Handle /nav_goal from brain_node."""
        text = msg.data.strip()
        if not text:
            return

        self.get_logger().info(f'Received nav goal: "{text}"')

        name, coords = self._resolve_location(text)

        if coords is None:
            self.get_logger().warning(f'Unknown location: "{text}"')
            self._say(
                f"I don't know where \"{text}\" is. "
                f"I know these locations: {', '.join(self.locations.keys())}."
            )
            return

        x = float(coords.get('x', 0.0))
        y = float(coords.get('y', 0.0))

        # Support both quaternion (oz/ow from RViz/amcl) and yaw (legacy).
        # Quaternion is preferred — avoids sin/cos conversion errors.
        if 'oz' in coords and 'ow' in coords:
            oz = float(coords['oz'])
            ow = float(coords['ow'])
        else:
            yaw = float(coords.get('yaw', 0.0))
            oz = math.sin(yaw / 2.0)
            ow = math.cos(yaw / 2.0)

        self.get_logger().info(
            f'Resolved "{text}" → {name} (x={x:.2f}, y={y:.2f}, oz={oz:.3f}, ow={ow:.3f})')
        self._say(f"Navigating to {name}.")

        self._send_nav2_goal(x, y, oz, ow, name)

    def _send_nav2_goal(self, x: float, y: float, oz: float, ow: float, name: str):
        """Send a NavigateToPose goal to Nav2."""
        if not _NAV2_AVAILABLE or self.nav2_client is None:
            self.get_logger().error(
                'Nav2 not available — cannot navigate. '
                'Install nav2_msgs and launch nav2_bringup.')
            self._say("Navigation hardware is not ready yet.")
            return

        if not self.nav2_client.wait_for_server(timeout_sec=self.nav2_timeout):
            self.get_logger().error(
                f'Nav2 action server not responding after {self.nav2_timeout}s')
            self._say("The navigation system is not responding. Please try again later.")
            return

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.z = oz
        goal.pose.pose.orientation.w = ow

        self.get_logger().info(f'Sending Nav2 goal: ({x:.2f}, {y:.2f}, oz={oz:.3f}, ow={ow:.3f})')
        future = self.nav2_client.send_goal_async(
            goal, feedback_callback=self._nav2_feedback)
        future.add_done_callback(
            lambda f: self._nav2_goal_response(f, name))

    def _nav2_feedback(self, feedback_msg):
        """Nav2 handles path planning and local obstacle avoidance."""
        pass

    def _nav2_goal_response(self, future, name: str):
        """Handle Nav2 goal acceptance/rejection."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warning(f'Nav2 rejected goal for "{name}"')
            self._say(f"I can't reach {name} right now. The path may be blocked.")
            return

        self.get_logger().info(f'Nav2 accepted goal for "{name}" — navigating')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f: self._nav2_result(f, name))

    def _nav2_result(self, future, name: str):
        """Handle Nav2 navigation result."""
        result = future.result()
        status = result.status

        # action_msgs/GoalStatus: STATUS_SUCCEEDED = 4
        if status == 4:
            self.get_logger().info(f'Arrived at {name}')
            self._say(f"I've arrived at {name}.")
        else:
            self.get_logger().warning(f'Navigation to {name} failed (status={status})')
            self._say(f"I couldn't reach {name}. Something blocked my path.")

    # ══════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════

    def _say(self, text: str):
        """Publish speech feedback on /robot_speech."""
        msg = String()
        msg.data = text
        self.speech_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointResolverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
