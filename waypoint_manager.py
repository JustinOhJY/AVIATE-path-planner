#!/usr/bin/env python

import rospy
import math
from geometry_msgs.msg import PoseStamped

# MAVROS services and messages
from mavros_msgs.srv import (
    SetMode,
    CommandBool,
    CommandTOL,
    WaypointPush,
    WaypointPushRequest
)
from mavros_msgs.msg import Waypoint
from path_planner import RRTStar

class WaypointManagerNode(object):
    """
    A ROS node that:
      - Maintains a list of MAVROS Waypoints.
      - Subscribes to /new_local_waypoint (optionally).
      - Runs a pre-mission sequence (GUIDED -> ARM -> TAKEOFF -> AUTO) once we have enough waypoints.
    """

    def __init__(self):
        rospy.init_node('waypoint_manager_node', anonymous=True)

        # Internal list of MAVROS Waypoint objects
        self.mavros_waypoints = []

        # Wait for MAVROS services to be available
        rospy.loginfo("Waiting for MAVROS services to become available...")
        rospy.wait_for_service('/minihawk_SIM/mavros/set_mode')
        rospy.wait_for_service('/minihawk_SIM/mavros/cmd/arming')
        rospy.wait_for_service('/minihawk_SIM/mavros/cmd/takeoff')
        rospy.wait_for_service('/minihawk_SIM/mavros/mission/push')
        rospy.loginfo("All required MAVROS services are available.")

        # Create service proxies
        self.set_mode_srv = rospy.ServiceProxy('/minihawk_SIM/mavros/set_mode', SetMode)
        self.arming_srv   = rospy.ServiceProxy('/minihawk_SIM/mavros/cmd/arming', CommandBool)
        self.takeoff_srv  = rospy.ServiceProxy('/minihawk_SIM/mavros/cmd/takeoff', CommandTOL)
        self.wp_push_srv  = rospy.ServiceProxy('/minihawk_SIM/mavros/mission/push', WaypointPush)

        # Subscribe to a local waypoint topic (optional)
        rospy.Subscriber("/new_local_waypoint", PoseStamped, self.local_waypoint_callback)

        # How many waypoints must we have before pushing/starting the mission?
        self.min_waypoints_needed = rospy.get_param("~min_waypoints_needed", 1)

        rospy.loginfo("WaypointManagerNode initialized. Will wait for %d waypoints before takeoff.",
                      self.min_waypoints_needed)

    def local_waypoint_callback(self, msg):
        """
        Callback for manual/other modules that publish local waypoints to /new_local_waypoint.
        These get converted into MAVROS Waypoints (demo: x->lat, y->lon, z->alt).
        NOTE: If you actually want local (x, y, z) in meters, consider MAV_FRAME_LOCAL_* 
              or do a proper lat/lon conversion here.
        """
        rospy.loginfo("Received a local waypoint from /new_local_waypoint.")
        new_wp = Waypoint()
        new_wp.frame = 3            # MAV_FRAME_GLOBAL_REL_ALT
        new_wp.command = 16         # MAV_CMD_NAV_WAYPOINT
        new_wp.is_current = False
        new_wp.autocontinue = True
        new_wp.x_lat = msg.pose.position.x
        new_wp.y_long = msg.pose.position.y
        new_wp.z_alt = msg.pose.position.z

        self.mavros_waypoints.append(new_wp)
        rospy.loginfo("Appended waypoint [%.3f, %.3f, %.3f]. Total = %d",
                      new_wp.x_lat, new_wp.y_long, new_wp.z_alt, len(self.mavros_waypoints))

    def push_waypoints_to_vehicle(self):
        """
        Push all stored MAVROS Waypoints (self.mavros_waypoints) to the flight controller.
        """
        if not self.mavros_waypoints:
            rospy.logwarn("No waypoints to push!")
            return

        req = WaypointPushRequest()
        req.start_index = 0
        req.waypoints = self.mavros_waypoints

        rospy.loginfo("Pushing %d waypoints to the vehicle...", len(req.waypoints))
        try:
            res = self.wp_push_srv(req)
            if res.success:
                rospy.loginfo("Waypoints successfully pushed: %s", res)
            else:
                rospy.logerr("Failed to push waypoints (success=False).")
        except rospy.ServiceException as e:
            rospy.logerr("Failed to call /mission/push: %s", e)

    def run_pre_mission_sequence(self):
        rospy.loginfo("Switching to GUIDED mode.")
        try:
            self.set_mode_srv(base_mode=0, custom_mode='GUIDED')
        except rospy.ServiceException as e:
            rospy.logerr("Failed to set GUIDED mode: %s", e)
            return

        rospy.loginfo("Arming the vehicle...")
        try:
            self.arming_srv(True)
        except rospy.ServiceException as e:
            rospy.logerr("Failed to arm vehicle: %s", e)
            return

        # Take off to 10 meters (example: using zeroed lat/lon to indicate "use current location")
        rospy.loginfo("Taking off to 10m altitude...")
        try:
            self.takeoff_srv(
                min_pitch=0.0,
                yaw=0.0,
                latitude=0.0,   # or lat0,
                longitude=0.0,  # or lon0,
                altitude=100.0
            )
        except rospy.ServiceException as e:
            rospy.logerr("Failed takeoff command: %s", e)
            return

        rospy.sleep(5.0)  # wait a bit for takeoff in SITL

        rospy.loginfo("Switching to AUTO mode to start mission...")
        try:
            self.set_mode_srv(base_mode=0, custom_mode='AUTO')
        except rospy.ServiceException as e:
            rospy.logerr("Failed to set AUTO mode: %s", e)
            return

        rospy.loginfo("Arming again in AUTO mode (if required by SITL).")
        try:
            self.arming_srv(True)
        except rospy.ServiceException as e:
            rospy.logerr("Could not arm in AUTO: %s", e)
            return

        rospy.loginfo("Pre-mission sequence complete! Vehicle should start the mission.")


def main():
    # 1) Create the Waypoint Manager Node
    node = WaypointManagerNode()

    # Reference latitude and longitude for SITL
    lat0 = -35.3632583
    lon0 = 149.1652068
    lat0_rad = math.radians(lat0)

    # 2) Run the RRTStar Path Planner to generate waypoints (in *local meters*)
    rospy.loginfo("Running RRTStar to generate a path...")

    space_bounds = [
        (-20, 120),
        (-20, 120),
        (0, 120)
    ]
    start = (0, 10, 0)
    goal = (80, 80, 40)
    obstacle_list = [
        ("rectprism", 40, 40, 30, 10, 10, 30),
        ("rectprism", 60, 20, 20, 5, 5, 10),
        ("rectprism", 12.5, 40, 30, 10, 10, 30),
        ("rectprism", 62.5, 40, 30, 10, 10, 30)
    ]

    rrt_star = RRTStar(
        start=start,
        goal=goal,
        obstacle_list=obstacle_list,
        space_bounds=space_bounds,
        step_size=5.0,
        expand_distance=2.0,
        goal_sample_rate=5,
        max_iter=5000,
        smoothness_weight=15.0,
        max_turn_angle_deg=60.0,
        vision_range=20.0
    )

    path = rrt_star.planning()
    if path is None:
        rospy.logwarn("RRTStar planning failed: no path found.")
    else:
        rospy.loginfo("RRTStar found a path! Discretizing the path at 2m steps...")
        discrete_path = rrt_star.discretize_path(path, step=2)
        rospy.loginfo("Discretized path has %d waypoints.", len(discrete_path))

        # Clear any previously stored waypoints
        node.mavros_waypoints = []

        # Convert the local path into MAVROS waypoints (global lat/lon)
        for (x_meters, y_meters, z_alt) in discrete_path:
            # Convert local XY (in meters) to lat/lon (in degrees)
            # Approx. 1 deg lat ~ 111,320m; 1 deg lon ~ 111,320m * cos(latitude)
            delta_lat_deg = y_meters / 111320.0
            delta_lon_deg = x_meters / (111320.0 * math.cos(lat0_rad))

            lat = lat0 + delta_lat_deg
            lon = lon0 + delta_lon_deg

            mav_wp = Waypoint()
            mav_wp.frame = 3            # MAV_FRAME_GLOBAL_REL_ALT
            mav_wp.command = 16         # MAV_CMD_NAV_WAYPOINT
            mav_wp.is_current = False
            mav_wp.autocontinue = True
            mav_wp.x_lat = lat
            mav_wp.y_long = lon
            mav_wp.z_alt = z_alt
            node.mavros_waypoints.append(mav_wp)

        rospy.loginfo("Added %d RRT-based waypoints to the manager.", len(node.mavros_waypoints))

    # 3) Main Loop: Once we have enough waypoints, push them & run the mission sequence
    rate = rospy.Rate(1)  # 1 Hz
    mission_started = False

    while not rospy.is_shutdown() and not mission_started:
        if len(node.mavros_waypoints) >= node.min_waypoints_needed:
            rospy.loginfo("We have at least %d waypoints, pushing to vehicle...",
                          node.min_waypoints_needed)
            node.push_waypoints_to_vehicle()
            node.run_pre_mission_sequence()
            mission_started = True

        rate.sleep()

    # Keep spinning for any remaining callbacks (e.g., /new_local_waypoint)
    rospy.spin()


if __name__ == '__main__':
    main()
