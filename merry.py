import argparse
import baxter_interface
import helpers as h
import math
import numpy as np
import random
import rospy

import sys
import tf

from baxter_pykdl import baxter_kinematics


from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Twist
)

from planner.rrt import RRT
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import JointState
from solver.ik_solver import IKSolver
from visualization_msgs.msg import InteractiveMarkerFeedback

__author__ = 'shaun howard'

# amount of time to wait for IK solver service
MAX_TRIALS = 10
TIMEOUT = 5.0

OK = "OK"
ERROR = "ERROR"


class Merry:

    closest_points = []

    # initialize and enable the robot and the named limb
    def __init__(self, left_speed=0.3, right_speed=0.3, accuracy=baxter_interface.JOINT_ANGLE_TOLERANCE):

        rospy.init_node("merry")

        # start this up early so it's ready before the robot can use it
        # self.ik_solver = IKSolver()

        # wait for robot to boot up
        rospy.sleep(5)

        # Create baxter_interface limb instances for right and left arms
        self.right_arm = baxter_interface.Limb("right")
        self.left_arm = baxter_interface.Limb("left")
        self.joint_states = dict()

        # Parameters which will describe joint position moves
        self.right_speed = right_speed
        self.left_speed = left_speed
        self._accuracy = accuracy

        # Verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable()
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()

        # self.gripper = baxter_interface.Gripper(limb_name)
        self.joint_states_subscriber = rospy.Subscriber("robot/joint_states", JointState, self.joint_state_cb)

        self.kinect_subscriber = rospy.Subscriber("kinect/depth/points", pc2.PointCloud2, self.kinect_cb)

        self.marker_subscriber = rospy.Subscriber("/merry_end_point_markers/feedback", InteractiveMarkerFeedback,
                                                  self.interactive_marker_cb)
        self.tf = tf.TransformListener()

        # self.kmeans = h.get_kmeans_instance(self)

        self.left_kinematics = baxter_kinematics("left")

        self.right_kinematics = baxter_kinematics("right")

        rospy.on_shutdown(self.clean_shutdown)

        self.left_goal = None

        self.right_goal = None

        self.left_obstacle_waves = []

        self.right_obstacle_waves = []

    def move_to_joint_positions(self, joint_positions, side, use_move=True):
        """
        Moves the Baxter robot end effector to the given dict of joint velocities keyed by joint name.
        :return: a status about the move execution; in success, "OK", and in error, "ERROR".
        """
        status = "OK"
        rospy.sleep(1.0)

        # Set joint position speed ratio for execution
        if not rospy.is_shutdown() and joint_positions is not None:
            if side is "right":
                self.right_arm.set_joint_position_speed(self.right_speed)
                if use_move:
                    self.right_arm.move_to_joint_positions(joint_positions)
                else:
                    self.right_arm.set_joint_positions(joint_positions)
            elif side is "left":
                self.left_arm.set_joint_position_speed(self.left_speed)
                if use_move:
                    self.left_arm.move_to_joint_positions(joint_positions)
                else:
                    self.left_arm.set_joint_positions(joint_positions)
        else:
            rospy.logerr("Joint angles are unavailable at the moment. \
                          Will try to get goal angles soon, but staying put for now.")
            status = ERROR
        return status

    def check_and_execute_goal_angles(self, goal_angles, side):
        status = ERROR
        if goal_angles is not None:
            rospy.loginfo("got joint angles to execute!")
            rospy.loginfo("goal angles: " + str(goal_angles))
            joint_positions = goal_angles
            # joint_positions = self.get_angles_dict(goal_angles, side)
            print joint_positions
            # set the goal joint angles to reach the desired goal
            status = self.move_to_joint_positions(joint_positions, side, True)
            if status is ERROR:
                rospy.logerr("could not set joint positions for ik solver...")
        return status

    def get_goal_point(self, side):
        if side is "left":
            goal = self.left_goal
        else:
            goal = self.right_goal
        if goal:
            return h.point_to_ndarray(goal.position)
        else:
            return None

    def get_goal_pose(self, side):
        if side is "left":
            goal = self.left_goal
        else:
            goal = self.right_goal
        if goal:
            return h.pose_to_ndarray(goal)
        else:
            return None

    def get_obs_for_side(self, side):
        if side == "left":
            return self.left_obstacle_waves
        else:
            return self.right_obstacle_waves

    def get_kin(self, side):
        if side == "left":
            return self.left_kinematics
        else:
            return self.right_kinematics

    def grow_rrt(self, side, q_start, x_goal, obs_mapping_fn, dist_thresh=0.1, p_goal=0.5):
        rrt = RRT(q_start, x_goal, self.get_kin(side), side)
        while rrt.dist_to_goal() > dist_thresh:
            p = random.uniform(0, 1)
            if p >= p_goal:
                rrt.extend_toward_goal(self.get_obs_for_side(side), obs_mapping_fn, dist_thresh)
            else:
                rrt.ik_extend_randomly(self.get_obs_for_side(side), obs_mapping_fn, dist_thresh)
        return rrt

    def approach_goals(self):
        """
        Tries to approach the current goals.
        First, plans for left endpoint.
        --not yet: Second, plans for right endpoint.
        Then, redoes planning over again.
        Loops forever, kill with ctrl-c.
        """
        while True:
            left_rrt = None
            if self.left_goal is not None:
                left_rrt = self.grow_rrt("left", self.left_arm.joint_angles(), self.left_goal,
                                         self.map_point_to_wavefront_index)
            if left_rrt:
                for node in left_rrt.nodes:
                    self.check_and_execute_goal_angles(node, "left")
                self.left_arm.set_joint_position_speed(0.0)

            # right_rrt = None
            # if self.right_goal is not None:
            #     right_rrt = self.grow_rrt("right", self.right_arm.endpoint_pose(), self.right_goal)
            #
            # if right_rrt:
            #     for node in right_rrt.nodes:
            #         self.move_to_joint_positions(node, "right")
            #     self.right_arm.set_joint_position_speed(0.0)
        return 0

    def map_point_to_wavefront_index(self, curr_point, goal_point, step_size=0.1):
        # points are np arrays
        if goal_point is not None:
            dist = np.linalg.norm(goal_point - curr_point)
            rank = dist / step_size
            return int(math.floor(rank))
        else:
            return 0

    def create_obstacle_wave_maps(self):
        # wave maps are lists of lists, indexed by distance step from goal point for each side
        closest_points = self.closest_points
        left_obstacle_waves = list()
        right_obstacle_waves = list()
        min_dist = 1000000
        max_dist = 0
        # do left goal distance mapping first
        for point in closest_points:
            left_goal_point = self.left_goal[:3]
            indx = self.map_point_to_wavefront_index(point, left_goal_point)
            if len(left_obstacle_waves) > indx:
                left_obstacle_waves[indx].append(point)
            else:
                left_obstacle_waves.append([point])

        # do right goal distance mapping second
        for point in closest_points:
            right_goal_point = self.right_goal[:3]
            indx = self.map_point_to_wavefront_index(point, right_goal_point)
            if len(right_obstacle_waves) > indx:
                right_obstacle_waves[indx].append(point)
            else:
                right_obstacle_waves.append([point])

        self.left_obstacle_waves = left_obstacle_waves
        self.right_obstacle_waves = right_obstacle_waves

    def kinect_cb(self, data, source="kinect_link", dest="base", min_dist=0.2, max_dist=2, min_height=1):
        """
        Receives kinect points from the kinect subscriber linked to the publisher stream.
        :return: kinect points numpy array
        """
        points = [p for p in pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z"))]
        transformed_points = h.transform_pcl2(self.tf, dest, source, points, 3)
        self.closest_points = [h.point_to_ndarray(p) for p in transformed_points if min_dist < math.sqrt(p.x**2 + p.y**2 + p.z**2) < max_dist
                               and p.z > min_height]
        self.create_obstacle_wave_maps()
        if len(self.closest_points) > 0:
            print "kinect cb: there are this many close points: " + str(len(self.closest_points))

    def joint_state_cb(self, data):
        names = data.name
        positions = data.position
        velocities = data.velocity
        efforts = data.effort
        for i in range(len(names)):
            self.joint_states[names[i]] = dict()
            self.joint_states[names[i]]["position"] = positions[i]
            self.joint_states[names[i]]["velocity"] = velocities[i]
            self.joint_states[names[i]]["effort"] = efforts[i]

    def interactive_marker_cb(self, feedback):
        # store feedback pose as goal
        goal = feedback.pose
        goal = h.pose_to_ndarray(goal)

        # set right or left goal depending on marker name
        if "right" in feedback.marker_name:
            self.right_goal = goal
        elif "left" in feedback.marker_name:
            self.left_goal = goal
        else:
            rospy.loginfo("got singular end-point goal")

    def get_joint_angles(self, side="right"):
        """Gets the joint angle dictionary of the specified arm side."""
        joint_angles = self.right_arm.joint_angles() if side is "right" else self.left_arm.joint_angles()
        return joint_angles

    def get_joint_velocities(self, side="right"):
        """Gets the joint angle dictionary of the specified arm side."""
        joint_velocities = self.right_arm.joint_velocities() if side is "right" else self.left_arm.joint_velocities()
        return joint_velocities

    def get_angles_dict(self, angles_list, side):
        i = 0
        angles_dict = dict()
        joint_names = self.right_arm.joint_names() if side is "right" else self.left_arm.joint_names()
        for name in joint_names:
            angles_dict[name] = angles_list[i]
            i += 1
        return angles_dict

    def clean_shutdown(self):
        print("\nExiting example...")
        if not self._init_state:
            print("Disabling robot...")
            self._rs.disable()
        return True


def run_robot(args):
    print("Initializing node... ")
    rospy.init_node("merry")
    merry = Merry(args.limb, args.speed, args.accuracy)
    rospy.on_shutdown(merry.clean_shutdown)
    status = merry.approach_goals()
    sys.exit(status)


if __name__ == '__main__':
    run_robot(h.get_args())
