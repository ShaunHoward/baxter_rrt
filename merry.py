import argparse
import baxter_interface
import helpers as h
import math
import numpy as np
import random
import rospy

import sys

#from baxter_pykdl import baxter_kinematics
from solver.ik_solver import KDLIKSolver

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Twist
)

from planner.rrt import RRT
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import JointState
from sensor_msgs.msg import PointCloud
from solver.ik_solver import RethinkIKSolver
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
    def __init__(self, left_speed=0.5, right_speed=0.5, accuracy=baxter_interface.JOINT_ANGLE_TOLERANCE):

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
        self.joint_states_subscriber = rospy.Subscriber("robot/joint_states", JointState, self.joint_state_cb,
                                                        queue_size=10)

        self.marker_subscriber = rospy.Subscriber("/merry_end_point_markers/feedback", InteractiveMarkerFeedback,
                                                  self.interactive_marker_cb, queue_size=5)

        self.left_obs_sub = rospy.Subscriber("left_arm_obstacles", PointCloud, self.left_obs_cb, queue_size=10)
        self.right_obs_sub = rospy.Subscriber("right_arm_obstacles", PointCloud, self.right_obs_cb, queue_size=10)

        # self.kmeans = h.get_kmeans_instance(self)

        self.left_kinematics = KDLIKSolver("left")

        self.right_kinematics = KDLIKSolver("right")

        rospy.on_shutdown(self.clean_shutdown)

        self.left_goal = None

        self.left_goal_arr = None

        self.right_goal = None

        self.right_goal_arr = None

        self.left_obstacles = []

        self.right_obstacles = []

        self.left_rrt = None

        self.right_rrt = None

    def unpack_obstacle_points(self, data, side):
        points_unpacked = []
        for point in data.points:
            points_unpacked.append((point.x, point.y, point.z))
        point_arr = np.mat(points_unpacked)
        if side == "left":
            self.left_obstacles = point_arr
        else:
            self.right_obstacles = point_arr

    def left_obs_cb(self, data):
        self.unpack_obstacle_points(data, "left")

    def right_obs_cb(self, data):
        self.unpack_obstacle_points(data, "right")

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
                #self.right_arm.set_joint_position_speed(self.right_speed)
                if use_move:
                    self.right_arm.move_to_joint_positions(joint_positions)
                else:
                    self.right_arm.set_joint_positions(joint_positions)
            elif side is "left":
                #self.left_arm.set_joint_position_speed(self.left_speed)
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
        # TODO, filter left arm for left cloud, filter right arm for right cloud
        if side == "left":
            return self.left_obstacles
        else:
            return self.right_obstacles

    def update_rrt_obstacles(self):
        if self.left_rrt is not None:
            self.left_rrt.update_obstacles(self.left_obstacles)
        if self.right_rrt is not None:
            self.right_rrt.update_obstacles(self.right_obstacles)

    def get_kin(self, side):
        if side == "left":
            return self.left_kinematics
        else:
            return self.right_kinematics

    def grow(self, rrt, dist_thresh, p_goal):
        if rrt is not None:
            while rrt.dist_to_goal() > dist_thresh:
                p = random.uniform(0, 1)
                if p < p_goal:
                    print "using jacobian extend"
                    rrt.extend_toward_goal(dist_thresh)
                else:
                    print "using ik random extend"
                    pos = self.left_arm.endpoint_pose()["position"]
                    rrt.ik_extend_randomly(np.array(pos), dist_thresh)
        return rrt

    def grow_rrt(self, side, q_start, goal_pose, dist_thresh=0.04, p_goal=0.5):
        obs = self.get_obs_for_side(side)
        if side == "left":
            self.left_rrt = RRT(q_start, goal_pose, self.get_kin(side), side, self.left_arm.joint_names(), obs,
                                self.check_and_execute_goal_angles)
            self.grow(self.left_rrt, dist_thresh, p_goal)
            return self.left_rrt
        else:
            self.right_rrt = RRT(q_start, goal_pose, self.get_kin(side), side, self.right_arm.joint_names(), obs,
                                 self.check_and_execute_goal_angles)
            self.grow(self.right_rrt, dist_thresh, p_goal)
            return self.right_rrt

    def get_goal(self, side):
        if side is "left":
            goal = self.left_goal
        else:
            goal = self.right_goal
        return goal

    def approach_goals(self):
        """
        Tries to approach the current goals.
        First, plans for left endpoint.
        Second, plans for right endpoint.
        Then, redoes planning over again.
        Loops forever, kill with ctrl-c.
        """
        while True:
            left_rrt = None
            if self.left_goal is not None:
                print "left goal: " + str(self.left_goal)
                left_joint_angles = [self.left_arm.joint_angle(name) for name in self.left_arm.joint_names()]
                left_rrt = self.grow_rrt("left", left_joint_angles, self.left_goal)
            if left_rrt:
                print "executed goal"
                # for node in left_rrt.nodes:
                #     print "approaching new node..."
                #     node_angle_dict = h.wrap_angles_in_dict(node, self.left_arm.joint_names())
                #     self.check_and_execute_goal_angles(node_angle_dict, "left")
                #     print "reached new node destination..."
            self.left_arm.set_joint_position_speed(0.0)

            # right_rrt = None
            # if self.right_goal is not None:
            #     print "right goal: " + str(self.right_goal)
            #     right_joint_angles = [self.right_arm.joint_angle(name) for name in self.right_arm.joint_names()]
            #     right_rrt = self.grow_rrt("right", right_joint_angles, self.right_goal)
            # if right_rrt:
            #     for node in right_rrt.nodes:
            #         node_angle_dict = h.wrap_angles_in_dict(node, self.right_arm.joint_names())
            #         self.check_and_execute_goal_angles(node_angle_dict, "right")
            self.right_arm.set_joint_position_speed(0.0)
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
        if self.left_goal_arr is not None:
            print "calculating left obstacle map"
            # do left goal distance mapping first
            for point in closest_points:
                left_goal_point = self.left_goal_arr[:3]
                indx = self.map_point_to_wavefront_index(point, left_goal_point)
                if len(left_obstacle_waves) > indx:
                    left_obstacle_waves[indx].append(point)
                else:
                    left_obstacle_waves.append([point])
        if self.right_goal_arr is not None:
            print "calculating right obstacle map"
            # do right goal distance mapping second
            for point in closest_points:
                right_goal_point = self.right_goal_arr[:3]
                indx = self.map_point_to_wavefront_index(point, right_goal_point)
                # if len(right_obstacle_waves) > indx:
                #     right_obstacle_waves[indx].append(point)
                # else:
                #     right_obstacle_waves.append([point])

        self.left_obstacles = left_obstacle_waves
        self.right_obstacles = right_obstacle_waves

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
        # goal = h.pose_to_ndarray(goal)

        # set right or left goal depending on marker name
        if "right" in feedback.marker_name:
            self.right_goal = goal
            self.right_goal_arr = h.pose_to_ndarray(goal)
            if self.right_rrt:
                self.right_rrt.update_goal(goal)
        elif "left" in feedback.marker_name:
            self.left_goal = goal
            self.left_goal_arr = h.pose_to_ndarray(goal)
            if self.left_rrt:
                self.left_rrt.update_goal(goal)
        else:
            rospy.loginfo("got singular end-point goal")

    def approach_single_goal(self, side, kin):
        """
        Attempts to successfully visit the goal point using the IK solver.
        :param side: the arm side to use
        :param kin: the KDLIKSolver instance
        :return: status of attempt to approach the goal point
        """
        status = OK

        goal = self.get_goal(side)

        # approach the goal points
        goal_met = False
        while goal_met is False and status is OK:
            goal = self.get_goal(side)
            obstacles = None
            # obstacles = h.get_critical_points_of_obstacles(self)
            # rospy.sleep(1)
            goal_angles = None
            if goal:
                # curr_angles = [self.left_arm.joint_angle(n) for n in self.left_arm.joint_names()]
                goal_angles = kin.solve(position=goal.position, orientation=goal.orientation)
                if goal_angles is not None:
                    goal_met = True
                else:
                    goal_met = False
            else:
                goal_met = True

            # execute goal angles if they are available
            if goal_angles is not None:
                angles_dict = h.wrap_angles_in_dict(goal_angles, self.left_arm.joint_names())
                # do left arm planning
                status = self.check_and_execute_goal_angles(angles_dict, side)
        print "found goal solution"
        return status

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
            print("stopping robot...")
            self.left_arm.set_joint_position_speed(0)
            self.right_arm.set_joint_position_speed(0)
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
