import argparse
import baxter_interface
import helpers as h
import math
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
        self.ik_solver = IKSolver()

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

        #self.kinect_subscriber = rospy.Subscriber("kinect/depth/points", PointCloud2, self.kinect_cb)

        self.marker_subscriber = rospy.Subscriber("/merry_end_point_markers/feedback", InteractiveMarkerFeedback,
                                                  self.interactive_marker_cb)
        self.tf = tf.TransformListener()

        self.kmeans = None

        self.left_kinematics = baxter_kinematics("left")

        self.right_kinematics = baxter_kinematics("right")

        rospy.on_shutdown(self.clean_shutdown)

        self.left_goal = None

        self.right_goal = None

        # self.bridge = CvBridge()

    def approach(self, goal_point_pairs):
        """
        Attempts to successfully visit all of the provided points in fifo order. Will try next point upon success or failure.
        :param goal_point_pairs: 3d goal points in the frame of the chosen (left or right) gripper to move to with x, y, z
        :return: status of attempt to approach at least one of the specified points based on the success of the planning
        """
        status = OK
        if goal_point_pairs is None:
            goal_point_pairs = [(self.left_goal, self.right_goal)]

        # approach the goal points
        for left_goal, right_goal in goal_point_pairs:
            goal_met = False

            if status is ERROR:
                # exit if in error state
                break

            while goal_met is False and status is OK:
                obstacles = h.get_critical_points_of_obstacles(self.closest_points)
                rospy.sleep(1)
                left_goal_angles = None
                right_goal_angles = None
                # TODO multithread this!!
                if left_goal:
                    left_goal_angles = self.ik_solver.solve("left", left_goal)
                if right_goal:
                    right_goal_angles = self.ik_solver.solve("right", right_goal)

                if obstacles is not None:
                    # apply obstacle avoidance
                    pass

                l_status = OK
                r_status = OK
                # execute goal angles if they are available
                if left_goal_angles is not None:
                    # do left arm planning
                    l_status = self.check_and_execute_goal_angles(left_goal_angles, "left")

                if right_goal_angles is not None:
                    # do right arm planning
                    r_status = self.check_and_execute_goal_angles(right_goal_angles, "right")
                status = h.determine_overall_status(l_status, r_status)
                goal_met = left_goal_angles and right_goal_angles and status is OK
        return status

    def move_to_joint_positions(self, joint_positions, side):
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
                self.right_arm.set_joint_positions(joint_positions)
            elif side is "left":
                self.left_arm.set_joint_position_speed(self.left_speed)
                self.left_arm.set_joint_positions(joint_positions)
        else:
            rospy.logerr("Joint angles are unavailable at the moment. \
                          Will try to get goal angles soon, but staying put for now.")
            status = ERROR
        return status

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

    def kinect_cb(self, data, source="kinect_link", dest="right_gripper", max_dist=2, min_height=1):
        """
        Receives kinect points from the kinect subscriber linked to the publisher stream.
        :return: kinect points numpy array
        """
        points = [p for p in pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z"))]
        transformed_points = h.transform_pcl2(self.tf, dest, source, points, 3)
        print "got a transformed cloud!!"
        self.closest_points = [p for p in transformed_points if math.sqrt(p.x**2 + p.y**2 + p.z**2) < max_dist \
                               and p.z > min_height]
        print "found closest points"
        print "there are this many close points: " + str(len(self.closest_points))

    def interactive_marker_cb(self, feedback):
        # store feedback pose as goal
        goal = feedback.pose

        # set right or left goal depending on marker name
        if "right" in feedback.marker_name:
            self.right_goal = goal
        elif "left" in feedback.marker_name:
            self.left_goal = goal
        else:
            rospy.loginfo("got singular end-point goal")

    def solve_ik(self, side, goal_pos, kin_solver_instance):
        goal_angles = None
        if goal_pos is not None:
            # goal = self.generate_goal_pose(side, (goal_pos.x, goal_pos.y, goal_pos.z))
            # do inverse kinematics for cart pose to joint angles, then convert joint angles to joint velocities
            goal_angles = kin_solver_instance.inverse_kinematics((goal_pos.x,
                                                                  goal_pos.y,
                                                                  goal_pos.z))
        return goal_angles

    def check_and_execute_goal_angles(self, goal_angles, side):
        status = ERROR
        if goal_angles is not None:
            rospy.loginfo("got joint angles to execute!")
            rospy.loginfo("goal angles: " + str(goal_angles))
            joint_positions = goal_angles
            # joint_positions = self.get_angles_dict(goal_angles, side)
            print joint_positions
            # set the goal joint angles to reach the desired goal
            status = self.move_to_joint_positions(joint_positions, side)
            if status is ERROR:
                rospy.logerr("could not set joint positions for ik solver...")
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

    def error_recovery(self, failed_move, point):
        """
        Tries to recover from APF planning local minima using the failed move as an example of what not to do.
        Applies a bout of random perturbation to the current arm pose to change planning ability.
        :return: status of recovery operation; "OK" on success, "ERROR" on error
        """
        status = ERROR
        return status

    def clean_shutdown(self):
        print("\nExiting example...")
        if not self._init_state:
            print("Disabling robot...")
            self._rs.disable()
        return True

    def generate_approach_path(self):
        return None

    def shake_hands(self):
        """
        Tries to shake the nearest hand possible using the limb instantiated.
        Loop runs forever; kill with ctrl-c.
        """
        path = self.generate_approach_path()
        if self.approach(path) is "OK":
            return 0
        return 1


def run_robot(args):
    print("Initializing node... ")
    rospy.init_node("merry")
    merry = Merry(args.limb, args.speed, args.accuracy)
    rospy.on_shutdown(merry.clean_shutdown)
    status = merry.shake_hands()
    sys.exit(status)


if __name__ == '__main__':
    run_robot(h.get_args())
