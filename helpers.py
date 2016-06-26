import argparse
import baxter_interface
import copy
import math
import numpy as np
import rospy

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Twist)

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

__author__ = 'shaun howard'

OK = "OK"
ERROR = "ERROR"


def determine_overall_status(l_status, r_status):
    if l_status is OK and r_status is OK:
       status = OK
    else:
       status = ERROR
    return status


def generate_goal_pose_w_same_orientation(dest_point, endpoint_pose):
    """Uses inverse kinematics to generate joint angles at destination point."""
    ik_pose = Pose()
    ik_pose.position.x = dest_point[0]
    ik_pose.position.y = dest_point[1]
    ik_pose.position.z = dest_point[2]
    current_pose = endpoint_pose
    if current_pose:
        ik_pose.orientation.x = current_pose['orientation'].x
        ik_pose.orientation.y = current_pose['orientation'].y
        ik_pose.orientation.z = current_pose['orientation'].z
        ik_pose.orientation.w = current_pose['orientation'].w
    return ik_pose


def get_args():
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=""""
                                     An interface to the CWRU robotics Merry robot for potential field motion planning.
                                     For help on usage, refer to the github README @ github.com/ShaunHoward/potential_\
                                     fields.""")
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '-l', '--limb', required=False, choices=['left', 'right'], default='right',
        help='limb to record/playback waypoints'
    )
    parser.add_argument(
        '-s', '--speed', default=0.3, type=float,
        help='joint position motion speed ratio [0.0-1.0] (default:= 0.3)'
    )
    parser.add_argument(
        '-a', '--accuracy',
        default=baxter_interface.JOINT_ANGLE_TOLERANCE, type=float,
        help='joint position accuracy (rad) at which waypoints must achieve'
    )
    return parser.parse_args(rospy.myargv()[1:])


def get_critical_points_of_obstacles(merry):
    """
    Considers the closest point the obstacle currently.
    In the future, will use obstacle detection to find the critical points of obstacles nearest to the robot.
    :return: a list of critical points paired with distances to end effector for each of the nearest obstacles
    """
    closest_point = None
    closest_dist = 15

    if merry.closest_points is not None and len(merry.closest_points) > 0:
        # copy points so they're not modified while in use
        curr_points = copy.deepcopy(merry.closest_points)
        normalized_points = scale(curr_points)
        if not merry.kmeans_initialized:
            predicition = merry.kmeans.fit(normalized_points)
            merry.kmeans_initialized = True
        else:
            prediction = merry.kmeans.predict(normalized_points)

        print(prediction)
        # run obstacle detection (K-means clustering) on the points closest to the robot
        # for p in closest_points:
        #     dist = math.sqrt(p.x ** 2 + p.y ** 2 + p.z ** 2)
        #     if closest_point is None or dist < closest_dist:
        #         closest_point = p
        #         closest_dist = dist
    return None if not (closest_point and closest_dist) else [(closest_point, closest_dist)]


def get_pose(x, y, z, ox=0, oy=0, oz=0, ow=1):
    pm = Pose()
    pm.position.x = x
    pm.position.y = y
    pm.position.z = z
    pm.orientation.x = ox
    pm.orientation.y = oy
    pm.orientation.z = oz
    pm.orientation.w = ow
    return pm


def get_current_endpoint_pose(arm):
    # retrieve current pose from endpoint
    current_pose = arm.endpoint_pose()
    pose_msg = Pose()
    pose_msg.position.x = current_pose['position'].x
    pose_msg.position.y = current_pose['position'].y
    pose_msg.position.z = current_pose['position'].z
    pose_msg.orientation.x = current_pose['orientation'].x
    pose_msg.orientation.y = current_pose['orientation'].y
    pose_msg.orientation.z = current_pose['orientation'].z
    pose_msg.orientation.w = current_pose['orientation'].w
    return pose_msg


def get_current_endpoint_velocities(arm):
    current_vels = arm._limb.endpoint_velocity()
    vel_msg = Twist()
    vel_msg.linear.x = current_vels['linear'].x
    vel_msg.linear.y = current_vels['linear'].y
    vel_msg.linear.z = current_vels['linear'].z
    vel_msg.angular.x = current_vels['angular'].x
    vel_msg.angular.y = current_vels['angular'].y
    vel_msg.angular.z = current_vels['angular'].z
    return vel_msg


def get_kmeans_instance(merry, num_clusts=10):
    merry.kmeans_initialized = False
    # seed numpy with the answer to the universe
    np.random.seed(42)
    kmeans = KMeans(init='k-means++', n_clusters=num_clusts, n_init=num_clusts)
    return kmeans


def lookup_transform(tf_, from_frame="kinect_link", to_frame="right_gripper"):
    """"
    :return: a set of points converted from_frame to_frame.
    """
    position = None
    quaternion = None
    # rosrun tf tf_echo right_gripper right_hand_camera
    if tf_.frameExists(to_frame) and tf_.frameExists(from_frame):
        t = tf_.getLatestCommonTime(to_frame, from_frame)
        position, quaternion = tf_.lookupTransform(to_frame, from_frame, t)
    return position, quaternion


def as_matrix2(tf_, target_frame, source_frame):
    return tf_.fromTranslationRotation(*lookup_transform(tf_, source_frame, target_frame))


def transform_pcl2(tf_, target_frame, source_frame, point_cloud, duration=2):
    # returns a list of transformed points
    tf_.waitForTransform(target_frame, source_frame, rospy.Time.now(), rospy.Duration(duration))
    mat44 = as_matrix2(tf_, target_frame, source_frame)
    if not isinstance(point_cloud[0], tuple) and not isinstance(point_cloud[0], list):
        point_cloud = [(p.x, p.y, p.z) for p in point_cloud]
    return [Point(*(tuple(np.dot(mat44, np.array([p[0], p[1], p[2], 1.0])))[:3])) for p in point_cloud]


def transform_points(tf, points, source="right_gripper", dest="base"):
    """
    Converts a given list of points into the specified dest. frame from the specified source frame.
    :return: points numpy array
    """
    transformed_points = transform_pcl2(tf, dest, source, points, 3)
    rospy.loginfo(''.join(["got a transformed cloud w/ mapping from ", source, " to ", dest, "!!"]))
    return transformed_points

# def move_to_start(self, start_angles=None):
#     print("Moving the {0} arm to start pose...".format(self._limb))
#     if not start_angles:
#         start_angles = dict(zip(self.JOINT_NAMES[:7], [0]*7))
#     self.set_joint_velocities(start_angles)
#     # self.gripper_open()
#     rospy.sleep(1.0)
#     print("Running. Ctrl-c to quit")

#
# def generate_goal_velocities(self, goal_point, obs, side="right"):
#     """
#     Generates the next move in a series of moves using an APF planning method
#     by means of the current robot joint states and the desired goal point.
#     :param goal_point: the goal 3-d point with x, y, z fields
#     :param obs: the obstacles for the planner to avoid
#     :param side: the arm side of the robot
#     :return: the status of the op and the goal velocities for the arm joints
#     """
#     return planner.plan(self.bkin, self.generate_goal_pose(goal_point), obs, self.get_current_endpoint_velocities(),
#                         self.get_joint_angles(side), side)
#
# def set_joint_velocities(self, joint_velocities):
#     """
#     Moves the Baxter robot end effector to the given dict of joint velocities keyed by joint name.
#     :param joint_velocities:  a list of join velocities containing at least those key-value pairs of joints
#      in JOINT_NAMES
#     :return: a status about the move execution; in success, "OK", and in error, "ERROR".
#     """
#     status = "OK"
#     rospy.sleep(1.0)
#
#     # Set joint position speed ratio for execution
#     self._limb.set_joint_position_speed(self._speed)
#
#     # execute next move
#     if not rospy.is_shutdown() and joint_velocities is not None and len(joint_velocities) == len(JOINT_NAMES):
#         vel_dict = dict()
#         curr_vel = 0
#         for name in JOINT_NAMES:
#             vel_dict[name] = joint_velocities[curr_vel]
#             curr_vel += 1
#         self._limb.set_joint_velocities(vel_dict)
#     else:
#         status = ERROR
#         rospy.logerr("Joint velocities are unavailable at the moment. \
#                       Will try to get goal velocities soon, but staying put for now.")
#
#     # Set joint position speed back to default
#     self._limb.set_joint_position_speed(0.3)
#     return status