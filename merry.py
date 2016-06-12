__author__ = 'shaun howard'
import baxter_interface
import cv2
import math
import numpy as np
import rospy
import std_msgs.msg
import struct
import tf
import planner

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs import JointState
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud
from sensor_msgs.msg import PointCloud2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    PointStamped,
    Quaternion,
    TransformStamped
)

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

from std_msgs.msg import (
    Header,
    Empty,
)

# amount of time to wait for IK solver service
TIMEOUT = 5.0

OK = "OK"
ERROR = "ERROR"


# G_HOME = Pose(1, 2, 3, 4, 5, 6)

def start_pcl():
    p = pcl.PointCloud()
    p.from_file("table_scene_lms400.pcd")
    fil = p.make_statistical_outlier_filter()
    fil.set_mean_k (50)
    fil.set_std_dev_mul_thresh (1.0)
    fil.filter().to_file("inliers.pcd")


# def extract_closest_points():
#     global points
#     for x, y, z in points:
#

def shake_hands(merry):
    """
    Tries to shake the nearest hand possible using the limb instantiated.
    Loop runs forever; kill with ctrl-c.
    """
    if merry.plan_and_execute() is "OK":
        return 0
    return 1


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


def transform_pcl2(tf_, target_frame, source_frame, point_cloud, duration=3):
    # returns a list of transformed points
    tf_.waitForTransform(target_frame, source_frame, rospy.Time.now(), rospy.Duration(duration))
    mat44 = as_matrix2(tf_, target_frame, source_frame)
    return [Point(*(tuple(np.dot(mat44, np.array([p[0], p[1], p[2], 1.0])))[:3])) for p in point_cloud]


def ik_request(self, pose):
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    ikreq = SolvePositionIKRequest()
    ikreq.pose_stamp.append(PoseStamped(header=hdr, pose=pose))
    try:
        resp = self._iksvc(ikreq)
    except (rospy.ServiceException, rospy.ROSException), e:
        rospy.logerr("Service call failed: %s" % (e,))
        return False
    # Check if result valid, and type of seed ultimately used to get solution
    # convert rospy's string representation of uint8[]'s to int's
    resp_seeds = struct.unpack('<%dB' % len(resp.result_type), resp.result_type)
    if resp_seeds[0] != resp.RESULT_INVALID:
        seed_str = {
            ikreq.SEED_USER: 'User Provided Seed',
            ikreq.SEED_CURRENT: 'Current Joint Angles',
            ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
        }.get(resp_seeds[0], 'None')
        if self._verbose:
            print("IK Solution SUCCESS - Valid Joint Solution Found from Seed Type: {0}".format(
                seed_str))
        # Format solution into Limb API-compatible dictionary
        limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
        if self._verbose:
            print("IK Joint Solution:\n{0}".format(limb_joints))
            print("------------------")
    else:
        rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
        return False
    return limb_joints


JOINT_NAMES = ["e0", "e1", "s0", "s1", "w0", "w1", "w2"]


class Merry(object):

    closest_points = []
    joint_states = []

    # initialize and enable the robot and the named limb
    def __init__(self, limb="right", speed=0.3, accuracy=baxter_interface.JOINT_ANGLE_TOLERANCE):

        rospy.init_node("merry")

        rospy.sleep(1)

        # Create baxter_interface limb instance
        self._arm = limb
        self._limb = baxter_interface.Limb(self._arm)
        self._joint_names = []

        # Parameters which will describe joint position moves
        self._speed = speed
        self._accuracy = accuracy

        ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        self._iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        rospy.wait_for_service(ns, 5.0)

        # Verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable()
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()

        # Create Navigator I/O
        self._navigator_io = baxter_interface.Navigator(self._arm)

        # self.gripper = baxter_interface.Gripper(limb_name)
        self.joint_states_subscriber = rospy.Subscriber("robot/joint_states", JointState, self.joint_state_cb)

        self.kinect_subscriber = rospy.Subscriber("kinect/depth/points", PointCloud2, self.kinect_cb)
        self.tf = tf.TransformListener()

        # self.bridge = CvBridge()

    def joint_state_cb(self, data):
        names = data.name
        positions = data.position
        velocities = data.velocity
        efforts = data.effort
        self.joint_states = dict()
        for i in range(len(names)):
            self.joint_states[names[i]] = dict()
            self.joint_states[names[i]]["position"] = positions[i]
            self.joint_states[names[i]]["velocity"] = velocities[i]
            self.joint_states[names[i]]["effort"] = efforts[i]

    def kinect_cb(self, data, source="kinect_link", dest="right_gripper", max_dist=2):
        """
        Receives kinect points from the kinect subscriber linked to the publisher stream.
        :return: kinect points numpy array
        """
        points = [p for p in pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z"))]
        transformed_points = transform_pcl2(self.tf, dest, source, points, 3)
        print "got a transformed cloud!!"
        self.closest_points = [p for p in transformed_points if math.sqrt(p.x**2 + p.y**2 + p.z**2) < max_dist]
        print "found closest points"
        print "there are this many close points: " + str(len(self.closest_points))

    def get_joint_angles(self, side="right"):
        """Gets the joint angle dictionary of the specified arm side."""
        joint_angles = dict()
        names = self.joint_states.keys()
        for name in names:
            if side in name:
                if name not in self._joint_names:
                    self._joint_names.append(name)
                joint_angles[name] = (self.joint_states[name]["position"], self.joint_states[name]["velocity"],
                                      self.joint_states[name]["effort"])
        return joint_angles

    def get_current_endpoint_pose(self):
        # retrieve current pose from endpoint
        current_pose = self._limb.endpoint_pose()
        ik_pose = Pose()
        ik_pose.position.x = current_pose['position'].x
        ik_pose.position.y = current_pose['position'].y
        ik_pose.position.z = current_pose['position'].z
        ik_pose.orientation.x = current_pose['orientation'].x
        ik_pose.orientation.y = current_pose['orientation'].y
        ik_pose.orientation.z = current_pose['orientation'].z
        ik_pose.orientation.w = current_pose['orientation'].w
        return ik_pose

    def generate_goal_angles(self, dest_point):
        """Uses inverse kinematics to generate joint angles at destination point."""
        ik_pose = Pose()
        ik_pose.position.x = dest_point[0]
        ik_pose.position.y = dest_point[1]
        ik_pose.position.z = dest_point[2]
        current_pose = self._limb.endpoint_pose()
        ik_pose.orientation.x = current_pose['orientation'].x
        ik_pose.orientation.y = current_pose['orientation'].y
        ik_pose.orientation.z = current_pose['orientation'].z
        ik_pose.orientation.w = current_pose['orientation'].w
        joint_angles = ik_request(ik_pose)
        return joint_angles

    def approach(self, points):
        """
        Attempts to successfully visit one of the provided points in fifo order. Will try another point if one fails.
        :param points: possible 3d points in the frame of the chosen (left or right) gripper to move to
        :return: status of attempt to approach at least one of the specified points based on the success of the planning
        """
        status = ERROR
        # approach the closest point then backups if needed
        for p in points:
            status = self.plan_and_execute(p)
            if status is OK:
                break
        return status

    def get_critical_points_of_obstacles(self):
        """
        Considers the closest point the obstacle currently.
        In the future, will use obstacle detection to find the critical points of obstacles nearest to the robot.
        :return: a list of critical points paired with distances to end effector for each of the nearest obstacles
        """
        closest_point = None
        closest_dist = 15
        # run obstacle detection on the points closest to the robot
        for p in self.closest_points:
            dist = math.sqrt(p.x**2 + p.y**2 + p.z**2)
            if closest_point is not None and dist < closest_dist:
                closest_dist = dist
        return [(closest_point, closest_dist)]

    def plan_and_execute(self, point, side="right"):
        """
        Moves to the 3-D goal coordinates that
        are given as a 3-tuple (x, y, z)
        :return: the status of the operation
        """
        # get obstacles
        obstacles = self.get_critical_points_of_obstacles()
        # generate the next goal
        next_joint_angles = self.generate_next_joint_angles(obstacles, side)
        # execute next step in movement
        status = self.move_to_next_angles(next_joint_angles)
        # check if the status is successful
        if status is not OK:
            # try to revisit move with perturbation
            status = self.error_recovery(next_joint_angles, point)
        return status

    def generate_next_joint_angles(self, obs, side="right"):
        """
        Generates the next move in a series of moves using an APF planning method
        by means of the current robot joint states and the desired goal point.
        :param obs: the obstacles for the planner to avoid
        :param side: the arm side of the robot
        :return: the next pose to move to in a series of APF planning steps
        """
        return planner.plan(obs, self.get_joint_angles(side), side)

    def move_to_next_angles(self, next_move):
        """
        Moves the Baxter robot to the location of the next move in a series of APF planning steps.
        :return: a status about the move execution; in success, "OK", and in error, "ERROR".
        """
        status = "OK"
        rospy.sleep(1.0)

        # Set joint position speed ratio for execution
        self._limb.set_joint_position_speed(self._speed)

        # execute next move
        if not rospy.is_shutdown() and next_move is not None:
            self._limb.move_to_joint_positions(next_move, timeout=20.0,
                                               threshold=self._accuracy)
            # Sleep for a few seconds between playback loops
            rospy.sleep(3.0)
        else:
            rospy.logerr("No joint angles provided for move to next joint angles. Not moving.")
            status = ERROR

        # Set joint position speed back to default
        self._limb.set_joint_position_speed(0.3)
        return status

    def move_to_start(self, start_angles=None):
        print("Moving the {0} arm to start pose...".format(self._limb))
        if not start_angles:
            start_angles = dict(zip(self._joint_names, [0]*7))
        self.move_to_next_angles(start_angles)
        # self.gripper_open()
        rospy.sleep(1.0)
        print("Running. Ctrl-c to quit")

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

    def curr_pose_callback(self):
        # get the arm side's current pose
        # use limb from class
        return Pose()

    def obstacles_callback(self):
        """
        :return: a list of obstacles and their locations as point_cloud vectors of points
        """
        return []

    def reset(self):
        """"
        Moves Merry's arms into the home pose.
        """
        pass


if __name__ == '__main__':
    goal = None
    merry = Merry()
    while not rospy.is_shutdown():
        #print "got in main loop"
        pass
        # if goal:
        #     status = merry.plan_and_execute(goal)
        #     if status is OK:
        #         # reset goal, wait for next oen
        #         goal = None
        #     else:
        #         # reset robot, leave planning loop running
        #         merry.reset()
