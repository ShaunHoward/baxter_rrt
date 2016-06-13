__author__ = 'shaun howard'
import baxter_interface
import cv2
import math
import numpy as np
import rospy
import struct
import tf
import planner

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import Image, JointState, PointCloud, PointCloud2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    PointStamped,
    Quaternion,
    TransformStamped,
    Twist
)

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

from std_msgs.msg import (
    Header,
    Empty,
)


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

# amount of time to wait for IK solver service
TIMEOUT = 5.0

OK = "OK"
ERROR = "ERROR"

# may need to change this order to make joint names correspond to joints 1-7 in the solver
JOINT_NAMES = ["s0", "s1", "e0", "e1", "w0", "w1", "w2"]
NUM_JOINTS = len(JOINT_NAMES)
MAX_TRIALS = 10

class Merry(object):

    closest_points = []

    # initialize and enable the robot and the named limb
    def __init__(self, limb="right", speed=0.3, accuracy=baxter_interface.JOINT_ANGLE_TOLERANCE):

        rospy.init_node("merry")

        rospy.sleep(1)

        # Create baxter_interface limb instance
        self._arm = limb
        self._limb = baxter_interface.Limb(self._arm)
        self._joint_names = []

        self.joint_states = {name: dict() for name in JOINT_NAMES}

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
        for i in range(len(JOINT_NAMES)):
            print positions[i]
            print velocities[i]
            print efforts[i]
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
        joint_angles = []
        for name in JOINT_NAMES:
            if side in name:
                if name not in self._joint_names:
                    self._joint_names.append(name)
                joint_angles.append(self.joint_states[name]["position"])
        return joint_angles

    def get_joint_velocities(self, side="right"):
        """Gets the joint angle dictionary of the specified arm side."""
        joint_velocities = []
        for name in JOINT_NAMES:
            print (self.joint_states)
            if side in name:
                if name not in self._joint_names:
                    self._joint_names.append(name)
                joint_velocities.append(self.joint_states[name]["velocity"])
        if len(joint_velocities) != NUM_JOINTS:
            joint_velocities = [0] * 7
        return np.array(joint_velocities)

    def get_current_endpoint_pose(self):
        # retrieve current pose from endpoint
        current_pose = self._limb.endpoint_pose()
        pose_msg = Pose()
        pose_msg.position.x = current_pose['position'].x
        pose_msg.position.y = current_pose['position'].y
        pose_msg.position.z = current_pose['position'].z
        pose_msg.orientation.x = current_pose['orientation'].x
        pose_msg.orientation.y = current_pose['orientation'].y
        pose_msg.orientation.z = current_pose['orientation'].z
        pose_msg.orientation.w = current_pose['orientation'].w
        return pose_msg

    def get_current_endpoint_velocities(self):
        current_vels = self._limb.endpoint_velocity()
        vel_msg = Twist()
        vel_msg.linear.x = current_vels['linear'].x
        vel_msg.linear.y = current_vels['linear'].y
        vel_msg.linear.z = current_vels['linear'].z
        vel_msg.angular.x = current_vels['angular'].x
        vel_msg.angular.y = current_vels['angular'].y
        vel_msg.angular.z = current_vels['angular'].z
        return vel_msg

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
            status = self.plan_and_execute_end_effector(p)
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
            if closest_point is None or dist < closest_dist:
                closest_point = p
                closest_dist = dist
        return [(closest_point, closest_dist)]

    def plan_and_execute_end_effector(self, side="right"):
        """
        Moves the side end effector to the closest obstacle.
        :return: the status of the operation
        """
        trial = 0
        status = ERROR
        while True:
            # get obstacles
            obstacles = self.get_critical_points_of_obstacles()
            # generate the goal joint velocities
            goal_velocities = self.generate_goal_velocities(obstacles, side)
            # set the goal joint velocities to reach the desired goal
            status = self.set_joint_velocities(goal_velocities)
            # check if the status is successful
            if status is not OK:
                # try to revisit move with perturbation
                rospy.logerr("unable to compute goal joint velocities...")
                # status = self.error_recovery(next_joint_angles, point)
                trial += 1
                if trial >= MAX_TRIALS:
                    rospy.loginfo("cannot find solution after max trials met...exiting...")
                    break
        return status

    def generate_goal_velocities(self, obs, side="right"):
        """
        Generates the next move in a series of moves using an APF planning method
        by means of the current robot joint states and the desired goal point.
        :param obs: the obstacles for the planner to avoid
        :param side: the arm side of the robot
        :return: the next pose to move to in a series of APF planning steps
        """
        return planner.plan(obs, self.get_current_endpoint_velocities(), self.get_joint_angles(side), side)

    def set_joint_velocities(self, joint_velocities):
        """
        Moves the Baxter robot end effector to the given dict of joint velocities keyed by joint name.
        :return: a status about the move execution; in success, "OK", and in error, "ERROR".
        """
        status = "OK"
        rospy.sleep(1.0)

        # Set joint position speed ratio for execution
        self._limb.set_joint_position_speed(self._speed)

        # execute next move
        if not rospy.is_shutdown() and joint_velocities is not None and len(joint_velocities) == len(JOINT_NAMES):
            vel_dict = dict()
            curr_vel = 0
            for name in JOINT_NAMES:
                vel_dict[name] = joint_velocities[curr_vel]
                curr_vel += 1
            self._limb.set_joint_velocities(vel_dict)
        else:
            status = ERROR
            rospy.logerr("Joint velocities are unavailable at the moment. \
                          Will try to get goal velocities soon, but staying put for now.")

        # Set joint position speed back to default
        self._limb.set_joint_position_speed(0.3)
        return status

    def set_joint_positions(self, joint_positions):
        """
        Moves the Baxter robot end effector to the given dict of joint velocities keyed by joint name.
        :return: a status about the move execution; in success, "OK", and in error, "ERROR".
        """
        status = "OK"
        rospy.sleep(1.0)

        # Set joint position speed ratio for execution
        self._limb.set_joint_position_speed(self._speed)

        # execute next move
        if not rospy.is_shutdown() and joint_positions is not None:
            self._limb.set_joint_positions(joint_positions)
        else:
            rospy.logerr("Joint angles are unavailable at the moment. \
                          Will try to get goal angles soon, but staying put for now.")
            status = ERROR

        # Set joint position speed back to default
        self._limb.set_joint_position_speed(0.3)
        return status

    def move_to_start(self, start_angles=None):
        print("Moving the {0} arm to start pose...".format(self._limb))
        if not start_angles:
            start_angles = dict(zip(self._joint_names, [0]*7))
        self.set_joint_velocities(start_angles)
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
