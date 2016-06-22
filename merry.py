import baxter_interface
import cv2
import math
import numpy as np
import rospy
import struct
import tf
import planner
import random

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import Image, JointState, PointCloud, PointCloud2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from visualization_msgs.msg import InteractiveMarker


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

from baxter_pykdl import baxter_kinematics

from std_msgs.msg import (
    Header,
    Empty,
)


__author__ = 'shaun howard'


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
    if not isinstance(point_cloud[0], tuple) and not isinstance(point_cloud[0], list):
        point_cloud = [(p.x, p.y, p.z) for p in point_cloud]
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
JOINT_NAMES = ["right_s0", "right_s1", "right_e0", "right_e1", "right_w0", "right_w1", "right_w2",
               "left_s0", "left_s1", "left_e0", "left_e1", "left_w0", "left_w1", "left_w2"]
NUM_JOINTS = len(JOINT_NAMES)
MAX_TRIALS = 10


def get_angles_dict(angles_list, side):
    i = 0
    angles_dict = dict()
    for name in JOINT_NAMES:
        if side in name:
            angles_dict[name] = angles_list[i]
            i += 1
    return angles_dict


class Merry(object):

    closest_points = []

    # initialize and enable the robot and the named limb
    def __init__(self, left_speed=0.3, right_speed=0.3, accuracy=baxter_interface.JOINT_ANGLE_TOLERANCE):

        rospy.init_node("merry")

        rospy.sleep(1)

        # Create baxter_interface limb instances for right and left arms
        self.right_arm = baxter_interface.Limb("right")
        self.left_arm = baxter_interface.Limb("left")
        self.joint_states = dict()

        # Parameters which will describe joint position moves
        self.right_speed = right_speed
        self.left_speed = left_speed
        self._accuracy = accuracy

        # ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        # self._iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        # rospy.wait_for_service(ns, 5.0)

        # Verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable()
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()

        # Create Navigator I/O
        # self._navigator_io = baxter_interface.Navigator(self._arm)

        # self.gripper = baxter_interface.Gripper(limb_name)
        self.joint_states_subscriber = rospy.Subscriber("robot/joint_states", JointState, self.joint_state_cb)

        self.kinect_subscriber = rospy.Subscriber("kinect/depth/points", PointCloud2, self.kinect_cb)

        self.marker_subscriber = rospy.Subscriber("/merry_end_point_markers/feedback", InteractiveMarker,
                                                  self.interactive_marker_cb)
        self.tf = tf.TransformListener()

        self.kmeans = None

        self.left_kinematics = baxter_kinematics("left")

        self.right_kinematics = baxter_kinematics("right")

        rospy.on_shutdown(self.clean_shutdown)

        self.left_goal = None

        self.right_goal = None

        # self.bridge = CvBridge()

    def get_kmeans_instance(self):
        # seed numpy with the answer to the universe
        np.random.seed(42)
        while len(self.closest_points) == 0:
            # busy wait until kinect update is received
            pass
        # copy points so they're not modified while in use
        curr_points = [x for x in self.closest_points]
        data = scale(curr_points)
        reduced_data = PCA(n_components=1).fit_transform(data)
        kmeans = KMeans(init='k-means++', n_clusters=len(data), n_init=10)
        kmeans.fit(reduced_data)
        return kmeans

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
        transformed_points = transform_pcl2(self.tf, dest, source, points, 3)
        print "got a transformed cloud!!"
        self.closest_points = [p for p in transformed_points if math.sqrt(p.x**2 + p.y**2 + p.z**2) < max_dist \
                               and p.z > min_height]
        print "found closest points"
        print "there are this many close points: " + str(len(self.closest_points))

    def interactive_marker_cb(self, feedback):
        print "marker callback entered!"
        goal = Point()
        goal.x = feedback.pose.position.x
        goal.y = feedback.pose.position.y
        goal.z = feedback.pose.position.z
        if "right" in feedback.marker_name:
            self.right_goal = goal
        elif "left" in feedback.marker_name:
            self.left_goal = goal
        else:
            rospy.loginfo("got singular end-point goal")
        print feedback.marker_name + " is now at " + str(goal.x) + ", " + str(goal.y) + ", " + str(goal.z)

    def get_joint_angles(self, side="right"):
        """Gets the joint angle dictionary of the specified arm side."""
        joint_angles = []
        for name in JOINT_NAMES:
            if side in name:
                joint_angles.append(self.joint_states[name]["position"])
        return joint_angles

    def get_joint_velocities(self, side="right"):
        """Gets the joint angle dictionary of the specified arm side."""
        joint_velocities = []
        for name in JOINT_NAMES:
            if side in name:
                joint_velocities.append(self.joint_states[name]["velocity"])
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

    def generate_goal_pose(self, dest_point):
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
        return ik_pose

    def transform_points(self, points, source="right_gripper", dest="base"):
            """
            Converts a given list of points into the specified dest. frame from the specified source frame.
            :return: points numpy array
            """
            transformed_points = transform_pcl2(self.tf, dest, source, points, 3)
            rospy.loginfo(''.join(["got a transformed cloud w/ mapping from ", source, " to ", dest, "!!"]))
            return transformed_points

    def get_critical_points_of_obstacles(self):
        """
        Considers the closest point the obstacle currently.
        In the future, will use obstacle detection to find the critical points of obstacles nearest to the robot.
        :return: a list of critical points paired with distances to end effector for each of the nearest obstacles
        """
        closest_point = None
        closest_dist = 15
        # TODO: add/test kmeans!!
        #        prediction = self.kmeans.predict(self.closest_points)
        #        print(prediction)
        # run obstacle detection (K-means clustering) on the points closest to the robot
        for p in self.closest_points:
            dist = math.sqrt(p.x ** 2 + p.y ** 2 + p.z ** 2)
            if closest_point is None or dist < closest_dist:
                closest_point = p
                closest_dist = dist
        return None if not (closest_point and closest_dist) else [(closest_point, closest_dist)]

    def solve_ik(self, goal_pos, kin_solver_instance):
        goal_angles = None
        if goal_pos is not None:
            goal = self.generate_goal_pose((goal_pos.x, goal_pos.y, goal_pos.z))
            # do inverse kinematics for cart pose to joint angles, then convert joint angles to joint velocities
            goal_angles = kin_solver_instance.inverse_kinematics((goal.position.x,
                                                                  goal.position.y,
                                                                  goal.position.z),
                                                                 (goal.orientation.x,
                                                                  goal.orientation.y,
                                                                  goal.orientation.z,
                                                                  goal.orientation.w))
        return goal_angles

    def check_and_execute_goal_angles(self, goal_angles, side):
        status = ERROR
        if goal_angles is not None:
            rospy.loginfo("got joint angles to execute!")
            rospy.loginfo("goal angles: " + str(goal_angles))
            joint_positions = get_angles_dict(goal_angles, side)
            print joint_positions
            # set the goal joint angles to reach the desired goal
            status = self.move_to_joint_positions(joint_positions)
            if status is ERROR:
                rospy.logerr("could not set joint positions for ik solver...")
        return status

    def approach(self, goal_points=[]):
        """
        Attempts to successfully visit all of the provided points in fifo order. Will try next point upon success or failure.
        :param goal_points: 3d goal points in the frame of the chosen (left or right) gripper to move to with x, y, z
        :return: status of attempt to approach at least one of the specified points based on the success of the planning
        """
        status = ERROR
        # approach the goal points
        # for goal in goal_points:
        goal_angles = None
        mode = "IK"
        while True:
            obstacles = self.get_critical_points_of_obstacles()
            if mode is "IK":
                rospy.sleep(1)
                left_goal_angles = self.solve_ik(self.left_goal, self.left_kinematics)
                right_goal_angles = self.solve_ik(self.right_goal, self.right_kinematics)

                if obstacles is not None:
                    # apply obstacle avoidance
                    pass

                # transform from the current gripper to the base frame for solving
                # ik_trans = self.transform_points(points, self._arm + "_gripper", "base")
                # execute goal angles if they are available
                if left_goal_angles is not None:
                    # do left arm planning
                    status = self.check_and_execute_goal_angles(left_goal_angles, "left")
                if right_goal_angles is not None and status is OK:
                    # do right arm planning
                    status = self.check_and_execute_goal_angles(right_goal_angles, "right")
                # if status is ERROR:
                #     rospy.logerr('could not find goal joint angles for at least one arm...')
                # else:
                #     # leave planning loop if solution was found
                #     # break
                #     pass
            else:
                #rospy.logerr("didn't get joint angles soln...")
                status = ERROR
                # if status == OK:
                # status = self.plan_and_execute_end_effector(goal)
            # elif mode is "IKAvoid":
            #     rospy.loginfo("this method of solving is not yet implemented...")
            #     pass
            # else:
            #     status = ERROR
            #     rospy.logerr("mode for operation not set.. aborting.")
            #     break
        return status

    def plan_and_execute_end_effector(self, goal, side="right"):
        """
        Moves the side end effector to the provided goal position.
        :param goal: the goal 3-d point with x, y, z fields
        :param side: the side of the robot to plan for, left or right
        :return: the status of the operation
        """
        trial = 0
        status = ERROR
        while True:
            # get obstacles
            obstacles = self.get_critical_points_of_obstacles()
            # generate the goal joint velocities
            status, goal_velocities = self.generate_goal_velocities(goal, obstacles, side)

            if status == OK:
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
            rospy.sleep(2)
        return status

    def generate_goal_velocities(self, goal_point, obs, side="right"):
        """
        Generates the next move in a series of moves using an APF planning method
        by means of the current robot joint states and the desired goal point.
        :param goal_point: the goal 3-d point with x, y, z fields
        :param obs: the obstacles for the planner to avoid
        :param side: the arm side of the robot
        :return: the status of the op and the goal velocities for the arm joints
        """
        return planner.plan(self.bkin, self.generate_goal_pose(goal_point), obs, self.get_current_endpoint_velocities(),
                            self.get_joint_angles(side), side)

    def set_joint_velocities(self, joint_velocities):
        """
        Moves the Baxter robot end effector to the given dict of joint velocities keyed by joint name.
        :param joint_velocities:  a list of join velocities containing at least those key-value pairs of joints
         in JOINT_NAMES
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
                self.right_arm.move_to_joint_positions(joint_positions)
            elif side is "left":
                self.left_arm.set_joint_position_speed(self.left_speed)
                self.left_arm.move_to_joint_positions(joint_positions)
        else:
            rospy.logerr("Joint angles are unavailable at the moment. \
                          Will try to get goal angles soon, but staying put for now.")
            status = ERROR
        return status

    # def move_to_start(self, start_angles=None):
    #     print("Moving the {0} arm to start pose...".format(self._limb))
    #     if not start_angles:
    #         start_angles = dict(zip(self.JOINT_NAMES[:7], [0]*7))
    #     self.set_joint_velocities(start_angles)
    #     # self.gripper_open()
    #     rospy.sleep(1.0)
    #     print("Running. Ctrl-c to quit")

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
