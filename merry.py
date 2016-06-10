__author__ = 'shaun howard'
import baxter_interface
import cv2
import math
import numpy as np
import rospy
import std_msgs.msg
import tf

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs import point_cloud2 as pc2
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


class Merry(object):

    curr_goal = None
    curr_pose = None
    latest_obstacles = []
    tf = None
    planner = None
    tf_broadcaster = None
    closest_points = []

    # initialize and enable the robot and the named limb
    def __init__(self, limb="right", speed=0.3, accuracy=baxter_interface.JOINT_ANGLE_TOLERANCE):

        rospy.init_node("merry")
        # store reference to planner
        #self.planner = planner

        rospy.sleep(1)

        # Create baxter_interface limb instance
        self._arm = limb
        self._limb = baxter_interface.Limb(self._arm)

        # Parameters which will describe joint position moves
        self._speed = speed
        self._accuracy = accuracy

        # Recorded waypoints
        self._waypoints = list()

        # Recording state
        self._is_recording = False

        # Verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable()
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()

        # Create Navigator I/O
        self._navigator_io = baxter_interface.Navigator(self._arm)

        #self.gripper = baxter_interface.Gripper(limb_name)

        #self.curr_goal = G_HOME
        self.kinect_subscriber = rospy.Subscriber("kinect/depth/points", PointCloud2, self.kinect_cb)
        self.tf = tf.TransformListener()

        self.bridge = CvBridge()
        self.tf_broadcaster = tf.TransformBroadcaster()

    def error_recovery(self, failed_move):
        """
        Tries to recover from APF planning local minima using the failed move as an example of what not to do.
        Applies a bout of random perturbation to the current arm pose to change planning ability.
        :return: status of recovery operation; "OK" on success, "ERROR" on error
        """
        status = ERROR
        return status

    def plan_and_execute(self):
        """
        Moves to the 3-D goal coordinates that
        are given as a 3-tuple (x, y, z)
        :param goal: a dictionary of coords and other goal data
        :return: the status of the operation
        """
        status = OK
        while status is OK:
            # generate the next goal
            next_move = self.generate_next_move()
            # execute next step in movement
            status = self.execute_next_move(next_move)
            # check if the status is successful
            if status is not OK:
                # try to revisit move with perturbation
                status = self.error_recovery(next_move)
        return status

    def find_closest_goal(self):
        obs = self.latest_obstacles
        closest_dist_to_bot = 1000000
        closest_ob = None
        for ob in obs:
            pass
        return closest_ob

    def generate_next_move(self):
        """
        Generates the next move in a series of moves using an APF planning method.
        :return: the next pose to move to in a series of APF planning steps
        """
        # store ref to obstacles
        obs = self.latest_obstacles
        closest_obstacle = self.find_closest_goal()
        self.curr_goal = closest_obstacle
        return self.curr_goal

    def execute_next_move(self, next_move):
        """
        Moves the Baxter robot to the location of the next move in a series of APF planning steps.
        :return: a status about the move execution; in success, "OK", and in error, "ERROR".
        """
        status = "OK"
        rospy.sleep(1.0)

        # Set joint position speed ratio for execution
        self._limb.set_joint_position_speed(self._speed)

        # execute next move
        if not rospy.is_shutdown():
            self._limb.move_to_joint_positions(next_move, timeout=20.0,
                                               threshold=self._accuracy)
            # Sleep for a few seconds between playback loops
            rospy.sleep(3.0)
        else:
            status = ERROR

        # Set joint position speed back to default
        self._limb.set_joint_position_speed(0.3)
        return status

    def get_current_goal(self):
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
