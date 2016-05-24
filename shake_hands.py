__author__ = 'shaun howard'
import baxter_interface
import rospy

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

# amount of time to wait for IK solver service
TIMEOUT = 5.0

OK = "OK"
ERROR = "ERROR"


G_HOME = Pose(1, 2, 3, 4, 5, 6)


class Merry():

    curr_goal = None
    curr_pose = None
    latest_obstacles = []

    # initialize and enable the robot and the named limb
    def __init__(self, limb_name="left", dist_to_remain_above=.15, is_debug=True):
        self.limb = baxter_interface.Limb(limb_name)
        self.limb_name = limb_name
        ik_service = "".join(["ExternalTools/", limb_name, "/PositionKinematicsNode/IKService"])
        self.dist_to_remain_above = dist_to_remain_above
        self.is_debug = is_debug
        self.gripper = baxter_interface.Gripper(limb_name)
        self.ik_service = rospy.ServiceProxy(ik_service, SolvePositionIK)
        rospy.wait_for_service(ik_service, TIMEOUT)
        self.state = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        self.state.enable()
        self.curr_goal = G_HOME

    def curr_pose_callback(self):
        # get the arm side's current pose
        # use limb from class
        return Pose()

    def obstacles_callback(self):
        """
        :return: a list of obstacles and their locations as point_cloud vectors of points
        """
        return []

    def error_recovery(self, failed_move):
        """
        Tries to recover from APF planning local minima using the failed move as an example of what not to do.
        Applies a bout of random perturbation to the current arm pose to change planning ability.
        :return: status of recovery operation; "OK" on success, "ERROR" on error
        """
        status = ERROR
        return status

    def plan_and_execute(self, goal):
        """
        Moves to the 3-D goal coordinates that
        are given as a 3-tuple (x, y, z)
        :param goal: a dictionary of coords and other goal data
        :return: the status of the operation
        """
        status = OK
        self.curr_goal = goal
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

    def generate_next_move(self):
        """
        Generates the next move in a series of moves using an APF planning method.
        :return: the next pose to move to in a series of APF planning steps
        """
        # store ref to obstacles
        obs = self.latest_obstacles
        return self.curr_pose

    def execute_next_move(self, next_move):
        """
        Moves the Baxter robot to the location of the next move in a series of APF planning steps.
        :return: a status about the move execution; in success, "OK", and in error, "ERROR".
        """
        status = "OK"
        return status

    def get_current_goal(self):
        # retrieve current pose from endpoint
        current_pose = self.limb.endpoint_pose()
        ik_pose = Pose()
        ik_pose.position.x = current_pose['position'].x
        ik_pose.position.y = current_pose['position'].y
        ik_pose.position.z = current_pose['position'].z
        ik_pose.orientation.x = current_pose['orientation'].x
        ik_pose.orientation.y = current_pose['orientation'].y
        ik_pose.orientation.z = current_pose['orientation'].z
        ik_pose.orientation.w = current_pose['orientation'].w
        return ik_pose

    def reset(self):
        """"
        Moves Merry's arms into the home pose.
        """
        pass

if __name__ == '__main__':
    goal = None
    merry = Merry()
    while not rospy.is_shutdown():
        if goal:
            status = merry.plan_and_execute(goal)
            if status is OK:
                # reset goal, wait for next oen
                goal = None
            else:
                # reset robot, leave planning loop running
                merry.reset()
