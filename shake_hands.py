__author__ = 'shaun howard'
import baxter_interface
import rospy

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

# amount of time to wait for IK solver service
TIMEOUT = 5.0


class Merry():

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

    def go_to_home(self, ):