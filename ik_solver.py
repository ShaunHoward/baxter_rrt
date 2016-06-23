import argparse
import sys

import rospy

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)


class IKSolver:

    def __init__(self):
        # rospy.init_node("rsdk_ik_service_client")

        self.r_ns = "ExternalTools/right/PositionKinematicsNode/IKService"
        self.r_iksvc = rospy.ServiceProxy(self.r_ns, SolvePositionIK)

        self.l_ns = "ExternalTools/left/PositionKinematicsNode/IKService"
        self.l_iksvc = rospy.ServiceProxy(self.l_ns, SolvePositionIK)

    def solve(self, limb, pos, orient):
        ikreq = SolvePositionIKRequest()
        print "ikreq: ", ikreq
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        pose = PoseStamped(header=hdr, pose=Pose(position=pos, orientation=orient))
        ikreq.pose_stamp.append(pose)
        if limb is "left":
            ns = self.l_ns
            iksvc = self.l_iksvc
        else:
            ns = self.r_ns
            iksvc = self.r_iksvc
        try:
            rospy.wait_for_service(ns, 5.0)
            resp = iksvc(ikreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            return 1
        if resp.isValid[0]:
            print("SUCCESS - Valid Joint Solution Found:")
            # Format solution into Limb API-compatible dictionary
            limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
            print limb_joints
            return limb_joints
        else:
            print("INVALID POSE - No Valid Joint Solution Found.")

        return None
