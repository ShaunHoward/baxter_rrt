import struct

import rospy
from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)
from geometry_msgs.msg import (
    Pose,
    PoseStamped
)
from std_msgs.msg import Header

from baxter_kdl.kdl_kinematics import KDLKinematics as kin
from urdf_parser_py.urdf import URDF


def dict_to_pose(q_dict):
    pose = Pose()
    keys = q_dict.keys()
    pose.position.x = q_dict[keys[0]]
    pose.position.y = q_dict[keys[1]]
    pose.position.z = q_dict[keys[2]]
    pose.orientation.x = q_dict[keys[3]]
    pose.orientation.x = q_dict[keys[4]]
    pose.orientation.x = q_dict[keys[5]]
    pose.orientation.x = q_dict[keys[6]]
    return pose


class KDLIKSolver:
    def __init__(self, limb):
        self._baxter = URDF.from_parameter_server(key='robot_description')
        self._base_link = self._baxter.get_root()
        self._tip_link = limb + '_gripper'
        self.solver = kin(self._baxter, self._base_link, self._tip_link)

    def solve(self, q_dict):
        pose = dict_to_pose(q_dict)
        return self.solver.inverse(pose)

    def solve_fwd_kin(self, q_dict):
        pose = dict_to_pose(q_dict)
        return self.solver.forward(pose)


class RethinkIKSolver:

    def __init__(self):
        self.r_ns = "ExternalTools/right/PositionKinematicsNode/IKService"
        self.r_iksvc = rospy.ServiceProxy(self.r_ns, SolvePositionIK)

        self.l_ns = "ExternalTools/left/PositionKinematicsNode/IKService"
        self.l_iksvc = rospy.ServiceProxy(self.l_ns, SolvePositionIK)

    def solve(self, limb, pose):
        ikreq = SolvePositionIKRequest()
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        pose = PoseStamped(header=hdr, pose=pose)
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
            return None

        # Check if result valid, and type of seed ultimately used to get solution
        # convert rospy's string representation of uint8[]'s to int'
        resp_seeds = struct.unpack('<%dB' % len(resp.result_type), resp.result_type)
        if resp_seeds[0] != resp.RESULT_INVALID:
            seed_str = {
                        ikreq.SEED_USER: 'User Provided Seed',
                        ikreq.SEED_CURRENT: 'Current Joint Angles',
                        ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                       }.get(resp_seeds[0], 'None')
            rospy.loginfo("IK Solution SUCCESS - Valid Joint Solution Found from Seed Type: {0}".format(seed_str))
            # Format solution into Limb API-compatible dictionary
            limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
            rospy.loginfo("IK Joint Solution:\n{0}".format(limb_joints))
        else:
            rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
            return None
        return limb_joints
