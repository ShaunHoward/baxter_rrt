import argparse
import baxter_interface
import helpers as h
import math
import numpy as np
import random
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

        self.kinect_subscriber = rospy.Subscriber("kinect/depth/points", pc2.PointCloud2, self.kinect_cb)

        self.marker_subscriber = rospy.Subscriber("/merry_end_point_markers/feedback", InteractiveMarkerFeedback,
                                                  self.interactive_marker_cb)
        self.tf = tf.TransformListener()

        self.kmeans = h.get_kmeans_instance(self)

        self.left_kinematics = baxter_kinematics("left")

        self.right_kinematics = baxter_kinematics("right")

        rospy.on_shutdown(self.clean_shutdown)

        self.left_goal = None

        self.right_goal = None

        # self.bridge = CvBridge()

    def ik_solution_exists(self, side, next_pose):
        goal_angles = self.ik_solver.solve(side, next_pose)
        if goal_angles is not None:
            return True, goal_angles
        else:
            return False, None

    def get_goal_point(self, side):
        if side is "left":
            goal = self.left_goal
        else:
            goal = self.right_goal
        return h.point_to_ndarray(goal.position)

    def has_collisions(self, pose, MIN_TOL=.1):
        # min tolerance in meters
        desired = pose.position
        collisions = []
        for p in self.closest_points:

            dist = np.linalg.norm(np.array((desired.x, desired.y, desired.z)) - np.array((p.x, p.y, p.z)))
            if dist <= MIN_TOL:
                # append the distance and the point
                collisions.append((dist, p))
        return len(collisions) == 0, collisions

    def compute_force_vetor_at_point(self, target_point, next_robot_point, att_potential_scale_factor=2,
                                     rep_potential_scaling_factor=2, rep_force_scale_factor=1, m=2, influence_zone=0.5):
        """
        Computes the force vector based on closest obstacles for the end effector point to travel to during
        its current planning step. Uses a randomly selected point to approach a better solution for motion
        planning at the current planning iteration step.
        :param target_point: the goal point for the end effector to reach
        :param next_robot_point: the next point the robot may travel to
        :param att_potential_scale_factor: the scaling factor for attractive force
        :param rep_potential_scaling_factor: the scaling factor for repulsive force
        :param m: determines the shape of the potential field, m=1 leads to conic well, m=2 leads to a parabolic well
        :param influence_zone: the radius of points to take into account for motion planning around the end effector
        :return: the force vector to determine the next direction to go in for the end effector
        """
        # return None if no obstacles, since this planning force is unnecessary
        if len(self.closest_points) == 0:
            return None
        # compute attractive force component
        p_rt = target_point - next_robot_point
        pt = np.linalg.norm(p_rt)
        # dont need potential due to force simplification
        # Uatt = att_scale_factor * pt**m
        Fatt = m * att_potential_scale_factor * (pt ** (m - 2)) * p_rt

        # compute repulsive energy and force
        closest_pts = [h.point_to_ndarray(p) for p in self.closest_points]
        poi = influence_zone
        Frep_l = []
        # TODO: filter points to be voxel grid!
        for obs in closest_pts:
            p_roi = obs - next_robot_point
            psi = np.linalg.norm(p_roi)
            n_roi = p_roi / psi
            F_rep_i = -rep_potential_scaling_factor * (1 / (psi ** 2)) * n_roi
            Frep_l.append(F_rep_i)
            # if psi <= poi:
            #     energy = rep_scaling_factor * ((1/psi) - (1/poi))
            # else:
            #     energy = 0
            # Urep_l.append(energy)
        # Urep = np.array(Urep_l).sum()
        F_rep = np.sum(Frep_l, 0)
        # divide F_rep by the number of closest points to normalize the repulsive force
        F_rep_norm = F_rep / len(closest_pts)
        F_tot = Fatt + (rep_force_scale_factor * F_rep_norm)
        return F_tot

    def genrate_and_execute_random_path_from_start_to_end(self, side, start_pose, DIST_THRESHOLD=0.05, MIN_THRESH=0.01, MAX_THRESH=0.5, MAX_ITERS=50, MAX_GUESSES=20):
        # hold orientation constant as it is in the goal pose
        curr_point = np.array((start_pose["position"].x, start_pose["position"].y, start_pose["position"].z))
        goal_point = self.get_goal_point(side)
        curr_dist = np.linalg.norm(goal_point-curr_point)
        next_point = curr_point.copy()
        niters = 0
        status = OK
        goal_met = False
        got_guess = False
        nguesses = 0
        next_direction = self.compute_force_vetor_at_point(goal_point, curr_point)
        goal_changed = False

        # TODO: parameter that needs to be fit for occasion
        t = 1

        # select random x, y, z coordinates to minimize distance to goal, then check IK solution for those points
        while not goal_met:
            if niters > MAX_ITERS or nguesses > MAX_GUESSES:
                rospy.loginfo("could not find solution in time, broke planner loop...")
                break
            # rospy.loginfo("generating next random point on path for " + side + " arm.")
            goal_point = self.get_goal_point(side)
            if side is "left":
                goal_pose = self.left_goal
            else:
                goal_pose = self.right_goal

            if goal_changed:
                # determine which direction to move in based on potential field/force approach
                F_tot = self.compute_force_vetor_at_point(goal_point, next_point)
                next_direction = F_tot / np.linalg.norm(F_tot)
                goal_changed = False

                """
                    linalg.solve(a, b)	Solve a linear matrix equation, or system of linear scalar equations.
                    linalg.tensorsolve(a, b[, axes])	Solve the tensor equation a x = b for x.
                    linalg.lstsq(a, b[, rcond])	Return the least-squares solution to a linear matrix equation.
                """

            for i in range(3):
                curr_coord = curr_point[i]
                goal_coord = goal_point[i]
                if next_direction is not None:
                    # do planning based on potential field / force approach
                    # compute coordinate in direction of force from current point on path to goal
                    # use parametric form of line: x=x0+tay=y0+tbz=z0+tc
                    next_point[i] = curr_point[i] + t * next_direction[i]
                else:
                    # do planning based on straight-line approach in range from current point to
                    # goal point for each coord
                    if curr_coord < goal_coord:
                        next_point[i] = h.generate_random_decimal(curr_coord, goal_coord)
                    else:
                        next_point[i] = h.generate_random_decimal(goal_coord, curr_coord)

            # compute only the potential at this point to determine if point is reachable
            # Uatt, Urep = self.compute_potential_at_point(goal_point, next_point)

            next_pose = h.get_pose(next_point[0],
                                   next_point[1],
                                   next_point[2],
                                   goal_pose.orientation.x,
                                   goal_pose.orientation.y,
                                   goal_pose.orientation.z,
                                   goal_pose.orientation.w)
            next_dist = math.fabs(np.linalg.norm(next_point-goal_point))
            # if MAX_THRESH > next_diff > MIN_THRESH:
            nguesses += 1
            if next_dist < curr_dist:
                rospy.loginfo("found next reasonable random goal")
                rospy.loginfo("checking if IK solution exists for next goal.")
                result, goal_angles = self.ik_solution_exists(side, next_pose)
                if result:
                    # check if generated next pose is within or near obstacle
                    has_collision, collisions = self.has_collisions(next_pose)
                    if not has_collision:
                        nguesses = 0
                        curr_dist = next_dist
                        curr_point = next_point
                        rospy.loginfo("IK goal solution found. executing goal segment.")
                        status = self.check_and_execute_goal_angles(goal_angles, side)
                        if status is OK:
                            goal_changed = True
                            rospy.loginfo("published next goal pose")
                    else:
                        rospy.loginfo("collisions found with object for generated endpoint goal")
                else:
                    status = ERROR
            if curr_dist <= DIST_THRESHOLD:
                goal_met = True
            niters += 1
        if goal_met:
            rospy.loginfo("met goal pose for " + side + " arm")
        return status

    def default_path(self):
        # hard-coded default path for both arms
        left_path = [h.get_pose(0.590916633606, 0.338178694248, 0.220857322216, 3.92595538301e-08, 0.687881827354, -9.44256584035e-09, 0.725822746754),
                     h.get_pose(0.715455174446, 0.338178694248, 0.227546870708, 3.92595538301e-08, 0.687881827354, -9.44256584035e-09, 0.725822746754)]
        right_path = [None, None]
        path = []
        for i in range(len(left_path)):
            path.append((left_path[i], right_path[i]))
        return path

    def approach_goals(self):
        """
        Tries to shake the nearest hand possible using the limb instantiated.
        Loop runs forever; kill with ctrl-c.
        """
        # path = self.generate_approach_path(self.left_goal)
        # path = self.default_path()
        while True:
            lstatus = OK
            rstatus = OK
            if self.left_goal:
                lstatus = self.genrate_and_execute_random_path_from_start_to_end("left", self.left_arm.endpoint_pose())
            if self.right_goal:
                rstatus = self.genrate_and_execute_random_path_from_start_to_end("right", self.right_arm.endpoint_pose())
            if lstatus is ERROR and rstatus is ERROR:
                self.left_arm.set_joint_position_speed(0.0)
                self.right_arm.set_joint_position_speed(0.0)
        # if self.approach(path) is "OK":
        #    return 0
        return 0

    def approach_single_goal(self, side, goal):
        """
        Attempts to successfully visit the goal point using the IK solver.
        :param side: the arm side to use
        :param goal: 3d goal point in the base frame with x, y, z
        :return: status of attempt to approach the goal point
        """
        status = OK
        if goal is None:
            if side is "left":
                goal = self.left_goal
            else:
                goal = self.right_goal

        # approach the goal points
        goal_met = False

        while goal_met is False and status is OK:
            obstacles = None
            # obstacles = h.get_critical_points_of_obstacles(self)
            rospy.sleep(1)
            goal_angles = None
            if goal:
                goal_angles = self.ik_solver.solve(side, goal)
                if goal_angles:
                    goal_met = True
                else:
                    goal_met = False
            else:
                goal_met = True

            if obstacles is not None:
                # apply obstacle avoidance
                pass
            # execute goal angles if they are available
            if goal_angles is not None:
                # do left arm planning
                status = self.check_and_execute_goal_angles(goal_angles, side)
        return status

    def move_to_joint_positions(self, joint_positions, side, use_move=True):
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
                if use_move:
                    self.right_arm.move_to_joint_positions(joint_positions)
                else:
                    self.right_arm.set_joint_positions(joint_positions)
            elif side is "left":
                self.left_arm.set_joint_position_speed(self.left_speed)
                if use_move:
                    self.left_arm.move_to_joint_positions(joint_positions)
                else:
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

    def kinect_cb(self, data, source="kinect_link", dest="base", min_dist=0.4, max_dist=2, min_height=1):
        """
        Receives kinect points from the kinect subscriber linked to the publisher stream.
        :return: kinect points numpy array
        """
        points = [p for p in pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z"))]
        transformed_points = h.transform_pcl2(self.tf, dest, source, points, 3)
        self.closest_points = [p for p in transformed_points if min_dist < math.sqrt(p.x**2 + p.y**2 + p.z**2) < max_dist
                               and p.z > min_height]
        if len(self.closest_points) > 0:
            print "kinect cb: there are this many close points: " + str(len(self.closest_points))

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
            status = self.move_to_joint_positions(joint_positions, side, True)
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


def run_robot(args):
    print("Initializing node... ")
    rospy.init_node("merry")
    merry = Merry(args.limb, args.speed, args.accuracy)
    rospy.on_shutdown(merry.clean_shutdown)
    status = merry.approach_goals()
    sys.exit(status)


if __name__ == '__main__':
    run_robot(h.get_args())
