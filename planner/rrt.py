import math
import numpy as np
import time

import helpers as h
from solver.collision_checker import CollisionChecker
from solver.ik_solver import RethinkIKSolver

__author__ = "Shaun Howard (smh150@case.edu)"


def ik_soln_exists(goal_pose, kin, use_rr=False):
    """
    Determines if an IK solution exists for the goal_pose using the provided kinematics solver instance, kin.
    This solver automatically checks if a solution is within joint limits. If a solution is found beyond joint limits,
    the values will be clipped to the limit for that joint.
    :param goal_pose: the goal pose to determine if a soln exists for
    :param kin: the kinematics solver instance
    :param goal_pos: the goal cart pos matrix
    :param use_rr: whether to use random-restart ik solver (if not, then use random points with IK)
    :return: whether a soln exists and the angles for that soln as a numpy array
    """
    goal_angles = None
    if goal_pose is not None:
        goal_angles = None
        if goal_pose:
            goal_angles = kin.solve("left", goal_pose)
            #goal_angles = kin.solve(pose=goal_pose, use_rr=use_rr)
            #if not kin.joints_in_limits(goal_angles):
                # if goals not within limits, clip and reset them to the limits
            #    goal_angles = kin.clip_joints_to_limits(goal_angles)
    if goal_angles is not None:
        return True, goal_angles
    else:
        return False, None

common_position_bounds = {"x_min":0.25, "x_max":1, "z_min":-.18, "z_max": .7}
right_position_bounds = common_position_bounds.copy()
right_position_bounds.update({"y_min": -.85, "y_max": -0.05})
left_position_bounds = common_position_bounds.copy()
left_position_bounds.update({"y_min": 0.05, "y_max": .7})


class RRT:
    """
    Class for Shaun Howard's online hybrid RRT-JT/Random IK joint angle planner for Merry, the Baxter robot
    at Case Western Reserve University. Designed for the EECS 499 Algorithmic Robotics class in Spring/Summer 2016.
    """

    # list of min/max values for left arm
    r_min = []
    r_max = []

    # list of min/max values for right arm

    l_min = []
    l_max = []

    def __init__(self, q_start, goal_pose, kin_solver, side, joint_names, step_size, obstacles, obs_dist_thresh,
                 dist_thresh=0.1):
        """
        Constructor for RRT. Accepts a numpy array of starting angles, the goal pose, the kinematics solver instance
        for the RRT side arm, an ordered list of joint_names from base to end effector, a list of obstacle points that
        should not include the current arm being planned for, and the method to execute the joint angles from the
        Merry object instance used to create this RRT instance.
        :param q_start: 7x1 numpy vector of starting joint angles
        :param goal_pose: the goal pose for the end effector
        :param kin_solver: the KDL kinematics solver instance
        :param side: the side, left or right, of arm to plan for
        :param joint_names: the ordered list of joint names from base to end effector
        :param step_size: the cartesian step size to grow the tree at intervals of
        :param obstacles: the list of obstacles not including the arm being planned for
        :param obs_dist_thresh: the distance threshold to avoid obstacles around the arm joint end points
        :param dist_thresh: the distance from th previous configuration that the jacobian planner is allowed before
         stopping
        """
        self.kin = kin_solver
        self.rethink_ik = RethinkIKSolver()
        self.q_start = q_start
        # define goals
        self.x_goal = None
        self.goal_pose = None
        # update the goals
        self.update_goal(goal_pose)
        self.nodes = []
        self.pos_nodes = []
        self.already_picked_nodes = []
        self.side = side
        self.collision_checker = CollisionChecker([], self.kin)
        # note: obstacles should be and are assumed to be sorted by distance from base link
        self.obstacles = obstacles
        self.obs_dist_thresh = obs_dist_thresh
        self.joint_names = joint_names
        self.step_size = step_size
        self.dist_thresh = dist_thresh

    def add_new_node(self, q_new):
        # self.cleanup_nodes()
        self.nodes.append(q_new)
        x_new = self.fwd_kin(q_new)[:3]
        dist_to_goal = self.workspace_delta_norm(self.goal_point() - x_new)
        print "new distance to goal: %f" % dist_to_goal
        self.pos_nodes.append((dist_to_goal, x_new))
        self.already_picked_nodes.append(q_new)

    def get_goal_pose(self):
        return self.goal_pose

    def goal_node(self):
        return self.x_goal

    def goal_point(self):
        return self.goal_node()[:3]

    def fwd_kin(self, q_list):
        return self.kin.solve_fwd_kin(q_list)

    def dist(self, start, stop):
        return np.linalg.norm(stop[:3] - start[:3])

    def _dist_to_goal(self, curr):
        return self.dist(curr, self.goal_node())

    def dist_to_goal(self, q_curr):
        return self._dist_to_goal(self.fwd_kin(q_curr))

    def already_picked(self, q):
        already_picked = False
        for n in self.already_picked_nodes:
            same = 0
            for i in range(len(n)):
                if n[i] == q[i]:
                    same += 1
            already_picked = len(n) == same
            if already_picked:
                break
        return already_picked

    def closest_node_to_goal(self, check_if_picked=True):
        # return the closest dist and q node to goal
        # assure that the node has not already been picked before during this step (as to not repeat history...)
        min_dist = self.dist_to_goal(self.q_start)
        min_dist_node = np.array(self.q_start)
        for i in range(len(self.pos_nodes)):
            node = self.nodes[i]
            pos_node = self.pos_nodes[i]
            norm_dist = math.fabs(pos_node[0])
            if check_if_picked and self.already_picked(node) and min_dist < norm_dist:
                continue
            if norm_dist < min_dist:
                min_dist = norm_dist
                min_dist_node = node
        if check_if_picked and not self.already_picked(min_dist_node):
            self.already_picked_nodes.append(min_dist_node)
        return min_dist, min_dist_node

    def workspace_delta_norm(self, x_curr):
        return np.linalg.norm(self.x_goal[:3] - x_curr[:3])

    def limited_cartesian_workspace_step(self, x_curr):
        # move along normal between two points from current end effector point to goal point, cap dist at step_size
        # use true difference between euler angles in hopes rotating to goal pose quickly
        goal_pos = self.x_goal[:3]
        goal_orientation = self.x_goal[3:]
        goal_euler_angles = h.get_euler_angles_for_orientation(goal_orientation)

        curr_pos = x_curr[:3]
        curr_orientation = x_curr[3:]
        curr_euler_angles = h.get_euler_angles_for_orientation(curr_orientation)

        u = (goal_pos - curr_pos) / np.linalg.norm(goal_pos - curr_pos)
        # generate a point limited by the step size along the line between x_curr and x_goal
        limited_x_diff = self.step_size * u

        # get the true orientation difference between current and goal euler angles
        euler_diff = goal_euler_angles - curr_euler_angles
        # euler_u = euler_angle_diff / np.linalg.norm(euler_angle_diff)
        # limited_euler_diff = self.step_size * euler_u
        # orientation_diff = goal_orientation[:3] - curr_orientation[:3]
        limited_workspace_step = np.concatenate((limited_x_diff, euler_diff))
        return limited_workspace_step

    def extend_toward_goal(self):
        """
        The goal-directed extension step of the RRT which uses the Jacobian Pseudo-inverse of the given arm's joint
        angles to extend the tree toward the goal limited per step by the cartesian step_size of the rrt instance.
        """
        # get the closest node to goal and try to complete the tree
        prev_dist, q_old = self.closest_node_to_goal()
        print "looking for goal-directed jacobian soln..."
        while True:
            x_old = self.fwd_kin(q_old)
            nd_x = self.workspace_delta_norm(x_old)
            if nd_x > self.dist_thresh:
                # solution reached at this point
                break
            else:
                d_x = self.limited_cartesian_workspace_step(x_old)
                J = self.kin.jacobian(q_old)
                JPINV = np.linalg.pinv(J)
                d_q = np.dot(JPINV, d_x).tolist()
                d_q = np.array(d_q[0])
                q_new = q_old + d_q
                q_new, all_at_limits = self.clip_joints_to_limits(q_new)
                curr_dist = self.dist_to_goal(q_new)
                if self.already_picked(q_new) or all_at_limits or curr_dist >= prev_dist + self.dist_thresh:
                    print "jac: soln already picked or joints at all limits or distance grew too far from goal..."
                    break
                else:
                    print "found new jacobian node!"
                    self.add_new_node(q_new)
                    q_old = q_new
                    prev_dist = curr_dist

    def generate_random_point_at_step(self, x_old):
        return h.generate_random_3d_point_at_length_away(x_old[:3], self.step_size)

    def generate_and_check_ik_soln(self, next_point, x_near):
        # try to generate an ik solution for the given next point with the orientation of the specified x_near.
        if True:  # self.collision_checker.check_collision(next_point, avoidance_diameter):
            curr_orientation = x_near[3:]
            new_pos_arr = np.concatenate((next_point, curr_orientation), 0)
            next_pose = h.pose_vector_to_pose_msg(new_pos_arr)
            solved, q_new = ik_soln_exists(next_pose, self.rethink_ik, False)
            if solved:
                return q_new
        return None

    def ik_extend_randomly(self, num_tries=5):
        """
        Random straight-line extension using KDL IK planner for the RRT step.
        Generates up to 5 random points at the given step size from the nearest node to the goal in the rrt. Tries
        to find IK solutions for each point. If none are found, no nodes are added to the tree.
        Adds the successful and valid step nodes to the RRT.
        :param num_tries: the number of times to regenerate a new random point at the given step size from
        the node closest to the goal in the rrt
        """
        prev_dist_to_goal, q_old = self.closest_node_to_goal(False)
        x_old = self.fwd_kin(q_old)
        num_tries_left = num_tries
        print "looking for random ik soln..."
        while num_tries_left > 0:
            next_point = self.generate_random_point_at_step(x_old)
            q_soln_dict = self.generate_and_check_ik_soln(next_point, x_old)
            q_soln = h.unwrap_angles_dict_to_list(q_soln_dict, self.joint_names)
            if q_soln is not None:
                self.add_new_node(q_soln)
                print "found new random ik node!!"
                break
            else:
                print "could not find ik soln for randomly generated point..."
                # planner might not have solution, so decrement # tries left
                num_tries_left -= 1

    def update_obstacles(self, new_obs):
        # note: obstacles should be and are assumed to be sorted by distance from base link
        self.obstacles = np.mat(new_obs)

    def update_goal(self, goal_pose):
        self.x_goal = h.pose_to_7x1_vector(goal_pose)
        self.goal_pose = goal_pose
        print "updating rrt goal"

    def clip_joints_to_limits(self, goal_angles):
        all_joints_at_limits = self.kin.all_joints_at_limits(goal_angles)
        if not self.kin.joints_in_limits(goal_angles):
            # if goals not within limits, clip and reset them to the limits
            goal_angles = self.kin.clip_joints_to_limits(goal_angles)
        return goal_angles, all_joints_at_limits

    def cleanup_nodes(self):
        min_dist_to_goal = 10000
        min_node_index = len(self.nodes) - 1
        for i in range(len(self.nodes)):
            curr_dist = math.fabs(self.pos_nodes[i][0])
            if curr_dist < min_dist_to_goal:
                min_dist_to_goal = curr_dist
                min_node_index = i
        self.nodes = self.nodes[:min_node_index+1]
        self.pos_nodes = self.pos_nodes[:min_node_index+1]

    def prune_closest_node(self):
        min_dist = 10000
        min_dist_node = self.q_start
        min_dist_index = -1
        i = -1
        for i in range(len(self.pos_nodes)):
            node = self.pos_nodes[i]
            norm_dist = math.fabs(node[0])
            if norm_dist < min_dist:
                min_dist = norm_dist
                min_dist_index = i
        if i > -1:
            del self.nodes[min_dist_index]
            del self.pos_nodes[min_dist_index]
            return True
        else:
            return False

    def get_pruned_tree(self):
        return self.nodes
        # nodes_to_keep = []
        # last_good_dist = 1000
        # for i in range(len(self.nodes)):
        #     node = self.nodes[i]
        #     curr_dist = self.dist_to_goal(node)
        #     if last_good_dist > curr_dist:
        #         last_good_dist = curr_dist
        #         nodes_to_keep.append(node)
        # return nodes_to_keep

    def most_recent_node(self):
        if len(self.nodes) == 0:
            return self.q_start
        return self.nodes[-1]