import math
import numpy as np
import time

import helpers as h
from solver.collision_checker import CollisionChecker

__author__ = "Shaun Howard (smh150@case.edu)"


def ik_soln_exists(goal_pose, kin):
    """
    Determines if an IK solution exists for the goal_pose using the provided kinematics solver instance, kin.
    This solver automatically checks if a solution is within joint limits. If a solution is found beyond joint limits,
    the values will be clipped to the limit for that joint.
    :param goal_pose: the goal pose to determine if a soln exists for
    :param kin: the kinematics solver instance
    :return: whether a soln exists and the angles for that soln as a numpy array
    """
    goal_angles = None
    if goal_pose is not None:
        goal_angles = None
        if goal_pose:
            goal_angles = kin.solve(position=goal_pose.position, orientation=goal_pose.orientation)
            if not kin.joints_in_limits(goal_angles):
                # if goals not within limits, clip and reset them to the limits
                goal_angles = kin.clip_joints_to_limits(goal_angles)
    if goal_angles is not None:
        return True, goal_angles
    else:
        return False, None


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

    def __init__(self, q_start, goal_pose, kin_solver, side, joint_names, obstacles, exec_angles_method):
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
        :param obstacles: the list of obstacles not including the arm being planned for
        :param exec_angles_method: the method to execute the joint angles on Merry
        """
        self.kin = kin_solver
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
        self.joint_names = joint_names
        self.exec_angles_method = exec_angles_method

        self.current_nodes = []

    def add_nodes(self, nodes_to_add):
        # q
        self.nodes.extend(nodes_to_add)

    def add_pos_nodes(self, nodes):
        # (dist, x_pos)
        self.pos_nodes.extend(nodes)

    def most_recent_node(self):
        if len(self.nodes) == 0:
            return self.q_start
        return self.nodes[-1]

    def get_goal_pose(self):
        return self.goal_pose

    def goal_node(self):
        return self.x_goal

    def goal_point(self):
        return self.goal_node()[:3]

    def fwd_kin(self, q_list):
        return self.kin.solve_fwd_kin(q_list)

    def dist(self, start, stop):
        return np.linalg.norm(stop - start)

    def _dist_to_goal(self, curr):
        return self.dist(curr, self.goal_node())

    def dist_to_goal(self, q_curr):
        return self._dist_to_goal(self.fwd_kin(q_curr))

    def already_picked(self, q):
        already_picked = False
        for n in self.current_nodes:
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
            node = self.pos_nodes[i]
            if check_if_picked and self.already_picked(node[1]):
                continue
            norm_dist = math.fabs(node[0])
            if norm_dist < min_dist:
                min_dist = norm_dist
                min_dist_node = self.nodes[i]
        if not self.already_picked(min_dist_node):
            self.already_picked_nodes.append(min_dist_node)
        return min_dist, min_dist_node

    def closest_node(self, q):
        min_dist = 10000
        min_dist_node = self.q_start
        for i in range(len(self.pos_nodes)):
            node = self.pos_nodes[i]
            norm_dist = math.fabs(node[0])
            if norm_dist < min_dist:
                min_dist = norm_dist
                min_dist_node = node[i]
        return min_dist_node

    def workspace_delta(self, x_curr):
        return (self.x_goal - x_curr)[:6]

    def update_obstacles(self, new_obs):
        # note: obstacles should be and are assumed to be sorted by distance from base link
        self.obstacles = np.mat(new_obs)

    def update_goal(self, goal_pose):
        self.x_goal = h.pose_to_7x1_vector(goal_pose)
        self.goal_pose = goal_pose
        print "updating rrt goal"

    def exec_angles(self, q):
        """
        Moves to the specified joint angles, q.
        :param q: a 7x1 vector of joint angles to approach
        :return: the status of the operation, 0 for success, 1 for error
        """
        q_dict = dict()
        curr = 0
        for n in self.joint_names:
            q_dict[n] = q[curr]
            curr += 1
        return self.exec_angles_method(q_dict, self.side)

    # def check_positions(self, x_positions):
    #     for x_pos in x_positions:
    #         if x_min < x_pos[0] < x_max and \
    #         y_min < x_pos[1] < y_max and \
    #         z_min < x_pos[2] < z_max:

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

    def extend_toward_goal(self, dist_thresh=0.02, max_dist_cap=0.1, use_pinv=True, use_adv_gradient_descent=True, time_limit=10):
        """
        Uses an online hybrid jacobian transpose and p-inverse RRT planning step to approach the goal
        to the provided distance threshold. Extends to approachable nodes generated by the planner as
        they are found.
        :param dist_thresh: the distance from the goal considered tolerable as reaching the goal pose
        :param max_dist_cap: the distance at which to cap solutions from traveling further away from the goal than
         currently at
        :param use_pinv: whether to use the jacobian pseudo-inverse (True if so, False for transpose)
        :param use_adv_gradient_descent: whether to use an advanced gradient descent step to the goal
        :param time_limit: the limit in seconds to limit planning to
        """
        # convert time limit from seconds to milliseconds
        time_limit *= 1000
        # get the closest node to goal and try to complete the tree
        prev_dist_to_goal, q_old = self.closest_node_to_goal()
        first = True
        Q_new = []
        X_new = []
        # convert seconds to milliseconds
        current_milli_time = lambda: int(round(time.time() * 1000))
        curr_time = current_milli_time()
        prev_time = curr_time
        while prev_dist_to_goal > dist_thresh and curr_time - prev_time < time_limit:
            print "looking for jacobian soln..."
            self.current_nodes.append(q_old)
            x_old = self.fwd_kin(q_old)
            d_x = self.workspace_delta(x_old)
            if use_adv_gradient_descent:
                J = self.kin.jacobian(q_old)
                if use_pinv:
                    J_T = np.linalg.pinv(J)
                else:
                    J_T = J.T
                direction = np.dot(J_T, np.dot(J, J_T)**-1)
            else:
                if use_pinv:
                    J_T = self.kin.jacobian_pinv(q_old)
                else:
                    J_T = self.kin.jacobian_transpose(q_old)
                direction = J_T
            d_q = np.dot(direction, d_x).tolist()
            d_q = np.array(d_q[0])
            q_new = q_old + d_q  # math.fabs(curr_dist_to_goal-prev_dist_to_goal) < max_dist_cap \
            # new_positions = self.kin.fwd_kin_all(q_new)
            q_new, all_at_limits = self.clip_joints_to_limits(q_new)
            curr_dist_to_goal = self.dist_to_goal(q_new)
            already_picked = self.already_picked(q_new)
            if not all_at_limits and not already_picked:
                # and self.check_positions(new_positions) \
                # and self.collision_checker.collision_free(q_new):
                print "jacobian goal step: curr dist to goal: " + str(curr_dist_to_goal)
                # self.exec_angles(q_new)
                Q_new.append(q_new)
                x_node = self.fwd_kin(q_new)
                X_new.append((curr_dist_to_goal, x_node))
                prev_dist_to_goal = curr_dist_to_goal
                q_old = q_new
                self.current_nodes.append(q_new)
            elif all_at_limits:
                print "jac: all joints at limits"
                break
            elif curr_dist_to_goal >= prev_dist_to_goal:
                print "jac: found node not close enough to goal"
                break
            else:
                print "jac: soln not collision-free or already picked"
                break
            # update the previous and current time
            prev_time = curr_time
            curr_time = current_milli_time()
        self.cleanup_nodes()
        self.add_nodes(Q_new)
        self.add_pos_nodes(X_new)

    def ik_extend_randomly(self, curr_pos, dist_thresh, offset=0.1, avoidance_radius=0.4, num_tries=5):
        """
        Random straight-line extension using KDL IK planner for the RRT step.
        Starts generating random points close to goal, from the goal out, until it finds a valid solution.
        Adds the successful and valid step nodes to the RRT.
        :param curr_pos: the 3x1 current position of the end effector
        :param dist_thresh: the distance threshold considered tolerable for reaching the goal state
        :param offset: the cartesian step offset upper limit of the range at which to generate points, starting from
        the goal and stepping out the range until a valid soln is found within that range.
        :param avoidance_radius: the radius to avoid collisions around the arm being planner for
        :param num_tries: the number of times to regenerate a new random point at a larger step away from goal
        than the previous try with step being offset
        """
        # TODO modify to start randomly near goal then move away from goal if no solns present
        # returns the nearest distance to goal from the last node added by this method
        # only add one node via random soln for now
        # first = True
        Q_new = []
        X_new = []
        prev_dist_to_goal, q_old = self.closest_node_to_goal(False)
        num_tries_left = num_tries
        first = True
        # start with soln at offset and work away from goal
        curr_diameter = offset
        while prev_dist_to_goal > dist_thresh and num_tries_left > 0:
            goal_pose = self.get_goal_pose()
            if first:
                first = False
                # first, try the goal point
                next_point = self.goal_point()
            else:
                goal_arr = self.goal_node()
                next_point = []
                for i in range(3):
                    curr_coord = curr_pos[i]
                    goal_coord = goal_arr[i]
                    radius = curr_diameter/2.0
                    if curr_coord < goal_coord:
                        next_point.append(h.generate_random_decimal(curr_coord-radius, goal_coord+radius))
                    else:
                        next_point.append(h.generate_random_decimal(goal_coord-radius, curr_coord+radius))
            print "looking for ik soln..."
            #if self.collision_checker.check_collision(next_point, avoidance_radius):
            next_pose = h.generate_goal_pose_w_same_orientation(next_point, goal_pose.orientation)
            solved, q_new = ik_soln_exists(next_pose, self.kin)
            if solved:
                curr_dist_to_goal = self._dist_to_goal(self.fwd_kin(q_new))
                # only add the point as a soln if the distance from this point to goal is less than that from the
                # last end effector point
                if curr_dist_to_goal < prev_dist_to_goal: # and self.collision_checker.collision_free(q_new):
                    print "random ik planner: curr dist to goal: " + str(curr_dist_to_goal)
                    # self.exec_angles(q_new)
                    Q_new.append(q_new)
                    X_new.append((curr_dist_to_goal, self.fwd_kin(q_new)))
                    prev_dist_to_goal = curr_dist_to_goal
                    curr_pos = next_point
                #    continue
                #elif curr_dist_to_goal >= prev_dist_to_goal:
                #    print "ik: soln not close enough to goal"
                #else:
                #    print "ik: soln not collision-free"
            else:
                print "could not find ik soln for generated point"
            # increment current range for generating random points by adding another offset amount
            curr_diameter += offset
            num_tries_left -= 1
        self.cleanup_nodes()
        self.add_nodes(Q_new)
        self.add_pos_nodes(X_new)
