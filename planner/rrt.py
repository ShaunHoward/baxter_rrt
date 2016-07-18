import numpy as np
import helpers as h


def ik_soln_exists(goal_pose, kin):
    goal_angles = None
    if goal_pose is not None:
        goal_angles = None
        if goal_pose:
            goal_angles = kin.solve(position=goal_pose.position, orientation=goal_pose.orientation)
    if goal_angles is not None:
        return True, goal_angles
    else:
        return False, None


class RRT:
    # one-way RRT

    def __init__(self, q_start, p_goal, kin_solver, side, joint_names, obstacles, exec_angles_method):
        self.kin = kin_solver
        self.q_start = q_start
        self.update_goal(p_goal)
        self.nodes = []
        self.side = side
        # note: obstacles should be and are assumed to be sorted by distance from base link
        self.obstacles = obstacles
        self.joint_names = joint_names
        self.exec_angles_method = exec_angles_method

    def add_nodes(self, nodes_to_add):
        self.nodes.extend(nodes_to_add)

    def curr_node(self):
        if len(self.nodes) == 0:
            return self.q_start
        return self.nodes[-1]

    def goal_pose(self):
        return self.p_goal

    def goal_node(self):
        return self.x_goal

    def goal_point(self):
        return self.goal_node()[:3]

    def dist(self, start, stop):
        return np.linalg.norm(stop - start)

    def _dist_to_goal(self, curr):
        return self.dist(curr, self.goal_node())

    def dist_to_goal(self):
        return self._dist_to_goal(self.fwd_kin(self.curr_node()))

    def closest_node_to_goal(self):
        return self.curr_node()

    def workspace_delta(self, x_curr):
        return (self.x_goal - x_curr)[:6]

    def fwd_kin(self, q_list):
        return self.kin.solve_fwd_kin(q_list)

    def joint_fwd_kin(self, q_list, end_link):
        return self.kin.joint_fwd_kin(q_list, "base", end_link)

    def fwd_kin_all(self, q_list):
        return self.kin.fwd_kin_all(q_list)

    def update_obstacles(self, new_obs):
        # note: obstacles should be and are assumed to be sorted by distance from base link
        self.obstacles = np.mat(new_obs)

    def update_goal(self, p_goal):
        self.x_goal = h.pose_to_ndarray(p_goal)
        self.p_goal = p_goal
        print "updating rrt goal"

    def _check_collision(self, x_3x1, avoidance_radius):
        if len(self.obstacles) > 1:
            for obs_point in self.obstacles[:]:
                dist = np.linalg.norm(obs_point - x_3x1)
                if dist > avoidance_radius:
                    # any obstacles outside of avoidance radius of robot since obstacles are sorted by distance
                    print "collisions highly unlikely based on closest points and obstacle avoidance radius..."
                    return True
        else:
            # no collisions if no obstacles
            return True
        return False

    def _check_collisions(self, link_pose_mat, avoidance_radius):
        for link_pose in link_pose_mat:
            # only use x,y,z from link pose
            x_3x1 = np.array((link_pose[0, 0], link_pose[0, 1], link_pose[0, 2]))
            if not self._check_collision(x_3x1, avoidance_radius):
                return False
        return True

    def collision_free(self, q_new_angles, avoidance_radius=0.2):
        # get the pose of each link in the arm
        # only take from the second on since the first two are always the same at 0,0,0
        link_pose_matrix = self.fwd_kin_all(q_new_angles)
        selected_collision_end_links = link_pose_matrix[len(link_pose_matrix)-4:]
        # check collisions for each link in the arm
        return self._check_collisions(selected_collision_end_links, avoidance_radius)

    def exec_angles(self, q):
        q_dict = dict()
        curr = 0
        for n in self.joint_names:
            q_dict[n] = q[curr]
            curr += 1
        self.exec_angles_method(q_dict, self.side)

    def extend_toward_goal(self, dist_thresh=0.02):
        # returns the distance of the last nearest node to goal generated by this method
        # get the closest node to goal and try to complete the tree
        q_old = self.closest_node_to_goal()
        first = True
        Q_new = []
        prev_dist_to_goal = self.dist_to_goal()
        while first or prev_dist_to_goal > dist_thresh:
            print "looking for jacobian soln..."
            if first:
                first = False
            J_T = self.kin.jacobian_transpose(q_old)
            x_old = self.fwd_kin(q_old)
            d_x = self.workspace_delta(x_old)
            d_q = np.dot(J_T, d_x).tolist()
            d_q = np.array(d_q[0])
            q_new = q_old + d_q
            curr_dist_to_goal = self._dist_to_goal(self.fwd_kin(q_new))
            if curr_dist_to_goal < prev_dist_to_goal and self.collision_free(q_new):
                print "jacobian goal step: curr dist to goal: " + str(curr_dist_to_goal)
                self.exec_angles(q_new)
                Q_new.append(q_new)
                q_old = q_new
                prev_dist_to_goal = curr_dist_to_goal
            else:
                print "jac: found non-collision free soln"
                break
        self.add_nodes(Q_new)

    def ik_extend_randomly(self, curr_pos, dist_thresh, offset=0.1):
        # returns the nearest distance to goal from the last node added by this method
        # only add one node via random soln for now
        # first = True
        Q_new = []
        prev_dist_to_goal = self.dist_to_goal()
        num_tries_left = 5
        first = True
        while prev_dist_to_goal > dist_thresh and num_tries_left > 0:
            goal_pose = self.goal_pose()
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
                    if curr_coord < goal_coord:
                        next_point.append(h.generate_random_decimal(curr_coord-offset, goal_coord+offset))
                    else:
                        next_point.append(h.generate_random_decimal(goal_coord-offset, curr_coord+offset))
            print "looking for ik soln..."
            if self._check_collision(next_point, 0.2):
                next_pose = h.generate_goal_pose_w_same_orientation(next_point, goal_pose.orientation)
                solved, q_new = ik_soln_exists(next_pose, self.kin)
                if solved:
                    curr_dist_to_goal = self._dist_to_goal(self.fwd_kin(q_new))
                    curr_pos = next_point
                    if curr_dist_to_goal < prev_dist_to_goal and self.collision_free(q_new):
                        print "random ik planner: curr dist to goal: " + str(curr_dist_to_goal)
                        self.exec_angles(q_new)
                        Q_new.append(q_new)
                        prev_dist_to_goal = curr_dist_to_goal
                        continue
                    else:
                        print "ik: soln not collision free..."
                else:
                    print "could not find ik soln for generated point"
            num_tries_left -= 1
        self.add_nodes(Q_new)
