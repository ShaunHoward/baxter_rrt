import numpy as np
import helpers as h


def ik_soln_exists(goal_pos, kin_solver_instance):
    goal_angles = None
    if goal_pos is not None:
        # goal = self.generate_goal_pose(side, (goal_pos.x, goal_pos.y, goal_pos.z))
        # do inverse kinematics for cart pose to joint angles, then convert joint angles to joint velocities
        goal_angles = kin_solver_instance.inverse_kinematics(goal_pos)
    if goal_angles:
        return True, goal_angles
    else:
        return False, None


def wrap_angles_in_dict(angles, keys):
    q_dict = dict()
    for i in range(len(keys)):
        q_dict[keys[i]] = angles[i]
    return q_dict


class RRT:
    # one-way RRT

    def __init__(self, q_start, x_goal, kin_solver, side):
        self.kin = kin_solver
        self.q_start = q_start
        self.x_goal = h.pose_to_ndarray(x_goal)
        self.nodes = []
        self.side = side

    def add_nodes(self, nodes_to_add):
        self.nodes.extend(nodes_to_add)

    def curr_node(self):
        if len(self.nodes) == 0:
            return self.q_start
        return self.nodes[-1]

    def goal_node(self):
        return self.x_goal

    def dist(self, start, stop):
        return np.linalg.norm(stop - start)

    def _dist_to_goal(self, curr):
        return self.dist(curr, self.goal_node())

    def dist_to_goal(self):
        return self._dist_to_goal(self.fwd_kin(self.curr_node()))

    def closest_node_to_goal(self):
        return self.curr_node()

    def workspace_delta(self, x_curr):
        return self.x_goal - x_curr

    def fwd_kin(self, q_dict):
        return self.kin.forward_position_kinematics(q_dict)

    def jacobian_transpose(self, q_curr):
        return self.kin.jacobian_transpose(q_curr)

    def collision_free(self, q_new, obstacle_waves, obs_mapping_fn, min_thresh=0.05):
        x_new = self.fwd_kin(q_new)
        return self._check_collisions(x_new, obstacle_waves, obs_mapping_fn, min_thresh)

    def _check_collisions(self, x, obstacle_waves, obs_mapping_fn, min_thresh):
        x_indx = obs_mapping_fn(x, self.side)
        if len(obstacle_waves) > x_indx:
            obs_wave = obstacle_waves[x_indx]
            for obs in obs_wave:
                if np.linalg.norm(obs-x) < min_thresh:
                    print "collisions found for rrt point..."
                    return False
        return True

    def extend_toward_goal(self, obstacle_waves, obs_mapping_fn, dist_thresh=0.1):
        # get the closest node to goal and try to complete the tree
        q_old = self.closest_node_to_goal()
        q_old_angles = np.array(q_old.values())
        first = True
        q_new = q_old.copy()
        Q_new = []
        while first or self._dist_to_goal(q_new) > dist_thresh:
            if first:
                first = False
            J_T = self.jacobian_transpose(q_old_angles)
            x_old = self.fwd_kin(q_old_angles)
            d_x = self.workspace_delta(x_old)
            d_q = J_T * d_x
            q_new_angles = q_old_angles + d_q
            if self.collision_free(q_new_angles, obstacle_waves, obs_mapping_fn):
                Q_new.append(wrap_angles_in_dict(q_new_angles, q_old.keys()))
            else:
                break
            q_old = q_new
        self.add_nodes(Q_new)

    def ik_extend_randomly(self, obstacle_waves, obs_mapping_fn, dist_thresh):
        # only add one node via random soln for now
        x_curr = self.fwd_kin(self.curr_node())
        goal = self.fwd_kin(self.goal_node())
        first = True
        Q_new = []
        while first or self.workspace_delta(x_curr) > dist_thresh:
            if first:
                first = False
            next_point = []
            for i in range(3):
                curr_coord = x_curr[i]
                goal_coord = goal[i]
                if curr_coord < goal_coord:
                    next_point.append(h.generate_random_decimal(curr_coord, goal_coord))
                else:
                    next_point.append(h.generate_random_decimal(goal_coord, curr_coord))

            if self._check_collisions(next_point, obstacle_waves, obs_mapping_fn, dist_thresh):
                solved, q_new = ik_soln_exists(next_point, self.kin)
                if solved:
                    if self.collision_free(q_new, obstacle_waves, obs_mapping_fn):
                        Q_new.append(wrap_angles_in_dict(q_new, self.curr_node().keys()))
                    else:
                        break
        self.add_nodes(Q_new)

