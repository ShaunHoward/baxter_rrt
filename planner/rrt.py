import numpy as np


class RRT:
    # one-way RRT

    def __init__(self, q_start, qx_goal, left_kin_solver, right_kin_solver):
        self.q_start = q_start
        self.q_goal = qx_goal[0]  # q goal is first in tuple
        self.x_goal = qx_goal[1]  # x goal is second in tuple
        self.node_edge_pairs = []
        self.left_kin = left_kin_solver
        self.right_kin = right_kin_solver

    def add_nodes(self, node_edge_pairs):
        self.node_edge_pairs.extend(node_edge_pairs)

    def curr_node(self):
        if len(self.node_edge_pairs) == 0:
            return self.q_start
        return self.node_edge_pairs[-1][0]

    def goal_node(self):
        return self.x_goal

    def dist(self, start, stop):
        return np.linalg.norm(stop - start)

    def _dist_to_goal(self, curr):
        return self.dist(self.goal_node(), curr)

    def dist_to_goal(self):
        return self._dist_to_goal(self.curr_node())

    def closest_node_to_goal(self):
        return self.q_goal

    def collision_free(self):
        pass

    def workspace_delta(self, q_curr):
        return self.q_goal - q_curr

    def get_kin(self, side):
        if side == "left":
            kin = self.left_kin
        else:
            kin = self.right_kin
        return kin

    def fwd_kin(self, q_dict, side):
        kin = self.get_kin(side)
        return kin.forward_position_kinematics(q_dict)

    def jacobian_transpose(self, q_curr, side):
        kin = self.get_kin(side)
        return kin.jacobian_transpose(q_curr)

    def extend_toward_goal(self, side, dist_thresh=0.1):
        q_old = self.closest_node_to_goal()
        first = True
        q_new = q_old
        Q_new = []
        while first or self._dist_to_goal(q_new) > dist_thresh:
            if first:
                first = False
            J_T = self.jacobian_transpose(q_old, side)
            d_x = self.workspace_delta(q_old)
            d_q = J_T * d_x
            q_new = q_old + d_q
            if self.collision_free(q_old, q_new):
                Q_new.append(q_new)
            else:
                return Q_new
            q_old = q_new
        self.add_nodes(Q_new)

    def extend_randomly(self):
        pass
