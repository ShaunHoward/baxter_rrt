def generate_and_execute_random_path_from_start_to_end(self, side, start_pose, DIST_THRESHOLD=0.05, MIN_THRESH=0.01,
                                                       MAX_THRESH=0.5, MAX_ITERS=50, MAX_GUESSES=20):
    # hold orientation constant as it is in the goal pose
    curr_point = np.array((start_pose["position"].x, start_pose["position"].y, start_pose["position"].z))
    goal_point = self.get_goal_point(side)
    curr_dist = np.linalg.norm(goal_point - curr_point)
    next_point = curr_point.copy()
    niters = 0
    status = OK
    goal_met = False
    got_guess = False
    nguesses = 0
    next_direction = self.compute_force_vetor_at_point(goal_point, curr_point)
    goal_changed = False
    using_force = False

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
            # # determine which direction to move in based on potential field/force approach
            # F_tot = self.compute_force_vetor_at_point(goal_point, next_point)
            # next_direction = F_tot / np.linalg.norm(F_tot)
            # goal_changed = False

            """
                linalg.solve(a, b)	Solve a linear matrix equation, or system of linear scalar equations.
                linalg.tensorsolve(a, b[, axes])	Solve the tensor equation a x = b for x.
                linalg.lstsq(a, b[, rcond])	Return the least-squares solution to a linear matrix equation.
            """

        for i in range(3):
            curr_coord = curr_point[i]
            goal_coord = goal_point[i]
            # if next_direction is not None:
            #     # do planning based on potential field / force approach
            #     # compute coordinate in direction of force from current point on path to goal
            #     # use parametric form of line: x=x0+tay=y0+tbz=z0+tc
            #     next_point[i] = curr_point[i] + t * next_direction[i]
            #     using_force = True
            # else:
            #     using_force = False
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
        next_dist = math.fabs(np.linalg.norm(next_point - goal_point))
        # if MAX_THRESH > next_diff > MIN_THRESH:
        nguesses += 1
        if using_force or next_dist < curr_dist:
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
    left_path = [h.get_pose(0.590916633606, 0.338178694248, 0.220857322216, 3.92595538301e-08, 0.687881827354,
                            -9.44256584035e-09, 0.725822746754),
                 h.get_pose(0.715455174446, 0.338178694248, 0.227546870708, 3.92595538301e-08, 0.687881827354,
                            -9.44256584035e-09, 0.725822746754)]
    right_path = [None, None]
    path = []
    for i in range(len(left_path)):
        path.append((left_path[i], right_path[i]))
    return path


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


def compute_force_vetor_at_point(self, target_point, next_robot_point, att_potential_scale_factor=2,
                                 rep_potential_scaling_factor=2, rep_force_scale_factor=1, m=2,
                                 influence_zone=0.5):
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
    i = 0
    for obs in closest_pts:
        # do every 5 points for efficiency
        if i % 5 == 0:
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
        i += 1
    # Urep = np.array(Urep_l).sum()
    F_rep = np.sum(Frep_l, 0)
    # divide F_rep by the number of closest points to normalize the repulsive force
    F_rep_norm = F_rep / len(closest_pts)
    F_tot = Fatt + (rep_force_scale_factor * F_rep_norm)
    return F_tot


    # def has_collisions(self, pose, MIN_TOL=.1):
    #     # min tolerance in meters
    #     desired = pose.position
    #     collisions = []
    #     i = 0
    #     for p in self.closest_points:
    #         # only do every 5 points for now to speed things up
    #         if i % 5 == 0:
    #             dist = np.linalg.norm(np.array((desired.x, desired.y, desired.z)) - np.array((p.x, p.y, p.z)))
    #             if dist <= MIN_TOL:
    #                 # append the distance and the point
    #                 collisions.append((dist, p))
    #         i += 1
    #     return len(collisions) == 0, collisions