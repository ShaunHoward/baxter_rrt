import numpy as np

__author__ = "Shaun Howard (smh150@case.edu)"


class CollisionChecker:
    """
    Class for checking collisions along one side of the Baxter robot.
    The class has an obstacle list of numpy 3x1 point vectors and a KDL kinematics solver instance.
    The collision_free method determines if a list of joint angles will cause any link on the arm to
    intersect any obstacles within a certain obstacle avoidance radius in meters.
    """

    def __init__(self, obstacles, kin):
        """
        Constructor for CollisionChecker. Takes nx3 obstacles vector and KDL kinematics solver instance.
        :param obstacles: nx3 obstacles vector
        :param kin: KDL kinematics solver instance
        """
        self.obstacles = obstacles
        self.kin = kin

    def update_obstacles(self, new_obs):
        """
        Completely replaces the current obstacles list with the new obstacles provided.
        :param new_obs: the new obstacles list of 3x1 point vectors
        """
        self.obstacles = new_obs

    def fwd_kin(self, q_list):
        """Solve fwd kinematics wrapper for kdl kin sovler using provided q_list."""
        return self.kin.solve_fwd_kin(q_list)

    def joint_fwd_kin(self, q_list, end_link):
        """Solves joint fwd kinematics from base to provided end_link using the q_list as starting point."""
        return self.kin.joint_fwd_kin(q_list, "base", end_link)

    def fwd_kin_all(self, q_list):
        """Gets the list of fwd_kinematic results for all robot arm links using the specified q_list."""
        return self.kin.fwd_kin_all(q_list)

    def check_collision(self, x_3x1, avoidance_radius):
        """
        Determines if the 3x1 point vector provided collides with any obstacles within the specified
        avoidance radius.
        :param x_3x1: the 3x1 numpy point vector containing x,y,z
        :param avoidance_radius: the radius in meters to avoid obstacles around point with
        :return: True if collision with 3x1 point, False otherwise
        """
        if len(self.obstacles) > 1:
            for obs_point in self.obstacles[:]:
                dist = np.linalg.norm(obs_point - x_3x1)
                if dist < avoidance_radius:
                    print "dist: " + str(dist)
                    # a collision was found within the avoidance radius
                    return True
        return False

    def _check_collisions(self, link_pose_mat, avoidance_radius):
        """
        Determines if any of the arm links will intersect with objects within the provided
        avoidance radius (in meters). Returns True if there are collisions along the specified arm link positions,
        False otherwise.
        :param link_pose_mat: the matrix of 3x1 point vectors of an arm on the robot
        :param avoidance_radius: the radius in meters to avoid obstacles within around the arm links
        :return: True if there are collisions, False if there are not
        """
        for link_pose in link_pose_mat:
            # only use x,y,z from link pose
            x_3x1 = np.array((link_pose[0, 0], link_pose[0, 1], link_pose[0, 2]))
            if self.check_collision(x_3x1, avoidance_radius):
                return True
        return False

    def collision_free(self, q_new_angles, avoidance_radius=0.3):
        """
        Determines if the provided vector of new angles are collision free around the given
        avoidance radius.
        :param q_new_angles: the 7x1 numpy vector of angles to check for collision, ordered from base to end effector
        :param avoidance_radius: the obstacle avoidance radius in meters
        :return: whether the arm will collide with obstacles when going to the specified q_new_angles
        """
        # get the poses of all links in the arm
        # only take from the first on since the first is always the same, closest to 0,0,0
        link_pose_matrix = self.fwd_kin_all(q_new_angles)
        selected_collision_end_links = link_pose_matrix[len(link_pose_matrix) - 5:]
        # check collisions for each link in the arm
        return not self._check_collisions(selected_collision_end_links, avoidance_radius)
