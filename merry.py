import baxter_interface
import helpers as h
import numpy as np
import random
import rospy
import sys
import threading

from sensor_msgs.msg import JointState
from sensor_msgs.msg import PointCloud
from visualization_msgs.msg import InteractiveMarkerFeedback

from planner.rrt import RRT
from solver.ik_solver import KDLIKSolver

__author__ = 'Shaun Howard (smh150@case.edu)'


class Merry:
    # TODO put in paper title below
    """
    A class for Merry, the Baxter Robot at Case Western Reserve University's Mobile Robotics Lab.
    One notable addition of Merry over the normal Baxter is the use of the Microsoft Kinect for obstacle detection.
    This code assumes that the kinect is running in order for Merry to detect obstacles. Otherwise, the code will
    assume that no obstacles are present, and plan arm motions based on a hybrid real-time RRT-JT described in the paper
    by Shaun Howard titled <INSERT TITLE HERE>. To stop the program, just use ctrl-c in the terminal.
    """

    def __init__(self, left_speed=0.5, right_speed=0.5, accuracy=0.1):
        """
        Constructor for Merry. The user may provide the left arm max speed, the right arm max speed, and the
        accuracy level of reaching the goal. The speeds are in meters per second (m/s) and the accuracy level
        is in meters (m). The accuracy level is how close Merry will come to the object to grasp before reaching
        the goal point and stopping to execute the next task.
        :param left_speed: the max speed in m/s that the left arm can reach
        :param right_speed: the max speed in m/s that the right arm can reach
        :param accuracy: the min threshold value for completing the current planning goal of the end effector
        """

        rospy.init_node("merry")

        # wait for robot to boot up
        rospy.sleep(5)

        # Create baxter_interface limb instances for right and left arms
        self.right_arm = baxter_interface.Limb("right")
        self.left_arm = baxter_interface.Limb("left")
        self.joint_states = dict()

        # Parameters which will describe planning parameters for arm max speed and accuracy level of reaching goal
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

        # subscribe to the joint states and store up to 10 messages
        self.joint_states_subscriber = rospy.Subscriber("robot/joint_states", JointState, self.joint_state_cb,
                                                        queue_size=10)

        # subscribe to the goal marker for planning with both end effectors
        self.marker_subscriber = rospy.Subscriber("/merry_end_point_markers/feedback", InteractiveMarkerFeedback,
                                                  self.interactive_marker_cb, queue_size=5)
        self.left_rrt = None

        self.right_rrt = None

        # subscribe to the obstacle list for both right and left arms
        self.left_obs_sub = rospy.Subscriber("left_arm_obstacles", PointCloud, self.left_obs_cb, queue_size=10)
        self.right_obs_sub = rospy.Subscriber("right_arm_obstacles", PointCloud, self.right_obs_cb, queue_size=10)

        # instantiate k-means clusterer for obstacle avoidance and planning purposes, really done in obstacle callback
        self.kmeans = None

        # create kinematics solvers for both the left and right arms
        self.left_kinematics = KDLIKSolver("left")

        self.right_kinematics = KDLIKSolver("right")

        # TODO
        # setup a clean shutdown process that will properly stop the robot upon termination
        rospy.on_shutdown(self.clean_shutdown)

        # initialize goal, obstacle and planner variables
        self.left_goal = None

        self.left_goal_arr = None

        self.right_goal = None

        self.right_goal_arr = None

        self.left_obstacles = []

        self.right_obstacles = []

    def unpack_obstacle_points(self, data, side):
        """
        Unpacks the point cloud points from the received message data dictionary into
        the provided side's 3xN obstacle matrix in this class.
        Left side obstacle points are added to left_obstacles and right side obstacle points
        are added to right_obstacles.
        """
        print "updating obstacles!!!!!"
        points_unpacked = []
        for point in data.points:
            points_unpacked.append((point.x, point.y, point.z))
        point_mat = np.mat(points_unpacked)
        if side == "left":
            self.left_obstacles = point_mat
        else:
            self.right_obstacles = point_mat
        print "obstacles updated..."

    def left_obs_cb(self, data):
        """
        Unpacks the point cloud obstacle points for the left arm into left_obstacles
        and updates the left rrt obstacles.
        :param data: the point cloud message data to unpack from the kinect transformer
        """
        self.unpack_obstacle_points(data, "left")
        self.update_rrt_obstacles("left")

    def right_obs_cb(self, data):
        """
        Unpacks the point cloud obstacle points for the right arm into right_obstacles
        and updates the right rrt obstacles.
        :param data: the point cloud message data to unpack from the kinect transformer
        """
        self.unpack_obstacle_points(data, "right")
        self.update_rrt_obstacles("right")

    def execute_joint_angles(self, joint_positions, side, use_move):
        """
        Moves or sets the joint positions for the provided side arm using
        the initialized max side speed for the arm to reach. The move or set
        position is determined by the use_move boolean. The move process will
        smooth the trajectory of the arm, whereas the set position process will
        have more rigid movement.
        :param joint_positions: the 7 joint position dictionary in format <JOINT_NAME>: <JOINT_VALUE>
        :param side: the side arm to move the joints to
        :param use_move:
        :return: the status of the operation, 0 if success, 1 if error due to data type, length or names in joint dict
        """
        # check type of joint positions and if any joint names were mis-given due to number or name
        if type(joint_positions) is dict and len(joint_positions.keys()) == 7:
            for joint_name in joint_positions.keys():
                if side not in joint_name:
                    # return error status
                    return 1
        if side is "right":
            self.right_arm.set_joint_position_speed(self.right_speed)
            if use_move:
                # wait for 8 seconds until move should be complete
                self.right_arm.move_to_joint_positions(joint_positions, timeout=2)
            else:
                self.right_arm.set_joint_positions(joint_positions)
            self.right_arm.set_joint_position_speed(0.0)
        elif side is "left":
            self.left_arm.set_joint_position_speed(self.left_speed)
            if use_move:
                self.left_arm.move_to_joint_positions(joint_positions)
            else:
                self.left_arm.set_joint_positions(joint_positions)
            self.left_arm.set_joint_position_speed(0.0)
        return 0

    def freeze_joint_angles_in_error(self):
        self.right_arm.set_joint_position_speed(0.0)
        self.left_arm.set_joint_position_speed(0.0)
        rospy.logerr("Joint angles are unavailable at the moment. \
                      Will try to get goal angles soon, but staying put for now.")
        return 1

    def move_to_joint_positions(self, joint_positions, side, use_move=True):
        """
        Moves the Baxter robot end effector to the given dict of joint velocities keyed by joint name.
        :return: a status about the move execution; in success, "OK", and in error, "ERROR".
        """
        rospy.sleep(1.0)
        # Set joint position speed ratio for execution
        if not rospy.is_shutdown() and joint_positions is not None:
            status = self.execute_joint_angles(joint_positions, side, use_move)
        else:
            status = self.freeze_joint_angles_in_error()
        return status

    def check_and_execute_goal_angles(self, goal_angles, side):
        """
        Determines if the joint goal angles are none and then
        tries to move to them at the initialized max side arm speed.
        :param goal_angles: the goal angles dictionary in format <JOINT_NAME>:<JOINT_VALUE>
        :param side: side arm to use, i.e. left or right
        :return: the status of the operation, 1 is error, 0 is success
        """
        status = 1
        if goal_angles is not None:
            rospy.loginfo("got joint angles to execute!")
            rospy.loginfo("goal angles: " + str(goal_angles))
            joint_positions = goal_angles
            # joint_positions = self.get_angles_dict(goal_angles, side)
            print joint_positions
            # set the goal joint angles to reach the desired goal
            status = self.move_to_joint_positions(joint_positions, side, True)
            if status == 1:
                rospy.logerr("could not set joint positions for ik solver...")
        return status

    def get_goal_point(self, side):
        """
        Returns a 7x1 goal point numpy vector for the provided side arm
        :param side: the left or right side to get point for
        :return: the 7x1 goal point vector or None if the goal isn't set
        """
        if side is "left":
            goal = self.left_goal
        else:
            goal = self.right_goal
        if goal:
            return h.point_to_3x1_vector(goal.position)
        else:
            return None

    def get_goal_pose(self, side):
        """
        Gets the pose for the goal of the specified side
        :param side: the left or right side arm to get goal of
        :return: the goal pose for the given side or None if the goal is not set
        """
        if side is "left":
            goal = self.left_goal
        else:
            goal = self.right_goal
        if goal:
            return goal
        else:
            return None

    def get_obs_for_side(self, side):
        """
        Gets obstacle matrix for the provided side.
        :param side: the side arm to get obstacles for
        :return: the side arm obstacle matrix
        """
        # TODO, filter left arm for left cloud, filter right arm for right cloud
        if side == "left":
            return self.left_obstacles
        else:
            return self.right_obstacles

    def update_rrt_obstacles(self, side):
        """
        Updates the obstacle matrix for the specified side rrt planner.
        :param side: the left or right side
        """
        if side == "left" and self.left_rrt is not None:
            self.left_rrt.update_obstacles(self.left_obstacles)
        elif side == "right" and self.right_rrt is not None:
            self.right_rrt.update_obstacles(self.right_obstacles)

    def get_kin(self, side):
        """
        Gets the kinematics solver for the specified side.
        :param side: the left or right side arm
        :return: the kinematics sovler instance for the given side
        """
        if side == "left":
            return self.left_kinematics
        else:
            return self.right_kinematics

    def prune_closest_node(self, rrt):
        return rrt.prune_closest_node()

    def grow(self, rrt, dist_thresh, p_goal, max_retries=10000):
        """
        Grows and executes an online hybrid RRT-JT/Randomized IK planner
        for the given rrt instance (left or right arm).
        Applies the JT goal approach method 1-p_goal * 100 percent of the time,
        while approaching the goal with a randomized IK planner p_goal * 100 percent of the time.
        :param rrt: the rrt instance to grow
        :param dist_thresh: the minimum distance threshold to reach from goal position for each goal
        :param p_goal: the probability of approach the goal using a randomized IK planner
        :param max_retries: the number of times to retry when not making progress in hopes of random hill climb/restart
        :return: the finalized rrt with all nodes after they have been executed
        """
        if rrt is not None:
            curr_dist_to_goal = rrt.closest_node_to_goal(check_if_picked=False)[0]
            times_retried = 0
            last_dist_to_goal = 1000
            while curr_dist_to_goal > dist_thresh and times_retried <= max_retries:
                rospy.loginfo("growing rrt...")
                curr_dist_to_goal = rrt.closest_node_to_goal(check_if_picked=False)[0]
                if curr_dist_to_goal < last_dist_to_goal:
                    print "current distance to goal (m): " + str(curr_dist_to_goal)
                    p = random.uniform(0, 1)
                    if p < p_goal:
                        rospy.loginfo("using jacobian extend")
                        rrt.extend_toward_goal(dist_thresh)
                    else:
                        rospy.loginfo("using ik random extend")
                        pos = self.left_arm.endpoint_pose()["position"]
                        rrt.ik_extend_randomly(np.array(pos), dist_thresh)
                else:
                    # back-track to last closest node
                    pruned_closest_node = self.prune_closest_node(rrt)
                    if pruned_closest_node:
                        # have possibility of making more progress
                        continue
                    else:
                        # not going to get any closer to goal most likely
                        break
                times_retried += 1
            rrt.cleanup_nodes()
        return rrt

    def grow_rrt(self, side, q_start, goal_pose, dist_thresh=0.05, p_goal=0.5):
        """
        Grows the rrt for the specified side of the robot. It starts planning from the q_start 7x1 joint angle vector
        and plans to the given goal pose. The distance threshold for accuracy to reach goal state and the probability
        of using a straight-line randomized IK planner to approach the goal may be given as parameters.
        :param side: the left or right arm side to plan for
        :param q_start: the 7x1 start goal angles vector in order from base to end effector
        :param goal_pose: the goal pose to reach for the side arm
        :param dist_thresh: the distance threshold in meters that determines if the goal pose has been meet
        :param p_goal: the probability of using a straight-line randomized IK planner to approach the goal
        :return: the rrt resulting from the planning session
        """
        obs = self.get_obs_for_side(side)
        if side == "left":
            self.left_rrt = RRT(q_start, goal_pose, self.get_kin(side), side, self.left_arm.joint_names(), obs,
                                self.check_and_execute_goal_angles)
            self.grow(self.left_rrt, dist_thresh, p_goal)
            return self.left_rrt
        else:
            self.right_rrt = RRT(q_start, goal_pose, self.get_kin(side), side, self.right_arm.joint_names(), obs,
                                 self.check_and_execute_goal_angles)
            self.grow(self.right_rrt, dist_thresh, p_goal)
            return self.right_rrt

    def approach_left_goal(self):
        last_goal = None
        execute_new_nodes = False
        while not rospy.is_shutdown():
            left_rrt = None
            if np.array_equal(last_goal, self.left_goal):
                # continue if goal already met
                continue
            if self.left_goal is not None:
                # print "left goal: " + str(self.left_goal)
                left_joint_angles = [self.left_arm.joint_angle(name) for name in self.left_arm.joint_names()]
                left_rrt = self.grow_rrt("left", left_joint_angles, self.left_goal)
                last_goal = self.left_goal
                if len(left_rrt.nodes) > 0:
                    execute_new_nodes = True
            if left_rrt and execute_new_nodes:
                print "left rrt created with %d nodes" % len(left_rrt.nodes)
                for node in left_rrt.nodes:
                    print "approaching new left node..."
                    print "[" + ", ".join([str(x) for x in node]) + "]"
                    node_angle_dict = h.wrap_angles_in_dict(node, self.left_arm.joint_names())
                    self.check_and_execute_goal_angles(node_angle_dict, "left")
                    print "reached new left node destination..."
                execute_new_nodes = False
                print "reached left goal"
                # rospy.loginfo("met left goal")

            self.left_arm.set_joint_position_speed(0.0)

    def approach_right_goal(self):
        last_goal = None
        execute_new_nodes = False
        while not rospy.is_shutdown():
            right_rrt = None
            if np.array_equal(last_goal, self.right_goal):
                continue
            if self.right_goal is not None:
                # print "right goal: " + str(self.right_goal)
                right_joint_angles = [self.right_arm.joint_angle(name) for name in self.right_arm.joint_names()]
                right_rrt = self.grow_rrt("right", right_joint_angles, self.right_goal)
                last_goal = self.right_goal
                if len(right_rrt.nodes) > 0:
                    execute_new_nodes = True
            if right_rrt and execute_new_nodes:
                print "right rrt created with %d nodes" % len(right_rrt.nodes)
                for node in right_rrt.nodes:
                    print "approaching new right node..."
                    print "[" + ", ".join([str(x) for x in node]) + "]"
                    node_angle_dict = h.wrap_angles_in_dict(node, self.right_arm.joint_names())
                    self.check_and_execute_goal_angles(node_angle_dict, "right")
                    print "reached new right node destination..."
                execute_new_nodes = False
                print "reached right goal"
                # rospy.loginfo("met right goal")

            self.right_arm.set_joint_position_speed(0.0)

    def approach_goals(self):
        """
        A multi-threaded process that plans for both left and right arms.
        Each arm uses a hybrid RRT-JT/Randomized IK planner that will essentially loop forever.
        It receives the latest goal and plans for that goal once it is received.
        The process can be killed with ctrl-c and the ros.on_shutdown callback should be set to stop the robot joints
        if this happens.
        """
        t_left = threading.Thread(target=self.approach_left_goal)
        t_left.start()
        t_right = threading.Thread(target=self.approach_right_goal)
        t_right.start()
        threads = [t_left, t_right]
        for thread in threads:
            thread.join()
        return 0

    def approach_single_goal(self, side, kin):
        """
        Attempts to successfully visit the goal point using the straight-line IK solver. This will currently
        loop forever but will receive the latest goal and retry for solutions on-demand when dynamic-planning is desired
        :param side: the arm side to use
        :param kin: the KDLIKSolver instance
        :return: status of attempt to approach the goal point
        """
        status = 0
        # approach the goal points
        goal_met = False
        while goal_met is False and status == 0:
            goal = self.get_goal_pose(side)
            # obstacles = None
            # obstacles = h.get_critical_points_of_obstacles(self)
            # rospy.sleep(1)
            goal_angles = None
            if goal:
                # curr_angles = [self.left_arm.joint_angle(n) for n in self.left_arm.joint_names()]
                goal_angles = kin.solve(position=goal.position, orientation=goal.orientation)
                if goal_angles is not None:
                    goal_met = True
                else:
                    goal_met = False
            else:
                goal_met = True

            # execute goal angles if they are available
            if goal_angles is not None:
                angles_dict = h.wrap_angles_in_dict(goal_angles, self.left_arm.joint_names())
                # do left arm planning
                status = self.check_and_execute_goal_angles(angles_dict, side)
        print "found goal solution"
        return status

    def joint_state_cb(self, data):
        """
        Callback for joint states that store position, velocity and effort values for each named joint on the robot.
        A joint_states dictionary is created in this class. The mapping is as follows:
            joint_states = dict
            joint_states[JOINT_NAME][<position, velocity, or effort string>] = VALUE
        :param data: the joint state data received from Baxter
        """
        names = data.name
        positions = data.position
        velocities = data.velocity
        efforts = data.effort
        for i in range(len(names)):
            self.joint_states[names[i]] = dict()
            self.joint_states[names[i]]["position"] = positions[i]
            self.joint_states[names[i]]["velocity"] = velocities[i]
            self.joint_states[names[i]]["effort"] = efforts[i]

    def interactive_marker_cb(self, feedback):
        """
        Callback for the interactive marker feedback data including the latest pose of the marker.
        This will determine if the feedback is from the left or right marker given the name,
        then the left or right goal will be updated, i.e. left_goal. Subsequently, the left or right goal array,
        aka 7x1 numpy goal vector, will be set, i.e. left_goal_arr. Then, the left or right rrt is updated with
        the latest goal that side.
        :param feedback: the Pose message from the interactive marker
        """
        # store feedback pose as goal
        goal = feedback.pose

        # set right or left goal depending on marker name
        if "right" in feedback.marker_name:
            self.right_goal = goal
            self.right_goal_arr = h.pose_to_7x1_vector(goal)
            if self.right_rrt:
                self.right_rrt.update_goal(goal)
        elif "left" in feedback.marker_name:
            self.left_goal = goal
            self.left_goal_arr = h.pose_to_7x1_vector(goal)
            if self.left_rrt:
                self.left_rrt.update_goal(goal)
        else:
            rospy.loginfo("got singular end-point goal")

    def get_joint_angles(self, side="right"):
        """Gets the joint angle dictionary of the specified arm side."""
        joint_angles = self.right_arm.joint_angles() if side is "right" else self.left_arm.joint_angles()
        return joint_angles

    def get_joint_velocities(self, side="right"):
        """Gets the joint velocity dictionary of the specified arm side."""
        joint_velocities = self.right_arm.joint_velocities() if side is "right" else self.left_arm.joint_velocities()
        return joint_velocities

    def get_angles_dict(self, angles_list, side):
        """
        Gets the joint angles dictionary from the provided ordered angles list from base to end effector by using
        the Baxter joint names list for that side and mapping the contents of the list to the dictionary by name.
        :param angles_list: the ordered list of angles from base to end effector
        :param side: the side arm to map the angles list to in a dictionary
        """
        joint_names = self.right_arm.joint_names() if side is "right" else self.left_arm.joint_names()
        angles_dict = dict()
        if len(joint_names) == len(angles_list):
            i = 0
            for name in joint_names:
                angles_dict[name] = angles_list[i]
                i += 1
        return angles_dict

    def clean_shutdown(self):
        """
        The clean shutdown method, important for stopping the robot arms from moving after shutdown.
        :return: a placeholder True value for a more advanced error-handling system
        """
        print("\nExiting example...")
        if not self._init_state:
            print("stopping robot...")
            self.left_arm.set_joint_position_speed(0)
            self.right_arm.set_joint_position_speed(0)
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
