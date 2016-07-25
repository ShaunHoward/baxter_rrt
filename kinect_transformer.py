import helpers as h
import math
import numpy as np
import rospy
import std_msgs.msg
import tf

from geometry_msgs.msg import Point32
from sensor_msgs.msg import PointCloud
from sensor_msgs import point_cloud2 as pc2
from solver.ik_solver import KDLIKSolver
from planner.collision_checker import CollisionChecker

__author__ = "Shaun Howard (smh150@case.edu)"


class KinectTransformer:
    # MAKE SURE KINECT STARTS WITH BAXTER_WORLD LAUNCH

    def __init__(self):
        rospy.init_node("kinect_transformer")
        self.kinect_depth_sub = rospy.Subscriber("kinect/depth/points", pc2.PointCloud2, self.kinect_cb, queue_size=10)
        self.left_obs_pub = rospy.Publisher("left_arm_obstacles", PointCloud, queue_size=10)
        self.right_obs_pub = rospy.Publisher("right_arm_obstacles", PointCloud, queue_size=10)
        self.tf = tf.TransformListener()
        self.closest_points = []
        # create kinematics solvers for both the left and right arms
        self.left_kinematics = KDLIKSolver("left")
        self.right_kinematics = KDLIKSolver("right")
        # create collision checkers with the left and right kin solver instances
        self.left_cc = CollisionChecker([], self.left_kinematics)
        self.right_cc = CollisionChecker([], self.right_kinematics)

    def filter_out_left_arm(self, points):
        # TODO use color points as filter

        return points

    def filter_out_right_arm(self, points):
        return points

    def build_and_publish_obstacle_point_clouds(self, points):
        obstacle_cloud = PointCloud()
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'kinect_pc_frame'
        obstacle_cloud.header = header
        left_filtered_pts = self.filter_out_left_arm(points)
        # update collision checker obstacle list
        self.left_cc.update_obstacles(left_filtered_pts)
        for point in left_filtered_pts:
            obstacle_cloud.points.append(Point32(point[0], point[1], point[2]))
        print "publishing new left obstacle cloud!"
        self.left_obs_pub.publish(obstacle_cloud)

        obstacle_cloud = PointCloud()
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'kinect_pc_frame'
        obstacle_cloud.header = header
        right_filtered_pts = self.filter_out_right_arm(points)
        # update collision checker obstacle list
        self.right_cc.update_obstacles(right_filtered_pts)
        for point in right_filtered_pts:
            obstacle_cloud.points.append(Point32(point[0], point[1], point[2]))
        print "publishing new right obstacle cloud!"
        self.right_obs_pub.publish(obstacle_cloud)

    def kinect_cb(self, data, source="kinect_pc_frame", dest="base", min_dist=0.1, max_dist=1.5, min_height=0.4):
        """
        Receives kinect points from the kinect subscriber linked to the publisher stream.
        :return: kinect points numpy array
        """
        # TODO add left and right arm points to filter out of published "obstacles" per side
        points = [p for p in pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z"))]
        print (points[0])
        print (points[1])
        transformed_points = h.transform_pcl2(self.tf, dest, source, points)
        points_list = []
        for p in transformed_points:
            dist = math.sqrt(p.x ** 2 + p.y ** 2 + p.z ** 2)
            if min_dist < dist < max_dist and p.z >= min_height:
                points_list.append((dist, p))
        sorted_pts = np.sort(np.array(points_list), 0)
        self.closest_points = [point for dist, point in sorted_pts]
        self.closest_points = points
        self.build_and_publish_obstacle_point_clouds(self.closest_points)
        if len(self.closest_points) > 0:
            print "kinect cb: there are this many close points: " + str(len(self.closest_points))

if __name__ == "__main__":
    KinectTransformer()
    rospy.spin()
