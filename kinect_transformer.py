import helpers as h
import math
import numpy as np
import rospy
import std_msgs.msg
import tf

from geometry_msgs.msg import Point32
from sensor_msgs.msg import PointCloud
from sensor_msgs import point_cloud2 as pc2


class KinectTransformer:

    def __init__(self):
        rospy.init_node("kinect_transformer")
        self.tf = tf.TransformListener()
        self.kinect_sub = rospy.Subscriber("kinect/depth/points", pc2.PointCloud2, self.kinect_cb, queue_size=10)
        self.left_obs_pub = rospy.Publisher("left_arm_obstacles", PointCloud, queue_size=10)
        self.right_obs_pub = rospy.Publisher("right_arm_obstacles", PointCloud, queue_size=10)
        self.closest_points = []

    def build_point_cloud(self, points):
        obstacle_cloud = PointCloud()
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'base'
        obstacle_cloud.header = header
        for point in points:
            obstacle_cloud.points.append(Point32(point[0], point[1], point[3]))
        print "publishing new obstacle clouds!"
        self.left_obs_pub.publish(obstacle_cloud)
        self.right_obs_pub.publish(obstacle_cloud)

    def kinect_cb(self, data, source="kinect_link", dest="base", min_dist=0.1, max_dist=2, min_height=0.4):
        """
        Receives kinect points from the kinect subscriber linked to the publisher stream.
        :return: kinect points numpy array
        """
        # TODO add left and right arm points to filter out of published "obstacles" per side
        points = [p for p in pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z"))]
        transformed_points = h.transform_pcl2(self.tf, dest, source, points, 3)
        self.closest_points = np.array([h.point_to_ndarray(p) for p in transformed_points if
                              min_dist < math.sqrt(p.x ** 2 + p.y ** 2 + p.z ** 2) < max_dist
                              and p.z > min_height])
        self.build_point_cloud(self.closest_points)
        # self.create_obstacle_wave_maps()
        # if self.left_rrt:
        #     self.left_rrt.update_obstacles(self.left_obstacle_waves)
        # if self.right_rrt:
        #     self.right_rrt.update_obstacles(self.right_obstacle_waves)
        if len(self.closest_points) > 0:
            print "kinect cb: there are this many close points: " + str(len(self.closest_points))

if __name__ == "__main__":
    KinectTransformer()
    rospy.spin()
