# potential_fields
Merry the robot with potential fields for joint control for EECS 499 Research Project.

# install rospy
* add it to your shell path

# install Cython, PCL for Python
* sudo apt-get install python-dev build-essential
* Download latest release from http://cython.org and untar it to ~/libraries
* cd ~/libraries/Cython-0.xx
* python setup.py install --user
* cd ~/libraries/
* git clone https://github.com/strawlab/python-pcl.git
* cd python-pcl
* python setup.py install --user

# install Kinect Libraries
* sudo apt-get install python-dev python-numpy
* opencv: https://help.ubuntu.com/community/OpenCV
* cd ~/libraries/
* sudo apt-get install python-freenect
* if above doesn't work, these are supposed to work but usually throw errors:
* -- git clone https://github.com/OpenKinect/libfreenect.git
* -- cd libfreenect/wrappers/python
* -- python setup.py build_ext --inplace

# install MoveIt!
* sudo add-apt-repository ppa:libccd-debs/ppa
* sudo apt-get update
* sudo apt-get install libccd-dev
* sudo apt-get install ros-indigo-moveit-full
* cd ~/ros_ws/src
* git clone https://github.com/ros-planning/moveit_robots.git

# run baxter:
* In all terminals with baxter nodes running:
* cd ~/projects/ros_ws/
* ./baxter.sh

* For gazebo simulator: roslaunch cwru_baxter_sim baxter_world.launch
* For motion planning: roslaunch baxter_moveit_config move_group.launch
* For visualization: rviz rviz

# Extract raw data from Kinect
* http://answers.ros.org/question/9803/extracting-raw-data-from-kinect/

now, in python you can:
from roslib import message
import sensor_msgs.point_cloud2 as pc2
import freenect
import pcl
