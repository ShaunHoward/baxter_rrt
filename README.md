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

# Extract raw data from Kinect
* http://answers.ros.org/question/9803/extracting-raw-data-from-kinect/

now, in python you can:
from roslib import message
import sensor_msgs.point_cloud2 as pc2
import freenect
import pcl