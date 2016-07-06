#! /bin/bash

# startup baxter simulator
roslaunch cwru_baxter_sim baxter_world.launch &

sleep 5

# startup rviz
rosrun rviz rviz &

sleep 5

# startup interactive marker script
python ./interactive_marker/3d_marker.py

# run robot including motion planner
# python merry.py