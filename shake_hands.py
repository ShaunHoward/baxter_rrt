#!/usr/bin/env python

__author__ = 'shaun howard'

import argparse
import rospy
import sys

import baxter_interface

from merry import Merry
from planner import PotentialFieldPlanner


def get_args():
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=""""
                                     An interface to the CWRU robotics Merry robot for potential field motion planning.
                                     For help on usage, refer to the github README @ github.com/ShaunHoward/potential_\
                                     fields.""")
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '-l', '--limb', required=True, choices=['left', 'right'],
        help='limb to record/playback waypoints'
    )
    parser.add_argument(
        '-s', '--speed', default=0.3, type=float,
        help='joint position motion speed ratio [0.0-1.0] (default:= 0.3)'
    )
    parser.add_argument(
        '-a', '--accuracy',
        default=baxter_interface.JOINT_ANGLE_TOLERANCE, type=float,
        help='joint position accuracy (rad) at which waypoints must achieve'
    )
    return parser.parse_args(rospy.myargv()[1:])


def run_robot(args):
    print("Initializing node... ")
    rospy.init_node("merry_pf %s" % (args.limb,))
    merry = Merry(args.limb, args.speed, args.accuracy, PotentialFieldPlanner())
    rospy.on_shutdown(merry.clean_shutdown)
    status = merry.shake_hands()
    sys.exit(status)


def robot_runner():
    run_robot(get_args())


if __name__ == '__main__':
    robot_runner()