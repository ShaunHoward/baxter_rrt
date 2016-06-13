#!/usr/bin/env python

__author__ = 'shaun howard'

import argparse
import baxter_interface
import rospy
import sys

from merry import Merry


def get_args():
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=""""
                                     An interface to the CWRU robotics Merry robot for potential field motion planning.
                                     For help on usage, refer to the github README @ github.com/ShaunHoward/potential_\
                                     fields.""")
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '-l', '--limb', required=False, choices=['left', 'right'], default='right',
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


def shake_hands(merry):
    """
    Tries to shake the nearest hand possible using the limb instantiated.
    Loop runs forever; kill with ctrl-c.
    """
    if merry.plan_and_execute_end_effector("right") is "OK":
        return 0
    return 1


def run_robot(args):
    print("Initializing node... ")
    rospy.init_node("merry")
    merry = Merry(args.limb, args.speed, args.accuracy)
    rospy.on_shutdown(merry.clean_shutdown)
    status = shake_hands(merry)
    sys.exit(status)


if __name__ == '__main__':
    run_robot(get_args())
