import math
import numpy as np
import random
import rospy
import scipy


def calculate_cartesian_jacobian(joint_angles):
    height = 6
    width = 7

    j0 = np.zeros((height, width))
    c1 = np.cos(joint_angles[0])
    c2 = np.cos(joint_angles[1])
    c3 = np.cos(joint_angles[2])
    c4 = np.cos(joint_angles[3])
    c5 = np.cos(joint_angles[4])
    c6 = np.cos(joint_angles[5])

    s1 = np.sin(joint_angles[0])
    s2 = np.sin(joint_angles[1])
    s3 = np.sin(joint_angles[2])
    s4 = np.sin(joint_angles[3])
    s5 = np.sin(joint_angles[4])
    s6 = np.sin(joint_angles[5])

    L2 = 0.37082  # may need to change this to 0.36435
    L3 = 0.37429
    L4 = 0.229525

    j11 = c1 * ((L3 + L4 * c6) * s3 * s4 + L4 * (c4 * c5 * s3 + c3 * c5) * s6) \
        + s1 * s2 * (L2 + (c4 * (L3 + L4 * c6)) - (L4 * c5 * s4 * s6)) \
        + c2 * (c3 * (L3 + L4 * c6) * s4 + L4 * (c3 * c4 * c5 - s3 * s5 * s6))

    j12 = c1 * (-1 * c2 * (L2 + c4 * (L3 + L4 * c6) - L4 * c5 * s4 * s6)
        + s2 * (c3 * (L3 + L4 * c6) * s4 + L4 * (c3 * c4 * c5 - s3 * s4) * s6))

    j13 = (L3 + L4 * c6) * (c3 * s1 + c1 * c2 * s3) * s4 \
        + L4 * (c4 * c5 * (c3 * s1 + c1 * c2 * s3) + (c1 * c2 * c3 - s1 * s3) * s5) * s6

    j14 = (L3 + L4 * c6) * (c4 * s1 * s3 + c1 * (-1 * c2 * c3 * c4 + s2 * s4)) \
        + L4 * c5 * (-1 * s1 * s3 * s4 + c1 * (c4 * s2 + c2 * c3 * s4)) * s6

    j15 = L4 * (c5 * (c3 * s1 + c1 * c2 * s3)
        + (-c4 * s1 * s3 + c1 * (c2 * c3 * c4 - s2 * s4)) * s5) * s6

    j16 = L4 * (s1 * (c6 * (c4 * c5 * s3 + c3 * s5) - s3 * s4 * s6)
        + c1 * (c6 * (c5 * s2 * s4 + c2 * (-1 * c3 * c4 * c5 + s3 * s5))
        + (c4 * s2 + c2 * c3 * s4) * s6))

    j17 = 0

    j21 = s1 * ((L3 + L4 * c6) * s3 * s4 + L4 * (c4 * c5 * s3 + c3 * s5) * s6) \
        - c1 * (s2 * (L2 + c4 * (L3 + L4 * c6) - L4 * c5 * s4 * s6)
        + c2 * (c3 * (L3 + L4 * c6) * s4 + L4 * (c3 * c4 * c5 - s3 * s5) * s6))

    j22 = s1 * (-1 * c2 * (L2 + c4 * (L3 + L4 * c6) - L4 * c5 * s4 * s6) + s2 * (c3 * (L3 + L4 * c6) * s4 + L4 * (c3 * c4 * c5 - s3 * s5) * s6))

    j23 = -1 * (L3 + L4 * c6) * (c1 * c3 - c2 * s1 * s3) * s4 + L4 * (c2 * s1 * (c4 * c5 * s3 + c3 * s5) + c1 * (-1 * c3 * c4 * c5 + s3 * s5)) * s6

    j24 = -1 * (L3 + L4 * c6) * (c4 * (c2 * c3 * s1 + c1 * s3) - s1 * s2 * s4) + L4 * c5 * (c4 * s1 * s2 + (c2 * c3 * s1 + c1 * s3) * s4) * s6

    j25 = L4 * (c1 * (-1 * c3 * c5 + c4 * s3 * s5) + s1 * (-1 * s2 * s4 * s5 + c2 * (c5 * s3 + c3 * c4 * s5))) * s6

    j26 = L4 * (c6 * (c5 * (-1 * c4 * (c2 * c3 * s1 + c1 * s3) + s1 * s2 * s4) + (-1 * c1 * c3 + c2 * s1 * s3) * s5)
        + (c4 * s1 * s2 + (c2 * c3 * s1 + c1 * s3) * s4) * s6)

    j27 = 0

    j31 = 0

    j32 = -1 * s2 * (L2 + c4 * (L3 + L4 * c6) - L4 * c5 * s4 * s6) \
        + c2 * (L4 * s3 * s5 * s6 - c3 * ((L3 + L4 * c6) * s4 + L4 * c4 * c5 * s6))

    j33 = s2 * ((L3 + L4 * c6) * s3 * s4 + L4 * (c4 * c5 * s3 + c3 * s5) * s6)

    j34 = -1 * (L3 + L4 * c6) * (c3 * c4 * s2 + c2 * s4) + L4 * c5 * (-1 * c2 * c4 + c3 * s2 * s4) * s6

    j35 = L4 * (c5 * s2 * s3 + (c3 * c4 * s2 + c2 * s4) * s5) * s6

    j36 = -1 * L4 * (c6 * (c5 * (c3 * c4 * s2 + c2 * s4) - s2 * s3 * s5) + (c2 * c4 - c3 * s2 * s4) * s6)

    j37 = 0

    j41 = 0

    j42 = s1

    j43 = -c1 * s2

    j44 = c3 * s1 + c1 * c2 * s3

    j45 = s1 * s3 * s4 - c1 * (c4 * s2 + c2 * c3 * s4)

    j46 = c5 * (c3 * s1 + c1 * c2 * s3) \
        + (-1 * c4 * s1 * s3 + c1 * (c2 * c3 * c4 - s2 * s4)) * s5

    j47 = s1 * (c6 * s3 * s4 + (c4 * c5 * s3 + c3 * s5) * s6) \
        + c1 * (-1 * c6 * (c4 * s2 + c2 * c3 * s4)
        + (c5 * s2 * s4 + c2 * (-1 * c3 * c4 * c5 + s3 * s5)) * s6)

    j51 = 0

    j52 = c1

    j53 = -s1 * s2

    j54 = -1 * c1 * c3 + c2 * s1 * s3

    j55 = -1 * c4 * s1 * s2 - (c2 * c3 * s1 + c1 * s3) * s4

    j56 = c1 * (-1 * c3 * c5 + c4 * s3 * s5) \
        + s1 * (-1 * s2 * s4 * s5 + c2 * (c5 * s3 + c3 * c4 * s5))

    j57 = -1 * c6 * (c4 * s1 * s2 + (c2 * c3 * s1 + c1 * s3) * s4) \
        - (c5 * (c4 * (c2 * c3 * s1 + c1 * s3) - s1 * s2 * s4) + (c1 * c3 - c2 * s1 * s3) * s5) * s6

    j61 = 1

    j62 = 0

    j63 = c2

    j64 = s2 * s3

    j65 = c2 * c4 - c3 * s2 * s4

    j66 = c5 * s2 * s3 + (c3 * c4 * s2 + c2 * s4) * s5

    j67 = c2 * (c4 * c6 - c5 * s4 * s6) + s2 * (s3 * s5 * s6 - c3 * (c6 * s4 + c4 * c5 * s6))

    j0[0][0] = j11
    j0[0][1] = j12
    j0[0][2] = j13
    j0[0][3] = j14
    j0[0][4] = j15
    j0[0][5] = j16
    j0[0][6] = j17

    j0[1][0] = j21
    j0[1][1] = j22
    j0[1][2] = j23
    j0[1][3] = j24
    j0[1][4] = j25
    j0[1][5] = j26
    j0[1][6] = j27

    j0[2][0] = j31
    j0[2][1] = j32
    j0[2][2] = j33
    j0[2][3] = j34
    j0[2][4] = j35
    j0[2][5] = j36
    j0[2][6] = j37

    j0[3][0] = j41
    j0[3][1] = j42
    j0[3][2] = j43
    j0[3][3] = j44
    j0[3][4] = j45
    j0[3][5] = j46
    j0[3][6] = j47

    j0[4][0] = j51
    j0[4][1] = j52
    j0[4][2] = j53
    j0[4][3] = j54
    j0[4][4] = j55
    j0[4][5] = j56
    j0[4][6] = j57

    j0[5][0] = j61
    j0[5][1] = j62
    j0[5][2] = j63
    j0[5][3] = j64
    j0[5][4] = j65
    j0[5][5] = j66
    j0[5][6] = j67
    return j0


OK = "OK"
ERROR = "ERROR"


def plan(goal, obstacles, end_effector_velocities, joint_angles, side, avoid_velocity=0.3):
    # validate parameters
    params = [goal, obstacles, end_effector_velocities, joint_angles, side, avoid_velocity]
    status = OK
    for param in params:
        if param is None:
            status = ERROR
            break
    if status == ERROR:
        rospy.logerr("Could not develop plan for Merry. No obstacles were found or a parameter was empty.")
        return status, None

    d_m = 2
    height = 6
    width = 7
    v0 = avoid_velocity

    goal_joint_velocities = np.zeros((len(joint_angles)))
    q_dot = np.zeros((len(joint_angles)))

    x_dot = np.array([end_effector_velocities.linear.x,
                      end_effector_velocities.linear.y,
                      end_effector_velocities.linear.z,
                      end_effector_velocities.angular.x,
                      end_effector_velocities.angular.y,
                      end_effector_velocities.angular.z])

    if len(obstacles) == 0:
        # do simple planning, tracking is more precise
        j0 = calculate_cartesian_jacobian(joint_angles)
        q_dot = np.dot(np.linalg.pinv(j0), x_dot)
    else:
        # otherwise, do more complicated planning
        for ob_critical_pt, ob_dist in obstacles:
            if ob_critical_pt is None:
                continue
            d0 = np.array([ob_critical_pt.x, ob_critical_pt.y, ob_critical_pt.z])
            d0_norm = np.linalg.norm(d0)
            n0 = d0 / d0_norm
            n0_t = n0.transpose()
            j0 = calculate_cartesian_jacobian(joint_angles)
            j_d0 = np.dot(n0_t, j0)

            a1 = np.zeros((height, width))
            for m in range(height):
                for n in range(width):
                    if d0_norm >= d_m:
                        a1[m][n] = (d_m / d0_norm) ^ n
                    else:
                        a1[m][n] = 2 * d_m / d_m + d0_norm

            a2 = np.zeros((height, width))
            for m in range(height):
                for n in range(width):
                    if d0_norm >= d_m:
                        a2[m][n] = (d_m / d0_norm) ^ n
                    else:
                        a2[m][n] = 1

            I = np.eye(height, width)
            N_prime_0 = I - np.dot(a2, np.linalg.pinv(j_d0), j_d0)
            x_dot_d0 = a1 * v0
            q_dot = np.dot(np.linalg.pinv(j_d0), x_dot_d0) + np.dot(N_prime_0, np.linalg.pinv(j0), x_dot)
            break

    if q_dot is not None and len(q_dot) == len(joint_angles) != 0:
        goal_joint_velocities = q_dot
    return status, goal_joint_velocities
