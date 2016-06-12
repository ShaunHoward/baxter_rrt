import math
import numpy as np
import random
import scipy


def calculate_cartesian_jacobian(joint_angles):
    height = 6
    width = 7
    J0 = np.zeros((height, width))
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
    J11 = c1 * ((L3 + L4 * c6) * s3 * s4 + L4 * (c4 * c5 * s3 + c3 * c5) * s6) \
        + s1 * s2 * (L2 + (c4 * (L3 + L4 * c6)) - (L4 * c5 * s4 * s6)) \
        + c2 * (c3 * (L3 + L4 * c6) * s4 + L4 * (c3 * c4 * c5 - s3 * s5 * s6))

    J12 = c1 * (-1 * c2 * (L2 + c4 * (L3 + L4 * c6) - L4 * c5 * s4 * s6)
        + s2 * (c3 * (L3 + L4 * c6) * s4 + L4 * (c3 * c4 * c5 - s3 * s4) * s6))

    J13 = (L3 + L4 * c6) * (c3 * s1 + c1 * c2 * s3) * s4 \
        + L4 * (c4 * c5 * (c3 * s1 + c1 * c2 * s3) + (c1 * c2 * c3 - s1 * s3) * s5) * s6

    J14 = (L3 + L4 * c6) * (c4 * s1 * s3 + c1 * (-1 * c2 * c3 * c4 + s2 * s4)) \
        + L4 * c5 * (-1 * s1 * s3 * s4 + c1 * (c4 * s2 + c2 * c3 * s4)) * s6

    J15 = L4 * (c5 * (c3 * s1 + c1 * c2 * s3)
        + (-c4 * s1 * s3 + c1 * (c2 * c3 * c4 - s2 * s4)) * s5) * s6

    J16 = L4 * (s1 * (c6 * (c4 * c5 * s3 + c3 * s5) - s3 * s4 * s6)
        + c1 * (c6 * (c5 * s2 * s4 + c2 * (-1 * c3 * c4 * c5 + s3 * s5))
        + (c4 * s2 + c2 * c3 * s4) * s6))

    J17 = 0

    J21 = s1 * ((L3 + L4 * c6) * s3 * s4 + L4 * (c4 * c5 * s3 + c3 * s5) * s6) \
        - c1 * (s2 * (L2 + c4 * (L3 + L4 * c6) - L4 * c5 * s4 * s6)
        + c2 * (c3 * (L3 + L4 * c6) * s4 + L4 * (c3 * c4 * c5 - s3 * s5) * s6))

    J22 = s1 * (-1 * c2 * (L2 + c4 * (L3 + L4 * c6) - L4 * c5 * s4 * s6) + s2 * (c3 * (L3 + L4 * c6) * s4 + L4 * (c3 * c4 * c5 - s3 * s5) * s6))

    J23 = -1 * (L3 + L4 * c6) * (c1 * c3 - c2 * s1 * s3) * s4 + L4 * (c2 * s1 * (c4 * c5 * s3 + c3 * s5) + c1 * (-1 * c3 * c4 * c5 + s3 * s5)) * s6

    J24 = -1 * (L3 + L4 * c6) * (c4 * (c2 * c3 * s1 + c1 * s3) - s1 * s2 * s4) + L4 * c5 * (c4 * s1 * s2 + (c2 * c3 * s1 + c1 * s3) * s4) * s6

    J25 = L4 * (c1 * (-1 * c3 * c5 + c4 * s3 * s5) + s1 * (-1 * s2 * s4 * s5 + c2 * (c5 * s3 + c3 * c4 * s5))) * s6

    J26 = L4 * (c6 * (c5 * (-1 * c4 * (c2 * c3 * s1 + c1 * s3) + s1 * s2 * s4) + (-1 * c1 * c3 + c2 * s1 * s3) * s5)
        + (c4 * s1 * s2 + (c2 * c3 * s1 + c1 * s3) * s4) * s6)

    J27 = 0

    J31 = 0

    J32 = -1 * s2 * (L2 + c4 * (L3 + L4 * c6) - L4 * c5 * s4 * s6) \
        + c2 * (L4 * s3 * s5 * s6 - c3 * ((L3 + L4 * c6) * s4 + L4 * c4 * c5 * s6))

    J33 = s2 * ((L3 + L4 * c6) * s3 * s4 + L4 * (c4 * c5 * s3 + c3 * s5) * s6)

    J34 = -1 * (L3 + L4 * c6) * (c3 * c4 * s2 + c2 * s4) + L4 * c5 * (-1 * c2 * c4 + c3 * s2 * s4) * s6

    J35 = L4 * (c5 * s2 * s3 + (c3 * c4 * s2 + c2 * s4) * s5) * s6

    J36 = -1 * L4 * (c6 * (c5 * (c3 * c4 * s2 + c2 * s4) - s2 * s3 * s5) + (c2 * c4 - c3 * s2 * s4) * s6)

    J37 = 0

    J41 = 0

    J42 = s1

    J43 = -c1 * s2

    J44 = c3 * s1 + c1 * c2 * s3

    J45 = s1 * s3 * s4 - c1 * (c4 * s2 + c2 * c3 * s4)

    J46 = c5 * (c3 * s1 + c1 * c2 * s3) \
        + (-1 * c4 * s1 * s3 + c1 * (c2 * c3 * c4 - s2 * s4)) * s5

    J47 = s1 * (c6 * s3 * s4 + (c4 * c5 * s3 + c3 * s5) * s6) \
        + c1 * (-1 * c6 * (c4 * s2 + c2 * c3 * s4)
        + (c5 * s2 * s4 + c2 * (-1 * c3 * c4 * c5 + s3 * s5)) * s6)

    J51 = 0

    J52 = c1

    J53 = -s1 * s2

    J54 = -1 * c1 * c3 + c2 * s1 * s3

    J55 = -1 * c4 * s1 * s2 - (c2 * c3 * s1 + c1 * s3) * s4

    J56 = c1 * (-1 * c3 * c5 + c4 * s3 * s5) \
        + s1 * (-1 * s2 * s4 * s5 + c2 * (c5 * s3 + c3 * c4 * s5))

    J57 = -1 * c6 * (c4 * s1 * s2 + (c2 * c3 * s1 + c1 * s3) * s4) \
        - (c5 * (c4 * (c2 * c3 * s1 + c1 * s3) - s1 * s2 * s4) + (c1 * c3 - c2 * s1 * s3) * s5) * s6

    J61 = 1

    J62 = 0

    J63 = c2

    J64 = s2 * s3

    J65 = c2 * c4 - c3 * s2 * s4

    J66 = c5 * s2 * s3 + (c3 * c4 * s2 + c2 * s4) * s5

    J67 = c2 * (c4 * c6 - c5 * s4 * s6) + s2 * (s3 * s5 * s6 - c3 * (c6 * s4 + c4 * c5 * s6))

    J0[0][0] = J11
    J0[0][1] = J12
    J0[0][2] = J13
    J0[0][3] = J14
    J0[0][4] = J15
    J0[0][5] = J16
    J0[0][6] = J17

    J0[1][0] = J21
    J0[1][1] = J22
    J0[1][2] = J23
    J0[1][3] = J24
    J0[1][4] = J25
    J0[1][5] = J26
    J0[1][6] = J27

    J0[2][0] = J31
    J0[2][1] = J32
    J0[2][2] = J33
    J0[2][3] = J34
    J0[2][4] = J35
    J0[2][5] = J36
    J0[2][6] = J37

    J0[3][0] = J41
    J0[3][1] = J42
    J0[3][2] = J43
    J0[3][3] = J44
    J0[3][4] = J45
    J0[3][5] = J46
    J0[3][6] = J47

    J0[4][0] = J51
    J0[4][1] = J52
    J0[4][2] = J53
    J0[4][3] = J54
    J0[4][4] = J55
    J0[4][5] = J56
    J0[4][6] = J57

    J0[5][0] = J61
    J0[5][1] = J62
    J0[5][2] = J63
    J0[5][3] = J64
    J0[5][4] = J65
    J0[5][5] = J66
    J0[5][6] = J67
    return J0


def plan(obstacles, joint_angles, side, avoid_velocity=0.3):
    d_m = 2
    height = 6
    width = 7
    # unpack joint angles
    # names = ["".join([side, x]) for x in ["e0", "e1", "s0", "s1", "w0", "w1", "w2"]]
    new_joint_angles = dict()
    for name in joint_angles.keys():
        if side in name:
            new_joint_angles[name.split("")[1]] = joint_angles[name]["position"]
    if len(obstacles) == 0:
        # do simple planning, tracking is more precise
        pass
    else:
        for ob in obstacles:
            d0 = np.array([ob.x, ob.y, ob.z])
            d0_norm = np.linalg.norm(d0)
            n0 = d0 / d0_norm
            n0_t = n0.transpose()
            J0 = calculate_cartesian_jacobian(new_joint_angles)
            J_d0 = np.dot(n0_t, J0)
            I = np.eye(height, width)

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

            N_prime_0 = I - np.dot(a2, J_d0.transpose(), J_d0)
            x_dot = np.array((height, width))

            v0 = avoid_velocity
            x_dot_d0 = a1 * v0
            q_dot = np.dot(J_d0.transpose(), x_dot_d0) + np.dot(N_prime_0, J0.transpose(), x_dot)
