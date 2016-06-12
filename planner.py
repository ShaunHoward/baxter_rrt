import math
import numpy as np
import random
import scipy


def calculate_cartesian_jacobian():
    height = 6
    width = 7
    J0 = np.array([[[0]*width]*height])
    c1 = 0
    c2 = 0
    c3 = 0
    c4 = 0
    c5 = 0
    c6 = 0
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    s5 = 0
    s6 = 0
    L2 = 0
    L3 = 0
    L4 = 0

    J11 = c1 * ((L3 + L4 * c6) * s3 * s4 + L4 * (c4 * c5 * s3 + c3 * c5 ) * s6) \
         + s1 * s2 * (L2 + c4 * (L3 + L4 * c6) − L4 * c5 * s4 * s6) \
         + c2 * (c3 * (L3 + L4 * c6) * s4 + L4 * (c3 * c4 * c5 − s3 * s5 * s6))

    J12 = c1 * (−c2 * (L2 + c4 * (L3 + L4 * c6) − L4 * c5 * s4 * s6) \
         + s2 * (c3 * (L3 + L4 * c6) * s4 + L4 * (c3 * c4 * c5 − s3 * s4) * s6))

    J13 = (L3 + L4 * c6) * (c3 * s1 + c1 * c2 * s3) * s4 \
         + L4 * (c4 * c5 * (c3 * s1 + c1 * c2 * s3) + (c1 * c2 * c3 − s1 * s3) * s5) * s6

    J14 = (L3 + L4 c6) (c4 s1 s3 + c1 (−c2 c3 c4 + s2 s4)) \
         + L4 c5(−s1 s3 s4 + c1 * (c4 * s2 + c2 * c3 * s4)) * s6

    J15 = L4 * (c5 * (c3 * s1 + c1 * c2 * s3) \
         + (−c4 * s1 * s3 + c1 * (c2 * c3 * c4 − s2 * s4)) * s5) * s6

    J16 = L4 * (s1 * (c6 * (c4 * c5 * s3 + c3 s5) − s3 * s4 * s6) \
         + c1 * (c6 * (c5 * s2 * s4 + c2(−c3 * c4 * c5 + s3 * s5)) \
         + (c4 * s2 + c2 * c3 * s4) * s6))

    J17 = 0

    J21 = s1 * ((L3 + L4 * c6) * s3 * s4 + L4 * (c4 * c5 * s3 + c3 * s5) * s6) \
         − c1 * (s2 * (L2 + c4 * (L3 + L4 * c6) − L4 * c5 * s4 * s6) \
         + c2 * (c3 * (L3 + L4 * c6) * s4 + L4 * (c3 * c4 * c5 − s3 * s5) * s6))

    J22 = s 1 (−c 2 (L 2 + c 4 (L 3 + L 4 c 6 ) − L 4 c 5 s 4 s 6 ) + s 2 (c 3 (L 3 + L 4 c 6 ) s 4 + L 4 (c 3 c 4 c 5 − s 3 s 5 ) s 6 ))

    J23 = − (L 3 + L 4 c 6 ) (c 1 c 3 − c 2 s 1 s 3 ) s 4 + L 4 (c 2 s 1 (c 4 c 5 s 3 + c 3 s 5 ) + c 1 (−c 3 c 4 c 5 + s 3 s 5 )) s 6 ,

    J24 = − (L 3 + L 4 c 6 ) (c 4 (c 2 c 3 s 1 + c 1 s 3 ) − s 1 s 2 s 4 ) + L 4 c 5 (c 4 s 1 s 2 + (c 2 c 3 s 1 + c 1 s 3 ) s 4 ) s 6

    J25 = L 4 (c 1 (−c 3 c 5 + c 4 s 3 s 5 ) + s 1 (−s 2 s 4 s 5 + c 2 (c 5 s 3 + c 3 c 4 s 5 ))) s 6

    J26 = L 4 (c 6 (c 5 (−c 4 (c 2 c 3 s 1 + c 1 s 3 ) + s 1 s 2 s 4 ) + (−c 1 c 3 + c 2 s 1 s 3 ) s 5 ) + (c 4 s 1 s 2 + (c 2 c 3 s 1 + c 1 s 3 ) s 4 ) s 6 )

    J27 = 0

    J31 = 0

    J32 = −s 2 (L 2 + c 4 (L3 + L 4 c 6 ) − L 4 c 5 s 4 s 6 )
        + c 2 (L 4 s 3 s 5 s 6 − c 3 ((L 3 + L 4 c 6 ) s 4 + L 4 c 4 c 5 s 6 ))

    J33 = s 2 ((L 3 + L 4 c 6 ) s 3 s 4 + L 4 (c 4 c 5 s 3 + c 3 s 5 ) s 6 )

    J34 = − (L 3 + L 4 c 6 ) (c 3 c 4 s 2 + c 2 s 4 ) + L 4 c 5 (−c 2 c 4 + c 3 s 2 s 4 ) s 6

    J35 = L 4 (c 5 s 2 s 3 + (c 3 c 4 s 2 + c 2 s 4 ) s 5 ) s 6

    J36 = −L 4 (c 6 (c 5 (c 3 c 4 s 2 + c 2 s 4 ) − s 2 s 3 s 5 ) + (c 2 c 4 − c 3 s 2 s 4 ) s 6 )

    J37 = 0

    J41 = 0

    J42 = s1

    J43 = -c1 * s2

    J44 = c3 * s1 + c1 * c2 * s3

    J45 = s 1 s 3 s 4 − c 1 (c 4 s 2 + c 2 c 3 s 4 )

    J46 = c 5 (c 3 s 1 + c 1 c 2 s 3 )
        + (−c 4 s 1 s 3 + c 1 (c 2 c 3 c 4 − s 2 s 4 )) s 5

    J47 = s 1 (c 6 s 3 s 4 + (c 4 c 5 s 3 + c 3 s 5 ) s 6 )
        + c 1 (−c 6 (c 4 s 2 + c 2 c 3 s 4 )
        + (c 5 s 2 s 4 + c 2 (−c 3 c 4 c 5 + s 3 s 5 )) s 6 )

    J51 = 0

    J52 = c1

    J53 = -s1 * s2

    J54 = −c 1 c 3 + c 2 s 1 s 3

    J55 = −c 4 s 1 s 2 − (c 2 c 3 s 1 + c 1 s 3 ) s 4

    J56 = c 1 (−c 3 c 5 + c 4 s 3 s 5 )
        + s 1 (−s 2 s 4 s 5 + c 2 (c 5 s 3 + c 3 c 4 s 5 ))

    J57 = −c 6 (c 4 s 1 s 2 + (c 2 c 3 s 1 + c 1 s 3 ) s 4 )
        − (c 5 (c 4 (c 2 c 3 s 1 + c 1 s 3 ) − s 1 s 2 s 4 ) + (c 1 c 3 − c 2 s 1 s 3 ) s 5 ) s 6

    J61 = 1

    J62 = 0

    J63 = c2

    J64 = s2 * s3

    J65 = c 2 c 4 − c 3 s 2 s 4

    J66 = c 5 s 2 s 3 + (c 3 c 4 s 2 + c 2 s 4 ) s 5

    J67 = c 2 (c 4 c 6 − c 5 s 4 s 6 )
        + s 2 (s 3 s 5 s 6 − c 3 (c 6 s 4 + c 4 c 5 s 6 ))

    return J0


def plan(obstacles, jointangles, side):
    # unpack joint angles
    #names = ["".join([side, x]) for x in ["e0", "e1", "s0", "s1", "w0", "w1", "w2"]]

    jointangles = dict()
    for name in jointangles.keys():
        if side in name:
            jointangles[name.split("")[1]] = jointangles[name]["position"]
    if len(obstacles) == 0:
        # do simple planning, tracking is more precise
        pass
    else:
        for ob in obstacles:
            d0 = np.array([ob.x, ob.y, ob.z])
            d0norm = np.linalg.norm(d0)
            n0 = d0 / d0norm
            n0t = n0.transpose()
            J0 = calculatecartesianjacobian()
            Jd0 = np.dot(n0t, J0)
