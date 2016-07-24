import baxter_interface
import math
import rospy
from scipy.stats import norm
virtualforce = False

__author__ = "Shaun Howard (smh150@case.edu)"


class Baxter():
    def __init__(self, p, obstacles, dt, m, fMax, diam):
        rospy.init_node("baxter_potential_field")
        rs = baxter_interface.RobotEnable()
        self.diam = diam
        self.fMax = fMax
        self.m = m
        self.dt = dt
        self.vx = 0
        self.vy = 0
        self.vz = 0
        self.x = p.x
        self.y = p.y
        self.z = p.z
        self.obstacles = obstacles
        self.target = obstacles[0]

    def in_range(self, ob, range_=0.6):
        dist = ob.distanceSq(self)
        if dist < range_:
            return True
        else:
            return False

    def update_position(self):
        x_dir = 0
        y_dir = 0
        z_dir = 0
        s_min = 200

        for ob in self.obstacles:
            squared_dist = ob.distanceSq(self)
            if squared_dist < 1:
                math.sin(1)
            dx = ob.charge * (ob.p.x - self.x) / squared_dist
            dy = ob.charge * (ob.p.y - self.y) / squared_dist
            dz = ob.charge * (ob.p.z - self.z) / squared_dist
            x_dir += dx
            y_dir += dy
            z_dir += dz

        norm_ = math.sqrt(x_dir*x_dir+y_dir*y_dir)
        x_dir /= norm_
        y_dir /= norm_

        for ob in self.obstacles:
            if not self.in_range(ob):
                continue
            squared_dist = ob.distanceSq(self)
            dx = (ob.p.x - self.x)
            dy = (ob.p.y - self.y)
            dz = (ob.p.z - self.z)
            # add normal noise to simulate the sonar effect
            dx = add_noise(dx, 0, 1)
            dy = add_noise(dy, 0, 1)
            dz = add_noise(dz, 0, 1)
            safety = squared_dist / (dx * x_dir + dy * y_dir + dz * z_dir)
            if (safety > 0) and (safety < s_min):
                s_min = safety

        if s_min < 5:
            oc = self.target.charge
            self.target.charge *= s_min/5
            print str(oc) + " DOWN TO " + str(self.target.charge)

        if s_min > 100:
            oc = self.target.charge
            self.target.charge *= s_min/100
            print str(oc) + " UP TO " + str(self.target.charge)

        yt_norm = s_min/2
        vtx = yt_norm * x_dir
        vty = yt_norm * y_dir
        vtz = yt_norm * z_dir
        fx = self.m * (vtx - self.vx) / self.dt
        fy = self.m * (vty - self.vy) / self.dt
        fz = self.m * (vtz - self.vz) / self.dt
        f_norm = math.sqrt(fx * fx + fy * fy)

        if f_norm > self.fMax:
            fx *= self.fMax / f_norm
            fy *= self.fMax / f_norm
            fz *= self.fMax / f_norm

        self.vx += (fx * self.dt) / self.m
        self.vy += (fy * self.dt) / self.m
        self.vz += (fz * self.dt) / self.m

        # virtual force component
        if virtualforce and self.target.charge < 1000 and self.x > 25 and self.y > 25:
            print "Virtual Force"
            self.target.charge *= s_min / 100
            self.vx += 5
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt
        self.z += self.vz * self.dt


def add_noise(val, mean, std_dev):
    r = norm.rvs(mean, std_dev)
    noise = std_dev * r + mean
    return val + noise
