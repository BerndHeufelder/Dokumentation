# -*- coding: utf-8 -*-


class PID_traj:

    def __init__(self):
        # Polvorgabe Trajektorienregelung
        omega0 = 3000.0
        d = 0.5
        h = 1.0
        self.kD = 2*omega0*d + h
        self.kP = omega0**2 + 2*h*omega0*d
        self.kI = omega0**2*h

    def control(self, e, e_int):
        dde = -self.kD*e[1] - self.kP*e[0] - self.kI*e_int
        return dde


class PID_motor:

    def __init__(self, kP, kI):
        self.kP = kP
        self.kI = kI

    def control(self, e, e_int):
        v_zk = -self.kP*e - self.kI*e_int
        return v_zk
