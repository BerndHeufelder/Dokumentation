import numpy as np
import sympy as sp
from sympy import lambdify
import scipy.io as sio
from scipy import interpolate
import settings as st


class Trajectory:

    def __init__(self, t0, T, y, z, phi, f0_rigid):
        # Generate trajectory functions
        if st.inputShaping:
            self.t_shift = 0.5/f0_rigid  # simulierter 10Hz offset fürs Input-shaping
            print('identified natural frequency: ', f0_rigid)
        else:
            self.t_shift = 0.0
        self.y = y
        self.z = z
        self.phi = phi
        self.T = T      # Endzeit der Bewegung
        self.t_set = st.t_set
        self.t0 = t0
        self.psa = self.s_curve(0, 1, 0, 1)

    def s_curve(self, t0, T, yA, yB):
        o = 7  # Polynomordnung
        t = sp.symbols('t')
        c = sp.symarray('c', o + 1)  # Symbolische Koeffizienten

        # Polynom ansetzen
        p0 = c[0]
        for k in range(1, o + 1):
            p0 += c[k]*t**k

        # Ableitungen bilden
        p = dict()
        p[0] = p0
        for k in range(1, o):
            p[k] = sp.diff(p[k - 1], t)

        # Gleichungssystem von Polynomen mit Anfangs- und Endbedingungen erstellen, Ruck soll Null sein bei t0 & T
        eqs = [
            p[0].subs(t, t0) - yA,
            p[0].subs(t, T) - yB,
            p[1].subs(t, t0) - 0,
            p[1].subs(t, T) - 0,
            p[2].subs(t, t0) - 0,
            p[2].subs(t, T) - 0,
            p[3].subs(t, t0) - 0,
            p[3].subs(t, T) - 0,
        ]

        # Polynomkoeffizienten anhand der Randbedingungen berechnen
        coeff = sp.solve(eqs)

        # Lambda Funktionen für die Polynome der Pos, Spd und Acc erstellen
        p_lam = dict()
        for i in range(0, len(p)):
            for j in range(0, len(c)):
                p[i] = p[i].subs(c[j], coeff[c[j]])
                p_lam[i] = lambdify(t, p[i])
        return p_lam

    def eval_ptf(self, t):
        ptf_val = list()
        if t <= 0:
            ptf_val.append(0)
            ptf_val.append(0)
            ptf_val.append(0)
        elif t >= 1:  # Position nach Ende der Bewegung beibehalten
            ptf_val.append(1)
            ptf_val.append(0)
            ptf_val.append(0)
        else:
            ptf_val.append(self.psa[0](t))
            ptf_val.append(self.psa[1](t))
            ptf_val.append(self.psa[2](t))
        return np.array(ptf_val)

    def eval_traj(self, t, y):
        # trajectory for z-Axis: y, dy, ddy
        deltaT = st.t_mve
        tau0 = (t - self.t0)/deltaT
        tau1 = (t - self.t0 - self.t_shift)/deltaT
        y0 = np.array([y[0], 0, 0])
        tscale = np.array([1, deltaT, deltaT**2])
        res = 0.5*np.array(self.eval_ptf(tau0) + self.eval_ptf(tau1))*(y[1] - y[0])/tscale + y0
        return res

    def eval_y_traj(self, t):
        return self.eval_traj(t, self.y)

    def eval_z_traj(self, t):
        return self.eval_traj(t, self.z)

    def eval_phi_traj(self, t):
        return self.eval_traj(t, self.phi)

    def setTraj_fromData(self, T):
        fmat = sio.loadmat(st.fpath_mat)

        zAxis = fmat['export'][0][0]
        yAxis = fmat['export'][1][0]
        pAxis = fmat['export'][2][0]

        yPos = yAxis[0]/1000
        ySpd = yAxis[1]/1000
        yAcc = yAxis[2]/1000

        zPos = zAxis[0]/1000
        zSpd = zAxis[1]/1000
        zAcc = zAxis[2]/1000

        pPos = pAxis[0]
        pSpd = pAxis[1]
        pAcc = pAxis[2]

        t = np.linspace(self.t0, self.T, len(yPos))
        t = np.append(t, self.T+self.t_set)

        self.f1yPos = interpolate.interp1d(t, np.append(yPos, yPos[-1]))
        self.f1ySpd = interpolate.interp1d(t, np.append(ySpd, ySpd[-1]))
        self.f1yAcc = interpolate.interp1d(t, np.append(yAcc, yAcc[-1]))

        self.f1zPos = interpolate.interp1d(t, np.append(zPos, zPos[-1]))
        self.f1zSpd = interpolate.interp1d(t, np.append(zSpd, zSpd[-1]))
        self.f1zAcc = interpolate.interp1d(t, np.append(zAcc, zAcc[-1]))

        self.f1pPos = interpolate.interp1d(t, np.append(pPos, pPos[-1]))
        self.f1pSpd = interpolate.interp1d(t, np.append(pSpd, pSpd[-1]))
        self.f1pAcc = interpolate.interp1d(t, np.append(pAcc, pAcc[-1]))

    def eval_all_trajectories(self, t):
        if st.trajFromData:
            y_traj = np.array([float(self.f1yPos(t)),
                               float(self.f1ySpd(t)),
                               float(self.f1yAcc(t))])
            z_traj = np.array([float(self.f1zPos(t)),
                               float(self.f1zSpd(t)),
                               float(self.f1zAcc(t))])
            phi_traj = np.array([float(self.f1pPos(t)),
                                 float(self.f1pSpd(t)),
                                 float(self.f1pAcc(t))])
        else:
            y_traj = self.eval_y_traj(t)
            z_traj = self.eval_z_traj(t)
            phi_traj = self.eval_phi_traj(t)
        return y_traj, z_traj, phi_traj
