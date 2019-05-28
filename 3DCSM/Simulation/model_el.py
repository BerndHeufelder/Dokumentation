import control
import param
import numpy as np


class Model:

    def __init__(self):
        self.motor_y = model_PMSM(kP=2.36262, kI=0.0, R=param.R_y, L=param.L_y, kbemf=param.kbemf_y)
        self.motor_z = model_PMSM(kP=3.3, kI=0.0, R=param.R_z, L=param.L_z, kbemf=param.kbemf_z)
        self.motor_p = model_PMSM(kP=50.0, kI=0.0, R=param.R_phi, L=param.L_phi, kbemf=param.kbemf_phi)

    def set_initial_values(self, im0):
        im_y = im0[0]
        im_z = im0[1]
        im_phi = im0[2]
        ey_int0 = 0.0
        ez_int0 = 0.0
        ep_int0 = 0.0
        z0 = [im_y, im_z, im_phi, ey_int0, ez_int0, ep_int0]
        return z0

    def calc_input(self, state_el, state_ctrl_el, u):
        [Fy_ist, Fz_ist, Mp_ist] = self.trans_motorCurrent_to_Force(state_el)
        [Fy_ref, Fz_ref, Mp_ref] = u

        # Regelfehler
        e_Fy = Fy_ist - Fy_ref
        e_Fz = Fz_ist - Fz_ref
        e_Mphi = Mp_ist - Mp_ref

        e_y_int = state_ctrl_el[0]
        e_z_int = state_ctrl_el[1]
        e_p_int = state_ctrl_el[2]

        v_zk_y = self.motor_y.ctrl.control(e_Fy, e_y_int)
        v_zk_z = self.motor_z.ctrl.control(e_Fz, e_z_int)
        v_zk_phi = self.motor_p.ctrl.control(e_Mphi, e_p_int)

        v_zk = [v_zk_y, v_zk_z, v_zk_phi]
        e = [e_Fy, e_Fz, e_Mphi]
        return v_zk, e

    def trans_motorCurrent_to_Force(self, im):
        My = im[0]*param.kbemf_y
        Mz = im[1]*param.kbemf_z
        Mp = im[2]*param.kbemf_phi
        Fy = My*2*np.pi/param.steig_y
        Fz = Mz*2*np.pi/param.steig_z
        return [Fy, Fz, Mp]

    def trans_Force_to_motorCurrent(self, Q):
        [Fy, Fz, Mp] = Q
        My = Fy*param.steig_y/(2*np.pi)
        Mz = Fz*param.steig_z/(2*np.pi)
        im_y = My/param.kbemf_y
        im_z = Mz/param.kbemf_z
        im_p = Mp/param.kbemf_phi

        im = [im_y, im_z, im_p]
        return im

    def sys_eq(self, dq_mech, state_el, u):
        # Elektrische Systemgleichungen beachten und Eingang des mech. Systems anpassen
        im_y = state_el[0]
        im_z = state_el[1]
        im_phi = state_el[2]

        dy = dq_mech[0]
        dz = dq_mech[1]
        wm_phi = dq_mech[2]

        # Eingangsgrößen der elektrischen Systeme
        v_zk_y = u[0]
        v_zk_z = u[1]
        v_zk_p = u[2]

        # Motorwinkelgesch. der Spindelschlitten berechnen
        wm_y = 2*np.pi*dy/param.steig_y
        wm_z = 2*np.pi*dz/param.steig_z

        # Zustandsdiff.-Gl.
        d_im_y = self.motor_y.motor_dynamics(v_zk=v_zk_y, im=im_y, wm=wm_y)
        d_im_z = self.motor_z.motor_dynamics(v_zk=v_zk_z, im=im_z, wm=wm_z)
        d_im_phi = self.motor_p.motor_dynamics(v_zk=v_zk_p, im=im_phi, wm=wm_phi)

        # Eingangsgrößen zwischenspeichern
        self.u = u

        return [d_im_y, d_im_z, d_im_phi]


class model_PMSM:

    def __init__(self, kP, kI, kbemf, R, L):
        self.ctrl = control.PID_motor(kP, kI)
        self.kbemf = kbemf
        self.R = R
        self.L = L
        self.z0 = None

    def set_initial_value(self, im=0.0):
        self.z0 = im

    def motor_dynamics(self, v_zk, im, wm):
        kbemf = self.kbemf
        R = self.R
        L = self.L
        Dz = 1/L*(v_zk - im*R - kbemf*wm)
        return Dz
