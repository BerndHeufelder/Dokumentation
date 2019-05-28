# -*- coding: utf-8 -*-
import matplotlib as mpl
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pickle
import model_mech
import settings as st
from sympy.utilities.iterables import flatten
from sympy.utilities.lambdify import lambdify
import scipy.io as sio
import param
from eq_of_motion import get_compensationTorque_from_phi


class plot_results:

    def __init__(self, suffix):
        print('Initialising ...')
        self.savefigs = st.savefigs
        self.color_ref = 'g'
        self.color_plt = 'b'
        self.idx = 1
        self.suffix = suffix
        self.elModel_allSims = False  # bool var to check if el Model is present in any simulation mentioned in suffix list
        self.figTitle = ''
        print('    ', 'Following Simulations will be plotted:')
        for i in range(0, len(suffix)):
            print('    ', i+1, ' ', suffix[i])
            self.figTitle += 'Sim ' + str(i+1) + ': ' + suffix[i] + '\n'

    def loadFile(self, current_suffix):
        print('   ', '... loading Data from Files ...')
        self.current_suffix = current_suffix
        if len(self.suffix) > 1:
            self.legend_suffix = 'Sim ' + str(self.idx)
        else:
            self.legend_suffix = ''
        filename = st.path_pickle + 'Simulation_result_' + current_suffix + '.pickle'

        # Plot from file
        with open(filename, 'rb') as inf:
            result = pickle.loads(inf.read())

        time = result['time'][:-1]
        res = result['res'][:-1, :]
        u = result['input'][:-1, :]
        trajData = result['trajData']
        self.rigid = result['model']['rigid']
        self.bool_elModel = result['settings']['bool_elModel']
        self.T = result['settings']['T']   # Zeit bei Bewegungsende, im Anschluss t_set
        if self.elModel_allSims==False:    # el Model Flag setzen falls bisher noch keines vorhanden war
            self.elModel_allSims = self.bool_elModel

        ref = trajData[0]

        oModel = model_mech.Model(self.rigid)
        oModel.time = time
        oModel.res = res
        oModel.u = u

        return oModel, ref

    def createFigures(self):
        print('Creating figures ...')
        # Figure für Endeffektor erzeugen
        self.fig_ee = plt.figure(figsize=(13, 8))
        plt.subplots_adjust(wspace=0.7, hspace=0.5, top=0.9, bottom=0.1, left=0.1, right=0.9)
        self.ax1ee = plt.subplot()
        gs = mpl.gridspec.GridSpec(3, 2)
        self.ax1ee = plt.subplot(gs[0:3, 0])
        self.ax1ee.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        self.ax2ee = plt.subplot(gs[0, 1])
        self.ax2ee.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        self.ax3ee = plt.subplot(gs[1, 1])
        self.ax3ee.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        self.ax4ee = plt.subplot(gs[2, 1])
        self.ax4ee.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        if len(self.suffix) > 1:
            self.fig_ee.suptitle(self.figTitle)

        # Figure für Motorachsen erzeugen
        self.fig_mt = plt.figure(figsize=(13, 8))
        ((self.ax1mt, self.ax2mt), (self.ax3mt, self.ax4mt), (self.ax5mt, self.ax6mt)) = self.fig_mt.subplots(3, 2)
        self.ax1mt.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        self.ax2mt.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        self.ax3mt.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        self.ax4mt.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        self.ax5mt.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        self.ax6mt.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        plt.subplots_adjust(wspace=0.4, hspace=0.7)
        if len(self.suffix) > 1:
            self.fig_mt.suptitle(self.figTitle)

        # Figure für Federn erzeugen
        numSprings = st.rigid.count(False) + 1  # Anzahl an Federn im System
        if numSprings > 2:
            self.fig_sp = plt.figure(figsize=(13, 8))
            ((self.ax1sp, self.ax2sp), (self.ax3sp, self.ax4sp)) = self.fig_sp.subplots(2, 2)
            self.ax1sp.yaxis.set_major_formatter(FormatStrFormatter('%g'))
            self.ax2sp.yaxis.set_major_formatter(FormatStrFormatter('%g'))
            self.ax3sp.yaxis.set_major_formatter(FormatStrFormatter('%g'))
            self.ax4sp.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        else:
            self.fig_sp = plt.figure(figsize=(13, 4))
            self.ax1sp, self.ax2sp = self.fig_sp.subplots(1, 2)
            self.ax1sp.yaxis.set_major_formatter(FormatStrFormatter('%g'))
            self.ax2sp.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        # plt.tight_layout()
        if len(self.suffix) > 1:
            self.fig_sp.suptitle(self.figTitle)

        # Figure für Kräfte und Moment erzeugen
        self.fig_fo = plt.figure(figsize=(13, 8))
        (self.ax1fo, self.ax2fo, self.ax3fo) = self.fig_fo.subplots(3, 1)
        self.ax1fo.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        self.ax2fo.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        self.ax3fo.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        # plt.tight_layout()
        if len(self.suffix) > 1:
            self.fig_fo.suptitle(self.figTitle)

        # Figure für Kräfte und Moment erzeugen
        self.fig_u = plt.figure(figsize=(12, 8))
        ((self.ax1u, self.ax2u, self.ax3u), (self.ax1du, self.ax2du, self.ax3du),
         (self.ax1ddu, self.ax2ddu, self.ax3ddu)) = self.fig_u.subplots(3, 3)
        self.ax1u.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        self.ax2u.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        self.ax3u.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        self.ax1du.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        self.ax2du.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        self.ax3du.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        self.ax1ddu.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        self.ax2ddu.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        self.ax3ddu.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        plt.subplots_adjust(wspace=0.4, hspace=0.7)
        if len(self.suffix) > 1:
            self.fig_u.suptitle(self.figTitle)

        # Figure für Kräfte und Moment erzeugen
        self.fig_ee_pva = plt.figure(figsize=(12, 8))
        ((self.axYee, self.axZee, self.axPee), (self.axdYee, self.axdZee, self.axdPee),
         (self.axddYee, self.axddZee, self.axddPee)) = self.fig_ee_pva.subplots(3, 3)
        self.axYee.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        self.axZee.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        self.axPee.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        self.axdYee.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        self.axdZee.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        self.axdPee.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        self.axddYee.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        self.axddZee.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        self.axddPee.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        plt.subplots_adjust(wspace=0.4, hspace=0.7)
        if len(self.suffix) > 1:
            self.fig_ee_pva.suptitle(self.figTitle)

        # if st.elModel:
        # Figure Zustände
        self.fig_elCurrent = plt.figure(figsize=(13, 8))
        (self.axy, self.axz, self.axp) = self.fig_elCurrent.subplots(3, 1)
        self.axy.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        self.axz.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        self.axp.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        plt.subplots_adjust(wspace=0.4, hspace=0.7)
        if len(self.suffix) > 1:
            self.fig_elCurrent.suptitle(self.figTitle)

    def calculateFromResults(self, oModel, ref):
        print('   ', '... start plotting ...')
        time = oModel.time[:, 0]
        res = oModel.res
        u = oModel.u

        # results
        y = res[:, 0]
        dy = res[:, 1]
        z = res[:, 2]
        dz = res[:, 3]
        phi = res[:, 4]
        dphi = res[:, 5]
        theta = res[:, 6]
        dtheta = res[:, 7]
        alpha = res[:, 8]
        dalpha = res[:, 9]
        g = res[:, 10]
        dg = res[:, 11]
        rho = res[:, 12]
        drho = res[:, 13]
        if self.bool_elModel:
            im_y = res[:, 17]
            im_z = res[:, 18]
            im_p = res[:, 19]
            q = np.array([y, z, phi, theta, alpha, g, rho, im_y, im_z, im_p])
        else:
            q = np.array([y, z, phi, theta, alpha, g, rho])

        # Ist-Bahn-Verlauf des EE
        dq = np.array([dy, dz, dphi, dtheta, dalpha, dg, drho])
        rb = oModel.eval_forwardKinematics(q)
        drb, ddrb, ddq = self.calc_EE_Acceleration(oModel, q, dq, u)  # ddq = [ddy, ddz, ddphi, ddtheta, ddalpha, ddg, ddrho]

        # Refernzbahnverlauf
        y_ref = ref[:-1, 0]
        dy_ref = ref[:-1, 1]
        ddy_ref = ref[:-1, 2]
        z_ref = ref[:-1, 3]
        dz_ref = ref[:-1, 4]
        ddz_ref = ref[:-1, 5]
        phi_ref = ref[:-1, 6]
        dphi_ref = ref[:-1, 7]
        ddphi_ref = ref[:-1, 8]

        q_ref = np.array([y_ref, z_ref, phi_ref, np.zeros(len(y_ref)), np.zeros(len(y_ref)), np.zeros(len(y_ref)), np.zeros(len(y_ref))])
        rb_ref = oModel.eval_forwardKinematics(q_ref)

        # Tracking Error des EE
        rb_err = rb - rb_ref

        # Tracking Errors aller 3 Achsen
        e_phi = (phi - phi_ref)*10**6
        e_y = (y - y_ref)*10**6
        e_z = (z - z_ref)*10**6

        err = {'rb': rb_err, 'e_phi': e_phi, 'e_y': e_y, 'e_z': e_z}
        ref = {'input': u,
               'rb': rb_ref, 'y': y_ref, 'z': z_ref, 'phi': phi_ref,
                             'dy': dy_ref, 'dz': dz_ref, 'dphi': dphi_ref,
                            'ddy': ddy_ref, 'ddz': ddz_ref, 'ddphi': ddphi_ref}
        return q, dq, ddq, rb, drb, ddrb, err, ref, time

    def plot_res_as_subplots(self, q, dq, ddq, rb, drb, ddrb, err, ref, time):
        rb_ref = ref['rb']
        y_ref = ref['y']
        z_ref = ref['z']
        phi_ref = ref['phi']
        dy_ref = ref['dy']
        dz_ref = ref['dz']
        dphi_ref = ref['dphi']
        ddy_ref = ref['ddy']
        ddz_ref = ref['ddz']
        ddphi_ref = ref['ddphi']

        rb_err = err['rb']
        e_y = err['e_y']
        e_z = err['e_z']
        e_phi = err['e_phi']

        u = ref['input']
        Fy = u[:, 0]
        Fz = u[:, 1]
        Mphi = u[:, 2]

        y = q[0]
        z = q[1]
        phi = q[2]
        theta = q[3]
        alpha = q[4]
        g = q[5]
        rho = q[6]
        if self.bool_elModel:
            im_y = q[7]
            im_z = q[8]
            im_p = q[9]

        dy = dq[0]
        dz = dq[1]
        dphi = dq[2]
        dtheta = dq[3]
        dalpha = dq[4]
        dg = dq[5]
        drho = dq[6]

        ddy = ddq[0]
        ddz = ddq[1]
        ddphi = ddq[2]
        ddtheta = ddq[3]
        ddalpha = ddq[4]
        ddg = ddq[5]
        ddrho = ddq[6]

        # Plot Endeffektor Position
        self.plot_on_Axis(self.ax1ee, [rb[0], rb_ref[0]], [rb[1], rb_ref[1]], color=[self.color_plt, self.color_ref], title='Movement in yz-Plane', xlabel=r'$y$ in $m$', ylabel=r'$z$ in $m$', label=[('EE-Position, ' + self.legend_suffix), ('EE-Referenzbahn, ' + self.legend_suffix)])
        self.ax1ee.set_aspect('equal')
        self.ax1ee.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        self.plot_on_Axis(self.ax2ee, time, rb_err[0]*10**6, color=self.color_plt, xlabel=r'$t$ in $s$', ylabel=r'$\mu m$', label='Tracking Error in y, '+ self.legend_suffix, mve_end_vLine=True)
        self.plot_on_Axis(self.ax3ee, time, rb_err[1]*10**6, color=self.color_plt, xlabel=r'$t$ in $s$', ylabel=r'$\mu m$', label='Tracking Error in z, ' + self.legend_suffix, mve_end_vLine=True)
        self.plot_on_Axis(self.ax4ee, time, (rb[2] - np.array(phi_ref))*10**6, color=self.color_plt, xlabel=r'$t$ in $s$', ylabel=r'in $\mu rad$', label='Tracking Error in \varphi, '+ self.legend_suffix, mve_end_vLine=True)

        # Plot rel. movement of all 3 axis and their traicking errors
        self.plot_on_Axis(self.ax1mt, [time, time], [y, y_ref], color=[self.color_plt, self.color_ref], title='y-Achse', xlabel='t in s', ylabel=r'$m$', label=[r'$y(t)$, ' + self.legend_suffix, r'$y_{ref}(t)$ ' + self.legend_suffix])
        self.plot_on_Axis(self.ax3mt, [time, time], [z, z_ref], color=[self.color_plt, self.color_ref], title='z-Achse', xlabel='t in s', ylabel=r'$m$', label=[r'$z(t)$ ' + self.legend_suffix, r'$z_{ref}(t)$ ' + self.legend_suffix])
        self.plot_on_Axis(self.ax5mt, [time, time], [(phi*180/np.pi), (np.asarray(phi_ref)*180/np.pi)], color=[self.color_plt, self.color_ref], title='RotX-Motorachse', xlabel='t in s', ylabel=r'$deg$', label=[r'$\varphi(t)$ ' + self.legend_suffix, r'$\varphi_{ref}(t)$ ' + self.legend_suffix])

        self.plot_on_Axis(self.ax2mt, time, e_y, color=self.color_plt, title='Tracking Error y-Achse', ylabel=r'$\mu m$', label=r'$e_{y}$ ' + self.legend_suffix, mve_end_vLine=True)
        self.plot_on_Axis(self.ax4mt, time, e_z, color=self.color_plt, title='Tracking Error z-Achse', ylabel=r'$\mu m$', label=r'$e_{z}$ ' + self.legend_suffix, mve_end_vLine=True)
        self.plot_on_Axis(self.ax6mt, time, e_phi*180/np.pi, color=self.color_plt, title='Tracking Error RotX-Achse', ylabel=r'$\mu deg$', label=r'$e_{\varphi}$ ' + self.legend_suffix, mve_end_vLine=True)

        # Plot Freiheitsgrade der im System befindlichen Federn
        lx_spring = np.zeros(len(phi))
        for i in range(0, len(phi)):
            lx_spring[i] = get_compensationTorque_from_phi(phi[i])

        self.plot_on_Axis(self.ax1sp, time, alpha*1e6, color=self.color_plt, title='Biegefeder Unterbau', ylabel=r'$\mu rad$', label=r'$\alpha(t)$ ' + self.legend_suffix)
        self.plot_on_Axis(self.ax2sp, time, lx_spring*1e3, color=self.color_plt, title='Trans.-Feder RotX Kompensation (effektive Länge)', xlabel='t in s', ylabel=r'$mm$', label=r'$l_x(t)$ ' + self.legend_suffix)
        if st.rigid.count(False)+1 > 2:
            self.plot_on_Axis(self.ax3sp, time, theta*1e6, color=self.color_plt, title='Biegefeder Rot-Body-Achse', ylabel=r'$\mu rad$', label=r'$\theta(t)$ ' + self.legend_suffix)
        if st.rigid.count(False) + 1 > 3:
            self.plot_on_Axis(self.ax4sp, time, rho*1e6, color=self.color_plt, title='Biegefeder Bodyholder (Konus)', xlabel='t in s', ylabel=r'$\mu rad$', label=r'$\rho(t)$ ' + self.legend_suffix)

        # Plot Motormomente
        self.plot_on_Axis(self.ax1fo, time, Fy, color=self.color_plt, title='Motor Y', ylabel=r'N', label=r'$F_{y}$ ' + self.legend_suffix)
        self.plot_on_Axis(self.ax2fo, time, Fz, color=self.color_plt, title='Motor Z', ylabel=r'N', label=r'$F_{z}$ ' + self.legend_suffix)
        self.plot_on_Axis(self.ax3fo, time, Mphi, color=self.color_plt, title='Motor Rot-X', xlabel='time in s', ylabel=r'Nm', label=r'$\tau_\phi(t)$, ' + self.legend_suffix)

        # Plot Eingangsverläufe
        self.plot_on_Axis(self.ax1u, time, y_ref, color=self.color_ref, label='y-Position, ' + self.legend_suffix, xlabel='t in s', ylabel=r'$m$')
        self.plot_on_Axis(self.ax2u, time, z_ref, color=self.color_ref, label='z-Position, ' + self.legend_suffix, xlabel='t in s', ylabel=r'$m$')
        self.plot_on_Axis(self.ax3u, time, phi_ref, color=self.color_ref, label='RotX-Position, ' + self.legend_suffix, xlabel='t in s', ylabel=r'$rad$')

        self.plot_on_Axis(self.ax1du, time, dy_ref, color=self.color_ref, label='y-Velocity, ' + self.legend_suffix, xlabel='t in s', ylabel=r'$m/s$')
        self.plot_on_Axis(self.ax2du, time, dz_ref, color=self.color_ref, label='z-Velocity, ' + self.legend_suffix, xlabel='t in s', ylabel=r'$m/s$')
        self.plot_on_Axis(self.ax3du, time, dphi_ref, color=self.color_ref, label='RotX-Angular-Velocity, ' + self.legend_suffix, xlabel='t in s', ylabel=r'$rad/s$')

        self.plot_on_Axis(self.ax1ddu, time, ddy_ref, color=self.color_ref, label='y-Acc, ' + self.legend_suffix, xlabel='t in s', ylabel=r'$m/s^2$')
        self.plot_on_Axis(self.ax2ddu, time, ddz_ref, color=self.color_ref, label='z-Acc, ' + self.legend_suffix, xlabel='t in s', ylabel=r'$m/s^2$')
        self.plot_on_Axis(self.ax3ddu, time, ddphi_ref, color=self.color_ref, label='RotX-Angular-Acc, ' + self.legend_suffix, xlabel='t in s', ylabel=r'$rad/s^2$')

        # Plot Endeffektor Pos, Geschw. Acc
        self.plot_on_Axis(self.axYee, time, rb[0], color=self.color_plt, label='EE y-Position, ' + self.legend_suffix, xlabel='t in s', ylabel=r'$m$')
        self.plot_on_Axis(self.axZee, time, rb[1], color=self.color_plt, label='EE z-Position, ' + self.legend_suffix, xlabel='t in s', ylabel=r'$m$')
        self.plot_on_Axis(self.axPee, time, rb[2], color=self.color_plt, label='EE Angle of x-Axis, ' + self.legend_suffix, xlabel='t in s', ylabel=r'$rad$')

        self.plot_on_Axis(self.axdYee, time, drb[0], color=self.color_plt, label='EE y-Velocity, ' + self.legend_suffix, xlabel='t in s', ylabel=r'$m/s$')
        self.plot_on_Axis(self.axdZee, time, drb[1], color=self.color_plt, label='EE z-Velocity, ' + self.legend_suffix, xlabel='t in s', ylabel=r'$m/s$')
        self.plot_on_Axis(self.axdPee, time, drb[2], color=self.color_plt, label='EE Angular-Velocity of x-Axis, ' + self.legend_suffix, xlabel='t in s', ylabel=r'$rad/s$')

        # [np.diff(drb[0])/(time[1] - time[0]), ddrb[0]], color = ['y', self.color_plt]
        self.plot_on_Axis(self.axddYee, time[:-1], np.diff(drb[0])/(time[1]-time[0]), color= self.color_plt, label='EE y-Acc, ' + self.legend_suffix, xlabel='t in s', ylabel=r'$m/s^2$')
        self.plot_on_Axis(self.axddZee, time[:-1], np.diff(drb[1])/(time[1]-time[0]), color= self.color_plt, label='EE z-Acc, ' + self.legend_suffix, xlabel='t in s', ylabel=r'$m/s^2$')
        self.plot_on_Axis(self.axddPee, time[:-1], np.diff(drb[2])/(time[1]-time[0]), color=self.color_plt, label='EE Angular-Acc of x-Axis, ' + self.legend_suffix, xlabel='t in s', ylabel=r'$rad/s^2$')

        # Plot Zustände
        if self.bool_elModel:
            self.plot_on_Axis(self.axy, time, im_y, color=self.color_plt, label='Current Motor Y, ' + self.legend_suffix, xlabel='t in s', ylabel=r'$A$')
            self.plot_on_Axis(self.axz, time, im_z, color=self.color_plt, label='Current Motor Z, ' + self.legend_suffix, xlabel='t in s', ylabel=r'$A$')
            self.plot_on_Axis(self.axp, time, im_p*param.kbemf_phi, color=self.color_plt, label='Current Motor RotX, ' + self.legend_suffix, xlabel='t in s', ylabel=r'$A$')

    def plot_on_Axis(self, ax, x, y, color='b', label=None, title=None, xlabel=None, ylabel=None, mve_end_vLine=False):
        if type(x) == list:
            L = len(x)
            for i in range(0, L):
                if type(label) == list and type(color) == list:
                    ax.plot(x[i], y[i], color[i], label=label[i])
                    ax.legend(loc='best')
                elif type(label) != list and type(color) == list:
                    ax.plot(x[i], y[i], color[i], label=label)
                    if label != None:
                        ax.legend(loc='best')
                elif type(label) != list and type(color) != list:
                    ax.plot(x[i], y[i], color, label=label)
                    if label != None:
                        ax.legend(loc='best')
        else:
            ax.plot(x, y, color, label=label)
            if label != None:
                ax.legend(loc='best')

        if title != None:
            ax.set_title(title)
        if xlabel != None:
            ax.set_xlabel(xlabel)
        if ylabel != None:
            ax.set_ylabel(ylabel)
        ax.grid(True)
        if mve_end_vLine:
            ax.axvline(self.T)

    def showPlots(self):
        if self.elModel_allSims==False:
            plt.close(self.fig_elCurrent)
        plt.show()

    def saveFigures(self):
        # Depreciated!!!
        if self.savefigs:
            self.fig_ee.savefig('../documentation/Latex/presentation/pics/endeffektor_' + self.suffix)
            self.fig_ee2.savefig('../documentation/Latex/presentation/pics/endeffektorZoom_' + self.suffix)
            self.fig_mt.savefig('../documentation/Latex/presentation/pics/posVerlaufAchsen_' + self.suffix)
            self.fig_sp.savefig('../documentation/Latex/presentation/pics/federn_' + self.suffix)
            self.fig_fo.savefig('../documentation/Latex/presentation/pics/force_' + self.suffix)
            self.fig_u.savefig('../documentation/Latex/presentation/pics/input_' + self.suffix)

    def calc_EE_Acceleration(self, oModel, q, dq, u):
        # 1) Berechnung von ddq = inv(M)*rhs in Abhängigkeit von q und dq
        # 2) Transformation der Achsbeschleunigungen auf den Endeffektor

        M = oModel.M
        rhs = oModel.rhs

        if self.bool_elModel:
            q = q[:-3, :]

        # q und dq verschachteln
        q_dq = np.zeros((len(q)*2, len(q[0])))
        for idx in range(0, len(q)):
            q_dq[2*idx] = q[idx]
            q_dq[2*idx + 1] = dq[idx]

        ddq = np.zeros((len(q), len(q[0])))
        for i in range(0, len(q[0])):
            cos_phi = np.cos(q[2][i])
            sin_phi = np.sin(q[2][i])
            arg = q_dq[:, i], u[i, :], cos_phi, sin_phi
            M_eval = M(*flatten(arg))
            oModel.Mi_eval[oModel.idx2_rigid] = np.linalg.inv(M_eval[oModel.idx2_rigid])
            rhs_eval = rhs(*flatten(arg))

            # Überschreibe Werte für starr geschaltete Federn
            if self.rigid[0]:
                rhs_eval[3] = 0.0
            if self.rigid[1]:
                rhs_eval[4] = 0.0
            if self.rigid[2]:
                rhs_eval[5] = 0.0
            if self.rigid[3]:
                rhs_eval[6] = 0.0
            # ddy, ddz, ddphi, ddtheta, ddalpha, ddg, ddrho
            ddq[:, i:i + 1] = np.matmul(oModel.Mi_eval, rhs_eval)

        # Vorwärtskineamtik für Achsbeschleunigungen
        dy_sym, dz_sym, dphi_sym, dtheta_sym, dalpha_sym, dg_sym, drho_sym = sp.symbols('dy dz dphi dtheta dalpha dg drho')
        ddy_sym, ddz_sym, ddphi_sym, ddtheta_sym, ddalpha_sym, ddg_sym, ddrho_sym = sp.symbols('ddy ddz ddphi ddtheta ddalpha ddg ddrho')
        dq_sym = sp.Matrix([[dy_sym], [dz_sym], [dphi_sym], [dtheta_sym], [dalpha_sym], [dg_sym], [drho_sym]])
        ddq_sym = sp.Matrix([[ddy_sym], [ddz_sym], [ddphi_sym], [ddtheta_sym], [ddalpha_sym], [ddg_sym], [ddrho_sym]])

        rb_sym, q_sym = oModel.forwardKinematics()
        Jac_rb = rb_sym.jacobian(q_sym)
        drb_sym = Jac_rb*dq_sym
        drb_lam = lambdify(flatten((q_sym, dq_sym)), drb_sym[0:2])

        H1 = dq_sym.transpose()*sp.hessian(rb_sym[0], q_sym)*dq_sym
        H2 = dq_sym.transpose()*sp.hessian(rb_sym[1], q_sym)*dq_sym
        Term_Hess = sp.Matrix([[H1[0]], [H2[0]]])
        J = Jac_rb*ddq_sym
        Term_Jac = sp.Matrix([[J[0]], [J[1]]])
        ddrb_sym = Term_Hess + Term_Jac
        ddrb_lam = lambdify(flatten((q_sym, dq_sym, ddq_sym)), ddrb_sym)

        drb_eval = np.zeros((3, len(q[0])))
        ddrb_eval = np.zeros((3, len(q[0])))
        for i in range(0, len(q[0])):
            drb_eval[0:2, i] = drb_lam(*flatten((q[:, i], dq[:, i])))
            drb_eval[2, i] = q[2, i] + q[3, i] + q[4, i] + q[6, i]
            ddrb_eval[0:2, i] = ddrb_lam(*flatten((q[:, i], dq[:, i], ddq[:, i])))[:, 0]
            ddrb_eval[2, i] = dq[2, i] + dq[3, i] + dq[4, i] + dq[6, i]

        return drb_eval, ddrb_eval, ddq

    def write_to_mat_file(self, rb, drb, ddrb):
        ee_mve = {'pos': rb, 'spd': drb, 'acc': ddrb}
        sio.savemat(st.path_mat + 'result_EE_mve_' + st.fname_mat, ee_mve)


if __name__ == "__main__":
    # Liste der zu plottenden Simulationen erstellen
    lstSuffix = st.lstSim_toPlot
    lstSuffix = [x for x in lstSuffix if x]

    oPlot = plot_results(lstSuffix)
    oPlot.createFigures()

    print('Plotting Simulation(s) ...')
    for z in lstSuffix:
        print('   ' + str(oPlot.idx) + '   ' + z)
        if oPlot.idx == 2:
            oPlot.color_ref = 'g--'
            oPlot.color_plt = 'm'
        if oPlot.idx == 3:
            oPlot.color_ref = 'g:'
            oPlot.color_plt = 'y'
        oModel, oTraj = oPlot.loadFile(z)
        q, dq, ddq, rb, drb, ddrb, err, ref, time = oPlot.calculateFromResults(oModel, oTraj)
        oPlot.plot_res_as_subplots(q, dq, ddq, rb, drb, ddrb, err, ref, time)
        del oModel
        del oTraj
        q, dq, ddq, rb, drb, ddrb, err, ref, time = None, None, None, None, None, None, None, None, None
        oPlot.idx += 1

    if st.save_EEmve_to_matFile:
        oPlot.write_to_mat_file(rb, drb, ddrb)

    oPlot.saveFigures()
    oPlot.showPlots()
