'''

'''

# -*- coding: utf-8 -*-
import model_full
import trajectory
import numpy as np
import pickle
import analysis
import settings as st
import matplotlib.pyplot as plt
from scipy.integrate import ode
import time


class Simulation:

    def __init__(self):
        self.dt = st.dt
        self.t0 = st.t0             # start time of movement
        self.T = st.t_mve + st.t0   # end time of movement
        self.num_steps = int((st.t_mve+st.t_set)/self.dt)  # Anzahl an Integrationsschritten, t_mve ist die Dauer der Bewegung

        # Gesamtmodell instanzieren
        self.oModel = model_full.Model()

        # Trajektorienobjekt erstellen
        if st.trajFromData:
            # Anfangswerte der Zustandsgrößen setzen
            self.y = 0.0
            self.z = 0.0
            self.phi = 0.0
            self.f0_rigid = 46.94
            self.oTraj = trajectory.Trajectory(st.t0, st.t_mve, self.y, self.z, self.phi, self.f0_rigid)
            self.oTraj.setTraj_fromData(self.T)
        else:
            # Start und Endpositinonen aller Achsen berechnen
            self.y, self.z, self.phi = self.oModel.mech.calc_axis_start_and_endposition(st.place_position_y, st.place_position_z, st.phi0, st.phiT)
            # Input-Shaping
            if st.inputShaping:
                self.f0_rigid = np.min(self.oModel.mech.eigen_frequencies([self.y[1], self.z[1], self.phi[1]], True))
            else:
                self.f0_rigid = 0
            # Trajektorien erzeugen
            self.oTraj = trajectory.Trajectory(st.t0, self.T, self.y, self.z, self.phi, self.f0_rigid)

        # Anfangswerte der Zustandsgrößen setzen
        self.oModel.set_initial_values(self.oTraj)
        # Arrays für Datenspeicherung erstellen
        self.res = np.empty([self.num_steps, len(self.oModel.z0)])
        self.u = np.empty([self.num_steps, 3])
        self.time = np.empty([self.num_steps, 1])
        self.ref = np.empty([self.num_steps, 9])  # storage variable for reference trajectories [y,dy,ddy,z,dz,ddz,phi,dphi,ddphi]
        # Anfangswerte in time und data array speichern
        self.res[0] = self.oModel.z0
        self.time[0] = self.t0
        self.u[0], e = self.oModel.mech.calc_input(self.t0, self.res[0], state_ctrl_mech=[0, 0, 0], oTraj=self.oTraj)
        y_traj, z_traj, phi_traj = self.oTraj.eval_all_trajectories(0.0)
        self.ref[0] = np.hstack([y_traj, z_traj, phi_traj])
        # ******************************************** init_end ********************************************

    def integrate(self):
        r = ode(self.oModel.system_equations).set_integrator('vode', rtol=1e-11, atol=1e-11)
        r.set_initial_value(self.oModel.z0, self.t0)
        r.set_f_params(self.oTraj)

        print('start integration...')
        time_s2 = time.time()
        i = 1  # iteration variable
        while r.successful() and i < self.num_steps-1:
            self.res[i] = r.integrate(r.t + self.dt)
            self.u[i] = self.oModel.mech.u
            self.time[i] = r.t
            y_traj, z_traj, phi_traj = self.oTraj.eval_all_trajectories(r.t)
            self.ref[i] = np.hstack([y_traj, z_traj, phi_traj])
            i += 1
            if (i % (self.num_steps//10)) == 1:
                print(np.round(r.t*1e6)/1e6)
        print('integration finished...')
        print('time:', time.time() - time_s2)
        # ******************************************** integrate_end ********************************************

    def save_results(self):
        # Simulationsergebnisse speichern
        result = {'time': self.time, 'res': self.res, 'input': self.u,
                  'trajData': [self.ref],
                  'model': {'rigid': self.oModel.mech.rigid},
                  'settings': {'bool_elModel': st.elModel, 'T': self.T}}
        with open(st.path_pickle + 'Simulation_result_' + st.suffix_sim + '.pickle', 'wb') as outf:
            outf.write(pickle.dumps(result))
            outf.close()

    def analyse_mech_eigenfreq(self):
        # Eigenfrequenzanalyse durchführen
        if st.plotAnalysis:
            analysis.analyse_eigenfrequencies(self.oModel.mech, st.place_position_y, st.place_position_z)
        plt.show()

if __name__ == "__main__":
    oSim = Simulation()
    oSim.integrate()
    oSim.save_results()
    oSim.analyse_mech_eigenfreq()
# plt.plot(oSim.time[:-1], oSim.e_mech[:-1], label='e_phi')
# plt.plot(oSim.time[:-1], oSim.state_ctrl_mech[:-1]*300, label='e_phi_int')
# plt.legend()
# plt.show()