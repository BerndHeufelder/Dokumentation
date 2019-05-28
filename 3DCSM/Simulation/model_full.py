'''
Modell full.py
'''
import model_mech
import model_el
import settings as st


class Model:
    ''' Model full class '''
    def __init__(self):
        ''' Model full init class '''
        # Mechanisches Modell instanzieren
        self.mech = model_mech.Model(st.rigid)
        self.mech.feedback = st.feedback  # Closed oder Open-Loop Simulation

        # Elektrisches Modell instanzieren
        self.el = model_el.Model()

    def set_initial_values(self, oTraj):
        z0_mech = self.mech.set_initial_values(oTraj)

        if st.elModel:  # el. Anfangsgrößen anhand mech. Anfangsgrößen berechnen und setzen
            u, e = self.mech.calc_input(st.t0, z0_mech,  state_ctrl_mech=[0, 0, 0], oTraj=oTraj)
            i_m = self.el.trans_Force_to_motorCurrent(u)
            z0_el = self.el.set_initial_values(i_m)
        else:
            z0_el = []
        self.z0 = z0_mech + z0_el

    def system_equations(self, t, state, oTraj):

        # Eingang des mechanischen Teilsystems (Fy, Fz, Mphi) berechnen
        state_mech = state[0:14]
        state_ctrl_mech = state[14:17]
        u_mech, e_mech = self.mech.calc_input(t, state_mech, state_ctrl_mech, oTraj)

        if st.elModel:
            state_el = state[17:20]
            state_ctrl_el = state[20:23]
            dy, dz, dphi = state[1], state[3], state[5]  # Lin. & Winkelgeschw. aus mech. werden für el. Sys benötigt

            # Eingang des el. Systems (Zwischenkreisspannung) berechnen
            u_el, e_el = self.el.calc_input(state_el, state_ctrl_el, u_mech)
            Dz_el = self.el.sys_eq([dy, dz, dphi], state_el, u_el)

            # Umrechnung Motorströme in Axialkraft für Spindelschlitten und Moment für RotX
            [Fy, Fz, Mp] = self.el.trans_motorCurrent_to_Force(state_el)
            u_mech = [Fy, Fz, Mp]
            Dz_ctrl_el = e_el
        else:
            Dz_el = []
            Dz_ctrl_el = []

        Dz_mech = self.mech.sys_eq(t, u_mech, state_mech)
        Dz_ctrl_mech = e_mech

        # self.e_mech = e_mech
        # self.state_ctrl_mech = state_ctrl_mech

        # *** State Overview ***
        # Dz_mech:      state 0 bis 13
        # Dz_ctrl_mech: state 14 bis 16
        # Dz_el:        state 17 bis 19
        # Dz_ctrl_el:   state 20 bis 22
        Dz = Dz_mech + Dz_ctrl_mech + Dz_el + Dz_ctrl_el
        return Dz

