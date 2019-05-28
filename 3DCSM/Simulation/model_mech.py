# -*- coding: utf-8 -*-
import numpy as np
import sympy as sp
import control
import param
from sympy.utilities.iterables import flatten
from sympy.utilities.lambdify import lambdify
import pickle
import settings as st


class Model:

    def __init__(self, rigid=[True, True, True, True]):
        # Gespeicherte Sympy Ausdrücke laden
        with open(st.path_pickle + 'M_matrix_' + st.suffix_eqs + '.pickle', 'rb') as inf:
            M_sym = pickle.loads(inf.read())
        with open(st.path_pickle + 'rhs_vector_' + st.suffix_eqs + '.pickle', 'rb') as inf:
            rhs_sym = pickle.loads(inf.read())

        # Gespeicherte Sympy Ausdrücke laden
        with open(st.path_pickle + 'M_matrix_rigid_' + st.suffix_eqs + '.pickle', 'rb') as inf:
            M_sym_rigid = pickle.loads(inf.read())
        with open(st.path_pickle + 'rhs_vector_rigid_' + st.suffix_eqs + '.pickle', 'rb') as inf:
            rhs_sym_rigid = pickle.loads(inf.read())

        self.M_sym = sp.Matrix(M_sym)
        self.rhs_sym = sp.Matrix(rhs_sym)
        self.rigid = rigid
        self.feedback = False

        # sympy Funktion in lambdify Funktion umwandeln
        self.M, self.rhs = self.gen_lam_from_sympyExpr(self.M_sym, self.rhs_sym)
        self.M_rigid, self.rhs_rigid = self.gen_lam_from_sympyExpr(M_sym_rigid, rhs_sym_rigid)
        self.inv_model = self.gen_inv_model_from_sympyExpr(M_sym_rigid, rhs_sym_rigid)

        # PID regler instanzieren
        self.ctrl_z = control.PID_traj()
        self.ctrl_y = control.PID_traj()
        self.ctrl_phi = control.PID_traj()

        # Inverse Massenmatrix als Einheitsmatrix instanzieren, während der Simulation werden nur die nicht
        # starren FG überschrieben, diese Indices für welche die Massenmatrix evaluiert werden soll,
        # werden in idx2_rigid gespeichert
        rigid = self.rigid[0:(self.M_sym.shape[0] - 3)]  # bool var für die Federn [theta, alpha, rho, g]
        idx = np.hstack(([True, True, True], np.logical_not(rigid)))  # ersten 3 True's für die FG [y,z,phi]
        dof = idx.shape[0]  # Anzahl der FG
        self.Mi_eval = np.eye(dof)
        self.idx2_rigid = np.ix_(idx, idx)  # index array in Abhängigkeit der pos. der true's erstellen

        # Umrechnung EE-Kraft auf Gelenksmomente
        self.JacT_lam = self.JacT_forwardKinematics()
        # end init

    def set_initial_values(self, oTraj):
        # initial values
        if st.trajFromData:
            y0 = float(oTraj.f1yPos(0))
            dy0 = float(oTraj.f1ySpd(0))
            z0 = float(oTraj.f1zPos(0))
            dz0 = float(oTraj.f1zSpd(0))
            phi0 = float(oTraj.f1pPos(0))
            dphi0 = float(oTraj.f1pSpd(0))
        else:
            y0 = oTraj.eval_y_traj(0.0)[0]
            dy0 = oTraj.eval_y_traj(0.0)[1]
            z0 = oTraj.eval_z_traj(0.0)[0]
            dz0 = oTraj.eval_z_traj(0.0)[1]
            phi0 = oTraj.eval_phi_traj(0.0)[0]
            dphi0 = oTraj.eval_phi_traj(0.0)[1]
        theta0 = 0.0
        dtheta0 = 0.0
        alpha0 = 0.0
        dalpha0 = 0.0
        g0 = 0.0
        dg0 = 0.0
        rho0 = 0.0
        drho0 = 0.0
        ey_int0 = 0.0
        ez_int0 = 0.0
        ep_int0 = 0.0
        z0 = [y0, dy0, z0, dz0, phi0, dphi0, theta0, dtheta0, alpha0, dalpha0,
              g0, dg0, rho0, drho0, ey_int0, ez_int0, ep_int0]
        return z0

    def gen_inv_model_from_sympyExpr(self, M_sym, rhs_sym):
        # Inverses starres Modell um aus den Refernzwerten y_ref, z_ref, phi_ref die benötigten Kräfte und Momente zu
        # berechnen
        z1, z2, z3, z4, z5, z6 = sp.symbols('z1 z2 z3 z4 z5 z6')
        z = z1, z2, z3, z4, z5, z6
        u0, u1, u2 = sp.symbols('u0 u1 u2')
        u = u0, u1, u2
        cos_phi, sin_phi = sp.symbols('cos_phi sin_phi')

        # Lambdify-Funktionen für inverses Model und rechte Seite
        M_lam = lambdify(flatten([z5, cos_phi, sin_phi]), M_sym, "numpy")
        rhs_lam = lambdify(flatten([z, u, cos_phi, sin_phi]), rhs_sym, [{'ImmutableMatrix': np.array}, 'numpy'])
        inv_model = lambda y: np.dot(M_lam(y[2, 0], np.sin(y[2, 0]), np.cos(y[2, 0])), y[:, 2]) - rhs_lam(
            *flatten([y[:, 0:2], 0.0, 0.0, 0.0, np.sin(y[2, 0]), np.cos(y[2, 0])]))[:, 0]
        return inv_model

    def gen_lam_from_sympyExpr(self, M_sym, rhs_sym):
        # Eingangs- und Zustandsvektor definieren
        z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13, z14 = sp.symbols('z1 z2 z3 z4 z5 z6 z7 z8 z9 z10 z11 z12 z13 z14')
        z = z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13, z14
        u0, u1, u2 = sp.symbols('u0 u1 u2')
        u = u0, u1, u2
        cos_phi, sin_phi, phi = sp.symbols('cos_phi sin_phi phi')
        # Lambdify-Funktionen für Massenmatrix und rechte Seite
        M_lam = lambdify(flatten([z, u, cos_phi, sin_phi]), M_sym, "numpy")
        rhs_lam = lambdify(flatten([z, u, cos_phi, sin_phi]), rhs_sym, [{'ImmutableMatrix': np.array}, 'numpy'])

        return M_lam, rhs_lam

    def sys_eq(self, t, u, state):  # integrations und modellgleichungs funktion

        # cos & sin phi auswerten
        cos_phi = np.cos(state[4])
        sin_phi = np.sin(state[4])

        arg = state, u, cos_phi, sin_phi

        # Massenmatrix und rechte Seite für momentane
        # Zustands- und Eingangsgröße zur Zeit t evaluieren (eval erfolgt nur für die nicht starr geschalteten FG)
        M_eval = self.M(*flatten(arg))
        self.Mi_eval[self.idx2_rigid] = np.linalg.inv(M_eval[self.idx2_rigid])
        rhs_eval = self.rhs(*flatten(arg))

        # ******************** TEST ********************
        if t > st.t_force and st.externalForce:
            JointCoord = state[0], state[2], state[4], state[6], state[8], 0, 0
            JacT = self.JacT_lam(*flatten(JointCoord))
            F_ee_g0 = np.array([[0], [-20]])
            angle_ee = state[4] + state[8]
            F_ee_gee = np.matmul(np.array([[np.cos(angle_ee), -np.sin(angle_ee)], [np.sin(angle_ee), np.cos(angle_ee)]]), F_ee_g0)
            F_joint = np.matmul(JacT, F_ee_gee)
            F_joint = np.append(F_joint, np.zeros((2, 1)))
            rhs_eval[:, 0] = rhs_eval[:, 0] + F_joint
        # ******************** TEST ********************

        ddq = np.zeros((7, 1))
        n = self.M_sym.shape[0]
        ddq[0:n, :] = np.dot(self.Mi_eval, rhs_eval)  # Inverse mit rechter Seite multiplizieren

        # DGL-System 1. Ordnung
        Dz1 = state[1]
        Dz2 = ddq[0]
        Dz3 = state[3]
        Dz4 = ddq[1]
        Dz5 = state[5]
        Dz6 = ddq[2]

        if self.rigid[0]:
            Dz7 = -state[6]
            Dz8 = -state[7]
        else:
            Dz7 = state[7]
            Dz8 = ddq[3]
        if self.rigid[1]:
            Dz9 = -state[8]
            Dz10 = -state[9]
        else:
            Dz9 = state[9]
            Dz10 = ddq[4]
        if self.rigid[2]:
            Dz11 = -state[10]
            Dz12 = -state[11]
        else:
            Dz11 = state[11]
            Dz12 = ddq[5]

        if self.rigid[3]:
            Dz13 = -state[12]
            Dz14 = -state[13]
        else:
            Dz13 = state[13]
            Dz14 = ddq[6]

        # Eingangsgrößen zwischenspeichern
        self.u = u

        # return mechanical state
        Dz = [Dz1, Dz2, Dz3, Dz4, Dz5, Dz6, Dz7, Dz8, Dz9, Dz10, Dz11, Dz12, Dz13, Dz14]
        return Dz

    def calc_input(self, t, state_mech, state_ctrl_mech, oTraj):
        y_traj, z_traj, phi_traj = oTraj.eval_all_trajectories(t)

        e_y = state_mech[0:2] - y_traj[0:2]
        e_z = state_mech[2:4] - z_traj[0:2]
        e_phi = state_mech[4:6] - phi_traj[0:2]

        if self.feedback:
            y_traj[2] += self.ctrl_y.control(e_y, state_ctrl_mech[0])
            z_traj[2] += self.ctrl_z.control(e_z, state_ctrl_mech[1])
            phi_traj[2] += self.ctrl_phi.control(e_phi, state_ctrl_mech[2])
        # inverse starres Modell: aus [pos,spd,acc] der Achsen [y,z,phi] können [Fy,Fz,Mp] berechnet werden Q=M*ddq+rhs
        u = self.inv_model(np.array([y_traj, z_traj, phi_traj]))
        e = [e_y[0], e_z[0], e_phi[0]]
        return u, e

    def eigen_frequencies(self, q, rigid_joints=True):
        # return Eigenfrequencies
        # rigid: aktuierte Achsen starr  (True) oder kräftefrei (False) setzen
        # q=(y,z,phi)$ Werte der Gelenkskoordinaten

        rigid = self.rigid[0:(self.M_sym.shape[0] - 3)]
        if rigid_joints:
            idx = np.hstack(([False, False, False], np.logical_not(rigid)))
        else:
            idx = np.hstack(([True, True, True], np.logical_not(rigid)))
        idx2 = np.ix_(idx, idx)
        # compute reduced symbolic matrices
        z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13, z14, cos_phi, sin_phi, u1, u2, u3 = sp.symbols(
            "z1 z2 z3 z4 z5 z6 z7 z8 z9 z10 z11 z12 z13 z14 cos_phi sin_phi u1 u2 u3")
        q_sym = [z1, z3, z5, z7, z9, z11, z13]
        M0_sym = self.M_sym.subs([(cos_phi, sp.cos(z5)), (sin_phi, sp.sin(z5))])
        rhs0 = self.rhs_sym.subs(
            [(z2, 0), (z4, 0), (z6, 0), (z8, 0), (z10, 0), (z12, 0), (z14, 0), (u1, 0), (u2, 0), (u3, 0),
             (cos_phi, sp.cos(z5)), (sin_phi, sp.sin(z5))])

        rhslin_sym = sp.Matrix(rhs0).jacobian(q_sym)
        # lambdify
        M0_lam = sp.lambdify(flatten([z1, z3, z5]), M0_sym, [{'ImmutableMatrix': np.array}, 'numpy'])
        rhslin_lam = sp.lambdify(flatten([z1, z3, z5]), rhslin_sym, [{'ImmutableMatrix': np.array}, 'numpy'])

        M0 = M0_lam(*flatten(q))[idx2]
        M0i = np.linalg.inv(M0)
        rhslin = rhslin_lam(*flatten(q))[idx2]
        rhslin = np.array(rhslin, dtype=float)
        A = np.dot(M0i, rhslin)
        eigs = np.linalg.eigvals(A)
        f = np.sqrt(-eigs)/2/np.pi
        return f

    def forwardKinematics(self):
        # Berechnung der EE-Pos anhand der Gelenkwinkel und Schlittenpositionen

        # Alternative mit DH-Transformation
        # symbolische Erstellung der Transformationsmatrix vom Ursprung zum Body
        phi_sym, alpha_sym, y_sym, z_sym, theta_sym, rho_sym, g_sym = sp.symbols('phi alpha y z theta rho g')
        q_sym = y_sym, z_sym, phi_sym, theta_sym, alpha_sym, g_sym, rho_sym
        T_oy = sp.Matrix([[sp.cos(alpha_sym), -sp.sin(alpha_sym), 0],
                          [sp.sin(alpha_sym), sp.cos(alpha_sym), param.zfeder_z],
                          [0, 0, 1]])
        T_om = sp.Matrix([[-sp.cos(phi_sym), sp.sin(phi_sym), y_sym + param.rx_y0],
                          [sp.sin(phi_sym), sp.cos(phi_sym), param.rx_z0 + z_sym],
                          [0, 0, 1]])
        T_md = sp.Matrix([[sp.cos(theta_sym), -sp.sin(theta_sym), param.lx_y],
                          [sp.sin(theta_sym), sp.cos(theta_sym), param.lx_z],
                          [0, 0, 1]])
        T_dw = sp.Matrix([[sp.cos(rho_sym), -sp.sin(rho_sym), 0],
                          [sp.sin(rho_sym), sp.cos(rho_sym), param.lw],
                          [0, 0, 1]])
        T_wb = sp.Matrix([[1, 0, param.dy],
                          [0, 1, g_sym + param.lb],
                          [0, 0, 1]])
        T = T_oy*T_om*T_md*T_dw*T_wb
        p0 = sp.Matrix([[0], [0], [1]])  # Punkt im Ursprung
        rb_sym = T*p0

        return rb_sym, q_sym

    def JacT_forwardKinematics(self):
        # Transponierte Jakobi Matrix für Umrechnung der Kraft am EE auf Gelenkskräfte F_joint = J^T*F_ee
        rb_sym, q_sym = self.forwardKinematics()
        phi_sym, alpha_sym, y_sym, z_sym, theta_sym, rho_sym, g_sym = sp.symbols('phi alpha y z theta rho g')
        JointCoord_sym = y_sym, z_sym, phi_sym, theta_sym, alpha_sym, g_sym, rho_sym
        Jac_T_sym = (rb_sym[0:2, 0].jacobian([y_sym, z_sym, phi_sym, theta_sym, alpha_sym]).transpose())
        Jac_T_lam = lambdify(flatten(JointCoord_sym), Jac_T_sym)
        return Jac_T_lam

    def eval_forwardKinematics(self, q):
        # Evaluierung des EE-Punktes mit den simulierten Verläufen der Zustandsgrößen
        y = q[0, :]
        z = q[1, :]
        phi = q[2, :]
        theta = q[3, :]
        alpha = q[4, :]
        g = q[5, :]
        rho = q[6, :]

        rb_sym, q_sym = self.forwardKinematics()
        phi_sym, alpha_sym, y_sym, z_sym, theta_sym, rho_sym, g_sym = sp.symbols('phi alpha y z theta rho g')
        JointCoord = y_sym, z_sym, phi_sym, theta_sym, alpha_sym, rho_sym, g_sym
        rb_lam = lambdify(JointCoord, rb_sym)

        rb_eval = np.zeros((3, len(phi)))
        for i in range(0, len(phi)):
            eval_arg = y[i], z[i], phi[i], theta[i], alpha[i], rho[i], g[i]
            rb_vec_hom = rb_lam(*flatten(eval_arg))
            rb_eval[0, i] = rb_vec_hom[0]
            rb_eval[1, i] = rb_vec_hom[1]
            rb_eval[2, i] = phi[i] + theta[i] + alpha[i] + rho[i]
        return rb_eval

    def forwardKinematicsRigid(self, q):
        y = q[0, :]
        z = q[1, :]
        phi = q[2, :]

        # Alternative mit DH-Transformation
        # symbolische Erstellung der Transformationsmatrix vom Ursprung zum Body
        phi_sym, y_sym, z_sym = sp.symbols('phi y z')
        T_om = sp.Matrix([[-sp.cos(phi_sym), sp.sin(phi_sym), param.rx_y0 + y_sym],
                          [sp.sin(phi_sym), sp.cos(phi_sym), param.rx_z0 + param.zfeder_z + z_sym],
                          [0, 0, 1]])
        T_mb = sp.Matrix([[1, 0, param.lx_y + param.dy],
                          [0, 1, param.lw + param.lb + param.lx_z],
                          [0, 0, 1]])
        T = T_om*T_mb
        p0 = sp.Matrix([[0], [0], [1]])  # Punkt im Ursprung
        rb = T*p0

        # Evaluierung des EE-Punktes mit den simulierten Verläufen der Zustandsgrößen
        arg_T = phi_sym, y_sym, z_sym
        rb_lam = lambdify(arg_T, rb)
        rb_eval = np.zeros((2, len(phi)))
        for i in range(0, len(phi)):
            eval_arg = phi[i], y[i], z[i]
            rb_vec_hom = rb_lam(*flatten(eval_arg))
            rb_eval[0, i] = rb_vec_hom[0]
            rb_eval[1, i] = rb_vec_hom[1]

        return rb_eval

    def inverseKinematicsRigid(self, yb, zb, phib):
        phi_sym = sp.symbols('phi')

        # symbolische Erstellung der Transformationsmatrix vom Ursprung zum Body
        T_om = sp.Matrix([[-sp.cos(phi_sym), sp.sin(phi_sym), param.rx_y0],
                          [sp.sin(phi_sym), sp.cos(phi_sym), param.rx_z0 + param.zfeder_z],
                          [0, 0, 1]])
        T_mb = sp.Matrix([[1, 0, param.lx_y + param.dy],
                          [0, 1, param.lw + param.lb + param.lx_z],
                          [0, 0, 1]])
        T = T_om*T_mb

        p0 = sp.Matrix([[0], [0], [1]])  # Punkt im Ursprung
        rb = T*p0

        # Evaluierung des EE-Punktes mit den simulierten Verläufen der Zustandsgrößen
        arg_T = phi_sym
        rb_lam = lambdify(arg_T, rb)
        rb_vec_hom = rb_lam(phib)
        y = yb - rb_vec_hom[0, 0]
        z = zb - rb_vec_hom[1, 0]
        return (y, z)

    # def posOffset_EE_gravity(self, y, z, phi, place_position_y, place_position_z):
    #     y0 = y[1]
    #     dy0 = 0.0
    #     z0 = z[1]
    #     dz0 = 0
    #     phi0 = phi[1]
    #     dphi0 = 0.0
    #
    #     test_arg = [y0, dy0, z0, dz0, phi0, dphi0]  # initial state
    #     Tges = self.Torque_ges_lam(*flatten(test_arg))
    #     print('Total Torque at z-spring: ', Tges, 'Nm')
    #     alpha_stat = Tges/param.kzr
    #     print('z-Spring angle as consequence of system weight', alpha_stat, 'rad')
    #     q = np.array([[y0], [z0], [phi0], [0], [alpha_stat], [0], [0]])
    #     dev_EE = self.eval_forwardKinematics(q)
    #     q = np.array([[y0], [z0], [phi0], [0], [0], [0], [0]])
    #     exa_EE = self.eval_forwardKinematics(q)
    #     err_EE = dev_EE - exa_EE
    #     print('Necessary Endeffector offset: y=', err_EE[0, 0], ' z=', err_EE[1, 0])
    #
    #     print('Place position and angle before correction', place_position_y, place_position_z, phi[1], 'm,m,rad')
    #     place_position_y += err_EE[0, 0]
    #     place_position_z += err_EE[1, 0]
    #     phi[1] += alpha_stat
    #     print('Place position and angle after correction:', place_position_y, place_position_z, phi[1], 'm,m,rad')
    #
    #     (y[1], z[1]) = self.inverseKinematicsRigid(place_position_y, place_position_z, phi[1])
    #     return y, z, phi

    def calc_axis_start_and_endposition(self, place_position_y, place_position_z, phi0, phiT):
        # Winkelverlauf RotX vorgeben
        phi = np.array([phi0, phiT])*(np.pi/180)
        # Gelenkskoordinaten y- und z-Achse abhängig von EE-Pos & -orientieurng berechnen
        (y0, z0) = self.inverseKinematicsRigid(place_position_y, place_position_z, phi[0])
        (yT, zT) = self.inverseKinematicsRigid(place_position_y, place_position_z, phi[1])
        y = np.array([y0, yT])
        z = np.array([z0, zT])

        return y, z, phi
