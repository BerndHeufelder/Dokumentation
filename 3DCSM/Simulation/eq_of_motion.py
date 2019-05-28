# -*- coding: utf-8 -*-
'''

'''
import sympy as sp
import param
import pickle
import settings as st

if __name__ == "__main__":
    calc_elastic = True
    calc_rigid = True

    # Define coordinates and other variables for both elastic and rigid system
    print("1. Define Variables")
    e1 = sp.Matrix([1, 0])
    e2 = sp.Matrix([0, 1])
    t = sp.symbols('t', real=True)
    # Definition der Symbole
    kwr, kwt, kbr, kzr, kxt = sp.symbols('k_w k_wt k_br k_alpha kxt')  # Federkonstanten
    Mz, My, Mx, Md, Mb = sp.symbols('m_z m_y m_x m_d m_b')  # Massen
    Jd, Jx, Jb, Jz, Jy = sp.symbols('J_d J_x J_b J_z J_y')  # Trägheitsmomente
    rx_z0, rx_y0, lx_y, lx_z, lw, lb, lcomx_y, lcomx_z, lcomw, lcomz0, lcomy_y0, lcomy_z0, lfederz = sp.symbols(
        'rx_z0 rx_y0 lx_y lx_z l_w l_b l_comx_y l_comx_z l_com_w l_comz l_comy_y0 l_comy_z0 l_federz')  # Längen
    cos_phi, sin_phi = sp.symbols('cos_phi sin_phi')
    grav = 9.81  # m/s^2

    # Freiheitsgrade in y und z-Richtung
    y = sp.Function('y')(t)
    z = sp.Function('z')(t)
    # Bewegung der pi-Feder
    alpha = sp.Function('alpha')(t)
    dalpha = sp.diff(alpha, t)
    # Motorwinkel
    phi = sp.Function('varphi')(t)
    # Winkel aufgrund Elastizität
    theta = sp.Function('theta')(t)
    dtheta = sp.diff(theta, t)
    # Translatorische Bewegung des Dornhalters
    g = sp.Function('g')(t)
    dg = sp.diff(g, t)
    # Biegefeder für Bodyholder
    rho = sp.Function('rho')(t)
    drho = sp.diff(rho, t)
    # Kraft auf y-Schlitten
    Fy = sp.Function('F_y')(t)
    # Kraft auf z-Schlitten
    Fz = sp.Function('F_z')(t)
    # Moment im Drehgelenk beim Winkel phi
    Mp = sp.Function('M_phi')(t)


def gen_eqs_of_motion():
    print('2. Calculate elastic equations of motion')

    # Einheitsvektoren z-Achse
    e1z = e1*sp.cos(alpha) + e2*sp.sin(alpha)
    e2z = -e1*sp.sin(alpha) + e2*sp.cos(alpha)

    # Einheitsvektoren Motorkoordinatensystem (Achtung: Definition der Koordinatensysteme beachten)
    e1m = -e1z*sp.cos(phi) + e2z*sp.sin(phi)
    e2m = e1z*sp.sin(phi) + e2z*sp.cos(phi)

    # Einheitsvektoren Dornkoordinatesystem
    e1d = e1m*sp.cos(theta) + e2m*sp.sin(theta)
    e2d = -e1m*sp.sin(theta) + e2m*sp.cos(theta)

    # Einheitsvektoren Bodyholderkoordinatensystem
    e1b = e1d*sp.cos(rho) + e2d*sp.sin(rho)
    e2b = -e1d*sp.sin(rho) + e2d*sp.cos(rho)

    # Koordinatentransformationen (Positionen)
    rx = (rx_z0 + z)*e2z + (rx_y0 + y)*e1z  # Position rotX, rx_z0,rx_y0 sind nominale Abstände zwischen Feder und rotX
    rd = rx + lx_y*e1m + lx_z*e2m  # Position Lager 1 RotBody
    rw = rd + lw*e2d  # Position DornEnde (Werkstückhalter), Konusverbindung
    rb = rw + (g + lb)*e2b

    # Position Massenschwerpunkte in Bezug auf globales Koordinatensystem
    comz = (lcomz0 + z)*e2z  # Position Massenschwerpunkt z-Schlitten inklusive Stator y-Schlitten
    comy = rx + lcomy_y0*e1z + lcomy_z0*e2z  # Position Massenschwerpunkt y-Schlitten inklusive Lager rotX
    # Achtung:  Abstände lcom_y, lcom_z sind relativ zu rotX

    comx = rx + lcomx_y*e1m + lcomx_z*e2m  # Position Massenschwerpunkt Rot-X System
    comd = rd + lcomw*e2d  # Position Massenschwerpunkt Dorn
    comb = rb

    # Koordinatentransformationen (Geschwindigkeiten)
    vcomz = sp.diff(comz, t)
    vcomd = sp.diff(comd, t)
    vcomx = sp.diff(comx, t)
    vcomy = sp.diff(comy, t)
    vcomb = sp.diff(comb, t)

    # Kinetische Energien
    Tt = Mz/2*sp.DotProduct(vcomz, vcomz) + \
         My/2*sp.DotProduct(vcomy, vcomy) + \
         Mx/2*sp.DotProduct(vcomx, vcomx) + \
         Md/2*sp.DotProduct(vcomd, vcomd) + \
         Mb/2*sp.DotProduct(vcomb, vcomb)
    Tr = Jz/2*sp.diff(alpha, t)**2 + \
         Jy/2*sp.diff(alpha, t)**2 + \
         Jx/2*sp.diff(phi + alpha, t)**2 + \
         Jd/2*sp.diff(phi + alpha + theta, t)**2 + \
         Jb/2*sp.diff(phi + alpha + theta + rho, t)**2  # Kinetische Energie für um alle 3 rot. Freiheitsgrade
    T = Tt + Tr

    # Potentielle Energien
    U_dyn = kwr*theta**2/2 + kzr*alpha**2/2 + kwt*g**2/2 + kbr*rho**2/2
    # ***** Berechnung RotX Kompensation *****
    if st.rotX_Spring:
        lx_spring, _ = get_compensationTorque_from_phi(phi)
        U_dyn += kxt*lx_spring**2/2

    U_grav = grav*(Mz*comz + My*comy + Mx*comx + Md*comd + Mb*comb)  # Potentielle Energien durch das Schwerefeld
    U = U_dyn + U_grav[1, 0]
    # END: Potentielle Energien

    # Lagrange Funktion
    L = (T - U)

    # Generalisierte Koordinaten und Ableitung
    q = sp.Matrix([y, z, phi, theta, alpha, g, rho])
    dq = q.diff(t)  # 1. Ableitung der verallgemeinerte Koordinaten

    print("2.1. Compute Mass-Matrix")
    # Matrizenberechnung
    M = sp.hessian(L, dq)  # Massenmatrix

    print("2.2. Compute RHS")
    tmp = sp.Matrix([L]).jacobian(dq)
    C = tmp.transpose().jacobian(q)
    F = sp.Matrix([L]).jacobian(q).transpose()
    Q = sp.Matrix([[Fy], [Fz], [Mp], [0], [0], [0], [0]])  # Verallgemeinerte Kräfte/Momente

    # Dissipationsfunktionen
    Dalpha = 0  # 1/2*param.dalpha*sp.diff(s, t)**2  # Translatorische Dämpfung des Unterbaus
    Dwt = 0  # 1/2*param.dwt*sp.diff(g, t)**2  # Translatorische Dämpfung des Bodyholders
    Dwr = 0  # 1/2*param.dwr*sp.diff(theta, t)**2  # Rotationsdämpfung des Werkzeughalters
    Dbr = 0  # 1/2*param.dbr*sp.diff(rho, t)**2  # Rotationsdämpfung des Bodyholders
    Dzr = 0  # 1/2*param.dzr*sp.diff(z, t)**2
    D = 0  # Ds + Dwt + Dwr + Dbr
    DMat = sp.Matrix([D]).jacobian(dq).transpose()

    # Gleichung für rechte Seite aufstellen: M*ddq = -C*dq + F + Q - D
    dqt = q.diff(t)
    rhs = -C*dqt + F + Q - DMat

    # Systemgrößen als Ersatz der generalisierten Koordinaten und Kräfte definieren
    u0, u1, u2 = sp.symbols('u0 u1 u2')
    z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13, z14 = sp.symbols(
        'z1 z2 z3 z4 z5 z6 z7 z8 z9 z10 z11 z12 z13 z14')
    dz1, dz2, dz3, dz4, dz5, dz6, dz7, dz8, dz9, dz10, dz11, dz12, dz13, dz14 = sp.symbols(
        'dz1 dz2 dz3 dz4 dz5 dz6 dz7 dz8 dz9 dz10 dz11 dz12 dz13 dz14')
    rep_sys = [(sp.diff(y, t, t), dz2), (sp.diff(y, t), z2),
               (sp.diff(z, t, t), dz4), (sp.diff(z, t), z4),
               (sp.diff(phi, t, t), dz6), (sp.diff(phi, t), z6),
               (sp.diff(theta, t, t), dz8), (sp.diff(theta, t), z8),
               (sp.diff(alpha, t, t), dz10), (sp.diff(alpha, t), z10),
               (sp.diff(g, t, t), dz12), (sp.diff(g, t), z12),
               (sp.diff(rho, t, t), dz14), (sp.diff(rho, t), z14),
               (y, z1), (z, z3), (phi, z5), (theta, z7), (alpha, z9), (g, z11), (rho, z13),
               (Fy, u0), (Fz, u1), (Mp, u2)]

    # Parameterwerte aus param.py beziehen
    rep_param = [(My, param.My), (Mx, param.Mx), (Md, param.Md), (Mb, param.Mb), (Mz, param.Mz),
                 (rx_z0, param.rx_z0), (rx_y0, param.rx_y0), (lx_z, param.lx_z), (lx_y, param.lx_y), (lw, param.lw),
                 (lb, param.lb),
                 (lcomx_z, param.lcomx_z), (lcomx_y, param.lcomx_y), (lcomw, param.lcomw), (lcomz0, param.lcomz0),
                 (lcomy_z0, param.lcomy_z0), (lcomy_y0, param.lcomy_y0),
                 (Jd, param.Jd), (Jx, param.Jx), (Jz, param.Jz), (Jy, param.Jy), (Jb, param.Jb),
                 (kwr, param.kwr), (kwt, param.kwt), (kzr, param.kzr), (kbr, param.kbr), (kxt, param.kxt)]

    # Linearisierung bezüglich der elastischen Koordinaten
    print("2.3. Approximate for small elastic Deflections")
    taylor_subs = [(rho, 0), (theta, 0), (g, 0), (alpha, 0)]
    small_variables = sp.Matrix([theta, dtheta, alpha, dalpha, g, dg, rho, drho])
    rhs_taylor_const = rhs.subs(taylor_subs).doit().expand()
    rhs_taylor_lin = rhs.jacobian(small_variables).subs(taylor_subs).doit().expand()
    M_taylor = M.subs(taylor_subs).doit().expand()

    print("2.4. Substitute Parameters")
    rhs_taylor_const = (rhs_taylor_const.subs(rep_param))
    rhs_taylor_lin = (rhs_taylor_lin.subs(rep_param))
    M_taylor = (M_taylor.subs(rep_param))
    rhs = rhs_taylor_const + rhs_taylor_lin*small_variables

    # Parameter und neue Systemgrößen einsetzen
    rhs = sp.expand_trig(rhs).subs([(sp.cos(phi), cos_phi), (sp.sin(phi), sin_phi)])
    rhs = rhs.subs(rep_sys)
    M_taylor = sp.expand_trig(M_taylor).subs([(sp.cos(phi), cos_phi), (sp.sin(phi), sin_phi)])
    M = M_taylor.subs(rep_sys)
    M = sp.simplify(sp.expand(M))
    rhs = sp.expand(rhs)

    with open(st.path_pickle + 'M_matrix_' + st.suffix_eqs + '.pickle', 'wb') as outf:
        outf.write(pickle.dumps(M))
        outf.close()
    with open(st.path_pickle + 'rhs_vector_' + st.suffix_eqs + '.pickle', 'wb') as outf:
        outf.write(pickle.dumps(rhs))
        outf.close()


def get_compensationTorque_from_phi(phi):
    L0 = param.L0  # Einbaulaenge in m
    l_P1 = param.l_P1  # Abstand vom Drehpunkt auf der RotX-Achse in m
    l_P2 = param.l_P2  # Abstand vom Drehpunkt am Gehäuse in m
    l_eff_0deg = param.l_eff_0deg  # Einbaulaenge bei RotX=0° in m
    phi_offset = param.phi_offset
    l_p1p2_15deg = param.l_p1p2_15deg

    cos_phiOff_phi = sp.cos(phi_offset)*sp.cos(phi) - sp.sin(phi_offset)*sp.sin(phi)
    sin_phiOff_phi = sp.sin(phi_offset)*sp.cos(phi) + sp.sin(phi)*sp.cos(phi_offset)
    P1 = sp.Matrix([[l_P1*cos_phiOff_phi], [-l_P1*sin_phiOff_phi]])
    P2 = sp.Matrix([[0.0], [-l_P2]])
    P2toP1 = P1 - P2
    l_P2toP1 = sp.sqrt(P2toP1[0, 0]**2 + P2toP1[1, 0]**2)
    l_delta = l_p1p2_15deg - l_P2toP1
    P1_rot90 = sp.Matrix([[P1[1, 0]], [-P1[0, 0]]])
    P2toP1_dir = - P2toP1
    P1_rot90_abs = sp.sqrt(P1_rot90[0, 0]**2 + P1_rot90[1, 0]**2)
    P2toP1_dir_abs = sp.sqrt(P2toP1_dir[0, 0]**2 + P2toP1_dir[1, 0]**2)
    beta_cos = sp.MatMul(P1_rot90.transpose(), P2toP1_dir).doit()[0, 0]/(P1_rot90_abs*P2toP1_dir_abs)
    lx_spring = (L0 - l_eff_0deg - l_delta)

    if __name__ == "__main__":
        F_res = kxt*lx_spring
        F_normal = F_res*beta_cos
        Mp_spring = F_normal*l_P1
        return lx_spring, Mp_spring
    else:
        return lx_spring


def gen_eqs_of_motion_rigid():
    print('3. Calculate rigid equations of motion')

    # Einheitsvektoren Motorkoordinatensystem
    e1m = -e1*sp.cos(phi) + e2*sp.sin(phi)
    e2m = e1*sp.sin(phi) + e2*sp.cos(phi)

    # Koordinatentransformationen (Positionen)
    rx = (rx_z0 + z)*e2 + (rx_y0 + y)*e1
    rd = rx + lx_y*e1m + lx_z*e2m  # Position Lager Dorn
    rw = rd + lw*e2m  # Position DornEnde (Werkstückhalter)
    rb = rw + lb*e2m

    # Position Massenschwerpunkte in Bezug auf globales Koordinatensystem
    comz = (lcomz0 + z)*e2  # Massenmittelpunkt z-Schlitten
    comy = rx + lcomy_y0*e1 + lcomy_z0*e2  # Massenmittelpunkt y-Schlitten
    comx = rx + lcomx_y*e1m + lcomx_z*e2m  # Position Massenschwerpunkt Rot-X System
    comd = rd + lcomw*e2m  # Position Massenschwerpunkt Dorn
    comb = rb

    # Koordinatentransformationen (Geschwindigkeiten)
    vcomz = sp.diff(comz, t)
    vcomd = sp.diff(comd, t)
    vcomx = sp.diff(comx, t)
    vcomy = sp.diff(comy, t)
    vcomb = sp.diff(comb, t)

    # Energien
    Tt = Mz/2*sp.DotProduct(vcomz, vcomz) + \
         My/2*sp.DotProduct(vcomy, vcomy) + \
         Mx/2*sp.DotProduct(vcomx, vcomx) + \
         Md/2*sp.DotProduct(vcomd, vcomd) + \
         Mb/2*sp.DotProduct(vcomb, vcomb)
    Tr = Jx/2*sp.diff(phi, t)**2 + \
         Jd/2*sp.diff(phi, t)**2 + \
         Jb/2*sp.diff(phi, t)**2  # Kinetische Energie für mb um alle 3 rot. Freiheitsgrade
    T = Tt + Tr
    U = grav*(Mz*comz + My*comy + Mx*comx + Md*comd + Mb*comb)
    L = T - U[1, 0]

    # Generalisierte Koordinaten und Ableitung
    q = sp.Matrix([y, z, phi])
    dq = q.diff(t)  # 1. Ableitung der verallgemeinerte Koordinaten

    print("3.1. Compute Mass-Matrix")
    # Matrizenberechnung
    M = sp.hessian(L, dq)  # Massenmatrix

    print("3.2. Compute RHS")
    tmp = sp.Matrix([L]).jacobian(dq)
    C = tmp.transpose().jacobian(q)

    F = sp.Matrix([L]).jacobian(q).transpose()

    if st.rotX_Spring:
        # Statische, winkelabhängiges Kompensationsmoment durch RotX-Feder berechnen
        _, Mp_spring = get_compensationTorque_from_phi(phi)
        Q = sp.Matrix([[Fy], [Fz], [Mp+Mp_spring]])  # Verallgemeinerte Kräfte/Momente
    else:
        Q = sp.Matrix([[Fy], [Fz], [Mp]])

    # Dissipationsfunktionen
    Dy = 0  # 1/2*param.dy*sp.diff(y, t)**2        # Reibung zwischen y- und z-Schlitten
    D = Dy
    DMat = sp.Matrix([D]).jacobian(dq).transpose()

    # Gleichung für rechte Seite aufstellen: M*ddq = -C*dq + F + Q - D
    dqt = q.diff(t)
    rhs = -C*dqt + F + Q - DMat

    # Systemgrößen als Ersatz der generalisierten Koordinaten und Kräfte definieren
    u0, u1, u2 = sp.symbols('u0 u1 u2')
    z1, z2, z3, z4, z5, z6 = sp.symbols('z1 z2 z3 z4 z5 z6')
    dz1, dz2, dz3, dz4, dz5, dz6 = sp.symbols('dz1 dz2 dz3 dz4 dz5 dz6')
    rep_sys = [(sp.diff(y, t, t), dz2), (sp.diff(y, t), z2),
               (sp.diff(z, t, t), dz4), (sp.diff(z, t), z4),
               (sp.diff(phi, t, t), dz6), (sp.diff(phi, t), z6),
               (y, z1), (z, z3), (phi, z5),
               (Fy, u0), (Fz, u1), (Mp, u2)]

    # Parameterwerte aus param.py beziehen
    rep_param = [(My, param.My), (Mx, param.Mx), (Md, param.Md), (Mb, param.Mb), (Mz, param.Mz),
                 (rx_z0, param.rx_z0), (rx_y0, param.rx_y0), (lx_z, param.lx_z), (lx_y, param.lx_y), (lw, param.lw),
                 (lb, param.lb),
                 (lcomx_z, param.lcomx_z), (lcomx_y, param.lcomx_y), (lcomw, param.lcomw), (lcomz0, param.lcomz0),
                 (lcomy_z0, param.lcomy_z0), (lcomy_y0, param.lcomy_y0),
                 (Jd, param.Jd), (Jx, param.Jx), (Jz, param.Jz), (Jy, param.Jy), (Jb, param.Jb),
                 (kxt, param.kxt)]

    # Linearisierung bezüglich der elastischen Koordinaten
    print("3.3. Substitute Parameters")
    rhs = (sp.simplify(rhs).subs(rep_param))
    M = (sp.simplify(M).subs(rep_param))

    # Parameter und neue Systemgrößen einsetzen
    rhs = rhs.subs(rep_sys)
    M = sp.simplify(M.subs(rep_sys))
    rhs = sp.simplify(sp.expand(rhs.subs(rep_sys)))

    with open(st.path_pickle + 'M_matrix_rigid_' + st.suffix_eqs + '.pickle', 'wb') as outf:
        outf.write(pickle.dumps(M))
        outf.close()
    with open(st.path_pickle + 'rhs_vector_rigid_' + st.suffix_eqs + '.pickle', 'wb') as outf:
        outf.write(pickle.dumps(sp.simplify(rhs)))
        outf.close()


if __name__ == "__main__":
    if calc_elastic:
        gen_eqs_of_motion()
    if calc_rigid:
        gen_eqs_of_motion_rigid()
