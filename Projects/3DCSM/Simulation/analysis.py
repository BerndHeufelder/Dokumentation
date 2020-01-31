import numpy as np
import matplotlib.pyplot as plt
import settings as st
import param


def autolabel(rects, ax, xpos='center'):
    # Funktion für Barheight-Wert über dem Balken
    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.4, 'left': 0.6}  # x_txt = x + w*off

    for rect in rects:
        height = int(round(rect.get_height()))
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                '{}'.format(height), ha=ha[xpos], va='bottom')


def analyse_eigenfrequencies(oModel, place_position_y, place_position_z):
    # Eigenfrequenzen an Bestückposition für verschiedene Federsteifigkeiten
    phi = np.array([0.0, 45.0, 90.0, 135.0, 150.0])
    f = np.zeros((2, 5))

    temp_dy = param.dy

    for ii in range(len(phi)):
        # Welche Federn werden starr geschalten [theta,alpha,g,rho]
        oModel.rigid = [False, False, False, False]
        # Berechne Eigenfrequenzen für
        (y, z) = oModel.inverseKinematicsRigid(place_position_y, place_position_z, phi[ii]/180*np.pi)
        f_new = oModel.eigen_frequencies([y, z, phi[ii]/180*np.pi], True)
        f[0, ii] = np.min(f_new)
        print("Phi=", phi[ii], ", y=", y, ", z=", z, ", F=", f[0, ii])

    oModel.dy = 0.0

    for ii in range(len(phi)):
        oModel.rigid = [False, False, False, False]
        # Berechne Eigenfrequenzen für
        (y, z) = oModel.inverseKinematicsRigid(place_position_y, place_position_z, phi[ii]/180*np.pi)
        f_new = oModel.eigen_frequencies([y, z, phi[ii]/180*np.pi], True)
        f[1, ii] = np.min(f_new)
        print("Phi=", phi[ii], ", y=", y, ", z=", z, ", F=", f_new)

    oModel.dy = temp_dy

    fig, ax = plt.subplots()
    width = 0.4

    rects1 = ax.bar(np.arange(5) - width/2, f[0, :], width, color='Blue', label='dy=-75mm')
    rects2 = ax.bar(np.arange(5) + width/2, f[1, :], width, color='Orange', label='dy=0mm')
    ax.grid()
    ax.set_xlabel('Winkel RotX in °')
    ax.set_ylabel('Eigenfrequenz in Hz')
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(phi)
    ax.legend()

    bottom, top = ax.get_ylim()  # return the current ylim
    ax.set_ylim((bottom, top*1.1))  # set the ylim to bottom, top

    autolabel(rects1, ax, "left")
    autolabel(rects2, ax, "right")
    fig.tight_layout()

    if st.savefigs:
        fig.savefig('../documentation/Latex/presentation/pics/eigenfreq_' + st.suffix)




