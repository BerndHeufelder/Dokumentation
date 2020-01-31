param.py
============

Das :py:mod:`param` Skript beinhaltet alle physikalischen Parameter des Body-Handling-Systems. Diese Parameter werden während
der Laufzeit der Klasse :py:mod:`eq\_of\_motion` durch deren Werte ersetzt.

.. automodule:: param
    :members:
    :undoc-members:
    :show-inheritance:
	
.. code:: python

	# -*- coding: utf-8 -*-
	import numpy as np

	# System parameters
	r0_z = 4.9e-3
	rotx0_z = r0_z + 0.7265 # z-Position rotX bei 0-Stellung z-Achse
	rotx0_y = 0.119  # y-Position rotX bei 0-Stellung y-Achse
	zfeder_z = -0.11  # z-Position z-Feder, Neg.VZ: z-Feder unterhalb des Ursprungs

	rx_z0 = rotx0_z - zfeder_z
	rx_y0 = rotx0_y

	# Massen in kg
	Mz = 77.29  # Masse z-Schlitten (bewegender Teil ohne Stator)
	My = 7.39   # y-Schlitten bis zum Motor
	Mx = 4.87   # Masse RotX Achse
	Md = 0.87+0.25   # Masse RotBody Achse, 0.25kg für den Artikelraum, 0.87 für die Achse selbst
	Mb = 0.024  # Masse des Bodies

	# Massenträgheitsmomente in kg*m^2 um jeweilige Massenmittelpunkte
	Jz = 7.191     # z-Achse
	Jy = 0.063029  # Y-Schlitten bis zum Motor
	Jx = 0.012638  # Trägheitsmoment Rot-X-Einheit
	Jd = 0.013826  # Trägheitsmoment Dorn bis zum Werkstück (Rot-Body)
	Jb = 0.0000077 # Trägheitsmoment des Werkzeugs (Bodies) um x-Achse

	# Geometrische Abmessungen in m
	r0 =    0.0     # Globaler Ursprung des Gesamtsystems
	lx_z =  0.039   # Abstand RotBody Lager 1 von RotX - Z-Richtung bei phi=0
	lx_y =  0.079   # Abstand RotBody Lager 1 von RotX - Y-Richtung
	lw =    0.06    # Stab bis zum Werkstück
	lb =    0.0286  # Versatz vom Konus zum Bodyschwerpunkt
	dy =    0.075   # Versatz über dem Durchmesser des Bodies in y-Richtung des Bodykoordinatensystems

	# Schwerpunktsabstände in m
	lcomx_y =  0.04533  # Abstand MMP RotX vom Motordrehgelenk (horizontal)
	lcomx_z =  4.14e-3  # Abstand MMP RotX vom Motordrehgelenk (vertikal)
	lcomw =    0.05355  # Abstand MMP RotBody von der Drehfeder aus
	lcomz0 =   r0_z + 0.09032 - zfeder_z  # Absolutposition MMP  z-Achse für dz=0
	lcomy_z0 = -0.1     # MMP  y-Achse von rotX-Bohrung für dz=0
	lcomy_y0 = 0.037    # MMP  y-Achse von rotX-Bohrung aus

	# Federkonstanten
	kwr = 0.217190e6    # Nm/rad, Drehfeder für Stab bis zum Werkstück
	kwt = 5995000       # N/m, translatroische Feder des Werkzeughalters, Konusverbindung zu Werkzeug
	kbr = 34899         # Nm/rad Biegefeder des Bodyholders
	kzr = 2.9024e6      # Nm/rad
	kxt = 1895        # N/m trans. Feder zur Eigengewichtskompensation der RotX-Achse

	# Dämpfungskonstanten
	# d = 2*xi*sqrt(k*m) xi ist der Dämpfungsgrad und für 1 entspricht das dem aperiodischem Fall
	dzr = 2*1*np.sqrt(kzr*Jz)       # Dämpfung z-Achse (Rotation)
	dwd = 2*0.01*np.sqrt(kwr*Jd)    # rotatorische Dämpfung des Werkzeughalters (RotBody-Achse)
	dwt = 2*1*np.sqrt(kwt*Md)       # translatorische Dämpfung des Werkzeughalters
	dbr = 2*1*np.sqrt(kbr*Jb)       # rot. Dämpfung des Bodyholders, mit 1 -> aperioderscher Fall

	# Schlittenspezifische Parameter
	steig_y = 0.016  # Spindelsteigung in m
	steig_z = 0.025  # Spindelsteigung in m
	kbemf_y = 0.3056 # Drehmomentenkonstante y-Motor in V/(rad/s)
	R_y = 3.0/2      # Spulenwiderstand pro Phase y-Motor in Ohm
	L_y = 5.4e-3/2   # Spuleninduktivität pro Phase y-Motor in Henry
	kbemf_z = 1.1364 # Drehmomentenkonstante z-Motor in V/(rad/s)
	R_z = 1.25/2     # Spulenwiderstand pro Phase z-Motor in Ohm
	L_z = 10.0e-3/2  # Spuleninduktivität pro Phase z-Motor in Henry
	kbemf_phi =0.2664# Drehmomentenkonstante RotX in V/(rad/s)
	R_phi = 1.111    # Spulenwiderstand RotX pro Phase in Ohm
	L_phi = 3.4e-3   # Spuleninduktivität RotX pro Phase in Henry

	# Parameter für Kompensationsfeder RotX
	L0 = 150.0e-3   # Einbaulaenge in m
	l_P1 = 30.7e-3  # Abstand vom Drehpunkt auf der RotX-Achse in m
	l_P2 = 59.5e-3  # Abstand vom Drehpunkt am Gehäuse in m
	l_eff_0deg = 90.5e-3  # Einbaulaenge bei RotX=0° in m
	phi_offset = 15*np.pi/180
	# Abstand des Punktes P1 zu P2 mit dem Offset von 15°
	l_p1p2_15deg = np.sqrt((l_P1*np.cos(phi_offset))**2+(l_P2-l_P1*np.sin(phi_offset))**2)

