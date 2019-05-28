settings.py
===============

Im :py:mod:`settings` File werden alle Einstellungen, betreffend einer Simulation gesetzt. Diese Einstellungen
werden dann während der Laufzeit der Klassen :py:mod:`eq\_of\_motion`, :py:mod:`simulation` und :py:mod:`plot\_results`.

.. automodule:: settings
    :members:
    :undoc-members:
    :show-inheritance:

.. code:: python

	# *** Equation of Motion Settings **************************************
	suffix_eqs = 'test'     # name suffix für pickle files der eq_of_motion, sim-result und plots
	path_pickle = './pickleFiles/'  # pfad zu den pickle files
	rotX_Spring = True

	# *** Simulation settings # ********************************************
	t0 = 0.0        # start time of movement
	t_mve = 0.1390  # duration of movement in s
	t_set = 0.1     # duration to watch settling behavior after movement has ended
	dt = 1e-4       # time step
	suffix_sim = 'test_InputShaping'  # suffix under which sim results will be saved
	feedback = True # True for Closed Loop, False for Open Loop

	# *** Plot settings # **************************************************
	lstSim_toPlot = ['test_noInputShaping', 'test_InputShaping', ''] # list of simulation results (
													# saved as pickle files) to plot

	# *** Model settings ***************************************************
	rigid = [True, False, True, True] # Federn starr schalten [theta, alpha, g, rho]
	elModel = False     # True to include el. Model, False to ignore el Model

	# *** Trajectory settings **********************************************
	trajFromData = False    # True to load Trajectory data from file
	# if trajFromData==True set following (also: correct end Time has to be set)
	path_mat = './JoanneumTrajectories/'
	fname_mat = 'cav1-9_lambda0.mat'
	fpath_mat = path_mat + fname_mat
	# if trajFromData==False set follwing
	inputShaping = True    # True to use input shaping
	place_position_y = 0.04
	place_position_z = 0.826
	phi0 = 0.0
	phiT = 80.0
	# *** end settings ****************************************************


	# *** Depreciated settings ********************************************
	savefigs = False    # speichern der erzeugten plots direkt in den latex ordner
	save_EEmve_to_matFile = False
	externalForce = False  # externe Kraft wirken lassen
	t_force = 0.3*t_mve     # if externalForce==True: Zeit zu der die Bestückkraft wirkt
	plotAnalysis = False    # analyse der mechanische Eigenfrequenz bei verschiedenen z-Stellungen
	# *** end depreciated settings ****************************************

