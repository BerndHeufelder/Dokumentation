Anwendungsbeispiele
====================

Beispiel 1: Simulation des starren Systems
############################################

1) Als erstes müssen die gewünschten Randbedingungen einer Simulation im :py:mod:`settings` File definiert werden.

.. admonition:: Simulationseinstellungen in :py:mod:`settings`

	* Durch Setzen der Variable 
	
		.. code:: python
		
			rigid = [True, True, True, True] # Federn starr schalten [theta, alpha, g, rho]
		
	  wird das System starr, ohne Federn, simuliert.
	  
	* Zeiteinstellungen (Startzeit: t0, Endzeit: t_mve+t_set, Zeitschritt: dt)
	
		.. code:: python
		
			t0 = 0.0        # start time of movement
			t_mve = 0.1390  # duration of movement in s
			t_set = 0.1     # duration to watch settling behavior after movement has ended
			dt = 1e-4       # time step
	
	* Das System soll ohne Berücksichtigung der Motormodelle
	
		.. code:: python
		
			elModel = False     # True to include el. Model, False to ignore el Model
			
	 nicht nur gesteuert, sondern geregelt
	 
		.. code:: python
		
			feedback = True # True for Closed Loop, False for Open Loop
			
	 werden.
	
	* Der Endeffektor soll einer Trajektorie folgen, welche in einem *.mat* File als Zeitfolge gespeichert ist.
	
		.. code:: python
		
			# *** Trajectory settings **********************************************
			trajFromData = True    # True to load Trajectory data from file
			# if trajFromData==True set following (also: correct end Time has to be set)
			path_mat = './JoanneumTrajectories/'
			fname_mat = 'cav1-9_lambda0.mat'
			
	* Benennungen der *.pickle* Files, die als Input und Output der 3 Hauptklassen verwendet werden
	
		.. code:: python 
		
			path_pickle = './pickleFiles/'  # pfad zu den pickle files
			suffix_eqs = 'Bsp1_rigid'     	# name suffix für pickle files of eq_of_motion
			suffix_sim = 'Bsp1_rigid'   	# suffix under which sim results will be saved
			lstSim_toPlot = ['Bsp1_rigid']  # list of simulation results (saved as pickle files) to plot


2) Durch Ausführen von :py:mod:`eq\_of\_motion` werden die *.pickle* Files

	* M_matrix_Bsp1_rigid.pickle
	* M_matrix_rigid_Bsp1_rigid.pickle
	* rhs_vector_Bsp1_rigid.pickle
	* rhs_vector_rigid_Bsp1_rigid.pickle

	erstellt. 
	
	**Console-Output:**
	
		.. code:: python
		
			1. Define Variables
			2. Calculate elastic equations of motion
			2.1. Compute Mass-Matrix
			2.2. Compute RHS
			2.3. Approximate for small elastic Deflections
			2.4. Substitute Parameters
			3. Calculate rigid equations of motion
			3.1. Compute Mass-Matrix
			3.2. Compute RHS
			3.3. Substitute Parameters

			Process finished with exit code 0
	
3) Anschließend kann die eigentliche Simulation durch ausführen von :py:mod:`simulation` gestartet werden.
   Es werden die *.pickle* Files aus vorherigem Schritt geladen und die erzeugten Simulationsergebnisse in
    
   * Simulation_result_Bsp1_rigid.pickle
   
   gespeichert. 
   
   **Console-Output:**
	
		.. code:: python
		
			start integration...
			0.0149
			0.0298
			0.0447
			0.0596
			0.0745
			0.0894
			0.1043
			0.1192
			0.1341
			integration finished...
			time: 49.72746253013611

			Process finished with exit code 0
   
4) Um diese Ergebnisse zu visualisieren wird die Klasse :py:mod:`plot\_results` ausgeführt. Da in *lstSim_toPlot* nur die gerade
   durchgeführte Simulation gelistet ist, wird keine Referenzsimulaiton geplottet.
   
   **Console-Output:**
	
		.. code:: python
		
			Initialising ...
				Following Simulations will be plotted:
				1   Bsp1_rigid
			Creating figures ...
			Plotting Simulation(s) ...
				1   Bsp1_rigid
				... loading Data from Files ...
				... start plotting ...

.. list-table:: **Plots zur Datenvisualisierung**
   :align: center

   * - Endeffektor-Pos der yz-Ebene mit Tracking-Error
     - Positionsverläufe der Achsen mit Tracking-Error 	  
   * - .. image:: img/Figure_1.png
			:target: ../../source/img/Figure_1.png
     - .. image:: img/Figure_2.png 
			:target: ../../source/img/Figure_2.png
   * - Längenänderung der Federn im System 
     - Momentenverläufe der einzelnen Achsen
   * - .. image:: img/Figure_3.png  
			:target: ../../source/img/Figure_3.png
     - .. image:: img/Figure_4.png
			:target: ../../source/img/Figure_4.png
   * - Pos, Spd, Acc der einzelnen Achsen
     - Pos, Spd, Acc des Endeffektors
   * - .. image:: img/Figure_5.png 
			:target: ../../source/img/Figure_5.png
     - .. image:: img/Figure_6.png 		
			:target: ../../source/img/Figure_6.png

Beispiel 2: Simulation des Systems mit Unterbaufeder
######################################################