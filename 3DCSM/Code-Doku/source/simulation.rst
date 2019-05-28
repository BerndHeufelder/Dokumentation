Simulation.py
=================

Die :py:mod:`simulation` Klasse dient zur simulation der mit :py:mod:`eq\_of\_motion` erzeugten Bewegungsgleichungen. Die Simulationsergebnisse
werden wiederum in *.pickle* Files zwischengespeichert und können schließlich durch die Klasse :py:mod:`plot\_results` visualisiert werden.

.. admonition:: Schnittstellen zu anderen Skripten bzw. Klassen

	* Input: 
		* M_matrix_<name>.pickle
		* M_matrix_rigid_<name>.pickle
		* rhs_vector_<name>.pickle
		* rhs_vector_rigid_<name>.pickle
	* Output: 
		* Simulation_result_<name>.pickle
		
	**Bemerkung:** Der *<name>* ist in :py:mod:`settings` durch die Variable *suffix_sim* definiert.

.. toctree::
		
	model_el
	model_full
	model_mech
	control
	trajectory
	analysis
		
.. automodule:: simulation

