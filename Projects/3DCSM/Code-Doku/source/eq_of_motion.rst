eq\_of\_motion.py
=====================

Das Skript :py:mod:`eq\_of\_motion` dient zur dynamischen Erzeugung der Bewegungsgleichungen anhand des Euler-Lagrange Formalismus.
Prinzipiell werden die erzeugten Gleichungen in Form von *.pickle*-Files abgespeichert und anschließend von der Klasse
:py:mod:`simulation` für die Simulaiton aufgerufen.

.. admonition:: Schnittstellen zu anderen Skripten bzw. Klassen

	* Input: 
		None (da dieses Skript das erste ist, das ausgeführt wird)
	* Output: 
		* M_matrix_<name>.pickle
		* M_matrix_rigid_<name>.pickle
		* rhs_vector_<name>.pickle
		* rhs_vector_rigid_<name>.pickle
		
	**Bemerkung:** Der *<name>* ist in :py:mod:`settings` durch die Variable *suffix_eqs* definiert.

.. automodule:: eq_of_motion
