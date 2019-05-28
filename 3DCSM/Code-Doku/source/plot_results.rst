plot\_results.py
====================

Die :py:mod:`plot\_results` Klasse ermöglicht es die Ergebnisse einer durchgeführten Simulation zu visualisieren.
Es ist auch möglich die Ergebnisse mehrerer Simulationen gleichzeitig zu plotten, um diese Vergleichen zu können.

.. admonition:: Schnittstellen zu anderen Skripten bzw. Klassen

	* Input: 
		* Simulation_result_<name1>.pickle
		* *Optional:* Simulation_result_<name2>.pickle
		* *Optional:* Simulation_result_<name3>.pickle
		* *Optional:* ...
	* Output: 
		* Matplot-Figures
		* *Optional:* Figures als *.jpg, .svg*
		
	**Bemerkung:** Die Namen *<name1>, <name2>, <name3>* sind in :py:mod:`settings` durch die Variable *lstSim_toPlot* definiert.

.. automodule:: plot_results
