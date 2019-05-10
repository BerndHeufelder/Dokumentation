Kinematik
***********


Trajektorien
=============

.. admonition:: :download:`Continuous Acceleration and Duty Time <files/Continuous Acceleration and Duty Time.pdf>`

	Berechnung der *Continuous Acceleration* anhand des *Duty Factors*.


Bremsdistanz berechnen
======================

.. note:: Berechnung der Bremsdistanz in Abhängigkeit von :math:`j(t), a(t), v(t), s(t)`

Die Berechnung der Bremsdistanz mit einem Jerk ungleich Null, ergibt ein trapezförmigen Verlauf der Beschleunigung. 
Die Beschleunigungsphasen können in drei Teilbereiche aufgeteilt werden. Die Anfangsbedingungen der einzelnen Teilbereiche
sind dabei von den Endwerten des vorangegangen Bereichs abhängig. Als Startbedingung wird eine konstante Geschwindigkeit,
und keine Beschleunigung angesetzt.


.. toctree:: 
   files/Breaking_distance.ipynb
   
Robotik
=======

.. admonition:: :download:`Statische EE-Kraft in Gelnekskräfte umrechnen <files/IntroductionToRobotics_Pearson_UMRECHNUNG_Endeffektorkraft_auf_Gelenksmomente.pdf>`

	.. image:: files/IntroductionToRobotics_Pearson_UMRECHNUNG_Endeffektorkraft_auf_Gelenksmomente-157.svg
		:width: 32%
	.. image:: files/IntroductionToRobotics_Pearson_UMRECHNUNG_Endeffektorkraft_auf_Gelenksmomente-158.svg
		:width: 32%
		
	Die Komponenten des Vektors :math:`f` der die Position des Endeffektors beschreibt, ergibt sich aus translatorischen und rotatorischen
	Koordinatentransformationen vom Ursprungskoordinatensystem in das Endeffektorkoordinatensystem. Die Transformationsmatrix zur Umrechnung
	kann anhand der Denavit-Hartenberger Transformation vorgenommen werden.
	
	.. image:: files/static_force_ee.jpg
		:width: 100%
	
sad