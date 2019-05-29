Mechanik
########

Grundlagen
**********

Federn und Dämpfer
==================

.. index::
	single: Federn (lin. & rot.)
	
.. admonition:: :download:`Statische Lin. & rot. Federn <files/federn_lin_rot.pdf>`

	**Inhalt:**

	* Federsteifigkeit
	* Eingenfrequenz
	* Hintereinanderschaltung von Federn


Kinematik
*********

Trajektorien
============

.. index::
	single: Duty-Time, Duty-Factor

.. admonition:: :download:`Continuous Acceleration and Duty Time <files/Continuous_Acceleration_and_Duty_Time.pdf>`

	Berechnung der *Continuous Acceleration* anhand des *Duty Factors*.
	
	.. image:: files/Continuous_Acceleration_and_Duty_Time-1.png
		:width: 32%
	.. image:: files/Continuous_Acceleration_and_Duty_Time-2.png
		:width: 32%
	.. image:: files/Continuous_Acceleration_and_Duty_Time-3.png
		:width: 32%

.. index::
	single: Bremsdistanz berechnen
	
	
.. sidebar:: Screenshots

	.. image:: files/Traj_break-1.jpg
		:width: 50%
	.. image:: files/Traj_break-2.jpg
		:width: 32%

.. admonition:: Bremsdistanz berechnen

	Berechnung der Bremsdistanz in Abhängigkeit von :math:`j(t), a(t), v(t), s(t)`



	Die Berechnung der Bremsdistanz mit einem Jerk ungleich Null, ergibt ein trapezförmigen Verlauf der Beschleunigung. 
	Die Beschleunigungsphasen können in drei Teilbereiche aufgeteilt werden. Die Anfangsbedingungen der einzelnen Teilbereiche
	sind dabei von den Endwerten des vorangegangen Bereichs abhängig. Als Startbedingung wird eine konstante Geschwindigkeit,
	und keine Beschleunigung angesetzt.


	.. toctree:: 
	   files/Breaking_distance.ipynb
   
Robotik
=======

.. index::
	single: Robotik, Statische Kräfte
	
.. sidebar:: Screenshots

	.. image:: files/IntroductionToRobotics-157.png
		:width: 49%
	.. image:: files/IntroductionToRobotics-158.png
		:width: 49%

.. admonition:: :download:`Statische EE-Kraft in Gelnekskräfte umrechnen <files/IntroductionToRobotics.pdf>`

	Die Komponenten des Vektors :math:`f` der die Position des Endeffektors beschreibt, ergibt sich aus translatorischen und rotatorischen
	Koordinatentransformationen vom Ursprungskoordinatensystem in das Endeffektorkoordinatensystem. Die Transformationsmatrix zur Umrechnung
	kann anhand der Denavit-Hartenberger Transformation vorgenommen werden.
	
	.. image:: files/static_force_ee.jpg
		:width: 50%
		:align: center
	
