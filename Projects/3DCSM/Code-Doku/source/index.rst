3DCSM-Bodyhandling Code-Dokumentation
##########################################################

.. admonition:: Dokumente zur dynamischen Modellbildung des Body-Handling-Systems (AP2) 
		
	:download:`Bericht: Dynamische Modellierung des Body-Handlings <../../Zwischenbericht/report/3DCSM_documentation.pdf>`
	
	:download:`Präsentation: Dynamische Modellierung des Body-Handlings  <../../Zwischenbericht/presentation/Zwischenpräsentation_Modellbildung.pdf>`

Einführung
**********

Zweck dieser Dokumentation ist eine Übersicht der Simulaionsumgebung bezüglich dessen Dateistruktur und Anwendung. 
Die einzelnen Klassen und ihre zugehörigen Member-Funktionen werden hier grob in ihrer Funktionalität beschrieben. 
Diese Dokumentation ist zum Einen an Personen gerichtet, die sich weiterführend mit dem Simulationscode beschäftigen 
und diesen Modifizieren möchten. Zum Anderen wird auch für den Nutzer, der die Simulaionsumgebung lediglich benutzen 
will, eine Anleitung, sowie Anwendungsbeispiele vorgegeben. 

Requirements
**************
Die Simulaionsumgebung wurde in Python 3.6.3 erstellt. Zusätzlich werden die folgenden Module
benötigt:

* numpy==1.14.3
* matplotlib==2.2.2
* sympy==1.3
* scipy==1.1.0


Wie ist die Simulationsumgebung zu verwenden?
**********************************************

Der komplette Ablauf einer Simulation, von der Erzeugung der Bewegungsgleichungen bis zur Darstellung 
der Ergebnisse, kann prinzipiell in drei Teilschritte aufgeteil werden:

1. **Erzeugung des Differentialgleichungssystems** durch Ausführung des Skriptes :py:mod:`eq\_of\_motion`
2. **Simulation** des erstellten Systems durch die Klasse :py:mod:`simulation`
3. **Visualisierung** der Ergebnisse durch die Klasse :py:mod:`plot\_results`

Die Randbedingungen, Einstellungen und Systemparameter werden durch die Skripte :py:mod:`param` und :py:mod:`settings`
definiert.

Offene Punkte
**************

* Parameterwahl für Stromregelkreise 
* Input-shaping zeigt nach Bewegungsende keine Verbesserung in der Schwingungsamplitude
	**Erkentnisse:** Durch input shaping wird die Trajektorie so verändert, dass die durch die urpsprüngliche Bewegung eingebrachten Schwingungen kompensiert werden. 
	Der dafür nötige Trajektorie ist jedoch nicht mehr so einfach zu folgen (mehrere Abstufungen des Beschleunigungsverlaufs).
	Dadurch enstehen nach Bewegungsende zwar geringere Schwingungen, jedoch ist die Regelabweichung während und nach der Bewegung größer.
	Es hängt demnach vom realisierbaren Positionsregelkreis, ob ein Input-shaping eine Verbesserung bringt 

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   modules
   examples
   
   

	


