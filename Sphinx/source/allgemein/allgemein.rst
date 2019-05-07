Dokumentieren mit Sphinx
==========================

Re-structered-text basics
---------------------------
Original Sphinx docu: `sphinx-doc.org <http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_

Beispiele: `Admonitions,Headings,Code,Footnotes,Citation,Comments,Tables,Body-Elements <https://pythonhosted.org/sphinxjp.themes.basicstrap/sample.html>`_

Überschriften
^^^^^^^^^^^^^^
* # with overline, for parts

* * with overline, for chapters

* =, for sections

* -, for subsections

* ^, for subsubsections

* ", for paragraphs

Admonition bzw. Farbboxen
^^^^^^^^^^^^^^^^^^^^^^^^^^
http://docutils.sourceforge.net/docs/ref/rst/directives.html#admonitions

**Beispiel:**

.. code::

	.. admonition:: And, by the way...

		You can make up your own admonition too.
		
.. admonition:: And, by the way...

		You can make up your own admonition too.

Python-Code mit Sphinx
-----------------------

Neue Projektdokumentation erstellen
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Requirement-Liste erstellen
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Modul `pireqs <https://pypi.org/project/pipreqs/>`_ zur automatischen Erzeugung einer Liste von benötigten Modulen
für ein bestimmtes Projekt erstellen.

.. code::

	$ pip install pipreqs

Durch den Befehl
	
.. code::

	$ pipreqs /path/to/project/folder
	
	
wird das File **requirement.txt** im Projektordner erzeugt und sieht beispielsweise so aus

.. code-block:: python

	requests==2.9.1
	pytest==3.0.5
	selenium==3.4.3
	mechanize==0.2.5
	python_dotenv==0.1.0
	Appium_Python_Client==0.24
	dotenv==0.0.5


