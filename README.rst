==================================================================================
scoreCART: Python package for building trees for probabilistic prediction
==================================================================================


Dependencies
~~~~~~~~~~~~

scoreCART is a Python package for the method introduced in the paper titled 'Building 
Trees for Probabilistic Prediction via Scoring Rules.'

This code requires Python (version 3.6 or later) and pip. 

Examples under ``examples/`` directory replicate figure 1 and figure 3.

Set up 
~~~~~~

We recommend creating a Python virtual environment within the working directory of scoreCART. 
If a virtual environment is created, scoreCART's required packages are installed and 
isolated from those installed a priori. Creating a virtual environment will also prevent
having conflicting packages on a user's machine. You may need to install the virtual 
environment on your system (if a user's system does not have it), for example, 
with 'apt install python3.9-venv'

1)Extract the zipped file.

2)From the command line, go to the directory of the source code.

3)Use the following command to create a virtual environment::

  python3 -m venv venv/  
  source venv/bin/activate  
 
We note that creating a virtual environment is not a required step. However, we tested this
procedure to prevent any conflict, and the code runs smoothly.

Installation
~~~~~~~~~~~~

To install the package:

1)Go to the directory of the source code (if a user has not done so yet).

2)Use the following command to install the required packages::

 pip install -r requirements.txt

3)From the command line, use the following command to install the package::

 pip install -e .

Once installed, a user should see ``build/`` directory created.
 

Instructions for running the illustrative examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To replicate Figure~1:

1)Go to the ``examples/`` directory.

2)Execute the followings from the command line::

 python3 Figure1.py

Running this script should not take more than 60 sec. See the figures (png files) saved under ``examples/`` directory.

Instructions for running the prominent empirical results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instructions are provided to replicate Figure~3.

Execute the following from the command line::

  python3 Figure3a.py
  python3 Figure3b.py
 
Running this script takes about 2.5hrs on a personal Mac laptop. 
Once completed, ``Figure3_easy.png`` is saved under ``examples/`` directory.
  
Final comments
~~~~~~~~~~~~~~

Type ``deactivate`` from the command line to deactivate the virtual environment if created.

Type ``pip uninstall scoreCART`` from the command line to uninstall the package.
