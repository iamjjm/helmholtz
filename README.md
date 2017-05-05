# A multi-dimensional Helmholtz solver using FEniCS 

This repository contains code to simulate the Helmholtz equation in 1,2, and 3 dimensions (see *HelmholtzSolver.py*), with an ability to incorporate Dirichlet, Neumann, and absorbing boundary conditions (via Perfectly Matched Layers - PMLs, implementation of which can be found in *PML.py*).  There are options to define non-uniform anisotropic meshes by a simple coordinate rescaling, which can be useful sometimes, as shown through *ex2d_interactingAtoms.py*. The file *GalerkinSolverComplex.py* can be adapted to define other weak formulations in FEniCS over the complex field, using mixed function spaces.  The PML implementation can be understood by studying the work detailed here <http://dx.doi.org/10.1002/nme.1105>. The files starting with "ex" are examples which can be run readily.

The code works on FEniCS version 2016.2.0 , the latest stable release as of May 2017.
