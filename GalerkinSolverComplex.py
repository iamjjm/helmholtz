"""
- This is a class that helps create a generic Galerkin solver for complex weak forms
- Please be aware that FEniCS by default uses Neumann boundary conditions

"""
#=======================================================================================

import os, sys, time
import numpy as np
import dolfin as df

#=======================================================================================

class GalerkinSolverComplex(object):
    """ The parent PDE solver class for complex variational forms"""
    #-----------------------------------------------------------------------------------
    def __init__(self, tDim, meshOpt, bcOpt,\
                 overloadBuildMesh=False,\
		 overloadBuildFunctionSpace=False):
    #-----------------------------------------------------------------------------------

	super(GalerkinSolverComplex,self).__init__()

	self.tDim        = tDim
	self.meshOpt     = meshOpt
	self.bcOpt       = bcOpt

	self._buildMeshCompleted                = False
	self._buildFunctionSpaceCompleted       = False
	self._prescribeDBCCompleted             = False
	self._boundaryCheckFunctionsConstructed = False

	if not(overloadBuildMesh):
	   self.buildMesh()
	   if meshOpt['stretchMesh']==True:
	      self.stretchMesh()
	if not(overloadBuildFunctionSpace) and self._buildMeshCompleted:
           self.buildFunctionSpace()

	self.constructBoundaryCheckFunctions()
	if self._buildFunctionSpaceCompleted and \
	   self._boundaryCheckFunctionsConstructed:
           self.prescribeDBC()

    #-----------------------------------------------------------------------------------
    def buildMesh(self):
    #-----------------------------------------------------------------------------------
        """ Discretize the domain """

        if self.tDim == 1:
           self.mesh = df.UnitIntervalMesh(self.meshOpt['nXElem'])
        if self.tDim == 2:
           self.mesh = df.UnitSquareMesh(self.meshOpt['nXElem'],\
                                         self.meshOpt['nYElem'])
        if self.tDim == 3:
	   self.mesh = df.UnitCubeMesh(self.meshOpt['nXElem'],\
                                       self.meshOpt['nYElem'],\
                                       self.meshOpt['nXElem'])

	self._buildMeshCompleted = True

	return

    #-----------------------------------------------------------------------------------
    def stretchMesh(self):
    #-----------------------------------------------------------------------------------
        """ Stretch the mesh """

        for iDim in range(self.tDim):
            s  = self.meshOpt['stretchParam'][iDim]
            x  = self.mesh.coordinates()[:,iDim]
	    xs = x**s
            self.mesh.coordinates()[:,iDim] = xs

	return

    #-----------------------------------------------------------------------------------
    def buildFunctionSpace(self):
    #-----------------------------------------------------------------------------------
        """ Built the product function space """

        ## Deprecated code from earlier version of Fenics
        #self.VReal = df.FunctionSpace(self.mesh,'CG',self.meshOpt['polynomialOrder'])
        #elf.VImag = df.FunctionSpace(self.mesh,'CG',self.meshOpt['polynomialOrder'])
        #self.V = df.MixedFunctionSpace([self.VReal,self.VImag])

        elem  = df.FiniteElement('CG',self.mesh.ufl_cell(),self.meshOpt['polynomialOrder'])
        self.VReal = df.FunctionSpace(self.mesh,elem)
        self.VImag = df.FunctionSpace(self.mesh,elem)
        self.V     = df.FunctionSpace(self.mesh,elem*elem)
        self.ur, self.ui = df.TrialFunctions(self.V)
        self.wr, self.wi = df.TestFunctions(self.V)

	self._buildFunctionSpaceCompleted = True

	return

    #-----------------------------------------------------------------------------------
    def constructBoundaryCheckFunctions(self):
    #-----------------------------------------------------------------------------------
        """ Construct functions that allow to check if point is on the boundary """

	def leftBoundary(x):
	    return x[0] < df.DOLFIN_EPS
	def rightBoundary(x):
	    return x[0] > 1.0 - df.DOLFIN_EPS
	def bottomBoundary(x):
	    return x[1] < df.DOLFIN_EPS
	def topBoundary(x):
	    return x[1] > 1.0 - df.DOLFIN_EPS
	def backBoundary(x):
	    return x[2] < df.DOLFIN_EPS
	def frontBoundary(x):
	    return x[2] > 1.0 - df.DOLFIN_EPS

        bFaceList = []

        ## 1 dimensional case
	self.bcOpt['left']['boundaryCheckFun']  = leftBoundary
	self.bcOpt['right']['boundaryCheckFun'] = rightBoundary
        bFaceList.extend(('left','right'))

        ## 2 dimensional case
        if self.tDim>1:
	   self.bcOpt['bottom']['boundaryCheckFun'] = bottomBoundary
	   self.bcOpt['top']['boundaryCheckFun']    = topBoundary
           bFaceList.extend(('bottom','top'))

        ## 3 dimensional case
        if self.tDim>2:
	   self.bcOpt['back']['boundaryCheckFun']   = backBoundary
	   self.bcOpt['front']['boundaryCheckFun']  = frontBoundary
           bFaceList.extend(('back','front'))

        self.bcOpt['bFaceList'] = bFaceList

	self._boundaryCheckFunctionsConstructed = True

	return

    #-----------------------------------------------------------------------------------
    def prescribeDBC(self):
    #-----------------------------------------------------------------------------------
        """
        Define Dirichlet boundary conditions for the product space
        """

	componentIndex = {'real':0,'imag':1}
        self.bc = []
        for bFace in self.bcOpt['bFaceList']:
	    if self.bcOpt[bFace]['DBC']==True:
	       for component in ('real','imag'):
                   self.bc.append(df.DirichletBC(self.V.sub(componentIndex[component]),\
                                                 self.bcOpt[bFace][component],\
                                                 self.bcOpt[bFace]['boundaryCheckFun']))

	self._prescribeDBCCompleted = True

	return

#=======================================================================================

class UserScalarExpression(df.Expression):
    #-----------------------------------------------------------------------------------
    def specifyUserFunction(self, userDefFun, *userDefFunArgs):
    #-----------------------------------------------------------------------------------
        self.userDefFun     = userDefFun
	self.userDefFunArgs = userDefFunArgs
    #-----------------------------------------------------------------------------------
    def eval(self, value, x):
    #-----------------------------------------------------------------------------------
        args = (x,) + self.userDefFunArgs
	value[0] = apply(self.userDefFun,args)

#=======================================================================================
