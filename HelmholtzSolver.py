"""
This is a class for defining variational problems where the PDE is the Helmholtz equation
-u,xx -k^2 u = f

"""
#=======================================================================================

import os, sys, time
import numpy as np
import dolfin as df

from GalerkinSolverComplex import *
from PML import *

#=======================================================================================

class HelmholtzSolver(GalerkinSolverComplex):
    #-----------------------------------------------------------------------------------
    def __init__(self,\
	         tDim,meshOpt,bcOpt,sourceOpt,
		 materialOpt,pmlOpt,\
		 **kwargs):
    #-----------------------------------------------------------------------------------
        super(HelmholtzSolver,self).__init__(tDim,meshOpt,bcOpt,\
					     **kwargs)

	self.sourceOpt   = sourceOpt
        self.materialOpt = materialOpt
        self.pmlOpt      = pmlOpt
        self.pmlOpt['minLimit'],self.pmlOpt['maxLimit'] = findPmlLimits(tDim,\
                                                          self.mesh,self.bcOpt,\
                                                          self.pmlOpt)
	self.constructPmlSubDomain()

	self.MS = {1:MaterialScalar1D,\
	           2:MaterialScalar2D,\
	           3:MaterialScalar3D}
	self.MT = {1:MaterialTensor1D,\
	           2:MaterialTensor2D,\
	           3:MaterialTensor3D}

	self.prescribeSource()
	self.constructGalerkinWeakFormulation()
	self.constructLinearFunctional()
        self.assemble()
        self.accountPointSource()
        self.solveLinearSystem()

    #-----------------------------------------------------------------------------------
    def constructPmlSubDomain(self):
    #-----------------------------------------------------------------------------------
        # Initialize mesh function for subdomains
        self.domains = df.CellFunction("size_t", self.mesh)
        self.domains.set_all(0)

	# create PML subdomain and mark elements therein as "1"
        PmlSD = PMLSubDomain(self.tDim,self.pmlOpt)
        PmlSD.mark(self.domains,1)

        # Define new measures associated with the PMl domains
        self.dx = df.Measure("dx", domain=self.mesh, subdomain_data=self.domains)

        return

    #-----------------------------------------------------------------------------------
    def constructGalerkinWeakFormulation(self):
    #-----------------------------------------------------------------------------------
	tDim   = self.tDim
        wr,wi  = self.wr,self.wi
        ur,ui  = self.ur,self.ui
	MS,MT  = self.MS,self.MT
	mOpt   = self.materialOpt
	pmlOpt = self.pmlOpt
	s      = sigmaFun

	MTR    = MT[tDim](mOpt,s,pmlOpt,returnReal = True,  degree=self.meshOpt['polynomialOrder'])
	MTI    = MT[tDim](mOpt,s,pmlOpt,returnReal = False, degree=self.meshOpt['polynomialOrder'])
	MSR    = MS[tDim](mOpt,s,pmlOpt,returnReal = True,  degree=self.meshOpt['polynomialOrder'])
	MSI    = MS[tDim](mOpt,s,pmlOpt,returnReal = False, degree=self.meshOpt['polynomialOrder'])

        bFormR = df.inner(df.grad(wr), MTR*df.grad(ur) - MTI*df.grad(ui))*df.dx \
	        -df.inner(df.grad(wi), MTI*df.grad(ur) + MTR*df.grad(ui))*df.dx \
                - ( df.inner(wr, MSR*ur - MSI*ui)*df.dx \
	           -df.inner(wi, MSI*ur + MSR*ui)*df.dx )

        bFormI = df.inner(df.grad(wr), MTI*df.grad(ur) + MTR*df.grad(ui))*df.dx \
	        +df.inner(df.grad(wi), MTR*df.grad(ur) - MTI*df.grad(ui))*df.dx \
                - ( df.inner(wr, MSI*ur + MSR*ui)*df.dx \
	           +df.inner(wi, MSR*ur - MSI*ui)*df.dx )

	self.bForm = bFormR + bFormI

	return

    #-----------------------------------------------------------------------------------
    def constructLinearFunctional(self):
    #-----------------------------------------------------------------------------------
        wr,wi = self.wr,self.wi
	fr,fi = self.source['real'],self.source['imag']

	self.lFuncR = df.inner(wr,fr)*df.dx - df.inner(wi,fi)*df.dx
	self.lFuncI = df.inner(wr,fi)*df.dx + df.inner(wi,fr)*df.dx

	self.lFunc  = self.lFuncR + self.lFuncI

	return

    #-----------------------------------------------------------------------------------
    def prescribeSource(self):
    #-----------------------------------------------------------------------------------
        self.source = dict.fromkeys(['real','imag'])

        for component in ('real','imag'):
            if self.sourceOpt[component]['choice']=='custom':
               self.source[component] = self.sourceOpt[component]['expression']
            if self.sourceOpt[component]['choice']=='none':
               self.source[component] = df.Constant(0.0)
	    if self.sourceOpt[component]['choice']=='pointSource':
               self.source[component] = df.Constant(0.0)

        return

    #-----------------------------------------------------------------------------------
    def assemble(self):
    #-----------------------------------------------------------------------------------
        self.A, self.rhsVec = df.assemble_system(self.bForm, self.lFunc, self.bc)

	return

    #-----------------------------------------------------------------------------------
    def accountPointSource(self):
    #-----------------------------------------------------------------------------------
	componentIndex = {'real':0,'imag':1}
        for sourceComponent in ('real','imag'):
	    if self.sourceOpt[sourceComponent]['choice']=='pointSource':
	       pointSourceMag = self.sourceOpt[sourceComponent]['pointSourceMag']
               pointSourceLoc = self.sourceOpt[sourceComponent]['pointSourceLoc']
               for funSpaceComponent in ('real','imag'):
	           if sourceComponent=='imag' and funSpaceComponent=='imag':
		      pointSourceMag = -pointSourceMag
                   PS = df.PointSource(self.V.sub(componentIndex[funSpaceComponent]),\
	                               df.Point(pointSourceLoc),pointSourceMag)
	           PS.apply(self.rhsVec)

	return

    #-----------------------------------------------------------------------------------
    def solveLinearSystem(self):
    #-----------------------------------------------------------------------------------
        self.uSoln = df.Function(self.V)
        df.solve(self.A, self.uSoln.vector(), self.rhsVec)
	self.uSolnReal, self.uSolnImag = self.uSoln.split()
	return

#=======================================================================================

class WaveNumber(object):
      def __init__(self,omega,waveSpeed,damping):
          self.omega     = omega
          self.damping   = damping
          self.waveSpeed = waveSpeed 
      def evalWaveNumber(self,x):
          return self.omega/self.waveSpeed(x) + 1j*self.damping(x)

#=======================================================================================
