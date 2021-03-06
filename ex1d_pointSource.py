import os, sys, time
import numpy as np
import dolfin as df

from HelmholtzSolver import *

## =====================================================================================

## -------------------------------------------------------------------------------------
def waveSpeed(x):
    return 1.0
## -------------------------------------------------------------------------------------
def damping(x):
    return 0.0
## -------------------------------------------------------------------------------------
def density(x):
    return 1.0
## -------------------------------------------------------------------------------------

## Define options
## ==============

omega      = 2*np.pi*7

meshOpt    = {'nXElem':80,\
              'polynomialOrder':1,\
	      'stretchMesh':False,\
	     }

bcOpt      = {'left':{'DBC':True,\
                      'real':df.Constant(0.0),\
                      'imag':df.Constant(0.0),},\
              'right':{'DBC':True,\
                       'real':df.Constant(0.0),\
                       'imag':df.Constant(0.0),},\
	      }

sourceOpt   = {'real':{'choice':'pointSource',\
	               'pointSourceLoc':np.array([0.5]),\
		       'pointSourceMag':1.0},
               'imag':{'choice':'none'},\
	      }


WN = WaveNumber(omega,waveSpeed,damping)
materialOpt = {'waveNumber':WN.evalWaveNumber,\
               'density':density,\
	      }

pmlOpt      = {'left':True,\
               'right':True,\
	       'exponent':2,\
	       'sigmaMax':5000,\
	       'numPmlWaveLengths':1,\
	       'pmlWaveNumber':omega,\
	      }

## Instantiate Helmoltz solver class with the options
## ==================================================
HS = HelmholtzSolver(1,meshOpt,bcOpt,sourceOpt,materialOpt,pmlOpt)

## Write numerical solution to vtk file
## ====================================
file = df.File("realNumericalSoln.pvd")
file << HS.uSolnReal
file = df.File("imagNumericalSoln.pvd")
file << HS.uSolnImag

## Plot the numerical soltion
## ==========================
'''
df.plot(HS.domains,title="domain partitioned")
df.plot(HS.uSolnReal,title="real(soln)")
df.plot(HS.uSolnImag,title="imag(soln)")
'''

## Determine analytical solution with exact solution u = i*exp(ik|x|)/2k
## =====================================================================
'''
class AnalyticalSoln(df.Expression):
    def __init__(self,sourceLoc,waveNumber,returnReal=True, **kwargs):
        self.sourceLoc  = sourceLoc
	self.waveNumber = waveNumber
	self.returnReal = returnReal
    def eval(self,value,x):
        z = np.abs(x[0]-self.sourceLoc[0])
        uExact = 1j*np.exp(1j*self.waveNumber*z)/(2.0*self.waveNumber)
        if self.returnReal:
           value[0] = np.real(uExact)
        else:
           value[0] = np.imag(uExact)

ASolnReal  = AnalyticalSoln(sourceOpt['real']['pointSourceLoc'],omega,\
                            degree=HS.meshOpt['polynomialOrder'])
ASolnImag  = AnalyticalSoln(sourceOpt['real']['pointSourceLoc'],omega,\
                            returnReal=False,degree=HS.meshOpt['polynomialOrder'])

pASolnReal = df.project(ASolnReal,HS.VReal)
pASolnImag = df.project(ASolnImag,HS.VImag)

df.plot(pASolnReal,title="real(analytical soln)")
df.plot(pASolnImag,title="imag(analytical soln)")
df.interactive()
'''
