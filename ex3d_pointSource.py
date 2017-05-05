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

omega      = 2.0*np.pi*3.0
meshOpt    = {'nXElem':30,\
              'nYElem':30,\
              'nZElem':30,\
              'polynomialOrder':1,\
	      'stretchMesh':False,\
	     }

bcOpt      = {'left':{'DBC':True,\
                      'real':df.Constant(0.0),\
                      'imag':df.Constant(0.0),},\
              'right':{'DBC':True,\
                       'real':df.Constant(0.0),\
                       'imag':df.Constant(0.0),},\
              'bottom':{'DBC':True,\
                        'real':df.Constant(0.0),\
                        'imag':df.Constant(0.0),},\
              'top':{'DBC':True,\
                     'real':df.Constant(0.0),\
                     'imag':df.Constant(0.0),},\
              'back':{'DBC':True,\
                      'real':df.Constant(0.0),\
                      'imag':df.Constant(0.0),},\
              'front':{'DBC':True,\
                       'real':df.Constant(0.0),\
                       'imag':df.Constant(0.0),},\
	      }

sourceOpt   = {'real':{'choice':'pointSource',\
	               'pointSourceLoc':np.array([0.5,0.5,0.5]),\
		       'pointSourceMag':1.0},
               'imag':{'choice':'none'},\
	      }

WN = WaveNumber(omega,waveSpeed,damping)
materialOpt  = {'waveNumber':WN.evalWaveNumber,\
                'density':density,\
	       }

pmlOpt       = {'left':True,\
                'right':True,\
                'bottom':True,\
                'top':True,\
                'back':True,\
                'front':True,\
                'exponent':2,\
                'sigmaMax':5000,\
                'numPmlWaveLengths':0.75,\
                'pmlWaveNumber':omega,\
               }

## Instantiate Helmoltz solver class with the options
## ==================================================
HS = HelmholtzSolver(3,meshOpt,bcOpt,sourceOpt,materialOpt,pmlOpt)

## Write numerical solution to vtk file
## ====================================

file = df.File("realNumericalSoln.pvd")
file << HS.uSolnReal
file = df.File("imagNumericalSoln.pvd")
file << HS.uSolnImag

## Determine analytical solution with exact solution u = exp(ik|x|)/4*pi*|x|
## =========================================================================
class AnalyticalSoln(df.Expression):
    def __init__(self,sourceLoc,waveNumber,returnReal=True, **kwargs):
        self.sourceLoc  = sourceLoc
	self.waveNumber = waveNumber
	self.returnReal = returnReal
    def eval(self,value,x):
        z = np.abs(x[0]-self.sourceLoc[0])
        uExact = np.exp(1j*self.waveNumber*z)/(4.0*np.pi*z)
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

## Write analytical solution to vtk file
## ====================================

file = df.File("realAnalyticalSoln.pvd")
file << HS.uSolnReal
file = df.File("imagAnalyticalSoln.pvd")
file << HS.uSolnImag
