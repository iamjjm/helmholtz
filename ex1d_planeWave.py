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

omega      = 2.0*np.pi*10

meshOpt    = {'nXElem':100,\
              'polynomialOrder':1,\
	      'stretchMesh':False,\
	     }

bcOpt      = {'left':{'DBC':True,\
                      'real':df.Constant(1.0),\
                      'imag':df.Constant(0.0),},\
              'right':{'DBC':True,\
                       'real':df.Constant(0.0),\
                       'imag':df.Constant(0.0),},\
	      }

sourceOpt   = {'real':{'choice':'none'},\
               'imag':{'choice':'none'},\
	      }

WN = WaveNumber(omega,waveSpeed,damping)
materialOpt = {'waveNumber':WN.evalWaveNumber,\
               'density':density,\
	      }

pmlOpt      = {'left':False,\
               'right':True,\
	       'exponent':2,\
	       'sigmaMax':5000,\
	       'numPmlWaveLengths':2,\
	       'pmlWaveNumber':omega,\
	      }

## Instantiate Helmoltz solver class with the options
## ==================================================
HS = HelmholtzSolver(1,meshOpt,bcOpt,sourceOpt,materialOpt,pmlOpt)

## Save the numerical soltion
## ==========================
file = df.File("realAnalyticalSoln.pvd")
file << HS.uSolnReal
file = df.File("imagAnalyticalSoln.pvd")
file << HS.uSolnImag

## Plot the numerical soltion
## ==========================
'''
df.plot(HS.domains,title="domain partitioned")
df.plot(HS.uSolnReal,title="real(soln)")
df.plot(HS.uSolnImag,title="imag(soln)")
'''

## Determine analytical solution with exact solution u = exp(ikx)
## =====================================================================
'''
class AnalyticalSoln(df.Expression):
    def __init__(self,waveNumber,returnReal=True):
	self.waveNumber = waveNumber
	self.returnReal = returnReal
    def eval(self,value,x):
        uExact = np.exp(1j*self.waveNumber*x)
        if self.returnReal:
           value[0] = np.real(uExact)
        else:
           value[0] = np.imag(uExact)

ASolnReal  = AnalyticalSoln(omega,degree=HS.meshOpt['polynomialOrder']))
ASolnImag  = AnalyticalSoln(omega,returnReal=False,degree=HS.meshOpt['polynomialOrder'])

pASolnReal = df.project(ASolnReal,HS.VReal)
pASolnImag = df.project(ASolnImag,HS.VImag)

df.plot(pASolnReal,title="real(analytical soln)")
df.plot(pASolnImag,title="imag(analytical soln)")
df.interactive()
'''
