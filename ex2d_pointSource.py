import os, sys, time
import numpy as np
import dolfin as df

from HelmholtzSolver import *

from scipy.special import hankel1
#import matplotlib.pylab as plt
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm

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

omega      = 2.0*np.pi*8.0

meshOpt    = {'nXElem':100,\
              'nYElem':100,\
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
              }

sourceOpt   = {'real':{'choice':'pointSource',\
	               'pointSourceLoc':np.array([0.5,0.5]),\
		       'pointSourceMag':1.0},
               'imag':{'choice':'none'},\
	      }


WN = WaveNumber(omega,waveSpeed,damping)
materialOpt = {'waveNumber':WN.evalWaveNumber,\
               'density':density,\
	       }

pmlOpt       = {'left':True,\
                'right':True,\
                'bottom':True,\
                'top':True,\
	        'exponent':2,\
	        'sigmaMax':5000,\
	        'numPmlWaveLengths':2,\
	        'pmlWaveNumber':omega,\
	       }

## Instantiate Helmoltz solver class with the options
## ==================================================
HS = HelmholtzSolver(2,meshOpt,bcOpt,sourceOpt,materialOpt,pmlOpt)

## Write numerical solution to vtk file
## ====================================
file = df.File("ex2d_pointSource_realNumericalSoln.pvd")
file << HS.uSolnReal
file = df.File("ex2d_pointSource_imagNumericalSoln.pvd")
file << HS.uSolnImag

## Plot the numerical soltion
## ==========================
'''
df.plot(HS.domains,title="domain partitioned")
df.plot(HS.uSolnReal,title="real(numerical soln)")
df.plot(HS.uSolnImag,title="imag(numerical soln)")
df.interactive()
'''

## Compare solution with exact solution u = i*H_0^1(ik|x|)/4
## =========================================================
'''
def analyticalSoln(X,Y,sourceLoc,waveNumber,returnReal=True):
    Z = np.sqrt((X-sourceLoc[0])**2 + (Y-sourceLoc[1])**2)
    uAnalyticalSoln = (1j/4.0)*hankel1(0,waveNumber*Z)
    if returnReal:
       return np.real(uAnalyticalSoln)
    else:
       return np.imag(uAnalyticalSoln)

x = np.linspace(0,1,501)
y = np.linspace(0,1,501)
X,Y = np.meshgrid(x,y)

sourceLoc = sourceOpt['real']['pointSourceLoc']

aSolnReal  = analyticalSoln(X,Y,sourceLoc,omega)
aSolnImag  = analyticalSoln(X,Y,sourceLoc,omega,returnReal = False)

fig = plt.figure(1)
fig.clf()

ax1   = fig.add_subplot(121)
plot1 = ax1.contourf(X,Y,aSolnReal,cmap=cm.coolwarm)
cbar1 = plt.colorbar(plot1)
cbar1.ax.set_ylabel('real(analytical solution)')

ax2   = fig.add_subplot(122)
plot2 = ax2.contourf(X,Y,aSolnImag,cmap=cm.coolwarm)
cbar2 = plt.colorbar(plot2)
cbar2.ax.set_ylabel('imag(analytical solution)')
plt.show()
'''
