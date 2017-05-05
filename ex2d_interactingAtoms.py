## =====================================================================================
'''
This problem is borrowed from "A Collection of 2D Elliptic Problems for Testing AdaptiveGrid Refinement Algorithms". It is highly oscillatory near the origin, with the wavelength decreasing closer to the origin. The number of oscillations, N , is determined by the parameter alpha = 1/(N *pi). We demonstrate that the code produces the right result for N = 10, but for N>10, the present mesh will not yield accurate results.
'''
## =====================================================================================

import os, sys, time
import numpy as np
import dolfin as df

from HelmholtzSolver import *

from numpy import sin,cos,power
#import matplotlib.pylab as plt
#from matplotlib import cm
## =====================================================================================
## Specify frequency of the problem
omega = 1.0
## Specify problem specific parameters
alpha = 1.0/(10*np.pi)
#-----------------------------------------------------------------------------------
class WaveSpeed(object):
    def __init__(self,a):
        self.a = a
    def waveSpeedFun(self,x):
        return np.power(self.a+np.sqrt(x[0]**2 + x[1]**2), 2)
#-----------------------------------------------------------------------------------
def damping(x):
    return 0.0
#-----------------------------------------------------------------------------------
def density(x):
    return 1.0
#-----------------------------------------------------------------------------------
def analyticalSolnFun(X,a):
    x = X[0]
    y = X[1]
    r = np.sqrt(x**2 + y**2)
    apr = a+r
    return sin(1.0/apr)
#-----------------------------------------------------------------------------------
def sourceFun(X,a):
    x = X[0]
    y = X[1]

    r = np.sqrt(x**2 + y**2)
    apr = a+r

    d2x2 = +2*(x**2)*cos(1/apr)/(power(r,2)*power(apr,3)) \
             +(x**2)*cos(1/apr)/(power(r,3)*power(apr,2)) \
                    -cos(1/apr)/(power(r,1)*power(apr,2)) \
             -(x**2)*sin(1/apr)/(power(r,2)*power(apr,4))

    d2y2 = +2*(y**2)*cos(1/apr)/(power(r,2)*power(apr,3)) \
             +(y**2)*cos(1/apr)/(power(r,3)*power(apr,2)) \
                    -cos(1/apr)/(power(r,1)*power(apr,2)) \
             -(y**2)*sin(1/apr)/(power(r,2)*power(apr,4))

    return -d2x2 -d2y2 -sin(1.0/apr)/power(apr,4)
#-----------------------------------------------------------------------------------

meshOpt    = {'nXElem':350,\
              'nYElem':350,\
              'polynomialOrder':1,\
	      'stretchMesh':True,\
	      'stretchParam':np.array([4,4]),\
	     }

analyticalSolnExpr = UserScalarExpression(degree=meshOpt['polynomialOrder'])
analyticalSolnExpr.specifyUserFunction(analyticalSolnFun,alpha)
bcOpt      = {'left':{'DBC':True,\
                      'real':analyticalSolnExpr,\
                      'imag':df.Constant(0.0),},\
              'right':{'DBC':True,\
                       'real':analyticalSolnExpr,\
                       'imag':df.Constant(0.0),},\
              'bottom':{'DBC':True,\
                        'real':analyticalSolnExpr,\
                        'imag':df.Constant(0.0),},\
              'top':{'DBC':True,\
                     'real':analyticalSolnExpr,\
                     'imag':df.Constant(0.0),},\
              }

elem = df.FiniteElement('DG', df.triangle, 0)
sourceExpr = UserScalarExpression(element=elem)
sourceExpr.specifyUserFunction(sourceFun,alpha)

sourceOpt   = {'real':{'choice':'custom',\
                       'expression':sourceExpr},\
               'imag':{'choice':'none'},\
	      }

WS = WaveSpeed(alpha)
WN = WaveNumber(omega,WS.waveSpeedFun,damping)
materialOpt  = {'waveNumber':WN.evalWaveNumber,\
               'density':density,\
	       }

pmlOpt       = {'left':False,\
                'right':False,\
                'bottom':False,\
                'top':False,\
               }

## Instantiate Helmoltz solver class with the options
## ==================================================
HS = HelmholtzSolver(2,meshOpt,bcOpt,sourceOpt,materialOpt,pmlOpt)

## Determine some helpful stats about the numerical problem
## ========================================================
print "max waveNumber = ",WN.evalWaveNumber(np.array([0,0]))
print "min waveNumber = ",WN.evalWaveNumber(np.array([1,1]))
print "min WaveLength = ",2*np.pi/WN.evalWaveNumber(np.array([0,0]))
print "max WaveLength = ",2*np.pi/WN.evalWaveNumber(np.array([1,1]))
print "smallest element =",HS.mesh.hmin()
print "largest  element =",HS.mesh.hmax()


## Plot the numerical soltion
## ==========================
'''
df.plot(HS.mesh,title="mesh")
df.plot(HS.uSolnReal,title="real(numerical soln)")
df.interactive()
'''

## Write numerical solution to vtk file
## ====================================
file = df.File("ex2d_interactingAtoms_realNumericalSoln.pvd")
file << HS.uSolnReal

## Determine analytical solution matplotlib interpolation
## ======================================================
'''
def analyticalSolnForMatplotLib(X,Y,a):
    R = np.sqrt(X**2 + Y**2)
    return sin(1/(a+R))

x = np.linspace(0,1,101)
y = np.linspace(0,1,101)
X,Y = np.meshgrid(x,y)

aSoln = analyticalSolnForMatplotLib(X,Y,alpha)

fig = plt.figure(1)
fig.clf()
ax1   = fig.add_subplot(111)
plot1 = ax1.contourf(X,Y,aSoln,cmap=cm.coolwarm)
cbar1 = plt.colorbar(plot1)
cbar1.ax.set_ylabel('analytical solution')

pASolnReal = df.interpolate(analyticalSolnExpr,HS.VReal)
df.plot(pASolnReal,title="real(analytical soln)")
plt.show()
'''
