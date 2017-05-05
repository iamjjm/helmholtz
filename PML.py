#=======================================================================================

import os, sys, time
import numpy as np
import dolfin as df
 
#=======================================================================================

class PMLSubDomain(df.SubDomain):
    """ A class that defines PML subdomains in 1D and 2D """
    #-----------------------------------------------------------------------------------
    def __init__(self,tDim,pmlOpt):
    #-----------------------------------------------------------------------------------
        df.SubDomain.__init__(self)
        self.tDim   = tDim
	self.pmlOpt = pmlOpt
    #-----------------------------------------------------------------------------------
    def inside(self, x, on_boundary):
    #-----------------------------------------------------------------------------------
        if self.tDim==1:
	   return not(df.between(x[0], \
	              (self.pmlOpt['minLimit'][0],self.pmlOpt['maxLimit'][0])))
        if self.tDim==2:
	   return not(df.between(x[0], \
	              (self.pmlOpt['minLimit'][0],self.pmlOpt['maxLimit'][0])) \
                      and \
                      df.between(x[1], \
                      (self.pmlOpt['minLimit'][1],self.pmlOpt['maxLimit'][1])))
        if self.tDim==3:
	   return not(df.between(x[0], \
	              (self.pmlOpt['minLimit'][0],self.pmlOpt['maxLimit'][0])) \
                      and \
                      df.between(x[1], \
                      (self.pmlOpt['minLimit'][1],self.pmlOpt['maxLimit'][1]))
                      and \
                      df.between(x[2], \
                      (self.pmlOpt['minLimit'][2],self.pmlOpt['maxLimit'][2])))

#=======================================================================================

class MaterialScalar1D(df.Expression):
    #-----------------------------------------------------------------------------------
    def __init__(self, materialOpt, sigma, pmlOpt, returnReal=True, **kwargs):
    #-----------------------------------------------------------------------------------
        self.waveNumber = materialOpt['waveNumber']
        self.density    = materialOpt['density']
        self.sigma      = sigma
	self.pmlOpt     = pmlOpt
	self.returnReal = returnReal
    #-----------------------------------------------------------------------------------
    def eval(self, value, x):
    #-----------------------------------------------------------------------------------
        kx = 1 + 1j*self.sigma(x[0],self.pmlOpt,axisDir=0)/self.waveNumber(x)
	if self.returnReal:
	   value[0] = np.real(kx*np.power(self.waveNumber(x),2))/self.density(x)
	else:
	   value[0] = np.imag(kx*np.power(self.waveNumber(x),2))/self.density(x)

#=======================================================================================

class MaterialTensor1D(df.Expression):
    #-----------------------------------------------------------------------------------
    def __init__(self, materialOpt, sigma, pmlOpt, returnReal=True, **kwargs):
    #-----------------------------------------------------------------------------------
        self.waveNumber = materialOpt['waveNumber']
        self.density    = materialOpt['density']
        self.sigma      = sigma
	self.pmlOpt     = pmlOpt
	self.returnReal = returnReal
    #-----------------------------------------------------------------------------------
    def eval(self, value, x):
    #-----------------------------------------------------------------------------------
	value[:] = np.zeros(self.value_size())
        kx = 1 + 1j*self.sigma(x[0],self.pmlOpt,axisDir=0)/self.waveNumber(x)
	if self.returnReal:
	   value[0] = np.real(1.0/kx)/self.density(x)
	else:
	   value[0] = np.imag(1.0/kx)/self.density(x)
    #-----------------------------------------------------------------------------------
    def value_shape(self):
    #-----------------------------------------------------------------------------------
        return (1,1)

#=======================================================================================

class MaterialScalar2D(df.Expression):
    #-----------------------------------------------------------------------------------
    def __init__(self, materialOpt, sigma, pmlOpt, returnReal=True, **kwargs):
    #-----------------------------------------------------------------------------------
        self.waveNumber = materialOpt['waveNumber']
        self.density    = materialOpt['density']
        self.sigma      = sigma
	self.pmlOpt     = pmlOpt
	self.returnReal = returnReal
    #-----------------------------------------------------------------------------------
    def eval(self, value, x):
    #-----------------------------------------------------------------------------------
        kx = 1 + 1j*self.sigma(x[0],self.pmlOpt,axisDir=0)/self.waveNumber(x)
        ky = 1 + 1j*self.sigma(x[1],self.pmlOpt,axisDir=1)/self.waveNumber(x)
	if self.returnReal:
	   value[0] = np.real(kx*ky*np.power(self.waveNumber(x),2))/self.density(x)
	else:
	   value[0] = np.imag(kx*ky*np.power(self.waveNumber(x),2))/self.density(x)

#=======================================================================================

class MaterialTensor2D(df.Expression):
    #-----------------------------------------------------------------------------------
    def __init__(self, materialOpt, sigma, pmlOpt, returnReal=True, **kwargs):
    #-----------------------------------------------------------------------------------
        self.waveNumber = materialOpt['waveNumber']
        self.density    = materialOpt['density']
        self.sigma      = sigma
	self.pmlOpt     = pmlOpt
	self.returnReal = returnReal
    #-----------------------------------------------------------------------------------
    def eval(self, value, x):
    #-----------------------------------------------------------------------------------
	value[:] = np.zeros(self.value_size())
        kx = 1 + 1j*self.sigma(x[0],self.pmlOpt,axisDir=0)/self.waveNumber(x)
        ky = 1 + 1j*self.sigma(x[1],self.pmlOpt,axisDir=1)/self.waveNumber(x)
	if self.returnReal:
	   value[0] = np.real(ky/kx)/self.density(x)
	   value[3] = np.real(kx/ky)/self.density(x)
	else:
	   value[0] = np.imag(ky/kx)/self.density(x)
	   value[3] = np.imag(kx/ky)/self.density(x)
    #-----------------------------------------------------------------------------------
    def value_shape(self):
    #-----------------------------------------------------------------------------------
        return (2,2)

#=======================================================================================

class MaterialScalar3D(df.Expression):
    #-----------------------------------------------------------------------------------
    def __init__(self, materialOpt, sigma, pmlOpt, returnReal=True, **kwargs):
    #-----------------------------------------------------------------------------------
        self.waveNumber = materialOpt['waveNumber']
        self.density    = materialOpt['density']
        self.sigma      = sigma
	self.pmlOpt     = pmlOpt
	self.returnReal = returnReal
    #-----------------------------------------------------------------------------------
    def eval(self, value, x):
    #-----------------------------------------------------------------------------------
        kx = 1 + 1j*self.sigma(x[0],self.pmlOpt,axisDir=0)/self.waveNumber(x)
        ky = 1 + 1j*self.sigma(x[1],self.pmlOpt,axisDir=1)/self.waveNumber(x)
        kz = 1 + 1j*self.sigma(x[1],self.pmlOpt,axisDir=2)/self.waveNumber(x)
	if self.returnReal:
	   value[0] = np.real(kx*ky*kz*np.power(self.waveNumber(x),2))/self.density(x)
	else:
	   value[0] = np.imag(kx*ky*kz*np.power(self.waveNumber(x),2))/self.density(x)

#=======================================================================================

class MaterialTensor3D(df.Expression):
    #-----------------------------------------------------------------------------------
    def __init__(self, materialOpt, sigma, pmlOpt, returnReal=True, **kwargs):
    #-----------------------------------------------------------------------------------
        self.waveNumber = materialOpt['waveNumber']
        self.density    = materialOpt['density']
        self.sigma      = sigma
	self.pmlOpt     = pmlOpt
	self.returnReal = returnReal
    #-----------------------------------------------------------------------------------
    def eval(self, value, x):
    #-----------------------------------------------------------------------------------
	value[:] = np.zeros(self.value_size())
        kx = 1 + 1j*self.sigma(x[0],self.pmlOpt,axisDir=0)/self.waveNumber(x)
        ky = 1 + 1j*self.sigma(x[1],self.pmlOpt,axisDir=1)/self.waveNumber(x)
        kz = 1 + 1j*self.sigma(x[2],self.pmlOpt,axisDir=2)/self.waveNumber(x)
	if self.returnReal:
	   value[0] = np.real(ky*kz/kx)/self.density(x)
	   value[4] = np.real(kz*kx/ky)/self.density(x)
	   value[8] = np.real(kx*ky/kz)/self.density(x)
	else:
	   value[0] = np.imag(ky*kz/kx)/self.density(x)
	   value[4] = np.imag(kz*kx/ky)/self.density(x)
	   value[8] = np.imag(kx*ky/kz)/self.density(x)
    #-----------------------------------------------------------------------------------
    def value_shape(self):
    #-----------------------------------------------------------------------------------
        return (3,3)

#=======================================================================================

#-----------------------------------------------------------------------------------
def findPmlLimits(tDim,mesh,bcOpt,pmlOpt, InteriorProb = True):
#-----------------------------------------------------------------------------------

    for bFace in bcOpt['bFaceList']:
        if pmlOpt[bFace]==True:
	   InteriorProb = False
	   break

    if not(InteriorProb):
       numPmlWaveLengths = pmlOpt['numPmlWaveLengths']
       pmlWaveNumber     = pmlOpt['pmlWaveNumber']
       lengthPml = numPmlWaveLengths*np.pi*2.0/pmlWaveNumber

    minLimit = dict.fromkeys([0,1,2])
    maxLimit = dict.fromkeys([0,1,2])

    ## 1 dimensional case
    if pmlOpt['left']==True:
       minLimit[0] = np.amin(mesh.coordinates(),axis=0)[0] + lengthPml
    else:
       minLimit[0] = np.amin(mesh.coordinates(),axis=0)[0]

    if pmlOpt['right']==True:
       maxLimit[0]= np.amax(mesh.coordinates(),axis=0)[0]  - lengthPml
    else:
       maxLimit[0]= np.amax(mesh.coordinates(),axis=0)[0]

    ## 2 dimensional case
    if tDim>1:
       if pmlOpt['bottom']==True:
          minLimit[1]= np.amin(mesh.coordinates(),axis=0)[1] + lengthPml
       else:
          minLimit[1]= np.amin(mesh.coordinates(),axis=0)[1]

       if pmlOpt['top']==True:
          maxLimit[1]= np.amax(mesh.coordinates(),axis=0)[1] - lengthPml
       else:
          maxLimit[1]= np.amax(mesh.coordinates(),axis=0)[1]

    ## 3 dimensional case
    if tDim>2:
       if pmlOpt['back']==True:
          minLimit[2]= np.amin(mesh.coordinates(),axis=0)[2] + lengthPml
       else:
          minLimit[2]= np.amin(mesh.coordinates(),axis=0)[2]

       if pmlOpt['front']==True:
          maxLimit[2]= np.amax(mesh.coordinates(),axis=0)[2] - lengthPml
       else:
          maxLimit[2]= np.amax(mesh.coordinates(),axis=0)[2]

    return minLimit,maxLimit

#---------------------------------------------------------------------------------------
def sigmaFun(x,pmlOpt,axisDir):
#---------------------------------------------------------------------------------------
    distToMinLimit = x - pmlOpt['minLimit'][axisDir]
    distToMaxLimit = x - pmlOpt['maxLimit'][axisDir]
    if (np.sign(distToMinLimit)==-1):
       sigma = pmlOpt['sigmaMax']*np.power(np.abs(distToMinLimit),pmlOpt['exponent'])
    elif (np.sign(distToMaxLimit)==+1):
       sigma = pmlOpt['sigmaMax']*np.power(np.abs(distToMaxLimit),pmlOpt['exponent'])
    else:
       sigma = 0.0

    return sigma

#=======================================================================================
