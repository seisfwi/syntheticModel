from operator import iadd
import opensimplex
import datetime
import numba
import numpy as np

@numba.njit(parallel=True)
def scaleVec(x,sc):
    xuse=x.copy()
    for i in numba.prange(x.shape[0]):
        xuse[i]=x[i]*sc
    return xuse

@numba.njit(parallel=True)
def addMat3(outA,inA,sc):
    for i3 in numba.prange(inA.shape[0]):
        for i2 in range(inA.shape[1]):
            for i1 in range(inA.shape[2]):
                outA[i3,i2,i1]+=inA[i3,i2,i1]*sc
    


class simplexNoise:
    """Interface for opensimplex"""
    
    def __init__(self,seed:int=None):
        if seed is None:
            seed=int(datetime.datetime.utcnow().timestamp())
        self.seed=seed
    
    def noise2D(self,x:np.ndarray,y:np.ndarray,n_octaves:int=3,frequency:float=1.0, \
      amplitude:float=1.0,lucanarity:float=2.,persistence:float=0.5)->np.ndarray:
    
        amps=[amplitude]
        freqs=[frequency]
        for i in range(n_octaves-1):
            amps.append(amps[i]*persistence)
            freqs.append(freqs[i]*lucanarity)
        
        return self.noise2DF(x,y,freqs,amps)
        
    def noise2DF(self,x:np.ndarray,y:np.ndarray,freqs:list,amps:list)->np.ndarray:
        
        ar=np.zeros((y.shape[0],x.shape[0]))
        for i in range(len(freqs)):
            opensimplex.seed(self.seed)
            self.seed+=1
            ar+=opensimplex.noise2array(x*4*freqs[i],y*4*freqs[i])*amps[i]
        return ar
        
    def noise3D(self,x:np.ndarray,y:np.ndarray,z:np.ndarray,frequency:float=1.0, \
      amplitude:float=1.0,n_octaves:int=3,persistence:float=0.5,lucanarity:float=2.)->np.ndarray:
    
        amps=[amplitude]
        freqs=[frequency]
        for i in range(n_octaves-1):
            amps.append(amps[i]*persistence)
            freqs.append(freqs[i]*lucanarity)
        
        return self.noise3DF(x,y,z,freqs,amps)

    def noise3DF(self,x:np.ndarray,y:np.ndarray,z:np.ndarray,freqs:list,amps:list)->np.ndarray:
        
        ar=np.zeros((z.shape[0],y.shape[0],x.shape[0]))
        for i in range(len(freqs)):
            opensimplex.seed(self.seed)
            self.seed+=1
            xu=scaleVec(x,4*freqs[i])
            yu=scaleVec(y,4*freqs[i])
            zu=scaleVec(z,4*freqs[i])
            xt=opensimplex.noise3array(xu,yu,zu)
            addMat3(ar,xt,amps[i])
        return ar   