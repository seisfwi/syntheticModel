import numba
import numpy as np
import math
import syntheticModel.event as event
import syntheticModel.syntheticModel as syntheticModel
import syntheticModel.simplexNoise as simplexNoise
import datetime

@numba.njit()
def getPath(beg2:float,beg3:float,rnd:np.ndarray):
    path=np.zeros((rnd.shape[1]+1,2))
    p2=beg2
    p3=beg3
    path[0,0]=beg2
    path[0,1]=beg3
    for i in range(rnd.shape[1]):
        path[i+1,0]=path[i,0]+math.cos(rnd[1,i])
        path[i+1,1]=path[i,1]+math.sin(rnd[1,i])
    return path


@numba.njit()
def putRiver(path:np.ndarray,fld:np.ndarray,fill,thick):
    myt=np.zeros((int(thick)))
    for i in range(int(thick)):
        myt[i]=math.cos(float(i)*math.pi/2/thick)*thick
    for i in range(path.shape[0]-1):
        dx=abs(path[i+1,0]-path[i,0])
        dy=abs(path[i+1,1]-path[i,1])
        for iz in range(int(thick)):
            bx=path[i,0]-dy*myt[iz]
            ex=path[i,0]+dy*myt[iz]
            by=path[i,1]-dx*myt[iz]
            ey=path[i,1]+dx*myt[iz]
            if bx >= 0 and bx < fld.shape[1] and by >=0 and by < fld.shape[0]:
                fld[by:ey+1,bx:ex+1,iz]=fill
                
        
class erodeRiver(event.event):
    """Default class for eroding a river"""
    def __init__(self,**kw):
        super().__init__(**kw)

    def applyBase(self,inM:syntheticModel.geoModel,start2:float=0.5,noiseFreq:float=2.,
            noiseLacunarity:float=2.,noisePeristence:float=.5,\
        start3:float=.5, dist:float=1.5, noiseOctaves:int=3,\
        fill_prop=3000., thick:float=20.,\
            begAng:float=0,endAng:float=360.,
            indicateF:bool=False,indicateI:bool=False, indicateValue=1., seed=None)->syntheticModel.geoModel:
        """
        Erode a river

        Arguements:
        
            noiseFreq - [4.] Higher frequency means more faster change
            noiseLacunarity [2.] How multiple octaves increase in frequency
            noisePeristence [.5] How multiple octaves max value changes 
            noiseOctaves   [3] Number of noise octaves
            start2 - [.5] Position (relative to axis length) to start river
            start3 - [.5] Position (relative) to start river
            dist   - [1.5] Length (relative) of river
            fill_prop - [3000] Fill value for deposition for river channel (dictionary)
            thick - [20.] Thickness of river channel
            indicateF - [False] Whether or not to indicate the location of an erode 
            river event
            indicateI - [False] Whether or not to indicate the location of an erode 
            river event
            indicateValue - [1] Value to mark for a river
            begAng - [0] Begining possible angle for the river
            endAng - [360] Ending possible angle for the river
            seed - [None]  Random seed 

        Returns

            outModel - Returns updated model  
        """
        axes=inM.getPrimaryProperty().getHyper().axes
        
        exts=[ax.d*ax.n for ax in axes]

        if seed is None:
            seed =int(datetime.datetime.utcnow().timestamp())
        outM=inM.expand(0,0)
        m=simplexNoise.simplexNoise(seed)
        nlen=int(dist*max(axes[1].n,axes[2].n))
        
        
        
        minA=begAng*math.pi/180
        maxA=endAng/180*math.pi
        rng=(maxA-minA)
        sc=1
        for i in range(1,noiseOctaves):
            sc+=math.pow(noisePeristence,i+1)
        
        x=np.linspace(0,1,int(nlen),endpoint=False)
        y=np.linspace(0,1,10,endpoint=False)
        rnd=m.noise2D(x,y,n_octaves=noiseOctaves,\
                        frequency=noiseFreq,lucanarity=noiseLacunarity,\
                      persistence=noisePeristence)/sc*rng+minA

        path=getPath(float(start2*axes[1].n),float(start3*axes[2].n),rnd)
        outM=inM.expand(0,0)
        inext=outM.getNewEvent()

        for fld in outM.getFloatFieldList():
            if fld != "indicator" and fld != "rgt":
                if isinstance(fill_prop,dict):
                    if fld not in fill_prop:
                        raise Exception(f"{fld} no in fill_prop list")
                    fuse=fill_prop[fld]
                else:
                    fuse=fill_prop
            putRiver(path,outM.getFloatField(fld).getNdArray(),fuse,thick/axes[0].d)
        if indicateF:
            putRiver(path,outM.getCreateFloatField("indicator",0,0).getNdArray(),indicateValue,thick/axes[0].d)
        
        if indicateI:
            putRiver(path,outM.getCreateIntField("indicator",0,0).getNdArray(),indicateValue,thick/axes[0].d)
        
    
        putRiver(path,outM.getLayer().getNdArray(),inext,thick/axes[0].d)
        
        return outM
class basic(erodeRiver):
    def __init__(self,**kw):
        """
        Basic  erode river, for now we have not specialized
        
        """
        super().__init__(**kw)    
