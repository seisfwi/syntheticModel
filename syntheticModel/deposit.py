from numpy.core.numeric import _isclose_dispatcher
import syntheticModel.syntheticModel as syntheticModel
import syntheticModel.simplexNoise as simplexNoise
import numpy as np
import numba
import random
import math
import syntheticModel.event as event
import datetime
import syntheticModel.sincTable as sincTable
debugBounds=False
@numba.njit()
def calc1DDeposition(n1:int,thickness):
  """
  Calcualte a 1-D property function
  
  n1 - Number of samples
  thickness - 1-D array of thicknesses

  Returns:

  layers - 1-D Array layer number
  outVals   - 1-D Array with values
  
  """
  layers=np.zeros((n1,),dtype=np.float32)
  curThick=0
  ilayer=0
  for i in range(n1):
    curThick+=1
    if thickness[ilayer] < curThick:
      layers[ilayer]=i
      ilayer+=1
      curThick=0
  ilayer+=1
  return layers,ilayer



@numba.njit(parallel=True,boundscheck=debugBounds)
def getHeight(inA):
  """
  Find the height of the bottom of the undefined top layer
  
  inA - Input array
  
  """
  height=np.zeros((inA.shape[0],inA.shape[1]),dtype=np.int32)
  for i3 in numba.prange(inA.shape[0]):
    for i2 in range(inA.shape[1]):
      height[i3,i2]=0
      while inA[i3,i2,height[i3,i2]]==-1 and height[i3,i2] <inA.shape[2]-1:
        height[i3,i2]+=1

  return height



@numba.njit(parallel=True)
def applyNoise(in3D,noise,dev):
  """
  Apply noise to an array

  in3D - Input Array
  noise - Noise array
  dev - StdDev for noise
  
  
  """
  out=noise.copy()
  for i3 in numba.prange(noise.shape[0]):
    for i2 in range(noise.shape[1]):
      for i1 in range(noise.shape[2]):
        out[i3,i2,i1]=in3D[i3,i2,i1]+noise[i3,i2,i1]*dev

  return out

@numba.njit(parallel=True,boundscheck=debugBounds)
def modifyRGT(rgt,height,ievent):
  """
  Update RGT field

  rgt - RGT field
  height - Height at various locations
  ievent - Event 
 
  
  """
  for i3 in numba.prange(rgt.shape[0]):
    for i2 in range(rgt.shape[1]):
      b0=ievent
      d=1./float(height[i3,i2])
      for i1 in range(rgt.shape[2]):
          rgt[i3,i2,i1]=b0+d*i1



def getParams(param:str, kw:dict)->(int,dict):
  """"
  Search parameters for all correpsonding to specified parameter
  
   param- Parameter to look for
   kw - Input dictionary
  """
      

  pars={ "dev":0.,"noiseFreq":1.,"noiseLacunarity":2.,"noisePersistence":.5,"noiseOctaves":3,\
       "noiseAbs":False,"ratio":1.,"depthGrad":0.,"noiseZStretch":1.,"noiseDev":0.,\
         "interbedPropDev":0.,"prop":2500,"layer":None}
  kout={}
  for k,v in pars.items():
    if "%s_%s"%(param,k) in kw:
      kout[k]=kw["%s_%s"%(param,k)]
    elif k in kw:
      kout[k]=kw[k]
    else:
      kout[k]=pars[k]
  if f"{param}_prop" in kw:
    return 2,kout
  elif f"{param}_ratio" in kw:
    kout["ratio"]=kw[f"{param}_ratio"]
    return 1, kout
  else:
    return 0, kout
   

@numba.njit(parallel=True,boundscheck=debugBounds)
def createPropArray(nlayer,grad,noise3D,layer1D,val1D,noiseV,izOff):
    """"
    Create the property array field

    nlayer - Number of layers
    grad.   - gradient
    noise3D - Layer boundary noise
    layer1D - 1-D layer boundaries
    val1D   - Property values
    noiseV  - SPatial noise
    izOff   - Random array of locations to begin grabing from noiseV


    """
    outA=noise3D.copy()

    for i3 in numba.prange(outA.shape[0]):
        for i2 in range(outA.shape[1]):
            ilast=0
            for i1 in range(nlayer):
                inext=max(0,min(outA.shape[2],int(.5+noise3D[i3,i2,i1]+layer1D[i1])))
                for i in range(ilast,inext):
                    outA[i3,i2,i]=val1D[i1]+grad*i+noiseV[i3,i2,i+izOff[i1]]
                ilast=inext
            for i in range(ilast,noise3D.shape[2]):
                outA[i3,i2,i]=val1D[nlayer-1]+grad*i+noiseV[i3,i2,i+izOff[nlayer-1]]  

    return outA

@numba.njit(parallel=True,boundscheck=debugBounds)
def setRGT(nlayer,noise3D,layer1D,val1D,noiseV,rgt,icur):
  """"
  Set RGT 

  nlayer - Number of layers
  noise3D - Layer boundary noise
  layer1D - 1-D layer boundaries
  val1D   - Property values
  noiseV  - SPatial noise
  rgt     - Time
  icur    - Current event
  
  """
  f0=icur
  df=1./float(nlayer)
  outA=noise3D.copy()
  for i3 in numba.prange(outA.shape[0]):
    for i2 in range(outA.shape[1]):
      ilast=0
      for i1 in range(nlayer):
        inext=max(0,min(outA.shape[2],int(.5+noise3D[i3,i2,i1]+layer1D[i1])))

        m0=f0+df*i1
        dm=df/float(inext-ilast)
        for i in range(ilast,inext):
          rgt[i3,i2,i]=m0+dm*i
        ilast=inext
     
      m0=f0+df*(nlayer-1)
      dm=df/float(noise3D.shape[2]-ilast)
      for i in range(ilast,noise3D.shape[2]):
        rgt[i3,i2,i]=m0+dm*i
          
  return outA
@numba.njit(parallel=True,boundscheck=debugBounds)
def modifyFloatArray(array,height,dep,sinc,expand):
  """"
  Modify float array inserting new deposition

  array - Array to be modified
  height - Size of deposition as  function  x,y
  dep - What to put into the depsotion
  sinc - Sinc table
  expand - Whether or not to expand
  
  """
  if expand:
    mn=height.min()
  for i3 in numba.prange(array.shape[0]):
    for i2 in range(array.shape[1]):
      tmp=np.zeros((dep.shape[2]+8),dtype=np.float32)
      for i in range(dep.shape[2]):
        tmp[4+i]=dep[i3,i2,i]
      for i in range(4):
        tmp[i]=tmp[4]
        tmp[tmp.shape[0]-4+i]=tmp[tmp.shape[0]-5]

      if expand:
        d=mn/height[i3,i2]
        #d=1.-float(height[i3,i2])/float(dep.shape[2])
      else:
        d=1.
      for i1 in range(height[i3,i2]):
        v=d*i1
        i=int(v)
        isinc=min(sinc.shape[0]-1,int((v-i)*10000+.5))
        array[i3,i2,i1]=0

        for j in range(8):
          array[i3,i2,i1]+=tmp[i+j]*sinc[isinc,j]


@numba.njit(parallel=True,boundscheck=debugBounds)
def modifyLayer(array,height,ievent):
  """
  array - Layer to modify
  height - Height as a function of x and y
  ievent - Event"""
  for i3 in numba.prange(array.shape[0]):
    for i2 in range(array.shape[1]):
      for i1 in range(height[i3,i2]):
        array[i3,i2,i1]=ievent


class deposit(event.event):
  """Base class for depositing"""
  def __init__(self,**kw):
      """Initialize the base class for deposition"""
      super().__init__(**kw)

  def applyBase(self,inM:syntheticModel.geoModel,thick:int=30,depthGrad:float=30.,\
     interbedThickAvg:float=7.5,interbedThickDev:float=0.,interbedPropDev:float=0.,\
     prop:float=2500.,expand:float=True,noiseFreq:float=1, noiseLacunarity:float=2., \
       noisePersistence:float=.5,noiseAbs:bool=False,noiseZStretch=1.,interbedOctaves=3,\
      interbedDev=0.,interbedFreq=2.,interbedLacunarity=2.,interbedPersistence=.5,\
      interbedZStretch=1.,seed=None,\
      noiseOctaves:int=3,noiseDev:float=0.,layer=None,**kw)->syntheticModel.geoModel:
      """Deposit a layer
      Arguments:
      modIn - Input geomodel
      seed  - [None] - Random seed,set for reproducibility
      thick - [30] - int; Thickness of the layer in number of samples
      depthGrad - [0.] - float; Gradient of primary parameter as a function of depth
      interbedThickAvg [7.5] - int; mean thickness of interbed layers in sample
      interbedThickDev  [0.] - int; std deviation of interbed layers  (stdDev)
      interbedPropDev [0.] Variation in interbed property  (stdDev)
      prop - [2500] Basic property value
      expand - [false] Whether or not expand (pinchout) or deposit horizontally
      layer - [None] floatTensor1D containing values for deposit
      noiseFreq - [1.] Higher frequency means more faster change
      noiseLacunarity [2.] How multiple octaves increase in frequency
      noisePeristence [.5] How multiple octaves max value changes 
      noiseZStretch [1.] Whether to stretch or compress Z axis to change noise psttern
      noseOctaves   [3] Number of noise octaves
      noiseAbs  [False] Whether or not to take the absolution value of the noise function
      noiseDev  [0.] StdDev to noise field
      interbedDev [0.] StDev of interbed layer boundaries
      interbedFreq     [2.] Higher frequency means faster change
      interbedLacunarity [2.] How multiple octaves increase in frequency
      interbedPeristence [.5] How multiple octaves max value changes 
      interbedZStretch  [1.]  How much to stretch Z axis when calculating layer boundary noise
      interbedOctaves [3] Number of octaves 
      
      
      Arguments that can be specified independantly for each parameter. For example, if you have a parameter
        'p1' to specify the deviation you would use 'p1_dev'.



      _noiseFreq - [2.] Higher frequency means more faster change
      _noiseLacunarity [2.] How multiple octaves increase in frequency
      _noisePeristence [.5] How multiple octaves max value changes 
      _noseOctaves   [3] Number of noise octaves
      _noiseAbs  [False] Whether or not to take the absolution value of the noise function
      _noiseZStretch [1.] How much to stretch z axis to create asymetrical noise

      Method 1:
      _ratio - Specify A ratio between base propety and this property
      _dev   - Specify deviation from ratio      


      Method 2:
      _depthGrad - [0.] - float; Gradient of primary parameter as a function of depth
      _noiseDev -[0.]  - float; Spatial deviation for noise
      _interbedPropDev [0.] StdDev in interbed property 
      _prop - [2500] Basic property value
      _layer - [null] floatTensor1D containing values for deposit





      Returns
      output geomodel
      """
      if seed is None:
        seed =int(datetime.datetime.utcnow().timestamp())
      np.random.seed(seed)
      random.seed(seed)

      z=inM.getPrimaryProperty()
    
      axes=inM.getPrimaryProperty().getHyper().axes
      allArgs={**locals(),**kw}
      del allArgs["kw"]
      exts=[ax.d*ax.n for ax in axes]
      # cpp or python modes
      n1Use=inM.findMaxDeposit()+thick
      maxE=max(max(axes[0].d*n1Use,exts[1]),exts[2])
      if layer==None:
        layer1D,nlayer=calc1DDeposition(n1Use,np.random.normal(interbedThickAvg/axes[0].d,interbedThickDev/axes[0].d,\
            (n1Use,)))
        if abs(interbedPropDev)<0.0001:
            nlayer=1
        val1D=np.random.normal(prop,interbedPropDev,(n1Use,))
      else:
        if not isinstance(layer,np.ndarray):
            raise Exception("layer must be a floatTensor")
        val1D=layer
        layer1D=np.linspace(0,val1D.shape[0]-1,val1D.shape[0])
        nlayer=val1D.shape[0]
        if val1D.shape[0] >= n1Use:
          raise Exception("layer is not large enough must be at least %d",n1Use)
            
      outM=inM.expand(thick,0)
      height=getHeight(outM.getLayer().getNdArray())

      z=np.linspace(0,3.*n1Use*axes[0].d/maxE*noiseZStretch*4,4*n1Use,dtype=np.float32)
      y=np.linspace(0,3.*exts[1]/maxE,axes[1].n,dtype=np.float32)
      x=np.linspace(0.,3.*exts[2]/maxE,axes[2].n,dtype=np.float32)
      zL=np.linspace(0,3.*n1Use*axes[0].d/maxE*interbedZStretch,n1Use,dtype=np.float32)

      layerNG=simplexNoise.simplexNoise(random.randint(0,1000*1000*1000))
      noiseG=simplexNoise.simplexNoise(random.randint(0,1000*1000*1000)) 
      noiseV=noiseG.noise3D(z,y,x,frequency=noiseFreq,lucanarity=noiseLacunarity,\
        persistence=noisePersistence,n_octaves=noiseOctaves,amplitude=noiseDev)
      ievent=outM.getNewEvent()

      noise3D=layerNG.noise3D(zL,y,x,n_octaves=interbedOctaves,amplitude=interbedDev)
      tst=random.sample(range(n1Use,3*n1Use),nlayer)
      sendList = numba.typed.List()
      [sendList.append(x) for x in tst]
      baseValues=createPropArray(nlayer,depthGrad/axes[0].d,noise3D,layer1D,val1D,noiseV,\
        sendList)
      mainProp=inM.getPrimary()
      sinc=sincTable.table
      for fld in outM.getFloatFieldList():
          modify=False
          if fld==mainProp:
            modify=True
            noiseT=baseValues
          elif fld=="rgt": 
            rgt=outM.getFloatField("rgt")
            setRGT(nlayer,noise3D,layer1D,val1D,noiseV,rgt,icur)
          else:        
            method,kProp=getParams(fld,allArgs)
            noiseG=simplexNoise.simplexNoise(random.randint(0,1000*1000*1000))
            if method!=0:
              z=np.linspace(0,3.*n1Use*axes[0].d/maxE*kProp["noiseZStretch"],n1Use,dtype=np.float32)
              propV=noiseG.noise3D(z,y,x,n_octaves=kProp["noiseOctaves"],frequency=kProp["noiseFreq"],\
                lucanarity=kProp["noiseLacunarity"],persistence=kProp["noisePersistence"],\
                  amplitude=kProp["noiseDev"])
              if kProp["noiseAbs"]:
                propV=abs(propV)              
              modify=True
              if method==1:
                propV=noiseG.noise3D(z,y,x,n_octaves=kProp["noiseOctaves"],frequency=kProp["noiseFreq"],\
                lucanarity=kProp["noiseLacunarity"],persistence=kProp["noisePersistence"])          
                if kProp["noiseAbs"]:
                  propV=abs(propV)   
                baseS=baseValues*kProp["ratio"]
                noiseT=applyNoise(baseS,propV,kProp["dev"])
              elif method==2:
                propV=noiseG.noise3D(z,y,x,n_octaves=kProp["noiseOctaves"],frequency=kProp["noiseFreq"],\
                lucanarity=kProp["noiseLacunarity"],persistence=kProp["noisePersistence"],\
                  amplitude=kProp["noiseDev"])       
                if kProp["layer"]==None:
                  lVal1D=np.random.normal(kProp["prop"],kProp["interbedPropDev"],(n1Use,))
                else:
                  if not isinstance(kProp["layer"],np.ndArray):
                      raise Exception("layer must be a np.ndarray")
                  loc1D=layer.getNdArray()
                  lVal1D=np.linspace(0,loc1D.shape[0]-1,loc1D.shape[0])
                  nlayerD=loc1D.shape[0]
                  if loc1D.shape[0] >= n1Use:
                    raise Exception("layer is not large enough must be at least %d",n1Use)
                noiseT=createPropArray(nlayer,kProp["depthGrad"]/axes[0].d,\
                  noise3D,layer1D,lVal1D,propV,sendList)
              else:
                raise Exception("Unknown methood to set param ",fld)
          if modify:
            out=outM.getFloatField(fld).getNdArray()
            modifyFloatArray(out,height,noiseT,sinc,expand)
      
      layer=outM.getLayer().getNdArray()
      modifyLayer(layer,height,ievent)
      return outM

          
      

class basic(deposit):
    def __init__(self,**kw):
        """
        Basic deposit, for now we have not specialized
        
        """
        super().__init__(**kw)


