"""This module is used to make realistic looking salt diapers in 3D.

A Perlin noise generator is used to create two 2D noise fields for the top and
bottom of a salt body, respectively. The values in the noise field initially
range from -1 to 1 but are scaled to range from 0 to the max depth indices. In
this way the values in the noise fields represent the depth locations of the top
and bottom of the salt body. For each x and y location, salt is filled between 
the top depth and bottom depth. When the bottom depth exceeds the top depth, no
salt is filled.
"""
import numpy as np
from syntheticModel.model_noise import fractal
from numba import jit, njit, prange
import math
import syntheticModel.syntheticModel as syntheticModel
import syntheticModel.event as event
import datetime
import scipy

RAND_INT_LIMIT = 100000
AMPLITUDE_LEVELS = [1, 0.5, 0.125, 0.0625]

@njit(parallel=True)
def limit(beg2:float,beg3:float,end2:float,end3:float,dieF:float,top,bot):
  b2_1=int(beg2*top.shape[1])
  b2_0=max(0,int((beg2-dieF)*top.shape[1]))
  e2_1=min(top.shape[1],int(end2*top.shape[1]))
  e2_0=min(top.shape[1],int((dieF+end2)*top.shape[1]))
  b3_1=int(beg3*top.shape[0])
  b3_0=max(0,int((beg3-dieF)*top.shape[0]))
  e3_1=min(top.shape[0],int(end3*top.shape[0]))
  e3_0=min(top.shape[0],int(top.shape[0]*(dieF+end3)))
    
  if b3_0==b3_1:
    b3_0=b3_1-1
  if b2_0 == b2_1:
    b2_0=b2_1-1
  if e2_0==e2_1:
    e2_0=e2_1+1
  if e3_0 ==e3_1:
    e3_0=e3_1+1

  topE=np.mean(bot)
  botE=np.mean(top)

  for i3 in prange(top.shape[0]):
    for i2 in prange(top.shape[1]):
      if i2 >= b2_1 and i2 < e2_1 and i3 >=b3_1 and i3< e3_1:
        pass;
      elif i3<  b3_0 or i3 >= e3_0 or i2< b2_0 or i2 >= e2_0:
        top[i3,i2]=topE
        bot[i3,i2]=botE
      else:
        sc2=1
        sc3=1
        if i3 >= b3_0 and i3 <b3_1:
          sc3=float(i3-b3_0)/float(b3_1-b3_0)
        if i3 >=e3_1 and i3 <e3_0:
          sc3=1.-float(i3-e3_1)/float(e3_0-e3_1)
        if i2 >= b2_0 and i2 <b2_1:
          sc2=float(i2-b2_0)/float(b2_1-b2_0)
        if i2 >=e2_1 and i2 <e2_0:
          sc2=1.-float(i2-e2_1)/float(e2_0-e2_1)
        dst=math.sqrt(sc2*sc2+sc3*sc3)
        
        top[i3,i2]=(1.-dst)*topE+dst*top[i3,i2]
        bot[i3,i2]=(1.-dst)*botE+dst*bot[i3,i2]




def create_3d(n1,
              n2,
              n3,
              d=1,
              seed=None,
              salt_value=1.0,
              thickness=0.2,
              n_octaves_top=4,
              n_octaves_bot=3,
              amplitude_top=0.66,
              amplitude_bot=0.66,
              mid_salt_depth=0.4,
              beg2=0.,
              beg3=0.,
              end2=1.,
              end3=1.,
              dieF=.01,
              baseFreq=3.,
              mirror_top=False):
  """create a 3d model with a salt body
  
  See module description for the overall approach to the algorithm. User
  specifies the shape of the output model (n1,n2,n3) and the spatial,
  discretization d. 

  Args:
      n1 (int): size of first dimension output noise field 
      n2 (int): size of second dimension output noise field 
      n3 (int): size of third dimension output noise field 
      d (int, optional): size if model vozel in each direction. Defaults to 1.
      seed ([type], optional): Random seed. Same seed will produce same models.
        Defaults to None.
      salt_value (float, optional): Value assigned to cells in model containing
        salt. Defaults to 1.0.
      thickness (float, optional): Scaled amount to thicken salt. Reasonable
        values will be from [-0.5,0.5]. Defaults to 0.2.
      n_octaves_top (int, optional): Number of Perlin octaves to add to top salt
        . Defaults to 4.
      n_octaves_bot (int, optional): Number of Perlin octaves to add to bottom
        salt. Defaults to 3.
      amplitude_top (float, optional): Scaling to apply to top salt amplitudes.
        Reasonable values range from [0,1]. Defaults to 0.66.
      amplitude_bot (float, optional): Scaling to apply to bottom salt
        amplitudes. Reasonable values range from [0,1]. Defaults to 0.66.
      mid_salt_depth (float, optional): Relative depth of the center of the salt
        ranging from [0,1] where 0 is zero depth. Defaults to 0.4.
      mirror_top (bool, optional): Whether to make the bottom salt field
        mirror the top. Defaults to False.
      beg2,beg3,end2,end3 Limit for salt
      dieF -  How fast salt dies off outside box
      baseFreq - Base frequency for creating salt

  Returns:
      np array: 3d model with salt with shape (n1,n2,n3).
  """
  if seed is None:
    seed =int(datetime.datetime.utcnow().timestamp())
  np.random.seed(seed)

  # find top salt noise field
  top = find_2d_noise_field(n1, n2, d=1.0, noise_levels=n_octaves_top,baseFreq=baseFreq)

  # convert noise field values to depth indices
  top = (top - np.mean(top)) / np.amax(np.abs(top)) * amplitude_top
  top = top * n3 / 2.0 + n3 * (mid_salt_depth)
  if thickness is not None:
    top -= n3 * thickness / 2.0

  # find bottom salt noise field
  if mirror_top:
    np.random.seed(seed=seed)
  bot = find_2d_noise_field(n1, n2, d, noise_levels=n_octaves_bot,baseFreq=baseFreq)

  # convert noise field values to depth indices
  bot = (bot - np.mean(bot)) / np.amax(np.abs(bot)) * amplitude_bot
  bot = bot * n3 / 2.0 + n3 * (mid_salt_depth)
  if thickness is not None:
    bot += n3 * thickness / 2.0
 
  limit(beg2,beg3,end2,end3,dieF,top,bot)

  # fill salt between top and bottom indices
  return fill_between_fields(top, bot, n3, value=salt_value)

def find_2d_noise_field(n1, n2, d=1.0,baseFreq=3. ,noise_levels=4, normalize=True):
  """Create a 2d field of coherent noise using perlin noise.

  Args:
      n1 (int): size of first dimension output noise field
      n2 (int): size of second dimension output noise field
      d (float, optional): voxel size in both directions. Defaults to 1.0.
      noise_levels (int, optional): [description]. Defaults to 4.
      normalize (bool, optional): [description]. Defaults to True.
      baseFreq (float) : Base freq level. Defaults to 3.

  Returns:
      np array: noise field with shape (n1,n2)
  """
  ""

  # randomly shift origin
  o1 = np.random.rand() * n1 * d
  o2 = np.random.rand() * n2 * d
  p1 = np.linspace(o1, o1 + n1 * d, n1, endpoint=False)
  p2 = np.linspace(o2, o2 + n2 * d, n2, endpoint=False)
  p = np.stack(np.meshgrid(p1, p2, indexing='ij'), -1)
  p = np.reshape(p, (-1, 2)).T

  FREQUENCY_LEVELS=[baseFreq]
  for i in range(3): 
    FREQUENCY_LEVELS.append(FREQUENCY_LEVELS[-1])

  # get freq and amplitude values for perlin octaves
  freq_levels = list(np.array(FREQUENCY_LEVELS[:noise_levels]) / (n1 * d))
  amp_levels = AMPLITUDE_LEVELS[:noise_levels]
  field = fractal(p,
                  frequency=freq_levels,
                  n_octaves=noise_levels,
                  amplitude=amp_levels)
  field = np.reshape(field, (n1, n2))

  # normalize between -1 and 1
  if normalize:
    field = field / np.amax(np.abs(field))

  return field



@njit(parallel=True)
def fill_between_fields(top, bot, n3, value=1.0):
  """Create a 3d model with salt where top < bot

  Value of top and bottom represent the indices of top and bottom salt,
  respectively. Since index 0 is at top of model, fill with salt where top < bot.
  Args:
      top (2d numpy array): depth indices of top salt
      bot (2d numpy array): depth indices of bottom salt
      n3 (int): size of depth axis of output model
      value (float, optional): Value to assign salt voxels. Defaults to 1.0.

  Returns:
      np array: model with shape (top.shape + n3) filled in with salt.
  """
  assert top.shape == bot.shape

  n1 = top.shape[0]
  n2 = top.shape[1]

  filled = np.zeros((n1, n2, n3))
  for i1 in prange(n1):
    for i2 in prange(n2):
      i3_top = int(top[i1, i2])
      i3_top = max(i3_top, 0)
      i3_bot = int(bot[i1, i2])
      i3_bot = min(n3 - 1, i3_bot)
      if i3_top < i3_bot:
        filled[i1, i2, i3_top:i3_bot] = value
      else:
        top[i1,i2]=-1
        bot[i1,i2]=-1

  return filled,top,bot

@njit(parallel=True)
def fillModel(salt,fld,val):
  for i3 in prange(fld.shape[0]):
    for i2 in prange(fld.shape[1]):
      for i1 in prange(fld.shape[2]):
        if salt[i3,i2,i1] >0.01:
          fld[i3,i2,i1]=val
        
@njit(parallel=True)
def createHillMap(salt:np.ndarray,up:np.ndarray,topE:np.ndarray,botE:np.ndarray,maxS:float):
    shiftF=np.zeros(salt.shape,dtype=np.float32)
    saltE=np.zeros(salt.shape,dtype=np.float32)
    
    for i3 in prange(up.shape[0]):
        for i2 in prange(up.shape[1]):
            if topE[i3,i2] == -1:
                for i1 in range(up.shape[2]):
                    shiftF[i3,i2,i1]=i1+up[i3,i2,i1]*maxS
            else:
                for i1 in range(up.shape[2]):
                    shiftF[i3,i2,i1]=i1+up[i3,i2,i1]*maxS
    return shiftF

            

@njit(parallel=True)
def applyHillMap(fld,mapH):
    for i3 in prange(fld.shape[0]):
        for i2 in prange(fld.shape[1]):
            for i1 in range(fld.shape[2]):
                fld[i3,i2,i1]=fld[i3,i2,int(.5+mapH[i3,i2,i1])]

class salt(event.event):
  """Base class for depositing"""
  def __init__(self,**kw):
      """Initialize the base class for deposition"""
      super().__init__(**kw)
      self.salt=[]

  def applyBase(self,inM:syntheticModel.geoModel,
          seed=None,
              method_2d:bool=True,
              salt_value:dict={},
              thickness:float=0.2,
              n_octaves_top:int=4,
              n_octaves_bot:int=3,
              amplitude_top:float=0.66,
              amplitude_bot:float=0.66,
              mid_salt_depth:float=0.4,
              beg_freq:float=3.,
              beg2:float=0.,
              beg3:float=0.,
              end2:float=1.,
              end3:float=1.,
              dieF:float=.01,
              maxShift:float=20,
              applyHills:bool=False,
                indicateI:bool=False,
                indicateMark:int=1,
              bound0:float=.0,
              saveSalt:bool=False,
              mirror_top:bool=False):
    """
      method_2d(True): Whether to use two surface method (ony method for now)
      seed ([type], optional): Random seed. Same seed will produce same models.
        Defaults to None.
      salt_value:dict: Value assigned to cells in model containing
        salt. If not specified for property ignored
      thickness (float, optional): Scaled amount to thicken salt. Reasonable
        values will be from [-0.5,0.5]. Defaults to 0.2.
      n_octaves_top (int, optional): Number of Perlin octaves to add to top salt
        . Defaults to 4.
      n_octaves_bot (int, optional): Number of Perlin octaves to add to bottom
        salt. Defaults to 3.
      beg2,beg3,end2,end3  [0,0,1.,1.] Limit for salt
      dieF  [.05] Salt die off fraction
      beg_freq [3.] Initial frequncy for noise
      amplitude_top (float, optional): Scaling to apply to top salt amplitudes.
        Reasonable values range from [0,1]. Defaults to 0.66.
      amplitude_bot (float, optional): Scaling to apply to bottom salt
        amplitudes. Reasonable values range from [0,1]. Defaults to 0.66.
      mid_salt_depth (float, optional): Relative depth of the center of the salt
        ranging from [0,1] where 0 is zero depth. Defaults to 0.4.
      mirror_top (bool, optional): Whether to make the bottom salt field
        mirror the top. Defaults to False.
      applyHills [False] Whether or not conform sediments above to salt
      bound0 [.0] Where to stop the conforming of the sediments to the top 
      indicateI - [False] Indicate salt
      IndicateMark [1] Value to use to indicate Salt
      maxShift [20.] Maximum points to shift
         
    """
        
    ns=inM.getPrimaryProperty().getHyper().getNs()

    if  method_2d:
        slt,top,bot=create_3d(ns[2],ns[1],ns[0],seed=seed,salt_value=1,\
        thickness=thickness, n_octaves_top=n_octaves_top,n_octaves_bot=n_octaves_bot,\
        amplitude_top=amplitude_top,amplitude_bot=amplitude_bot,beg2=beg2,\
        beg3=beg3,end2=end2,end3=end3,dieF=dieF,\
                mid_salt_depth=mid_salt_depth,mirror_top=mirror_top,\
                           baseFreq=beg_freq)

        


        if applyHills:    
            shift=slt.copy()
            up=scipy.ndimage.gaussian_filter(shift,6.)
            mapH=createHillMap(slt,up,top,bot,maxShift)
        
        if saveSalt:
            self.salt.append(slt)
        outM=inM.expand(0,0)
        ievent=outM.getNewEvent()
        
        if indicateI:
            indicateInt=outM.getCreateIntField("indicator",0,0).getNdArray()
            fillModel(slt,indicateInt.getNdArray(),indicateMark)

        
        for fld in outM.getFloatFieldList():
            if applyHills:
                applyHillMap(outM.getFloatField(fld).getNdArray(),mapH)
            if fld in salt_value:
              fillModel(slt,outM.getFloatField(fld).getNdArray(),salt_value[fld])
        for fld in outM.getIntFieldList():
            if applyHills:
                applyHillMap(outM.getIntField(fld).getNdArray(),mapH)
            if fld in salt_value:
              fillModel(slt,outM.getIntField(fld).getNdArray(),salt_value[fld])
            elif fld == "layer":
              fillModel(slt,outM.getIntField(fld).getNdArray(),ievent)
           
            
    else:
        raise Exception("Only method_2d is valid for now")
    return outM
class basic(salt):
    def __init__(self,**kw):
        """
        Basic deposit, for now we have not specialized
        
        """
        super().__init__(**kw)