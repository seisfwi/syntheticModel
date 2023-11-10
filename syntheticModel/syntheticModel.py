import re
import math
import numba
import datetime
import random
import numpy as np
from datetime import datetime
import copy
from scipy.signal import butter, lfilter

usingSepVector=False

if "ON"=="@USE_SEP@":
  usingSepVector=True
  import SepTensor
  import genericIO
  import  Hypercube
  def getGeoField(hyper:Hypercube.hypercube,storage:str="dataFloat"):
    return SepTensor.getSepTensor(hyper,storage=storage)
else:
  import syntheticModel.regField as regField
  import syntheticModel.Hypercube as Hypercube
  def getGeoField(hyper:Hypercube.hypercube,storage:str="dataFloat"):
    return regField.regField(hyper,storage=storage)


@numba.njit(parallel=True)
def findMaxD(arr:np.ndarray):
    bot=np.zeros((arr.shape[0]),dtype=np.int32)
    for i3 in numba.prange(arr.shape[0]):
        bot[i3]=0
        for i2 in range(arr.shape[1]):
            i1=0
            found=False
           
            while i1 < arr.shape[2]-1 and not found:
                if arr[i3,i2,i1]>0:
                    found=True
                    if bot[i3] < i1:
                        bot[i3]=i1
                i1+=1
    return np.max(bot)
                    


        

class geoModel:

    def __init__(self, hyper,*args, **kw):
        """Create a new geomodel
            Create a new geoModel
          Method 1: geoModel(hyper,basement,primary,seed=-1)

            hyper : [no default] - Hypercube; A Hypercube describing the initial model. Should have three dimensions.
            basement : [no default] - dictionary; a dictionary of the basement property keys and their assosciated values
            primary : [no default] - string; primary property for the model. Must be a key in the provided 'basement' arguemen
              rgt : [False] - bool; whether to keep a float field of Relative Geologic 
                  Time (RGT)
          Method 2: geoModel(hyper,oldModel=oldModel)
            oldModel - Previous geoModel


        """
        if len(args)>0:
            basement=args[0]
            primary=args[1]

            if not isinstance(basement,dict):
                raise Exception("Expecting basement do be a dictionary")

            if not isinstance(hyper,Hypercube.hypercube):
                raise Exception("Expecting hyper to be a hypercube")

            if not isinstance(primary,str):
                raise Exception("Expecting primary to be a string")

            if not primary in basement:
                raise Exception("Expecting %s tobe a key in basment"%primary)
            self.intFields={"layer":getGeoField(hyper,storage="dataInt")}
            self.basements={"layer":-1}
            self.intFields["layer"].getNdArray().fill(0)
            self.floatFields={}
            self.fills={"layer":-1}
            for  fld in basement.keys():
                self.floatFields[fld]=getGeoField(hyper)
                self.floatFields[fld].getNdArray().fill(basement[fld])
                
                self.fills[fld]=-1
                self.basements[fld]=basement[fld]
            self._primary=primary
            self.currentEvent=0

            self.ax2=hyper.getAxis(2)
            self.ax3=hyper.getAxis(3)
            rgt=False
            if"rgt" in kw:
                if not isinstance(rgt, bool):
                    raise Exception("Expecting rgt to be a bool")
                    if kw["rgt"]:
                        self.getCreateFloatField("rgt",-1,0)
        else:
            if "oldModel" not in kw:
                raise Exception("Expeciting oldModel as parameter")
            if not isinstance(kw["oldModel"], geoModel):
                raise Exception("Expecting oldModel to be a geoModel")
            old=kw["oldModel"]
            self._primary=old._primary
            self.ax2=old.ax2
            self.ax3=old.ax3
            self.fills=copy.deepcopy(old.fills)
            self.basements=copy.deepcopy(old.basements)
            self.floatFields={}
            self.intFields={}
            for fld in old.floatFields.keys():
                self.floatFields[fld]=getGeoField(hyper)
            for fld in old.intFields.keys():
                self.intFields[fld]=getGeoField(hyper,storage="dataInt")
            self.currentEvent=old.currentEvent


    def get_primary_field_name(self):
        """Return primary field name"""
        return self._primary

    def get(self,param=None):
        """Return field
        Arguments:
        param : [none] - string: parameter to return
        Returns: geoField for param
        """
        if param is None:
          param=self.get_primary_field_name()

        return  self.getFloatField(param)


    def getCreateFloatField(self,param,fill,basement):
        """"
        Return (and create if it doesn't exist float field
          param - Name of field
          fill - What to fill the field with
          basement - What to put at the bottom of the field
        """
        
        if param not in self.floatFields:
            self.floatFields[param]=getGeoField(self.getPrimaryProperty().getHyper())
            self.basements[param]=basement
            self.fills[param]=fill
        return self.floatFields[param]

    def getCreateIntField(self,param,fill,basement):
        """"
        Return (and create if it doesn't exist float field
          param - Name of field
          fill - What to fill the field with
          basement - What to put at the bottom of the field
        """
        if param not in self.intFields:
            self.intFields[param]=getGeoField(self.getPrimaryProperty().getHyper(),storage="dataInt")
            self.basements[param]=basement
            self.fills[param]=fill
        return self.intFields[param]
        
    
    def getIntField(self,param:str):
        """Return field
        Arguments:
        param : [none] - string: parameter to return
        Returns: GeoGield for param
        """
        return  self.intFields[param]
    
    def getFloatField(self,param:str=None):
        """Return field
        Arguments:
        param : [none] - string: parameter to return
        Returns: getGeoField for param
        """
        if param is None:
          param=self._primary
        return self.floatFields[param]


    def getIndicatorI(self):
        """Return indicator int form"""
        return self.getIntField("indicator")

    def getIndicatorF(self):
        """Return indicator float form"""
        return self.getFloatField("indicator")

    def getBasementFloat(self,par):
        """Return basement value associated with a float field

          par - Name of float field
        """
        return self.basements[par]

    def findMaxDeposit(self):
        """Find the maximum i1 of a deposited layer"""
        return findMaxD(self.getLayer().getNdArray())

    def getCurrentEvent(self):
        """Get Current event number"""
        return self.currentEvent

    def getBasementInt(self,par):
        """Return basement value associated with an int field

          par - Name of int field
        """
        return self.basements[par]
    def getFillFloat(self,par):
        """Return fill value associated with a float field

          par - Name of float field
        """
        return self.fills[par]

    def getFillInt(self,par):
        """Return fil value associated with an int field

          par - Name of int field
        """
        return self.fills[par]

    def getLayer(self):
        """Return layer"""
        return self.getIntField("layer")

    def getPrimaryProperty(self):
        """Return primary property of model


        """
        return self.getFloatField()

    def getPrimary(self):
        """Get primary"""
        return self._primary

    def getNewEvent(self):
        """
          Get the event number fo the next event (after creating it)
        """
        self.currentEvent+=1
        return self.currentEvent

    def floatFieldExists(self,fld:str):
        """Check whether or not float field exists"""
        return fld in self.floatFields


    def getFloatFieldList(self):
        """
          Return a list of the float fields associated with model
        """
        return self.floatFields.keys()

    def getIntFieldList(self):
        """
          Return a list of the int fields assoicated with model
        """
        return self.intFields.keys()

    def expand(self,beg:int,end:int):
        """"
        Expand model and return a new copy

        beg -Amount to expand on top
        end - Amount to expand on bottom

        return @geoModel
        """
        axIn=self.getPrimaryProperty().getHyper().getAxis(1)
        ax1=Hypercube.axis(n=axIn.n+beg+end,o=axIn.o,d=axIn.d,label=axIn.label,unit=axIn.unit)
        hyp=Hypercube.hypercube(axes=[ax1,self.ax2,self.ax3])
        
        outM=geoModel(hyp,oldModel=self)
        
        
        for k,v in self.floatFields.items():
            base=self.basements[k]
            fill=self.fills[k]
            outA=outM.floatFields[k].getNdArray()
            inA=self.floatFields[k].getNdArray()
            outA[:,:,:beg]=self.fills[k]
            outA[:,:,beg:beg+axIn.n]=inA
            outA[:,:,beg+axIn.n:]=self.basements[k]
        for k,v in self.intFields.items():
            outA=outM.intFields[k].getNdArray()
            base=self.basements[k]
            fill=self.fills[k]
            inA=self.intFields[k].getNdArray()
            outA[:,:,:beg]=self.fills[k]
            outA[:,:,beg:beg+axIn.n]=inA
            outA[:,:,beg+axIn.n:]=self.basements[k]
        return outM

    def getGeoFields(self,*args):
        """returns all fields combined into one GeoFields"""
        #get number of Properties
        nProp = len(args)

        #primary hyper
        new_hyper = self.get().getHyper().clone()

        #add axis to hypercube
        new_hyper.addAxis(Hypercube.axis(n=nProp, o=0, d=1, label="field"))

        #create sepVec that will be written to disk
        out_sepVec =getGeoField(new_hyper)

        #copy from each property to geoGield
        for field_count, field in enumerate(args):
          #copy field to sepVec that will be written to disk
          out_sepVec.getNdArray()[field_count,
                                  ...] = self.get(param=field).getNdArray()

        # write to disk
        return out_sepVec

    def writeModel(self, filename, min_clip={}, max_clip={}):
        """write the model to disk. slow axis is 'properties' and fast axis is 
          depth,z
        Arguements:
          filename : string; file to write model
          min_clip : [0.0] - python dictionary; a dictionary with keys being 
            properties and values being the the lower clip allowed to be written 
            to disk
        """
        # get all fields into geoField
        out_sepVec = self.getGeoField()

        #get number of Properties
        nProp = len(self.basement)

        #copy from each property to geoField
        for field_count, field in enumerate(self.basement):
          # min clip defaults to zero unless otherwise provided in min_clip 
          # dictionary
          cur_min_clip = 0
          if field in min_clip:
            cur_min_clip = min_clip[field]
          if field in max_clip:
            cur_max_clip = max_clip[field]
          else:
            cur_max_clip = None

          #clip each field
          out_sepVec.getNdArray()[field_count, ...] = np.clip(
              out_sepVec.getNdArray()[field_count, ...],
              a_min=cur_min_clip,
          a_max=cur_max_clip)

        # write to disk
        out_sepVec.writeVec(filename)

    def writeLayers(self, filename):
        """write the layers field to disk.
        Arguements:
          filename : string; file to write model to
        """
        float_layers_sep = getGeoField(self.getLayer().getHyper())
        float_layers_sep.getNdArray()[:] = self.getLayer().getNdArray()
        # write to disk
        float_layers_sep.writeVec(filename)
