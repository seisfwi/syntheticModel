import syntheticModel.syntheticModel as syntheticModel
import syntheticModel.event as event
import datetime

class erodeFlat(event.event):
    """Base class for eroding a plane"""
    def __init__(self,**kw):
        """Initialize the base class for erode flat"""
        super().__init__(**kw)

    def applyBase(self,inM:syntheticModel.geoModel,depth:float=50.)->syntheticModel.geoModel:
        """Squish a model

        Arguements:

        depth - [50.] Depth (axis 1) to slice off

        Returns

        outModel - Returns updated model  
        """

        axes=inM.getPrimaryProperty().getHyper().axes
        outM=inM.expand(0,0)
        
        ic=int(depth/axes[0].d)
        
        lay=outM.getLayer().getNdArray()
        
        lay[:,:,:ic]=-1
        
        return outM



class basic(erodeFlat):
    def __init__(self,**kw):
        """
        Basic erosion plane, for now we have not specialized
        
        """
        super().__init__(**kw)       