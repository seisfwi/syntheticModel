import numpy as np
import syntheticModel.Hypercube as Hypercube

class regField:
    """Basic wrapper class for a numpy array"""
    def __init__(self,hyper:Hypercube.hypercube,storage:str="dataFloat"):
        """hyper - Hypercube
           storage - Storage type """
        
        self._hyper=hyper
        ns=self._hyper.getNs()
        if storage=="dataFloat":

            self._vals=np.zeros((ns[2],ns[1],ns[0]),dtype=np.float32)
        elif storage=="dataInt":
            self._vals=np.zeros((ns[2],ns[1],ns[0]),dtype=np.int32)
    
    def getHyper(self):
        """Return the hypercube"""
        return self._hyper
    
    def getNdArray(self):
        """Return the numpy array"""
        return self._vals

    

