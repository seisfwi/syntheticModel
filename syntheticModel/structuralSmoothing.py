import pathlib
import numpy as np
import os
jar_filepath=pathlib.Path(os.environ["JTK_BASE"]+"/edu-mines-jtk-1.0.0.jar")
assert jar_filepath.exists()

import jpype
import jpype.imports

if not jpype.isJVMStarted():  
  jpype.startJVM(classpath=[str(jar_filepath)])

import edu.mines.jtk.dsp as dsp
import edu.mines.jtk.util.ArrayMath as am

def smoother(input_np,mod=None,planar_semblance:bool=True, hw:int =2, hw2:int =2):
  """function for applying structual smoothing to a 3d image.
    Args:
        input_np - 3d numpy array - input image to be smoothed
        planar_semblance - boolean - whether to apply smoothing along
            planar features (True) or linear features (False)
        hw - float - half-width of semblance filter.
        hw2 - float - half-width of local smoothing filter.
        mod - 3d numpy array to smooth (use input if not specified)
    Returns:
        A 3d numpy array which is the input array smoothed along geologic layers
    """
  data_j = jpype.JArray(jpype.JFloat, input_np.ndim)(input_np)

  # build LocalOrientFilter which is used to make local estimates
  # of orientations of features in images.
  lof = dsp.LocalOrientFilter(4.0)

  # get 3d structure eigentensor (3x3) by applying lof filter to input data
  s = lof.applyForTensors(data_j)

  # Estimate semblance
  lsf = dsp.LocalSemblanceFilter(hw, hw)
  if planar_semblance:
    smb = lsf.semblance(dsp.LocalSemblanceFilter.Direction3.VW, s, data_j)
    s.setEigenvalues(0.0, 1.0, 1.0)
  else:
    smb = lsf.semblance(dsp.LocalSemblanceFilter.Direction3.W, s, data_j)
    s.setEigenvalues(0.0, 0.0, 1.0)
  smb = am.mul(smb, smb)
  smb = am.mul(smb, smb)
  if not isinstance(mod,np.ndarray):
    data_in=data_j
  else:
    data_in = jpype.JArray(jpype.JFloat, input_np.ndim)(mod)


  # Smooth with semblance
  datasm = am.copy(data_in)
  c = hw2 * (hw2 + 1) /6.
  smooth = dsp.LocalSmoothingFilter()
  
  smooth.apply(s,c,smb, data_in, datasm)
  return np.array(datasm, dtype=np.float32)
