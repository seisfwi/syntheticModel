import syntheticModel.syntheticModel as syntheticModel
import syntheticModel.Hypercube as Hypercube
import re
import math
import numba
import numpy as np
from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = lfilter(b, a, data)
	return y




import numba
import scipy
@numba.njit()
def reflectivity(vel):
    ref=vel.copy()
    for i3 in range(1,ref.shape[0]-1):
        for i2 in range(1,ref.shape[1]-1):
            ref[i3,i2,0]=0
            for i1 in range(1,ref.shape[2]-1):
                ref[i3,i2,i1]=(vel[i3,i2,i1]*6-(vel[i3,i2,i1-1]+vel[i3,i2,i1+1]\
                                                   +vel[i3,i2+1,i1]+vel[i3,i2-1,i1]\
                                                +vel[i3+1,i2,i1]+vel[i3-1,i2,i1]))
    return ref
@numba.njit()
def reflectivityDamp(vel,damp):
    ref=vel.copy()
    for i3 in range(1,ref.shape[0]-1):
        for i2 in range(1,ref.shape[1]-1):
            ref[i3,i2,0]=0
            for i1 in range(1,ref.shape[2]-1):
                ref[i3,i2,i1]=(vel[i3,i2,i1]*6-(vel[i3,i2,i1-1]+vel[i3,i2,i1+1]\
                                                   +vel[i3,i2+1,i1]+vel[i3,i2-1,i1]\
                                    +vel[i3+1,i2,i1]+vel[i3-1,i2,i1]))*damp[i3-1,i2,i1]
    return ref
@numba.njit(parallel=True)
def convolve(ref,wavelet):
    out=ref.copy()
    for i3 in numba.prange(ref.shape[0]):
        for i2 in range(ref.shape[1]):
            for i1 in range(ref.shape[2]):
                out[i3,i2,i1]=0
                for it in range(wavelet.shape[0]):
                  val=0
                  if i1 -it >= 0:
                      val=ref[i3,i2,i1-it]
                  out[i3,i2,i1]+=val*wavelet[it]
    return out