import numba
import numpy as np
import scipy
import math

@numba.njit(parallel=True)
def mkToepTable(ntab,lsinc):
    pi = math.pi
    pi2 = pi * 2;
    snyq = 0.5;
    snat = snyq * (0.066 + 0.265 * math.log(float(lsinc)));
    s0 = 0.0;
    ds = (snat - s0) / (float(lsinc) * 2 - 1);

    b=np.zeros((ntab,lsinc))
    c=np.zeros((ntab,lsinc))

    duse=1./float(ntab)
    for it in range(ntab):
        eta = lsinc / 2 - 1.0 + duse*it;
        for j in range(lsinc):
            s=s0
            while s < snat:
                b[it,j]+=math.cos(pi2*s*j)
                c[it,j]+=math.cos(pi2*s*(eta-j))
                s=s+ds                
    return b,c

def sincTable(ntab:int=10000,nsinc:int=8):
    table=np.zeros((ntab,nsinc))
    b,c= mkToepTable(ntab,nsinc)
    for itab in range(ntab):
        table[itab,:]=scipy.linalg.solve_toeplitz(b[itab,:],c[itab,:])
    return table

table=sincTable()