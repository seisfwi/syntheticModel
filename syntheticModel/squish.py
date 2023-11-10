import syntheticModel.event as event
import syntheticModel.sincTable as sincTable
import syntheticModel.syntheticModel as syntheticModel
from syntheticModel.model_noise import fractal
from numba import njit, prange, float64
import numpy as np
import datetime
import math

RAND_INT_LIMIT = 100000
FREQUENCY_LEVELS = [3, 6, 12, 24]
AMPLITUDE_LEVELS = [1, 0.5, 0.125, 0.0625]
# @njit(float64[:,:](float,float64[:,:],int,int),parallel=True)
@njit(parallel=True)
def rotateField(azimuth: float, field: np.ndarray, n1: int, n2: int):
    cIn = int(field.shape[0] / 2.0)
    cOut1 = int(n1 / 2)
    cOut2 = int(n2 / 2)
    rot = azimuth / 180 * math.pi
    out = np.zeros((n2, n1))
    for i2 in prange(n2):
        f2 = i2 - cOut2
        for i1 in range(n1):
            f1 = i1 - cOut1
            b1 = math.cos(rot) * f1 - f2 * math.sin(rot) + cIn
            b2 = math.cos(rot) * f2 + f1 * math.sin(rot) + cIn
            il1 = int(b1)
            il2 = int(b2)
            b1 -= il1
            b2 -= il2

            out[i2, i1] = (
                (1.0 - b1) * (1.0 - b2) * field[il2, il1]
                + (b1) * (1.0 - b2) * field[il2, il1 + 1]
                + (1.0 - b1) * (b2) * field[il2 + 1, il1]
                + (b1) * (b2) * field[il2 + 1, il1 + 1]
            )

            # out[i2,i1]=field[il2,il1]
    return out


def getNoiseField(
    n1, n2, maxS, widthInLine, widthCrossLine, azimuth, noise_levels, begFreq
):

    nmax = int(max(n1, n2) * 1.8)
    # randomly shift origin
    o1 = np.random.rand() * nmax
    o2 = np.random.rand() * nmax
    p1 = np.linspace(o1, o1 + nmax * widthInLine, nmax, endpoint=False)
    p2 = np.linspace(o2, o2 + nmax * widthCrossLine, nmax, endpoint=False)
    p = np.stack(np.meshgrid(p1, p2, indexing="ij"), -1)
    p = np.reshape(p, (-1, 2)).T

    # get freq and amplitude values for perlin octaves
    freq = [begFreq]
    for i in range(3):
        freq.append(freq[-1] * 2)

    freq_levels = list(np.array(freq[:noise_levels]) / (n1))
    amp_levels = AMPLITUDE_LEVELS[:noise_levels]
    field = fractal(
        p, frequency=freq_levels, n_octaves=noise_levels, amplitude=amp_levels
    )
    field = np.reshape(field, (nmax, nmax))

    # normalize between -1 and 1
    mx = np.max(field)
    mn = np.min(field)
    field = (field + mn) / (mx - mn) * maxS
    m = rotateField(azimuth, field, n1, n2)
    return m


@njit(parallel=True)
def shiftFieldF(
    maxS: int,
    sincT: np.ndarray,
    outF: np.ndarray,
    inF: np.ndarray,
    shiftF: np.ndarray,
    fill: float,
    basement: float,
):
    for i3 in prange(outF.shape[0]):
        for i2 in prange(outF.shape[1]):
            useF = maxS - shiftF[i3, i2]
            imin = math.ceil(useF)
            itab = min(sincT.shape[0] - 1, int((imin - useF) * sincT.shape[0] + 0.5))
            for i1 in range(outF.shape[2]):
                # outF[i3,i2,i1]=inF[i3,i2,max]
                outF[i3, i2, i1] = 0
                for isinc in range(sincT.shape[1]):
                    il = min(inF.shape[2] - 1, max(0, i1 - 3 + isinc - imin))
                    outF[i3, i2, i1] += inF[i3, i2, il] * sincT[itab, isinc]


@njit(parallel=True)
def shiftFieldI(
    maxS: int,
    outF: np.ndarray,
    inF: np.ndarray,
    shiftF: np.ndarray,
    fill: float,
    basement: float,
):
    for i3 in prange(outF.shape[0]):
        for i2 in prange(outF.shape[1]):
            imin = int(maxS - shiftF[i3, i2] + 0.5)
            for i1 in range(imin):
                outF[i3, i2, i1] = fill
            for i1 in range(inF.shape[2]):
                outF[i3, i2, i1 + imin] = inF[i3, i2, i1]
            for i1 in range(inF.shape[2] + imin, outF.shape[2]):
                outF[i3, i2, i1] = basement


class squish(event.event):
    """Default class for squishing"""

    def __init__(self, **kw):
        super().__init__(**kw)

    def applyBase(
        self,
        inM: syntheticModel.geoModel,
        azimuth: float = 0.0,
        widthInLine: float = 1000.0,
        widthCrossLine: float = 100.0,
        maxShift: float = 50,
        seed=None,
        begFreq: float = 6.0,
        nfreq=3,
    ) -> syntheticModel.geoModel:
        """

        Squish a model

        Arguements:

        inM - Input model
        azimuth - Azimuth for squishing
        widthInLine - Approximate width for random patterns, axis 1
        widthCrosssLine - Approximate width for random patterns, axis 2
        maxShift -  Maximum shift in z
        seed - [null] int; random seed to initialize noise functions
        nfreq -3 Number of frequencies for noise (more means higher frequency bumpiness)
        begFreq - 3. Basic frequency level for noise (lower means lowe spatial frequency)

        Returns

            outModel - Returns updated model
        """
        if seed is None:
            seed = int(datetime.datetime.utcnow().timestamp())
        np.random.seed(seed)

        axes = inM.getPrimaryProperty().getHyper().axes

        # calculate the maximum shift and then add cells to the top
        maxS = math.ceil(maxShift / axes[0].d)
        outM = inM.expand(maxS, 0)

        sincT = sincTable.table

        shifts = getNoiseField(
            axes[1].n,
            axes[2].n,
            maxS,
            widthInLine / axes[1].n / axes[1].d,
            widthCrossLine / axes[2].d / axes[2].n,
            azimuth,
            nfreq,
            begFreq,
        )
        shifts -= shifts.min()
        self.shifts = shifts

        for fld in outM.getFloatFieldList():
            inp = inM.getFloatField(fld).getNdArray()
            base = inM.getBasementFloat(fld)
            outp = outM.getFloatField(fld).getNdArray()

            shiftFieldF(maxS, sincT, outp, inp, shifts, inM.getFillFloat(fld), base)

        for fld in outM.getIntFieldList():
            inp = inM.getIntField(fld).getNdArray()
            base = inM.getBasementInt(fld)

            shiftFieldI(
                maxS,
                outM.getIntField(fld).getNdArray(),
                inp,
                shifts,
                base,
                inM.getFillInt(fld),
            )
        return outM


class basic(squish):
    def __init__(self, **kw):
        """
        Basic squish, for now we have not specialized

        """
        super().__init__(**kw)
