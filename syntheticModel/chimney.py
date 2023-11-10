from os import defpath
import syntheticModel.sincTable as sincTable
import syntheticModel.simplexNoise as simplexNoise
import numba
import math
import numpy as np
import random
import argparse
import syntheticModel.syntheticModel as syntheticModel
import datetime
import syntheticModel.event as event
import syntheticModel.structuralSmoothing as structuralSmoothing
@numba.njit(parallel=True)
def calcShiftDamp(iZeroBeg: int,iFullBeg: int ,iFullEnd: int,iZeroEnd:int,\
    centerX,centerY,halfWidthA,uplift,dampV,pockDepth,ns,ds):
    """
    Calculate the shift and cut fields for creating chimneys
     
     iZeroBeg,iFullBeg, iFullEnd, iZeroEnd - The chimney begining and end along with zero to 1
     centerX,centerY - Location of the center of the chimney
     halfWidthA - Width of chimney as a function of depth and angle
     uplift - Maximum amount to uplift
     dampV - How much to damp amplitudes below the chimney
     pockDepth - Depth for pockmark
     ns,ds - Number of samples and sampling of 3-d cubes
    
    """




    shift=np.zeros((ns[2],ns[1],ns[0]),dtype=np.float32)
    dampT=np.zeros((ns[2],ns[1],ns[0]),dtype=np.float32)
    dampT.fill(1.)
    dPctZ=1./float(ns[0])
    dSamp=2*math.pi/180 #sampling in angle table
    for i3 in numba.prange(ns[2]):
        y=i3*ds[2]
        for i2 in range(ns[1]):
            x=i2*ds[1]
            found=False
            for i1 in range(ns[0]):
                z=i1*ds[0]
                if z< pockDepth*3:
                    dist2=x-centerX[iZeroBeg]
                    dist3=y-centerY[iZeroBeg]
                    dist=math.sqrt(dist2*dist2+dist3*dist3)
                    ang=math.atan2(dist2,dist3)
                    fXNoise=(math.cos(ang)+math.pi)/dSamp
                    fYNoise=(math.sin(ang)+math.pi)/dSamp
                    iXNoise=int(fXNoise+.5)
                    iYNoise=int(fYNoise+.5)
                    haltWidthAtLocation= halfWidthA[iYNoise,iXNoise,iZeroBeg]*1.2
                    if dist<=haltWidthAtLocation: #we are inside
                        fr=dist/haltWidthAtLocation
                        zscale=1
                        if z> pockDepth:
                            zscale=1.-(z-pockDepth)/(2*pockDepth)
                        shift[i3,i2,i1]=-pockDepth*math.cos(fr*math.pi/2.)*\
                        zscale/ds[0]
            
 

                else:
                    pct=i1*dPctZ
                    zscale=1.
                    if i1 < iZeroBeg or i1 > iZeroEnd:
                        zscale=0
                    elif i1 >= iZeroBeg and i1 < iFullBeg:
                        zscale=float(i1-iZeroBeg)/float(iFullBeg-iZeroBeg)
                    elif i1 >iFullEnd and i1 <= iZeroEnd:
                        zscale=float(iZeroEnd-i1)/float(iZeroEnd-iFullEnd)
                    
                    dist2=x-centerX[i1]
                    dist3=y-centerY[i1]

                    dist=math.sqrt(dist2*dist2+dist3*dist3)
                    ang=math.atan2(dist2,dist3)
                    fXNoise=(math.cos(ang)+math.pi)/dSamp
                    fYNoise=(math.sin(ang)+math.pi)/dSamp
                    iXNoise=int(fXNoise)
                    iYNoise=int(fYNoise)
                    fXNoise-=iXNoise
                    fYNoise-=iYNoise
                    halfAtLocation=(1.-fXNoise)*(1.-fYNoise)*halfWidthA[iYNoise,iXNoise,i1]+\
                    (fXNoise)*(1.-fYNoise)*halfWidthA[iYNoise,iXNoise+1,i1]+\
                    (1.-fXNoise)*(fYNoise)*halfWidthA[iYNoise+1,iXNoise,i1]+\
                    (fXNoise)*(fYNoise)*halfWidthA[iYNoise+1,iXNoise+1,i1]

                    if dist <= halfAtLocation: #we are inside
                        fr=dist/halfAtLocation
                        found=True
                        shift[i3,i2,i1]=uplift*math.cos(fr*math.pi/2.)*\
                        math.cos(fr*math.pi/2.)*zscale/ds[0]

                        dampT[i3,i2,i1]=dampV
                
    ic2=centerX[iZeroBeg]/ds[1]
    ic3=centerY[iZeroBeg]/ds[2]
    return shift,dampT

@numba.njit(parallel=True)
def multDamp(mat1,mat2):
    """Simple function to multiply to damping fields"""
    for i3 in numba.prange(mat1.shape[0]):
        for i2 in range(mat1.shape[1]):
            for i1 in range(mat1.shape[2]):
                mat1[i3,i2,i1]*=mat2[i3,i2,i1]
                
               

@numba.njit(parallel=True)
def applyShiftInt(fldIn,fldOut,shift,fill):
    """
    Apply the shift field to an integer array
    
    fldIn, fldOut - Input and output arrays
    shift -SHift fields
    fill - What to fill pock marked array values

    """
    for i3 in numba.prange(fldIn.shape[0]):
        for i2 in range(fldIn.shape[1]):
            for i1 in range(fldIn.shape[2]):
                iloc=int(shift[i3,i2,i1])
                if iloc+i1 <0:
                    fldOut[i3,i2,i1]=fill

                else:
                    fldOut[i3,i2,i1]=fldIn[i3,i2,i1+iloc]


                        
@numba.njit(parallel=True)
def fixHalfWidth(widthIn,mean,stdDev,grad,minV):
    """
    Make the halfWidth change as a function of depth
    
    widthIn - Input width field
    mean - Average width
    stdDev - Standard deviation for the width
    grad - Gradient to apply to the width
    minV - Minimum width
    
    """
    widthOut=np.copy(widthIn)
    for i3 in numba.prange(widthOut.shape[0]):
        for i2 in range(widthOut.shape[1]):
            for i1 in range(widthOut.shape[2]):
                old=(widthIn[i3,i2,i1]-mean)/stdDev
                widthOut[i3,i2,i1]=max(minV,widthIn[i3,i2,i1]+grad*i1)

    return widthOut  

@numba.njit(parallel=True)
def applyShiftFloat(fldIn,fldOut,shift,sincTab,fill):
    """
    Apply shift field to float array

    fldIn, fldOut - Float arrays
    shift - AMount toshift
    sincTab - A tabulated sinc function
    fill - Value to fill in pockmarked values
    """
    for i3 in numba.prange(fldIn.shape[0]):
        for i2 in range(fldIn.shape[1]):
            tmp=np.zeros((fldIn.shape[2],),dtype=np.float32)
            for i1 in range(4):
                tmp[i1]=fldIn[i3,i2,0]
                tmp[i1+4+fldIn.shape[2]]=fldIn[i3,i2,fldIn.shape[2]-1]
            for i1 in range(fldIn.shape[2]):
                tmp[i1+4]=fldIn[i3,i2,i1]
            ia=0
            while  abs(tmp[ia]-fill)<.001:
                ia+=1
            tmp2=tmp.copy()
            tmp2[ia-4:ia]=tmp[ia]


            for i1 in range(fldIn.shape[2]):
                fract=shift[i3,i2,i1]
                iloc=int(fract)
                fract-=iloc
                itab=int(fract*10000)
                if iloc+i1 < 0 or abs(tmp2[iloc+i1]-fill)<.001:
                    fldOut[i3,i2,i1]=fill

                else:
                    fldOut[i3,i2,i1]=0
                    for ifilt in range(8):
                        fldOut[i3,i2,i1]+=sincTab[itab,ifilt]*tmp[iloc+ifilt+i1]
 



class chimney(event.event):
    """Parent class to make a gas chimney"""

    def __init__(self,**kw):
        """Root class for making a chimney. Call with care, use sub-classes"""
        super().__init__(**kw)

    def applyBase(self,modIn:syntheticModel.geoModel,center2:float=.5 ,center3:float=.5, \
                width:float=.02, pockDepth:float=.0, seed=None,\
                devEdge:float=0., devCenter:float=0, uplift:float=.07,zeroBeg:float=.02,\
                fullBeg:float=.04, fullEnd:float=.88, zeroEnd:float=.9,dampV:float=.9,\
                minWidth:float=.015, gradWidth:float=.1, applySmoothing:bool=False, filterLen:int=3,\
                smoothLen:int=60)->syntheticModel.geoModel:
        """
        Create a gas chimney

            modIn - Input model
            center2 - Center of chimney as fraction of axis 2
            center3 - Center of chimney as fraction of axis 3
            width - Width at top of chimn
            pockDepth - Depth of pock mark (fraction of axis 1)
            devEdge   -Deviation from vertical well width
            devCenter - Deviation of the the center of the well
            uplift - plift for chimney, ratio of the first axis 
            zeroBeg - Top of chimney, no movement
            fullBeg - Top of chimney, full movement
            fullEnd - Bottom of chimney, fill movement
            zeroENd - Bottom of chimney, no movement
            dampV  - Damping of amplitude 
            minWidth - Minimum width of well
            gradWidth - radient increase of chimney increase
            applySmoothing- apply structural smoothing"
            smoothLen -Default structural smoothing length
            filterLen - Length of filter for capturing local covariance
            seed  - [None] Random seed 

            Return

                syntheticModel.model object
            
        """
        axes=modIn.getPrimaryProperty().getHyper().axes
        ns=[ax.n for ax in axes]
        ds=[ax.d for ax in axes]
        nsN = numba.typed.List()
        [nsN.append(x) for x in ns]
        dsN =numba.typed.List()
        [dsN.append(x) for x in ds]
        exts=[ax.d*ax.n for ax in axes]
        #convert parameters to locations
        center2=center2*exts[1]
        center3=center3*exts[2]
        halfWidth=width*max(exts[1],exts[2])/2.
        minWidth=minWidth*max(exts[1],exts[2])/2
        gradWidth=gradWidth/ns[0]*ds[0]
        iZeroBeg=int(zeroBeg*exts[0]/axes[0].d)
        iFullBeg=int(fullBeg*exts[0]/axes[0].d)
        iFullEnd=int(fullEnd*exts[0]/axes[0].d)
        iZeroEnd=int(zeroEnd*exts[0]/axes[0].d)
        puckDepth=pockDepth*exts[0]
        uplift=uplift*exts[0]
        if iZeroBeg < 0: raise Exception("Illegal zeroBeg must be greater than 0")
        if iFullBeg < iZeroBeg: raise Exception("Illegal chimnet description iFullBeg<iZeroBeg")
        if iFullEnd < iFullBeg: raise Exception("Illegal chiment description iFullEnd<iFullBeg")
        if iZeroEnd < iFullEnd or iZeroEnd >= axes[0].n:
            raise Exception("Invalid chimney descrption zeroEnd < fullEnd or zeroEnd>=1.")

            
        if seed is None:
            seed =int(datetime.datetime.utcnow().timestamp())
        
        random.seed(seed)
        np.random.seed(seed)
            
        noiseCenter=simplexNoise.simplexNoise(random.randint(0,1000*1000*1000))
        noiseEdge=simplexNoise.simplexNoise(random.randint(0,1000*1000*1000))       

        pi=math.atan(1.)*2
        ang1L=np.linspace(-pi,pi+.1,185,dtype=np.float32)
        ang2L=np.linspace(-pi,pi+.1,185,dtype=np.float32)
        zL=np.linspace(0.,9.,ns[0],dtype=np.float32)

        centerX=noiseCenter.noise2D(zL,zL[:10],amplitude=devCenter/1.875,n_octaves=4)[0,:]+center2
        centerY=noiseCenter.noise2D(zL,zL[:10],amplitude=devCenter/1.875,n_octaves=4)[0,:]+center3

        halfWidthA=halfWidth+noiseEdge.noise3D(zL,ang2L,ang1L,amplitude=devEdge/1.875)

        halfWidthA=fixHalfWidth(halfWidthA,halfWidth,devEdge,gradWidth,minWidth)
        del ang1L
        del ang2L
 
        shift,dampT=calcShiftDamp(iZeroBeg,iFullBeg,iFullEnd,iZeroEnd,centerX,
                                    centerY,halfWidthA,uplift,dampV,puckDepth,nsN,dsN)
        
        
        del centerX
        del centerY
        del halfWidthA
        modIn.getCreateFloatField("damp",1.,1.)
        modOut=modIn.expand(0,0)

                    
        sinc=sincTable.table
        
    
        for fld in modOut.getFloatFieldList():
            fill=modOut.getFillFloat(fld)
            applyShiftFloat(modIn.getFloatField(fld).getNdArray(),\
            modOut.getFloatField(fld).getNdArray(),shift,sinc,fill)
        for fld in modOut.getIntFieldList():
            fill=modOut.getFillInt(fld)
            applyShiftInt(modIn.getIntField(fld).getNdArray(),\
                modOut.getIntField(fld).getNdArray(),shift,fill)
            
        if applySmoothing:
            dist=width+gradWidth*ds[0]*ns[0]
            b2=max(0,int((center2-dist)/ds[1]-int(smoothLen)))
            e2=min(ns[1],int((center2+dist)/ds[1]+int(smoothLen)))
            b3=max(0,int((center3-dist)/ds[2]-int(smoothLen)))
            e3=min(ns[2],int((center3+dist)/ds[2]+int(smoothLen)))

            vS=modOut.getPrimaryProperty().getNdArray()[b3:e3,b2:e2,:]
            dS=dampT[b3:e3,b2:e2,:]
            d2=structuralSmoothing.smoother(vS,mod=dS,hw=filterLen,hw2=smoothLen)
            dampT[b3:e3,b2:e2,:]=d2
            print(b2,e2,b3,e3)
            multDamp(modOut.getFloatField("damp").getNdArray(),dampT)
        else:
            multDamp(modOut.getFloatField("damp").getNdArray(),dampT)
        return modOut


class big(chimney):
    """Create a large chimney"""
    def __init__(self,center2:float=.5,center3:float=.5, depth:float=.2,width=.02,pockMark:bool=True,\
            uplift:float=.07):
        """center2, center3  - Location in X,Y space (0-1.)
            width - Width of chimney
            pockMark - Whether or not to create a pockmark
            uplift - Amount to uplify  (fraction of first axis
            depth - Begining depth of chimney"""
        
        kws=locals()
        if pockMark:
            kws["pockDepth"]=.015
        else:
            kws["pockDepth"]=0
        
        kws["devEdge"]=width/1.5
        kws["devCenter"]=width/2

        del kws["pockMark"]
        
        if depth<0.02 and depth > .7:
            print("Depth must be between .02 and .7")
        
        #Assume the basement takes .15 
        len=.82-depth
        kws["zeroBeg"]=depth 
        kws["fullBeg"]=depth+.1*len
        kws["fullEnd"]=depth+.8*len
        kws["zeroEnd"]=.82
        del kws["self"]
        del kws["depth"]
        del kws["__class__"]
        kws["minWidth"]=width*.76
        kws["filterLen"]=3
        kws["smoothLen"]=60
        super().__init__(**kws)


    

class small(chimney):
    """Create a large chimney"""
    def __init__(self,center2:float=.5,center3:float=.5, depth:float=.4,width=.008,pockMark:bool=True,\
            uplift:float=.07):
        """center2, center3  - Location in X,Y space (0-1.)
            width - Width of chimney
            pockMark - Whether or not to create a pockmark
            uplift - Amount to uplify  (fraction of first axis
            depth - Begining depth of chimney"""
        
        kws=locals()
        if pockMark:
            kws["pockDepth"]=.015
        else:
            kws["pockDepth"]=0
        
        kws["devEdge"]=width/1.5
        kws["devCenter"]=width/2

        del kws["pockMark"]
        
        if depth<0.02 and depth > .7:
            print("Depth must be between .02 and .7")
        
        #Assume the basement takes .15 
        len=.82-depth
        kws["zeroBeg"]=depth 
        kws["fullBeg"]=depth+.1*len
        kws["fullEnd"]=depth+.8*len
        kws["zeroEnd"]=.82
        del kws["self"]
        del kws["depth"]
        del kws["__class__"]
        kws["minWidth"]=width*.76
        kws["filterLen"]=10
        kws["smoothLen"]=24

        super().__init__(**kws)



        
