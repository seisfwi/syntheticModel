import syntheticModel.syntheticModel as syntheticModel
import math
import syntheticModel.event as event
import math
import numba
import syntheticModel.sincTable as sincTable
import syntheticModel.model_noise as model_noise
import datetime
import numpy as np

from numba import prange

@numba.njit(parallel=True)
   # "void(int32[:,:,:],int64[:,:],List(float64),boolean[:],float64,float64,float32[:,:],float32[:,:],float32[:,:],int32[:,:,:])",parallel=True)
def shiftIntField(fieldIn,ilocs,ds,changed,fill,basement,shift1,shift2,shift3,fieldOut):
    for ipos in numba.prange(shift1.shape[0]):
        i2=ilocs[1,ipos]
        i3=ilocs[0,ipos]
        if changed[ipos]:
            for i1 in range(fieldIn.shape[2]):
                l1=int(i1 - shift1[ipos,i1]/ds[0]+.5)
                l2=max(0,min(fieldIn.shape[1]-1,int(i2-shift2[ipos,i1]/ds[1]+.5)))
                l3=max(0,min(fieldIn.shape[2]-1,int(i3-shift3[ipos,i1]/ds[2]+.5)))
                if l1 < 0:
                    fieldOut[i3,i2,i1]=fill
                elif l1 >= fieldIn.shape[2]:
                    fieldOut[i3,i2,i1]=basement
                else:
                    fieldOut[i3,i2,i1]=fieldIn[l3,l2,l1]
        else:
            fieldOut[i3,i2,:]=fieldIn[i3,i2,:]
            


@numba.njit("void(float64[:,:],float64,float64,float64,float64[:,:])",parallel=True)
def calcAzPerp(loc,cylinder_center_axis_2,cylinder_center_axis_3,fault_azimuth,az_perp):
    if fault_azimuth < 0.: fault_azimuth+360 #Put azimuth between 0-360
    fault_azimuth*=np.pi/180.
    csn=math.cos(fault_azimuth)
    sn=math.sin(fault_azimuth)
                  
    for ipos in prange(loc.shape[1]):
        d1=loc[1,ipos]-cylinder_center_axis_2
        d2=loc[0,ipos]-cylinder_center_axis_3
        az_perp[ipos,0]=csn*d1+sn*d2
        az_perp[ipos,1]=-sn*d1+csn*d2
        if abs(az_perp[ipos,0])<.001:
            az_perp[ipos,0]=.0001

@numba.njit(parallel=True,boundscheck=False)
def calcShifts(iloc,locs,theta0,thetatDie,radius,noisyRadius,distDie,cylinderCenterAxis1,cylinderCenterAxis2,cylinderCenterAxis3,azPerp,os,ds,scalePerp,\
            minAxisDiscretization, cylinderRotationTheta,indicateVal,faultHalf,thetaDie,minToIndicate,fault_azimuth,\
            direction,shiftFieldAxis1,shiftFieldAxis2,shiftFieldAxis3,changedPos,\
              indicateI,indicateF,fake):
    csn=math.cos(fault_azimuth/180*math.pi)
    sn=math.sin(fault_azimuth/180*math.pi)
    first=True
    for ipos in numba.prange(azPerp.shape[0]):
        i3=iloc[0,ipos]
        i2=iloc[1,ipos]
        for i1 in range(shiftFieldAxis1.shape[1]):
            loc_axis_1 = ds[0] * i1 - cylinderCenterAxis1
            thetaOld = math.atan2(loc_axis_1, azPerp[ipos,0]) * 180. / math.pi
            
            thetaOld+=360

            thetaCompare = math.atan(loc_axis_1 / azPerp[ipos,0]) * 180. / math.pi + 360.;
            # True radius of current point
            radius = math.sqrt(azPerp[ipos,0] * azPerp[ipos,0] + loc_axis_1 * loc_axis_1);
            ratioAz = abs(noisyRadius[ipos] - radius) / distDie;
            ratioTheta = abs(thetaOld - theta0) / thetaDie;
            if ratioAz < 1 and ratioTheta < 1:
                scaleAz= math.cos(ratioAz*math.pi/2.)
                scaleTheta = math.cos(ratioTheta*math.pi/2.)
                scaleTot= scaleAz*scaleTheta*math.cos(math.pi/2.*scalePerp[ipos])
            
                #shift in theta
                shiftTheta=cylinderRotationTheta*scaleTot
                thetaNew = thetaOld + shiftTheta;
                
                if direction < 0 and radius < noisyRadius[ipos]:
                    thetaNew = thetaOld - shiftTheta
                if direction > 0 and radius > noisyRadius[ipos]:
                    thetaNew= thetaOld - shiftTheta
                #Check to see if we are at the fault
                if abs(radius- noisyRadius[ipos]) / minAxisDiscretization < .999 and\
                  shiftTheta/cylinderRotationTheta > minToIndicate:
                    
                    indicateI[i3,i2,i1]+=indicateVal
                    indicateF[i3,i2,i1]+=scaleTot*fake
                    test=True
            
                #if abs(radius-noisyRadius[ipos]) < faultHalf:
                #changedPos[ipos]=True
                #scale[ipos,i1]=1.3*scaleTot+(1.-scaleTot)
                
                dPR= radius * math.cos(thetaNew*math.pi/180.)
                dpO= radius * math.cos(thetaOld*math.pi/180.)
                newZ = radius * math.sin(thetaNew*math.pi/180.)+cylinderCenterAxis1
                newX = csn * dPR - sn * azPerp[ipos,1] + cylinderCenterAxis2;
                newY = sn * dPR + csn  * azPerp[ipos,1] + cylinderCenterAxis3;
                oldX = csn * dpO - sn * azPerp[ipos,1] + cylinderCenterAxis2
                oldY = sn * dpO + csn * azPerp[ipos,1] + cylinderCenterAxis3
                shiftFieldAxis1[ipos,i1] = newZ - (os[0]+ds[0]*i1);
                shiftFieldAxis2[ipos,i1]= newX -  locs[1,ipos];
                shiftFieldAxis3[ipos,i1] = newY - locs[0,ipos];
                if abs(shiftFieldAxis1[ipos,i1]) >.01 or abs(shiftFieldAxis2[ipos,i1])>.01\
                  or abs(shiftFieldAxis3[ipos,i1]) >.01:
                    changedPos[ipos]=True

@numba.njit(parallel=True)
#"void(float32[:,:,:],int64[:,:],List(float64),boolean[:],float64,float64,float32[:,:],float32[:,:],float32[:,:],float32[:,:],float32[:,:,:])",parallel=True)
def shiftFloatField(fieldIn,ilocs,ds,changed,fill,basement,shift1,shift2,shift3,syncT,fieldOut):
    for ipos in numba.prange(shift1.shape[0]):
        i2=ilocs[1,ipos]
        i3=ilocs[0,ipos]
        if changed[ipos]:
            for i1 in range(fieldIn.shape[2]):
                if abs(shift1[ipos,i1]) >.01 or abs(shift2[ipos,i1])>.01 or abs(shift3[ipos,i1])>.01:
                    l1=i1-shift1[ipos,i1]/ds[0]
                    l2=i2-shift2[ipos,i1]/ds[1]
                    l3=i3-shift3[ipos,i1]/ds[2]
                    if l1 < 0:
                        fieldOut[i3,i2,i1]=fill
                    elif l1 >= fieldIn.shape[2]:
                        fieldOut[i3,i2,i1]=basement
                    else:
                        il1=int(l1)
                        il2=int(l2)
                        il3=int(l3)
                        is1=max(0,min(syncT.shape[0]-1,int((l1-int(l1))*syncT.shape[0]+.5)))
                        is2=max(0,min(syncT.shape[0]-1,int((l2-int(l2))*syncT.shape[0]+.5)))
                        is3=max(0,min(syncT.shape[0]-1,int((l3-int(l3))*syncT.shape[0]+.5)))

                        sm=0
                        for ic in range(syncT.shape[1]):
                            for ib in range(syncT.shape[1]):
                                for ia in range(syncT.shape[1]):
                                    v=fieldIn[max(0,min(fieldOut.shape[0]-1,il3-3+ic)), \
                                      max(0,min(fieldOut.shape[1]-1,il2-3+ib)),\
                                      max(0,min(fieldOut.shape[2]-1,il1-3+ia))]*\
                                      syncT[is3,ic]*syncT[is2,ib]*syncT[is1,ia]
                                    sm+=v
                        fieldOut[i3,i2,i1]=sm
                else:
                    fieldOut[i3,i2,i1]=fieldIn[i3,i2,i1]
        else:
            fieldOut[i3,i2,:]=fieldIn[i3,i2,:]


class fault(event.event):
    """Default class for faulting"""
    def __init__(self,**kw):
        super().__init__(**kw)

    def applyBase(self,inM:syntheticModel.geoModel,azimuth:float=0.,indicateI:bool=False,\
        indicateF:bool=False,dir:int=1,faultZone:float=0.,indicateValue:float=.05,\
        indicateMark:int=1, seed=None, begx:float=.5, begy:float=.5,begz:float=.5,\
        perp_die:float=.1,dz:float=0.,daz:float=.5,deltaTheta:float=.1,dist_die:float=.2,\
        theta_die:float=5.,theta_shift:float=3.,radius_freq:float=4.,radiusSkew:float=.02,\
            **kw)->syntheticModel.geoModel:
        """
            Fault a model
            
            Arguements:

            azimuth - [0.] Azimuth of fault
            indicateF [False] Whether or not mark faults
            indicateI [False] Whether or not mark faults
            dir - [1] Direction of fault movement
            faultZone [0.] Size of crushed rock zone
            indicateValue [.05] What fraction of fault movement to mark as a fault
            indicateMark [1] What value to use to indicate a fault
            seed - [null] int; random seed to initialize noise functions

            Old method: (overrides new method)
                begx    - [.5] Relative location of the beginning of fault x (fraction 
                of model)
                begy    - [.5] Relative location of the beginning of fault y fraction 
                of model)
                begz    - [.5] Relative location of the beginning of fault z (fraction 
                of model)
                perp_die - [0.1] Dieoff of fault in in perpendicular distance (fraction 
                of model)
                dz      - [0.] Distance away from the center of a circle in z (fraction 
                of model)
                daz     - [.5] Distance away in azimuth (fraction of model)
                deltaTheta - [.1] Dieoff in theta away from the fault
                dist_die - [0.2] Distance dieoff of fault (fraction of model)
                theta_die - [5.] Distance dieoff in theta
                theta_shift - [3.] Shift in theta for fault
                radiusFreq -[4]    Frequency variation
                radiusSkew - [.02] Skew
 
            New method:
                xpos  - Middle axis 2
                ypos  - Middle axis 3
                zpos  - Middle axis 1
                extentInLine - How far away movement occurs inline with the fault
                extentCrossLine - How far movement perpendicular to fault inline 
                position
                radius - Radius of the circle describing motion (larger circle means 
                less fault curvature)
                shift - Maximum shift along the fault
                ruptureLength - Total rupture length of the fault
                angle - Fault angle. Zero would have the rupture perpendicular at fault 
                center

            Returns

                outModel - Returns updated model  
        """
        
        axes = inM.get().getHyper().axes

        if "xpos" in kw:
            begx = (kw["xpos"] - axes[2].o) / (axes[2].d * axes[2].n)
        if "ypos" in kw:
            begy = (kw["ypos"] - axes[1].o) / (axes[1].d * axes[1].n)
        if "zpos" in kw:
            begz = (kw["zpos"] - axes[0].o) / (axes[0].d * axes[0].n)
        if "extentInLine" in kw:
            dist_die = kw["extentInLine"] / max(axes[1].d * axes[1].n,
                                                    axes[2].d * axes[2].n)
        if "extentCrossLine" in kw:
            perp_die = kw["extentCrossLine"] / max(axes[1].d * axes[1].n,
                                                        axes[2].d * axes[2].n)
        if "radius" in kw:
            radius=kw["radius"]
            if not "angle" in kw:
                kw["angle"] = 0
            dz = math.sin(.01745 * kw["angle"]) * kw["radius"] / max(\
                axes[1].d * axes[1].n, axes[2].d * axes[2].n)
            daz = math.cos(.01745 * kw["angle"]) * kw["radius"] / max(\
            axes[1].d * axes[1].n, axes[2].d * axes[2].n)
        else:
            da = max(axes[1].d * axes[1].n, axes[2].d * axes[2].n) * .5
            if "daz" in kw:
                da = max(axes[1].d * axes[1].n, axes[2].d * axes[2].n) * kw["daz"]
            dz = 0
            if "dz" in kw:
                dz = axes[0].d * axes[0].n * kw["dz"]
            radius = math.sqrt(dz * dz + da * da)
        if "ruptureLength" in kw :
            theta_die = kw["ruptureLength"] / kw["radius"] / math.pi * 180.
        
        if theta_die > 20:
                print(\
            "Warning very large theta_die %f. Reduce theta_die or  decrease \
                rupture_length or increase radius."
            % theta_die)
        if "shift" in kw:
            theta_shift = (kw["shift"] / math.pi / radius / 2.) * 180.
            

        axes=inM.getPrimaryProperty().getHyper().axes
        
        exts=[ax.d*ax.n for ax in axes]
        ns=[ax.n for ax in axes]
        os=[ax.o for ax in axes]
        ds=[ax.d for ax in axes]
        
        osN= numba.typed.List()
        dsN= numba.typed.List()
        [dsN.append(x) for x in ds]
        [osN.append(x) for x in os]

        
        
        faultAzimuth=azimuth
        faultBegAxis3=begx*exts[2]
        faultBegAxis2=begy*exts[1]
        faultBegAxis1=begz*exts[0]
        
 
        
        #Distance away along x-y plane
        dz=dz*exts[0]
        daz=daz*max(exts[1],exts[2])
        
        
        radius=math.sqrt(dz*dz+daz*daz)

        #amount cylinder rotates in degrees
        cylinderRotationTheta=theta_shift
        
        #fault die out perp tp cylinder
        perpDie=perp_die*max(exts[1],exts[2])
        
        #fault die out distance along cylinder wall
        distDie=dist_die*max(exts[1],exts[2])
        
        #theta die out 
        thetaDie=theta_die
        
     
        #radius of fault crumble zone
        faultHalf=faultZone
        
        if cylinderRotationTheta >thetaDie/2.:
            raise Exception("A shift more than 1/2 theta_die is unrealistic and unalloawed")
            
        direction=dir
        
        theta0=(math.atan2(dz,daz))*180./math.pi
        theta0+=360

        faultAzimuth=azimuth
        if faultAzimuth<0:
            faultAzimuth+=360 #Put azimuth in 0-360

        
        #fault die out perp tp cylinder
        perpDie=perp_die*max(exts[1],exts[2])
        
        #fault diemuth+=360 #Put azimuth in 0-360
            
        #Cylinderical coordniate 0,0,0 is at
        cylinderCenterAxis1=faultBegAxis1-dz
        cylinderCenterAxis2=faultBegAxis2 -daz * math.cos(faultAzimuth/180*math.pi)
        cylinderCenterAxis3=faultBegAxis3-daz* math.sin(faultAzimuth/180*math.pi)
        
     
        if seed is None:
            seed=int(datetime.datetime.utcnow().timestamp())
        np.random.seed(seed)
        
        indicateVal=indicateValue
        
        tmpI=np.zeros((ns[2],ns[1],ns[0]),dtype=np.int32)
        tmpF=np.zeros((ns[2],ns[1],ns[0]))
        if indicateF:
            indicateFloat=inM.getCreateFloatField("indicator",0.,0.).getNdArray()
            fake=1.
        else:
            fake=0.
            
        minToIndicate=indicateValue
        if indicateI:
            indicateInt=inM.getCreateIntField("indicator",0,0).getNdArray()
            indicateVal=indicateMark
        else:
            indicateVal=0
        
        
            
        minAxisDiscretization=min(ds[0],min(ds[1],ds[2]))
        
        loc1 = np.linspace(os[1], os[1] + ns[1] * ds[1], ns[1], endpoint=False)
        loc2 = np.linspace(os[2], os[2] + ns[2] * ds[2], ns[2], endpoint=False)
        locs = np.stack(np.meshgrid(loc2, loc1, indexing='ij'), -1)
        locs = np.reshape(locs, (-1, 2)).T
        

        indexes = np.stack(np.meshgrid(range(ns[2]), range(ns[1]), indexing='ij'), -1)
        indexes = np.reshape(indexes, (-1, 2)).T

        
        perpScale=max(exts[1],exts[2])
        azPerp=np.zeros((ns[2]*ns[1],2),dtype=np.float64)
        calcAzPerp(locs,cylinderCenterAxis2,cylinderCenterAxis3,faultAzimuth,azPerp) 
        perpS=azPerp[:,1]/perpScale+np.random.rand()*max(ds[1]*ns[1],ds[2]*ns[2])
        noiseyRadius= radius * (1. + radiusSkew * model_noise.fractal(perpS))
        ratioPerp = azPerp[:,1]/perpDie
        ratioAz=  abs(radius  - azPerp[:,0])/distDie
        scalePerp=abs(ratioPerp.clip(-1.00,1.00))
        
        shiftFieldAxis1=np.zeros((ns[2]*ns[1],ns[0]),np.float32)
        shiftFieldAxis2=np.zeros((ns[2]*ns[1],ns[0]),np.float32)
        shiftFieldAxis3=np.zeros((ns[2]*ns[1],ns[0]),np.float32)
        
        junk=np.zeros((ns[2],ns[1],ns[0]))
        changedPos=np.zeros((ns[1]*ns[2]),bool)
        calcShifts(indexes,locs,theta0,thetaDie,radius,noiseyRadius,distDie,cylinderCenterAxis1,cylinderCenterAxis2,\
        cylinderCenterAxis3,azPerp,osN,dsN,scalePerp,minAxisDiscretization, cylinderRotationTheta,\
        indicateVal,faultHalf,thetaDie,minToIndicate,faultAzimuth,direction,shiftFieldAxis1,\
        shiftFieldAxis2,shiftFieldAxis3,changedPos,tmpI,tmpF,fake)
        

        outM=inM.expand(0,0)
        for fld in outM.getIntFieldList():
            inF=inM.getIntField(fld).getNdArray()
            outF=outM.getIntField(fld).getNdArray()
            base=inM.getBasementInt(fld)
            fill=inM.getFillInt(fld)
            shiftIntField(inF,indexes,dsN,changedPos,fill,base,shiftFieldAxis1,shiftFieldAxis2,shiftFieldAxis3,outF)
                                        
        
        sincT=sincTable.table
        for fld in outM.getFloatFieldList():
            inF=inM.getFloatField(fld).getNdArray()
            outF=outM.getFloatField(fld).getNdArray()
            base=inM.getBasementFloat(fld)
            fill=inM.getFillFloat(fld)
            shiftFloatField(inF,indexes,dsN,changedPos,fill,base,shiftFieldAxis1,shiftFieldAxis2,shiftFieldAxis3,\
                            sincT,outF)  
        if indicateI:
            j=outM.getIntField("indicator").getNdArray()
            j+=tmpI
        if indicateF:
            j=outM.getFloatField("indicator").getNdArray()
            j+=tmpF

        return outM


class basic(fault):
    def __init__(self,**kw):
        """
        Basic  erode river, for now we have not specialized
        
        """
        super().__init__(**kw)