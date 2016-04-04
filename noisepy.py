# -*- coding: utf-8 -*-
"""
A python module for ambient seismic noise interferometry, receiver function analysis and Surface Wave Tomography.

:Methods:
    aftan analysis (compiled from aftan-1.1)
    SNR analysis based on aftan results
    C3(Correlation of coda of Cross-Correlation) computation
    Generate Predicted Phase Velocity Curves for an array
    Generate Input for Barmin's Surface Wave Tomography Code
    Automatic Receiver Function Analysis( Iterative Deconvolution and Harmonic Stripping )
    Eikonal Tomography
    Helmholtz Tomography (To be added soon)
    Bayesian Monte Carlo Inversion of Surface Wave and Receiver Function datasets (To be added soon)
    Stacking/Rotation for Cross-Correlation Results from SEED2CORpp
    
:Dependencies:
    numpy 1.9.1
    matplotlib 1.4.3
    numexpr 2.3.1
    ObsPy 0.10.2
    pyfftw3 0.2.1
    pyaftan( compiled from aftan 1.1 )
    GMT 5.x.x (For Eikonal/Helmholtz Tomography)
    CURefPy ( A submodule for noisepy, designed for automatic receiver function analysis, by Lili Feng)
    
:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""

# import obspy.sac.sacio as sacio
import obspy.geodetics as obsGeo
import obspy.taup.taup
import obspy
import pyaftan as ftan  # Comment this line if you do not have pyaftan
import numpy as np
import glob, os
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.pylab as plb
import copy
import scipy.signal
import numexpr as npr
from functools import partial
import multiprocessing as mp
import math
try:
    import fftw3 # pyfftw3-0.2
    useFFTW=True;
except:
    useFFTW=False;
import time
import shutil
# import CURefPy as ref # Comment this line if you do not have CURefPy
from subprocess import call
from mpl_toolkits.basemap import Basemap
import warnings

#LDPATH = os.environ['LD_LIBRARY_PATH']
#sys.path.append(LDPATH)

class ftanParam(object):
    """
    Basic FTAN parameters:
    nfout1_1 - output number of frequencies for arr1, (integer*4)
    arr1_1   - preliminary results.
              Description: real*8 arr1(8,n), n >= nfin)
              arr1(1,:) -  central periods, s
              arr1(2,:) -  observed periods, s
              arr1(3,:) -  group velocities, km/s
              arr1(4,:) -  phase velocities, km/s or phase if nphpr=0, rad
              arr1(5,:) -  amplitudes, Db
              arr1(6,:) -  discrimination function
              arr1(7,:) -  signal/noise ratio, Db
              arr1(8,:) -  maximum half width, s
              arr1(9,:) -  amplitudes, nm/m
    arr2_1   - final results
    nfout2_1 - output number of frequencies for arr2, (integer*4)
              Description: real*8 arr2(7,n), n >= nfin)
              If nfout2 == 0, no final result.
              arr2(1,:) -  central periods, s
              arr2(2,:) -  observed periods, s
              arr2(3,:) -  group velocities, km/sor phase if nphpr=0, rad
              arr2(4,:) -  phase velocities, km/s
              arr2(5,:) -  amplitudes, Db
              arr2(6,:) -  signal/noise ratio, Db
              arr2(7,:) -  maximum half width, s
              arr2(8,:) -  amplitudes, nm/m
    tamp_1      -  time to the beginning of ampo table, s (real*8)
    nrow_1      -  number of rows in array ampo, (integer*4)
    ncol_1      -  number of columns in array ampo, (integer*4)
    amp_1       -  Ftan amplitude array, Db, (real*8)
    ierr_1   - completion status, =0 - O.K.,           (integer*4)
                                 =1 - some problems occures
                                 =2 - no final results
    ==========================================================
    Phase-Matched-Filtered FTAN parameters:
    nfout1_2 - output number of frequencies for arr1, (integer*4)
    arr1_2   - preliminary results.
             Description: real*8 arr1(8,n), n >= nfin)
             arr1(1,:) -  central periods, s (real*8)
             arr1(2,:) -  apparent periods, s (real*8)
             arr1(3,:) -  group velocities, km/s (real*8)
             arr1(4,:) -  phase velocities, km/s (real*8)
             arr1(5,:) -  amplitudes, Db (real*8)
             arr1(6,:) -  discrimination function, (real*8)
             arr1(7,:) -  signal/noise ratio, Db (real*8)
             arr1(8,:) -  maximum half width, s (real*8)
             arr1(9,:) -  amplitudes, nm/m
    arr2_2   - final results
    nfout2_2 - output number of frequencies for arr2, (integer*4)
             Description: real*8 arr2(7,n), n >= nfin)
             If nfout2 == 0, no final results.
             arr2(1,:) -  central periods, s (real*8)
             arr2(2,:) -  apparent periods, s (real*8)
             arr2(3,:) -  group velocities, km/s (real*8)
             arr1(4,:) -  phase velocities, km/s (real*8)
             arr2(5,:) -  amplitudes, Db (real*8)
             arr2(6,:) -  signal/noise ratio, Db (real*8)
             arr2(7,:) -  maximum half width, s (real*8)
             arr2(8,:) -  amplitudes, nm/m
    tamp_2      -  time to the beginning of ampo table, s (real*8)
    nrow_2      -  number of rows in array ampo, (integer*4)
    ncol_2      -  number of columns in array ampo, (integer*4)
    amp_2       -  Ftan amplitude array, Db, (real*8)
    ierr_2   - completion status, =0 - O.K.,           (integer*4)
                                =1 - some problems occures
                                =2 - no final results
    """
    def __init__(self):
        # Parameters for first iteration
        self.nfout1_1=0
        self.arr1_1=np.array([])
        self.nfout2_1=0
        self.arr2_1=np.array([])
        self.tamp_1=0.
        self.nrow_1=0
        self.ncol_1=0
        self.ampo_1=np.array([],dtype='float32')
        self.ierr_1=0
        # Parameters for second iteration
        self.nfout1_2=0
        self.arr1_2=np.array([])
        self.nfout2_2=0
        self.arr2_2=np.array([])
        self.tamp_2=0.
        self.nrow_2=0
        self.ncol_2=0
        self.ampo_2=np.array([])
        self.ierr_2=0
        # Flag for existence of predicted phase dispersion curve
        self.preflag=False

    def writeDISP(self, fnamePR):
        """
        Write FTAN parameters to DISP files given a prefix.
        fnamePR: file name prefix
        _1_DISP.0: arr1_1
        _1_DISP.1: arr2_1
        _2_DISP.0: arr1_2
        _2_DISP.1: arr2_2
        """
        if self.nfout1_1!=0:
            f10=fnamePR+'_1_DISP.0';
            
            # Lf10=self.nfout1_1;
            # outArrf10=np.arange(Lf10);
            # for i in np.arange(7):
            #     outArrf10=np.append(outArrf10, self.arr1_1[i,:Lf10]);
            # outArrf10=outArrf10.reshape((8,Lf10));
            # outArrf10=outArrf10.T;
            # np.savetxt(f10+'new', outArrf10, fmt='%4d %10.4lf %10.4lf %12.4lf %12.4lf %12.4lf %12.4lf %8.3lf');
            
            f=open(f10,'w')
            for i in np.arange(self.nfout1_1):
                tempstr='%4d %10.4lf %10.4lf %12.4lf %12.4lf %12.4lf %12.4lf %8.3lf  \n' %( i, self.arr1_1[0,i] , self.arr1_1[1,i] , self.arr1_1[2,i] , self.arr1_1[3,i]  \
                    , self.arr1_1[4,i] , self.arr1_1[5,i] , self.arr1_1[6,i] )
                f.writelines(tempstr)
            f.close()
        if self.nfout2_1!=0:
            f11=fnamePR+'_1_DISP.1'
            
            # Lf11=self.nfout2_1;
            # outArrf11=np.arange(Lf11);
            # for i in np.arange(6):
            #     outArrf11=np.append(outArrf11, self.arr2_1[i,:Lf11]);
            # outArrf11=outArrf11.reshape((7,Lf11));
            # outArrf11=outArrf11.T;
            # np.savetxt(f11+'new', outArrf11, fmt='%4d %10.4lf %10.4lf %12.4lf %12.4lf %12.4lf %8.3lf');
            
            f=open(f11,'w')
            for i in np.arange(self.nfout2_1):
                tempstr='%4d %10.4lf %10.4lf %12.4lf %12.4lf %12.4lf %8.3lf  \n' %( i, self.arr2_1[0,i], self.arr2_1[1,i] , self.arr2_1[2,i] , self.arr2_1[3,i]  \
                    , self.arr2_1[4,i] , self.arr2_1[5,i]  )
                f.writelines(tempstr)
            f.close()
        if self.nfout1_2!=0:
            f20=fnamePR+'_2_DISP.0';
            
            # Lf20=self.nfout1_2;
            # outArrf20=np.arange(Lf20);
            # for i in np.arange(7):
            #     outArrf20=np.append(outArrf20, self.arr1_2[i,:Lf20]);
            # outArrf20=outArrf20.reshape((8,Lf20));
            # outArrf20=outArrf20.T;
            # np.savetxt(f20+'new', outArrf20, fmt='%4d %10.4lf %10.4lf %12.4lf %12.4lf %12.4lf %12.4lf %8.3lf');
            
            f=open(f20,'w')
            for i in np.arange(self.nfout1_2):
                tempstr='%4d %10.4lf %10.4lf %12.4lf %12.4lf %12.4lf %12.4lf %8.3lf \n' %( i, self.arr1_2[0,i], self.arr1_2[1,i] , self.arr1_2[2,i] , self.arr1_2[3,i]  \
                    , self.arr1_2[4,i] , self.arr1_2[5,i] , self.arr1_2[6,i] )
                f.writelines(tempstr)
            f.close()
        if self.nfout2_2!=0:
            f21=fnamePR+'_2_DISP.1';
            
            # Lf21=self.nfout2_2;
            # outArrf21=np.arange(Lf21);
            # for i in np.arange(6):
            #     outArrf21=np.append(outArrf21, self.arr2_2[i,:Lf21]);
            # outArrf21=outArrf21.reshape((7,Lf21));
            # outArrf21=outArrf21.T;
            # np.savetxt(f21+'new', outArrf21, fmt='%4d %10.4lf %10.4lf %12.4lf %12.4lf %12.4lf %8.3lf');
            
            f=open(f21,'w')
            for i in np.arange(self.nfout2_2):
                tempstr='%4d %10.4lf %10.4lf %12.4lf %12.4lf %12.4lf %8.3lf  \n' %( i, self.arr2_2[0,i], self.arr2_2[1,i] , self.arr2_2[2,i] , self.arr2_2[3,i]  \
                    , self.arr2_2[4,i] , self.arr2_2[5,i]  )
                f.writelines(tempstr)
            f.close()
        return

    def FTANcomp(self, inftanparam, compflag=4):
        """
        Compare aftan results for two ftanParam objects.
        """
        fparam1=self
        fparam2=inftanparam
        if compflag==1:
            obper1=fparam1.arr1_1[1,:fparam1.nfout1_1]
            gvel1=fparam1.arr1_1[2,:fparam1.nfout1_1]
            phvel1=fparam1.arr1_1[3,:fparam1.nfout1_1]
            obper2=fparam2.arr1_1[1,:fparam2.nfout1_1]
            gvel2=fparam2.arr1_1[2,:fparam2.nfout1_1]
            phvel2=fparam2.arr1_1[3,:fparam2.nfout1_1]
        elif compflag==2:
            obper1=fparam1.arr2_1[1,:fparam1.nfout2_1]
            gvel1=fparam1.arr2_1[2,:fparam1.nfout2_1]
            phvel1=fparam1.arr2_1[3,:fparam1.nfout2_1]
            obper2=fparam2.arr2_1[1,:fparam2.nfout2_1]
            gvel2=fparam2.arr2_1[2,:fparam2.nfout2_1]
            phvel2=fparam2.arr2_1[3,:fparam2.nfout2_1]
        elif compflag==3:
            obper1=fparam1.arr1_2[1,:fparam1.nfout1_2]
            gvel1=fparam1.arr1_2[2,:fparam1.nfout1_2]
            phvel1=fparam1.arr1_2[3,:fparam1.nfout1_2]
            obper2=fparam2.arr1_2[1,:fparam2.nfout1_2]
            gvel2=fparam2.arr1_2[2,:fparam2.nfout1_2]
            phvel2=fparam2.arr1_2[3,:fparam2.nfout1_2]
        else:
            obper1=fparam1.arr2_2[1,:fparam1.nfout2_2]
            gvel1=fparam1.arr2_2[2,:fparam1.nfout2_2]
            phvel1=fparam1.arr2_2[3,:fparam1.nfout2_2]
            obper2=fparam2.arr2_2[1,:fparam2.nfout2_2]
            gvel2=fparam2.arr2_2[2,:fparam2.nfout2_2]
            phvel2=fparam2.arr2_2[3,:fparam2.nfout2_2]
        plb.figure()
        ax = plt.subplot()
        ax.plot(obper1, gvel1, '--k', lw=3) #
        ax.plot(obper2, gvel2, '-.b', lw=3)
        plt.xlabel('Period(s)')
        plt.ylabel('Velocity(km/s)')
        plt.title('Group Velocity Comparison')
        if (fparam1.preflag==True and fparam2.preflag==True):
            plb.figure()
            ax = plt.subplot()
            ax.plot(obper1, phvel1, '--k', lw=3) #
            ax.plot(obper2, phvel2, '-.b', lw=3)
            plt.xlabel('Period(s)')
            plt.ylabel('Velocity(km/s)')
            plt.title('Phase Velocity Comparison')
        return

class snrParam(object):
    """
    SNR parameters:
        suffix: p=positve lag;  n=negative lag; s=symmetric lag
        amp: largest amplitude measurement for each period
        snr: SNR measurement for each period
        nrms: noise rms measurement for each period
        oper: observed period

    """
    def __init__(self):
        self.amp_p=np.array([])
        self.snr_p=np.array([])
        self.nrms_p=np.array([])
        self.oper_p=np.array([])
        self.amp_n=np.array([])
        self.snr_n=np.array([])
        self.nrms_n=np.array([])
        self.oper_n=np.array([])
        self.amp_s=np.array([])
        self.snr_s=np.array([])
        self.nrms_s=np.array([])
        self.oper_s=np.array([])

    def writeAMPSNR(self, fnamePR):
        """
        writeAMPSNR:
        Write output SNR parameters to text files
        _pos_amp_snr - positive lag
        _neg_amp_snr - negative lag
        _amp_snr     - symmetric lag
        """
        len_p=len(self.amp_p)
        len_n=len(self.amp_n)
        len_s=len(self.amp_s)
        if len_p!=0:
            fpos=fnamePR+'_pos_amp_snr'
            f=open(fpos,'w')
            for i in np.arange(len_p):
                tempstr='%8.4f   %.5g  %8.4f  \n' %(  self.oper_p[i] , self.amp_p[i],  self.snr_p[i] )
                f.writelines(tempstr)
            f.close()
        if len_n!=0:
            fneg=fnamePR+'_neg_amp_snr'
            f=open(fneg,'w')
            for i in np.arange(len_n):
                tempstr='%8.4f   %.5g  %8.4f  \n' %(   self.oper_n[i] , self.amp_n[i],  self.snr_n[i] )
                f.writelines(tempstr)
            f.close()
        if len_s!=0:
            fsym=fnamePR+'_amp_snr'
            f=open(fsym,'w')
            for i in np.arange(len_s):
                tempstr='%8.4f   %.5g  %8.4f  \n' %(   self.oper_s[i] , self.amp_s[i],  self.snr_s[i] )
                f.writelines(tempstr)
            f.close()
        return

class noisetrace(obspy.core.trace.Trace):
    """
    noisetrace:
    A derived class inherited from obspy.core.trace.Trace. This derived class have a variety of new member functions
    """
    def init_ftanParam(self):
        """
        Initialize ftan parameters
        """
        self.ftanparam=ftanParam()

    def init_snrParam(self):
        """
        Initialize SNR parameters
        """
        self.SNRParam=snrParam()

    def reverse(self):
        """
        Reverse the trace
        """
        self.data=self.data[::-1]
        return

    def makesym(self):
        """
        Turn the double lagged cross-correlation data to one single lag
        """
        if abs(self.stats.sac.b+self.stats.sac.e)>self.stats.delta:
            raise ValueError('Error: Not symmetric trace!');
        if self.stats.npts%2!=1:
            raise ValueError('Error: Incompatible begin and end time!');
        nhalf=(self.stats.npts-1)/2+1;
        neg=self.data[:nhalf];
        pos=self.data[nhalf-1:self.stats.npts];
        neg=neg[::-1];
        self.data=npr.evaluate( '(pos+neg)/2' );
        self.stats.npts=nhalf;
        self.stats.starttime=self.stats.starttime+self.stats.sac.e;
        self.stats.sac.b=0.;
        return

    def getneg(self):
        """
        Get the negative lag of a cross-correlation record
        """
        if abs(self.stats.sac.b+self.stats.sac.e)>self.stats.delta:
            raise ValueError('Error: Not symmetric trace!');
        negTr=self.copy();
        t=self.stats.starttime;
        L=(int)((self.stats.npts-1)/2)+1;
        negTr.data=negTr.data[:L];
        negTr.data=negTr.data[::-1];
        negTr.stats.npts=L;
        negTr.stats.sac.b=0.;
        negTr.stats.starttime=t-self.stats.sac.b;
        return negTr;

    def getpos(self):
        """
        Get the positive lag of a cross-correlation record
        """
        if abs(self.stats.sac.b+self.stats.sac.e)>self.stats.delta:
            raise ValueError('Error: Not symmetric trace!');
        posTr=self.copy();
        t=self.stats.starttime;
        L=(int)((self.stats.npts-1)/2)+1;
        posTr.data=posTr.data[L-1:];
        posTr.stats.npts=L;
        posTr.stats.sac.b=0.;
        posTr.stats.starttime=t-self.stats.sac.b;
        return posTr;

    def aftan(self, pmf=True, piover4=-1.0, vmin=1.5, vmax=5.0, tmin=4.0, \
        tmax=30.0, tresh=20.0, ffact=1.0, taperl=1.0, snr=0.2, fmatch=1.0,phvelname=''):

        """ (Automatic Frequency-Time ANalysis) aftan analysis:
        This function read SAC file, make it symmtric (if it is not), and then do aftan analysis.
        -----------------------------------------------------------------------------------------------------
        Input Parameters:
        pmf         - flag for Phase-Matched-Filtered output (default: True)
        piover4     - phase shift = pi/4*piover4, for cross-correlation piover4 should be -1.0
        vmin        - minimal group velocity, km/s
        vmax        - maximal group velocity, km/s
        tmin        - minimal period, s
        tmax        - maximal period, s
        tresh       - treshold for jump detection, usualy = 10, need modifications
        ffact       - factor to automatic filter parameter, usualy =1
        taperl      - factor for the left end seismogram tapering, taper = taperl*tmax,    (real*8)
        snr         - phase match filter parameter, spectra ratio to determine cutting point for phase matched filter
        fmatch      - factor to length of phase matching window
        fname       - SAC file name
        phvelname   - predicted phase velocity file name
        
        Output:
        self.ftanparam, a object of ftanParam class, to store output aftan results
        -----------------------------------------------------------------------------------------------------
        References:
        Levshin, A. L., and M. H. Ritzwoller. Automated detection, extraction, and measurement of regional surface waves.
             Monitoring the Comprehensive Nuclear-Test-Ban Treaty: Surface Waves. Birkhäuser Basel, 2001. 1531-1545.
        Bensen, G. D., et al. Processing seismic ambient noise data to obtain reliable broad-band surface wave dispersion measurements.
             Geophysical Journal International 169.3 (2007): 1239-1260.
        """
        try:
            self.ftanparam
        except:
            self.init_ftanParam()
        try:
            dist=self.stats.sac.dist;
        except:
            dist=self.stats.distance;
        if (phvelname==''):
            phvelname='./ak135.disp';
        nprpv = 0
        phprper=np.zeros(300)
        phprvel=np.zeros(300)
        if os.path.isfile(phvelname):
            php=np.loadtxt(phvelname)
            phprper=php[:,0]
            phprvel=php[:,1]
            nprpv = php[:,0].size
            phprper=np.append(phprper,np.zeros(300-phprper.size))
            phprvel=np.append(phprvel,np.zeros(300-phprvel.size))
            self.ftanparam.preflag=True
        nfin = 64
        npoints = 5  #  only 3 points in jump
        perc    = 50.0 # 50 % for output segment
        tempsac=self.copy()
        if abs(tempsac.stats.sac.b+tempsac.stats.sac.e)<tempsac.stats.delta:
            tempsac.makesym()
        tb=tempsac.stats.sac.b
        length=len(tempsac.data)
        if length>32768:
            print "Warning: length of seismogram is larger than 32768!"
            nsam=32768
            tempsac.data=tempsac.data[:nsam]
            tempsac.stats.e=(nsam-1)*tempsac.stats.delta+tb
            sig=tempsac.data
        else:
            sig=np.append(tempsac.data,np.zeros( float(32768-tempsac.data.size) ) )
            nsam=int( float (tempsac.stats.npts) )### for unknown reasons, this has to be done, nsam=int(tempsac.stats.npts)  won't work as an input for aftan
        dt=tempsac.stats.delta
        try:
            dist=tempsac.stats.sac.dist;
        except:
            dist=tempsac.stats.distance;
        # Start to do aftan utilizing pyaftan
        self.ftanparam.nfout1_1,self.ftanparam.arr1_1,self.ftanparam.nfout2_1,self.ftanparam.arr2_1,self.ftanparam.tamp_1,\
        self.ftanparam.nrow_1,self.ftanparam.ncol_1,self.ftanparam.ampo_1, self.ftanparam.ierr_1= ftan.aftanpg(piover4, nsam, \
        sig, tb, dt, dist, vmin, vmax, tmin, tmax, tresh, ffact, perc, npoints, taperl, nfin, snr, nprpv, phprper, phprvel)
        if pmf==True:
            if self.ftanparam.nfout2_1<3:
                return
            npred = self.ftanparam.nfout2_1
            tmin2 = self.ftanparam.arr2_1[1,0]
            tmax2 = self.ftanparam.arr2_1[1,self.ftanparam.nfout2_1-1]
            pred=np.zeros((2,300))
            pred[:,0:100]=self.ftanparam.arr2_1[1:3,:]
            pred=pred.T
            self.ftanparam.nfout1_2,self.ftanparam.arr1_2,self.ftanparam.nfout2_2,self.ftanparam.arr2_2,self.ftanparam.tamp_2, \
            self.ftanparam.nrow_2,self.ftanparam.ncol_2,self.ftanparam.ampo_2, self.ftanparam.ierr_2=ftan.aftanipg(piover4,nsam, \
            sig,tb,dt,dist,vmin,vmax,tmin2,tmax2,tresh,ffact,perc,npoints,taperl,nfin,snr,fmatch,npred,pred,nprpv,phprper,phprvel)
        return

    def findcoda(self, tfactor=2, Lwin=1200, Tmin=5.0, Tmax=10.0, method='stehly' ):
        """findcoda:
        Find the coda for a given trace, return both negative and positive trace and begin/end time
        This function find the coda window for a given noisetrace, the arrival of surface wave package is determined by aftan analysis.
        Two methods are utilized to find the coda.
        -----------------------------------------------------------------------------------------------------
        Input Parameters:
        tfactor - 
            method='stehly'
            begin time is the arrival time of surface wave multiplied by tfactor
            method='ma'
            begin time is the arrival time of surface wave plus tfactor( in sec)
        Lwin    - coda window length
        Tmin    - minimum period for FTAN analysis
        Tmax    - maximum period for FTAN analysis
        method  - 'stehly' or 'ma'
        
        Output:
        neg     - negative lag data (numpy array)
        pos     - positive lag data (numpy array)
        Tbeg    - begin time of coda window
        Tend    - end time of coda window
        -----------------------------------------------------------------------------------------------------
        References:
        Stehly, L., et al. Reconstructing Green's function by correlation of the coda of the correlation (C3) of ambient seismic noise.
             Journal of Geophysical Research: Solid Earth (1978–2012) 113.B11 (2008).
        Ma, Shuo, and Gregory C. Beroza. Ambient field Green's functions from asynchronous seismic observations. Geophysical Research Letters 39.6 (2012).
        """
        npts=self.stats.npts
        Nhalf=(npts-1)/2+1
        neg=self.data[:Nhalf]
        pos=self.data[Nhalf-1:]
        seis=self.copy()
        seis.aftan(tmin=Tmin, tmax=Tmax)
        self.ftanparam=seis.ftanparam
        vposition=np.where(np.logical_and(self.ftanparam.arr2_1[1,:]>Tmin, self.ftanparam.arr2_1[1,:]<Tmax))[0]
        if vposition.size==0:
            vposition=np.where(np.logical_and(self.ftanparam.arr1_1[1,:]>Tmin, self.ftanparam.arr1_1[1,:]<Tmax))[0]
            Vmean=self.ftanparam.arr1_1[2,vposition].mean()
            print "Warning: no jump corrected result for: "+self.stats.sac.kevnm.split('\x00')[0]+"_"+self.stats.station+" "+self.stats.channel+' V:' +str(Vmean)
        else:
            Vmean=self.ftanparam.arr2_1[2,vposition].mean()

        if method=='ma':
            Tbeg=max( (int)( (self.stats.sac.dist/Vmean)/self.stats.delta ) +(int)(tfactor/self.stats.delta) , 1)
            Tend=Tbeg+(int)(Lwin/self.stats.delta)
        else:
            Tbeg=(int)( (self.stats.sac.dist/Vmean)/self.stats.delta * tfactor ) +1
            Tend=Tbeg+(int)(Lwin/self.stats.delta)
        if Tend>Nhalf:
            Tend=Nhalf
            print "Warning: The default coda window end excess the record length!"
        neg=neg[::-1]
        neg=neg[Tbeg-1:Tend]
        pos=pos[Tbeg-1:Tend]
        return neg, pos, Tbeg, Tend

    def findcodaTime(self, tfactor=2, Lwin=1200, Tmin=5.0, Tmax=10.0, method='stehly' ):
        """findcodaTime:
        Simlar to findcoda but only return begin/end time.
        """
        npts=self.stats.npts
        Nhalf=(npts-1)/2+1
        seis=self.copy()
        seis.aftan(tmin=Tmin, tmax=Tmax)
        self.ftanparam=seis.ftanparam
        vposition=np.where(np.logical_and(self.ftanparam.arr2_1[1,:]>Tmin, self.ftanparam.arr2_1[1,:]<Tmax))[0]
        if vposition.size==0:
            vposition=np.where(np.logical_and(self.ftanparam.arr1_1[1,:]>Tmin, self.ftanparam.arr1_1[1,:]<Tmax))[0]
            Vmean=self.ftanparam.arr1_1[2,vposition].mean()
            print "Warning: no jump corrected result for: "+self.stats.sac.kevnm.split('\x00')[0]+"_"+self.stats.station+" "+self.stats.channel+' V:' +str(Vmean)
        else:
            Vmean=self.ftanparam.arr2_1[2,vposition].mean()
        if Vmean<0.5 or Vmean >6 or np.isnan(Vmean):
            return -1, -1
        if method=='ma':
            Tbeg=max( (int)( (self.stats.sac.dist/Vmean)/self.stats.delta ) +(int)(tfactor/self.stats.delta) , 1)
            Tend=Tbeg+(int)(Lwin/self.stats.delta)
        else:
            Tbeg=(int)( (self.stats.sac.dist/Vmean)/self.stats.delta * tfactor ) +1
            Tend=Tbeg+(int)(Lwin/self.stats.delta)
        if Tend>Nhalf:
            Tend=-1
            print "Warning: The default coda window end excess the record length!"
        return Tbeg, Tend;

    def findcommoncoda(self, Tbeg1, Tbeg2, Tend1, Tend2):
        """findcommoncoda:
        Return common coda trace for two noisetraces
        -----------------------------------------------------------------------------------------------------
        Input Parameters:
        Tbeg1, Tend1 - coda window time for the first trace
        Tbeg1, Tend1 - coda window time for the second trace
        
        Output:
        neg     - common negative lag data (numpy array)
        pos     - common positive lag data (numpy array)
        Tbeg    - begin time of common coda window
        Tend    - end time of common coda window
        -----------------------------------------------------------------------------------------------------
        """
        npts=self.stats.npts
        Nhalf=(npts-1)/2+1
        neg=self.data[:Nhalf]
        pos=self.data[Nhalf-1:]
        if Tbeg1>Tbeg2:
            Tbeg=Tbeg1
            Tend=Tend1
        else:
            Tbeg=Tbeg2
            Tend=Tend2
        neg=neg[::-1]
        neg=neg[Tbeg-1:Tend]
        pos=pos[Tbeg-1:Tend]
        return neg, pos, Tbeg, Tend;

    def plotftan(self, plotflag=3, sacname=''):
        """
        Plot ftan diagram:
        This function plot ftan diagram.
        -----------------------------------------------------------------------------------------------------
        Input Parameters:
        plotflag -
            0: only Basic FTAN
            1: only Phase Matched Filtered FTAN
            2: both
            3: both in one figure
        sacname - sac file name than can be used as the title of the figure
        -----------------------------------------------------------------------------------------------------
        """
        try:
            fparam=self.ftanparam
            if fparam.nfout1_1==0:
                return "Error: No Basic FTAN parameters!"
            dt=self.stats.delta
            dist=self.stats.sac.dist
            if (plotflag!=1 and plotflag!=3):
                v1=dist/(fparam.tamp_1+np.arange(fparam.ncol_1)*dt)
                ampo_1=fparam.ampo_1[:fparam.ncol_1,:fparam.nrow_1]
                obper1_1=fparam.arr1_1[1,:fparam.nfout1_1]
                gvel1_1=fparam.arr1_1[2,:fparam.nfout1_1]
                phvel1_1=fparam.arr1_1[3,:fparam.nfout1_1]
                plb.figure()
                ax = plt.subplot()
                p=plt.pcolormesh(obper1_1, v1, ampo_1, cmap='gist_rainbow',shading='gouraud')
                ax.plot(obper1_1, gvel1_1, '--k', lw=3) #
                if (fparam.preflag==True):
                    ax.plot(obper1_1, phvel1_1, '--w', lw=3) #

                if (fparam.nfout2_1!=0):
                    obper2_1=fparam.arr2_1[1,:fparam.nfout2_1]
                    gvel2_1=fparam.arr2_1[2,:fparam.nfout2_1]
                    phvel2_1=fparam.arr2_1[3,:fparam.nfout2_1]
                    ax.plot(obper2_1, gvel2_1, '-k', lw=3) #
                    if (fparam.preflag==True):
                        ax.plot(obper2_1, phvel2_1, '-w', lw=3) #
                cb = plt.colorbar(p, ax=ax)
                Tmin1=obper1_1[0]
                Tmax1=obper1_1[fparam.nfout1_1-1]
                vmin1= v1[fparam.ncol_1-1]
                vmax1=v1[0]
                plt.axis([Tmin1, Tmax1, vmin1, vmax1])
                plt.xlabel('Period(s)')
                plt.ylabel('Velocity(km/s)')
                plt.title('Basic FTAN Diagram '+sacname,fontsize=15)

            if fparam.nfout1_2==0 and plotflag!=0:
                return "Error: No PMF FTAN parameters!"
            if (plotflag!=0 and plotflag!=3):
                v2=dist/(fparam.tamp_2+np.arange(fparam.ncol_2)*dt)
                ampo_2=fparam.ampo_2[:fparam.ncol_2,:fparam.nrow_2]
                obper1_2=fparam.arr1_2[1,:fparam.nfout1_2]
                gvel1_2=fparam.arr1_2[2,:fparam.nfout1_2]
                phvel1_2=fparam.arr1_2[3,:fparam.nfout1_2]
                plb.figure()
                ax = plt.subplot()
                p=plt.pcolormesh(obper1_2, v2, ampo_2, cmap='gist_rainbow',shading='gouraud')
                ax.plot(obper1_2, gvel1_2, '--k', lw=3) #
                if (fparam.preflag==True):
                    ax.plot(obper1_2, phvel1_2, '--w', lw=3) #

                if (fparam.nfout2_2!=0):
                    obper2_2=fparam.arr2_2[1,:fparam.nfout2_2]
                    gvel2_2=fparam.arr2_2[2,:fparam.nfout2_2]
                    phvel2_2=fparam.arr2_2[3,:fparam.nfout2_2]
                    ax.plot(obper2_2, gvel2_2, '-k', lw=3) #
                    if (fparam.preflag==True):
                        ax.plot(obper2_2, phvel2_2, '-w', lw=3) #
                cb = plt.colorbar(p, ax=ax)
                Tmin2=obper1_2[0]
                Tmax2=obper1_2[fparam.nfout1_2-1]
                vmin2= v2[fparam.ncol_2-1]
                vmax2=v2[0]
                plt.axis([Tmin2, Tmax2, vmin2, vmax2])
                plt.xlabel('Period(s)')
                plt.ylabel('Velocity(km/s)')
                plt.title('PMF FTAN Diagram '+sacname,fontsize=15)

            if ( plotflag==3 ):
                v1=dist/(fparam.tamp_1+np.arange(fparam.ncol_1)*dt)
                ampo_1=fparam.ampo_1[:fparam.ncol_1,:fparam.nrow_1]
                obper1_1=fparam.arr1_1[1,:fparam.nfout1_1]
                gvel1_1=fparam.arr1_1[2,:fparam.nfout1_1]
                phvel1_1=fparam.arr1_1[3,:fparam.nfout1_1]
                plb.figure(num=None, figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
                ax = plt.subplot(2,1,1)
                p=plt.pcolormesh(obper1_1, v1, ampo_1, cmap='gist_rainbow',shading='gouraud')
                ax.plot(obper1_1, gvel1_1, '--k', lw=3) #
                if (fparam.preflag==True):
                    ax.plot(obper1_1, phvel1_1, '--w', lw=3) #
                if (fparam.nfout2_1!=0):
                    obper2_1=fparam.arr2_1[1,:fparam.nfout2_1]
                    gvel2_1=fparam.arr2_1[2,:fparam.nfout2_1]
                    phvel2_1=fparam.arr2_1[3,:fparam.nfout2_1]
                    ax.plot(obper2_1, gvel2_1, '-k', lw=3) #
                    if (fparam.preflag==True):
                        ax.plot(obper2_1, phvel2_1, '-w', lw=3) #
                cb = plt.colorbar(p, ax=ax)
                Tmin1=obper1_1[0]
                Tmax1=obper1_1[fparam.nfout1_1-1]
                vmin1= v1[fparam.ncol_1-1]
                vmax1=v1[0]
                plt.axis([Tmin1, Tmax1, vmin1, vmax1])
                plt.xlabel('Period(s)')
                plt.ylabel('Velocity(km/s)')
                plt.title('Basic FTAN Diagram '+sacname)

                v2=dist/(fparam.tamp_2+np.arange(fparam.ncol_2)*dt)
                ampo_2=fparam.ampo_2[:fparam.ncol_2,:fparam.nrow_2]
                obper1_2=fparam.arr1_2[1,:fparam.nfout1_2]
                gvel1_2=fparam.arr1_2[2,:fparam.nfout1_2]
                phvel1_2=fparam.arr1_2[3,:fparam.nfout1_2]

                ax = plt.subplot(2,1,2)
                p=plt.pcolormesh(obper1_2, v2, ampo_2, cmap='gist_rainbow',shading='gouraud')
                ax.plot(obper1_2, gvel1_2, '--k', lw=3) #
                if (fparam.preflag==True):
                    ax.plot(obper1_2, phvel1_2, '--w', lw=3) #

                if (fparam.nfout2_2!=0):
                    obper2_2=fparam.arr2_2[1,:fparam.nfout2_2]
                    gvel2_2=fparam.arr2_2[2,:fparam.nfout2_2]
                    phvel2_2=fparam.arr2_2[3,:fparam.nfout2_2]
                    ax.plot(obper2_2, gvel2_2, '-k', lw=3) #
                    if (fparam.preflag==True):
                        ax.plot(obper2_2, phvel2_2, '-w', lw=3) #
                cb = plt.colorbar(p, ax=ax)
                Tmin2=obper1_2[0]
                Tmax2=obper1_2[fparam.nfout1_2-1]
                vmin2= v2[fparam.ncol_2-1]
                vmax2=v2[0]
                plt.axis([Tmin2, Tmax2, vmin2, vmax2])
                plt.xlabel('Period(s)')
                plt.ylabel('Velocity(km/s)')
                plt.title('PMF FTAN Diagram '+sacname)
        except AttributeError:
            print 'Error: FTAN Parameters are not available!'
        return

    def GaussianFilter(self, fcenter, fhlen=0.008):
        """
        Gaussian Filter designed for SNR analysis, utilize pyfftw to do fft
        exp( (-0.5/fhlen^2)*(f-fcenter)^2 )
        -----------------------------------------------------------------------------------------------------
        Input parameters:
        fcenter - central period
        fhlen   - half length of Gaussian width
        -----------------------------------------------------------------------------------------------------
        """
        npts=self.stats.npts
        Ns=1<<(npts-1).bit_length()
        df=1.0/self.stats.delta/Ns
        nhalf=Ns/2+1
        fmax=(nhalf-1)*df
        if fcenter>fmax:
            fcenter=fmax
        alpha = -0.5/(fhlen*fhlen)
        F=np.arange(Ns)*df
        gauamp = F - fcenter
        sf=npr.evaluate('exp(alpha*gauamp**2)')
        sp, Ns=FFTW(self.data, direction='forward')
        filtered_sp=npr.evaluate('sf*sp')
        filtered_seis, Ns=FFTW(filtered_sp, direction='backward')
        filtered_seis=filtered_seis[:npts].real
        return filtered_seis

    def getSNR(self, foutPR='', fhlen=0.008, pmf=True, piover4=-1.0, vmin=1.5, vmax=5.0, tmin=4.0, \
        tmax=30.0, tresh=20.0, ffact=1.0, taperl=1.0, snr=0.2, fmatch=1.0,phvelname=''):
        """getSNR
        Get the SNR for signal window based on FTAN analysis.
        If input noisetrace is double-lagged, it will do SNR analysis for pos/neg lag; otherwise it will do SNR analysis for sym lag.
        -----------------------------------------------------------------------------------------------------
        Input Parameters:
        foutPR      - Output file prefix for positive and negative lags, default is ''(NOT to save positive/negative files)
        fhlen       - half length of Gaussian width
        pmf         - flag for Phase-Matched-Filtered output (default: True)
        piover4     - phase shift = pi/4*piover4, for cross-correlation piover4 should be -1.0
        vmin        - minimal group velocity, km/s
        vmax        - maximal group velocity, km/s
        tmin        - minimal period, s
        tmax        - maximal period, s
        tresh       - treshold for jump detection, usualy = 10, need modifications
        ffact       - factor to automatic filter parameter, usualy =1
        taperl      - factor for the left end seismogram tapering, taper = taperl*tmax,    (real*8)
        snr         - phase match filter parameter, spectra ratio to determine cutting point for phase matched filter
        fmatch      - factor to length of phase matching window
        fname       - SAC file name
        phvelname   - predicted phase velocity file name
        
        Output:
        self.SNRParam, a object of snrParam class to store output SNR results
        -----------------------------------------------------------------------------------------------------
        References:
        Tian, Ye, and Michael H. Ritzwoller. "Directionality of ambient noise on the Juan de Fuca plate:
            implications for source locations of the primary and secondary microseisms." Geophysical Journal International 201.1 (2015): 429-443.
        """
        try:
            self.SNRParam
        except:
            self.init_snrParam()
        try:
            dist=self.stats.sac.dist
            begT=self.stats.sac.b
            endT=self.stats.sac.e
            symFlag=False
            dt=self.stats.delta
            if ( abs(begT+endT) < dt ):
                symFlag=True
                begT=0.
                temp_pos=self.getpos()
                temp_neg=self.getneg()
                temp_pos.aftan(pmf=pmf, piover4=piover4, vmin=vmin, vmax=vmax, tmin=tmin, \
                    tmax=tmax, tresh=tresh, ffact=ffact, taperl=taperl, snr=snr, fmatch=fmatch,phvelname=phvelname)
                temp_neg.aftan(pmf=pmf, piover4=piover4, vmin=vmin, vmax=vmax, tmin=tmin, \
                    tmax=tmax, tresh=tresh, ffact=ffact, taperl=taperl, snr=snr, fmatch=fmatch,phvelname=phvelname)
                fparam_p=temp_pos.ftanparam
                fparam_n=temp_neg.ftanparam
                if foutPR!='':
                    temp_pos.ftanparam.writeDISP(foutPR+'_pos')
                    temp_neg.ftanparam.writeDISP(foutPR+'_neg')
            if symFlag==True:
                if fparam_p.nfout2_2!=0:
                    o_per_p=fparam_p.arr2_2[1,:]
                    g_vel_p=fparam_p.arr2_2[2,:]
                    self.SNRParam.oper_p=o_per_p
                ### positive lag
                for i in np.arange(fparam_p.nfout2_2):
                    filtered_tr=temp_pos.GaussianFilter(1./o_per_p[i], fhlen=fhlen)
                    minT = dist/g_vel_p[i]-o_per_p[i]/2.
                    maxT = dist/g_vel_p[i]+o_per_p[i]/2.
                    if g_vel_p[i]<0 or o_per_p[i]<0:
                         self.SNRParam.snr_p=np.append(self.SNRParam.snr_p, -1.);
                         self.SNRParam.amp_p=np.append(self.SNRParam.amp_p, -1.);
                         continue;
                    if(minT<begT):
                        minT=begT
                    if(maxT>endT):
                        maxT=endT
                    ib = (int)(minT/dt)
                    ie = (int)(maxT/dt)+2
                    tempTr_p=filtered_tr[ib:ie]
                    tempTr_p=npr.evaluate('abs(tempTr_p)')
                    tempmax_p=tempTr_p.max()
                    self.SNRParam.amp_p=np.append(self.SNRParam.amp_p, tempmax_p)
                    # Noise window
                    minT = maxT + o_per_p[i] * 5 + 500.
                    skipflag=False
                    if( (endT - minT) < 50. ):
                        self.SNRParam.snr_p=np.append(self.SNRParam.snr_p, -1.)
                        skipflag=True
                    elif( (endT - minT) < 1100. ):
                        maxT = endT - 10.
                    else:
                        minT = endT - 1100.
                        maxT = endT - 100.
                    if skipflag==False:
                        ib = (int)(minT/dt)
                        ie = (int)(maxT/dt)+2
                        tempnoise_p=filtered_tr[ib:ie]
                        tempnoise_p=npr.evaluate('tempnoise_p**2')
                        noiserms_p=math.sqrt(npr.evaluate('sum(tempnoise_p)')/(ie-ib-1.))
                        self.SNRParam.nrms_p=np.append(self.SNRParam.nrms_p, noiserms_p)
                        tempSNR_p=tempmax_p/noiserms_p
                        self.SNRParam.snr_p=np.append(self.SNRParam.snr_p, tempSNR_p)
                ### negative lag
                if fparam_n.nfout2_2!=0:
                    o_per_n=fparam_n.arr2_2[1,:]
                    g_vel_n=fparam_n.arr2_2[2,:]
                    self.SNRParam.oper_n=o_per_n
                for i in np.arange(fparam_n.nfout2_2):
                    minT = dist/g_vel_n[i]-o_per_n[i]/2.
                    maxT = dist/g_vel_n[i]+o_per_n[i]/2.
                    if g_vel_n[i]<0 or o_per_n[i]<0:
                         self.SNRParam.snr_n=np.append(self.SNRParam.snr_n, -1.);
                         self.SNRParam.amp_n=np.append(self.SNRParam.amp_n, -1.);
                         continue;
                    filtered_tr=temp_neg.GaussianFilter(1./o_per_n[i], fhlen=fhlen)
                    if(minT<begT):
                        minT=begT
                    if(maxT>endT):
                        maxT=endT
                    ib = (int)(minT/dt)
                    ie = (int)(maxT/dt)+2
                    # print ib,ie, minT, maxT, g_vel_n[i], o_per_n[i]
                    tempTr_n=filtered_tr[ib:ie]
                    tempTr_n=npr.evaluate('abs(tempTr_n)')
                    tempmax_n=tempTr_n.max()
                    self.SNRParam.amp_n=np.append(self.SNRParam.amp_n, tempmax_n)
                    # Noise window
                    minT = maxT + o_per_n[i] * 5 + 500.
                    skipflag=False
                    if( (endT - minT) < 50. ):
                        self.SNRParam.snr_n=np.append(self.SNRParam.snr_n, -1.)
                        skipflag=True
                    elif( (endT - minT) < 1100. ):
                        maxT = endT - 10.
                    else:
                        minT = endT - 1100.
                        maxT = endT - 100.
                    if skipflag==False:
                        ib = (int)(minT/dt)
                        ie = (int)(maxT/dt)+2
                        tempnoise_n=filtered_tr[ib:ie]
                        tempnoise_n=npr.evaluate('tempnoise_n**2')
                        noiserms_n=math.sqrt(npr.evaluate('sum(tempnoise_n)')/(ie-ib-1.))
                        self.SNRParam.nrms_n=np.append(self.SNRParam.nrms_n, noiserms_n)
                        tempSNR_n=tempmax_n/noiserms_n
                        self.SNRParam.snr_n=np.append(self.SNRParam.snr_n, tempSNR_n)
            else:
                fparam=self.ftanparam
                if fparam.nfout2_2!=0:
                    o_per=fparam.arr2_2[1,:]
                    g_vel=fparam.arr2_2[2,:]
                    self.SNRParam.oper_s=o_per
                for i in np.arange(fparam.nfout2_2):
                    filtered_tr=self.GaussianFilter(1./o_per[i], fhlen=fhlen)
                    minT = dist/g_vel[i]-o_per[i]/2.
                    maxT = dist/g_vel[i]+o_per[i]/2.
                    if g_vel[i]<0 or o_per[i]<0:
                         self.SNRParam.snr_s=np.append(self.SNRParam.snr_s, -1.);
                         self.SNRParam.amp_s=np.append(self.SNRParam.amp_s, -1.)
                         continue;
                    if(minT<begT):
                        minT=begT
                    if(maxT>endT):
                        maxT=endT
                    ib = (int)(minT/dt)
                    ie = (int)(maxT/dt)+2
                    tempTr_s=filtered_tr[ib:ie]
                    tempTr_s=npr.evaluate('abs(tempTr_s)')
                    tempmax_s=tempTr_s.max()
                    self.SNRParam.amp_s=np.append(self.SNRParam.amp_s, tempmax_s)
                    # Noise window
                    minT = maxT + o_per[i] * 5 + 500.
                    skipflag=False
                    if( (endT - minT) < 50. ):
                        self.SNRParam.snr_s=np.append(self.SNRParam.snr_s, -1.)
                        skipflag=True
                    elif( (endT - minT) < 1100. ):
                        maxT = endT - 10.
                    else:
                        minT = endT - 1100.
                        maxT = endT - 100.
                    if skipflag==False:
                        ib = (int)(minT/dt)
                        ie = (int)(maxT/dt)+2
                        tempnoise_s=filtered_tr[ib:ie]
                        tempnoise_s=npr.evaluate('tempnoise_s**2')
                        noiserms_s=math.sqrt(npr.evaluate('sum(tempnoise_s)')/(ie-ib-1.))
                        self.SNRParam.nrms_s=np.append(self.SNRParam.nrms_s, noiserms_s)
                        tempSNR_s=tempmax_s/noiserms_s
                        self.SNRParam.snr_s=np.append(self.SNRParam.snr_s, tempSNR_s)
        except AttributeError:
            print 'Error: FTAN Parameters are not available!'
        return

def FFTW(indata, direction, flags=['estimate']):
    """
    FFTW: a function utilizes fftw, a extremely fast library to do FFT computation (pyfftw3 need to be installed)
    -----------------------------------------------------------------------------------------------------
    Input Parameters:
    indata      - Input data
    direction   - direction of FFT
    flags       - list of fftw-flags to be used in planning
    -----------------------------------------------------------------------------------------------------
    Functions that using this function:
        noisetrace.GaussianFilter()
    """
    npts=indata.size
    Ns=1<<(npts-1).bit_length()
    INput = np.zeros((Ns), dtype=complex)
    OUTput = np.zeros((Ns), dtype=complex)
    fftw = fftw3.Plan(INput, OUTput, direction=direction, flags=flags)
    INput[:npts]=indata
    fftw()
    nhalf=Ns/2+1
    if direction == 'forward':
        OUTput[nhalf:]=0
        OUTput[0]/=2
        OUTput[nhalf-1]=OUTput[nhalf-1].real+0.j
    if direction =='backward':
        OUTput=2*OUTput/Ns
    return OUTput, Ns

class C3Param(object): ###
    """
    A subclass to store input parameters for C3 computation.
    -----------------------------------------------------------------------------------------------------
    Parameters:
    c3staList   - list to store stacode for C3 computation
    tfactor     - time window factor 
    Lwin        - length of coda window
    method      - defines how to choose C3 coda window, seed findcoda in noisetrace class for more details
    sepflag     - whether to seperate different component of C3 or stack them
    chan        - channel list
    Tmin        - minimum period
    Tmax        - maximum period
    -----------------------------------------------------------------------------------------------------
    """
    def __init__(self):
        self.c3staList=[]
        self.tfactor=2
        self.Lwin=1200
        self.method='stehly'
        self.sepflag=True
        self.chan=['BHZ']
        self.Tmin=5.
        self.Tmax=10.

class InputFtanParam(object): ###
    """
    A subclass to store input parameters for aftan analysis and SNR Analysis
    -----------------------------------------------------------------------------------------------------
    Parameters:
    pmf         - flag for Phase-Matched-Filtered output (default: True)
    piover4     - phase shift = pi/4*piover4, for cross-correlation piover4 should be -1.0
    vmin        - minimal group velocity, km/s
    vmax        - maximal group velocity, km/s
    tmin        - minimal period, s
    tmax        - maximal period, s
    tresh       - treshold for jump detection, usualy = 10, need modifications
    ffact       - factor to automatic filter parameter, usualy =1
    taperl      - factor for the left end seismogram tapering, taper = taperl*tmax,    (real*8)
    snr         - phase match filter parameter, spectra ratio to determine cutting point for phase matched filter
    fmatch      - factor to length of phase matching window
    fhlen       - half length of Gaussian width
    dosnrflag   - whether to do SNR analysis or not
    -----------------------------------------------------------------------------------------------------
    """
    def __init__(self):
        self.pmf=True
        self.piover4=-1.0
        self.vmin=1.5
        self.vmax=5.0
        self.tmin=4.0
        self.tmax=30.0
        self.tresh=20.0
        self.ffact=1.0
        self.taperl=1.0
        self.snr=0.2
        self.fmatch=1.0
        self.fhlen=0.008
        self.dosnrflag=True

    def setInParam(self, pmf=True, piover4=-1.0, vmin=1.5, vmax=5.0, tmin=4.0, \
        tmax=30.0, tresh=20.0, ffact=1.0, taperl=1.0, snr=0.2, fmatch=1.0, fhlen=0.008, dosnrflag=True):
        """
        Set the parameters
        """
        self.pmf=pmf
        self.piover4=piover4
        self.vmin=vmin
        self.vmax=vmax
        self.tmin=tmin
        self.tmax=tmax
        self.tresh=tresh
        self.ffact=ffact
        self.taperl=taperl
        self.snr=snr
        self.fmatch=fmatch
        self.fhlen=fhlen
        self.dosnrflag=dosnrflag
        return

class StaInfo(object):
    """
    An object contains a station information several methods for station related analysis.
    -----------------------------------------------------------------------------------------------------
    General Parameters:
    stacode     - station name
    network     - network
    virtual_Net - virtula network name
    chan        - channels for analysis
    lon,lat     - position for station
    elevation   - elevation
    start_date  - start date of deployment of the station
    end_date    - end date of deployment of the station
    chan        - channel name
    ccflag      - cross-correlation flag, used to control staPair generation ( not necessary for cross-correlation)
    -----------------------------------------------------------------------------------------------------
    """
    def __init__(self, stacode=None, network='', virtual_Net=None, lat=None, lon=None, \
        elevation=None,start_date=None, end_date=None, ccflag=None, chan=[]):

        self.stacode=stacode
        self.network=network
        self.virtual_Net=virtual_Net
        self.lon=lon
        self.lat=lat
        self.elevation=elevation
        self.start_date=start_date
        self.end_date=end_date
        self.chan=[]
        self.ccflag=ccflag
  
    # Member Functions for Receiver Function Analysis
    def init_RefDataBase(self, RFType='R'):
        """
        Initialize Receiver Function Database
        ------------------------------------------------------------------------------------------
        Parameters:
        RFType - Receiver function type, default is radial receiver function, represented as 'R'
        fnameNume   - List of file names for numerator (R/T component)
        fnameDeno   - List of file names for denominator (Z component)
        eventT      - List of event time(used as event id)
        deconvfname - List of deconvoled absolute path file names
        RFLst       - List of RFTrace object( RFStream object )
        PostLst     - List of PostRefDatabase object (PostRefLst object)
        Q1Lst       - PostRefLst that passes the step 1 quality control
        RBLst       - PostRefLst that passes the remove_bad quality control
        Q2Lst       - PostRefLst that passes the step 2 quality control
        ------------------------------------------------------------------------------------------
        """
        self.RFType=RFType
        self.fnameNume=[]
        self.fnameDeno=[]
        self.eventT=[]
        self.deconvfname=[]
        self.RFLst=ref.RFStream()
        self.PostLst=ref.PostRefLst()
        self.Q1Lst=ref.PostRefLst()
        self.RBLst=ref.PostRefLst()
        self.Q2Lst=ref.PostRefLst()
        return
    
    def GetPathfromSOD(self, datadir, netFlag=True):
        """
        Get the path of raw R(T)/Z component data downloaded by SOD
        ----------------------------------------------------------------------------------
        Input format:
        datadir/Event_*/net.stacode..Z.sac
        Output format:
        eventT    - year_month_day_hour_minute_second as event ID
        fnameNume - file names(not absolute path) for numerator (R/T component)
        fnameDeno - file names(not absolute path) for denominator (Z component)
        ----------------------------------------------------------------------------------
        """
        try:
            self.RFType
        except:
            self.init_RefDataBase()
            
        if netFlag==True and self.network!='':
            pattern=datadir+'/*/'+self.network+'.'+self.stacode+'*Z.sac'
        else:
            pattern=datadir+'/*/*.'+self.stacode+'*Z.sac'
        for absoluteP in glob.glob(pattern):
            absoluteP=absoluteP.split('/')
            eventdir=absoluteP[len(absoluteP)-2]
            fnameDeno=absoluteP[len(absoluteP)-1]
            tempfname=fnameDeno.split('.')
            chanDeno=tempfname[len(tempfname)-2]
            chanPRX=chanDeno.split('Z')[0]
            chanNume=chanPRX+self.RFType
            tempEvent=eventdir.split('_')
            year=tempEvent[1]
            month=tempEvent[2]
            day=tempEvent[3]
            hour=tempEvent[4]
            minute=tempEvent[5]
            second=tempEvent[6]
            self.eventT.append(year+'_'+month+'_'+day+'_'+hour+'_'+minute+'_'+second)
            fnameNume=''
            for i in np.arange(len(tempfname)-2):
                fnameNume=fnameNume+tempfname[i]+'.'
            fnameNume=fnameNume+chanNume+'.sac'
            self.fnameNume.append(fnameNume)
            self.fnameDeno.append(fnameDeno)
        return
    
    def fromSOD2Ref(self, datadir, outdir='', saveflag=True, tdel=5., f0 = 2.5, niter=200, minderr=0.001, phase='P', tbeg=-10., tend=30.,
                    PostFlag=True, outLstFlag=True):
        """
        Compute Receiver function from raw R(T)/Z data downloaded by SOD
        ----------------------------------------------------------------------------------
        Input Parameters:
        saveflag   - Save deconvolved receiver function or not
        tdel       - phase delay
        f0         - Gaussian width factor
        niter      - number of maximum iteration
        minderr    - minimum misfit improvement, iteration will stop if improvement between two steps is smaller than minderr
        phase      - phase name, default is P
        tbeg       - begin time, default is -10 sec before the arrival of given phase
        tend       - end time, default is 30 sec after the arrival of given phase
        PostFlag   - whether to save RFTrace to RFLst for post processing
        outLstFlag - whether to save absolute path file name to deconvfname
        
        Input format:
        datadir/Event_eventT/fnameNume AND  datadir/Event_eventT/fnameDeno
        
        Output format:
        deconvfname - List of deconvoled absolute path file names (optional)
        RFLst       - List of RFTrace object( RFStream object ), inData in RFTrace object is deleted to save memory (optional)
        outdir/stacode/stacode_eventT.eqr/eqt - SAC binary of deconvolve receiver function (optional)
        ----------------------------------------------------------------------------------
        """
        if len(self.eventT)!=len(self.fnameNume) or len(self.fnameDeno)!=len(self.fnameNume):
            raise ValueError('Error: Incompatible Ref DataBase!')
        for i in np.arange(len(self.eventT)):
            ZFname=datadir+'/Event_'+self.eventT[i]+'/'+self.fnameDeno[i]
            RTFname=datadir+'/Event_'+self.eventT[i]+'/'+self.fnameNume[i]
            RefTr=ref.RFTrace()
            if RefTr.ReadData(ZFname, RTFname, phase=phase, tbeg=tbeg, tend=tend): 
                RefTr.Iterdeconv(tdel=tdel, f0 = f0, niter=niter, minderr=minderr, phase=phase)
                outfname=outdir+'/'+self.stacode+'/'+self.stacode+'_'+self.eventT[i]
                if self.RFType=='R':
                    outfname=outfname+'.eqr'
                elif self.RFType=='T':
                    outfname=outfname+'.eqt'
                else:
                    raise ValueError('Error: Unknown ReF Type!')
                if outLstFlag==True:
                    self.deconvfname.append(outfname)
                if PostFlag==True:
                    del RefTr.inData
                    self.RFLst.append(RefTr)
                if saveflag==True and outdir!='':
                    with open(outfname, 'wb') as fout:
                        RefTr.WriteSacBinary(fout)
                    # RefTr.write(outfname,format='SAC')
        return
    
    def ReadDeconvRef(self, datadir):
        """
        Read Receiver function data.
        ----------------------------------------------------------------------------------
        Input format:
        datadir/stacode/stacode*.eqr
        Output format:
        RFLst  - RFStream object
        eventT - event ID
        ----------------------------------------------------------------------------------
        """
        pattern=datadir+'/'+self.stacode+'/'+self.stacode+'*.eqr'
        print pattern
        for absolutefname in glob.glob(pattern):
            try:
                absoluteP=absolutefname.split('/')
                fname=absoluteP[len(absoluteP)-1]
                tempEvent=fname.split('.')[0].split('_')
                year=tempEvent[1]
                month=tempEvent[2]
                day=tempEvent[3]
                hour=tempEvent[4]
                minute=tempEvent[5]
                second=tempEvent[6]
            except IndexError:
                print 'Error eqr file:' , absoluteP;
                return
            with open(absolutefname,'rb') as fin:
                RefTr=ref.RFTrace(fin);
            self.RFLst.append(RefTr);
            self.eventT.append(year+'_'+month+'_'+day+'_'+hour+'_'+minute+'_'+second)
        return

    def PostProcess1(self, outdir, freeDeconvFlag=True, VR=80., tdiff=0.08):
        """
        Post Processing receiver function (step 1), processing tecniques include:
        MoveOut     - MoveOut of receiver function
        StretchBack - Stretch receiver function back
        
        """
        print 'Start to do MoveOut!'
        for i in np.arange(len(self.RFLst)):
            print 'Do:' + self.eventT[i]
            deconvTr=self.RFLst[i]
            # deconvTr.MoveOutOLD()
            # deconvTr.StretchBackOLD()
            deconvTr.MoveOut()
            deconvTr.StretchBack()
            deconvTr.postdatabase.eventT=self.eventT[i]
            self.PostLst.append(deconvTr.postdatabase)
        # self.SavePostDataBase(outdir=outdir)
        if freeDeconvFlag==True:
            del self.RFLst
        print 'End of MoveOut!'
        return
    
    def PostProcess2(self, outdir, freeDeconvFlag=True, VR=80., tdiff=0.08):
        """
        Post Processing receiver function(step 2), processing tecniques include:
        QControl_s1 - Discard results with variance 
        remove_bad  - 
        """
        print 'Start to do Quality Control!'
        self.Q1Lst=self.PostLst.QControl_s1(VR=VR);
        outdir=outdir+'/'+self.stacode
        if not os.path.isdir(outdir):
            os.makedirs(outdir);
        # self.RBLst=self.Q1Lst.remove_badOLD(outdir=outdir);
        self.RBLst=self.Q1Lst.remove_bad(outdir=outdir);
        self.Q2Lst=self.RBLst.QControl_s2(tdiff=tdiff);
        print 'Start to do Harmonic Stripping!'
        # self.PostLst.HarmonicStrippingV3(outdir=outdir);
        # self.PostLst.HarmonicStrippingV3OLD(outdir=outdir);
        self.Q2Lst.HarmonicStrippingV1(stacode=self.stacode, outdir=outdir);
        
        self.WriteLst(outdir=outdir)
        return

    
    def PostProcess(self, outdir, freeDeconvFlag=True, VR=80., tdiff=0.08, saveHStxtFlag=False):
        """
        Post Processing receiver function , processing tecniques include:
        MoveOut     - MoveOut of receiver function
        StretchBack - Stretch receiver function back
        QControl_s1 - Discard results with variance reduction less than VR
        remove_bad & QControl_s2  - remove bad according to preliminary harmonic stripping results
        HarmonicStrippingV3 - Harmonic Stripping analysis for non-quality-controlled data
        HarmonicStrippingV1 - Harmonic Stripping analysis for quality-controlled data
        ----------------------------------------------------------------------------------
        Input format:
        RFLst     - RFStream object
        
        Output format:
        wmean.txt - 
        
        
        ----------------------------------------------------------------------------------
        """
        print 'Do PostProcessing for: ', self.stacode;
        for i in np.arange(len(self.RFLst)):
            # print 'Do:' + self.eventT[i];
            deconvTr=self.RFLst[i];
            if not deconvTr.MoveOut():
                print 'MoveOut Out of range:', self.eventT[i], self.stacode;
                continue;
            deconvTr.StretchBack();
            deconvTr.postdatabase.eventT=self.eventT[i];
            self.PostLst.append(deconvTr.postdatabase);
        if freeDeconvFlag==True:
            del self.RFLst
        if len(self.PostLst)==0:
            return
        # print 'Start to do Quality Control:', self.stacode;
        self.Q1Lst=self.PostLst.QControl_s1(VR=VR);
        if len(self.Q1Lst)==0:
            return
        outdir=outdir+'/'+self.stacode
        if not os.path.isdir(outdir):
            os.makedirs(outdir);
        self.RBLst=self.Q1Lst.remove_bad(outdir=outdir);
        if len(self.RBLst)==0:
            return
        self.Q2Lst=self.RBLst.QControl_s2(tdiff=tdiff);
        if len(self.Q2Lst)==0:
            return
        # print 'Start to do Harmonic Stripping:', self.stacode;
        # self.PostLst.HarmonicStrippingV3(outdir=outdir);
        warnings.filterwarnings('ignore', category=UserWarning, append=True);
        self.Q2Lst.HarmonicStrippingV1(stacode=self.stacode, outdir=outdir, saveHStxtFlag=saveHStxtFlag);
        # self.Q2Lst.HSDataBase.PlotHSStreams(outdir=outdir, stacode=self.stacode, longitude=self.lon, latitude=self.lat);
        # self.WriteLst(outdir=outdir);
        return
    
    def ReadHSDataBase(self, datadir, prefix='QC_'):
        """
        Read Harmonic Stripping results from QC_*.mseed files
        ----------------------------------------------------------------------------------
        Input format:
        datadir/stacode/QC_stacode_obs.mseed   - observed recever function, backazimuth is save to channel header
        datadir/stacode/QC_stacode_diff.mseed  - residual recever function
        datadir/stacode/QC_stacode_rep.mseed   - predicted recever function
        datadir/stacode/QC_stacode_rep0.mseed  - A0 recever function
        datadir/stacode/QC_stacode_rep1.mseed  - A1 recever function
        datadir/stacode/QC_stacode_rep2.mseed  - A2 recever function
        
        ----------------------------------------------------------------------------------
        """
        datadir=datadir+'/'+self.stacode;
        try:
            self.HSDataBase=ref.HarmonicStrippingDataBase();
            self.HSDataBase.LoadHSDatabase(datadir, stacode=prefix+self.stacode);
        except:
            print 'Error in Loading HSDataBase for station: ', self.stacode;
        return;
    
    def PlotHSDataBase(self, outdir, ampfactor=40, targetDT=0.2, browseflag=True, saveflag=True,\
            obsflag=1, diffflag=0, repflag=1, rep0flag=1, rep1flag=1, rep2flag=1):
        outdir=outdir+'/'+self.stacode;
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        try:
            self.HSDataBase.PlotHSStreams(outdir=outdir, stacode=self.stacode,ampfactor=ampfactor, targetDT=targetDT,\
                longitude=self.lon, latitude=self.lat, obsflag=obsflag, diffflag=diffflag, repflag=repflag,\
                rep0flag=rep0flag, rep1flag=rep1flag, rep2flag=rep2flag );
        except:
            print 'No HSDataBase!'
        return;
        
    
    def SavePostDataBase(self, outdir):
        """
        Save Post Processing receiver function data
        """
        VarR=np.array([])
        MoFlags=np.array([])
        # value1=np.array([])
        L=0
        for PostData in self.PostLst:
            outfname=outdir+'/'+self.stacode+'/0.06_'+self.stacode+'_'+PostData.eventT+'.out'
            np.savetxt(outfname,PostData.ampC, fmt='%g')
            outfname=outdir+'/'+self.stacode+'/stre_'+str(int(PostData.baz))+'_'+self.stacode+'_'+PostData.eventT+'.out'
            np.savetxt(outfname,PostData.ampTC, fmt='%g')
            outfname=outdir+'/'+self.stacode+'/stre_'+str(int(PostData.baz))+'_'+self.stacode+'_'+PostData.eventT+'.out.back'
            np.savetxt(outfname,PostData.strback, fmt='%g')
            VarR=np.append(VarR,PostData.VR)
            MoFlags=np.append(MoFlags,PostData.MOFlag)
            # value1=np.append(value1,PostData.value1)
            L=L+1
        CateArray=np.append(VarR,MoFlags)
        CateArray=CateArray.reshape((2,L))
        CateArray=CateArray.T
        outfname=outdir+'/'+self.stacode+'/'+self.stacode+'_cate.lst'
        np.savetxt(outfname, CateArray, fmt='%g')
        # CateArray=np.append(CateArray,value1)
        return
    
    def LoadPostDataBase(self, outdir):
        """
        Load Post Processing receiver function data
        """
        incatefname=outdir+'/'+self.stacode+'/'+self.stacode+'_cate.lst'
        CateArray=np.loadtxt(incatefname)
        VarR=CateArray[:,0]
        MOFlags=CateArray[:,1]
        for i in np.arange(len(self.eventT)):
            eventT=self.eventT[i]
            outfname=outdir+'/'+self.stacode+'/0.06_'+self.stacode+'_'+eventT+'.out'
            ampC=np.loadtxt(outfname)
            outfname=outdir+'/'+self.stacode+'/stre_'+str(int(self.RFLst[i].GetHvalue('baz')))+'_'+self.stacode+'_'+eventT+'.out'
            ampTC=np.loadtxt(outfname)
            outfname=outdir+'/'+self.stacode+'/stre_'+str(int(self.RFLst[i].GetHvalue('baz')))+'_'+self.stacode+'_'+eventT+'.out.back'
            strback=np.loadtxt(outfname)
            PostData=ref.PostRefDatabase()
            PostData.baz=int(self.RFLst[i].GetHvalue('baz'))
            PostData.ampC=ampC
            PostData.ampTC=ampTC
            PostData.strback=strback
            PostData.MOFlag=MOFlags[i]
            PostData.VR=VarR[i]
            PostData.eventT=eventT
            self.PostLst.append(PostData)
        return
    
    def WriteLst(self, outdir):
        """
        Write quality controlled stre_back to n.n1.lst
        """
        fname=outdir+'/n.n1.lst'
        f=open(fname,'wb')
        for PostData in self.Q2Lst:
            outfname='stre_'+str(int(PostData.baz))+'_'+self.stacode+'_'+PostData.eventT+'.out.back'
            f.writelines('%s \n' % (outfname) )
        f.close()
        return
    
    def GetTravelTimeFile(self, SLst, per, datadir, outdir, dirtin, \
                        minlon, maxlon, minlat, maxlat, tin='COR', dx=0.2, filetype='phase', chpair=['LHZ', 'LHZ'] ):
        """
        Generate Travel Time files for a given station(regarded as virtual source)/event.
        'surface' command in GMT 5.x.x is used to interpolate station points to grid points, with tension=0(minimum curvature).
        Input:
        SLst - station list( a StaLst object )
        per - period
        dx - dlon/dlat
        ----------------------------------------------------------------------------------
        Input format:
        datadir/dirtin/sta1/tin_sta1_chpair[0]_sta2_chpair[1].SAC*
        e.g. datadir/DISP/MONP/COR_MONP_LHZ_109C_LHZ.SAC_2_DISP.1 AND datadir/DISP/MONP/COR_MONP_LHZ_109C_LHZ.SAC_amp_snr
        
        Output format:
        outdir/travel_time_stacode.filetype.c.txt - travel time file
        outdir/sta1_TravelT.lst - travel time file for SNR>0 station points
        outdir/travel_time_stacode.filetype.c.txt.HD - interpolated travel time file 
        ----------------------------------------------------------------------------------
        """
        station1=self
        Tfname=outdir+'/travel_time_'+station1.stacode+'.'+filetype+'.c.txt'
        f=open(Tfname,'wb') 
        LonLst=np.array([])
        LatLst=np.array([])
        TLst=np.array([])
        Length=0
        # print 'Start to get travel time map for:',self.stacode;
        for station2 in SLst.stations:
            if station1.stacode>=station2.stacode:
                continue
            sta1=station1.stacode
            sta2=station2.stacode
            # print sta1, sta2 
            ### May need to be modified before running the code
            sacfname=datadir+"/"+dirtin+"/"+sta1+"/"+tin+"_"+sta1+"_"+chpair[0]+"_"+sta2+"_"+chpair[1]+".SAC";
            ###
            # sacfname=datadir+"/"+dirtin+"/"+sta2+"."+sta1+".BXZ.sem.sac" ### For Artie Project, Note the exchange of sta1 and sta2 !!!
            
            if not os.path.isfile(sacfname+'_2_DISP.1') or not os.path.isfile(sacfname+'_amp_snr'):
                sacfname=datadir+"/"+dirtin+"/"+sta1+"/"+tin+"_"+sta1+"_"+sta2+".SAC";
                if not os.path.isfile(sacfname+'_2_DISP.1') or not os.path.isfile(sacfname+'_amp_snr'):
                    continue;
            fDISP=noisefile(sacfname+'_2_DISP.1', 'DISP')
            fsnr=noisefile(sacfname+'_amp_snr','SNR')
            (pvel,gvel) = fDISP.get_phvel(per);
            (snr,signal1,noise1) = fsnr.get_snr(per);
            dist, az, baz=obsGeo.base.gps2dist_azimuth(station1.lat, station1.lon, station2.lat, station2.lon ) # distance is in m
            dist=dist/1000.
            if filetype=='phase':
                TravelT=dist/pvel
                if math.isnan(TravelT) or math.isnan(pvel) or math.isnan(snr):
                    print 'NaN Detected for:',sacfname
                    continue;
                f.writelines('%g %g %g %g %s %d \n' % ( station2.lon, station2.lat, TravelT, pvel, station2.stacode, int(snr) ) )
            else:
                TravelT=dist/gvel
                if math.isnan(TravelT) or math.isnan(pvel) or math.isnan(snr):
                    print 'NaN Detected for:',sacfname
                    continue;
                f.writelines('%g %g %g %g %s %d \n' % ( station2.lon, station2.lat, TravelT, gvel, station2.stacode, int(snr) ) )
            if snr > 0:
                LonLst=np.append(LonLst, station2.lon)
                LatLst=np.append(LatLst, station2.lat)
                TLst=np.append(TLst, TravelT)
                Length=Length+1
        try:
            lsnr=snr
        except:
            lsnr=10000
        f.writelines('%g %g %g %g %s %d \n' % ( station1.lon, station1.lat, 0., 999., station1.stacode, int(lsnr) ) )
        f.close()
        LonLst=np.append(LonLst, station1.lon)
        LatLst=np.append(LatLst, station1.lat)
        TLst=np.append(TLst, 0.)
        Length=Length+1
        if Length==1:
            return
        tempTout=np.append(LonLst, LatLst)
        tempTout=np.append(tempTout, TLst)
        tempTout=tempTout.reshape((3,Length))
        tempTout=tempTout.T
        tempoutfname=outdir+'/'+sta1+'_TravelT.lst'
        TfnameHD=Tfname+'.HD'
        np.savetxt(tempoutfname, tempTout, fmt='%g')
        tempGMT=outdir+'/'+sta1+'_GMT.sh'
        grdfile=outdir+'/'+str(per)+'_'+sta1+'.grd'
        f=open(tempGMT,'wb')
        npts_x=int((maxlon-minlon)/dx)+1
        npts_y=int((maxlat-minlat)/dx)+1
        maxlon=(npts_x-1)*dx+minlon
        maxlat=(npts_y-1)*dx+minlat
        REG='-R'+str(minlon)+'/'+str(maxlon)+'/'+str(minlat)+'/'+str(maxlat)
        f.writelines('gmtset MAP_FRAME_TYPE fancy \n');
        f.writelines('surface %s -T0.0 -G%s -I%g %s \n' %( tempoutfname, grdfile, dx, REG ));
        f.writelines('grd2xyz %s %s > %s \n' %( grdfile, REG, TfnameHD ));
        f.close()
        call(['bash', tempGMT])
        os.remove(grdfile)
        os.remove(tempGMT)
        return 
    
    def CheckTravelTimeCurvature(self, per, outdir, minlon, npts_x, minlat, npts_y, dx=0.2, filetype='phase'):
        """
        Check travel time curvature and discard those points with large curvatures.
        Station points at boundary will be discarded.
        Two interpolation schemes with different tension (0, 0.2) will be applied to the quality controlled travel time file. 
        ----------------------------------------------------------------------------------
        Input format:
        outdir/travel_time_stacode.filetype.c.txt.HD - interpolated travel time file 
        
        Output format:
        outdir/travel_time_stacode.filetype.c.txt_v1 - travel time file with flags( to label curvature ), points at boundaries are discarded.
        outdir/sta1_TravelT_v1.lst - travel time file for SNR>0 station points
        outdir/travel_time_stacode.filetype.c.txt_v1.HD - interpolated travel time file 
        outdir/travel_time_stacode.filetype.c.txt.HD_0.2 - interpolated travel time file with tension=0.2
        ----------------------------------------------------------------------------------
        """
        radius = 6371.1391285;
        TfnameHD=outdir+'/travel_time_'+self.stacode+'.'+filetype+'.c.txt.HD'
        if not os.path.isfile(TfnameHD):
            print 'HD Travel Time Not exist:',TfnameHD
            return
        InarrayHD=np.loadtxt(TfnameHD)
        tr_t=InarrayHD[:,2]
        ################# Need Double Check!!!
        tr_t=tr_t.reshape((npts_y,npts_x));
        tr_t=tr_t.T;
        tr_t=tr_t[:,::-1];
        ###############
        dy=dx;
        lat_temp = minlat+np.arange(npts_y)*dy
        PI=math.pi
        lat_temp=npr.evaluate('arctan(0.993277 * tan(lat_temp/180.*PI))*180./PI')
        dx_km=npr.evaluate('radius*sin( (90.-lat_temp)/180.*PI )*dx/180.*PI')
        dy_km = radius*dy/180.*PI
        Tfname=outdir+'/travel_time_'+self.stacode+'.'+filetype+'.c.txt'
        f=open(Tfname,'rb')
        Tfnamev1=outdir+'/travel_time_'+self.stacode+'.'+filetype+'.c.txt_v1'
        fout=open(Tfnamev1,'wb')
        LonLst=np.array([])
        LatLst=np.array([])
        TLst=np.array([])
        Length=0
        
        for tl in f.readlines():
            tl1 = tl.split();
            lon2 = float(tl1[0]);
            lat2 = float(tl1[1]);
            temp2 = float(tl1[2]);
            vel2 = float(tl1[3]);
            tname = tl1[4];
            tflag = int(tl1[5]);
            if (tflag<=0):
                fout.write("%g %g %g %g %s %d\n" % (lon2,lat2,temp2,vel2,tname,tflag));
                continue;
            if (temp2 < 2.*per):
                tflag = 3; # do not remove these points
                fout.write("%g %g %g %g %s %d\n" % (lon2,lat2,temp2,vel2,tname,tflag));
                LonLst=np.append(LonLst, lon2)
                LatLst=np.append(LatLst, lat2)
                TLst=np.append(TLst, temp2)
                Length=Length+1;
                continue;

            i = int((lon2-minlon)/dx + 0.5);
            j = int((lat2-minlat)/dy + 0.5);
            # print i, j, lon2, lat2, temp2-tr_t[i,j], dx_km[j], dy_km
            # print lon2, lat2, lat_temp[j]
            
            if (i<2 or j<2 or i>=npts_x-2 or j >= npts_y-2):
                continue;
            temp_x = (tr_t[i+2,j]/-12. + tr_t[i+1,j]*4./3. + tr_t[i,j]*-5./2. + tr_t[i-1,j]*4./3. + tr_t[i-2,j]/-12.)/dx_km[j]/dx_km[j];
            temp_y = (tr_t[i,j+2]/-12. + tr_t[i,j+1]*4./3. + tr_t[i,j]*-5./2. + tr_t[i,j-1]*4./3. + tr_t[i,j-2]/-12.)/dy_km/dy_km;
    
            temp = temp_x + temp_y; ### Why nor L2 norm???
            if (temp>0.005 or temp<-0.005):
                tflag = -3;
                fout.write("%g %g %g %g %s %g\n" % (lon2,lat2,temp2,vel2,tname,tflag));
            else:
                fout.write("%g %g %g %g %s %d\n" % (lon2,lat2,temp2,vel2,tname,tflag));
                LonLst=np.append(LonLst, lon2)
                LatLst=np.append(LatLst, lat2)
                TLst=np.append(TLst, temp2)
                Length=Length+1;
        fout.close();
        f.close();
        tempoutfname=outdir+'/'+self.stacode+'_TravelT_v1.lst'
        TfnameHD=Tfnamev1+'.HD'
        tempTout=np.append(LonLst, LatLst)
        tempTout=np.append(tempTout, TLst)
        tempTout=tempTout.reshape((3,Length))
        tempTout=tempTout.T
        np.savetxt(tempoutfname, tempTout, fmt='%g')
        if Length < 10:
            print 'Curvature quality control discard '+self.stacode+' at '+str(per)+' sec, '+str(Length)+' points!'
            return
        tempGMT=outdir+'/'+self.stacode+'_v1_GMT.sh'
        grdfile=outdir+'/'+self.stacode+'_v1.grd'
        f=open(tempGMT,'wb')
        maxlon=(npts_x-1)*dx+minlon
        maxlat=(npts_y-1)*dx+minlat
        REG='-R'+str(minlon)+'/'+str(maxlon)+'/'+str(minlat)+'/'+str(maxlat)
        f.writelines('gmtset MAP_FRAME_TYPE fancy \n');
        f.writelines('surface %s -T0.0 -G%s -I%g %s \n' %( tempoutfname, grdfile, dx, REG ));
        f.writelines('grd2xyz %s %s > %s \n' %( grdfile, REG, TfnameHD ));
        f.writelines('surface %s -T0.2 -G%s -I%g %s \n' %( tempoutfname, grdfile+'.T0.2', dx, REG ));
        f.writelines('grd2xyz %s %s > %s \n' %( grdfile+'.T0.2', REG, TfnameHD+'_0.2' ));
        f.close()
        call(['bash', tempGMT])
        os.remove(grdfile+'.T0.2')
        os.remove(grdfile)
        os.remove(tempGMT)
        return 
        
    def TravelTime2Slowness(self, datadir, outdir, per, minlon, npts_x, minlat, npts_y, dx=0.2, cdist=None, filetype='phase' ):
        """
        Generate Slowness Maps from Travel Time Maps.
        Two interpolated travel time file with different tension will be used for quality control.
        ----------------------------------------------------------------------------------
        Input format:
        datadir/travel_time_stacode.pflag.txt_v1.HD - interpolated travel time file (tension=0)
        datadir/travel_time_stacode.pflag.txt_v1.HD_0.2 - interpolated travel time file (tension=0.2)
        
        Output format:
        outdir/slow_azi_stacode.pflag.txt.HD.2.v2 - Slowness map
        ----------------------------------------------------------------------------------
        """
        if cdist==None:
            cdist=12.*per;
        period=per;
        cdist_1 = period * 4. * 3. + 50.;
        dy=dx;
        x0 = minlon;
        y0 = minlat;
        pflag=filetype+'.c';
        radius=6371.1391285;
        pi=math.pi;
        x1=x0+(npts_x-1)*dx;
        ### NOTE: Both dx_km and dy_km has nothing to do with longitude!
        y1=y0+(npts_y-1)*dy;
        Inarraydx=np.loadtxt('dx_km.txt');
        dx_km_in=Inarraydx[:,0];
        latlstin=Inarraydx[:,1];
        Inarraydy=np.loadtxt('dy_km.txt');
        dy_km_in=Inarraydy[:,0];
        i0=int((y0+90.)/dy+0.1);
        ilast=int((y1+90.)/dy+0.1);
        dx_km=dx_km_in[i0:ilast+1];
        dy_km=dy_km_in[i0:ilast+1];
        if dx_km.size !=npts_y or abs(y0-latlstin[i0])>0.01 or abs(y1-latlstin[ilast]) >0.01:
            raise ValueError('Incompatible npts for grid points:' + self.stacode );
        sta1=self.stacode;
        sta1_lon=self.lon;
        if sta1_lon<0:
            sta1_lon = sta1_lon + 360.;
        sta1_lat=self.lat;
        fnamev1HD=datadir+"/travel_time_"+sta1+"."+pflag+".txt_v1.HD";
        fnamev1HD02=datadir+"/travel_time_"+sta1+"."+pflag+".txt_v1.HD_0.2";
        # fnamev1=datadir+"/travel_time_"+sta1+"."+pflag+".txt_v1";
        fnamev1=datadir+"/"+sta1+"_TravelT_v1.lst";
        print fnamev1
        if not ( os.path.isfile(fnamev1HD) and os.path.isfile(fnamev1HD02) and os.path.isfile(fnamev1) ):
            print "Lack Travel Time File(s) for:" + sta1;
            return;
        fnameSlow=outdir+"/slow_azi_"+sta1+"."+pflag+".txt.HD.2.v2";
        fSlow=open(fnameSlow, 'wb');
        InarrayV1=np.loadtxt(fnamev1);
        lonin=InarrayV1[:,0];
        latin=InarrayV1[:,1];
        ttin=InarrayV1[:,2];
        Inv1HD=np.loadtxt(fnamev1HD);
        lonv1HD=Inv1HD[:,0];
        latv1HD=Inv1HD[:,1];
        tempv1HD=Inv1HD[:,2];
        Inv1HD02=np.loadtxt(fnamev1HD02);
        lonv1HD02=Inv1HD02[:,0];
        latv1HD02=Inv1HD02[:,1];
        tempv1HD02=Inv1HD02[:,2];
        difflonsum=npr.evaluate('sum(abs(lonv1HD-lonv1HD02))');
        difflatsum=npr.evaluate('sum(abs(latv1HD-latv1HD02))');
        if difflonsum>0.01 or difflatsum > 0.01:
            print "HD and HD_0.2 files not compatiable!";
            return;
        tr_t=tempv1HD;
        difftemp=tempv1HD-tempv1HD02;
        tr_t=tr_t*((difftemp<2.)*(difftemp>-2.));
        #######
        tr_t=tr_t.reshape((npts_y,npts_x));
        tr_t=tr_t.T;
        tr_t=tr_t[:,::-1];
        lon_arr=lonv1HD.reshape((npts_y,npts_x));
        lon_arr=lon_arr.T;
        lon_arr=lon_arr[:,::-1];
        lat_arr=latv1HD.reshape((npts_y,npts_x));
        lat_arr=lat_arr.T;
        lat_arr=lat_arr[:,::-1];
        ########
        reason_n=np.ones(tr_t.size);
        reason_n1=reason_n*(difftemp>2.);
        reason_n2=reason_n*(difftemp<-2.);
        reason_n=npr.evaluate('reason_n1+reason_n2');
        #########
        reason_n=reason_n.reshape((npts_y,npts_x));
        reason_n=reason_n.T;
        reason_n=reason_n[:,::-1];

        for i in np.arange(npts_x):
            for j in np.arange(npts_y):
                if reason_n[i,j]==1:
                    continue;
                lon=lon_arr[i,j];
                lat=lat_arr[i,j];
                temp_x=x0+i*dx;
                temp_y=y0+j*dy;
                if abs(lon-temp_x) > 0.01 or abs(lat-temp_y) > 0.01:
                    print 'Input npts not compatible with HD travel time file for: '+self.stacode;
                    return
                marker_EN=np.zeros((2,2));
                marker_nn=4;
                tflag = False;
                for ista in np.arange(len(lonin)):
                    lon2=lonin[ista];
                    lat2=latin[ista];
                    if abs(lon2-lon) > cdist/110. or abs(lat2-lat) > cdist/110.:
                        continue;
                    if lon2-lon<0:
                        marker_E=0;
                    else:
                        marker_E=1;
    
                    if lat2-lat<0:
                        marker_N=0;
                    else:
                        marker_N=1;
                    if marker_EN[marker_E , marker_N]!=0:
                        continue;
                    dist, az, baz=obsGeo.base.gps2dist_azimuth(lat,lon,lat2,lon2);
                    dist=dist/1000.
                    if dist< cdist*2 and dist >= 1:
                        marker_nn=marker_nn-1;
                        if marker_nn==0:
                            tflag = True;
                            break;
                        marker_EN[marker_E, marker_N]=1;
                if tflag==False :
                    tr_t[i,j]=0;
                    reason_n[i,j] = 2;
        # Start to Compute Gradient
        for i in np.arange(npts_x-2)+1:
            for j in np.arange(npts_y-2)+1:
                
                if reason_n[i,j] == 2 or reason_n[i,j] == 1:
                    fSlow.writelines("%lf %lf 0 999 %d\n" % (x0+i*dx, y0+j*dy, reason_n[i,j] ) );
                    # print x0+i*dx, y0+j*dy
                    continue;
                temp1=(tr_t[i+1,j]-tr_t[i-1,j])/2.0/dx_km[j];
                temp2=(tr_t[i,j+1]-tr_t[i,j-1])/2.0/dy_km[j];
                if abs(temp2)<0.00001:
                    temp2=0.00001;
                temp=math.sqrt(temp1**2+temp2**2); # magnitude of gradient

                if temp>0.6 or temp<0.2:
                    reason_n[i,j] = 3;
                    fSlow.writelines("%lf %lf 0 999 %d\n" % (x0+i*dx, y0+j*dy, reason_n[i,j] ) );
                elif tr_t[i+1, j]==0 or tr_t[i-1, j]==0 or tr_t[i, j+1]==0 or tr_t[i, j-1]==0:
                    reason_n[i,j] = 4;
                    fSlow.writelines("%lf %lf 0 999 %d\n" % (x0+i*dx, y0+j*dy, reason_n[i,j] ) );
                else:
                    lon = x0+i*dx;
                    lat = y0+j*dy;
                    tempdist=npr.evaluate('112.*sqrt(((lonin-lon)**2)*cos(lat*pi/180.)**2 + (latin-lat)**2)'); 
                    imin=np.argmin(tempdist);
                    mdist, az, baz=obsGeo.base.gps2dist_azimuth(lat,lon,latin[imin],lonin[imin]);
                    mdist=mdist/1000.
                    distevent, az, baz=obsGeo.base.gps2dist_azimuth(lat, lon, sta1_lat, sta1_lon); #### NEED TO BE CHECK!!!
                    distevent=distevent/1000.
                    # print "DISTEVENT: ", distevent
                    az = az + 180.;
                    az = 90.-az;
                    baz = 90.-baz;
                    if az > 180.:
                        az = az - 360.;
                    if az < -180.:
                        az = az + 360.;
                    if baz > 180.:
                        baz = baz - 360.;
                    if baz < -180.:
                        baz = baz + 360.;
                    ag1 = az;
                    ag2 = math.atan2(temp2, temp1)/pi*180.;
                    diffa = ag2 - ag1;
    
                    if diffa < -180.:
                        diffa = diffa + 360.;
                    if diffa > 180.:
                        diffa = diffa - 360.;
                    if distevent < cdist+50.:
                        fSlow.writelines("%lf %lf 0 999 5\n" % (lon, lat ) );
                    else:
                        # print x0+i*dx, y0+j*dy, temp, ag2, mdist, distevent, az, baz, diffa;
                        fSlow.writelines("%lf %lf %lf %lf %g %g %g %g %g\n" % (lon, lat, temp, ag2, mdist, distevent, az, baz, diffa) );
        fSlow.close();
        return

    def GeneratePrePhaseDISP(self, SLst, outdir, mapfile='./MAPS/smpkolya_phv'):
        """
        Generate Predicted Phase V Dispersion Curves for a StaInfo.
        Input:
        SLst - StaLst object
        outdir - output directories
        mapfile - Phase V maps
        ------------------------------------------------------------
        Input format:
        prephaseEXE pathfname mapfile perlst staname
        
        Output format:
        outdirL(outdirR)/sta1.sta2.pre
        ------------------------------------------------------------
        """
        pathfname=self.stacode+'_pathfile';
        ista=0;
        prephaseEXE='./mhr_grvel_predict/lf_mhr_predict_earth';
        perlst='./mhr_grvel_predict/perlist_phase';
        if not os.path.isfile(prephaseEXE):
            print 'lf_mhr_predict_earth executable does not exist!';
            return;
        if not os.path.isfile(perlst):
            print 'period list does not exist!';
            return;
        f=open(pathfname,'wb');
        for station in SLst.stations:
            if self.stacode==station.stacode or ( abs(self.lon-station.lon) < 0.1 and abs(self.lat-station.lat)<0.1 ):
                continue;
            ista=ista+1;
            f.writelines('%5d%5d %15s %15s %10.5f %10.5f %10.5f %10.5f \n' %(1, ista, self.stacode, station.stacode, self.lat, self.lon, station.lat, station.lon ));
        f.close();
        call([prephaseEXE, pathfname, mapfile, perlst, self.stacode]);
        os.remove(pathfname);
        outdirL=outdir+'_L';
        outdirR=outdir+'_R';
        if not os.path.isdir(outdirL):
            os.makedirs(outdirL);
        if not os.path.isdir(outdirR):
            os.makedirs(outdirR);
        fout=open(self.stacode+'_temp','wb');
        for l1 in open('PREDICTION_L'+'_'+self.stacode):
            l2 = l1.rstrip().split();
            if (len(l2)>8):
                fout.close();
                outname = outdirL + "/%s.%s.pre" % (l2[3],l2[4]);
                fout = open(outname,"w");
            else:
                fout.write("%g %g\n" % (float(l2[0]),float(l2[1])));
        
        for l1 in open('PREDICTION_R'+'_'+self.stacode):
            l2 = l1.rstrip().split();
            if (len(l2)>8):
                fout.close();
                outname = outdirR + "/%s.%s.pre" % (l2[3],l2[4]);
                fout = open(outname,"w");
            else:
                fout.write("%g %g\n" % (float(l2[0]),float(l2[1])));
        fout.close();
        os.remove(self.stacode+'_temp');
        os.remove('PREDICTION_L'+'_'+self.stacode);
        os.remove('PREDICTION_R'+'_'+self.stacode);
        return
                    
    def SODRF(self, datadir, outdir='', RFType='R', netFlag=True, saveflag=True, tdel=5., \
              f0 = 2.5, niter=200, minderr=0.001, phase='P', tbeg=-10., tend=30., PostFlag=True, outLstFlag=True):
        """
        Computation receiver function from data downloaded with SOD (has been rotated to R/T components)
        
        """
        print 'Do Receiver Function Analysis for:'+self.network+self.stacode
        self.init_RefDataBase(RFType=RFType)
        self.GetPathfromSOD(datadir=datadir, netFlag=netFlag)
        self.fromSOD2Ref(datadir=datadir, outdir=outdir, saveflag=saveflag, tdel=tdel, f0 = f0, niter=niter, minderr=minderr, phase=phase, tbeg=tbeg, tend=tend,
                    PostFlag=PostFlag, outLstFlag=outLstFlag);
        return
    
    def setChan(self, chan):
        self.chan=copy.copy(chan)

    def appendChan(self,chan):
        self.chan.append(copy.copy(chan))

    def get_contents(self):
        if self.stacode==None:
            print 'StaInfo NOT Initialized yet!'
            return
        if self.network!='':
            print 'Network:%16s' %(self.network)
        print 'Station:%20s' %(self.stacode)
        print 'Longtitude:%17.3f' %(self.lon)
        print 'Latitude:  %17.3f' %(self.lat)

        return

    def addHSlowness(self, datadir, prefix='', suffix=''):
        indir=datadir+'/'+self.stacode
        pattern='/'+prefix+'*'+suffix
        print indir+pattern
        for fname in glob.glob(indir+pattern):
            sacf=sacio.SacIO()
            with open(fname, 'rb') as fin:
                sacf.ReadSacHeader(fin)
            dist=sacf.GetHvalue('dist')
            if abs( dist ) <0.1:
                evla=sacf.GetHvalue('evla')
                evlo=sacf.GetHvalue('evlo')
                stla=sacf.GetHvalue('stla')
                stlo=sacf.GetHvalue('stlo')
                dist, az, baz=obsGeo.base.gps2dist_azimuth(evla, evlo, stla, stlo) # distance is in m
                dist=dist/1000.
                sacf.SetHvalue('dist',dist)
            depth=sacf.GetHvalue('evdp')/1000.
            dist=obsGeo.kilometer2degrees(dist)
            tt=obspy.taup.taup.getTravelTimes(delta=dist, depth=depth, model='iasp91',phase_list=['P'])
            if len(tt)!=0:
                Hslowness=tt[0]['dT/dD']
            else:
                Hslowness=-1
            sacf.SetHvalue('user4',Hslowness)
            with open(fname, 'rb+') as fout:
                sacf.WriteSacHeader(fout)
        return

    def Decimate(self, factor, datadir, outdir, prefix='',suffix='' ):
        indir=datadir+'/'+self.stacode
        Odir=outdir+'/'+self.stacode
        pattern=prefix+'*'+suffix
        print indir
        os.chdir(indir)
        for fname in glob.glob(pattern):
            sacf=sacio.SacIO()
            st = obspy.core.read(fname)
            tr = st[0]
            tr.decimate(factor=factor, no_filter=True)
            sac1=obspy.sac.sacio.SacIO()
            sac1.readTrace(tr)
            sac2=obspy.sac.sacio.SacIO()
            with open(fname) as fh:
                sac2.ReadSacHeader(fh)
            sac1.SetHvalue('nzyear',sac2.GetHvalue('nzyear'))
            sac1.SetHvalue('nzjday',sac2.GetHvalue('nzjday'))
            sac1.SetHvalue('nzhour',sac2.GetHvalue('nzhour'))
            sac1.SetHvalue('nzmin',sac2.GetHvalue('nzmin'))
            sac1.SetHvalue('nzsec',sac2.GetHvalue('nzsec'))
            sac1.SetHvalue('nzmsec',sac2.GetHvalue('nzmsec'))
            with open(Odir+'/'+fname,'wb') as fh:
                sac1.WriteSacBinary(fh)
        return

    def ChangeChName(self, datadir, outdir, mlist, inchan,outchan):
        sta=self.stacode
        for mon in mlist:
            for day in np.arange(31):
                day=day+1
                mday=mon+'.'+str(day)
                indir=datadir+'/'+mon+'/'+mday
                if not os.path.isdir(indir):
                    continue
                Odir=outdir+'/'+mon+'/'+mday
                if not os.path.isdir(Odir):
                    os.makedirs(Odir)
                infname=indir+'/ft_'+mday+'.'+sta+'.'+inchan+'.SAC'
                if not os.path.isfile(infname):
                    continue
                outfname=Odir+'/ft_'+mday+'.'+sta+'.'+outchan+'.SAC'
                sac_ft=obspy.sac.sacio.SacIO()
                with open(infname,'rb') as fh:
                    sac_ft.ReadSacFile(fh)
                sac_ft.SetHvalue('kcmpnm',outchan)
                with open(outfname,'wb') as fh:
                    sac_ft.WriteSacBinary(fh)
                if os.path.isfile(infname+'_rec'):
                    shutil.copy(infname+'_rec',outfname+'_rec')
                if os.path.isfile(infname+'_rec1'):
                    shutil.copy(infname+'_rec1',outfname+'_rec1')
                if os.path.isfile(infname+'_rec2'):
                    shutil.copy(infname+'_rec2',outfname+'_rec2')
                print infname +'::'+outfname
#                sac_am=obspy.sac.sacio.SacIO()
#                sac_ph=obspy.sac.sacio.SacIO()
#                with open(infname+'.am','rb') as fh:
#                    sac_am.ReadSacFile(fh)
#                with open(infname+'.ph','rb') as fh:
#                    sac_ph.ReadSacFile(fh)
#                sac_am.SetHvalue('kcmpnm',outchan)
#                sac_ph.SetHvalue('kcmpnm',outchan)
#                with open(outfname+'.am','wb') as fh:
#                    sac_am.WriteSacBinary(fh)
#                with open(outfname+'.ph','wb') as fh:
#                    sac_ph.WriteSacBinary(fh)
        return


    def getDISP(self, datadir, minlon, maxlon, minlat, maxlat, dlon, dlat, outdir=None, crifactor=1., dirPFX='', dirSFX='', fPRX='', fSFX='.phv'):
        """
        Get the dispersion curve from near geographycal nodes
        ------------------------------------------------------------
        Input format:
        datadir/dirPFX+ind+dirSFX/fPRX+ind+fSFX
        e.g. datadir/100_23/100_23.phv
        
        Output format:
        outdir/self.stacode+fSFX
        e.g. outdir/AK.AKBB.phv
        ------------------------------------------------------------
        """
        self.GetPoslon()
        lon=self.lon
        lat=self.lat
        dist_x, az, baz = obsGeo.base.gps2dist_azimuth(float(int(lat)), float(int(lon)), float(int(lat)), float(int(lon))+dlon);
        dist_x=dist_x/1000.;
        Nx=int(111.1949/dist_x)+1;
        Ny=5;
        nearpoints={}
        weight_sum=0.;
        
        for ilon in np.arange(2*Nx+1)-Nx:
            for ilat in np.arange(2*Ny+1)-Ny:
                clon=float(int(lon))+ilon*dlon;
                clat=float(int(lat))+ilat*dlat;
                if float(clon)>maxlon or float(clon) < minlon or float(clat)>maxlat or float(clat) < minlat:
                    continue
                dist, az, baz = obsGeo.base.gps2dist_azimuth(float(clat), float(clon) , float(lat), float(lon))
                dist=dist/1000.
                if dist > crifactor*111.1949:
                    # print clat, dist, crifactor*111.1949
                    continue
                nloc = "%g_%g" % (clon, clat);
                infname=datadir+'/'+dirPFX+nloc+dirSFX+'/'+fPRX+nloc+fSFX;
                if not os.path.isfile(infname):
                    continue
                nearpoints[nloc]=1.-dist/(crifactor*111.1949);
                weight_sum=weight_sum+1.-dist/(crifactor*111.1949);
        if len(nearpoints) == 0:
            print 'No near points for:' + self.stacode;
            return;
        ind0=nearpoints.keys()[0];
        infname=datadir+'/'+dirPFX+ind0+dirSFX+'/'+fPRX+ind0+fSFX;
        avgArr=nearpoints[ind0]/weight_sum*np.loadtxt(infname);
        for ind in nearpoints.keys()[1:]:
            weight=nearpoints[ind]/weight_sum;
            infname=datadir+'/'+dirPFX+ind+dirSFX+'/'+fPRX+ind+fSFX;
            Inarray=weight*np.loadtxt(infname);
            avgArr=avgArr+Inarray;
            # print ind, weight, self.lon, self.lat, self.stacode;
        self.avgArr=avgArr;
        if outdir!=None:
            outfname=outdir+'/'+self.stacode+fSFX;
            np.savetxt(outfname, avgArr, fmt='%g');
        return
    
    def CheckDispersion(self, datadir, minlon, maxlon, minlat, maxlat, outdir=None, BrowseFlag=False, SaveFlag=True):
        """
        Compute predicted group velocity curve from phase velocity curve(centered difference scheme), and plot them.
        ---------------------------------------------------------------------------------
        Input format:
        datadir/stacode.phv AND datadir/stacode.grv
        Output format:
        outdir/stacode_DISP.ps
        
        Fomula for group V computation:
        U=c/(1-c*T*dc/dT) where U - group V; c - phase V; T - period
        ---------------------------------------------------------------------------------
        """
        if outdir==None:
            outdir=datadir;
        self.GetPoslon()
        PI=math.pi;
        infname_ph=datadir+'/'+self.stacode+'.phv';
        infname_gr=datadir+'/'+self.stacode+'.grv';
        if not ( os.path.isfile(infname_ph) and os.path.isfile(infname_gr) ):
            print 'Dispersion Curve not exist: ', self.stacode;
            return
        Inarr_ph=np.loadtxt(infname_ph);
        Inarr_gr=np.loadtxt(infname_gr);
        Tph=Inarr_ph[:,0];
        Vph=Inarr_ph[:,1];
        Tgr=Inarr_gr[:,0];
        Vgr=Inarr_gr[:,1];
        Tmin=Tph.min();
        Tmax=Tph.max();
        dT=2.
        T_int=np.arange((Tmax-Tmin)/dT)*dT+Tmin;
        V_intph=np.interp(T_int, Tph, Vph);
        V_intgr=np.interp(T_int, Tgr, Vgr);
        Sc_intph=1./V_intph;
        Sc_for_ph=Sc_intph[2:];
        Sc_bak_ph=Sc_intph[:len(V_intph)-2];
        # V_cen_ph=V_intph[1:len(V_intph)-1];
        Sc_intph=Sc_intph[1:len(V_intph)-1];
        T_cal_gr=T_int[1:len(V_intph)-1];
        diff_cal_gr=(Sc_for_ph-Sc_bak_ph)/2./dT;
        Sc_cal_gr=Sc_intph-T_cal_gr*diff_cal_gr;
        V_cal_gr=1./Sc_cal_gr
        # Start to Plot
        fig=plb.figure(num=None, figsize=(8, 12), dpi=80, facecolor='w', edgecolor='k');
        ax=plt.subplot(2,1,1);
        m = Basemap(llcrnrlon=minlon, llcrnrlat=minlat, urcrnrlon=maxlon, urcrnrlat=maxlat, \
                rsphere=(6378137.00,6356752.3142), resolution='c', projection='merc')
        lon = self.lon
        lat = self.lat
        x,y = m(lon, lat)
        m.plot(x, y, 'r^', markersize=6)
        m.drawcoastlines()
        m.etopo()
        # draw parallels
        m.drawparallels(np.arange(-90,90,10),labels=[1,1,0,1])
        # draw meridians
        m.drawmeridians(np.arange(-180,180,10),labels=[1,1,0,1])
        self.GetNeglon()
        plt.title(self.stacode+' longitude: %g latitude: %g' %( self.lon, self.lat ))
        ax=plt.subplot(2,1,2);
        line_ph, = ax.plot(Tph, Vph, '-b', lw=2);
        line_gr, = ax.plot(T_int, V_intgr, '--r', lw=2);
        line_cal, = ax.plot(T_cal_gr, V_cal_gr, '-.k', lw=2);
        ax.legend([line_ph, line_gr, line_cal], ['Phase V', 'Group V', 'Predicted Group V'], loc=0)
        plt.xlabel('Period(s)');
        plt.ylabel('Velocity(km/s)');
        # plt.title(self.stacode+' longitude: %g latitude: %g' %( self.lon, self.lat ));
        if BrowseFlag==True:
            plt.draw()
            plt.pause(1) # <-------
            raw_input("<Hit Enter To Close>")
            plt.close('all')
        if SaveFlag==True and outdir!='':
            fig.savefig(outdir+'/'+self.stacode+'_DISP.ps', format='ps')
        return;
    
    
    def GetPoslon(self):
        if self.lon<0:
            self.lon=self.lon+360.;
        return
    
    def GetNeglon(self):
        if self.lon>180.:
            self.lon=self.lon-360.;
        return
    
class StaLst(object):
    """
    An object contains a station list(a list of StaInfo object) information several methods for station list related analysis.
        stations: list of StaInfo
    """
    def __init__(self,stations=None):
        self.stations=[]
        if isinstance(stations, StaInfo):
            stations = [stations]
        if stations:
            self.stations.extend(stations)

    def __add__(self, other):
        """
        Add two StaLst with self += other.
        """
        if isinstance(other, StaInfo):
            other = StaLst([other])
        if not isinstance(other, StaLst):
            raise TypeError
        stations = self.stations + other.stations
        return self.__class__(stations=stations)

    def __len__(self):
        """
        Return the number of Traces in the StaLst object.
        """
        return len(self.stations)

    def __getitem__(self, index):
        """
        __getitem__ method of obspy.Stream objects.
        :return: Trace objects
        """
        if isinstance(index, slice):
            return self.__class__(stations=self.stations.__getitem__(index))
        else:
            return self.stations.__getitem__(index)

    def append(self, station):
        """
        Append a single StaInfo object to the current StaLst object.
        """
        if isinstance(station, StaInfo):
            self.stations.append(station)
        else:
            msg = 'Append only supports a single StaInfo object as an argument.'
            raise TypeError(msg)
        return self

    def ReadStaList(self, stafile):
        """
        Read Sation List from a txt file
        stacode longitude latidute network
        """
        f = open(stafile, 'r')
        Sta=[]
        for lines in f.readlines():
            lines=lines.split()
            stacode=lines[0]
            lon=float(lines[1])
            lat=float(lines[2])
            network=''
            ccflag=None
            if len(lines)==5:
                try:
                    ccflag=int(lines[3])
                    network=lines[4]
                except ValueError:
                    ccflag=int(lines[4])
                    network=lines[3]
            if len(lines)==4:
                try:
                    ccflag=int(lines[3])
                except ValueError:
                    network=lines[3]
            netsta=network+'.'+stacode
            if Sta.__contains__(netsta):
                index=Sta.index(netsta)
                if abs(self[index].lon-lon) >0.01 and abs(self[index].lat-lat) >0.01:
                    raise ValueError('Incompatible Station Location:' + netsta+' in Station List!')
                else:
                    print 'Warning: Repeated Station:' +netsta+' in Station List!'
                    continue
            Sta.append(netsta)
            self.append(StaInfo (stacode=stacode, network=network, lon=lon, lat=lat, ccflag=ccflag ))
            f.close()
        return

    def MakeDirs(self, outdir, dirtout='COR'):
        """
        Create directories for the station list.
        directories format:
        outdir/dirtout/stacode
        """
        for station in self.stations:
            if dirtout=='':
                odir=outdir+'/'+station.stacode
            else:
                odir=outdir+'/'+dirtout+'/'+station.stacode
            if not os.path.isdir(odir):
                os.makedirs(odir)
        return

    def addHSlowness(self, datadir, prefix='', suffix=''):
        """
        Add horizontal slowness according to a StaLst. The slowness is computed with taup.
        """
        for station in self.stations:
            station.addHSlowness(datadir=datadir, prefix=prefix, suffix=suffix)
        print 'End of Adding Horizontal Slowness!'
        return

    def addHSlownessParallel(self, datadir, prefix='', suffix=''):
        ADDSLOW = partial(StationsAddHSlowness, datadir=datadir, prefix=prefix, suffix=suffix)
        pool =mp.Pool()
        pool.map(ADDSLOW, self.stations) #make our results with a map call
        pool.close() #we are not adding any more processes
        pool.join() #tell it to wait until all threads are done before going on
        print 'End of Adding Horizontal Slowness  ( Parallel ) !'
        return

    def Decimate(self, factor, datadir, outdir, prefix='',suffix='' ):
        for station in self.stations:
            station.Decimate(factor=factor, datadir=datadir, outdir=outdir, prefix=prefix, suffix=suffix)
        print 'End of Decimation!'
        return

    def DecimateParallel(self, factor, datadir, outdir, prefix='', suffix=''):
        DECIMATE = partial(StationsDecimate, factor=factor, datadir=datadir, outdir=outdir, prefix=prefix, suffix=suffix)
        pool = mp.Pool()
        pool.map(DECIMATE, self.stations) #make our results with a map call
        pool.close() #we are not adding any more processes
        pool.join() #tell it to wait until all threads are done before going on
        print 'End of Decimation  ( Parallel ) !'
        return

    def ChangeChName(self, datadir, outdir, Mbeg, Mend, inchan, outchan):
        """
        TODO: if paralle, use mlist
        """
        mlist=GeMonLst( Mbeg, Mend )
        for station in self.stations:
            station.ChangeChName(datadir=datadir, outdir=outdir, mlist=mlist, inchan=inchan, outchan=outchan)
        print 'End of Changing Channel Names!'
        return

    def SODRF(self, datadir, outdir='', RFType='R', netFlag=True, saveflag=True, tdel=5., \
              f0 = 2.5, niter=200, minderr=0.001, phase='P', tbeg=-10., tend=30., PostFlag=False, outLstFlag=True):
        for station in self.stations:
            station.SODRF(datadir=datadir, outdir=outdir, RFType=RFType, netFlag=netFlag, saveflag=saveflag, tdel=tdel, \
              f0 = f0, niter=niter, minderr=minderr, phase=phase, tbeg=tbeg, tend=tend, PostFlag=PostFlag, outLstFlag=outLstFlag)
        print 'End of Receiver Function Analysis from SOD Data!'
        return
    
    def SODRFParallel(self, datadir, outdir='', RFType='R', netFlag=True, saveflag=True, tdel=5., \
              f0 = 2.5, niter=200, minderr=0.001, phase='P', tbeg=-10., tend=30., PostFlag=False, outLstFlag=True):
        SODRECEIVER=partial(StationsSODRF, datadir=datadir, outdir=outdir, RFType=RFType, netFlag=netFlag, saveflag=saveflag, tdel=tdel, \
              f0 = f0, niter=niter, minderr=minderr, phase=phase, tbeg=tbeg, tend=tend, PostFlag=PostFlag, outLstFlag=outLstFlag)
        pool =mp.Pool();
        pool.map(SODRECEIVER, self.stations) #make our results with a map call
        pool.close() #we are not adding any more processes
        pool.join() #tell it to wait until all threads are done before going on
        print 'End of DReceiver Function Analysis from SOD Data ( Parallel ) !'
        return
    
    def PostProcess(self, datadir, outdir, freeDeconvFlag=True, VR=80., tdiff=0.08, saveHStxtFlag=False):
        """
        Post Processing receiver function , processing tecniques include:
        MoveOut     - MoveOut of receiver function
        StretchBack - Stretch receiver function back
        QControl_s1 - Discard results with variance reduction less than VR
        remove_bad & QControl_s2  - remove bad according to preliminary harmonic stripping results
        HarmonicStrippingV3 - Harmonic Stripping analysis for non-quality-controlled data
        HarmonicStrippingV1 - Harmonic Stripping analysis for quality-controlled data
        ----------------------------------------------------------------------------------
        Input format:
        RFLst                  - RFStream object
        
        Output format:
        wmean.txt              - 
        average_vr.dat         - 
        variance_reduction.dat - 
        QC_stacode_obs.mseed   - observed recever function, backazimuth is save to channel header
        QC_stacode_diff.mseed  - residual recever function
        QC_stacode_rep.mseed   - predicted recever function
        QC_stacode_rep0.mseed  - A0 recever function
        QC_stacode_rep1.mseed  - A1 recever function
        QC_stacode_rep2.mseed  - A2 recever function
        ----------------------------------------------------------------------------------
        """
        for station in self.stations:
            station.init_RefDataBase();
            station.ReadDeconvRef(datadir=datadir);
            station.PostProcess(outdir=outdir, freeDeconvFlag=freeDeconvFlag, VR=VR, tdiff=tdiff, saveHStxtFlag=saveHStxtFlag);
        print 'End of Post Processing of Receiver Functions!'
        return
        
    def PostProcessParallel(self, datadir, outdir, freeDeconvFlag=True, VR=80., tdiff=0.08, saveHStxtFlag=False):
        """
        Parallel version of PostProcess
        """
        POSTPROCESS=partial(StationsPostProcess, datadir=datadir, outdir=outdir,\
            freeDeconvFlag=freeDeconvFlag, VR=VR, tdiff=tdiff, saveHStxtFlag=saveHStxtFlag)
        pool =mp.Pool()
        pool.map(POSTPROCESS, self.stations) #make our results with a map call
        pool.close() #we are not adding any more processes
        pool.join() #tell it to wait until all threads are done before going on
        print 'End of Post Processing of Receiver Functions ( Parallel ) !'
        return
    
    def PlotHSDataBase(self, datadir, outdir, prefix='QC_', ampfactor=40, targetDT=0.2, browseflag=True, saveflag=True,\
            obsflag=1, diffflag=0, repflag=1, rep0flag=1, rep1flag=1, rep2flag=1):
        for station in self.stations:
            print 'Plot Harmonic Stripping Results for: ',station.stacode;
            station.ReadHSDataBase(datadir=datadir, prefix=prefix);
            station.PlotHSDataBase(outdir=outdir, ampfactor=ampfactor, targetDT=targetDT, browseflag=browseflag, saveflag=saveflag,\
                obsflag=obsflag, diffflag=diffflag, repflag=repflag, rep0flag=rep0flag, rep1flag=rep1flag, rep2flag=rep2flag);
        print 'End of Plotting Harmonic Stripping Results!';
        return
        
    
    
    
    
    def GetTravelTimeFile(self, SLst, perlst, datadir, outdir, dirtin, \
                        minlon, maxlon, minlat, maxlat, tin='COR', dx=0.2, filetype='phase', chpair=['LHZ', 'LHZ'] ):
        """
        Generate Travel Time files for a given station(regarded as virtual source)/event.
        'surface' command in GMT 5.x.x is used to interpolate station points to grid points, with tension=0(minimum curvature).
        Input:
        SLst - station list( a StaLst object )
        per - period
        dx - dlon/dlat
        ----------------------------------------------------------------------------------
        Input format:
        datadir/dirtin/sta1/tin_sta1_chpair[0]_sta2_chpair[1].SAC*
        e.g. datadir/DISP/MONP/COR_MONP_LHZ_109C_LHZ.SAC_2_DISP.1 AND datadir/DISP/MONP/COR_MONP_LHZ_109C_LHZ.SAC_amp_snr
        
        Output format:
        outdir/travel_time_stacode.filetype.c.txt - travel time file
        outdir/sta1_TravelT.lst - travel time file for SNR>0 station points
        outdir/travel_time_stacode.filetype.c.txt.HD - interpolated travel time file 
        ----------------------------------------------------------------------------------
        """
        for per in perlst:
            outdirin=outdir+'/'+str(int(per))+'sec';
            if not os.path.isdir(outdirin):
                os.makedirs(outdirin)
            for station in self.stations:
                station.GetTravelTimeFile(SLst=SLst, per=per, datadir=datadir, outdir=outdirin, dirtin=dirtin,\
                    minlon=minlon, maxlon=maxlon, minlat=minlat, maxlat=maxlat, tin=tin, dx=dx, filetype=filetype, chpair=chpair );
        print 'End of Getting Travel Time Maps for Eikonal/Helmhotz Tomography!'
        return;
    
    def CheckTravelTimeCurvature(self, perlst, outdir, minlon, maxlon, minlat, maxlat, dx=0.2, filetype='phase'):
        """
        Check travel time curvature and discard those points with large curvatures.
        Station points at boundary will be discarded.
        Two interpolation schemes with different tension (0, 0.2) will be applied to the quality controlled travel time file. 
        ----------------------------------------------------------------------------------
        Input format:
        outdir/travel_time_stacode.filetype.c.txt.HD - interpolated travel time file 
        
        Output format:
        outdir/travel_time_stacode.filetype.c.txt_v1 - travel time file with flags( to label curvature ), points at boundaries are discarded.
        outdir/sta1_TravelT_v1.lst - travel time file for SNR>0 station points
        outdir/travel_time_stacode.filetype.c.txt_v1.HD - interpolated travel time file 
        outdir/travel_time_stacode.filetype.c.txt.HD_0.2 - interpolated travel time file with tension=0.2
        ----------------------------------------------------------------------------------
        """
        npts_x=int((maxlon-minlon)/dx)+1
        npts_y=int((maxlat-minlat)/dx)+1
        for per in perlst:
            outdirin=outdir+'/'+str(int(per))+'sec';
            if not os.path.isdir(outdirin):
                os.makedirs(outdirin)
            for station in self.stations:
                station.CheckTravelTimeCurvature(per=per, outdir=outdirin,\
                                        minlon=minlon, npts_x=npts_x, minlat=minlat, npts_y=npts_y, dx=dx, filetype=filetype );
        print 'End of Checking Travel Time Curvature!'
        return;

    def GetTravelTimeFileParallel(self, SLst, perlst, datadir, outdir, dirtin, \
                        minlon, maxlon, minlat, maxlat, tin='COR', dx=0.2, filetype='phase', chpair=['LHZ', 'LHZ'] ):
        """
        Parallel version of GetTravelTimeFile
        """
        for per in perlst:
            outdirin=outdir+'/'+str(int(per))+'sec';
            if not os.path.isdir(outdirin):
                os.makedirs(outdirin)
            TRAVELTIME=partial(StationsTravelTime, SLst=SLst, per=per, datadir=datadir, outdir=outdirin, dirtin=dirtin,\
                    minlon=minlon, maxlon=maxlon, minlat=minlat, maxlat=maxlat, tin=tin, dx=dx, filetype=filetype, chpair=chpair);
            pool = mp.Pool()
            pool.map(TRAVELTIME, self.stations) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        print 'End of Getting Travel Time Maps for Eikonal/Helmhotz Tomography ( Parallel ) !'
        return
    
    def CheckTravelTimeCurvatureParallel(self, perlst, outdir, minlon, maxlon, minlat, maxlat, dx=0.2, filetype='phase'):
        """
        Parallel version of CheckTravelTimeCurvature
        """
        npts_x=int((maxlon-minlon)/dx)+1;
        npts_y=int((maxlat-minlat)/dx)+1;
        for per in perlst:
            outdirin=outdir+'/'+str(int(per))+'sec';
            if not os.path.isdir(outdirin):
                os.makedirs(outdirin);
            CHECKCURVATURE=partial(StationsCheckTimeCurvature, per=per, outdir=outdirin,\
                                        minlon=minlon, npts_x=npts_x, minlat=minlat, npts_y=npts_y, dx=dx, filetype=filetype);
            pool = mp.Pool()
            pool.map(CHECKCURVATURE, self.stations) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        print 'End of Checking Travel Time Curvature ( Parallel ) !'
        return;
    
    def TravelTime2Slowness(self, datadir, outdir, perlst, minlon, maxlon, minlat, maxlat, dx=0.2, filetype='phase' ):
        """
        Generate Slowness Maps from Travel Time Maps.
        Two interpolated travel time file with different tension will be used for quality control.
        ----------------------------------------------------------------------------------
        Input format:
        datadir/travel_time_stacode.pflag.txt_v1.HD - interpolated travel time file (tension=0)
        datadir/travel_time_stacode.pflag.txt_v1.HD_0.2 - interpolated travel time file (tension=0.2)
        
        Output format:
        outdir/slow_azi_stacode.pflag.txt.HD.2.v2 - Slowness map
        ----------------------------------------------------------------------------------
        """
        npts_x=int((maxlon-minlon)/dx)+1
        npts_y=int((maxlat-minlat)/dx)+1
        for per in perlst:
            outdirin=outdir+'/'+str(int(per))+'sec';
            datadirin=datadir+'/'+str(int(per))+'sec';
            if not os.path.isdir(outdirin):
                os.makedirs(outdirin);
            cdist=12.*per;
            for station in self.stations:
                station.TravelTime2Slowness(datadir=datadirin, outdir=outdirin, per=per, \
                            minlon=minlon, npts_x=npts_x, minlat=minlat, npts_y=npts_y, dx=dx, cdist=cdist, filetype=filetype );
        print 'End of Converting Travel Time to Slowness!'
        return;
        
    def TravelTime2SlownessParallel(self, datadir, outdir, perlst, minlon, maxlon, minlat, maxlat, dx=0.2, filetype='phase' ):
        """
        Parallel version of TravelTime2Slowness
        """
        npts_x=int((maxlon-minlon)/dx)+1
        npts_y=int((maxlat-minlat)/dx)+1
        for per in perlst:
            outdirin=outdir+'/'+str(int(per))+'sec';
            datadirin=datadir+'/'+str(int(per))+'sec';
            if not os.path.isdir(outdirin):
                os.makedirs(outdirin);
            cdist=12.*per;
            TRAVELT2SLOWNESS=partial(StationsTravelTime2Slowness, datadir=datadirin, outdir=outdirin, per=per, \
                            minlon=minlon, npts_x=npts_x, minlat=minlat, npts_y=npts_y, dx=dx, cdist=cdist, filetype=filetype );
            pool = mp.Pool()
            pool.map(TRAVELT2SLOWNESS, self.stations) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        print 'End of Converting Travel Time to Slowness (Parallel)!'
        return;
            
    def GeneratePrePhaseDISP(self, SLst, outdir, mapfile='./MAPS/smpkolya_phv'):
        """
        Generate Predicted Phase V Dispersion Curves for a StaLst.
        Input:
        SLst - StaLst object
        outdir - output directories
        mapfile - Phase V maps
        ------------------------------------------------------------
        Input format:
        prephaseEXE pathfname mapfile perlst staname
        
        Output format:
        outdirL(outdirR)/sta1.sta2.pre
        ------------------------------------------------------------
        """
        outdirL=outdir+'_L';
        outdirR=outdir+'_R';
        if not os.path.isdir(outdirL):
            os.makedirs(outdirL);
        if not os.path.isdir(outdirR):
            os.makedirs(outdirR);
        for station in self.stations:
            station.GeneratePrePhaseDISP(SLst=SLst, outdir=outdir, mapfile=mapfile);
        print 'End of Generating Predicted Phase V Dispersion Curves!'
        return
        
    def GeneratePrePhaseDISPParallel(self, SLst, outdir, mapfile='./MAPS/smpkolya_phv'):
        """
        Parallel Version of GeneratePrePhaseDISP
        """
        outdirL=outdir+'_L';
        outdirR=outdir+'_R';
        if not os.path.isdir(outdirL):
            os.makedirs(outdirL);
        if not os.path.isdir(outdirR):
            os.makedirs(outdirR);
        GENERATEPREPHASE=partial(Stations2PrePhase, SLst=SLst, outdir=outdir, mapfile=mapfile );
        pool = mp.Pool()
        pool.map(GENERATEPREPHASE, self.stations) #make our results with a map call
        pool.close() #we are not adding any more processes
        pool.join() #tell it to wait until all threads are done before going on
        print 'End of Generating Predicted Phase V Dispersion Curves (Parallel)!'
        return
   
    def getDISP(self, datadir, minlon, maxlon, minlat, maxlat, dlon, dlat, outdir, crifactor=1., dirPFX='', dirSFX='', fPRX='', fSFX='.phv'):
        """
        Get the dispersion curve from near geographycal nodes
        ------------------------------------------------------------
        Input format:
        datadir/dirPFX+ind+dirSFX/fPRX+ind+fSFX
        e.g. datadir/100_23/100_23.phv
        
        Output format:
        outdir/self.stacode+fSFX
        e.g. outdir/AK.AKBB.phv
        -------------------------------------------------------------
        """
        if not os.path.isdir(outdir):
            os.makedirs(outdir);
        for station in self.stations:
            station.getDISP(datadir=datadir, minlon=minlon, maxlon=maxlon, minlat=minlat, maxlat=maxlat,\
                dlon=dlon, dlat=dlat, outdir=outdir, crifactor=crifactor, dirPFX=dirPFX, dirSFX=dirSFX, fPRX=fPRX, fSFX=fSFX);
        return;
    
    def CheckDispersion(self, datadir, minlon, maxlon, minlat, maxlat, outdir=None, BrowseFlag=False, SaveFlag=True):
        """
        Compute predicted group velocity curve from phase velocity curve, and plot them.
        ---------------------------------------------------------------------------------
        Input format:
        datadir/stacode.phv AND datadir/stacode.grv
        Output format:
        outdir/stacode_DISP.ps
        ---------------------------------------------------------------------------------
        """
        for station in self.stations:
            station.CheckDispersion(datadir=datadir, minlon=minlon, maxlon=maxlon, minlat=minlat, maxlat=maxlat, outdir=outdir, BrowseFlag=BrowseFlag, SaveFlag=SaveFlag)
            plt.close('all');
        return
    
    def GetGridStaLst(self, network='LF', PRX='EA', SFX='S'):
        NewSLst=StaLst();
        for station in self.stations:
            if abs(station.lon-int(station.lon)) > 0.1 or abs(station.lat-int(station.lat))>0.1:
                continue;
            station.stacode=PRX+str(int(station.lon))+SFX+str(int(station.lat))
            station.network=network
            NewSLst.append(station);
        # print 'End of Converting SES3D seismograms to SAC files !'
        return NewSLst;
  
  
        
def Slowness2IsoAniMap(stafile, perlst, datadir, outdir, minlon, maxlon, minlat, maxlat, dx=0.2, pflag=1, crifactor=12, minazi=-180, maxazi=180, N_bin=20 ):
    """
    Generate Isotropic/Anisotropic phase velocity Maps from Slowness Maps.
    ----------------------------------------------------------------------------------
    Input format:
    datadir/{per}sec/slow_azi_stacode.pflag.txt.HD.2.v2 - Slowness map
    
    Output format:
    
    ----------------------------------------------------------------------------------
    References:
    Lin, Fan-Chi, Michael H. Ritzwoller, and Roel Snieder. "Eikonal tomography: surface wave tomography by phase front tracking across a regional broad-band seismic array."
        Geophysical Journal International 177.3 (2009): 1091-1110.
    """
    slow2isoani_exe='./Slowness2IsoAniMap';
    npts_x=int((maxlon-minlon)/dx)+1
    npts_y=int((maxlat-minlat)/dx)+1
    if not os.path.isfile(slow2isoani_exe):
        print "No Slowness2IsoAniMap executable! Please Compile the Slowness2IsoAniMap.C at first!";
        return
    for per in perlst:
        data_pre=datadir+'/'+str(int(per))+'sec/';
        out_pre=outdir+'/'+str(int(per))+'sec/';
        if not os.path.isdir(outdir+'/'+str(int(per))+'sec'):
            os.makedirs(outdir+'/'+str(int(per))+'sec');
        cridist=crifactor*per;
        call([slow2isoani_exe, stafile, str(minazi), str(maxazi), str(N_bin), \
            data_pre, out_pre, str(dx), str(minlon), str(npts_x),str(minlat),str(npts_y), str(pflag), str(cridist) ]);
    return
    
### Non-member functions for Station List manipulations ###
def StationsAddHSlowness(STA, datadir, prefix='', suffix=''):
    STA.addHSlowness(datadir=datadir, prefix=prefix, suffix=suffix)
    return

def StationsDecimate(STA, factor, datadir, outdir, prefix='', suffix=''):
    STA.Decimate(factor=factor, datadir=datadir, outdir=outdir, prefix=prefix, suffix=suffix)
    return

def StationsSODRF(STA, datadir, outdir='', RFType='R', netFlag=True, saveflag=True, tdel=5., \
              f0 = 2.5, niter=200, minderr=0.001, phase='P', tbeg=-10., tend=30., PostFlag=False, outLstFlag=True):
    STA.SODRF(datadir=datadir, outdir=outdir, RFType=RFType, netFlag=netFlag, saveflag=saveflag, tdel=tdel, \
              f0 = f0, niter=niter, minderr=minderr, phase=phase, tbeg=tbeg, tend=tend, PostFlag=PostFlag, outLstFlag=outLstFlag)
    return

def StationsPostProcess(STA, datadir, outdir, freeDeconvFlag=True, VR=80., tdiff=0.08, saveHStxtFlag=False):
    STA.init_RefDataBase();
    STA.ReadDeconvRef(datadir=datadir);
    STA.PostProcess(outdir=outdir, freeDeconvFlag=freeDeconvFlag, VR=VR, tdiff=tdiff, saveHStxtFlag=saveHStxtFlag);
    return;


def StationsTravelTime(STA, SLst, per, datadir, outdir, dirtin, \
                        minlon, maxlon, minlat, maxlat, tin='COR', dx=0.2, filetype='phase', chpair=['LHZ', 'LHZ'] ):
    STA.GetTravelTimeFile(SLst=SLst, per=per, datadir=datadir, outdir=outdir, dirtin=dirtin,\
                    minlon=minlon, maxlon=maxlon, minlat=minlat, maxlat=maxlat, tin=tin, dx=dx, filetype=filetype, chpair=chpair );
    return;
    
def StationsCheckTimeCurvature(STA, per, outdir, minlon, npts_x, minlat, npts_y, dx=0.2, filetype='phase'):
    STA.CheckTravelTimeCurvature( per=per, outdir=outdir,\
                minlon=minlon, npts_x=npts_x, minlat=minlat, npts_y=npts_y, dx=dx, filetype=filetype );
    return

def StationsTravelTime2Slowness(STA, datadir, outdir, per, minlon, npts_x, minlat, npts_y, dx=0.2, cdist=None, filetype='phase' ):
    STA.TravelTime2Slowness(datadir=datadir, outdir=outdir, per=per, \
                            minlon=minlon, npts_x=npts_x, minlat=minlat, npts_y=npts_y, dx=dx, cdist=cdist, filetype=filetype );
    return;

def Stations2PrePhase(STA, SLst, outdir, mapfile='./MAPS/smpkolya_phv'):
    STA.GeneratePrePhaseDISP(SLst=SLst, outdir=outdir, mapfile=mapfile);
    return;


### End of non-member functions for Station List manipulations ###

class staPair(object):
    """
    An object contains a station pair and several method for pair analysis.
        sta1, sta2: station 1 & 2
        chan: channels for analysis
        lon1, lat1: position for station 1
        lon2, lat2: position for station 2
        datadir: input data directory
        outdir: output data directory
        tin: input title, default is COR
        tin: output title, default is COR
        c3Param: an object to store input parameters for C3 computation

    """

    def __init__(self, sta1=StaInfo(), sta2=StaInfo()):
        self.station1=sta1
        self.station2=sta2
        self.dist=None
        self.az=None
        self.baz=None

    def init_C3Param(self):
        self.c3Param=C3Param()

    def set_PRE_PHASE(self, predir):
        phvelname=predir+'/'+self.station1.stacode+'.'+self.station2.stacode+'.pre'
        if os.path.isfile(phvelname):
            self.phvelname=phvelname

    def get_contents(self):
        pair=self.station1.stacode+'_'+self.station2.stacode
        print 'Station Pair: %15s' %(pair)
        self.station1.get_contents()
        self.station2.get_contents()

        return

    def staPair2C3( self, datadir, outdir, tin, tout, skipexiflag=True):
        """
        TO BE TESTED!
        """
        try:
            self.c3Param
        except:
            return
        sta1=self.station1.stacode
        sta2=self.station2.stacode
        c3staList=self.c3Param.c3staList
        chanR1=self.station1.chan
        chanR2=self.station2.chan
        chan=self.c3Param.chan # channel of virtual source
        Lwin=self.c3Param.Lwin
        tfactor=self.c3Param.tfactor
        method=self.c3Param.method
        Tmin=self.c3Param.Tmin
        Tmax=self.c3Param.Tmax
        sepflag=self.c3Param.sepflag
        # Initialization
        c3trList=[]
        c3fList=[]
        c3allneg=[]
        c3allpos=[]
        for chan1 in chanR1:
            for chan2 in chanR2:
                f12=datadir+'/'+tin+'/'+sta1+'/'+tin+'_'+sta1+'_'+chan1+'_'+sta2+'_'+chan2+'.SAC'
                c3f12=outdir+'/'+tout+'/'+sta1+'/'+tout+'_'+sta1+'_'+chan1+'_'+sta2+'_'+chan2+'.SAC'
                f11=datadir+'/'+tin+'/'+sta1+'/'+tin+'_'+sta1+'_'+chan1+'_'+sta1+'_'+chan1+'.SAC'
                f22=datadir+'/'+tin+'/'+sta2+'/'+tin+'_'+sta2+'_'+chan2+'_'+sta2+'_'+chan2+'.SAC'
                print f12
                try:
                    c3tr=obspy.core.read(f12)[0]
                    c3tr.data=np.zeros((int) (2*Lwin/c3tr.stats.delta+1))
                except:
                    if os.path.isfile(f11) and os.path.isfile(f22):
                        try:
                            c11=obspy.core.read(f11)[0]
                            c3tr=obspy.core.read(f22)[0]
                            c3tr.data=np.zeros((int) (2*Lwin/c3tr.stats.delta+1))
                            c3tr.stats.sac.kevnm=c11.stats.sac.kevnm
                            c3tr.stats.sac.evla=c11.stats.sac.evla
                            c3tr.stats.sac.evlo=c11.stats.sac.evlo
                            dist, az, baz=obsGeo.base.gps2dist_azimuth(c3tr.stats.sac.evla, c3tr.stats.sac.evlo, \
                                c3tr.stats.sac.stla, c3tr.stats.sac.stlo) # distance is in m
                            c3tr.stats.sac.dist=dist/1000.
                            c3tr.stats.sac.az=az
                            c3tr.stats.sac.baz=baz
                            c3tr.stats.channel=chan1+chan2
                        except TypeError:
                            return
                    else:
                        return
                c3trList.append(c3tr)
                c3fList.append(c3f12)
                c3allpos.append(np.zeros((len(chan), (int)(Lwin*2/c3tr.stats.delta)+1 )))
                c3allneg.append(np.zeros((len(chan), (int)(Lwin*2/c3tr.stats.delta)+1 )))

        if skipexiflag==True:
            returnflag=True
            for outf in c3fList:
                if not os.path.isfile(outf):
                    returnflag=False
                    break
            if returnflag==True:
                print 'Exist: C3 for: '+sta1+'_'+sta2
                return
        c3neg=copy.deepcopy(c3allneg)
        c3pos=copy.deepcopy(c3allneg)
        c3all=copy.deepcopy(c3allneg)
        dt=c3trList[0].stats.delta
        stackNo=0
        print 'Do C3 for: '+sta1+'_'+sta2
        for S in c3staList:
            if sta1==S or sta2==S:
                continue
            skipflag=False
            self.c3Param.fname1=[]
            self.c3Param.fname2=[]
            for ch1 in np.arange(len(chanR1)):
                if skipflag==True:
                    break
                for ch2 in np.arange(len(chanR2)):
                    chan1=chanR1[ch1]
                    chan2=chanR2[ch2]
                    listNo=ch1*3+ch2
                    for i in np.arange(len(chan)):
                        chanS=chan[i]
                        flag1=False
                        flag2=False
                        if sta1>S:
                            f1=datadir+'/'+tin+'/'+S+'/'+tin+'_'+S+'_'+chanS+'_'+sta1+'_'+chan1+'.SAC'
                            cor1=tin+'_'+S+'_'+chanS+'_'+sta1+'_'+chan1+'.SAC'
                        else:
                            f1=datadir+'/'+tin+'/'+sta1+'/'+tin+'_'+sta1+'_'+chan1+'_'+S+'_'+chanS+'.SAC'
                            cor1=tin+'_'+sta1+'_'+chan1+'_'+S+'_'+chanS+'.SAC'
                            flag1=True
                        if sta2>S:
                            f2=datadir+'/'+tin+'/'+S+'/'+tin+'_'+S+'_'+chanS+'_'+sta2+'_'+chan2+'.SAC'
                            cor2=tin+'_'+S+'_'+chanS+'_'+sta2+'_'+chan2+'.SAC'
                        else:
                            f2=datadir+'/'+tin+'/'+sta2+'/'+tin+'_'+sta2+'_'+chan2+'_'+S+'_'+chanS+'.SAC'
                            cor2=tin+'_'+sta2+'_'+chan2+'_'+S+'_'+chanS+'.SAC'
                            flag2=True

                        if (os.path.isfile(f1) and os.path.isfile(f2)):
                            c3neg[listNo][i],c3pos[listNo][i] , skipflag=Cor2C3 (f1, f2, sta1, sta2, dt, tfactor, Lwin, Tmin, Tmax, method, flag1, flag2)
                            if skipflag==True:
                                break
                            if np.isnan(c3neg[listNo][i].sum()) or np.isnan(c3pos[listNo][i].sum()) or abs(c3neg[listNo][i].max())>10e10 or abs(c3pos[listNo][i].max())> 10e10 :
                                print 'Warning! NaN output for: ' + sta1 +'_'+sta2 +' with '+S
                                skipflag=True
                                break
            if skipflag==False:
                for k in np.arange(len(chanR1)*len(chanR2)):
                    c3allneg[k]+=c3neg[k]
                    c3allpos[k]+=c3pos[k]
                stackNo+=1
        # End of loop over virtual sources
        if stackNo==0:
            return
        for i in np.arange(len(chanR1)*len(chanR2)):
            c3all[i]=c3allpos[i]+c3allneg[i]
            # SAC header
            c3tr=c3trList[i]
            c3f12=c3fList[i]
            c3tr.stats.npts=c3all[i][0].size
            beg=c3tr.stats.sac.b
            c3tr.stats.sac.b=-(c3all[i][0].size-1)/2*c3tr.stats.delta
            c3tr.stats.sac.e=-c3tr.stats.sac.b
            c3tr.stats.starttime=c3tr.stats.starttime+(c3tr.stats.sac.b-beg)
            c3tr.stats.sac.user0=stackNo
            if sepflag==True:
                for j in np.arange(len(chan)):
                    c3tr.data=c3all[i][j]
                    c3tr.write(c3f12+'_'+chan[j],format="SAC")
            c3tr.data=c3all[i].sum(axis=0)
            c3tr.write(c3f12,format="SAC")
        return

    def Rotation(self, datadir, outdir, tin, tout, CPR='BH'):
        """
        To be TESTED!!!
        """
        sta1=self.station1.stacode
        sta2=self.station2.stacode
        chan1=self.station1.chan
        chan2=self.station2.chan

        PI=math.pi
        fEE=datadir+"/"+tin+"/"+sta1+"/"+tin+"_"+sta1+"_"+chan1[0]+"_"+sta2+"_"+chan2[0]+".SAC"
        fNN=datadir+"/"+tin+"/"+sta1+"/"+tin+"_"+sta1+"_"+chan1[1]+"_"+sta2+"_"+chan2[1]+".SAC"
        fEN=datadir+"/"+tin+"/"+sta1+"/"+tin+"_"+sta1+"_"+chan1[0]+"_"+sta2+"_"+chan2[1]+".SAC"
        fNE=datadir+"/"+tin+"/"+sta1+"/"+tin+"_"+sta1+"_"+chan1[1]+"_"+sta2+"_"+chan2[0]+".SAC"
        if os.path.isfile(fEE) and os.path.isfile(fNN) and os.path.isfile(fEN) and os.path.isfile(fNE):
            print "Do Rotation(RT) for:"+sta1+" and "+sta2+" at dir:"+outdir
            with open(fEE, 'r') as f1:
                c3EE=sacio.SacIO(f1)
            with open(fNN,'r') as f2:
                c3NN=sacio.SacIO(f2)
            with open(fEN,'r') as f3:
                c3EN=sacio.SacIO(f3)
            with open(fNE,'r') as f4:
                c3NE=sacio.SacIO(f4)
            theta=c3EN.GetHvalue('az')
            psi=c3EN.GetHvalue('baz')
            Ctheta=math.cos(PI*theta/180.)
            Stheta=math.sin(PI*theta/180.)
            Cpsi=math.cos(PI*psi/180.)
            Spsi=math.sin(PI*psi/180.)
            tempTT=-Ctheta*Cpsi*c3EE.seis+Ctheta*Spsi*c3EN.seis - \
                Stheta*Spsi*c3NN.seis + Stheta*Cpsi*c3NE.seis

            tempRR=- Stheta*Spsi*c3EE.seis - Stheta*Cpsi*c3EN.seis \
                - Ctheta*Cpsi*c3NN.seis - Ctheta*Spsi*c3NE.seis

            tempTR=-Ctheta*Spsi*c3EE.seis - Ctheta*Cpsi*c3EN.seis  \
                + Stheta*Cpsi*c3NN.seis + Stheta*Spsi*c3NE.seis

            tempRT=-Stheta*Cpsi*c3EE.seis +Stheta*Spsi*c3EN.seis \
                + Ctheta*Spsi*c3NN.seis - Ctheta*Cpsi*c3NE.seis

            fTT=outdir+"/"+tout+"/"+sta1+"/"+tout+"_"+sta1+"_"+CPR+"T_"+sta2+"_"+CPR+"T.SAC"
            fRR=outdir+"/"+tout+"/"+sta1+"/"+tout+"_"+sta1+"_"+CPR+"R_"+sta2+"_"+CPR+"R.SAC"
            fTR=outdir+"/"+tout+"/"+sta1+"/"+tout+"_"+sta1+"_"+CPR+"T_"+sta2+"_"+CPR+"R.SAC"
            fRT=outdir+"/"+tout+"/"+sta1+"/"+tout+"_"+sta1+"_"+CPR+"R_"+sta2+"_"+CPR+"T.SAC"
            c3EE.seis=tempTT
            c3EE.SetHvalue('kcmpnm',CPR+'T'+CPR+'T')
            with open(fTT,'wb') as fout1:
                c3EE.WriteSacBinary(fout1)

            c3EE.seis=tempRR
            c3EE.SetHvalue('kcmpnm',CPR+'R'+CPR+'R')
            with open(fRR,'wb') as fout2:
                c3EE.WriteSacBinary(fout2)

            c3EE.seis=tempTR
            c3EE.SetHvalue('kcmpnm',CPR+'T'+CPR+'R')
            with open(fTR,'wb') as fout3:
                c3EE.WriteSacBinary(fout3)

            c3EE.seis=tempRT
            c3EE.SetHvalue('kcmpnm',CPR+'R'+CPR+'T')
            with open(fRT,'wb') as fout4:
                c3EE.WriteSacBinary(fout4)
        if len(chan1)==2 or len(chan2)==2:
            return
        fEZ=datadir+"/"+tin+"/"+sta1+"/"+tin+"_"+sta1+"_"+chan1[0]+"_"+sta2+"_"+chan2[2]+".SAC"
        fZE=datadir+"/"+tin+"/"+sta1+"/"+tin+"_"+sta1+"_"+chan1[2]+"_"+sta2+"_"+chan2[0]+".SAC"
        fNZ=datadir+"/"+tin+"/"+sta1+"/"+tin+"_"+sta1+"_"+chan1[1]+"_"+sta2+"_"+chan2[2]+".SAC"
        fZN=datadir+"/"+tin+"/"+sta1+"/"+tin+"_"+sta1+"_"+chan1[2]+"_"+sta2+"_"+chan2[1]+".SAC"
        if os.path.isfile(fEZ) and os.path.isfile(fZE) and os.path.isfile(fNZ) and os.path.isfile(fZN):
            print "Do Rotation(RTZ) for:"+sta1+" and "+sta2+" at dir:" + outdir
            with open(fEZ,'r') as f5:
                c3EZ=sacio.SacIO(f5)
            with open(fZE,'r') as f6:
                c3ZE=sacio.SacIO(f6)
            with open(fNZ,'r') as f7:
                c3NZ=sacio.SacIO(f7)
            with open(fZN,'r') as f8:
                c3ZN=sacio.SacIO(f8)
            theta=c3EZ.GetHvalue('az')
            psi=c3EZ.GetHvalue('baz')
            Ctheta=math.cos(PI*theta/180.)
            Stheta=math.sin(PI*theta/180.)
            Cpsi=math.cos(PI*psi/180.)
            Spsi=math.sin(PI*psi/180.)
            tempRZ = Ctheta*c3NZ.seis + Stheta*c3EZ.seis

            tempZR = - Cpsi*c3ZN.seis -Spsi*c3ZE.seis

            tempTZ = -Stheta*c3NZ.seis + Ctheta*c3EZ.seis

            tempZT =  Spsi*c3ZN.seis - Cpsi*c3ZE.seis

            fRZ=outdir+"/"+tout+"/"+sta1+"/"+tout+"_"+sta1+"_"+CPR+"R_"+sta2+"_"+CPR+"Z.SAC"
            fZR=outdir+"/"+tout+"/"+sta1+"/"+tout+"_"+sta1+"_"+CPR+"Z_"+sta2+"_"+CPR+"R.SAC"
            fTZ=outdir+"/"+tout+"/"+sta1+"/"+tout+"_"+sta1+"_"+CPR+"T_"+sta2+"_"+CPR+"Z.SAC"
            fZT=outdir+"/"+tout+"/"+sta1+"/"+tout+"_"+sta1+"_"+CPR+"Z_"+sta2+"_"+CPR+"T.SAC"
            c3EZ.seis=tempRZ
            c3EZ.SetHvalue('kcmpnm',CPR+'R'+CPR+'Z')
            with open(fRZ,'wb') as fout5:
                c3EZ.WriteSacBinary(fout5)

            c3EZ.seis=tempZR
            c3EZ.SetHvalue('kcmpnm',CPR+'Z'+CPR+'R')
            with open(fZR,'wb') as fout6:
                c3EZ.WriteSacBinary(fout6)

            c3EZ.seis=tempTZ
            c3EZ.SetHvalue('kcmpnm',CPR+'T'+CPR+'Z')
            with open(fTZ,'wb') as fout7:
                c3EZ.WriteSacBinary(fout7)

            c3EZ.seis=tempZT
            c3EZ.SetHvalue('kcmpnm',CPR+'Z'+CPR+'T')
            with open(fZT,'wb') as fout8:
                c3EZ.WriteSacBinary(fout8)
        return

    def Stacking(self, datadir, outdir, tin, tout, mlist):
        """
        TO BE TESTED
        """
        sta1=self.station1.stacode
        sta2=self.station2.stacode
        chan1=self.station1.chan
        chan2=self.station2.chan
        tempSac=[]
        allSac=[]
        iniFlag=True
        print 'Do Stacking for:'+sta1+'_'+sta2
        for mon in mlist:
            skipflag=False
            for ch1 in np.arange(len(chan1)):
                if skipflag==True:
                    break
                for ch2 in np.arange(len(chan2)):
                    fname=datadir+'/'+mon+'/'+tin+'/'+sta1+'/'+tin+'_'+sta1+'_'+chan1[ch1]+'_'+sta2+'_'+chan2[ch2]+'.SAC'
                    if not os.path.isfile(fname):
                        skipflag=True
                        break
                    try:
                        tr=obspy.core.read(fname)[0]
                    except TypeError:
                        print 'Error SAC File: '+fname
                        skipflag=True
                    if np.isnan(tr.data.sum()) or abs(tr.data.max())>10e10:
                        print 'Warning! NaN Monthly SAC for: ' + sta1 +'_'+sta2 +' Month: '+mon
                        skipflag=True
                        break
                    tempSac.append(tr)
            if len(tempSac)!=len(chan1)*len(chan2) or skipflag==True:
                tempSac=[]
                continue
            if iniFlag==True:
                allSac=tempSac
                iniFlag=False
            else:
                for trNo in np.arange(len(tempSac)):
                    mtr=tempSac[trNo]
                    allSac[trNo].data+=mtr.data
                    allSac[trNo].stats.sac.user0+=mtr.stats.sac.user0
            tempSac=[]

        if len(allSac)==len(chan1)*len(chan2):
            print 'Finished Stacking for:'+sta1+'_'+sta2
            for ch1 in np.arange(len(chan1)):
                for ch2 in np.arange(len(chan2)):
                    outfname=outdir+'/'+tout+'/'+sta1+'/'+tout+'_'+sta1+'_'+chan1[ch1]+'_'+sta2+'_'+chan2[ch2]+'.SAC'
                    stackedTr=allSac[ch1*3+ch2]
                    stackedTr.write(outfname,format='SAC')
        return

    def Makesym(self, datadir, outdir, tin, tout, dirtin, dirtout ):
        sta1=self.station1.stacode
        sta2=self.station2.stacode
        Chan1=self.station1.chan
        Chan2=self.station2.chan
        for chan1 in Chan1:
            for chan2 in Chan2:
                infname=datadir+"/"+dirtin+"/"+sta1+"/"+tin+"_"+sta1+"_"+chan1+"_"+sta2+"_"+chan2+".SAC"
                outfname=outdir+"/"+dirtout+"/"+sta1+"/"+tout+"_"+sta1+"_"+chan1+"_"+sta2+"_"+chan2+".SAC"
                if (os.path.isfile(infname)):
                    try:
                        tr=obspy.core.read(infname)[0]
                        ltr=noisetrace(tr.data,tr.stats)
                        ltr.makesym()
                        ltr.write(outfname, format="SAC")
                    except TypeError:
                        print 'Error Data for: '+infname
        return

    def CheckCOR(self, datadir, tin, allf, errorf):
        sta1=self.station1.stacode
        sta2=self.station2.stacode
        Chan1=self.station1.chan
        Chan2=self.station2.chan
        for chan1 in Chan1:
            for chan2 in Chan2:
                fname=datadir+"/"+tin+"/"+sta1+"/"+tin+"_"+sta1+"_"+chan1+"_"+sta2+"_"+chan2+".SAC"
                #print 'Checking Data for: ' +fname
                if (os.path.isfile(fname)):
                    tr=obspy.core.read(fname)[0]
#                    print 'Checking Data for: '+ sta1+'_'+sta2+' channel: '+chan1+'_'+chan2
                    allf.writelines('Checking Data for: %s_%s %s_%s \n' % ( sta1, sta2, chan1, chan2) )
                    if np.isnan(tr.data.sum()) or abs(tr.data.max() ) > 10e10 or abs(tr.data.min() )>10e10:
                        errorf.writelines('Error Data Detected: %s_%s %s_%s \n' % ( sta1, sta2, chan1, chan2) )
#                        print 'Error Data Detected: '+ sta1+'_'+sta2+' channel: '+chan1+'_'+chan2
        return

    def aftanSNR(self, datadir, outdir, tin, tout, dirtin, dirtout, inftan, crifactor=12.):
        """
        aftan and SNR analysis for a staPair 
        Input format:
        datadir/dirtin/tin_sta1_chan1_sta2_chan2.SAC OR datadir/dirtin/tin_sta1_sta2.SAC
        Output format:
        outdir/dirtout/tout_sta1_chan1_sta2_chan2.SAC OR outdir/dirtout/tout_sta1_sta2.SAC
        inftan - Input ftan parameters
        """
        sta1=self.station1.stacode
        sta2=self.station2.stacode
        Chan1=self.station1.chan
        Chan2=self.station2.chan
        try:
            self.phvelname
        except:
            print 'Error: No predicted phase V curve for:' +sta1+'_'+sta1
            return
        for chan1 in Chan1:
            for chan2 in Chan2:
                sacfname=datadir+"/"+dirtin+"/"+sta1+"/"+tin+"_"+sta1+"_"+chan1+"_"+sta2+"_"+chan2+".SAC";
                foutPR=outdir+"/"+dirtout+"/"+sta1+"/"+tout+"_"+sta1+"_"+chan1+"_"+sta2+"_"+chan2+".SAC";
                Fname=tin+"_"+sta1+"_"+chan1+"_"+sta2+"_"+chan2+".SAC"
                if not os.path.isfile(sacfname):
                    sacfname=datadir+"/"+dirtin+"/"+sta1+"/"+tin+"_"+sta1+"_"+sta2+".SAC";
                    foutPR=outdir+"/"+dirtout+"/"+sta1+"/"+tout+"_"+sta1+"_"+sta2+".SAC";
                    Fname=tin+"_"+sta1+"_"+sta2+".SAC";
                    if not os.path.isfile(sacfname):
                        print 'No input SAC for:' +Fname
                        continue
                try:
                    st=obspy.core.read(sacfname)[0]
                except TypeError:
                    print 'Error Data for: '+sacfname
                    continue
                if st.stats.sac.dist< crifactor*inftan.tmin:
                    print 'Inter-distance is too small: ',st.stats.sac.dist, ' ', sta1+" "+sta2
                    return;
                if np.isnan(st.data.sum()) or abs(st.data.max() ) > 10e10 or abs(st.data.min() )>10e10:
                        print 'Error Data Detected: '+ sta1+'_'+sta2+' channel: '+chan1+'_'+chan2
                print ' Do aftan for:' + Fname
                tr=noisetrace(st.data, st.stats)
                tr.aftan(pmf=inftan.pmf, piover4=inftan.piover4, vmin=inftan.vmin, vmax=inftan.vmax, tmin=inftan.tmin, \
                    tmax=inftan.tmax, tresh=inftan.tresh, ffact=inftan.ffact, taperl=inftan.taperl, snr=inftan.snr, fmatch=inftan.fmatch, phvelname=self.phvelname)
                if np.isnan(tr.ftanparam.arr1_1.sum()) or np.isnan(tr.ftanparam.arr2_1.sum()) or np.isnan(tr.ftanparam.arr1_2.sum()) or np.isnan(tr.ftanparam.arr2_2[:5].sum()):
                    continue
                
                tr.ftanparam.writeDISP(foutPR)
                if inftan.dosnrflag==True:
                    tr.getSNR(foutPR=foutPR, fhlen=inftan.fhlen, pmf=inftan.pmf, piover4=inftan.piover4, vmin=inftan.vmin, vmax=inftan.vmax, tmin=inftan.tmin, \
                        tmax=inftan.tmax, tresh=inftan.tresh, ffact=inftan.ffact, taperl=inftan.taperl, snr=inftan.snr, fmatch=inftan.fmatch, phvelname=self.phvelname)
                    tr.makesym()
                    tr.getSNR(fhlen=inftan.fhlen)
                    tr.SNRParam.writeAMPSNR(foutPR)
        return

    def CHStartTime(self, datadir, outdir, tin, tout, dirtin, dirtout, outST=obspy.core.utcdatetime.UTCDateTime(0)):
        sta1=self.station1.stacode
        sta2=self.station2.stacode
        Chan1=self.station1.chan
        Chan2=self.station2.chan
        for chan1 in Chan1:
            for chan2 in Chan2:
                infname=datadir+"/"+dirtin+"/"+sta1+"/"+tin+"_"+sta1+"_"+chan1+"_"+sta2+"_"+chan2+".SAC"
                outfname=outdir+"/"+dirtout+"/"+sta1+"/"+tout+"_"+sta1+"_"+chan1+"_"+sta2+"_"+chan2+".SAC"

                if (os.path.isfile(infname)):
                    tr=obspy.core.read(infname)[0]
                    tr.stats.starttime=outST+tr.stats.sac.b
#                try:
                    tr.write(outfname, format='SAC')
#                except:
#                    print 'Unable to Save File: '+ outfname
        return

    def CheckExistence(self, outfile1, dir1, t1, dirt1, outfile2=None, dir2=None, t2=None, dirt2=None, outfileall=None):
        sta1=self.station1.stacode
        sta2=self.station2.stacode
        Chan1=self.station1.chan
        Chan2=self.station2.chan
        CheckFlag=2
        if outfile2==None or dir2==None or t2 == None or dirt2==None or outfileall==None:
            CheckFlag=1
        for chan1 in Chan1:
            for chan2 in Chan2:
                sacf1=dir1+"/"+dirt1+"/"+sta1+"/"+t1+"_"+sta1+"_"+chan1+"_"+sta2+"_"+chan2+".SAC"
                if CheckFlag==1:
                    if not os.path.isfile(sacf1):
                        outfile1.writelines('%s \n' %(sacf1))
                else:
                    sacf2=dir2+"/"+dirt2+"/"+sta1+"/"+t2+"_"+sta1+"_"+chan1+"_"+sta2+"_"+chan2+".SAC"
                    if (not os.path.isfile(sacf1) ) and os.path.isfile(sacf2):
                        outfile1.writelines('%s  %s \n' %(sacf1, sacf2))
                    elif (os.path.isfile(sacf1) ) and (not os.path.isfile(sacf2) ):
                        outfile2.writelines('%s  %s \n' %(sacf2, sacf1))
                    elif (not os.path.isfile(sacf1) ) and (not os.path.isfile(sacf2) ):
                        outfileall.writelines('%s  %s \n' %(sacf1, sacf2))
        return

    def getDistAzBaz(self):
        if self.station1.lat==self.station2.lat and self.station1.lon == self.station2.lon:
            print 'Error: Common locations for event and station!'
            return
        dist, az, baz=obsGeo.base.gps2dist_azimuth(self.station1.lat, self.station1.lon, self.station2.lat, self.station2.lon ) # distance is in m
        self.az=az
        self.baz=baz
        self.dist=dist/1000.
        return

    def getTomoInput(self, per, datadir, tin, dirtin, chpair=['LHZ', 'LHZ']):
        sta1=self.station1.stacode
        sta2=self.station2.stacode
        sacfname=datadir+"/"+dirtin+"/"+sta1+"/"+tin+"_"+sta1+"_"+chpair[0]+"_"+sta2+"_"+chpair[1]+".SAC"
        fDISP=noisefile(sacfname+'_2_DISP.1', 'DISP')
        fsnr=noisefile(sacfname+'_sym_amp_snr','SNR') ### 
#        print sta1+'_'+sta2
        if not os.path.isfile(fDISP.fname) or not os.path.isfile(fsnr.fname):
            return -1, -1, -1, -1, -1
        (pvel,gvel) = fDISP.get_phvel(per);
        (snr,signal1,noise1) = fsnr.get_snr(per);
        return pvel, gvel, snr, signal1,noise1

class StaPairLst(object):
    """
    An object contains a staPair list information and several methods for staPair list related analysis.
        stapairs: list of staPair
    """
    def __init__(self,stapairs=None):
        self.stapairs=[]
        if isinstance(stapairs, staPair):
            stapairs = [stapairs]
        if stapairs:
            self.stapairs.extend(stapairs)

    def __add__(self, other):
        """
        Add two StaPairLst with self += other.
        """
        if isinstance(other, staPair):
            other = StaLst([other])
        if not isinstance(other, StaLst):
            raise TypeError
        stapairs = self.stapairs + other.stapairs
        return self.__class__(stapairs=stapairs)

    def __len__(self):
        """
        Return the number of staPair in the StaPairLst object.
        """
        return len(self.stapairs)

    def __getitem__(self, index):
        """

        """
        if isinstance(index, slice):
            return self.__class__(stapairs=self.stapairs.__getitem__(index))
        else:
            return self.stapairs.__getitem__(index)

    def append(self, stapair):
        """
        Append a single staPair object to the current StaLst object.
        """
        if isinstance(stapair, staPair):
            self.stapairs.append(stapair)
        else:
            msg = 'Append only supports a single staPair object as an argument.'
            raise TypeError(msg)
        return self

    def GenerateC3Lst(self, staLst, chan=['BHZ'], chanS=['BHZ'],
        tin='COR', tout='C3', tfactor=2, Lwin=1200, Tmin=5.0, Tmax=10.0, method = 'stehly', sepflag=True):

        for sta1 in staLst:
            for sta2 in staLst:
                if sta2.stacode<=sta1.stacode:
                    continue
                tempPair=staPair(sta1=sta1, sta2=sta2)
                tempPair.station1.setChan(chan)
                tempPair.station2.setChan(chan)
                # Set parameters specified for C3 computation
                tempPair.init_C3Param()
                tempPair.c3Param.chan=chanS
                tempPair.c3Param.tfactor=tfactor
                tempPair.c3Param.Lwin=Lwin
                tempPair.c3Param.Tmin=Tmin
                tempPair.c3Param.Tmax=Tmax
                tempPair.c3Param.method=method
                tempPair.c3Param.sepflag=sepflag
                StaS=[]
                for S in staLst:
                    if sta1.stacode==S.stacode or sta2.stacode==S.stacode:
                        continue
                    StaS.append(S.stacode)
                tempPair.c3Param.c3staList=StaS
                self.append(tempPair)
        return

    def cc2c3(self, datadir, outdir, tin, tout, skipexiflag=True):
        for stapair in self.stapairs:
            stapair.staPair2C3( datadir=datadir, outdir=outdir, tin=tin, tout=tout, skipexiflag=skipexiflag)
        print 'End of C3 Computation!'
        return

    def cc2c3Parallel(self, datadir, outdir, tin, tout, skipexiflag=True):
        CC2C3 = partial(Spair2C3, datadir=datadir, outdir=outdir, tin=tin, tout=tout, skipexiflag=skipexiflag)
        pool =mp.Pool()
        pool.map(CC2C3, self.stapairs) #make our results with a map call
        pool.close() #we are not adding any more processes
        pool.join() #tell it to wait until all threads are done before going on
        print 'End of C3 Computation ( Parallel ) !'
        return

    def GenerateSPairLst(self, staLst, stationflag=0, chanAll=[['BHZ']] ):
        """
        Input:
        staLst: Station List  (a StaLst object)
        stationflag:
            0: sta1 < sta2; 1: sta1 <= sta2; 2: any
        chanAll: Channel List ( MUST be a nested list)
        example 1: [['BHE', 'BHN', 'BHZ'], ['BH1', 'BH2', 'BHZ'], ['LHE', 'LHN', 'LHZ']]
        example 2: [ ['LHE', 'LHN', 'LHZ'] ]
        example 3: [ ['LHZ'] ]
        """
        for sta1 in staLst:
            for sta2 in staLst:
                if sta2.stacode<=sta1.stacode and stationflag==0:
                    continue
                if sta2.stacode<sta1.stacode and stationflag==1:
                    continue
                for chan in chanAll:
                    tempPair=staPair(sta1=sta1, sta2=sta2)
                    tempPair.station1.setChan(chan)
                    tempPair.station2.setChan(chan)
                    self.append(tempPair)
        return

    def Stacking(self, datadir, outdir, tin, tout, Mbeg, Mend):
        mlist=GeMonLst( Mbeg, Mend)
        for stapair in self.stapairs:
            stapair.Stacking( datadir=datadir, outdir=outdir, tin=tin, tout=tout, mlist=mlist)
        print 'End of Stacking!'
        return

    def StackingParallel(self, datadir, outdir, tin, tout, Mbeg, Mend):
        mlist=GeMonLst( Mbeg, Mend)
        STACK = partial(SpairStacking, datadir=datadir, outdir=outdir, tin=tin, tout=tout, mlist=mlist)
        pool =mp.Pool()
        pool.map(STACK, self.stapairs) #make our results with a map call
        pool.close() #we are not adding any more processes
        pool.join() #tell it to wait until all threads are done before going on
        print 'End of Stacking  ( Parallel ) !'
        return

    def Rotation(self, datadir, outdir, tin, tout, CPR='BH'):
        for stapair in self.stapairs:
            stapair.Rotation( datadir=datadir, outdir=outdir, tin=tin, tout=tout, CPR=CPR)
        print 'End of Rotation!'
        return

    def RotationParallel(self, datadir, outdir, tin, tout, CPR='BH'):
        ROTATION = partial(SpairRotation, datadir=datadir, outdir=outdir, tin=tin, tout=tout, CPR=CPR)
        pool =mp.Pool()
        pool.map(ROTATION, self.stapairs) #make our results with a map call
        pool.close() #we are not adding any more processes
        pool.join() #tell it to wait until all threads are done before going on
        print 'End of Rotation  ( Parallel ) !'
        return

    def Makesym(self, datadir, outdir, tin, tout, dirtin, dirtout ):
        for stapair in self.stapairs:
            stapair.Makesym( datadir=datadir, outdir=outdir, tin=tin, tout=tout, dirtin=dirtin, dirtout=dirtout)
        print 'End of Makesym!'
        return

    def MakesymParallel(self, datadir, outdir, tin, tout, dirtin, dirtout ):
        MAKESYM = partial(SpairMakesym, datadir=datadir, outdir=outdir, tin=tin, tout=tout, dirtin=dirtin, dirtout=dirtout)
        pool =mp.Pool()
        pool.map(MAKESYM, self.stapairs) #make our results with a map call
        pool.close() #we are not adding any more processes
        pool.join() #tell it to wait until all threads are done before going on
        print 'End of Makesym  ( Parallel ) !'
        return

    def CheckCOR(self, datadir, tin, allfname, errorfname ):
        allf=open(allfname,'wb')
        errorf=open(errorfname,'wb')
        for stapair in self.stapairs:
            stapair.CheckCOR( datadir=datadir, tin=tin, allf=allf, errorf=errorf)
        allf.close()
        errorf.close()
        print 'End of Checking Cross-Correlation!'
        return

    def set_PRE_PHASE(self, predir):
        """
        Set the directory for predicted phase V curves.
        """
        self.setPREFlag=True
        for stapair in self.stapairs:
            stapair.set_PRE_PHASE( predir=predir)
        print 'End of setting PRE_PHASE!'
        return

    def aftanSNR(self, datadir, outdir, tin, tout, dirtin, dirtout, inftan=InputFtanParam()):
        """
        aftan and SNR analysis for a staPair List
        ----------------------------------------------------------------------------------
        Input format:
        datadir/dirtin/tin_sta1_chan1_sta2_chan2.SAC OR datadir/dirtin/tin_sta1_sta2.SAC
        Output format:
        outdir/dirtout/tout_sta1_chan1_sta2_chan2.SAC OR outdir/dirtout/tout_sta1_sta2.SAC
        inftan - Input ftan parameters
        ----------------------------------------------------------------------------------
        """
        try:
            self.setPREFlag
        except:
            raise ValueError('Predicted phase V curve not set yet!')
        for stapair in self.stapairs:
            stapair.aftanSNR( datadir=datadir, outdir=outdir, tin=tin, tout=tout, dirtin=dirtin, dirtout=dirtout, inftan=inftan)
        print 'End of aftan and SNR analysis!'
        return

    def aftanSNRParallel(self, datadir, outdir, tin, tout, dirtin, dirtout, inftan=InputFtanParam()):
        """
        Parallel version of aftanSNR
        """
        try:
            self.setPREFlag
        except:
            raise ValueError('Predicted phase V curve not set yet!')
        AFTANSNR = partial(SpairaftanSNR, datadir=datadir, outdir=outdir, tin=tin, tout=tout, dirtin=dirtin, dirtout=dirtout, inftan=inftan)
        pool = mp.Pool()
        pool.map(AFTANSNR, self.stapairs) #make our results with a map call
        pool.close() #we are not adding any more processes
        pool.join() #tell it to wait until all threads are done before going on
        print 'End of aftan and SNR analysis  ( Parallel ) !'
        return

    def CHStartTime(self, datadir, outdir, tin, tout, dirtin, dirtout, outST=obspy.core.utcdatetime.UTCDateTime(0)):
        for stapair in self.stapairs:
            stapair.CHStartTime( datadir=datadir, outdir=outdir, tin=tin, tout=tout, dirtin=dirtin, dirtout=dirtout, outST=outST)
        print 'End of Changing StartTime!'
        return

    def CHStartTimeParallel(self, datadir, outdir, tin, tout, dirtin, dirtout, outST=obspy.core.utcdatetime.UTCDateTime(0)):
        CHANGETIME = partial(SpairChangeST, datadir=datadir, outdir=outdir, tin=tin, tout=tout, dirtin=dirtin, dirtout=dirtout, outST=outST)
        pool = mp.Pool()
        pool.map(CHANGETIME, self.stapairs) #make our results with a map call
        pool.close() #we are not adding any more processes
        pool.join() #tell it to wait until all threads are done before going on
        print 'End of aftan and SNR analysis  ( Parallel ) !'
        return

    def CheckExistence(self, outfname1, dir1, t1, dirt1, outfname2=None, dir2=None, t2=None, dirt2=None, outfnameall=None):
        outfile1=open(outfname1, 'wb')
        if outfname2==None or outfnameall==None or dir2==None or t2 == None or dirt2==None:
            outfile2=None
            outfileall=None
        else:
            outfile2=open(outfname2, 'wb')
            outfileall=open(outfnameall, 'wb')
        for stapair in self.stapairs:
            stapair.CheckExistence( outfile1=outfile1, dir1=dir1, t1=t1, dirt1=dirt1, outfile2=outfile2, dir2=dir2, t2=t2, dirt2=dirt2, outfileall=outfileall)
        outfile1.close()
        try:
            outfile2.close()
            outfileall.close()
            print 'Checking Type: 2'
        except:
            print 'Checking Type: 1'
        print 'End of Checking Existence!'
        return

    def getTomoInput(self, outdir, per_array, datadir, tin='COR', dirtin='COR', chpairList=[['BHZ', 'BHZ']], outPRE='tomo_in_' ):
        """
        Generate Input files for Surface Wave Tomography.
        outdir:
            Output directory
        per_array:
            Period array
        chpairList:
            List for channel pairs, default is [['BHZ', 'BHZ']] (only ZZ component)
        outPRE:
            Prefix for output files, default is 'tomo_in_'
        """
        for stapair in self.stapairs:
            stapair.getDistAzBaz()
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        start = time.time();
        for per in per_array:
            print 'Generating Tomo Input for period:', per
            for chpair in chpairList:
                f=open(outdir+'/'+outPRE+str(per)+'_'+chpair[0]+'_'+chpair[1]+'.lst', 'w')
                i=0
                for stapair in self.stapairs:
                    pvel, gvel, snr, signal1,noise1=stapair.getTomoInput(per, datadir=datadir, tin=tin, dirtin=dirtin, chpair=chpair)
                    if pvel < 0 or gvel < 0 or pvel>10 or gvel>10 or snr>10e10 or signal1>10e10 or noise1>10e10:
                        i = i+1;
                        continue;
                    if stapair.station1.lon<0:
                        stapair.station1.lon=stapair.station1.lon+360.
                    if stapair.station2.lon<0:
                        stapair.station2.lon=stapair.station2.lon+360.
                    f.writelines("%d %g %g %g %g %g %g %g %g %g %g %g %g %s %s \n" %(i, stapair.station1.lat, stapair.station1.lon, \
                        stapair.station2.lat, stapair.station2.lon, pvel, gvel, stapair.dist, stapair.az, stapair.baz, snr, signal1, noise1,\
                        stapair.station1.stacode, stapair.station2.stacode))
                    i=i+1
                    if (math.fmod(i,1000) == 0):
                        print i, stapair.station1.stacode, stapair.station2.stacode, time.time() - start;
                f.close()
        print 'End of Generating Tomography Input File!'
        return

    def getTomoInputParallel(self, outdir, per_array, datadir, tin='COR', dirtin='COR', chpairList=[['BHZ', 'BHZ']], outPRE='tomo_in_' ):
        """
        Parallel Version of getTomoInput
        """
        for stapair in self.stapairs:
            stapair.getDistAzBaz()
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        per_list=per_array.tolist()
        GETTOMOINPUT=partial(PerTomoInput, stapairs=self.stapairs, outdir=outdir, datadir=datadir, tin=tin, dirtin=dirtin, chpairList=chpairList, outPRE=outPRE )
        pool = mp.Pool()
        pool.map(GETTOMOINPUT, per_list) #make our results with a map call
        pool.close() #we are not adding any more processes
        pool.join() #tell it to wait until all threads are done before going on
        print 'End of Generating Tomography Input File  ( Parallel ) !'
        return

    def getMishaInput(self, outdir, per_array, datadir, tin='COR', dirtin='COR', chpairList=[['BHZ', 'BHZ']], outPRE='MISHA_in_'):
        """
        Generate Input files for Misha's Surface Wave Tomography code.
        outdir:
            Output directory
        per_array:
            Period array
        chpairList:
            List for channel pairs, default is [['BHZ', 'BHZ']] (only ZZ component)
        outPRE:
            Prefix for output files, default is 'MISHA_in_'
        -----------------------------------------------------------------------
        Input format:
        datadir/dirtin/tin_sta1_chpair[0]_sta2_chpair[1].SAC_2.DISP.1
        datadir/dirtin/tin_sta1_chpair[0]_sta2_chpair[1].SAC_amp_snr
        
        Output format:
        outdir/outPRE+per__chpair[0]_sta2_chpair[1]_ph.lst
        -----------------------------------------------------------------------
        """
        for stapair in self.stapairs:
            stapair.getDistAzBaz()
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        start = time.time();
        for per in per_array:
            print 'Generating Tomo Input for period:', per
            for chpair in chpairList:

                fname_ph=outdir+'/'+outPRE+'%g'%( per ) +'_'+chpair[0]+'_'+chpair[1]+'_ph.lst' %( per );
                fname_gr=outdir+'/'+outPRE+'%g'%( per ) +'_'+chpair[0]+'_'+chpair[1]+'_gr.lst' %( per );
                fph=open(fname_ph, 'w');
                fgr=open(fname_gr, 'w');
                i=-1;
                for stapair in self.stapairs:
                    pvel, gvel, snr, signal1,noise1=stapair.getTomoInput(per, datadir=datadir, tin=tin, dirtin=dirtin, chpair=chpair)
                    if pvel < 0 or gvel < 0 or pvel>10 or gvel>10 or snr >10e10 or signal1 >10e10 or noise1 >10e10:
                        continue;
                    if stapair.station1.lon<0:
                        stapair.station1.lon=stapair.station1.lon+360.
                    if stapair.station2.lon<0:
                        stapair.station2.lon=stapair.station2.lon+360.
                    i=i+1;
                    if snr < 15.:
                        continue;
                    if stapair.dist < 2.*per*3.5:
                        continue
                    fph.writelines("%d %g %g %g %g %g 1. %s %s 1 1 \n" %(i, stapair.station1.lat, stapair.station1.lon, \
                        stapair.station2.lat, stapair.station2.lon, pvel, stapair.station1.stacode, stapair.station2.stacode));
                    fgr.writelines("%d %g %g %g %g %g 1. %s %s 1 1 \n" %(i, stapair.station1.lat, stapair.station1.lon, \
                        stapair.station2.lat, stapair.station2.lon, gvel, stapair.station1.stacode, stapair.station2.stacode));
                    if (math.fmod(i,1000) == 0):
                        print i, stapair.station1.stacode, stapair.station2.stacode, time.time() - start;
                fph.close();
                fgr.close();
        print 'End of Generating Misha Tomography Input File!'
        return
    
    def getMishaInputParallel(self, outdir, per_array, datadir, tin='COR', dirtin='COR', chpairList=[['BHZ', 'BHZ']], outPRE='MISHA_in_'):
        """
        Parallel version of getMishaInput.
        """
        for stapair in self.stapairs:
            stapair.getDistAzBaz()
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        per_list=per_array.tolist()
        GETMISHAINPUT=partial(PerMishaInput, stapairs=self.stapairs, outdir=outdir, datadir=datadir, tin=tin, dirtin=dirtin, chpairList=chpairList, outPRE=outPRE )
        pool = mp.Pool()
        pool.map(GETMISHAINPUT, per_list) #make our results with a map call
        pool.close() #we are not adding any more processes
        pool.join() #tell it to wait until all threads are done before going on
        print 'End of Generating Misha Tomography Input File ( Parallel ) !'
        return



### Non-member functions for staPair List manipulation ###
def Spair2C3(Spair, datadir, outdir, tin, tout, skipexiflag=True ):
#    Spair.get_contents()
    Spair.staPair2C3( datadir=datadir, outdir=outdir, tin=tin, tout=tout, skipexiflag=skipexiflag)
    return

def Cor2C3 (fname1, fname2, sta1, sta2, dt=1.0, tfactor=2, Lwin=1200, Tmin=5.0,
    Tmax=10.0, method='stehly', flag1=False, flag2=False):
    """
    Computing C3 from Cross-Correlations
    fname1, fname2: input SAC file
    sta1, sta2: station 1 & 2
    S: station serves as vritual source
    dt: sampling interval
    tfactor: time factor, see "findcoda" for more details
    Lwin: window length( length of C3: -Lwin ~ Lwin)
    Tmin: minimum period
    Tmax: maximum period
    method: 'stehly' or 'ma', see "findcoda" for more details
    flag1, flag2, whether to reverse the cc record or not( S>sta1, flag1=True; S> sta2, flag2=True)
    """
    if sta1<sta2:
        try:
            cc1=obspy.core.read(fname1)[0]
            cc2=obspy.core.read(fname2)[0]
        except TypeError:
            return np.zeros((int)(Lwin*2/dt)+1), np.zeros((int)(Lwin*2/dt)+1), True
        if flag1==True:
            cc1.data=cc1.data[::-1]
        if flag2==True:
            cc2.data=cc2.data[::-1]
    else:
        try:
            cc2=obspy.core.read(fname1)[0]
            cc1=obspy.core.read(fname2)[0]
        except TypeError:
            return np.zeros((int)(Lwin*2/dt)+1), np.zeros((int)(Lwin*2/dt)+1), True
        if flag2==True:
            cc1.data=cc1.data[::-1]
        if flag1==True:
            cc2.data=cc2.data[::-1]
    end1=cc1.stats.sac.e
    start1=cc1.stats.starttime
    end2=cc2.stats.sac.e
    start2=cc2.stats.starttime
    if len(cc1.data) > 32000 or len(cc2.data) >32000:
        cutL=(int)(16000.0*cc1.stats.delta)
        cc1=cc1.slice(start1+end1-cutL, start1+end1+cutL)
        cc1.stats.sac.b=-cutL
        cc1.stats.sac.e=cutL
        cc2=cc2.slice(start2+end2-cutL, start2+end2+cutL)
        cc2.stats.sac.b=-cutL
        cc2.stats.sac.e=cutL

    if abs(cc1.data.max())> 10e10 or abs(cc2.data.max()) > 10e10 or np.isnan(cc1.data.sum()) or np.isnan(cc2.data.sum()) or cc1.stats.sac.dist<Tmax*9 or cc2.stats.sac.dist<Tmax*9 :
        return np.zeros((int)(Lwin*2/dt)+1), np.zeros((int)(Lwin*2/dt)+1), True

    cc1=noisetrace(cc1.data, cc1.stats)
    cc2=noisetrace(cc2.data, cc2.stats)
    Tbeg1, Tend1=cc1.findcodaTime(tfactor=tfactor, Lwin=Lwin, Tmin=Tmin, Tmax=Tmax, method=method)
    Tbeg2, Tend2=cc2.findcodaTime(tfactor=tfactor, Lwin=Lwin, Tmin=Tmin, Tmax=Tmax, method=method)
    if Tend1==-1 or Tend2==-1:
        return np.zeros((int)(Lwin*2/dt)+1), np.zeros((int)(Lwin*2/dt)+1), True
    neg1, pos1, Tbeg, Tend=cc1.findcommoncoda(Tbeg1, Tbeg2, Tend1, Tend2)
    neg2, pos2, Tbeg, Tend=cc2.findcommoncoda(Tbeg1, Tbeg2, Tend1, Tend2)

    c3neg=scipy.signal.fftconvolve(neg1[::-1],neg2,"full")
    c3pos=scipy.signal.fftconvolve(pos1[::-1],pos2,"full")
    return c3neg, c3pos, False

def SpairStacking(Spair, datadir, outdir, tin, tout, mlist):
    Spair.Stacking( datadir=datadir, outdir=outdir, tin=tin, tout=tout, mlist=mlist)
    return

def SpairRotation(Spair, datadir, outdir, tin, tout, CPR='BH'):
    Spair.Rotation( datadir=datadir, outdir=outdir, tin=tin, tout=tout, CPR=CPR)
    return

def SpairMakesym(Spair, datadir, outdir, tin, tout, dirtin, dirtout):
    Spair.Makesym( datadir=datadir, outdir=outdir, tin=tin, tout=tout, dirtin=dirtin, dirtout=dirtout)
    return

def SpairaftanSNR(Spair, datadir, outdir, tin, tout, dirtin, dirtout, inftan=InputFtanParam()):
    Spair.aftanSNR( datadir=datadir, outdir=outdir, tin=tin, tout=tout, dirtin=dirtin, dirtout=dirtout, inftan=inftan)
    return

def SpairChangeST(Spair, datadir, outdir, tin, tout, dirtin, dirtout, outST=obspy.core.utcdatetime.UTCDateTime(0)):
    Spair.CHStartTime( datadir=datadir, outdir=outdir, tin=tin, tout=tout, dirtin=dirtin, dirtout=dirtout, outST=outST)
    return

def PerTomoInput(per, stapairs, outdir, datadir, tin, dirtin, chpairList=[['BHZ', 'BHZ']], outPRE='tomo_in_'):
    print 'Generating Tomo Input for period:', per
    for chpair in chpairList:
        f=open(outdir+'/'+outPRE+str(per)+'_'+chpair[0]+'_'+chpair[1]+'.lst', 'w')
        i=0
        for stapair in stapairs:
            pvel, gvel, snr, signal1,noise1=stapair.getTomoInput(per, datadir=datadir, tin=tin, dirtin=dirtin, chpair=chpair)
            if pvel < 0 or gvel < 0 or pvel>10 or gvel>10 or snr>10e10 or signal1>10e10 or noise1>10e10:
                i = i+1;
                continue;
            if stapair.station1.lon<0:
                stapair.station1.lon=stapair.station1.lon+360.
            if stapair.station2.lon<0:
                stapair.station2.lon=stapair.station2.lon+360.
            f.writelines("%d %g %g %g %g %g %g %g %g %g %g %g %g %s %s \n" %(i, stapair.station1.lat, stapair.station1.lon, \
                stapair.station2.lat, stapair.station2.lon, pvel, gvel, stapair.dist, stapair.az, stapair.baz, snr, signal1, noise1,\
                stapair.station1.stacode, stapair.station2.stacode))
            i=i+1
        f.close()
    return

def PerMishaInput(per, stapairs, outdir, datadir, tin='COR', dirtin='COR', chpairList=[['BHZ', 'BHZ']], outPRE='MISHA_in_'):
    print 'Generating Misha Tomo Input for period:', per
    for chpair in chpairList:
        fname_ph=outdir+'/'+outPRE+'%g'%( per ) +'_'+chpair[0]+'_'+chpair[1]+'_ph.lst';
        fname_gr=outdir+'/'+outPRE+'%g'%( per ) +'_'+chpair[0]+'_'+chpair[1]+'_gr.lst';
        fph=open(fname_ph, 'w');
        fgr=open(fname_gr, 'w');
        i=-1;
        for stapair in stapairs:
            pvel, gvel, snr, signal1,noise1=stapair.getTomoInput(per, datadir=datadir, tin=tin, dirtin=dirtin, chpair=chpair)
            if pvel < 0 or gvel < 0 or pvel>10 or gvel>10 or snr >10e10 or signal1 >10e10 or noise1 >10e10:
                continue;
            if stapair.station1.lon<0:
                stapair.station1.lon=stapair.station1.lon+360.
            if stapair.station2.lon<0:
                stapair.station2.lon=stapair.station2.lon+360.
            i=i+1;
            if snr < 15.:
                continue;
            if stapair.dist < 2.*per*3.5:
                continue
            fph.writelines("%d %g %g %g %g %g 1. %s %s 1 1 \n" %(i, stapair.station1.lat, stapair.station1.lon, \
                stapair.station2.lat, stapair.station2.lon, pvel, stapair.station1.stacode, stapair.station2.stacode));
            fgr.writelines("%d %g %g %g %g %g 1. %s %s 1 1 \n" %(i, stapair.station1.lat, stapair.station1.lon, \
                stapair.station2.lat, stapair.station2.lon, gvel, stapair.station1.stacode, stapair.station2.stacode));
        fph.close();
        fgr.close();
    return

### End of Non-member functions for staPair List manipulation ###

class noisefile(object):
    """
    An object for file manipunation.
        fname: file name
        format: file format
            
    """
    def __init__(self, fname='', format='SAC'):
        self.fname=fname
        self.format=format
        return

    def get_phvel(self, per):
        """
        Get phase & group velocity for a given period from a DISP file
        """
        ovel1 = -1.;
        ovel2 = -1.;
        fname = self.fname;
        if self.format !='DISP':
            print "Not DISP format",fname;
            return (ovel1,ovel2);

        if  not os.path.isfile(fname) :
            print "cannot find file ",fname;
            return (ovel1,ovel2);
        ap_1 = 0.;
        flag = 0;
        #### New Version
        Inarray=np.loadtxt(fname)
        a_per = Inarray[:,2];
        gv = Inarray[:,3];
        pv = Inarray[:,4];
        tempLper=a_per[a_per<per]
        tempUper=a_per[a_per>per]
        if len(tempLper)==0 or len(tempUper)==0:
            ovel1=-1.
            ovel2=-1.
        else:
            Lgv=gv[a_per<per][-1]
            Ugv=gv[a_per>per][0]
            Lpv=pv[a_per<per][-1]
            Upv=pv[a_per>per][0]
            Lper=tempLper[-1]
            Uper=tempUper[0]
            ovel1=(Upv - Lpv)/(Uper - Lper)*(per - Lper) + Lpv;
            ovel2=(Ugv - Lgv)/(Uper - Lper)*(per - Lper) + Lgv;
        #### Old Version
        # for l1 in open(fname):
        #     l1 = l1.rstrip();
        #     l2 = l1.split();
        #     ap_2 = float(l2[2]);
        #     gv_2 = float(l2[3]);
        #     pv_2 = float(l2[4]);
        #     if (ap_2 > per):
        #         if (ap_1==0.):
        #             return (ovel1,ovel2);
        #         ovel1 = (pv_2 - pv_1)/(ap_2 - ap_1)*(per - ap_1) + pv_1;
        #         ovel2 = (gv_2 - gv_1)/(ap_2 - ap_1)*(per - ap_1) + gv_1;
        #         flag = 1;
        #     ap_1 = ap_2;
        #     pv_1 = pv_2;
        #     gv_1 = gv_2;
        #     if (flag == 1):
        #         break;
        return (ovel1,ovel2);

    def get_snr (self, per):
        """
        Get SNR and amplitude for a given period from a SNR file
        """
        osnr = -1.;
        osig = -1.;
        onoise = -1.;
        flag = 0;
        ap_1 = 0.;
        fname=self.fname;
        if self.format !='SNR':
            print "Not SNR format",fname;
            return (osnr,osig,onoise);
        if (os.path.exists(fname) != True):
            print "cannot find file ",fname;
            return (osnr,osig,onoise);
        
        #### New Version
        Inarray=np.loadtxt(fname)
        a_per = Inarray[:,0];
        sig = Inarray[:,1];
        snr = Inarray[:,2];
        tempLper=a_per[a_per<per]
        tempUper=a_per[a_per>per]
        if len(tempLper)==0 or len(tempUper)==0:
            osnr=-1.
            osig=-1.
            onoise = -1.;
        else:
            Lsig=sig[a_per<per][-1]
            Usig=sig[a_per>per][0]
            Lsnr=snr[a_per<per][-1]
            Usnr=snr[a_per>per][0]
            Lper=tempLper[-1]
            Uper=tempUper[0]
            osig=(Usig - Lsig)/(Uper - Lper)*(per - Lper) + Lsig;
            osnr=(Usnr - Lsnr)/(Uper - Lper)*(per - Lper) + Lsnr;
            try:
                onoise = osig/osnr;
            except ZeroDivisionError:
                print 'Error SNR File:',fname, osig, osnr
                return (osnr,osig,onoise);
        return (osnr,osig,onoise);
        #### Old Version
        # for l1 in open(fname):
        #     l1 = l1.rstrip();
        #     l2 = l1.split();
        #     ap_2 = float(l2[0]); ## Apparent Period
        #     sig_2 = float(l2[1]);  ## Amplitude
        #     snr_2 = float(l2[2]); ## SNR
        #     try:
        #         noise_2 = sig_2/snr_2;
        #     except ZeroDivisionError:
        #         print 'Error SNR File:',fname, sig_2, snr_2
        #         return (osnr,osig,onoise);
        # 
        #     if (ap_2 > per):
        #         if (ap_1 == 0.):
        #             return (osnr,osig,onoise);
        #         osnr = (snr_2 - snr_1)/(ap_2 - ap_1)*(per - ap_1) + snr_1;
        #         osig = (sig_2 - sig_1)/(ap_2 - ap_1)*(per - ap_1) + sig_1;
        #         onoise = (noise_2 - noise_1)/(ap_2 - ap_1)*(per - ap_1) + noise_1;
        #         flag = 1;
        #     snr_1 = snr_2;
        #     sig_1 = sig_2;
        #     noise_1 = noise_2;
        #     ap_1 = ap_2;
        #     if (flag == 1):
        #         break;
        return (osnr,osig,onoise);

    def getStaLst(self):
        """
        Get a list of stations from a station file
        """
        stafile=self.fname
        if self.format !='SLT':
            print "Not SLT format",fname;
            return [];
        f = open(stafile, 'r')
        stalst=[]
        Sta=[]
        for lines in f.readlines():
            lines=lines.split()
            if Sta.__contains__(lines[0]):
                index=Sta.index(lines[0])
                if stalst[index][1]-float(lines[1]) or stalst[index][2]!=float(lines[2]):
                    print 'Error! Incompatible Station:' + lines[0]+' in Station List!'
                    return []
                else:
                    print 'Warning: Repeated Station:' +lines[0]+' in Station List!'
                    continue
            Sta.append(lines[0])
            stalst.append([lines[0],float(lines[1]),float(lines[2])])

        return stalst


### Other Non-member Functions
def FileComp(fname1, x1c, y1c, fname2, x2c, y2c,xLabel='Data x', yLabel='Data y'):
    """
    Compare data from two files
    """
    f1 = open(fname1, 'r')
    f2 = open(fname2, 'r')
    x1=np.array([])
    y1=np.array([])
    x2=np.array([])
    y2=np.array([])
    for lines in f1.readlines():
        lines=lines.split()
        x1=np.append(x1, float(lines[x1c-1]))
        y1=np.append(y1, float(lines[y1c-1]))
    for lines in f2.readlines():
        lines=lines.split()
        x2=np.append(x2, float(lines[x2c-1]))
        y2=np.append(y2, float(lines[y2c-1]))
    plb.figure()
    ax = plt.subplot()
    ax.plot(x1, y1, 'xk', lw=3) #
    ax.plot(x2, y2, 'ob', lw=3)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title('File Date Comparison')
    return

def FileDiff(fname1, x1c, y1c, fname2, x2c, y2c, xmin, xmax, npoints, diffFlag='percentage',\
    xLabel='Data x', yLabel='Data y', title='File Date Comparison'):
    """
    Generate the difference in data from two files
    """
    f1 = open(fname1, 'r')
    f2 = open(fname2, 'r')
    x1=np.array([])
    y1=np.array([])
    x2=np.array([])
    y2=np.array([])
    for lines in f1.readlines():
        lines=lines.split()
        x1=np.append(x1, float(lines[x1c-1]))
        y1=np.append(y1, float(lines[y1c-1]))
    for lines in f2.readlines():
        lines=lines.split()
        x2=np.append(x2, float(lines[x2c-1]))
        y2=np.append(y2, float(lines[y2c-1]))
    x=np.arange(npoints)*(xmax-xmin)/(npoints-1)+xmin
    yinterp1=np.interp(x, x1, y1)
    yinterp2=np.interp(x, x2, y2)
    if diffFlag=='percentage':
        ydiff_temp=(yinterp1-yinterp2)/yinterp2
        ydiff= (ydiff_temp <=10) *ydiff_temp+(ydiff_temp >10) *1
    else:
        ydiff= yinterp1-yinterp2
    ax = plt.subplot()
    ax.plot(x, ydiff, '--b', lw=1) #
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    return

def ReadStaionList(stafile):
    """
    Read station list from a station file
    """
    f = open(stafile, 'r')
    stalst=[]
    Sta=[]
    for lines in f.readlines():
        lines=lines.split()
        if Sta.__contains__(lines[0]):
            index=Sta.index(lines[0])
            if stalst[index][1]-float(lines[1]) or stalst[index][2]!=float(lines[2]):
                print 'Error! Incompatible Station:' + lines[0]+' in Station List!'
                return []
            else:
                print 'Warning: Repeated Station:' +lines[0]+' in Station List!'
                continue
        Sta.append(lines[0])
        stalst.append([lines[0],float(lines[1]),float(lines[2])])

    return stalst

def GeMonLst( Mbeg, Mend):
    """
    Generate Month List.
    """
    Month=['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    mlist=[]
    cMon=Mbeg
    while (True):
        yy=cMon[0]
        mm=Month[cMon[1]-1]
        mlist.append(str(yy)+'.'+mm)
        if cMon[1]==12:
            cMon[1]=1
            cMon[0]+=1
        else:
            cMon[1]+=1
        if cMon[0]==Mend[0] and cMon[1]==Mend[1]:
            yy=cMon[0]
            mm=Month[cMon[1]-1]
            mlist.append(str(yy)+'.'+mm)
            break
    return mlist

def PrintElaspedTime(tbeg, tend):
    elapsedtime=tend-tbeg
    ehour=int(elapsedtime/3600)
    emin=int((elapsedtime-ehour*3600)/60)
    esec=(elapsedtime-ehour*3600-emin*60)
    print 'Elapsed Time:',str(ehour)+' hours '+str( emin )+' mins '+str(esec)+' secs'
    return

def GetdxdyDataBase(dx=0.2):
    latLst=-90.+np.arange(180./dx+1)*dx
    midlon=0.
    dx_km=np.array([])
    dy_km=np.array([])
    for lat in latLst:
        if lat == -90:
            continue
        dist, az, baz=obsGeo.base.gps2dist_azimuth(lat, midlon, lat-dx, midlon )
        dy_km=np.append(dy_km, dist/1000.)
        dist, az, baz=obsGeo.base.gps2dist_azimuth(lat, midlon, lat, midlon+dx )
        dx_km=np.append(dx_km, dist/1000.)
        
    dist, az, baz=obsGeo.base.gps2dist_azimuth(-90., midlon, -90., midlon+dx )
    dx_km=np.append(dist/1000., dx_km)
    dx_km=np.append(dx_km, latLst);
    dx_km=dx_km.reshape(2,len(latLst))
    dx_km=dx_km.T
    np.savetxt('dx_km_%g.txt' %(dx), dx_km, fmt='%g')
    dy_km=np.append(dy_km[0], dy_km)
    dy_km=np.append(dy_km, latLst);
    dy_km=dy_km.reshape(2,len(latLst))
    dy_km=dy_km.T
    np.savetxt('dy_km_%g.txt' %(dx), dy_km, fmt='%g');
    return;

### Functions used as glue to run Misha's Tomography code
def RunMishaSmooth(per, datadir, outdir, minlon, maxlon, minlat, maxlat, \
        dlon=0.5, dlat=0.5, stepInte=0.2, lengthcell=1.0, datatype='ph', chpair=['BHZ', 'BHZ'], alpha=3000, sigma=500, beta=100, data_pre='MISHA_in_', outpre='N_INIT_'):
    """
    Run Misha's Tomography Code with large regularization parameters.
    This function is designed to do an inital test run, the output can be used to discard outliers in aftan results.
    
    IsoMishaexe - Path to Misha's Tomography code executable (isotropic version)
    contourfname - Path to contour file (see the manual for detailed description)
    ----------------------------------------------------------------------------
    Input format:
    datadir/data_pre+str(per)+'_'+chpair[0]+'_'+chpair[1]+'_'+datatype+'.lst'
    e.g. datadir/MISHA_in_20.0_BHZ_BHZ_ph.lst
    
    Output format:
    e.g. 
    Prefix: outdir/10.0_ph/N_INIT_3000_500_100
    output file: outdir/10.0_ph/N_INIT_3000_500_100_10.0.1 etc. (see the manual for detailed description of output suffix)
    ----------------------------------------------------------------------------
    References:
    Barmin, M. P., M. H. Ritzwoller, and A. L. Levshin. "A fast and reliable method for surface wave tomography."
        Monitoring the Comprehensive Nuclear-Test-Ban Treaty: Surface Waves. Birkhäuser Basel, 2001. 1351-1375.
    """
    IsoMishaexe='./TOMO_MISHA/itomo_sp_cu_shn';
    contourfname='./contour.ctr';
    if not os.path.isfile(IsoMishaexe):
        print 'IsoMishaexe does not exist!';
        return
    if not os.path.isfile(contourfname):
        print 'Contour file does not exist!';
        return;
    infname=datadir+'/'+data_pre+'%g'%( per ) +'_'+chpair[0]+'_'+chpair[1]+'_'+datatype+'.lst';
    outper=outdir+'/'+'%g'%( per ) +'_'+datatype;
    if not os.path.isdir(outper):
        os.makedirs(outper);
    outpre=outper+'/'+outpre+str(alpha)+'_'+str(sigma)+'_'+str(beta);
    temprunsh='temp_'+'%g_Smooth.sh' %(per);
    f=open(temprunsh,'wb')
    f.writelines('%s %s %s %g << EOF \n' %(IsoMishaexe, infname, outpre, per ));
    # if paraFlag==False:
    #     f.writelines('me \n' );
    f.writelines('me \n4 \n5 \n%g \n6 \n%g \n%g \n%g \n' %( beta, alpha, sigma, sigma) );
    f.writelines('7 \n%g %g %g \n8 \n%g %g %g \n12 \n%g \n%g \n16 \n' %(minlat, maxlat, dlat, minlon, maxlon, dlon, stepInte, lengthcell) );
    # if paraFlag==False:
    #     f.writelines('v \n' );
    f.writelines('v \nq \ngo \nEOF \n' );
    f.close();
    call(['bash', temprunsh]);
    # os.remove(temprunsh);
    return;

def RunMishaSmoothParallel(per_array, datadir, outdir, minlon, maxlon, minlat, maxlat, \
        dlon=0.5, dlat=0.5, stepInte=0.2, lengthcell=1.0, datatype='ph', chpair=['BHZ', 'BHZ'], alpha=3000, sigma=500, beta=100, data_pre='MISHA_in_', outpre='N_INIT_'):
    """
    Parallelly run Misha's Tomography Code with large regularization parameters for a period array.
    This function is designed to do an inital test run, the output can be used to discard outliers in aftan results.
    
    IsoMishaexe - Path to Misha's Tomography code executable (isotropic version)
    contourfname - Path to contour file (see the manual for detailed description)
    ----------------------------------------------------------------------------
    Input format:
    datadir/data_pre+per+'_'+chpair[0]+'_'+chpair[1]+'_'+datatype+'.lst'
    e.g. datadir/MISHA_in_20_BHZ_BHZ_ph.lst
    
    Output format:
    e.g. 
    Prefix: outdir/10_ph/N_INIT_3000_500_100
    output file: outdir/10_ph/N_INIT_3000_500_100_10.1 etc. (see the manual for detailed description of output suffix)
    ----------------------------------------------------------------------------
    References:
    Barmin, M. P., M. H. Ritzwoller, and A. L. Levshin. "A fast and reliable method for surface wave tomography."
        Monitoring the Comprehensive Nuclear-Test-Ban Treaty: Surface Waves. Birkhäuser Basel, 2001. 1351-1375.
    """
    IsoMishaexe='./TOMO_MISHA/itomo_sp_cu_shn';
    contourfname='./contour.ctr';
    if not os.path.isfile(IsoMishaexe):
        print 'IsoMishaexe does not exist!';
        return
    if not os.path.isfile(contourfname):
        print 'Contour file does not exist!';
        return;
    per_list=per_array.tolist();
    RUNMISHASMOOTH=partial(RunMishaSmooth, datadir=datadir, outdir=outdir, minlon=minlon, maxlon=maxlon, minlat=minlat, maxlat=maxlat,\
        dlon=dlon, dlat=dlat, stepInte=stepInte, lengthcell=lengthcell, datatype=datatype, chpair=chpair, alpha=alpha, sigma=sigma, beta=beta,\
        data_pre=data_pre, outpre=outpre); 
    pool = mp.Pool()
    pool.map(RUNMISHASMOOTH, per_list) #make our results with a map call
    pool.close() #we are not adding any more processes
    pool.join() #tell it to wait until all threads are done before going on
    print 'End of Running Smooth Misha Tomography ( Parallel ) !'
    return;

def RunMishaQC(per, isoFlag, datadir, outdir, minlon, maxlon, minlat, maxlat, \
        dlon=0.5, dlat=0.5, stepInte=0.1, lengthcell=0.5, datatype='ph', Wavetype='R',\
        alpha=850, sigma=175, beta=1, crifactor=0.5, crilimit=10., data_pre='N_INIT_', alpha1=3000, sigma1=500, beta1=100, outpre='QC_', \
        lengthcellAni=1.0, AniparaFlag=1, xZone=2, alphaAni0=1200, betaAni0=1, sigmaAni0=200, alphaAni1=1000, sigmaAni1=100):
    """
    Run Misha's Tomography Code with quality control based on preliminary run of RunMishaSmooth.
    This function is designed to discard outliers in aftan results (Quality Control), and then do tomography.
    
    Mishaexe - Path to Misha's Tomography code executable ( isotropic/anisotropic version, determined by isoFlag )
    contourfname - Path to contour file (see the manual for detailed description)
    ----------------------------------------------------------------------------
    Input format:
    datadir+'/'+per+'_'+datatype+'/'+data_pre+str(alpha1)+'_'+str(sigma1)+'_'+str(beta1)+'_'+per+'.resid'
    e.g. datadir/10_ph/N_INIT_3000_500_100_10.0.resid
    
    Intermediate output format:
    outdir+'/'+per+'_'+datatype+'/QC_'+per+'_'+Wavetype+'_'+datatype+'.lst'
    e.g. outdir/10_ph/QC_10_R_ph.lst
    
    Output format:
    e.g. 
    Prefix: outdir/10_ph/QC_850_175_1  OR outdir/10_ph/QC_AZI_R_1200_200_1000_100_1
    
    Output file:
    outdir/10_ph/QC_850_175_1_10.1 etc. 
    OR
    outdir/10_ph/QC_AZI_R_1200_200_1000_100_1_10.1 etc. (see the manual for detailed description of output suffix)
    ----------------------------------------------------------------------------
    References:
    Barmin, M. P., M. H. Ritzwoller, and A. L. Levshin. "A fast and reliable method for surface wave tomography."
        Monitoring the Comprehensive Nuclear-Test-Ban Treaty: Surface Waves. Birkhäuser Basel, 2001. 1351-1375.
    """
    if isoFlag==True:
        Mishaexe='./TOMO_MISHA/itomo_sp_cu_shn';
    else:
        Mishaexe='./TOMO_MISHA_AZI/tomo_sp_cu_s_shn-.1/tomo_sp_cu_s_shn_.1';
        outpre=outpre+'AZI_';
    contourfname='./contour.ctr';
    if not os.path.isfile(Mishaexe):
        print 'Mishaexe does not exist!';
        return
    if not os.path.isfile(contourfname):
        print 'Contour file does not exist!';
        return;
    infname=datadir+'/'+'%g'%( per ) +'_'+datatype+'/'+data_pre+str(alpha1)+'_'+str(sigma1)+'_'+str(beta1)+'_'+'%g'%( per )+'.resid' ;
    Inarray=np.loadtxt(infname);   
    res_tomo=Inarray[:,7];
    cri_res=crifactor*per;
    if cri_res>crilimit:
        cri_res=crilimit;
    QC_arr= Inarray[abs(res_tomo)<cri_res,:];
    outArray=QC_arr[:,:8];
    outper=outdir+'/'+'%g'%( per ) +'_'+datatype ;
    if not os.path.isdir(outper):
        os.makedirs(outper);
    QCfname=outper+'/QC_'+'%g'%( per ) +'_'+Wavetype+'_'+datatype+'.lst';
    np.savetxt(QCfname, outArray, fmt='%g');
    if isoFlag==True:
        outpre=outper+'/'+outpre+str(alpha)+'_'+str(sigma)+'_'+str(beta);
    else:
        outpre=outper+'/'+outpre+Wavetype+'_'+str(alphaAni0)+'_'+str(sigmaAni0)+'_'+str(alphaAni1)+'_'+str(sigmaAni1)+'_'+str(betaAni0);
    temprunsh='temp_'+'%g_QC.sh' %(per);
    f=open(temprunsh,'wb')
    f.writelines('%s %s %s %g << EOF \n' %(Mishaexe, QCfname, outpre,per ));
    if isoFlag==True:
        f.writelines('me \n4 \n5 \n%g \n6 \n%g \n%g \n%g \n' %( beta, alpha, sigma, sigma) );
        f.writelines('7 \n%g %g %g \n8 \n%g %g %g \n12 \n%g \n%g \n16 \n' %(minlat, maxlat, dlat, minlon, maxlon, dlon, stepInte, lengthcell) );
        f.writelines('v \nq \ngo \nEOF \n' );
    else:
        if datatype=='ph':
            Dtype='P'
        else:
            Dtype='G'
        f.writelines('me \n4 \n5 \n%g %g %g \n6 \n%g %g %g \n' %( minlat, maxlat, dlat, minlon, maxlon, dlon) );
        f.writelines('10 \n%g \n%g \n%s \n%s \n%g \n%g \n11 \n%d \n' %(stepInte, xZone, Wavetype, Dtype, lengthcell, lengthcellAni, AniparaFlag) );
        f.writelines('12 \n%g \n%g \n%g \n%g \n' %(alphaAni0, betaAni0, sigmaAni0, sigmaAni0) );
        f.writelines('13 \n%g \n%g \n%g \n' %(alphaAni1, sigmaAni1, sigmaAni1) );
        f.writelines('19 \n25 \n' );
        f.writelines('v \nq \ngo \nEOF \n' );
    f.close();
    call(['bash', temprunsh]);
    os.remove(temprunsh);
    return;

def RunMishaQCParallel(per_array, isoFlag, datadir, outdir, minlon, maxlon, minlat, maxlat, \
        dlon=0.5, dlat=0.5, stepInte=0.1, lengthcell=0.5, datatype='ph', Wavetype='R',\
        alpha=850, sigma=175, beta=1, crifactor=0.5, crilimit=10., data_pre='N_INIT_', alpha1=3000, sigma1=500, beta1=100, outpre='QC_', \
        lengthcellAni=1.0, AniparaFlag=1, xZone=2, alphaAni0=1200, betaAni0=1, sigmaAni0=200, alphaAni1=1000, sigmaAni1=100):
    """
    Parallelly run Misha's Tomography Code with quality control based on preliminary run of RunMishaSmooth for a period array.
    This function is designed to discard outliers in aftan results (Quality Control), and then do tomography.
    
    Mishaexe - Path to Misha's Tomography code executable ( isotropic/anisotropic version, determined by isoFlag )
    contourfname - Path to contour file (see the manual for detailed description)
    ----------------------------------------------------------------------------
    Input format:
    datadir+'/'+per+'_'+datatype+'/'+data_pre+str(alpha1)+'_'+str(sigma1)+'_'+str(beta1)+'_'+per+'.resid'
    e.g. datadir/10_ph/N_INIT_3000_500_100_10.resid
    
    Intermediate output format:
    outdir+'/'+per+'_'+datatype+'/QC_'+str(per)+'_'+Wavetype+'_'+datatype+'.lst'
    e.g. outdir/10_ph/QC_10_R_ph.lst
    
    Output format:
    e.g. 
    Prefix: outdir/10_ph/QC_850_175_1  OR outdir/10_ph/QC_AZI_R_1200_200_1000_100_1
    
    Output file:
    outdir/10_ph/QC_850_175_1_10.1 etc. 
    OR
    outdir/10_ph/QC_AZI_R_1200_200_1000_100_1_10.1 etc. (see the manual for detailed description of output suffix)
    ----------------------------------------------------------------------------
    References:
    Barmin, M. P., M. H. Ritzwoller, and A. L. Levshin. "A fast and reliable method for surface wave tomography."
        Monitoring the Comprehensive Nuclear-Test-Ban Treaty: Surface Waves. Birkhäuser Basel, 2001. 1351-1375.
    """
    per_list=per_array.tolist();
    RUNMISHAQC=partial(RunMishaQC, isoFlag=isoFlag, datadir=datadir, outdir=outdir, minlon=minlon, maxlon=maxlon, minlat=minlat, maxlat=maxlat,\
        dlon=dlon, dlat=dlat, stepInte=stepInte, lengthcell=lengthcell, datatype=datatype, Wavetype=Wavetype,\
        alpha=alpha, sigma=sigma, beta=beta, crifactor=crifactor, crilimit=crilimit, data_pre=data_pre, alpha1=alpha1, sigma1=sigma1, beta1=beta1, outpre=outpre, \
        lengthcellAni=lengthcellAni, AniparaFlag=AniparaFlag, xZone=xZone, alphaAni0=alphaAni0, betaAni0=betaAni0, sigmaAni0=sigmaAni0, alphaAni1=alphaAni1, sigmaAni1=sigmaAni1); 
    pool = mp.Pool()
    pool.map(RUNMISHAQC, per_list) #make our results with a map call
    pool.close() #we are not adding any more processes
    pool.join() #tell it to wait until all threads are done before going on
    print 'End of Running Quality Controlled Misha Tomography ( Parallel ) !'
    return;

def GetCorrectedMap(per, glbdir, regdir, outdir, reg_pre='QC_850_175_1_', glbpre='smpkolya_phv_R_', outpre='smpkolya_phv_R_'):
    """
    Get corrected global phave V map using a regional phase V map.
    ----------------------------------------------------------------
    Input format:
    glbdir/glbpre+per - global phase V map
    regdir/str(float(per))_ph/reg_pre+per.1 - e.g. regdir/40.0_ph/QC_850_175_1_40.1

    Output format:
    outdir/outpre+str(int(per))
    ----------------------------------------------------------------
    """
    inglobalfname=glbdir+'/'+glbpre+str(int(per));
    inregfname=regdir+'/'+'%g'%( per ) +'_ph'+'/'+reg_pre+str(float(per))+'.1' ;
    if not ( os.path.isfile(inglobalfname) and os.path.isfile(inregfname) ):
        inregfname=regdir+'/'+'%g'%( per ) +'_ph'+'/'+reg_pre+'%g.1' %(per);
        if not ( os.path.isfile(inglobalfname) and os.path.isfile(inregfname) ):
            print 'File not exists for period: ', per;
            print inglobalfname,inregfname
            return
    outfname=outdir+'/'+outpre+'%g' %(per);
    print inglobalfname, inregfname, outfname
    InregArr=np.loadtxt(inregfname);
    InglbArr=np.loadtxt(inglobalfname);
    outArr=InglbArr;
    (Lglb, m)=InglbArr.shape;
    (Lreg, m)=InregArr.shape;
    for i in np.arange(Lglb):
        lonG=InglbArr[i,0];
        latG=InglbArr[i,1];
        phVG=InglbArr[i,2];
        for j in np.arange(Lreg):
            lonR=InregArr[j,0];
            latR=InregArr[j,1];
            phVR=InregArr[j,2];
            if abs(lonR-lonG)<0.05 and abs(latR-latG)<0.05:
                phVG=phVR;
        outArr[i,2]=phVG
    np.savetxt(outfname, outArr, fmt='%g %g %.4f');
    return;
        
def GetCorrectedMapParallel(per_array, glbdir, regdir, outdir, reg_pre='QC_850_175_1_', glbpre='smpkolya_phv_R_', outpre='smpkolya_phv_R_'):
    """
    Get corrected global phave V map using a regional phase V map, do for a period array parallelly.
    ----------------------------------------------------------------
    Input format:
    glbdir/glbpre+per - global phase V map
    regdir/per_ph/reg_pre+per.1 - e.g. regdir/40_ph/QC_850_175_1_40.1

    Output format:
    outdir/outpre+per
    ----------------------------------------------------------------
    """
    if not os.path.isdir(outdir):
        os.makedirs(outdir);
    per_list=per_array.tolist();
    GETCMAP=partial(GetCorrectedMap, glbdir=glbdir, regdir=regdir, outdir=outdir, reg_pre=reg_pre, glbpre=glbpre, outpre=outpre); 
    pool = mp.Pool()
    pool.map(GETCMAP, per_list) #make our results with a map call
    pool.close() #we are not adding any more processes
    pool.join() #tell it to wait until all threads are done before going on
    print 'End of Get Corrected Global Phase V Maps( Parallel ) !'
    return;



    

