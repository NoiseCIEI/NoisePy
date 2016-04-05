#!/usr/bin/env python
import obspy
import obspy.signal.tf_misfit as obsTF
import obspy.signal
import pyaftan as ftan
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.pylab as plb
import os
import numexpr as npr
import fftw3
import spectrum
from functools import partial
import multiprocessing as mp

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
              arr1(9,:) -  amplitudes
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
              arr2(8,:) -  amplitudes
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
             arr1(9,:) -  amplitudes 
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
             arr2(8,:) -  amplitudes
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
            f=open(f10,'w')
            for i in np.arange(self.nfout1_1):
                tempstr='%4d %10.4lf %10.4lf %12.4lf %12.4lf %12.4lf %12.4lf %8.3lf  \n' %( i, self.arr1_1[0,i] , self.arr1_1[1,i] , self.arr1_1[2,i] , self.arr1_1[3,i]  \
                    , self.arr1_1[4,i] , self.arr1_1[5,i] , self.arr1_1[6,i] )
                f.writelines(tempstr)
            f.close()
        if self.nfout2_1!=0:
            f11=fnamePR+'_1_DISP.1'
            f=open(f11,'w')
            for i in np.arange(self.nfout2_1):
                tempstr='%4d %10.4lf %10.4lf %12.4lf %12.4lf %12.4lf %8.3lf  \n' %( i, self.arr2_1[0,i], self.arr2_1[1,i] , self.arr2_1[2,i] , self.arr2_1[3,i]  \
                    , self.arr2_1[4,i] , self.arr2_1[5,i]  )
                f.writelines(tempstr)
            f.close()
        if self.nfout1_2!=0:
            f20=fnamePR+'_2_DISP.0';
            f=open(f20,'w')
            for i in np.arange(self.nfout1_2):
                tempstr='%4d %10.4lf %10.4lf %12.4lf %12.4lf %12.4lf %12.4lf %8.3lf \n' %( i, self.arr1_2[0,i], self.arr1_2[1,i] , self.arr1_2[2,i] , self.arr1_2[3,i]  \
                    , self.arr1_2[4,i] , self.arr1_2[5,i] , self.arr1_2[6,i] )
                f.writelines(tempstr)
            f.close()
        if self.nfout2_2!=0:
            f21=fnamePR+'_2_DISP.1';
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
        return;

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

class symtrace(obspy.core.trace.Trace):
    """
    symtrace:
    A derived class inherited from obspy.core.trace.Trace. This derived class have a variety of new member functions
    """
    def init_ftanParam(self):
        """
        Initialize ftan parameters
        """
        self.ftanparam=ftanParam()
        return
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

    def aftan(self, pmf=True, piover4=-1.0, vmin=1.5, vmax=5.0, tmin=0.5, \
        tmax=100.0, tresh=20.0, ffact=1.0, taperl=1.0, snr=0.2, fmatch=1.0, phvelname='', predV=np.array([]) ):

        """ (Automatic Frequency-Time ANalysis) aftan analysis:
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
             Monitoring the Comprehensive Nuclear-Test-Ban Treaty: Surface Waves. Birkh?user Basel, 2001. 1531-1545.
        Bensen, G. D., et al. Processing seismic ambient noise data to obtain reliable broad-band surface wave dispersion measurements.
             Geophysical Journal International 169.3 (2007): 1239-1260.
        """
        try:
            self.ftanparam
        except:
            self.init_ftanParam()
        if (phvelname==''):
            phvelname='./ak135.disp';
        nprpv = 0 ;
        phprper=np.zeros(300)
        phprvel=np.zeros(300)
        
        if predV.size != 0:
            phprper=predV[:,0]
            phprvel=predV[:,1]
            nprpv = predV[:,0].size
            phprper=np.append(phprper,np.zeros(300-phprper.size))
            phprvel=np.append(phprvel,np.zeros(300-phprvel.size))
            self.ftanparam.preflag=True
        elif os.path.isfile(phvelname):
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
        tb=self.stats.sac.b;
        length=self.data.size;
        if length>32768:
            print "Warning: length of seismogram is larger than 32768!"
            nsam=32768
            self.data=self.data[:nsam]
            self.stats.e=(nsam-1)*self.stats.delta+tb
            sig=self.data
        else:
            sig=np.append(self.data, np.zeros( float(32768-self.data.size) ) )
            nsam=int( float (self.stats.npts) )### for unknown reasons, this has to be done, nsam=int(self.stats.npts)  won't work as an input for aftan
        dt=self.stats.delta
        dist=self.stats.sac.dist;
        # Start to do aftan utilizing pyaftan
        self.ftanparam.nfout1_1, self.ftanparam.arr1_1, self.ftanparam.nfout2_1, self.ftanparam.arr2_1, self.ftanparam.tamp_1,\
        self.ftanparam.nrow_1, self.ftanparam.ncol_1, self.ftanparam.ampo_1, self.ftanparam.ierr_1= ftan.aftanpg(piover4, nsam, \
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
        F=np.arange(Ns)*df
        gauamp = F - fcenter
        
        # alpha = -0.5/(fhlen*fhlen);
        # sf=npr.evaluate('exp(alpha*gauamp**2)')

        alpha = -20.*np.sqrt(self.stats.sac.dist/1000.0)
        sf=npr.evaluate('exp(alpha*(gauamp/fcenter)**2)')
        sf1=npr.evaluate('alpha*(gauamp/fcenter)**2')
        sf=sf*(np.abs(sf1)<40.);
        
        sp, Ns=FFTW(self.data, direction='forward')
        filtered_sp=npr.evaluate('sf*sp')
        filtered_seis, Ns=FFTW(filtered_sp, direction='backward')
        filtered_seis=filtered_seis[:npts].real
        return filtered_seis
    
    def GaussianFilterTaper(self, fcenter, ib, ie, fhlen=0.008):
        """
        Gaussian Filter designed for SNR analysis, utilize pyfftw to do fft
        exp( (-0.5/fhlen^2)*(f-fcenter)^2 )
        -----------------------------------------------------------------------------------------------------
        Input parameters:
        fcenter - central period
        fhlen   - half length of Gaussian width
        -----------------------------------------------------------------------------------------------------
        """
        tempdata=self.data[ib:ie];
        npts=tempdata.size;
        Ns=1<<(npts-1).bit_length()
        df=1.0/self.stats.delta/Ns
        nhalf=Ns/2+1
        fmax=(nhalf-1)*df
        if fcenter>fmax:
            fcenter=fmax
        F=np.arange(Ns)*df
        gauamp = F - fcenter
        
        # alpha = -0.5/(fhlen*fhlen);
        # sf=npr.evaluate('exp(alpha*gauamp**2)')

        alpha = -20.*np.sqrt(self.stats.sac.dist/1000.0)
        sf=npr.evaluate('exp(alpha*(gauamp/fcenter)**2)')
        sf1=npr.evaluate('alpha*(gauamp/fcenter)**2')
        # sf=sf*(np.abs(sf1)<40.);
        
        Win=spectrum.window.window_tukey(npts, r=0.5);
        sp, Ns=FFTW(tempdata, direction='forward')
        filtered_sp=npr.evaluate('sf*sp')
        filtered_seis, Ns=FFTW(filtered_sp, direction='backward')
        filtered_seis=filtered_seis[:npts].real
        return filtered_seis

    def getSNRTime(self, fhlen=0.008 ):
        """getSNRTime
        Get the SNR for signal window based on FTAN analysis.
        If input noisetrace is double-lagged, it will do SNR analysis for pos/neg lag; otherwise it will do SNR analysis for sym lag.
        -----------------------------------------------------------------------------------------------------
        Input Parameters:
        fhlen       - half length of Gaussian width
        
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
            fparam=self.ftanparam
            if fparam.nfout1_1!=0:
                o_per=fparam.arr1_1[1,:]
                g_vel=fparam.arr1_1[2,:]
                self.SNRParam.oper_s=o_per[:fparam.nfout1_1]
            for i in np.arange(fparam.nfout1_1):
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
                ib = (int)(minT/dt);
                ie = (int)(maxT/dt)+2;
                filtered_tr=self.GaussianFilter(1./o_per[i], fhlen=fhlen)
                tempTr_s=filtered_tr[ib:ie]
                tempTr_envelope=obspy.signal.filter.envelope(tempTr_s)
                
                # filtered_tr=self.GaussianFilterTaper(1./o_per[i], fhlen=fhlen, ib=ib, ie=ie)
                # tempTr_envelope=obspy.signal.filter.envelope(filtered_tr)
                
                # # # tempTr_s=npr.evaluate('abs(tempTr_s)')
                tempmax_s=tempTr_envelope.max()
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
                    noiserms_s=np.sqrt(npr.evaluate('sum(tempnoise_s)')/(ie-ib-1.))
                    self.SNRParam.nrms_s=np.append(self.SNRParam.nrms_s, noiserms_s)
                    tempSNR_s=tempmax_s/noiserms_s
                    self.SNRParam.snr_s=np.append(self.SNRParam.snr_s, tempSNR_s)
        except AttributeError:
            print 'Error: FTAN Parameters are not available!'
        return
    
    def getSNRFreq(self, vmin=1.5, vmax=5.0, perLst=np.array([]) ):
        
        try:
            self.SNRParam
        except:
            self.init_snrParam()
        dist=self.stats.sac.dist
        begT=self.stats.sac.b
        endT=self.stats.sac.e
        dt=self.stats.delta
        minT=dist/vmax;
        maxT=dist/vmin;
        ib=int(minT/dt);
        ie=int(maxT/dt)+2;
        tempsac=self.copy();
        tempsac.data=self.data[ib:ie];
        L=tempsac.data.size;
        Win=spectrum.window.window_tukey(L, r=0.5);
        tempsac.data=tempsac.data*Win;
        # print perLst
        tempsac.getSpec(1./perLst);
        self.SNRParam.amp_s=tempsac.hf;
        self.SNRParam.snr_s=np.ones(perLst.size)*100;
        self.SNRParam.nrms_s=np.ones(perLst.size);
        self.SNRParam.oper_s=perLst;
        # print perLst
        return;
    
    def getSNRaftan(self ):
        
        try:
            self.SNRParam
        except:
            self.init_snrParam()
        self.SNRParam.amp_s=self.ftanparam.arr1_1[8,:self.ftanparam.nfout1_1];
        self.SNRParam.snr_s=np.ones(self.ftanparam.nfout1_1)*100;
        self.SNRParam.nrms_s=np.ones(self.ftanparam.nfout1_1);
        self.SNRParam.oper_s=self.ftanparam.arr1_1[1,:self.ftanparam.nfout1_1];
        # print perLst
        return;
    
    def plotfreq(self):
        try:
            freq=self.freq;
            hf=self.hf;
        except:
            print 'No frequency data yet!'
            return;
        plt.semilogx(freq, np.abs(hf), lw=3)
        # plt.plot([fmin,fmin],[0.0, np.max(np.abs(hf))],'r--')
        # plt.text(1.1*fmin, 0.5*np.max(np.abs(hf)), 'fmin')
        # plt.plot([fmax,fmax],[0.0, np.max(np.abs(hf))],'r--')
        # plt.text(1.1*fmax, 0.5*np.max(np.abs(hf)), 'fmax')
        # plt.xlim(0.1*fmin,10.0*fmax)
        plt.xlabel('frequency [Hz]')
        plt.title('source time function (frequency domain)')
        plt.xlim(0.005,10.0)
        # plt.ylim(0,0.1)
        plt.show()
    
    def getSpec(self, FreqLst=np.array( [] ) ):
        npts=self.data.size;
        Ns=1<<(npts-1).bit_length()
        INput = np.zeros((Ns), dtype=complex)
        OUTput = np.zeros((Ns), dtype=complex)
        INput[:npts]=self.data
        nhalf=Ns/2+1;
        OUTput = np.fft.fft(INput);
        OUTput[nhalf:]=0
        OUTput[0]/=2
        OUTput[nhalf-1]=OUTput[nhalf-1].real+0.j
        self.hf = OUTput[:nhalf-1];
        # self.hf = np.fft.fft(self.data);
        # self.hf = np.fft.fft(self.data)/sqrt(len(self.hf));
        freq = np.fft.fftfreq(len(OUTput), self.stats.delta);
        self.freq=freq[:nhalf-1];
        if FreqLst.size !=0:
            # fmin=FreqLst.min();
            # fmax=FreqLst.max();
            hf=np.interp(FreqLst, self.freq, np.abs(self.hf ) );
            self.hf=hf;
            self.freq=FreqLst;    
        # self.mydata=np.abs(np.fft.ifft(self.hf));
        # # sp, Ns=FFTW(self.data, direction='forward')
        # # self.mydata, Ns=FFTW(sp, direction='backward')
        # self.mydata=(self.mydata[:self.data.size]).real
        # print Ns
        return;
    
    def plotTfr(self, t0=0.0, fmin=0.5, fmax=5.0, nf=100, w0=6, left=0.1,
                bottom=0.1, h_1=0.2, h_2=0.6, w_1=0.2, w_2=0.6, w_cb=0.01, d_cb=0.0, show=True,
                plot_args=[u'k', u'k'], clim=0.0, cmap=None, mode=u'absolute', fft_zero_pad_fac=0):
        obsTF.plotTfr(self.data, dt=self.stats.delta, t0=t0, fmin=fmin, fmax=fmax, nf=nf, w0=w0, left=left,
                bottom=bottom, h_1=h_1, h_2=h_2, w_1=w_1, w_2=w_2, w_cb=w_cb, d_cb=d_cb, show=show,
                plot_args=plot_args, clim=clim, cmap=cmap, mode=mode, fft_zero_pad_fac=fft_zero_pad_fac);
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
        OUTput=2*OUTput/Ns ###
    return OUTput, Ns


class InputFtanParam(object): ###
    """
    A subclass to store input parameters for aftan analysis and SNR Analysis
    -----------------------------------------------------------------------------------------------------
    Parameters:
    pmf         - flag for Phase-Matched-Filtered output (default: Fasle)
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
        self.pmf=False
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
        self.dosnrflag=False

    def setInParam(self, pmf=False, piover4=-1.0, vmin=1.5, vmax=5.0, tmin=4.0, \
        tmax=30.0, tresh=20.0, ffact=1.0, taperl=1.0, snr=0.2, fmatch=1.0, fhlen=0.008, dosnrflag=False, predV=np.array([]) ):
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
        self.predV=predV
        return

class Specfem2dDataBase(object):
    def __init__(self, enx, enz, StaLst, dspacing=1000):
        print ' Attention: Input source location unit should be in meter! ';
        self.symStream=obspy.core.Stream();
        self.dspacing=dspacing;
        self.StaLst=StaLst;
        self.enx=enx;
        self.enz=enz;
        return
    
    def ReadtxtSeismograms(self, datadir, sfx='.BXY.semd'):
        print 'Start Reading Seismograms (txt) !'
        for sta in self.StaLst:
            infname = datadir+'/' + sta.network+'.' +sta.stacode+sfx;
            InArr=np.loadtxt(infname);
            time=InArr[:,0];
            data=InArr[:,1];
            tr=obspy.core.Trace();
            tr.data=data;
            tr.stats['sac']={};
            tr.stats.sac.stlo=sta.x/1000.;
            tr.stats.sac.stla=sta.z/1000.;
            tr.stats.sac.user0=self.dspacing;
            tr.stats.station=sta.stacode;
            tr.stats.delta=time[1]-time[0];
            tr.stats.network=sta.network;
            tr.stats.sac.b=time[0];
            tr.stats.sac.evlo=self.enx/self.dspacing;
            tr.stats.sac.evla=self.enz/self.dspacing;
            tr.stats.sac.dist=np.sqrt( (tr.stats.sac.stlo-tr.stats.sac.evlo)**2
                + (tr.stats.sac.stla-tr.stats.sac.evla)**2 );
            self.symStream.append(tr);
        print 'End of Reading Seismograms (txt) !'
        return;
    
    def SaveSeismograms(self, outdir):
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        for tr in self.symStream:
            sacfname=outdir +'/'+ tr.stats.network+'.' +tr.stats.station+'..SAC';
            tr.write(sacfname, format='sac');
        return;
    
    def ReadSeismograms(self, datadir):
        print 'Start Reading Seismograms!'
        for sta in self.StaLst:
            infname = datadir+'/' + sta.network+'.' +sta.stacode+'..SAC';
            tr=obspy.read(infname)[0]
            self.symStream.append(tr);
        print 'End of Reading Seismograms!'
        return;
    
    def aftan(self, outdir, fhlen=0.008, inftan=InputFtanParam() ):
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        print 'Start aftan analysis!'
        for mytrace in self.symStream:
            if mytrace.stats.sac.dist < 0.01:
                print 'Too close to source, skip aftan for: '+mytrace.id+'SAC';
                continue;
            Strace=symtrace(mytrace.data, mytrace.stats);
            fPRX=outdir+'/'+Strace.id+'SAC';
            print fPRX
            Strace.aftan(pmf=inftan.pmf, piover4=inftan.piover4, vmin=inftan.vmin,
                vmax=inftan.vmax, tmin=inftan.tmin, tmax=inftan.tmax, tresh=inftan.tresh,
                ffact=inftan.ffact, taperl=inftan.taperl, snr=inftan.snr, fmatch=inftan.fmatch, predV=inftan.predV);
            Strace.ftanparam.writeDISP(fPRX);
            Strace.getSNRaftan();
            Strace.SNRParam.writeAMPSNR(fPRX);
        print 'End of aftan analysis!'
        return;
    
    def aftanParallel(self, outdir, fhlen=0.008, inftan=InputFtanParam() ):
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        print 'Start aftan analysis ( Parallel ) !'
        AFTAN = partial( FDataBaseaftan, outdir=outdir, fhlen=fhlen, inftan=inftan );
        pool =mp.Pool()
        pool.map(AFTAN, self.symStream) #make our results with a map call
        pool.close() #we are not adding any more processes
        pool.join() #tell it to wait until all threads are done before going on
        print 'End of aftan analysis ( Parallel ) !'
        return
    
    
    def GetField2dFile(self, datadir, outdir, perLst, factor=0.0, datatype='both', outfmt='txt'):
        if not os.path.isdir(outdir):
            os.makedirs(outdir);
        for per in perLst:
            TphLst=np.array([]);
            TgrLst=np.array([]);
            AmpLst=np.array([]);
            XLst=np.array([]);
            ZLst=np.array([]);
            Length=0;
            for sta in self.StaLst:
                X=sta.x;
                Z=sta.z;
                sacfname=datadir+'/' + sta.network+'.' +sta.stacode+'..SAC';
                fDISP=noisefile(sacfname+'_1_DISP.0', 'DISP')
                fsnr=noisefile(sacfname+'_amp_snr','SNR')
                (pvel, gvel) = fDISP.get_phvel(per);
                (snr, signal1, noise1) = fsnr.get_snr(per);
                dist=np.sqrt( (self.enx-X)**2 +  (self.enz-Z)**2 )/self.dspacing;
                if dist < per * factor:
                    print 'Skipping: '+sacfname;
                    continue;
                TravelTph=dist/pvel;
                TravelTgr=dist/gvel;
                
                XLst=np.append( XLst, X* self.dspacing/1000. )
                ZLst=np.append( ZLst, Z* self.dspacing/1000. )
                TphLst=np.append(TphLst, TravelTph);
                TgrLst=np.append(TgrLst, TravelTgr);
                AmpLst=np.append(AmpLst, signal1);
                Length=Length+1;
            ### End of Reading aftan results
            Tphfname = outdir+'/TravelT.ph.'+str(per)+'.'+outfmt;
            Tgrfname = outdir+'/TravelT.gr.'+str(per)+'.'+outfmt;
            Ampfname = outdir+'/Amplitude.'+str(per)+'.'+outfmt;
            
            OutArrTph = np.append(XLst, ZLst);
            OutArrTph = np.append(OutArrTph, TphLst);
            OutArrTph = OutArrTph.reshape((3,Length));
            OutArrTph = OutArrTph.T;
            if outfmt=='txt':
                np.savetxt(Tphfname, OutArrTph, fmt='%g');
            elif outfmt=='npy':
                np.save(Tphfname, OutArrTph, fmt='%g');
                
            OutArrTgr = np.append(XLst, ZLst);
            OutArrTgr = np.append(OutArrTgr, TgrLst);
            OutArrTgr = OutArrTgr.reshape((3,Length));
            OutArrTgr = OutArrTgr.T;
            if outfmt=='txt':
                np.savetxt(Tgrfname, OutArrTgr, fmt='%g');
            elif outfmt=='npy':
                np.save(Tgrfname, OutArrTgr, fmt='%g');
            
            OutArrAmp = np.append(XLst, ZLst);
            OutArrAmp = np.append(OutArrAmp, AmpLst);
            OutArrAmp = OutArrAmp.reshape((3,Length));
            OutArrAmp = OutArrAmp.T;
            if outfmt=='txt':
                np.savetxt(Ampfname, OutArrAmp, fmt='%g');
            elif outfmt=='npy':
                np.save(Ampfname, OutArrAmp, fmt='%g');
        return 
                    
def FDataBaseaftan(STrace, outdir, fhlen=0.008, inftan=InputFtanParam() ):
    if STrace.stats.sac.dist < 0.1:
        print 'Too close to source, skip aftan for: '+STrace.id+'SAC';
        return;
    Strace=symtrace(STrace.data, STrace.stats);
    Strace.aftan(pmf=inftan.pmf, piover4=inftan.piover4, vmin=inftan.vmin, vmax=inftan.vmax, tmin=inftan.tmin, tmax=inftan.tmax,
        tresh=inftan.tresh, ffact=inftan.ffact, taperl=inftan.taperl, snr=inftan.snr, fmatch=inftan.fmatch, predV=inftan.predV);
    fPRX=outdir+'/'+Strace.id+'SAC';
    Strace.ftanparam.writeDISP(fPRX);
    
    # Strace.getSNRTime(fhlen=fhlen);
    Strace.getSNRaftan()
#     perLst=Strace.ftanparam.arr1_1[1,:Strace.ftanparam.nfout1_1];
#     Strace.getSNRFreq(perLst=perLst);
    Strace.SNRParam.writeAMPSNR(fPRX);
    return;


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


            
            
            
            
            
        
        
        
        
        
        
    