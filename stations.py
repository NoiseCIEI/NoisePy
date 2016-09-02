import numpy as np



class StaInfo(object):
    """
    An object contains a station information several methods for station related analysis.
    ===========================================================================================================
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
    ===========================================================================================================
    """
    def __init__(self, stacode=None, network='', virtual_Net=None, lat=None, lon=None, \
        elevation=None, start_date=None, end_date=None, ccflag=None, chan=[]):

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

    def read(self, stafile):
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