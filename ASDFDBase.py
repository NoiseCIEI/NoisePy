import pyasdf
import numpy as np
import matplotlib.pyplot as plt
import obspy
import warnings
import copy
import os, shutil
import numba
from functools import partial
import multiprocessing


sta_info_default={'rec_func': 0, 'xcorr': 1, 'isnet': 0}

xcorr_header_default={'netcode1': '', 'stacode1': '', 'netcode2': '', 'stacode2': '', 'chan1': '', 'chan2': '',
        'npts': 12345, 'b': 12345, 'e': 12345, 'delta': 12345, 'dist': 12345, 'az': 12345, 'baz': 12345, 'stackday': 0}

xcorr_sacheader_default = {'knetwk': '', 'kstnm': '', 'kcmpnm': '', 'stla': 12345, 'stlo': 12345, 
            'kuser0': '', 'kevnm': '', 'evla': 12345, 'evlo': 12345, 'evdp': 0., 'dist': 0., 'az': 12345, 'baz': 12345, 
                'delta': 12345, 'npts': 12345, 'user0': 0, 'b': 12345, 'e': 12345}

monthdict={1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN', 7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}


class noiseASDF(pyasdf.ASDFDataSet):
    
    def init_working_env(self, datadir, workingdir):
        self.datadir    = datadir
        self.workingdir = workingdir
    
    def write_stationxml(self, staxml, source='CIEI'):
        inv=obspy.core.inventory.inventory.Inventory(networks=[], source=source)
        for staid in self.waveforms.list():
            inv+=self.waveforms[staid].StationXML
        inv.write(staxml, format='stationxml')
        return
    
    def write_stationtxt(self, stafile):
        """Write obspy inventory to txt station list(format used in SEED2COR)
        """
        try:
            auxiliary_info=self.auxiliary_data.StaInfo
            isStaInfo=True
        except:
            isStaInfo=False
        with open(stafile, 'w') as f:
            for staid in self.waveforms.list():
                stainv=self.waveforms[staid].StationXML
                netcode=stainv.networks[0].code
                stacode=stainv.networks[0].stations[0].code
                lon=stainv.networks[0].stations[0].longitude
                lat=stainv.networks[0].stations[0].latitude
                if isStaInfo:
                    staid_aux=netcode+'/'+stacode
                    ccflag=auxiliary_info[staid_aux].parameters['xcorr']
                    f.writelines('%s %3.4f %3.4f %d %s\n' %(stacode, lon, lat, ccflag, netcode) )
                else:
                    f.writelines('%s %3.4f %3.4f %s\n' %(stacode, lon, lat, netcode) )        
        return
    
    def read_stationtxt(self, stafile, source='CIEI', chans=['BHZ', 'BHE', 'BHN'], dnetcode='TA'):
        """Read txt station list 
        """
        sta_info=sta_info_default.copy()
        with open(stafile, 'r') as f:
            Sta=[]
            site=obspy.core.inventory.util.Site(name='01')
            creation_date=obspy.core.utcdatetime.UTCDateTime(0)
            inv=obspy.core.inventory.inventory.Inventory(networks=[], source=source)
            total_number_of_channels=len(chans)
            for lines in f.readlines():
                lines=lines.split()
                stacode=lines[0]
                lon=float(lines[1])
                lat=float(lines[2])
                netcode=dnetcode
                ccflag=None
                if len(lines)==5:
                    try:
                        ccflag=int(lines[3])
                        netcode=lines[4]
                    except ValueError:
                        ccflag=int(lines[4])
                        netcode=lines[3]
                if len(lines)==4:
                    try:
                        ccflag=int(lines[3])
                    except ValueError:
                        netcode=lines[3]
                netsta=netcode+'.'+stacode
                if Sta.__contains__(netsta):
                    index=Sta.index(netsta)
                    if abs(self[index].lon-lon) >0.01 and abs(self[index].lat-lat) >0.01:
                        raise ValueError('Incompatible Station Location:' + netsta+' in Station List!')
                    else:
                        print 'Warning: Repeated Station:' +netsta+' in Station List!'
                        continue
                channels=[]
                if lon>180.:
                    lon-=360.
                for chan in chans:
                    channel=obspy.core.inventory.channel.Channel(code=chan, location_code='01', latitude=lat, longitude=lon,
                            elevation=0.0, depth=0.0)
                    channels.append(channel)
                station=obspy.core.inventory.station.Station(code=stacode, latitude=lat, longitude=lon, elevation=0.0,
                        site=site, channels=channels, total_number_of_channels = total_number_of_channels, creation_date = creation_date)
                network=obspy.core.inventory.network.Network(code=netcode, stations=[station])
                networks=[network]
                inv+=obspy.core.inventory.inventory.Inventory(networks=networks, source=source)
                staid_aux=netcode+'/'+stacode
                if ccflag!=None:
                    sta_info['xcorr']=ccflag
                self.add_auxiliary_data(data=np.array([]), data_type='StaInfo', path=staid_aux, parameters=sta_info)
        print 'Writing obspy inventory to ASDF dataset'
        self.add_stationxml(inv)
        print 'End writing obspy inventory to ASDF dataset'
        return 
    
    def wsac_xcorr(self, netcode1, stacode1, netcode2, stacode2, chan1, chan2, outdir='.', pfx='COR'):
        """Write cross-correlation data from ASDF to sac file
        ==============================================================================
        Input Parameters:
        netcode1, stacode1, chan1   - network/station/channel name for station 1
        netcode2, stacode2, chan2   - network/station/channel name for station 2
        outdir                      - output directory
        pfx                         - prefix
        Output:
        e.g. outdir/COR/TA.G12A/COR_TA.G12A_BHT_TA.R21A_BHT.SAC
        ==============================================================================
        """
        subdset=self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][chan1][chan2]
        sta1=self.waveforms[netcode1+'.'+stacode1].StationXML.networks[0].stations[0]
        sta2=self.waveforms[netcode2+'.'+stacode2].StationXML.networks[0].stations[0]
        xcorr_sacheader=xcorr_sacheader_default.copy()
        xcorr_sacheader['kuser0']=netcode1
        xcorr_sacheader['kevnm']=stacode1
        xcorr_sacheader['knetwk']=netcode2
        xcorr_sacheader['kstnm']=stacode2
        xcorr_sacheader['kcmpnm']=chan1+chan2
        xcorr_sacheader['evla']=sta1.latitude
        xcorr_sacheader['evlo']=sta1.longitude
        xcorr_sacheader['stla']=sta2.latitude
        xcorr_sacheader['stlo']=sta2.longitude
        xcorr_sacheader['dist']=subdset.parameters['dist']
        xcorr_sacheader['az']=subdset.parameters['az']
        xcorr_sacheader['baz']=subdset.parameters['baz']
        xcorr_sacheader['b']=subdset.parameters['b']
        xcorr_sacheader['e']=subdset.parameters['e']
        xcorr_sacheader['delta']=subdset.parameters['delta']
        xcorr_sacheader['npts']=subdset.parameters['npts']
        xcorr_sacheader['user0']=subdset.parameters['stackday']
        sacTr=obspy.io.sac.sactrace.SACTrace(data=subdset.data.value, **xcorr_sacheader)
        if not os.path.isdir(outdir+'/'+pfx+'/'+netcode1+'.'+stacode1):
            os.makedirs(outdir+'/'+pfx+'/'+netcode1+'.'+stacode1)
        sacfname=outdir+'/'+pfx+'/'+netcode1+'.'+stacode1+'/'+ \
                pfx+'_'+netcode1+'.'+stacode1+'_'+chan1+'_'+netcode2+'.'+stacode2+'_'+chan2+'.SAC'
        sacTr.write(sacfname)
        return
    
    def wsac_xcorr_all(self, netcode1, stacode1, netcode2, stacode2, outdir='.', pfx='COR'):
        subdset=self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2]
        channels1=subdset.list()
        channels2=subdset[channels1[0]].list()
        for chan1 in channels1:
            for chan2 in channels2:
                self.wsac_xcorr(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                    stacode2=stacode2, chan1=chan1, chan2=chan2, outdir=outdir, pfx=pfx)
        return
    
    def xcorr_stack(self, datadir, startyear, startmonth, endyear, endmonth, pfx='COR', chantype=1, outdir=None, inchannels=None, fnametype='LF'):
        """Stack cross-correlation data from monthly-stacked sac files
        """
        utcdate=obspy.core.utcdatetime.UTCDateTime(startyear, startmonth, 1)
        ylst=np.array([], dtype=int)
        mlst=np.array([], dtype=int)
        while (utcdate.year<endyear or (utcdate.year<=endyear and utcdate.month<=endmonth) ):
            ylst=np.append(ylst, utcdate.year)
            mlst=np.append(mlst, utcdate.month)
            try:
                utcdate.month+=1
            except ValueError:
                utcdate.year+=1
                utcdate.month=1
        mnumb=mlst.size
        if inchannels!=None:
            try:
                if not isinstance(inchannels[0], obspy.core.inventory.channel.Channel):
                    channels=[]
                    for inchan in inchannels:
                        channels.append(obspy.core.inventory.channel.Channel(code=inchan, location_code='01',
                                        latitude=0, longitude=0, elevation=0, depth=0) )
                else:
                    channels=inchannels
            except:
                inchannels=None
        if inchannels==None:
            fnametype=='LF'
        else:
            if len(channels)!=1:
                fnametype=='LF'
        staLst=self.waveforms.list()
        for staid1 in staLst:
            for staid2 in staLst:
                netcode1, stacode1=staid1.split('.')
                netcode2, stacode2=staid2.split('.')
                if stacode1 >= stacode2:
                    continue
                stackedST=[]
                cST=[]
                initflag=True
                if inchannels==None:
                    channels1=self.waveforms[staid1].StationXML.networks[0].stations[0].channels
                    channels2=self.waveforms[staid2].StationXML.networks[0].stations[0].channels
                else:
                    channels1=channels
                    channels2=channels
                for im in xrange(mnumb):
                    skipflag=False
                    for chan1 in channels1:
                        if skipflag:
                            break
                        for chan2 in channels2:
                            month=monthdict[mlst[im]]
                            yrmonth=str(ylst[im])+'.'+month
                            if fnametype=='LF':
                                fname=datadir+'/'+yrmonth+'/'+pfx+'/'+stacode1+'/'+pfx+'_'+stacode1+'_'+chan1.code+'_'+stacode2+'_'+chan2.code+'.SAC'
                            elif fnametype=='YT':
                                fname=datadir+'/'+yrmonth+'/'+pfx+'/'+stacode1+'/'+pfx+'_'+stacode1+'_'+stacode2+'.SAC'
                            if not os.path.isfile(fname):
                                skipflag=True
                                break
                            try:
                                tr=obspy.core.read(fname)[0]
                            except TypeError:
                                warnings.warn('Unable to read SAC for: ' + stacode1 +'_'+stacode2 +' Month: '+yrmonth, UserWarning, stacklevel=1)
                                skipflag=True
                            if np.isnan(tr.data).any() or abs(tr.data.max())>1e20:
                                warnings.warn('NaN monthly SAC for: ' + stacode1 +'_'+stacode2 +' Month: '+yrmonth, UserWarning, stacklevel=1)
                                skipflag=True
                                break
                            cST.append(tr)
                    if len(cST)!=len(channels1)*len(channels2) or skipflag:
                        cST=[]
                        continue
                    if initflag:
                        stackedST=copy.deepcopy(cST)
                        initflag=False
                    else:
                        for itr in xrange(len(cST)):
                            mtr=cST[itr]
                            stackedST[itr].data+=mtr.data
                            stackedST[itr].stats.sac.user0+=mtr.stats.sac.user0
                    cST=[]
                if len(stackedST)==len(channels1)*len(channels2):
                    print 'Finished Stacking for:'+stacode1+'_'+stacode2
                    # create sac output directory 
                    if outdir!=None:
                        if not os.path.isdir(outdir+'/'+pfx+'/'+netcode1+'.'+stacode1):
                            os.makedirs(outdir+'/'+pfx+'/'+netcode1+'.'+stacode1)
                    # write cross-correlation header information
                    xcorr_header=xcorr_header_default.copy()
                    xcorr_header['b']=stackedST[0].stats.sac.b
                    xcorr_header['e']=stackedST[0].stats.sac.e
                    xcorr_header['netcode1']=netcode1
                    xcorr_header['netcode2']=netcode2
                    xcorr_header['stacode1']=netcode1
                    xcorr_header['stacode2']=netcode1
                    xcorr_header['npts']=stackedST[0].stats.npts
                    xcorr_header['delta']=stackedST[0].stats.delta
                    xcorr_header['stackday']=stackedST[0].stats.sac.user0
                    try:
                        xcorr_header['dist']=stackedST[0].stats.sac.dist
                        xcorr_header['az']=stackedST[0].stats.sac.az
                        xcorr_header['baz']=stackedST[0].stats.sac.baz
                    except AttributeError:
                        lon1=self.waveforms[staid1].StationXML.networks[0].stations[0].longitude
                        lat1=self.waveforms[staid1].StationXML.networks[0].stations[0].latitude
                        lon2=self.waveforms[staid2].StationXML.networks[0].stations[0].longitude
                        lat2=self.waveforms[staid2].StationXML.networks[0].stations[0].latitude
                        dist, az, baz=obspy.geodetics.gps2dist_azimuth(lat1, lon1, lat2, lon2)
                        dist=dist/1000.
                        xcorr_header['dist']=dist
                        xcorr_header['az']=az
                        xcorr_header['baz']=baz
                    staid_aux=netcode1+'/'+stacode1+'/'+netcode2+'/'+stacode2
                    i=0
                    for chan1 in channels1:
                        for chan2 in channels2:
                            stackedTr=stackedST[i]
                            if outdir!=None:
                                outfname=outdir+'/'+pfx+'/'+netcode1+'.'+stacode1+'/'+ \
                                    pfx+'_'+netcode1+'.'+stacode1+'_'+chan1.code+'_'+netcode2+'.'+stacode2+'_'+chan2.code+'.SAC'
                                stackedTr.write(outfname,format='SAC')
                            xcorr_header['chan1']=chan1.code
                            xcorr_header['chan2']=chan2.code
                            self.add_auxiliary_data(data=stackedTr.data, data_type='NoiseXcorr', path=staid_aux+'/'+chan1.code+'/'+chan2.code, parameters=xcorr_header)
                            i+=1
        return
    
    def xcorr_stack_mp(self, datadir, outdir, startyear, startmonth, endyear, endmonth,
                    pfx='COR', inchannels=None, fnametype='LF', subsize=1000, deletesac=True):
        
        utcdate=obspy.core.utcdatetime.UTCDateTime(startyear, startmonth, 1)
        ylst=np.array([], dtype=int)
        mlst=np.array([], dtype=int)
        print 'Preparing data for stacking'
        while (utcdate.year<endyear or (utcdate.year<=endyear and utcdate.month<=endmonth) ):
            ylst=np.append(ylst, utcdate.year)
            mlst=np.append(mlst, utcdate.month)
            try:
                utcdate.month+=1
            except ValueError:
                utcdate.year+=1
                utcdate.month=1
        mnumb=mlst.size
        staLst=self.waveforms.list()
        if inchannels!=None:
            try:
                if not isinstance(inchannels[0], obspy.core.inventory.channel.Channel):
                    channels=[]
                    for inchan in inchannels:
                        channels.append(obspy.core.inventory.channel.Channel(code=inchan, location_code='01',
                                        latitude=0, longitude=0, elevation=0, depth=0) )
                else:
                    channels=inchannels
            except:
                inchannels=None
        if inchannels==None:
            fnametype=='LF'
        else:
            if len(channels)!=1:
                fnametype=='LF'
        stapairInvLst=[]
        for staid1 in staLst:
            if not os.path.isdir(outdir+'/'+pfx+'/'+staid1):
                os.makedirs(outdir+'/'+pfx+'/'+staid1)
            for staid2 in staLst:
                netcode1, stacode1=staid1.split('.')
                netcode2, stacode2=staid2.split('.')
                if stacode1 >= stacode2:
                    continue
                inv = self.waveforms[staid1].StationXML+self.waveforms[staid2].StationXML
                if inchannels!=None:
                    inv.networks[0].stations[0].channels=channels
                    inv.networks[1].stations[0].channels=channels
                stapairInvLst.append(inv) 
        print 'Start stacking (MP)!'
        if len(stapairInvLst) > subsize:
            Nsub = int(len(stapairInvLst)/subsize)
            for isub in xrange(Nsub):
                print isub,'in',Nsub
                cstapairs=stapairInvLst[isub*subsize:(isub+1)*subsize]
                STACKING = partial(stack4mp, datadir=datadir, outdir=outdir, ylst=ylst, mlst=mlst, pfx=pfx, fnametype=fnametype)
                pool = multiprocessing.Pool()
                pool.map_async(STACKING, cstapairs) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
            cstapairs=stapairInvLst[(isub+1)*subsize:]
            STACKING = partial(stack4mp, datadir=datadir, outdir=outdir, ylst=ylst, mlst=mlst, pfx=pfx, fnametype=fnametype)
            pool = multiprocessing.Pool()
            pool.map_async(STACKING, cstapairs) 
            pool.close() 
            pool.join() 
        else:
            STACKING = partial(stack4mp, datadir=datadir, outdir=outdir, ylst=ylst, mlst=mlst, pfx=pfx, fnametype=fnametype)
            pool = multiprocessing.Pool()
            pool.map_async(STACKING, stapairInvLst) 
            pool.close() 
            pool.join() 
        print 'End of stacking  ( MP ) !'
        print 'Reading data into ASDF database'
        for inv in stapairInvLst:
            channels1=inv.networks[0].stations[0].channels
            netcode1=inv.networks[0].code
            stacode1=inv.networks[0].stations[0].code
            channels2=inv.networks[1].stations[0].channels
            netcode2=inv.networks[1].code
            stacode2=inv.networks[1].stations[0].code
            skipflag=False
            xcorr_header=xcorr_header_default.copy()
            xcorr_header['netcode1']=netcode1
            xcorr_header['netcode2']=netcode2
            xcorr_header['stacode1']=stacode1
            xcorr_header['stacode2']=stacode2
            staid_aux=netcode1+'/'+stacode1+'/'+netcode2+'/'+stacode2
            for chan1 in channels1:
                if skipflag:
                    break
                for chan2 in channels2:
                    sacfname=outdir+'/'+pfx+'/'+netcode1+'.'+stacode1+'/'+ \
                        pfx+'_'+netcode1+'.'+stacode1+'_'+chan1.code+'_'+netcode2+'.'+stacode2+'_'+chan2.code+'.SAC'
                    try:
                        tr=obspy.read(sacfname)[0]
                        # cross-correlation header 
                        xcorr_header['b']=tr.stats.sac.b
                        xcorr_header['e']=tr.stats.sac.e
                        xcorr_header['npts']=tr.stats.npts
                        xcorr_header['delta']=tr.stats.delta
                        xcorr_header['stackday']=tr.stats.sac.user0
                        try:
                            xcorr_header['dist']=tr.stats.sac.dist
                            xcorr_header['az']=tr.stats.sac.az
                            xcorr_header['baz']=tr.stats.sac.baz
                        except AttributeError:
                            lon1=inv.networks[0].stations[0].longitude
                            lat1=inv.networks[0].stations[0].latitude
                            lon2=inv.networks[1].stations[0].longitude
                            lat2=inv.networks[1].stations[0].latitude
                            dist, az, baz=obspy.geodetics.gps2dist_azimuth(lat1, lon1, lat2, lon2)
                            dist=dist/1000.
                            xcorr_header['dist']=dist
                            xcorr_header['az']=az
                            xcorr_header['baz']=baz
                        xcorr_header['chan1']=chan1.code
                        xcorr_header['chan2']=chan2.code
                        self.add_auxiliary_data(data=tr.data, data_type='NoiseXcorr', path=staid_aux+'/'+chan1.code+'/'+chan2.code, parameters=xcorr_header)
                    except IOError:
                        skipflag==True
                        break
        if deletesac:
            shutil.rmtree(outdir+'/'+pfx)
        print 'End read data into ASDF database'
        return
                    
    def xcorr_rotation(self, outdir=None, pfx='COR'):
        staLst=self.waveforms.list()
        for staid1 in staLst:
            for staid2 in staLst:
                netcode1, stacode1=staid1.split('.')
                netcode2, stacode2=staid2.split('.')
                if stacode1 >= stacode2:
                    continue
                chan1E=None; chan1N=None; chan1Z=None; chan2E=None; chan2N=None; chan2Z=None
                try:
                    channels1=self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2].list()
                    channels2=self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][channels1[0]].list()
                    cpfx1=channels1[0][:2]
                    cpfx2=channels2[0][:2]
                    for chan in channels1:
                        if chan[2]=='E':
                            chan1E=chan
                        if chan[2]=='N':
                            chan1N=chan
                        if chan[2]=='Z':
                            chan1Z=chan
                    for chan in channels2:
                        if chan[2]=='E':
                            chan2E=chan
                        if chan[2]=='N':
                            chan2N=chan
                        if chan[2]=='Z':
                            chan2Z=chan
                except AttributeError:
                    continue
                subdset=self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2]
                if chan1E==None or chan1N==None or chan2E==None or chan2N==None:
                    continue
                if chan1Z==None or chan2Z==None:
                    print 'Do rotation(RT) for:'+netcode1+'.'+stacode1+' and '+netcode2+'.'+stacode2
                else:
                    print 'Do rotation(RTZ) for:'+netcode1+'.'+stacode1+' and '+netcode2+'.'+stacode2
                dsetEE=subdset[chan1E][chan2E]
                dsetEN=subdset[chan1E][chan2N]
                dsetNE=subdset[chan1N][chan2E]
                dsetNN=subdset[chan1N][chan2N]
                temp_header=dsetEE.parameters.copy()
                chan1R=cpfx1+'R'; chan1T=cpfx1+'T'; chan2R=cpfx2+'R'; chan2T=cpfx2+'T'
                theta=temp_header['az']
                psi=temp_header['baz']
                Ctheta=np.cos(np.pi*theta/180.)
                Stheta=np.sin(np.pi*theta/180.)
                Cpsi=np.cos(np.pi*psi/180.)
                Spsi=np.sin(np.pi*psi/180.)
                tempTT=-Ctheta*Cpsi*dsetEE.data.value+Ctheta*Spsi*dsetEN.data.value - \
                    Stheta*Spsi*dsetNN.data.value + Stheta*Cpsi*dsetNE.data.value
                tempRR=- Stheta*Spsi*dsetEE.data.value - Stheta*Cpsi*dsetEN.data.value \
                    - Ctheta*Cpsi*dsetNN.data.value - Ctheta*Spsi*dsetNE.data.value
                tempTR=-Ctheta*Spsi*dsetEE.data.value - Ctheta*Cpsi*dsetEN.data.value  \
                    + Stheta*Cpsi*dsetNN.data.value + Stheta*Spsi*dsetNE.data.value
                tempRT=-Stheta*Cpsi*dsetEE.data.value +Stheta*Spsi*dsetEN.data.value \
                    + Ctheta*Spsi*dsetNN.data.value - Ctheta*Cpsi*dsetNE.data.value
                staid_aux=netcode1+'/'+stacode1+'/'+netcode2+'/'+stacode2
                temp_header['chan1']=chan1T; temp_header['chan2']=chan2T
                self.add_auxiliary_data(data=tempTT, data_type='NoiseXcorr', path=staid_aux+'/'+chan1T+'/'+chan2T, parameters=temp_header)
                
                temp_header['chan1']=chan1R; temp_header['chan2']=chan2R
                self.add_auxiliary_data(data=tempRR, data_type='NoiseXcorr', path=staid_aux+'/'+chan1R+'/'+chan2R, parameters=temp_header)
                
                temp_header['chan1']=chan1T; temp_header['chan2']=chan2R
                self.add_auxiliary_data(data=tempTR, data_type='NoiseXcorr', path=staid_aux+'/'+chan1T+'/'+chan2R, parameters=temp_header)
                
                temp_header['chan1']=chan1R; temp_header['chan2']=chan2T
                self.add_auxiliary_data(data=tempRT, data_type='NoiseXcorr', path=staid_aux+'/'+chan1R+'/'+chan2T, parameters=temp_header)
                # write to sac files
                if outdir!=None:
                    self.wsac_xcorr(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                            stacode2=stacode2, chan1=chan1T, chan2=chan2T, outdir=outdir, pfx=pfx)
                    self.wsac_xcorr(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                            stacode2=stacode2, chan1=chan1R, chan2=chan2R, outdir=outdir, pfx=pfx)
                    self.wsac_xcorr(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                            stacode2=stacode2, chan1=chan1T, chan2=chan2R, outdir=outdir, pfx=pfx)
                    self.wsac_xcorr(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                            stacode2=stacode2, chan1=chan1R, chan2=chan2T, outdir=outdir, pfx=pfx)
                # RTZ rotation
                if chan1Z!=None and chan2Z!=None:
                    dsetEZ=subdset[chan1E][chan2Z]
                    dsetZE=subdset[chan1Z][chan2E]
                    dsetNZ=subdset[chan1N][chan2Z]
                    dsetZN=subdset[chan1Z][chan2N]
                    tempRZ = Ctheta*dsetNZ.data.value + Stheta*dsetEZ.data.value
                    tempZR = - Cpsi*dsetZN.data.value -Spsi*dsetZE.data.value
                    tempTZ = -Stheta*dsetNZ.data.value + Ctheta*dsetEZ.data.value
                    tempZT =  Spsi*dsetZN.data.value - Cpsi*dsetZE.data.value
                    temp_header['chan1']=chan1R; temp_header['chan2']=chan2Z
                    self.add_auxiliary_data(data=tempRZ, data_type='NoiseXcorr', path=staid_aux+'/'+chan1R+'/'+chan2Z, parameters=temp_header)
                    
                    temp_header['chan1']=chan1Z; temp_header['chan2']=chan2R
                    self.add_auxiliary_data(data=tempZR, data_type='NoiseXcorr', path=staid_aux+'/'+chan1Z+'/'+chan2R, parameters=temp_header)
                    
                    temp_header['chan1']=chan1T; temp_header['chan2']=chan2Z
                    self.add_auxiliary_data(data=tempTZ, data_type='NoiseXcorr', path=staid_aux+'/'+chan1T+'/'+chan2Z, parameters=temp_header)
                    
                    temp_header['chan1']=chan1Z; temp_header['chan2']=chan2T
                    self.add_auxiliary_data(data=tempZT, data_type='NoiseXcorr', path=staid_aux+'/'+chan1Z+'/'+chan2T, parameters=temp_header)
                    # write to sac files
                    if outdir!=None:
                        self.wsac_xcorr(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                                stacode2=stacode2, chan1=chan1Z, chan2=chan2R, outdir=outdir, pfx=pfx)
                        self.wsac_xcorr(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                                stacode2=stacode2, chan1=chan1R, chan2=chan2Z, outdir=outdir, pfx=pfx)
                        self.wsac_xcorr(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                                stacode2=stacode2, chan1=chan1T, chan2=chan2Z, outdir=outdir, pfx=pfx)
                        self.wsac_xcorr(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                                stacode2=stacode2, chan1=chan1Z, chan2=chan2T, outdir=outdir, pfx=pfx)
        return
               

            
def stack4mp(inv, datadir, outdir, ylst, mlst, pfx, fnametype):
    stackedST=[]
    cST=[]
    initflag=True
    channels1=inv.networks[0].stations[0].channels
    channels2=inv.networks[1].stations[0].channels
    netcode1=inv.networks[0].code
    stacode1=inv.networks[0].stations[0].code
    netcode2=inv.networks[1].code
    stacode2=inv.networks[1].stations[0].code
    mnumb=mlst.size
    for im in xrange(mnumb):
        skipflag=False
        for chan1 in channels1:
            if skipflag:
                break
            for chan2 in channels2:
                month=monthdict[mlst[im]]
                yrmonth=str(ylst[im])+'.'+month
                if fnametype=='LF':
                    fname=datadir+'/'+yrmonth+'/'+pfx+'/'+stacode1+'/'+pfx+'_'+stacode1+'_'+chan1.code+'_'+stacode2+'_'+chan2.code+'.SAC'
                elif fnametype=='YT':
                    fname=datadir+'/'+yrmonth+'/'+pfx+'/'+stacode1+'/'+pfx+'_'+stacode1+'_'+stacode2+'.SAC'
                if not os.path.isfile(fname):
                    skipflag=True
                    break
                try:
                    tr=obspy.core.read(fname)[0]
                except TypeError:
                    warnings.warn('Unable to read SAC for: ' + stacode1 +'_'+stacode2 +' Month: '+yrmonth, UserWarning, stacklevel=1)
                    skipflag=True
                if np.isnan(tr.data).any() or abs(tr.data.max())>1e20:
                    warnings.warn('NaN monthly SAC for: ' + stacode1 +'_'+stacode2 +' Month: '+yrmonth, UserWarning, stacklevel=1)
                    skipflag=True
                    break
                cST.append(tr)
        if len(cST)!=len(channels1)*len(channels2) or skipflag:
            cST=[]
            continue
        if initflag:
            stackedST=copy.deepcopy(cST)
            initflag=False
        else:
            for itr in xrange(len(cST)):
                mtr=cST[itr]
                stackedST[itr].data+=mtr.data
                stackedST[itr].stats.sac.user0+=mtr.stats.sac.user0
        cST=[]
    if len(stackedST)==len(channels1)*len(channels2):
        print 'Finished Stacking for:'+stacode1+'_'+stacode2
        i=0
        for chan1 in channels1:
            for chan2 in channels2:
                stackedTr=stackedST[i]
                outfname=outdir+'/'+pfx+'/'+netcode1+'.'+stacode1+'/'+ \
                    pfx+'_'+netcode1+'.'+stacode1+'_'+chan1.code+'_'+netcode2+'.'+stacode2+'_'+chan2.code+'.SAC'
                stackedTr.write(outfname,format='SAC')
                i+=1
    return
    
    