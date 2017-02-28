# -*- coding: utf-8 -*-
"""
A python wrapper to run Misha Barmin's straight ray surface wave tomography
The code creates a datadbase based on hdf5 data format

:Dependencies:
    numpy >=1.9.1
    matplotlib >=1.4.3
    h5py 
    
:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
    
:References:
    Barmin, M. P., M. H. Ritzwoller, and A. L. Levshin. "A fast and reliable method for surface wave tomography."
            Monitoring the Comprehensive Nuclear-Test-Ban Treaty: Surface Waves. Birkh?user Basel, 2001. 1351-1375.
"""
import numpy as np
import numpy.ma as ma
import h5py
import os, shutil
from subprocess import call
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
import colormaps
import obspy

class RayTomoDataSet(h5py.File):
    
    def set_input_parameters(self, minlon, maxlon, minlat, maxlat, pers=np.array([]), data_pfx='raytomo_in_', smoothpfx='N_INIT_', qcpfx='QC_'):
        """
        Set input parameters for tomographic inversion.
        =================================================================================================================
        Input Parameters:
        minlon, maxlon  - minimum/maximum longitude
        minlat, maxlat  - minimum/maximum latitude
        pers            - period array, default = np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
        data_pfx        - input data file prefix
        smoothpfx       - prefix for smooth run files
        smoothpfx       - prefix for qc(quanlity controlled) run files
        =================================================================================================================
        """
        if pers.size==0:
            # pers=np.arange(13.)*2.+6.
            pers=np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
        self.attrs.create(name = 'period_array', data=pers, dtype='f')
        self.attrs.create(name = 'minlon', data=minlon, dtype='f')
        self.attrs.create(name = 'maxlon', data=maxlon, dtype='f')
        self.attrs.create(name = 'minlat', data=minlat, dtype='f')
        self.attrs.create(name = 'maxlat', data=maxlat, dtype='f')
        self.attrs.create(name = 'data_pfx', data=data_pfx)
        self.attrs.create(name = 'smoothpfx', data=smoothpfx)
        self.attrs.create(name = 'qcpfx', data=qcpfx)
        return
        
    def run_smooth(self, datadir, outdir, datatype='ph', channel='ZZ', dlon=0.5, dlat=0.5, stepinte=0.2, lengthcell=1.0, alpha1=3000, alpha2=100, sigma=500,
            runid=0, comments='', deletetxt=False, contourfname='./contour.ctr', IsoMishaexe='./TOMO_MISHA/itomo_sp_cu_shn' ):
        """
        Run Misha's Tomography Code with large regularization parameters.
        This function is designed to do an inital test run, the output can be used to discard outliers in aftan results.
        =================================================================================================================
        Input Parameters:
        datadir/outdir      - data/output directory
        datatype            - ph: phase velocity inversion, gr: group velocity inversion
        channel             - channel for analysis (default: ZZ, xcorr ZZ component)
        dlon/dlat           - longitude/latitude interval
        stepinte            - step of integral
        lengthcell          - size of main cell (degree)
        alpha1,alpha2,sigma - regularization parameters for isotropic tomography
                                alpha1: smoothing coefficient, alpha2: path density damping, sigma: Gaussian smoothing
        runid               - id number for the run
        comments            - comments for the run
        deletetxt           - delete txt output or not
        contourfname        - path to contour file (see the manual for detailed description)
        IsoMishaexe         - path to Misha's Tomography code executable (isotropic version)
        ------------------------------------------------------------------------------------------------------------------
        Input format:
        datadir/data_pfx+'%g'%( per ) +'_'+channel+'_'+datatype+'.lst' (e.g. datadir/raytomo_10_ZZ_ph.lst)
        e.g. datadir/MISHA_in_20.0_BHZ_BHZ_ph.lst
        
        Output format:
        e.g. 
        Prefix: outdir/10_ph/N_INIT_3000_500_100
        output file: outdir/10.0_ph/N_INIT_3000_500_100_10.0.1 etc. 
        =================================================================================================================
        """
        if not os.path.isfile(IsoMishaexe): raise AttributeError('IsoMishaexe does not exist!')
        if not os.path.isfile(contourfname): raise AttributeError('Contour file does not exist!')
        pers = self.attrs['period_array']
        minlon=self.attrs['minlon']
        maxlon=self.attrs['maxlon']
        minlat=self.attrs['minlat']
        maxlat=self.attrs['maxlat']
        data_pfx=self.attrs['data_pfx']
        smoothpfx=self.attrs['smoothpfx']
        if not os.path.isdir(outdir): deleteall=True
        for per in pers:
            infname=datadir+'/'+data_pfx+'%g'%( per ) +'_'+channel+'_'+datatype+'.lst'
            outper=outdir+'/'+'%g'%( per ) +'_'+datatype
            if not os.path.isdir(outper):
                os.makedirs(outper)
            outpfx=outper+'/'+smoothpfx+str(alpha1)+'_'+str(sigma)+'_'+str(alpha2)
            temprunsh='temp_'+'%g_Smooth.sh' %(per)
            with open(temprunsh,'wb') as f:
                f.writelines('%s %s %s %g << EOF \n' %(IsoMishaexe, infname, outpfx, per ))
                # if paraFlag==False:
                #     f.writelines('me \n' );
                f.writelines('me \n4 \n5 \n%g \n6 \n%g \n%g \n%g \n' %( alpha2, alpha1, sigma, sigma) )
                f.writelines('7 \n%g %g %g \n8 \n%g %g %g \n12 \n%g \n%g \n16 \n' %(minlat, maxlat, dlat, minlon, maxlon, dlon, stepinte, lengthcell) )
                # if paraFlag==False:
                #     f.writelines('v \n' );
                f.writelines('v \nq \ngo \nEOF \n' )
            call(['bash', temprunsh])
            # os.remove(temprunsh)
        # save to hdf5 dataset
        create_group=False
        while (not create_group):
            try:
                group=self.create_group( name = 'smooth_run_'+str(runid) )
                create_group=True
            except:
                runid+=1
                continue
        group.attrs.create(name = 'comments', data=comments)
        group.attrs.create(name = 'dlon', data=dlon)
        group.attrs.create(name = 'dlat', data=dlat)
        group.attrs.create(name = 'step_of_integration', data=stepinte)
        group.attrs.create(name = 'datatype', data=datatype)
        group.attrs.create(name = 'channel', data=channel)
        group.attrs.create(name = 'alpha1', data=alpha1)
        group.attrs.create(name = 'alpha2', data=alpha2)
        group.attrs.create(name = 'sigma', data=sigma)
        for per in pers:
            subgroup=group.create_group(name='%g_sec'%( per ))
            outper=outdir+'/'+'%g'%( per ) +'_'+datatype
            outpfx=outper+'/'+smoothpfx+str(alpha1)+'_'+str(sigma)+'_'+str(alpha2)
            v0fname     = outpfx+'_%g.1' %(per)
            dvfname     = outpfx+'_%g.1' %(per)+'_%_'
            azifname    = outpfx+'_%g.azi' %(per)
            residfname  = outpfx+'_%g.resid' %(per)
            resfname    = outpfx+'_%g.res' %(per)
            inArr=np.loadtxt(v0fname); v0Arr=inArr[:,2];
            v0dset=subgroup.create_dataset(name='velocity', data=v0Arr)
            inArr=np.loadtxt(dvfname); dvArr=inArr[:,2];
            dvdset=subgroup.create_dataset(name='Dvelocity', data=dvArr)
            inArr=np.loadtxt(azifname); aziArr=inArr[:,2:4]
            azidset=subgroup.create_dataset(name='azi_coverage', data=aziArr)
            inArr=np.loadtxt(residfname)
            residdset=subgroup.create_dataset(name='residual', data=inArr)
			#residdset.attrs.create(name='res_tomo', data=8)
            inArr=np.loadtxt(resfname); resArr=inArr[:,2:]
            resdset=subgroup.create_dataset(name='path_density', data=resArr)
            if deletetxt: shutil.rmtree(outper)
        if deletetxt and deleteall: shutil.rmtree(outdir)
        return
    
    def run_qc(self, outdir, runid=0, smoothid=0, isotropic=False, datatype='ph', wavetype='R', crifactor=0.5, crilimit=10., dlon=0.5, dlat=0.5, stepinte=0.1,
            lengthcell=0.5, alpha=850, beta=1, sigma=175, lengthcellAni=1.0, anipara=1, xZone=2, alphaAni0=1200, betaAni0=1, sigmaAni0=200,
            alphaAni2=1000, sigmaAni2=100, alphaAni4=1200, sigmaAni4=500, comments='', deletetxt=False, contourfname='./contour.ctr', 
            IsoMishaexe='./TOMO_MISHA/itomo_sp_cu_shn', AniMishaexe='./TOMO_MISHA_AZI/tomo_sp_cu_s_shn-.1/tomo_sp_cu_s_shn_.1'):
        """
        Run Misha's Tomography Code with quality control based on preliminary run of run_smooth.
        This function is designed to discard outliers in aftan results (Quality Control), and then do tomography.
        =================================================================================================================
        Input Parameters:
        outdir              - output directory
        smoothid            - smooth run id number
        isotropic           - use isotropic or anisotropic version
        datatype            - ph: phase velocity inversion, gr: group velocity inversion
        wavetype            - wave type(R: Rayleigh, L: Love)
        crifactor/crilimit  - criteria for quality control
                                largest residual is min( crifactor*period, crilimit)
        dlon/dlat           - longitude/latitude interval
        stepinte            - step of integral
        lengthcell          - size of main cell (degree)
        alpha,beta,sigma    - regularization parameters for isotropic tomography (isotropic==True)
                                alpha: smoothing coefficient, beta: path density damping, sigma: Gaussian smoothing
        lengthcellAni       - size of anisotropic cell (degree)
        anipara             - anisotropic paramter(0: isotropic, 1: 2 psi anisotropic, 2: 2&4 psi anisotropic)
        xZone               -
        alphaAni0,betaAni0,sigmaAni0 
                            - regularization parameters for isotropic term in anisotropic tomography  (isotropic==False)
                                alphaAni0: smoothing coefficient, betaAni0: path density damping, sigmaAni0: Gaussian smoothing
        alphaAni2,sigmaAni2 - regularization parameters for 2 psi term in anisotropic tomography  (isotropic==False)
                                alphaAni2: smoothing coefficient, sigmaAni2: Gaussian smoothing
        
        alphaAni4,sigmaAni4 - regularization parameters for 4 psi term in anisotropic tomography  (isotropic==False)
                                alphaAni4: smoothing coefficient, sigmaAni4: Gaussian smoothing
        comments            - comments for the run
        deletetxt           - delete txt output or not
        contourfname        - path to contour file (see the manual for detailed description)
        IsoMishaexe         - path to Misha's Tomography code executable (isotropic version)
        AniMishaexe         - path to Misha's Tomography code executable (anisotropic version)
        ------------------------------------------------------------------------------------------------------------------
        Intermediate output format:
        outdir+'/'+per+'_'+datatype+'/QC_'+per+'_'+wavetype+'_'+datatype+'.lst'
        e.g. outdir/10_ph/QC_10_R_ph.lst
        
        Output format:
        e.g. 
        prefix: outdir/10_ph/QC_850_175_1  OR outdir/10_ph/QC_AZI_R_1200_200_1000_100_1
        
        Output file:
        outdir/10_ph/QC_850_175_1_10.1 etc. 
        OR
        outdir/10_ph/QC_AZI_R_1200_200_1000_100_1_10.1 etc. (see the manual for detailed description of output suffix)
        =================================================================================================================
        """
        pers = self.attrs['period_array']
        minlon=self.attrs['minlon']
        maxlon=self.attrs['maxlon']
        minlat=self.attrs['minlat']
        maxlat=self.attrs['maxlat']
        smoothpfx=self.attrs['smoothpfx']
        qcpfx=self.attrs['qcpfx']
        if isotropic:
            mishaexe=IsoMishaexe
        else:
            mishaexe=AniMishaexe
            qcpfx=qcpfx+'AZI_'
        contourfname='./contour.ctr'
        if not os.path.isfile(mishaexe): raise AttributeError('mishaexe does not exist!')
        if not os.path.isfile(contourfname): raise AttributeError('Contour file does not exist!')
        smoothgroup=self['smooth_run_'+str(smoothid)]
        for per in pers:
            try:
                residdset = smoothgroup['%g_sec'%( per )+'/residual']
                inArr=residdset.value
            except:
                raise AttributeError('Residual data: '+ str(per)+ ' sec does not exist!')
            res_tomo=inArr[:,7]
            cri_res=min(crifactor*per, crilimit)
            QC_arr= inArr[np.abs(res_tomo)<cri_res, :]
            outArr=QC_arr[:,:8]
            outper=outdir+'/'+'%g'%( per ) +'_'+datatype
            if not os.path.isdir(outper): os.makedirs(outper)
            QCfname=outper+'/QC_'+'%g'%( per ) +'_'+wavetype+'_'+datatype+'.lst'
            np.savetxt(QCfname, outArr, fmt='%g')
            # start to run tomography code
            if isotropic:
                outpfx=outper+'/'+qcpfx+str(alpha)+'_'+str(sigma)+'_'+str(beta)
            else:
                outpfx=outper+'/'+qcpfx+wavetype+'_'+str(alphaAni0)+'_'+str(sigmaAni0)+'_'+str(alphaAni2)+'_'+str(sigmaAni2)+'_'+str(betaAni0)
            temprunsh='temp_'+'%g_QC.sh' %(per)
            with open(temprunsh,'wb') as f:
                f.writelines('%s %s %s %g << EOF \n' %(mishaexe, QCfname, outpfx, per ))
                if isotropic:
                    f.writelines('me \n4 \n5 \n%g \n6 \n%g \n%g \n%g \n' %( beta, alpha, sigma, sigma) )
                    f.writelines('7 \n%g %g %g \n8 \n%g %g %g \n12 \n%g \n%g \n16 \n' %(minlat, maxlat, dlat, minlon, maxlon, dlon, stepinte, lengthcell) )
                    f.writelines('v \nq \ngo \nEOF \n' )
                else:
                    if datatype=='ph':
                        Dtype='P'
                    else:
                        Dtype='G'
                    f.writelines('me \n4 \n5 \n%g %g %g \n6 \n%g %g %g \n' %( minlat, maxlat, dlat, minlon, maxlon, dlon) )
                    f.writelines('10 \n%g \n%g \n%s \n%s \n%g \n%g \n11 \n%d \n' %(stepinte, xZone, wavetype, Dtype, lengthcell, lengthcellAni, anipara) )
                    f.writelines('12 \n%g \n%g \n%g \n%g \n' %(alphaAni0, betaAni0, sigmaAni0, sigmaAni0) )
                    f.writelines('13 \n%g \n%g \n%g \n' %(alphaAni2, sigmaAni2, sigmaAni2) )
                    if anipara==2:
                        f.writelines('14 \n%g \n%g \n%g \n' %(alphaAni4, sigmaAni4, sigmaAni4) )
                    f.writelines('19 \n25 \n' )
                    f.writelines('v \nq \ngo \nEOF \n' )
            call(['bash', temprunsh])
            os.remove(temprunsh)
        # save to hdf5 dataset
        create_group=False
        while (not create_group):
            try:
                group=self.create_group( name = 'qc_run_'+str(runid) )
                create_group=True
            except:
                runid+=1
                continue
        group.attrs.create(name = 'isotropic', data=isotropic)
        group.attrs.create(name = 'datatype', data=datatype)
        group.attrs.create(name = 'wavetype', data=wavetype)
        group.attrs.create(name = 'crifactor', data=crifactor)
        group.attrs.create(name = 'crilimit', data=crilimit)
        group.attrs.create(name = 'dlon', data=dlon)
        group.attrs.create(name = 'dlat', data=dlat)
        group.attrs.create(name = 'step_of_integration', data=stepinte)
        group.attrs.create(name = 'lengthcell', data=lengthcell)
        group.attrs.create(name = 'alpha', data=alpha)
        group.attrs.create(name = 'beta', data=beta)
        group.attrs.create(name = 'sigma', data=sigma)
        group.attrs.create(name = 'lengthcellAni', data=lengthcellAni)
        group.attrs.create(name = 'anipara', data=anipara)
        group.attrs.create(name = 'xZone', data=xZone)
        group.attrs.create(name = 'alphaAni0', data=alphaAni0)
        group.attrs.create(name = 'betaAni0', data=betaAni0)
        group.attrs.create(name = 'sigmaAni0', data=sigmaAni0)
        group.attrs.create(name = 'alphaAni2', data=alphaAni2)
        group.attrs.create(name = 'sigmaAni2', data=sigmaAni2)
        group.attrs.create(name = 'alphaAni4', data=alphaAni4)
        group.attrs.create(name = 'sigmaAni4', data=sigmaAni4)
        group.attrs.create(name = 'comments', data=comments)
        if anipara==0 or isotropic:
            index0={'vel_iso': 0}
        elif anipara==1:
            index0={'vel_iso': 0, 'vel_rmod': 1, 'dm': 2, 'amp2': 3, 'psi2': 4, 'Acos2': 5, 'Asin2': 6}
        elif anipara==2:
            index0={'vel_iso': 0, 'vel_rmod': 1, 'dm': 2, 'amp2': 3, 'psi2': 4, 'Acos2': 5, 'Asin2': 6, 'amp4': 7, 'psi4': 8, 'Acos4': 9, 'Asin4': 10}
        for per in pers:
            subgroup=group.create_group(name='%g_sec'%( per ))
            outper=outdir+'/'+'%g'%( per ) +'_'+datatype
            if isotropic:
                outpfx=outper+'/'+qcpfx+str(alpha)+'_'+str(sigma)+'_'+str(beta)
            else:
                outpfx=outper+'/'+qcpfx+wavetype+'_'+str(alphaAni0)+'_'+str(sigmaAni0)+'_'+str(alphaAni2)+'_'+str(sigmaAni2)+'_'+str(betaAni0)
            v0fname     = outpfx+'_%g.1' %(per)
            dvfname     = outpfx+'_%g.1' %(per)+'_%_'
            azifname    = outpfx+'_%g.azi' %(per)
            residfname  = outpfx+'_%g.resid' %(per)
            reafname    = outpfx+'_%g.rea' %(per)
            resfname    = outpfx+'_%g.res' %(per)
            inArr=np.loadtxt(v0fname); v0Arr=inArr[:,2:]
            v0dset=subgroup.create_dataset(name='velocity', data=v0Arr)
            if not isotropic:
                lonlatArr=inArr[:,:2]; lonlatdset=subgroup.create_dataset(name='lons_lats', data=lonlatArr)
            inArr=np.loadtxt(dvfname); dvArr=inArr[:,2]
            dvdset=subgroup.create_dataset(name='Dvelocity', data=dvArr)
            inArr=np.loadtxt(azifname); aziArr=inArr[:,2:]
            azidset=subgroup.create_dataset(name='azi_coverage', data=aziArr)
            inArr=np.loadtxt(residfname)
            residdset=subgroup.create_dataset(name='residual', data=inArr)
            if not isotropic:
                inArr=np.loadtxt(reafname); reaArr=inArr[:,2:]
                readset=subgroup.create_dataset(name='resolution', data=reaArr)
                lonlatArr=inArr[:,:2]; lonlatdset_rea=subgroup.create_dataset(name='lons_lats_rea', data=lonlatArr)
            inArr=np.loadtxt(resfname); resArr=inArr[:,2:]
            resdset=subgroup.create_dataset(name='path_density', data=resArr)
            if deletetxt: shutil.rmtree(outper)
        if deletetxt and deleteall: shutil.rmtree(outdir)
        return
    
    def _get_basemap(self, projection='lambert', geopolygons=None, resolution='i'):
        """Get basemap for plotting results
        """
        # fig=plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
        minlon=self.attrs['minlon']
        maxlon=self.attrs['maxlon']
        minlat=self.attrs['minlat']
        maxlat=self.attrs['maxlat']
        lat_centre = (maxlat+minlat)/2.0
        lon_centre = (maxlon+minlon)/2.0
        if projection=='merc':
            m=Basemap(projection='merc', llcrnrlat=minlat-5., urcrnrlat=maxlat+5., llcrnrlon=minlon-5.,
                      urcrnrlon=maxlon+5., lat_ts=20, resolution=resolution)
            # m.drawparallels(np.arange(minlat,maxlat,dlat), labels=[1,0,0,1])
            # m.drawmeridians(np.arange(minlon,maxlon,dlon), labels=[1,0,0,1])
            m.drawparallels(np.arange(-80.0,80.0,5.0), labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,5.0), labels=[1,0,0,1])
            m.drawstates(color='g', linewidth=2.)
        elif projection=='global':
            m=Basemap(projection='ortho',lon_0=lon_centre, lat_0=lat_centre, resolution=resolution)
            # m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,1])
            # m.drawmeridians(np.arange(-170.0,170.0,10.0), labels=[1,0,0,1])
        elif projection=='regional_ortho':
            m1 = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution='l')
            m = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution=resolution,\
                llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/mapfactor, urcrnry=m1.urcrnry/3.5)
            m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,0],  linewidth=2,  fontsize=20)
            # m.drawparallels(np.arange(-90.0,90.0,30.0),labels=[1,0,0,0], dashes=[10, 5], linewidth=2,  fontsize=20)
            # m.drawmeridians(np.arange(10,180.0,30.0), dashes=[10, 5], linewidth=2)
            m.drawmeridians(np.arange(-170.0,170.0,10.0),  linewidth=2)
        elif projection=='lambert':
            distEW, az, baz=obspy.geodetics.gps2dist_azimuth(minlat, minlon, minlat, maxlon) # distance is in m
            distNS, az, baz=obspy.geodetics.gps2dist_azimuth(minlat, minlon, maxlat+2., minlon) # distance is in m
            m = Basemap(width=distEW, height=distNS, rsphere=(6378137.00,6356752.3142), resolution='l', projection='lcc',\
                lat_1=minlat, lat_2=maxlat, lon_0=lon_centre, lat_0=lat_centre+1)
            m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=1, dashes=[2,2], labels=[1,1,0,0], fontsize=15)
            m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1, dashes=[2,2], labels=[0,0,1,0], fontsize=15)
            # m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=0.5, dashes=[2,2], labels=[1,0,0,0], fontsize=5)
            # m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=0.5, dashes=[2,2], labels=[0,0,0,1], fontsize=5)
        m.drawcoastlines(linewidth=1.0)
        m.drawcountries(linewidth=1.)
        # m.drawmapboundary(fill_color=[1.0,1.0,1.0])
        m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        # m.drawlsmask(land_color='0.8', ocean_color='#99ffff')
        m.drawmapboundary(fill_color="white")
        try:
            geopolygons.PlotPolygon(inbasemap=m)
        except:
            pass
        return m
    
    def _get_lon_lat_arr(self, dataid):
        """Get longitude/latitude array
        """
        minlon=self.attrs['minlon']
        maxlon=self.attrs['maxlon']
        minlat=self.attrs['minlat']
        maxlat=self.attrs['maxlat']
        dlon=self[dataid].attrs['dlon']
        dlat=self[dataid].attrs['dlat']
        self.lons=np.arange((maxlon-minlon)/dlon+1)*dlon+minlon
        self.lats=np.arange((maxlat-minlat)/dlat+1)*dlat+minlat
        self.Nlon=self.lons.size; self.Nlat=self.lats.size
        self.lonArr, self.latArr = np.meshgrid(self.lons, self.lats)
        return
    
    def _numpy2ma(self, inarray, reason_n=None):
        """Convert input numpy array to masked array
        """
        if reason_n==None:
            outarray=ma.masked_array(inarray, mask=np.zeros(self.reason_n.shape) )
            outarray.mask[self.reason_n!=0]=1
        else:
            outarray=ma.masked_array(inarray, mask=np.zeros(reason_n.shape) )
            outarray.mask[reason_n!=0]=1
        return outarray
    
    def np2ma(self):
        """Convert numpy data array to masked data array
        """
        try:
            reason_n=self.reason_n
        except:
            raise AttrictError('No reason_n array!')
        self.vel_iso=self._numpy2ma(self.vel_iso)
        self.dv=self._numpy2ma(self.dv)
        self.pdens=self._numpy2ma(self.pdens)
        self.pdens1=self._numpy2ma(self.pdens1)
        self.pdens2=self._numpy2ma(self.pdens2)
        self.azicov1=self._numpy2ma(self.azicov1)
        self.azicov2=self._numpy2ma(self.azicov2)
        try:
            self.amp2=self._numpy2ma(self.amp2)
            self.psi2=self._numpy2ma(self.psi2)
        except:
            pass
        try:
            self.amp4=self._numpy2ma(self.amp4)
            self.psi4=self._numpy2ma(self.psi4)
        except:
            pass
 
        return
    
    def get_data4plot(self, dataid, period):
        """
        Get data for plotting
        =======================================================================================
        Input Parameters:
        dataid              - dataid (e.g. smooth_run_0, qc_run_0 etc.)
        period              - period
        ---------------------------------------------------------------------------------------
        generated data arrays:
        ----------------------------------- isotropic version ---------------------------------
        self.vel_iso        - isotropic velocity
        self.dv             - velocity perturbation
        self.pdens          - path density (R1 and R2)
        self.pdens1         - path density (R1)
        self.pdens2         - path density (R2)
        self.azicov1        - azimuthal coverage, squared sum method(0-10)
        self.azicov2        - azimuthal coverage, maximum value method(0-180)
        ---------------------------------- anisotropic version --------------------------------
        include all the array above(but will be converted to masked array), and
        self.psi2/amp2      - fast axis/amplitude for psi2 anisotropy
        self.psi4/amp4      - fast axis/amplitude for psi4 anisotropy
        self.cradius        - cone radius (resolution)
        self.reason_n       - array to represent valid/invalid data points
        =======================================================================================
        """
        self._get_lon_lat_arr(dataid)
        subgroup=self[dataid+'/%g_sec'%( period )]
        self.period=period
        self.datatype=self[dataid].attrs['datatype']
        try:
            self.isotropic=self[dataid].attrs['isotropic']
        except:
            self.isotropic=True
        if self.isotropic:
            self.vel_iso=subgroup['velocity'].value
            self.vel_iso=self.vel_iso.reshape(self.Nlat, self.Nlon)
            self.dv=subgroup['Dvelocity'].value
            self.dv=self.dv.reshape(self.Nlat, self.Nlon)
            self.pdens=subgroup['path_density'].value
            self.pdens=self.pdens.reshape(self.Nlat, self.Nlon)
            self.azicov1=(subgroup['azi_coverage'].value)[:,0]
            self.azicov1=self.azicov1.reshape(self.Nlat, self.Nlon)
            self.azicov2=(subgroup['azi_coverage'].value)[:,1]
            self.azicov2=self.azicov2.reshape(self.Nlat, self.Nlon)
        else:
            self.anipara=self[dataid].attrs['anipara']
            # initialize dataset
            self.vel_iso=np.zeros(self.lonArr.shape)
            if self.anipara!=0:
                self.amp2=np.zeros(self.lonArr.shape)
                self.psi2=np.zeros(self.lonArr.shape)
            if self.anipara==2:
                self.amp4=np.zeros(self.lonArr.shape)
                self.psi4=np.zeros(self.lonArr.shape)
            self.dv=np.zeros(self.lonArr.shape)
            self.pdens=np.zeros(self.lonArr.shape)
            self.pdens1=np.zeros(self.lonArr.shape)
            self.pdens2=np.zeros(self.lonArr.shape)
            self.azicov1=np.zeros(self.lonArr.shape)
            self.azicov2=np.zeros(self.lonArr.shape)
            self.cradius=np.zeros(self.lonArr.shape)
            self.reason_n=np.ones(self.lonArr.shape)
            # read data from hdf5 database
            lon_lat_array=subgroup['lons_lats'].value
            vel_iso=(subgroup['velocity'].value)[:,0]
            dv=subgroup['Dvelocity'].value
            if self.anipara!=0:
                amp2=(subgroup['velocity'].value)[:,3]
                psi2=(subgroup['velocity'].value)[:,4]
            if self.anipara==2:
                amp4=(subgroup['velocity'].value)[:,7]
                psi4=(subgroup['velocity'].value)[:,8]
            inlon=lon_lat_array[:,0]
            inlat=lon_lat_array[:,1]
            pdens=(subgroup['path_density'].value)[:,0]
            pdens1=(subgroup['path_density'].value)[:,1]
            pdens2=(subgroup['path_density'].value)[:,2]
            azicov1=(subgroup['azi_coverage'].value)[:,0]
            azicov2=(subgroup['azi_coverage'].value)[:,1]
            # cradius=(subgroup['resolution'].value)[:,0]
            for i in xrange(inlon.size):
                lon=inlon[i]
                lat=inlat[i]
                index = np.where((self.lonArr==lon)*(self.latArr==lat))
                # print index
                self.reason_n[index[0], index[1]]=0
                self.vel_iso[index[0], index[1]]=vel_iso[i]
                if self.anipara!=0:
                    self.amp2[index[0], index[1]]=amp2[i]
                    self.psi2[index[0], index[1]]=psi2[i]
                if self.anipara==2:
                    self.amp4[index[0], index[1]]=amp4[i]
                    self.psi4[index[0], index[1]]=psi4[i]
                self.dv[index[0], index[1]]=dv[i]
                self.pdens[index[0], index[1]]=pdens[i]
                self.pdens1[index[0], index[1]]=pdens1[i]
                self.pdens2[index[0], index[1]]=pdens2[i]
                self.azicov1[index[0], index[1]]=azicov1[i]
                self.azicov2[index[0], index[1]]=azicov2[i]
                # self.cradius[index[0], index[1]]=cradius[i]
            self.np2ma()
        return
            
    
    def plot_vel_iso(self, projection='lambert', fastaxis=False, geopolygons=None, showfig=True, vmin=None, vmax=None):
        """Plot isotropic velocity
        """
        m=self._get_basemap(projection=projection, geopolygons=geopolygons)
        x, y=m(self.lonArr, self.latArr)
        cmap = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        im=m.pcolormesh(x, y, self.vel_iso, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        cb = m.colorbar(im, "bottom", size="3%", pad='2%')
        cb.set_label('V'+self.datatype+' (km/s)', fontsize=12, rotation=0)
        plt.title(str(self.period)+' sec', fontsize=20)
        if fastaxis:
            try:
                self.plot_fast_axis(inbasemap=m)
            except:
                pass
        if showfig:
            plt.show()
        return
        
    def plot_fast_axis(self, projection='lambert', inbasemap=None, factor=1, showfig=False, psitype=2):
        """Plot fast axis(psi2 or psi4)
        """
        if inbasemap==None:
            m=self._get_basemap(projection=projection)
        else:
            m=inbasemap
        x, y=m(self.lonArr, self.latArr)
        if psitype==2:
            psi=self.psi2
        elif psitype==4:
            psi=self.psi4
        U=np.sin(psi)
        V=np.cos(psi)
        if factor!=None:
            x=x[0:self.Nlat:factor, 0:self.Nlon:factor]
            y=y[0:self.Nlat:factor, 0:self.Nlon:factor]
            U=U[0:self.Nlat:factor, 0:self.Nlon:factor]
            V=V[0:self.Nlat:factor, 0:self.Nlon:factor]
        Q = m.quiver(x, y, U, V, scale=50, width=0.001, headaxislength=0)
        if showfig:
            plt.show()
        return
    
    def plot_array(self, inarray, title='', label='', projection='lambert', fastaxis=False, geopolygons=None, showfig=True, vmin=None, vmax=None):
        """Plot input array
        """
        if inarray.shape!=self.lonArr.shape:
            raise ValueError('Shape of input array is not compatible with longitude/latitude array!')
        m=self._get_basemap(projection=projection, geopolygons=geopolygons)
        x, y=m(self.lonArr, self.latArr)
        cmap = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        im=m.pcolormesh(x, y, inarray, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        cb = m.colorbar(im, "bottom", size="3%", pad='2%')
        cb.set_label(label, fontsize=12, rotation=0)
        plt.title(title+str(self.period)+' sec', fontsize=20)
        if fastaxis:
            try:
                self.plot_fast_axis(inbasemap=m)
            except:
                pass
        if showfig:
            plt.show()
        return
    
    def generate_corrected_map(self, dataid, glbdir, outdir, pers=np.array([]), glbpfx='smpkolya_phv_R_', outpfx='smpkolya_phv_R_'):
        """
        Generate corrected global phave velocity map using a regional phase velocity map.
        =================================================================================================================
        Input Parameters:
        dataid              - dataid for regional phase velocity map
        glbdir              - location of global reference phase velocity map files
        outdir              - output directory
        pers                - period array for correction (default is 4)
        glbpfx              - prefix for global reference phase velocity map files
        outpfx              - prefix for output reference phase velocity map files
        -----------------------------------------------------------------------------------------------------------------
        Output format:
        outdir/outpfx+str(int(per))
        =================================================================================================================
        """
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        if pers.size==0:
            pers=np.append( np.arange(7.)*10.+40., np.arange(2.)*25.+125.)
        for per in pers:
            inglobalfname=glbdir+'/'+glbpfx+str(int(per))
            try:
                self.get_data4plot(dataid=dataid, period=per)
            except:
                print 'No regional data for period =',per,'sec'
                continue
            if not os.path.isfile(inglobalfname):
                print 'No global data for period =',per,'sec'
                continue
            outfname=outdir+'/'+outpfx+'%g' %(per)
            InglbArr=np.loadtxt(inglobalfname)
            outArr=InglbArr.copy()
            lonArr=self.lonArr.reshape(self.lonArr.size)
            latArr=self.latArr.reshape(self.latArr.size)
            vel_iso=ma.getdata(self.vel_iso)
            vel_iso=vel_iso.reshape(vel_iso.size)
            for i in xrange(InglbArr[:,0].size):
                lonG=InglbArr[i,0]
                latG=InglbArr[i,1]
                phVG=InglbArr[i,2]
                for j in xrange(lonArr.size):
                    lonR=lonArr[j]
                    latR=latArr[j]
                    phVR=vel_iso[j]
                    if abs(lonR-lonG)<0.05 and abs(latR-latG)<0.05 and phVR!=0:
                        outArr[i,2]=phVR
            np.savetxt(outfname, outArr, fmt='%g %g %.4f')
        return
    
    def plot_global_map(self, period, resolution='i', inglbpfx='./MAPS/smpkolya_phv_R', geopolygons=None, showfig=True, vmin=None, vmax=None):
        """
        Plot global phave velocity map 
        =================================================================================================================
        Input Parameters:
        period              - input period
        resolution          - resolution in Basemap object
        inglbpfx            - prefix of input global velocity map files
        geopolygons         - geopolygons for plotting
        showfig             - show figure or not
        vmin/vmax           - minimum/maximum value for plotting
        =================================================================================================================
        """
        inglbfname=inglbpfx+'_'+str(int(period))
        inArr = np.loadtxt(inglbfname)
        lonArr=inArr[:,0]
        lonArr[lonArr>180]=lonArr[lonArr>180]-360
        lonArr=lonArr.reshape(181, 360)
        latArr=inArr[:,1]
        latArr=latArr.reshape(181, 360)
        phvArr=inArr[:,2]
        phvArr=phvArr.reshape(181, 360)
        minlon=self.attrs['minlon']
        maxlon=self.attrs['maxlon']
        minlat=self.attrs['minlat']
        maxlat=self.attrs['maxlat']
        lat_centre = (maxlat+minlat)/2.0
        lon_centre = (maxlon+minlon)/2.0
        m=Basemap(projection='moll',lon_0=lon_centre, lat_0=lat_centre, resolution=resolution)
        x, y=m(lonArr, latArr)
        cmap = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        im=m.pcolormesh(x, y, phvArr, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        m.drawcoastlines(linewidth=1.0)
        cb = m.colorbar(im, "bottom", size="3%", pad='2%')
        cb.set_label('Vph'+' (km/s)', fontsize=12, rotation=0)
        plt.title(str(period)+' sec', fontsize=20)
        # m.readshapefile('./tectonicplates/PB2002_plates', 
        #         name='tectonic_plates', 
        #         drawbounds=True, 
        #         color='red')
        if showfig:
            plt.show()
        return
        
        
