# -*- coding: utf-8 -*-
"""
A python module to run surface wave Eikonal tomography
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
    Lin, Fan-Chi, Michael H. Ritzwoller, and Roel Snieder. "Eikonal tomography: surface wave tomography by phase front tracking across a regional broad-band seismic array."
        Geophysical Journal International 177.3 (2009): 1091-1110.
    Lin, Fan-Chi, and Michael H. Ritzwoller. "Helmholtz surface wave tomography for isotropic and azimuthally anisotropic structure."
        Geophysical Journal International 186.3 (2011): 1104-1120.
"""
import numpy as np
import numpy.ma as ma
import h5py, pyasdf
import os, shutil
from subprocess import call
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
import colormaps
import obspy
import field2d_earth

class EikonalTomoDataSet(h5py.File):
    
    def set_input_parameters(self, minlon, maxlon, minlat, maxlat, pers=np.array([]), dlon=0.2, dlat=0.2):
        """
        Set input parameters for tomographic inversion.
        =================================================================================================================
        Input Parameters:
        minlon, maxlon  - minimum/maximum longitude
        minlat, maxlat  - minimum/maximum latitude
        pers            - period array, default = np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
        dlon, dlat      - longitude/latitude interval
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
        self.attrs.create(name = 'dlon', data=dlon)
        self.attrs.create(name = 'dlat', data=dlat)
        Nlon=(maxlon-minlon)/dlon+1
        Nlat=(maxlat-minlat)/dlat+1
        self.attrs.create(name = 'Nlon', data=Nlon)
        self.attrs.create(name = 'Nlat', data=Nlat)
        return
    
    def xcorr_eikonal(self, inasdffname, workingdir, fieldtype='Tph', channel='ZZ', data_type='FieldDISPpmf2interp', runid=0):
        create_group=False
        while (not create_group):
            try:
                group=self.create_group( name = 'Eikonal_run_'+str(runid) )
                create_group=True
            except:
                runid+=1
                continue
        group.attrs.create(name = 'fieldtype', data=fieldtype)
        inDbase=pyasdf.ASDFDataSet(inasdffname)
        pers = self.attrs['period_array']
        minlon=self.attrs['minlon']
        maxlon=self.attrs['maxlon']
        minlat=self.attrs['minlat']
        maxlat=self.attrs['maxlat']
        dlon=self.attrs['dlon']
        dlat=self.attrs['dlat']
        fdict={ 'Tph': 2, 'Tgr': 3}
        evLst=inDbase.waveforms.list()
        for per in pers:
            del_per=per-int(per)
            if del_per==0.:
                persfx=str(int(per))+'sec'
            else:
                dper=str(del_per)
                persfx=str(int(per))+'sec'+dper.split('.')[1]
            working_per=workingdir+'/'+str(per)+'sec'
            per_group=group.create_group( name='%g_sec'%( per ) )
            for evid in evLst:
                netcode1, stacode1=evid.split('.')
                try:
                    subdset = inDbase.auxiliary_data[data_type][netcode1][stacode1][channel][persfx]
                except KeyError:
                    print 'No travel time field for: '+evid
                    continue
                lat1, elv1, lon1=inDbase.waveforms[evid].coordinates.values()
                if lon1<0.:
                    lon1+=360.
                dataArr = subdset.data.value
                field2d=field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon, minlat=minlat, maxlat=maxlat, dlat=dlat, period=per, fieldtype=fieldtype)
                # return field2d
                Zarr=dataArr[:, fdict[fieldtype]]
                distArr=dataArr[:, 5]
                field2d.read_array(lonArr=np.append(lon1, dataArr[:,0]), latArr=np.append(lat1, dataArr[:,1]), ZarrIn=np.append(0., distArr/Zarr) )
                outfname=evid+'_'+fieldtype+'_'+channel+'.lst'
                field2d.interp_surface(workingdir=working_per, outfname=outfname)
                field2d.check_curvature(workingdir=working_per, outpfx=evid+'_'+channel+'_')
                field2d.gradient_qc(workingdir=working_per, evlo=lon1, evla=lat1, inpfx=evid+'_'+channel+'_', nearneighbor=True, cdist=None)
                field2d.evlo=lon1; field2d.evla=lat1
                # save data to hdf5 dataset
                event_group=per_group.create_group(name=evid)
                event_group.attrs.create(name = 'evlo', data=lon1)
                event_group.attrs.create(name = 'evla', data=lat1)
                appVdset     = event_group.create_dataset(name='appV', data=field2d.appV)
                reason_ndset = event_group.create_dataset(name='reason_n', data=field2d.reason_n)
                proAngledset = event_group.create_dataset(name='proAngle', data=field2d.proAngle)
                azdset       = event_group.create_dataset(name='az', data=field2d.az)
                bazdset      = event_group.create_dataset(name='baz', data=field2d.baz)
                Tdset        = event_group.create_dataset(name='travelT', data=field2d.Zarr)
                londset      = event_group.create_dataset(name='lonArr', data=field2d.lonArr)
                latdset      = event_group.create_dataset(name='latArr', data=field2d.latArr)
                # return field2d
        return
    
    def eikonal_stacking(self, runid=0):
        
        pers = self.attrs['period_array']
        minlon=self.attrs['minlon']
        maxlon=self.attrs['maxlon']
        minlat=self.attrs['minlat']
        maxlat=self.attrs['maxlat']
        dlon=self.attrs['dlon']
        dlat=self.attrs['dlat']
        Nlon=self.attrs['Nlon']
        Nlat=self.attrs['Nlat']
        group=self['Eikonal_run_'+str(runid)]
        
        for per in pers:
            per_group=group['%g_sec'%( per )]
            Treason=np.ones((Nlat-4, Nlon-4))
            Nmeasure=np.zeros((Nlat-4, Nlon-4))
            velArr=np.zeros((Nlat-4, Nlon-4))
            for evid in per_group.keys():
                event_group=per_group[evid]
                reason_n=event_group['reason_n'].value
                appV=event_group['appV'].value
                velArr[reason_n==0]+=appV[reason_n==0]
                oneArr=np.ones((Nlat-4, Nlon-4))
                oneArr[reason_n!=0]=0
                Nmeasure+=oneArr
        velArr[Nmeasure>15]=velArr[Nmeasure>15]/Nmeasure[Nmeasure>15]
        # self.velArr=velArr
        self.reason_n=np.zeros((Nlat-4, Nlon-4))
        self.reason_n[Nmeasure<15]=1
        self.velArr=self._numpy2ma(velArr)
        
        
        self.lons=np.arange((maxlon-minlon)/dlon-3)*dlon+minlon+2*dlon
        self.lats=np.arange((maxlat-minlat)/dlat-3)*dlat+minlat+2*dlat
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
    
    def _get_lon_lat_arr(self, dataid):
        """Get longitude/latitude array
        """
        minlon=self.attrs['minlon']
        maxlon=self.attrs['maxlon']
        minlat=self.attrs['minlat']
        maxlat=self.attrs['maxlat']
        dlon=self[dataid].attrs['dlon']
        dlat=self[dataid].attrs['dlat']
        self.lons=np.arange((maxlon-minlon)/dlon+1)*dlon+minlon+2*dlon
        self.lats=np.arange((maxlat-minlat)/dlat+1)*dlat+minlat+2*dlat
        self.Nlon=self.lons.size-4; self.Nlat=self.lats.size-4
        self.lonArr, self.latArr = np.meshgrid(self.lons, self.lats)
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
        m.drawstates()
        m.drawmapboundary(fill_color="white")
        try:
            geopolygons.PlotPolygon(inbasemap=m)
        except:
            pass
        return m
    
            
    
    def plot_vel_iso(self, projection='lambert', fastaxis=False, geopolygons=None, showfig=True, vmin=None, vmax=None):
        """Plot isotropic velocity
        """
        m=self._get_basemap(projection=projection, geopolygons=geopolygons)
        x, y=m(self.lonArr, self.latArr)
        cmap = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        im=m.pcolormesh(x, y, self.velArr, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        cb = m.colorbar(im, "bottom", size="3%", pad='2%')
        # cb.set_label('V'+self.datatype+' (km/s)', fontsize=12, rotation=0)
        # plt.title(str(self.period)+' sec', fontsize=20)
        # if fastaxis:
        #     try:
        #         self.plot_fast_axis(inbasemap=m)
        #     except:
        #         pass
        if showfig:
            plt.show()
        
        
    