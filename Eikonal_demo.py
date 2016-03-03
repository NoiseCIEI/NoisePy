import noisepy as npy
import numpy as np
import obspy

datadir='/lustre/janus_scratch/life9360/ALASKA_COR'
outdir='/lustre/janus_scratch/life9360/ALASKA_COR/Eikonal_out'
stafile='/lustre/janus_scratch/life9360/ALASKA_COR/station_ALASKA.lst'
per_array=np.arange((32-6)/2+1)*2.+6.

#
staLst=npy.StaLst()
staLst.ReadStaList(stafile)
tempstaLst=staLst[:12];
# staLst.GetTravelTimeFileParallel(staLst, per_array, datadir=datadir, outdir=outdir, \
#                 dirtin='DISP', minlon=-170, maxlon=-130, minlat=51, maxlat=72, dx=0.2, \
#                 filetype='phase', chpair=['BHZ', 'BHZ'] )
# staLst.CheckTravelTimeCurvatureParallel(perlst=per_array, outdir=outdir, minlon=-170,\
#                             maxlon=-130, minlat=51, maxlat=72, dx=0.2, filetype='phase');

# staLst.TravelTime2SlownessParallel(datadir=outdir, outdir=outdir, perlst=per_array, minlon=-170,\
#                             maxlon=-130, minlat=51, maxlat=72, dx=0.2, filetype='phase')
npy.Slowness2IsoAniMap(stafile=stafile, perlst=np.array([22]), datadir=outdir, outdir=outdir, minlon=-170,\
                            maxlon=-130, minlat=51, maxlat=72, dx=0.2, pflag=1, crifactor=12, minazi=-180, maxazi=180, N_bin=20)

# Slowness2IsoAniMap(stafile, perlst=np.array([10]), datadir=outdir, outdir=outdir+'final',\
#     minlon=-170, maxlon=-130, minlat=51, maxlat=72, pflag=1, crifactor=12, minazi=-180, maxazi=180, N_bin=20, dx=0.2 )

# station=staLst[191];
# station.etTravelTimeFile(staLst, per=10, datadir, outdir, dirtin, \
#                         minlon, maxlon, minlat, maxlat, tin='COR', dx=0.2, filetype='phase', chpair=['LHZ', 'LHZ'] )
# dx_km=np.loadtxt('dx_km.txt');
# dy_km=np.loadtxt('dy_km.txt');
# station.TravelTime2Slowness(datadir=outdir,outdir=outdir, per=12., minlon=109, npts_x=201, minlat=21, npts_y=201 )#!/usr/bin/env python

