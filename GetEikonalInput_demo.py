import noisepy as npy
import numpy as np
import obspy

datadir='/projects/life9360/SYN_TEST_ARTIE/DATA/TOMO_s29ea'
outdir='/projects/life9360/SYN_TEST_ARTIE/DATA/TOMO_s29ea/TEST'
stafile='/projects/life9360/SYN_TEST_ARTIE/DATA/TOMO_s29ea/station.lst'
per_array=np.arange((32-6)/2+1)*2.+6.
#
staLst=npy.StaLst()
staLst.ReadStaList(stafile)
station=staLst[0]
# npts_x, npts_y=station.GetTravelTimeFile(staLst, 12, datadir=datadir, outdir=outdir,
#         dirtin='SAC_Z', minlon=109, maxlon= 149 , minlat=21, maxlat= 61 )
npy.GetdxdyDataBase()
# station.CheckTravelTimeCurvature(outdir=outdir, per=12., minlon=109, npts_x=201, minlat=21, npts_y=201)
dx_km=np.loadtxt('dx_km.txt');
dy_km=np.loadtxt('dy_km.txt');
station.TravelTime2Slowness(datadir=outdir,outdir=outdir, per=12., minlon=109, npts_x=201, minlat=21, npts_y=201 )
# StapairLst=npy.StaPairLst()
# StapairLst.GenerateSPairLst(staLst, stationflag=2)
# tbeg=obspy.core.utcdatetime.UTCDateTime()
# StapairLst.getTomoInput(outdir=outdir, per_array=per_array, datadir=datadir, tin='COR', dirtin='DISP', chpairList=[['BHZ', 'BHZ']] )
# tend=obspy.core.utcdatetime.UTCDateTime()
# npy.PrintElaspedTime(tbeg, tend)#!/usr/bin/env python

