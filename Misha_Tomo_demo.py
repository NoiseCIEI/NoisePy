import noisepy as npy
import numpy as np
import obspy
import GeoPoint
### C3 Test
datadir='/lustre/janus_scratch/life9360/ALASKA_COR'
outdir='/lustre/janus_scratch/life9360/ALASKA_test'
stafile='/lustre/janus_scratch/life9360/ALASKA_COR/station_ALASKA.lst'
CHAN=['BHE','BHN','BHZ']
per_array=np.arange((32-6)/2+1)*2.+6.

# staLst=npy.StaLst()
# staLst.ReadStaList(stafile)
# StapairLst=npy.StaPairLst()
# StapairLst.GenerateSPairLst(staLst, stationflag=2)
# tbeg=obspy.core.utcdatetime.UTCDateTime()
# StapairLst.getMishaInputParallel(outdir=outdir, per_array=per_array, datadir=datadir, tin='COR', dirtin='DISP', chpairList=[['BHZ', 'BHZ']], outPRE='MISHA_in_')
# tend=obspy.core.utcdatetime.UTCDateTime()
# npy.PrintElaspedTime(tbeg, tend)

## Initial Run
# npy.RunMishaSmooth(10.0, datadir=outdir, outdir='/lustre/janus_scratch/life9360/ALASKA_tomo_test', minlon=185, maxlon=230, minlat=50, maxlat=72)
# npy.RunMishaSmoothParallel(per_array, datadir=outdir, outdir='/lustre/janus_scratch/life9360/ALASKA_tomo_test', minlon=185, maxlon=230, minlat=50, maxlat=72)
# smooth_pre='N_INIT_'
# for per in per_array:
#     indir='/lustre/janus_scratch/life9360/ALASKA_tomo_test/'+'%g' %(per)+'_ph';
#     infname=indir+'/'+smooth_pre+'3000_500_100_'+'%g' %(per)+'.1'
#     GeoPoint.PlotTomoMap(infname, title='Phase Velocity Map'+'( '+'%g' %(per)+' sec )')
# 
outdir='/lustre/janus_scratch/life9360/ALASKA_tomo_test'
npy.RunMishaQCParallel(per_array, isoFlag=False, datadir=outdir, outdir=outdir, minlon=185, maxlon=230, minlat=50, maxlat=72)
# # npy.RunMishaQC(20., isoFlag=False, datadir=outdir, outdir=outdir, minlon=185, maxlon=230, minlat=50, maxlat=72)
QC_pre='QC_AZI_R_1200_200_1000_100_1_'
for per in per_array:
    indir='/lustre/janus_scratch/life9360/ALASKA_tomo_test/'+'%g' %(per)+'_ph';
    infname=indir+'/'+QC_pre+'%g' %(per)+'.1'
    GeoPoint.PlotTomoMap(infname, title='Phase Velocity Map'+'( '+'%g' %(per)+' sec )')
