import noisepy as npy
import numpy as np
import obspy
### C3 Test
datadir='/lustre/janus_scratch/life9360/ALASKA_COR'
outdir='/lustre/janus_scratch/life9360/ALASKA_test'
stafile='/lustre/janus_scratch/life9360/ALASKA_COR/station_ALASKA.lst'
CHAN=['BHE','BHN','BHZ']
per_array=np.arange((32-6)/2+1)*2.+6.
#
staLst=npy.StaLst()
staLst.ReadStaList(stafile)
StapairLst=npy.StaPairLst()
StapairLst.GenerateSPairLst(staLst, stationflag=2)
tbeg=obspy.core.utcdatetime.UTCDateTime()
StapairLst.getTomoInput(outdir=outdir, per_array=per_array, datadir=datadir, tin='COR', dirtin='DISP', chpairList=[['BHZ', 'BHZ']] )
tend=obspy.core.utcdatetime.UTCDateTime()
npy.PrintElaspedTime(tbeg, tend)
