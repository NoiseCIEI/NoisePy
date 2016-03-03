import noisepy as npy

### C3 Test
datadir='/lustre/janus_scratch/life9360/COR_YP_backup'
outdir='/lustre/janus_scratch/life9360/COR_YP_test2'
stafile='/lustre/janus_scratch/life9360/COR_YP_backup/station.lst'
CHAN=['BHE','BHN','BHZ']
#
staLst.ReadStaList(stafile)
staLst=npy.StaLst()
staLst.MakeDirs(outdir=outdir, dirtout='C3') ### Important!

StapairLst=npy.StaPairLst()
StapairLst.GenerateC3Lst(staLst, chan=['BHE','BHN','BHZ'], chanS=['BHE','BHN','BHZ'],
        tin='COR', tout='C3', tfactor=2, Lwin=1200, Tmin=5.0, Tmax=10.0, method = 'stehly', sepflag=True)

testLst=StapairLst[:3]
testLst.cc2c3(datadir=datadir, outdir=outdir, tin='COR', tout='C3')

