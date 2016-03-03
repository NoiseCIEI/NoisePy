import noisepy as npy

datadir='/lustre/janus_scratch/life9360/ALASKA_COR'
outdir='/lustre/janus_scratch/life9360/ALASKA_COR_test'
stafile='/lustre/janus_scratch/life9360/ALASKA_COR/station_ALASKA.lst'

CHAN=[ ['BHR','BHT','BHZ'] ]

staLst=npy.StaLst()
staLst.ReadStaList(stafile)
staLst.MakeDirs(outdir=outdir, dirtout='COR') ### Important!

StapairLst=npy.StaPairLst()
StapairLst.GenerateSPairLst(staLst, chanAll=CHAN)

## For Test
templst=StapairLst[:300]
templst.CHStartTimeParallel(datadir=datadir, outdir=outdir, tin='COR', tout='COR', dirtin='COR', dirtout='COR')
