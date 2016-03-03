import noisepy as npy

#datadir='/lustre/janus_scratch/life9360/Benchmark_TA'
#outdir='/lustre/janus_scratch/life9360/Benchmark_TA_devPy'
#stafile='/lustre/janus_scratch/life9360/Benchmark_TA/station.lst'
datadir='/lustre/janus_scratch/life9360/COR_YP_backup'
outdir='/lustre/janus_scratch/life9360/COR_YP_backup'
stafile='/lustre/janus_scratch/life9360/COR_YP_backup/station.lst'
CHAN=[ ['BHE','BHN','BHZ'] ]

staLst=npy.StaLst()
staLst.ReadStaList(stafile)
staLst.MakeDirs(outdir=outdir, dirtout='COR') ### Important!

StapairLst=npy.StaPairLst()
StapairLst.GenerateSPairLst(staLst, chanAll=CHAN)
StapairLst.StackingParallel(datadir=datadir, outdir=outdir, tin='COR', tout='COR', Mbeg= [2008, 1], Mend=[2009, 12])
StapairLst.RotationParellel(datadir=outdir, outdir=outdir, tin='COR', tout='COR')
# Makesym
staLst.MakeDirs(outdir=outdir, dirtout='SYM') ### Important!
StapairLst1=npy.StaPairLst()
StapairLst1.GenerateSPairLst(staLst, chanAll=[ ['BHR','BHT','BHZ'] ])
StapairLst1.MakesymParallel(datadir=outdir, outdir=outdir, tin='COR', tout='SYM', dirtin='COR', dirtout='SYM' )

