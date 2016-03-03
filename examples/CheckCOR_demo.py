import noisepy as npy

#datadir='/lustre/janus_scratch/life9360/Benchmark_TA'
#outdir='/lustre/janus_scratch/life9360/Benchmark_TA_devPy'
#stafile='/lustre/janus_scratch/life9360/Benchmark_TA/station.lst'
datadir='/rc_scratch/life9360/COR_YP'
outdir='/rc_scratch/life9360/COR_YP'
stafile='/lustre/janus_scratch/life9360/COR_YP_backup/station.lst'
allfname='/lustre/janus_scratch/life9360/COR_YP_backup/check.lst'
errorfname='/lustre/janus_scratch/life9360/COR_YP_backup/error.lst'
CHAN=[ ['BHE','BHN','BHZ'] ]

staLst=npy.StaLst()
staLst.ReadStaList(stafile)
StapairLst=npy.StaPairLst()
StapairLst.GenerateSPairLst(staLst, chanAll=CHAN)
StapairLst.CheckCOR(datadir=datadir, tin='COR', allfname=allfname, errorfname=errorfname)
