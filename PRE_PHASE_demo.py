import noisepy as npy

datadir='/lustre/janus_scratch/life9360/ancc-1.0-0'
outdir='/lustre/janus_scratch/life9360/ancc-1.0-0'
stafile='/lustre/janus_scratch/life9360/ancc-1.0-0/Station.lst_forTest'
staLst=npy.StaLst()
staLst.ReadStaList(stafile)
staLst.GeneratePrePhaseDISP(staLst, outdir='./testP');

predir='/projects/life9360/code/PRE_PHASE/WUS_R'
CHAN=[ ['BHZ'] ]
# inftan=npy.InputFtanParam()
# inftan.setInParam(tmin=4.0, tmax=70.0)  # Set Input aftan parameters
# staLst=npy.StaLst()
# staLst.ReadStaList(stafile)
# staLst.MakeDirs(outdir=outdir, dirtout='DISP') ### Important!
# 
# StapairLst=npy.StaPairLst()
# StapairLst.GenerateSPairLst(staLst, chanAll=CHAN)
# #StapairLst.set_PRE_PHASE(predir=predir)
# ## For Test
# templst=StapairLst[:300]
# templst.set_PRE_PHASE(predir=predir)
# templst.aftanSNRParallel(datadir=datadir, outdir=outdir, tin='COR', tout='COR', dirtin='COR', dirtout='DISP', inftan=inftan)

