import noisepy as npy
import numpy as np
datadir='/lustre/janus_scratch/life9360/ancc-1.0-0'
outdir='/lustre/janus_scratch/life9360/ancc-1.0-0'
stafile='/lustre/janus_scratch/life9360/ancc-1.0-0/Station.lst_forTest'
staLst=npy.StaLst();
### Read station list
staLst.ReadStaList(stafile); 
### Generate Predicted Phase V Dispersion
# staLst.GeneratePrePhaseDISPParallel(staLst, outdir=outdir+'/PREPHASE');

### aftan analysis
# Prepare database
# predir=outdir+'/PREPHASE_R';
# CHAN=[ [''] ]
# inftan=npy.InputFtanParam() 
# inftan.setInParam(tmin=4.0, tmax=70.0);  # Set Input aftan parameters
# staLst.MakeDirs(outdir=outdir, dirtout='DISP'); ### Important!
# StapairLst=npy.StaPairLst(); # 
# StapairLst.GenerateSPairLst(staLst, chanAll=CHAN); # Generate station pair list from station list
# StapairLst.set_PRE_PHASE(predir=predir);
# # Now do aftan
# L=len(StapairLst);
# Lm=int(L/20)
# for i in np.arange(Lm):
#     print i
#     tempstapair=StapairLst[i*20:(i+1)*20];
#     tempstapair.set_PRE_PHASE(predir=predir);
#     tempstapair.aftanSNRParallel(datadir=datadir, outdir=outdir, tin='COR', tout='COR', dirtin='COR', dirtout='DISP', inftan=inftan);
# tempstapair=StapairLst[Lm*20:];
# tempstapair.set_PRE_PHASE(predir=predir);
# tempstapair.aftanSNRParallel(datadir=datadir, outdir=outdir, tin='COR', tout='COR', dirtin='COR', dirtout='DISP', inftan=inftan);
# StapairLst.aftanSNRParallel(datadir=datadir, outdir=outdir, tin='COR', tout='COR', dirtin='COR', dirtout='DISP', inftan=inftan);

### Eikonal Tomography
per_array=np.arange((32-6)/2+1)*2.+6.
# staLst.GetTravelTimeFileParallel(staLst, per_array, datadir=datadir, outdir=outdir+'/Eikonal_out', \
#                 dirtin='DISP', minlon=-125, maxlon=-105, minlat=31, maxlat=50, dx=0.2, \
#                 filetype='phase', chpair=['', ''] )
# staLst.CheckTravelTimeCurvatureParallel(perlst=per_array, outdir=outdir+'/Eikonal_out', minlon=-125, maxlon=-105, minlat=31, maxlat=50, dx=0.2, filetype='phase');


datadir='/lustre/janus_scratch/life9360/ancc-1.0-0/Eikonal_out'
outdir='/lustre/janus_scratch/life9360/ancc-1.0-0/Eikonal_out'
# staLst.TravelTime2SlownessParallel(datadir=outdir, outdir=outdir, perlst=per_array, minlon=-125, maxlon=-105, minlat=31, maxlat=50, filetype='phase')
npy.Slowness2IsoAniMap(stafile=stafile, perlst=per_array, datadir=outdir, outdir=outdir, minlon=-125, maxlon=-105, minlat=31, maxlat=50,\
    dx=0.2, pflag=1, crifactor=12, minazi=-180, maxazi=180, N_bin=20)
