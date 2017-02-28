import obspy
import pyaftan
import matplotlib.pyplot as plt
import numpy as np
# 
# prefname='/lustre/janus_scratch/life9360/PRE_PHP/ses3d_2016_R/E000.98S43.pre'
# f='SES.174S110.SAC'
# f='./sac_data/SES.98S43.SAC'
# f='/work3/leon/COR_TEST/2008.APR/COR/109C/COR_109C_BHZ_R21A_BHZ.SAC'
f='./aftan_test_data/COR_SUMG_NEEM.SAC'
st=obspy.read(f)
tr=st[0]
tr1=pyaftan.aftantrace(tr.data, tr.stats)
# tr1.stats.sac.b=-0.0
tr1.makesym()
# tr1.aftanf77(piover4=-1., pmf=True, vmin=2.4, vmax=3.5, ffact=1. , tmin=4.0, tmax=30.0)
inftan=pyaftan.InputFtanParam()
tr1.aftanf77(pmf=inftan.pmf, piover4=inftan.piover4, vmin=inftan.vmin, vmax=inftan.vmax, tmin=inftan.tmin, tmax=inftan.tmax,
            tresh=inftan.tresh, ffact=inftan.ffact, taperl=inftan.taperl, snr=inftan.snr, fmatch=inftan.fmatch, nfin=inftan.nfin,
                npoints=inftan.npoints, perc=inftan.perc)
tr1.plotftan(plotflag=0)
plt.show()
# # 
# prefname='/lustre/janus_scratch/life9360/PRE_PHP/ses3d_2016_R/E000.98S47.pre'
# f='./sac_major_second_seismogram/SES.98S47.SAC'
# st=obspy.read(f)
# tr=st[0]
# tr2=symdata.ses3dtrace(tr.data, tr.stats)
# # tr1.stats.sac.b=-0.0
# tr2.aftan(piover4=-1., pmf=True, vmin=2.4, vmax=3.5, ffact=1. , tmin=6.0, tmax=15.0, phvelname=prefname)
# tr1.ftanparam.FTANcomp(tr2.ftanparam,  compflag=4)
# 
# plt.show()
