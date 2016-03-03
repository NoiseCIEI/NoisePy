import noisepy as npy
import numpy as np
import obspy
import warnings
stafile="/lustre/janus_scratch/life9360/Ref_TEST/station.lst"
datadir="/rc_scratch/life9360/ALASKA_receiver/seismograms_P_ALASKA"
outdir="/lustre/janus_scratch/life9360/Ref_TEST/my_receiverF"

staLst=npy.StaLst()
staLst.ReadStaList(stafile)
staLst.MakeDirs(outdir=outdir, dirtout='')
tempSLST=staLst[:20]
station=staLst[219]
L=int(len(staLst)/20);

datadir='/lustre/janus_scratch/life9360/for_Weisen/ALASKA_receiver/P_receiver_R';
outdir='/lustre/janus_scratch/life9360/Ref_TEST/stalst_ref_test';

# for i in np.arange(L):
#     print i
#     tempSLST=staLst[i*20:(i+1)*20]
#     tempSLST.PostProcessParallel(datadir=datadir, outdir=outdir);
#     tempSLST.PlotHSDataBase(datadir=outdir, outdir=outdir)
# tbeg=obspy.core.utcdatetime.UTCDateTime()
# staLst.PostProcessParallel(datadir=datadir, outdir=outdir);
# tend=obspy.core.utcdatetime.UTCDateTime()
# npy.PrintElaspedTime(tbeg, tend)
staLst.PlotHSDataBase(datadir=outdir, outdir=outdir);
# station.init_RefDataBase()
# # # # station.GetPathfromSOD(datadir=datadir)
# # # # station.fromSOD2Ref(datadir=datadir, outdir=outdir,PostFlag=True)
# station.ReadDeconvRef(datadir=datadir);
# station.PostProcess(outdir=outdir);
# # station.LoadPostDataBase(outdir=outdir)
# tbeg=obspy.core.utcdatetime.UTCDateTime()
# # station.PostProcess1(outdir=outdir)
# # station.PostProcess(outdir=outdir)
# datadir='/lustre/janus_scratch/life9360/Ref_TEST/my_receiverF';
# station.ReadHSDataBase(datadir=datadir);
# station.PlotHSDataBase(outdir='/lustre/janus_scratch/life9360/Ref_TEST');
# tend=obspy.core.utcdatetime.UTCDateTime()
# npy.PrintElaspedTime(tbeg, tend)

# station.PostProcess1(outdir=outdir)
# station.PostProcess2(outdir=outdir)


# np.savetxt('stre_'+str(int(station.Q1Lst[N].baz))+'_'+station.stacode+'_'+station.Q1Lst[N].eventT+'.out.back', station.Q1Lst[N].strback)
# station.SODRF(datadir=datadir, outdir=outdir)
# tempSLST.SODRFParallel(datadir=datadir, outdir=outdir)

