import obspy
import noisepy as npy
import matplotlib.pyplot as plt
import numpy as np
# f='/lustre/janus_scratch/life9360/ALASKA_COR/COR/SAW/COR_SAW_BHZ_SII_BHZ.SAC'
# f1='/lustre/janus_scratch/life9360/SES3D_WorkingDir/OUTPUT_SAC_ak135/LF.S005032..BXZ.SAC'
f1='/projects/life9360/SYN_TEST_ARTIE/DATA/TOMO_1D.ak135/SAC_Z/S017021.SS.BXZ.sem.sac'
f1='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/OUTPUT_SAC_EA_10sec_1km_001/LF.EA123S32..BXZ.SAC'
# f2='/projects/life9360/SYN_TEST_ARTIE/DATA/TOMO_1D.ak135/SAC_Z/S005035.SS.BXZ.sem.sac'
# f1='INSTASEIS.LXZ.SAC'
# f2='/projects/life9360/SYN_TEST_ARTIE/DATA/TOMO_1D.ak135/SAC_Z/S019025.SS.BXZ.sem.sac'
# f2='/lustre/janus_scratch/life9360/SES3D_WorkingDir/OUTPUT_SAC/LF.S005032..BXZ.SAC'
# f2='/projects/life9360/SYN_TEST_ARTIE/DATA/TOMO_1D.ak135/SAC_Z/S006001.SS.BXZ.sem.sac'
# f2='/lustre/janus_scratch/life9360/SES3D_WorkingDir/OUTPUT_SAC_vel/LF.S002008..BXZ.SAC'
# f='./instaseis_ftan/LF.S000.SAC'
# f2='./instaseis_ftan/LF.S000.SAC'
f='./instaseis_ftan/LF.S000.SAC'
# f2='./instaseis_ftan/LF.S000.SAC'
#prefname='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/OUTPUT_DISP_EA_10sec_1km_001/PREPHASE_R/NKNT.EA123S32.pre'
prefname='/projects/life9360/code/PRE_PHASE/ALASKA_L/CLCO.N25K.pre'
f='/projects/life9360/code/fk/ak135_Q_15_1/30.grn.0'
f='/projects/life9360/code/CPSPy/sac_dir/B00511ZVF.sac'
f='/lustre/janus_scratch/life9360/sw4_working_dir_trials/ak135/SW4.38S1000.u'
st=obspy.read(f)
tr=st[0]
tr1=npy.noisetrace(tr.data, tr.stats)
tr1.aftan(piover4=-1., pmf=True, tmin=2.0, tmax=50.0, phvelname=prefname)
# st=obspy.read(f2)
# tr=st[0]
# tr2=npy.noisetrace(tr.data*2., tr.stats)
# tr2.aftan(piover4=-1., pmf=True, tmin=2.0, tmax=50.0, phvelname=prefname)
# tr1=npy.noisetrace(tr.data, tr.stats)
# tr1.aftan(piover4=0, pmf=True, tmin=2.0, tmax=50.0, phvelname=prefname)
# tr1.ftanparam.FTANcomp(tr.ftanparam)
# st=obspy.read(f2)
# tr=st[0]
# tr2=npy.noisetrace(tr.data, tr.stats)
# tr2.aftan(piover4=0., pmf=True, tmin=2.0, tmax=100.0,phvelname='ak135.disp')
# tr1.ftanparam.writeDISP(f)
# tr1.getSNR (tmin=4.0, tmax=70.0)
# tr1.SNRParam.writeAMPSNR(f)
# tr1.ftanparam.FTANcomp(tr2.ftanparam)
# 
# InAK135Arr=np.loadtxt('prem.disp')
# T=InAK135Arr[:,0];
# V=InAK135Arr[:,1];
# ax = plt.subplot()
# ax.plot(T, V, '--r', lw=3);
# # 
# InAK135Arr=np.loadtxt('ak135.disp')
# T=InAK135Arr[:,0];
# V=InAK135Arr[:,1];
# ax = plt.subplot()
# ax.plot(T, V, '--g', lw=3);
tr1.plotftan()
plt.show()


