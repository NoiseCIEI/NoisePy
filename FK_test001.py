import instaseis
import obspy
import numpy as np
import noisepy
import matplotlib.pyplot as plt
from scipy import stats
from obspy.io.sac.util import obspy_to_sac_header
from obspy.geodetics import kilometer2degrees
import glob
Miso=1.0000e+24
dDelta=1;
Nsta=20;
Ni=5;
lat=0;
evla=0.0;
evlo=0.0;

fkStream=obspy.Stream();
fkdatadir='/projects/life9360/code/fk/ak135_Q_15';
for sacfname in glob.glob(fkdatadir+"/*0"):
    Tr=obspy.read(sacfname)[0];
    Tr.stats.distance=Tr.stats.sac.dist;
    fkStream.append(Tr)

DeltaArr=np.array([]);
VgrArr=np.array([]);
AmpArr=np.array([]);
DistArr=np.array([]);
per=15.;
for trace in fkStream:
    nTr=noisepy.noisetrace(trace.data, trace.stats);
    # nTr=symData2d.symtrace(trace.data, trace.stats);
    DistArr=np.append(DistArr, nTr.stats.sac.dist);
    DeltaArr=np.append(DeltaArr, kilometer2degrees(nTr.stats.sac.dist) );
    nTr.aftan(tmin=5, tmax=30,ffact=10.);
    # nTr.getSNR();
    
    a_per = nTr.ftanparam.arr1_1[1,:];
    gv = nTr.ftanparam.arr1_1[2,:];
    pv = nTr.ftanparam.arr1_1[3,:];
    amp = nTr.ftanparam.arr1_1[8,:];
    # amp=np.append(nTr.SNRParam.amp_s, np.zeros(36))
    tempLper=a_per[a_per<per]
    tempUper=a_per[a_per>per]
    if len(tempLper)==0 or len(tempUper)==0:
        raise ValueError('Wrong period!');
    else:
        Lgv=gv[a_per<per][-1]
        Ugv=gv[a_per>per][0]
        Lpv=pv[a_per<per][-1]
        Upv=pv[a_per>per][0]
        Lamp=amp[a_per<per][-1]
        Uamp=amp[a_per>per][0]
        Lper=tempLper[-1]
        Uper=tempUper[0]
        
        Vph=(Upv - Lpv)/(Uper - Lper)*(per - Lper) + Lpv;
        Vgr=(Ugv - Lgv)/(Uper - Lper)*(per - Lper) + Lgv;
        Amp=(Uamp - Lamp)/(Uper - Lper)*(per - Lper) + Lamp;
    
    VgrArr=np.append(VgrArr, Vgr);
    # AmpArr=np.append(AmpArr, nTr.SNRParam.amp_s[0]);
    AmpArr=np.append(AmpArr, Amp);
    
minindex=np.argmin(DistArr)
plt.figure();
# plt.plot(DistArr, DistArr/VgrArr, 'o');
# plt.plot(DistArr, VgrArr, 'o');
plt.plot(DistArr, (VgrArr-VgrArr[minindex])/VgrArr[minindex]*100.,'o' );
plt.title('original aftan')
# plt.title('modified aftan')

plt.ylabel('Relative Difference in Vgr (%)');
# plt.ylabel('Vgr(km/s)');
plt.xlabel('Distance(km)');

plt.figure();
# plt.plot(DistArr, VgrArr, 'x');
# plt.plot(DistArr, AmpArr*np.sqrt(np.sin(DeltaArr*np.pi/180.) ));
plt.plot(DistArr, AmpArr*1e9,'o' );
plt.title('original aftan')
# plt.title('modified aftan')
plt.ylabel('Amplitude(nm)');
plt.xlabel('Distance(km)');

plt.figure();
# plt.plot(DistArr, VgrArr, 'x');
# plt.plot(DistArr, AmpArr*np.sqrt(np.sin(DeltaArr*np.pi/180.) ));
# CampArr=AmpArr*np.sqrt( np.sin(DeltaArr*np.pi/180.) ) /np.sqrt(np.sin(DeltaArr[minindex]*np.pi/180.) )
# CampArr=AmpArr*np.sqrt(DistArr/DistArr[minindex] )
# CampArr=AmpArr*np.sin(DeltaArr*np.pi/180.)  /np.sin(DeltaArr[0]*np.pi/180.) 
CampArr=AmpArr*DistArr/ DistArr[minindex]
# CampArr=AmpArr*(DistArr/ DistArr[0])**(0.8)
plt.plot(DistArr, (CampArr-CampArr[minindex])/CampArr[minindex]*100.,'o' );
# plt.plot(DistArr, CampArr,'o' );
plt.title('original aftan')
# plt.title('modified aftan')
plt.ylabel('Relative Difference in Corrected Amp (%)');
plt.xlabel('Distance(km)');


# plt.axis([ DistArr.min(), DistArr.max(), CampArr.min(), CampArr.max()])
plt.show()
# slope, intercept, r_value, p_value, std_err = stats.linregress(DistArr, DistArr/VgrArr);
slope, intercept, r_value, p_value, std_err = stats.linregress(DistArr, CampArr*1e9);
print slope, intercept, r_value, p_value, std_err
    
    
fkStream.plot(type='section',  alpha=1)#     
