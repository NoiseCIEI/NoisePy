import instaseis
import obspy
import numpy as np
import noisepy
import matplotlib.pyplot as plt
from scipy import stats
from obspy.io.sac.util import obspy_to_sac_header
#import symData2d

Miso=1.0000e+24
dDelta=1;
Nsta=40;
Ni=12;
lat=0;
evla=0.0;
evlo=0.0;

outdir='./instaseis_ftan';
# db = instaseis.open_db("/home/lili/code/10s_PREM_ANI_FORCES")
#db = instaseis.open_db("/projects/life9360/instaseis_seismogram/10s_PREM_ANI_FORCES")
db = instaseis.open_db("/lustre/janus_scratch/life9360/instaseis_db")
source = instaseis.Source(
    latitude=evla, longitude=evlo, depth_in_m=1000,
    m_rr = Miso / 1E7,
    m_tt = Miso / 1E7,
    m_pp = Miso / 1E7,
    m_rt = Miso / 1E7,
    m_rp = Miso / 1E7,
    m_tp = Miso / 1E7,
    origin_time=obspy.UTCDateTime(2011, 1, 2, 3, 4, 5))

InstaStream=obspy.Stream();
DeltaArr=np.array([]);
for n in np.arange(Nsta):
    staname = str('S%03d' %n)
    lon=n*dDelta+Ni;
    DeltaArr=np.append(DeltaArr, lon)
    stla=0;
    stlo=lon;
    # print staname
    network="LF"
    receiver = instaseis.Receiver(
        latitude=lat, longitude=lon, network=network, station=staname)
    trZ = db.get_seismograms(source=source, receiver=receiver)[0]
    outfname=outdir+'/'+network+'.'+staname+'.SAC';
    trZ.stats.sac=obspy_to_sac_header(trZ.stats)
    trZ.stats.sac.evlo=evlo
    trZ.stats.sac.evla=evla
    trZ.stats.sac.stlo=stlo
    trZ.stats.sac.stla=stla
    #trZ.stats.sac.b=7.38470105346
    trZ.stats.sac.b=-1.53764949208
    trZ.stats.sac.e=trZ.stats.npts*trZ.stats.delta;
    dist, az, baz=obspy.geodetics.base.gps2dist_azimuth(evla, evlo, stla, stlo);
    trZ.stats.sac.az=az;
    trZ.stats.sac.baz=baz;
    trZ.stats.sac.dist=dist/1000.;
    trZ.stats.distance=dist/1000.;
    InstaStream.append(trZ)
    trZ.write(outfname,format='SAC');



VgrArr=np.array([]);
AmpArr=np.array([]);
DistArr=np.array([]);
per=12.;
for trace in InstaStream:
    staname = str('S%03d' %n)
    lon=n*dDelta+Ni
    stla=0;
    stlo=lon;
    nTr=noisepy.noisetrace(trace.data, trace.stats);
    # nTr=symData2d.symtrace(trace.data, trace.stats);
    DistArr=np.append(DistArr, nTr.stats.sac.dist);
    nTr.aftan(tmin=5, tmax=30);
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
    
    
plt.figure();
# plt.plot(DistArr, DistArr/VgrArr, 'o');
#plt.plot(DistArr, VgrArr, 'o');
plt.plot(DistArr, (VgrArr-VgrArr[0])/VgrArr[0]*100.,'o' );
plt.ylabel('Relative Difference in Vgr (%)');
#plt.ylabel('Vgr(km/s)');
plt.xlabel('Distance(km)');

plt.figure();
# plt.plot(DistArr, VgrArr, 'x');
# plt.plot(DistArr, AmpArr*np.sqrt(np.sin(DeltaArr*np.pi/180.) ));
plt.plot(DistArr, AmpArr*1e9,'o' );
plt.ylabel('Amplitude(nm)');
plt.xlabel('Distance(km)');

plt.figure();
# plt.plot(DistArr, VgrArr, 'x');
# plt.plot(DistArr, AmpArr*np.sqrt(np.sin(DeltaArr*np.pi/180.) ));
#CampArr=AmpArr*np.sqrt(np.sin(DeltaArr*np.pi/180.) )/np.sqrt(np.sin(DeltaArr[0]*np.pi/180.) )
#CampArr=AmpArr*np.sqrt(DistArr/DistArr[0] )  
CampArr=AmpArr*DistArr/ DistArr[0] 
#plt.plot(DistArr, (CampArr-CampArr[0])/CampArr[0]*100.,'o' );
plt.plot(DistArr, CampArr,'o' );
plt.ylabel('Amplitude(nm)');
plt.xlabel('Distance(km)');
# plt.axis([ DistArr.min(), DistArr.max(), CampArr.min(), CampArr.max()])
plt.show()
slope, intercept, r_value, p_value, std_err = stats.linregress(DistArr, DistArr/VgrArr);
print slope, intercept, r_value, p_value, std_err
    
    
# InstaStream.plot(type='section', norm_method='stream', alpha=1)
#     
