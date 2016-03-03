import GeoPoint as gpt;
import numpy as np
import noisepy as npy
# tomodatadir='/lustre/janus_scratch/life9360/ALASKA_tomo_final';
# tomof_pre='QC_AZI_R_1200_200_1000_100_1_';
# tomof_sfx='.1';
# outdir='/lustre/janus_scratch/life9360/ALASKA_TOMO/geomap';
# per_array=np.arange((32-6)/2+1)*2.+6.;
# per_arr2=np.arange(6)*5.+35.;
# per_array=np.append(per_array,per_arr2);
# mapdata=gpt.MapDatabase(tomodatadir=tomodatadir, tomof_pre=tomof_pre, tomof_sfx=tomof_sfx, perarray=per_array);
# mapdata.ReadTomoResult(datatype='gr');
# mapdata.TomoMap2GeoPoints(datatype='gr');
# # mapdata.geomap.SavePhDisp(outdir=outdir);
# mapdata.geomap.SaveGrDisp(outdir=outdir);

#
datadir='/lustre/janus_scratch/life9360/ALASKA_TOMO/geomap';
outdir='/lustre/janus_scratch/life9360/ALASKA_TOMO/StationDISP';
stafile='/lustre/janus_scratch/life9360/ALASKA_COR/station_ALASKA.lst';
staLst=npy.StaLst();
staLst.ReadStaList(stafile);
# staLst.getDISP(datadir=datadir, minlon=185, maxlon=230, minlat=50, maxlat=72, dlon=0.5, dlat=0.5, outdir=outdir, fSFX='.grv');
station=staLst[0];
staLst.CheckDispersion(datadir=outdir,minlon=185, maxlon=230, minlat=50, maxlat=72)