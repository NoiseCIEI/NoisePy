import noisepy as npy

datadir='/lustre/janus_scratch/life9360/for_Weisen/ALASKA_receiver/P_receiver_R'
stafile='/lustre/janus_scratch/life9360/ALASKA_COR/station_ALASKA.lst'
staLst=npy.StaLst()

staLst.ReadStaList(stafile)
temlst=staLst[:12]
temlst.addHSlownessParallel(datadir=datadir, suffix='eqr')


