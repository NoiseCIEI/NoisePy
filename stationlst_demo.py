import noisepy as npy
#datadir='/lustre/janus_scratch/life9360/for_Weisen/ALASKA_receiver/P_receiver_R'
#outdir='/lustre/janus_scratch/life9360/for_Weisen/ALASKA_receiver/P_receiver_R_10'
#stafile='/lustre/janus_scratch/life9360/ALASKA_COR/station_ALASKA.lst'
#
#staLst=npy.StaLst()
#staLst.ReadStaList(stafile)
#staLst.MakeDirs(outdir=outdir, dirtout='')
#temlst=staLst[:12]
#temlst.DecimateParallel(factor=10, datadir=datadir, outdir=outdir, suffix='eqr')

stafile="/projects/life9360/YP_data/station.lst"
datadir="/projects/life9360/YP_data/seismograms_YP"
# outdir="/lustre/janus_scratch/life9360/COR_US_PO_test"

staLst=npy.StaLst()
staLst.ReadStaList(stafile)

