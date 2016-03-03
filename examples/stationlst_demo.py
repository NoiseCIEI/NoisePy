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

stafile="/lustre/janus_scratch/life9360/COR_US_PO/station.lst"
datadir="/lustre/janus_scratch/life9360/COR_US_PO"
outdir="/lustre/janus_scratch/life9360/COR_US_PO_test"

staLst=npy.StaLst()
staLst.ReadStaList(stafile)
#staLst.MakeDirs(outdir=outdir, dirtout='')
staLst.ChangeChName(datadir=datadir, outdir=outdir, Mbeg=[2012, 12], Mend=[2015, 5], inchan='HHZ', outchan='LHZ')
