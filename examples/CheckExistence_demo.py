import noisepy as npy

dir1='/lustre/janus_scratch/life9360/COR_YP'
dir2='/lustre/janus_scratch/life9360/COR_YP'
stafile='/lustre/janus_scratch/life9360/COR_YP/station.lst'
outfname1='/lustre/janus_scratch/life9360/COR_YP/exist1.lst'
outfname2='/lustre/janus_scratch/life9360/COR_YP/exist2.lst'
outfnameall='/lustre/janus_scratch/life9360/COR_YP/notexistall.lst'
CHAN=[ ['BHR','BHT','BHZ'] ]

staLst=npy.StaLst()
staLst.ReadStaList(stafile)

StapairLst=npy.StaPairLst()
StapairLst.GenerateSPairLst(staLst, chanAll=CHAN)

## For Test
StapairLst.CheckExistence(outfname1=outfname1, dir1=dir1, t1='COR', dirt1='COR',
    outfname2=outfname2, dir2=dir2, t2='C3', dirt2='C3', outfnameall=outfnameall)
