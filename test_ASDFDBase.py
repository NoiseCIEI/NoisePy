import ASDFDBase

dset=ASDFDBase.noiseASDF('../COR_TA_TEST.h5')

# dset.write_stationtxt('./test_sta.lst')
dset.read_stationtxt('/work3/leon/COR_TEST/station.lst')
# dset.xcorr_stack('/work3/leon/COR_TEST', 2008, 1, 2009, 12, outdir='/work3/leon/COR_TEST', inchannels=['BHZ'])
dset.xcorr_stack_mp('/work3/leon/COR_TEST', '/work3/leon/COR_TEST', 2008, 1, 2009, 12)#, inchannels=['BHZ'])
# dset.xcorr_stack('/work3/leon/COR_TEST', 2008, 1, 2009, 12)
# dset.write_stationtxt('./test_sta3.lst')
# dset.write_staxml('../test.xml')
# dset.add_stationxml('../test.xml')

dset.xcorr_rotation(outdir='.')