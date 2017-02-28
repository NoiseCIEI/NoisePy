import ASDFDBase

dset=ASDFDBase.noiseASDF('CC_JdF_4_plotSta.h5')
# dset.read_stationtxt(stafile='/lustre/janus_scratch/howa1663/Seis_Data/SEED/station_plot.lst', source='CIEI', chans=['BHZ'])
# dset.get_limits_lonlat()

#dset.read_stationtxt(stafile='/work2/weisen/PAPER/FIG1/station_ChinaArray2.lst', source='CIEI', chans=['BHZ'], dnetcode='TA')
dset.my_plot_stations()
