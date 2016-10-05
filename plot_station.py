import ASDFDBase
import GeoPolygon

basins=GeoPolygon.GeoPolygonLst()
basins.ReadGeoPolygonLst('basin1')

dset=ASDFDBase.noiseASDF('./EA_chinaarray2.h5')
#dset.read_stationtxt(stafile='station.lst', source='CIEI', chans=['BHZ', 'BHE', 'BHN'], dnetcode='TA')

#dset.read_stationtxt(stafile='/work2/weisen/PAPER/FIG1/station_ChinaArray2.lst', source='CIEI', chans=['BHZ'], dnetcode='TA')
dset.plot_stations(geopolygons=basins)
