import ASDFDBase
import numpy as np
import GeoPolygon
basins=GeoPolygon.GeoPolygonLst()
basins.ReadGeoPolygonLst('basin1')
dset=ASDFDBase.quakeASDF('../EA_quake.h5')
# dset.get_events(startdate='2008-1-01', enddate='2011-12-31', Mmin=4.0, Mmax=5.5, minlatitude=25, maxlatitude=35, minlongitude=100, maxlongitude=110,\
#                 magnitudetype='mw', maxdepth=10.)
# dset.get_events(startdate='2008-1-01', enddate='2011-12-31', Mmin=4.0, Mmax=5.5, minlatitude=15, maxlatitude=30, minlongitude=120, maxlongitude=125,\
#                 magnitudetype='mw', maxdepth=10.)
dset.plot_events(geopolygons=basins)