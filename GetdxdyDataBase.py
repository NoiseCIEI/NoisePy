#!/usr/bin/env python
import obspy.core.util.geodetics as obsGeo
import numpy as np
dx=0.2
latLst=-90.+np.arange(180./dx+1)*0.2
midlon=0.
dx_km=np.array([])
dy_km=np.array([])
for lat in latLst:
    if lat == -90:
        continue
    dist, az, baz=obsGeo.gps2DistAzimuth(lat, midlon, lat-0.2, midlon )
    dy_km=np.append(dy_km, dist/1000.)
    dist, az, baz=obsGeo.gps2DistAzimuth(lat, midlon, lat, midlon+0.2 )
    dx_km=np.append(dx_km, dist/1000.)

dist, az, baz=obsGeo.gps2DistAzimuth(-90., midlon, -90., midlon+0.2 )
dx_km=np.append(dist/1000., dx_km)
dx_km=np.append(dx_km, latLst);
dx_km=dx_km.reshape(2,len(latLst))
dx_km=dx_km.T
np.savetxt('dx_km.txt', dx_km, fmt='%g')
dy_km=np.append(dy_km[0], dy_km)
dy_km=np.append(dy_km, latLst);
dy_km=dy_km.reshape(2,len(latLst))
dy_km=dy_km.T
np.savetxt('dy_km.txt', dy_km, fmt='%g')

# midlon=31.223
# dx_km1=np.array([])
# dy_km1=np.array([])
# for lat in latLst:
#     if lat == -90:
#         continue
#     dist, az, baz=obsGeo.gps2DistAzimuth(lat-0.2, midlon, lat, midlon )
#     dy_km1=np.append(dy_km1, dist/1000.)
#     dist, az, baz=obsGeo.gps2DistAzimuth(lat, midlon+0.2, lat, midlon )
#     dx_km1=np.append(dx_km1, dist/1000.)


