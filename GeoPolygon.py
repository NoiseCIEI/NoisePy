import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

class GeoPolygon(object):
    def __init__(self):
        self.lonArr=np.array([]);
        self.latArr=np.array([]);
        
    def PlotPolygon(self, inbasemap, alpha=0.7, hatch='x'):
        
        x, y = inbasemap(self.lonArr, self.latArr);
        basinArr=np.append(x,y);
        N=x.size;
        basinArr=basinArr.reshape((2,N));
        basinArr=basinArr.T;
        # poly=Polygon(basinArr, edgecolor= (0.5019607843137255, 0.5019607843137255, 0.5019607843137255),\
        #             alpha=alpha, closed=True, fill=False, hatch=hatch, lw=2);
        # poly=Polygon(basinArr, edgecolor= 'r',\
        #             alpha=None, closed=True, fill=False, hatch=hatch, lw=2)
        poly=Polygon(basinArr, edgecolor= 'k',\
                    alpha=alpha, closed=True, fill=False, hatch=hatch, lw=2)
        # poly=Polygon(basinArr, edgecolor='k',\
        #             alpha=None, closed=True, fill=False, hatch=hatch, lw=2);
        plt.gca().add_patch(poly)
        return

class GeoPolygonLst(object):
    """
    A object contains a list of GeoPolygon
    """
    def __init__(self,geopolygons=None):
        self.geopolygons=[]
        if isinstance(geopolygons, GeoPolygon):
            geopolygons = [geopolygons]
        if geopolygons:
            self.geopolygons.extend(geopolygons)

    def __add__(self, other):
        """
        Add two GeoPolygonLsts with self += other.
        """
        if isinstance(other, GeoPolygon):
            other = GeoPolygonLst([other])
        if not isinstance(other, GeoPolygonLst):
            raise TypeError
        geopolygons = self.geopolygons + other.geopolygons
        return self.__class__(geopolygons=geopolygons)

    def __len__(self):
        """
        Return the number of GeoPolygons in the GeoPolygonLst object.
        """
        return len(self.geopolygons)

    def __getitem__(self, index):
        """
        __getitem__ method of GeoPolygonLst objects.
        :return: GeoPolygon objects
        """
        if isinstance(index, slice):
            return self.__class__(geopolygons=self.geopolygons.__getitem__(index))
        else:
            return self.geopolygons.__getitem__(index)

    def append(self, geopolygon):
        """
        Append a single GeoPolygon object to the current GeoPolygonLst object.
        """
        if isinstance(geopolygon, GeoPolygon):
            self.geopolygons.append(geopolygon)
        else:
            msg = 'Append only supports a single GeoPolygon object as an argument.'
            raise TypeError(msg)
        return self
    
    
    def ReadGeoPolygonLst(self, polygonLst ):
        """
        Read GeoPolygon List from a txt file
        longitude latitude
        """
        f = open(polygonLst, 'r');
        NumbC=0;
        newpolygon=False;
        for lines in f.readlines():
            lines=lines.split();
            if newpolygon==True:
                lon=lines[0];
                if lon=='>':
                    newpolygon=False;
                    self.append(geopolygon)
                    continue;
                else:
                    lon=float(lines[0]);
                    lat=float(lines[1]);
                    geopolygon.lonArr=np.append(geopolygon.lonArr, lon);
                    geopolygon.latArr=np.append(geopolygon.latArr, lat);
            a=lines[0];
            b=lines[1];
            if a=='#' and b!='@P':
                continue;
            if b=='@P':
                NumbC=NumbC+1;
                newpolygon=True;
                geopolygon=GeoPolygon();
                continue;
            f.close()
        print 'End of reading', NumbC, 'geological polygons';
        return
    def PlotPolygon(self, inbasemap, alpha=0.3, hatch='x'):
        for geopolygon in  self.geopolygons:
            geopolygon.PlotPolygon(inbasemap, alpha=alpha, hatch=hatch)
        return
    
        
        
    # 
    # def Trim(self, maxlon=360, minlon=0, maxlat=90, minlat=-90):
    #     TrimedGeoPolygonLst=GeoPolygonLst()
    #     for geopolygon in self.geopolygons:
    #         if geopolygon.IsInRegion(maxlon=maxlon, minlon=minlon, maxlat=maxlat, minlat=minlat):
    #             TrimedGeoPolygonLst.append(geopolygon)
    #     return TrimedGeoPolygonLst
    # 
    # def GetINTGeoPolygonLst(self):
    #     NewGeoPolygonLst=GeoPolygonLst();
    #     for geopolygon in self.geopolygons:
    #         if abs(geopolygon.lon-int(geopolygon.lon)) > 0.1 or abs(geopolygon.lat-int(geopolygon.lat))>0.1:
    #             continue;
    #         geopolygon.SetName()
    #         NewGeoPolygonLst.append(geopolygon);
    #     # print 'End of Converting SES3D seismograms to SAC files !'
#     #     return NewGeoPolygonLst;     
# minlat=25.;
# maxlat=52.;
# minlon=90.;
# maxlon=143.;
# from mpl_toolkits.basemap import Basemap, shiftgrid, cm
# m1 = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution='l')
# m = Basemap(projection='ortho',lon_0=minlon,lat_0=minlat, resolution='i',\
#     llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/2.5, urcrnry=m1.urcrnry/2.5)
# # labels = [left,right,top,bottom]
# m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
# m.drawmeridians(np.arange(-170.0,170.0,10.0))
# m.drawcoastlines()
# m.drawcountries(color='b')
# mygeopolygons=GeoPolygonLst();
# mygeopolygons.ReadGeoPolygonLst('basin1');
# mygeopolygons.PlotPolygon(mybasemap=m);
# 
# plt.show()