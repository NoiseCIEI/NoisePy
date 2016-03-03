// use HD and HD_0.8

// espetially for US
// change the marker_nn to 4
// change the distance cri to 150.
// use dx-dy from precomputed values;

#define MAIN
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include </projects/life9360/EIKONAL_SCRIPT/dx.h>
#include </projects/life9360/EIKONAL_SCRIPT/dy.h>

using namespace std;

double get_dist(double lat1,double lon1,double lat2,double lon2)
{
    double theta,pi,temp;
    double radius=6371;
    pi=4.0*atan(1.0);
    lat1=atan(0.993277*tan(lat1/180*pi))*180/pi;
    lat2=atan(0.993277*tan(lat2/180*pi))*180/pi;
    temp=sin((90-lat1)/180*pi)*cos(lon1/180*pi)*sin((90-lat2)/180*pi)*cos(lon2/180*pi)+sin((90-lat1)/180*pi)*sin(lon1/180*pi)*sin((90-lat2)/180*pi)*sin(lon2/180*pi)+cos((90-lat1)/180*pi)*cos((90-lat2)/180*pi);
    if(temp>1)
    {
        temp=1;
    }
    if(temp<-1)
    {
        temp=-1;
    }
    theta=fabs(acos(temp));
    return theta*radius;
}
// End of get_dist
int get_dist2(double lat1, double lon1, double lat2, double lon2, double *dist, double *azi, double *bazi)
{
    double pi;
    pi = 4.0*atan(1.0);
    double cva = 6378.137;
    double cvb = 6356.7523142;
    double f = 1/298.257223563;
    double L = 0.00;
    double jcvA, jcvB;
    L = lon1-lon2;
    double U1 = 0;
    U1 = atan((1-f)*tan(lat1/180*pi));
    double U2 = 0;
    U2 = atan((1-f)*tan(lat2/180*pi));
    double cv,cv1,cv2,cv3,cv4,cv5,cvC,numda1;
    L = L*pi/180;
    double numda = L;
    numda1 = numda;
    do
    {
        numda = numda1;
        cv1 =  sqrt( (cos(U2)*sin(numda))*(cos(U2)*sin(numda))+ (cos(U1)*sin(U2)-sin(U1)*cos(U2)*cos(numda))*(cos(U1)*sin(U2)-sin(U1)*cos(U2)*cos(numda)) ); // cv1 sin(quan)
        cv2 = sin(U1)*sin(U2)+ cos(U1)*cos(U2)*cos(numda);
        cv = atan2(cv1,cv2);
        cv3 = cos(U1)*cos(U2)*sin(numda)/sin(cv);
        cv4 = 1 - cv3*cv3;
        if (cv4 == 0)
            cv4 = 0.0000000001;
        cv5 = cos(cv) - 2*sin(U1)*sin(U2)/cv4;
        cvC = f/16*cv4*(4 + f*(4 - 3*cv4));
        numda1 = L + (1-cvC)*f*cv3*(cv + cvC*cv1*(cv5 + cvC*cv2*(-1 +2*cv5*cv5)));
    }
    while (fabs(numda - numda1) > 0.0000000001);
    double mius, cvA, cvB, deltacv,s;
    mius = cv4*(cva*cva - cvb*cvb)/(cvb*cvb);
    cvA = 1+mius/16384*(4096 + mius*(-768 + mius*(320 - 175*mius)));
    cvB = mius/1024*(256+ mius*(-128 + mius*(74 - 47*mius)));
    deltacv = cvB*cv1*(cv5 +cvB/4*(cv2*(-1 + 2*cv5*cv5)-cvB/6*cv5*(-3+4*cv1*cv1)*(-3+4*cv5*cv5) ));
    s = cvb * cvA *(cv - deltacv);
    jcvA = atan2( (cos(U2)*sin(numda1)),(cos(U1)*sin(U2)-sin(U1)*cos(U2)*cos(numda1)))*180/pi;
    jcvB = atan2( (cos(U1)*sin(numda1)),(-sin(U1)*cos(U2)+sin(U2)*cos(U1)*cos(numda1)))*180/pi;
    if (jcvB>180) jcvB = jcvB-180;
    else jcvB = 180 - jcvB;
    if (jcvA>180) jcvA = jcvA-180;
    else if (jcvA <0 )
        jcvA = -jcvA;
    else jcvA = 360 - jcvA;
    *dist = s;
    *azi = jcvA;
    *bazi = jcvB;
    return 1;
}
// End of get_dist2
void TravelTime2Slowness(string stalst, double period, string out_prefix, double dxdy, double x0, int npts_x, double  y0, int npts_y, string pflag, double cridist)
{
    if(na<10)
    {
        cout<<"usage:travel_time_to_velocity_map [station_morgan.lst] [period] [dx/dy] [x0] [xn] [y0] [yn] [pflag:1-phase/!1-group] [cdist] [optional: sta]"<<endl;
        return 0;
    }
    double ag1,ag2,diffa,diffb;
    FILE *ff,*fin,*fin2,*fin3,*fout,*file1;
    int i,j,cc,tflag;
    char buff1[300],sta1[10],pflag[5],cvstr[300];
    double lat,lon,lat2,lon2,t_lat,t_lon,radius,pi,sta1_lon,sta1_lat;
    double lonin[40000],latin[40000],ttin[40000];
    int nn,cvi,cvii;
    int t_i,t_j,xn,yn;
    int marker_nn,marker_EN[2][2],marker_E,marker_N;
    double dist;
    double plat,plon,tdist,cdist;
    double cdist1;
    double az,baz;
    double dst;
    double xi,xj,xk;
    double yi,yj,yk;
    double loni,lati;
    double ang_n,slow_n;
    cdist = period * 4. * 3. + 50.;
    cdist1 = cridist; // per*12

    if (atoi(arg[8]) == 1)
    {
        sprintf(pflag,"phase.c");
    }
    else
    {
        sprintf(pflag,"group.c");
    }

    radius=6371.1391285;
    pi=4.0*atan(1.0);
    double dx,dy,x1,y1,temp,temp1,temp2,lat_temp;
    npts_x=xn;
    npts_y=yn;
    fprintf(stderr,"Memory check!!\n");
    double tr_t[npts_x][npts_y];
    double dx_km[npts_y],dy_km[npts_y];
    double mdist1[npts_x][npts_y];
    double mdist2;
    int cvn_lat;

    int reason_n[npts_x][npts_y];
    fprintf(stderr,"Memory enough!!\n");

    dx=dxdy;//degree
    dy=dxdy;//degree
    x1=x0+(npts_x-1)*dx;
    y1=y0+(npts_y-1)*dy;
    for(j=1; j<npts_y-1; j++)
    {
        lat_temp=y0+j*dy;
        cvn_lat = int(lat_temp/0.2+0.1);
        if (cvn_lat>449) cvn_lat = 449;
        dx_km[j]=dx_km1[cvn_lat]*dx/0.2;
        dy_km[j]=dy_km1[cvn_lat]*dy/0.2;
    }
    file1=fopen(arg[1],"r");
    clock_t t,t1;
    t = clock();
    double trash0,trash1,trash2,trash3;
    long int trash4;
    char trash5[6];
    for(;;)
    {
        if(fscanf(file1,"%s %lf %lf",&sta1,&sta1_lon,&sta1_lat)==EOF)
            break;
        if (na == 11)
        {
            if (strcmp (sta1, arg[10]) != 0) continue;
        }
        if (sta1_lon < 0)
        {
            sta1_lon = sta1_lon + 360.;
        }
        sprintf(buff1,"travel_time_%s.%s.txt_v1.HD",sta1,pflag);
        if((fin=fopen(buff1,"r"))==NULL)
        {
            cout<<buff1<<" not exist!!"<<endl;
            continue;
        }
        sprintf(buff1,"travel_time_%s.%s.txt_v1.HD_0.2",sta1,pflag);
        if((fin2=fopen(buff1,"r"))==NULL)
        {
            cout<<buff1<<" not exist!!"<<endl;
            continue;
        }
        sprintf(buff1,"slow_azi_%s.%s.txt.HD.2.v2",sta1,pflag);
        fout=fopen(buff1,"w");
        for(i=0; i<npts_x; i++)
        {
            for(j=0; j<npts_y; j++)
            {
                tr_t[i][j]=0;
                reason_n[i][j] = 0;
            }
        }
        sprintf(buff1,"travel_time_%s.%s.txt_v1",sta1,pflag);
        if((fin3=fopen(buff1,"r"))==NULL)
        {
            cout<<buff1<<" not exist!!"<<endl;
            return 1;
        }
        fclose(fin3);
        nn = 0;
        fin3=fopen(buff1,"r");
        for(cvi=0; cvi<2000; cvi++)
        {
            lonin[cvi]=0.;
            latin[cvi]=0.;
            ttin[cvi]=-1.;
        }
        cvi = 0;
        for (;;)
        {
            if (fgets(cvstr,100,fin3) == NULL) // travel_time_%s.%s.txt_v1
            {
                break;
            }
            sscanf(cvstr,"%lf %lf %lf %lf %s %ld",&(trash0),&(trash1),&trash2, &trash3, &(trash5[0]),&trash4);
            if (trash4>0) // flag
            {
                lonin[cvi] = trash0;
                latin[cvi] = trash1;
                ttin[cvi] = trash2;
                cvi = cvi + 1;
            }
            else
            {
                continue;
            }
            //if (fabs(lon2-lon) > cdist1/110. || fabs(lat2-lat) > cdist1/110. ) continue;
            //dist=get_dist(lat,lon,lat2,lon2);
            //if (mdist1[i][j] > dist) {mdist1[i][j] = dist;  }
        }
//	for (i=0; i<=cvi; i++)
//		cout<<lonin[i]<<" "<<latin[i]<<endl;
        nn = cvi;
        cout<<nn<<" "<<buff1<<endl;
        fclose(fin3);
        cout<<"read in file ok! "<<nn<<endl;
        cout <<"now read "<<buff1<<endl;

        t1 = clock();
        fprintf(stderr,"now time is : %g\n",(float)(t1-t)/CLOCKS_PER_SEC);

        for(;;)
        {
            if(fscanf(fin,"%lf %lf %lf",&lon,&lat,&temp)==EOF) break; // "travel_time_%s.%s.txt_v1.HD"
            if(fscanf(fin2,"%lf %lf %lf",&lon2,&lat2,&temp2)==EOF) break; // "travel_time_%s.%s.txt_v1.HD_0.2"
            if(lon!=lon2||lat!=lat2)
            {
                fprintf(stderr,"HD and HD_0.2 files not compatiable!!\n");
                return 0;
            }
            if(lon>x1+0.01||lon<x0-0.01|lat>y1+0.01||lat<y0-0.01) // NOTE!
            {
                continue;
            }
            i=int((lon-x0)/dx+0.1);
            j=int((lat-y0)/dy+0.1);
            if(temp<temp2-2||temp>temp2+2)   // 2 period criterior
            {
                tr_t[i][j]=0;
                reason_n[i][j] = 1;
		    //cout<<lon<<" "<<lat<<endl;
                continue;
            }
            mdist1[i][j] = 1000.;
            marker_nn = 4;
            marker_EN[0][0]=0;
            marker_EN[0][1]=0;
            marker_EN[1][0]=0;
            marker_EN[1][1]=0;

            tflag = 0;
            for(cvi=0; cvi<nn; cvi++)  // looking for stations close to the point. nn effective number of points for snrflag>0
            {
                lon2 = lonin[cvi];
                lat2 = latin[cvi];
                if (fabs(lon2-lon) > cdist1/110. || fabs(lat2-lat) > cdist1/110. ) continue;  // too far, continue
                if (lon2 < 0) lon2 = lon2+360.;

                if(lon2-lon<0)
                    marker_E=0;
                else
                    marker_E=1;

                if(lat2-lat<0)
                    marker_N=0;
                else
                    marker_N=1;

                dist=get_dist(lat,lon,lat2,lon2);
                if(marker_EN[marker_E][marker_N]!=0)
                    continue;
                if( dist< cdist1 && dist >= 1)
                {
                    marker_nn--;
                    if(marker_nn==0)
                    {
                        tflag = 1; // more than 4 stations close to the point
                        break;
                    }
                    marker_EN[marker_E][marker_N]++;
                }
            }
            if (tflag < 1) // not enough stations close to the point
            {
                temp = 0.;
                temp2 = 0.;
                reason_n[i][j] = 2;
            }
            tr_t[i][j]=temp;
        }
        fclose(fin);
        fclose(fin2);

        cvi = 0;
        for(i=1; i<npts_x-1; i++) for(j=1; j<npts_y-1; j++)
            {
                if (tr_t[i][j]>0) cvi++;
            }
        cout<<cvi<<endl; // cvi: number of effective points

        t1 = clock();
        fprintf(stderr,"now time is : %g\n",(float)(t1-t)/CLOCKS_PER_SEC);

        cout<<"now  wright!"<<endl;
        for(i=1; i<npts_x-1; i++)
        {
            for(j=1; j<npts_y-1; j++)
            {
//cout<<x0+i*dx<<" "<<y0+j*dy<<" "<<tr_t[i][j]<<endl;
                if (reason_n[i][j] == 2 || reason_n[i][j] == 1) // if not enough stations close to the point OR T0.0 map not compatible with T0.2 map
                {
                    fprintf(fout,"%lf %lf 0 999 %d\n",x0+i*dx,y0+j*dy,reason_n[i][j]);
		    //cout<<x0+i*dx<<" "<<y0+j*dy<<endl;
                    continue;
                }

                temp1=(tr_t[i+1][j]-tr_t[i-1][j])/2.0/dx_km[j];
                temp2=(tr_t[i][j+1]-tr_t[i][j-1])/2.0/dy_km[j];
                if(temp2==0)
                {
                    temp2=0.00001;
                }
                temp=sqrt(temp1*temp1+temp2*temp2);
		//cout<<x0+i*dx<<" "<<y0+j*dy<<" "<<tr_t[i][j]<<" "<<tr_t[i+1][j]<<" "<<tr_t[i-1][j]<<" "<<tr_t[i][j+1]<<" "<<tr_t[i][j-1]<<" "<<" "<<dx_km[j]<<" "<<dy_km[j]<<" "<<temp<<endl;
                if(temp>0.6||temp<0.2)  // if the gradient is too large or too small( 1.67 km/s or 5km/s  )
                {
                    reason_n[i][j] = 3;
//cout<<x0+i*dx<<" "<<y0+j*dy<<endl;
                    fprintf(fout,"%lf %lf 0 999 %d\n",x0+i*dx,y0+j*dy,reason_n[i][j]);
                }
                else if ( tr_t[i+1][j]==0||tr_t[i-1][j]==0||tr_t[i][j+1]==0||tr_t[i][j-1]==0 ) // if 0 travel time
                {
                    reason_n[i][j] = 4;
                    fprintf(fout,"%lf %lf 0 999 %d\n",x0+i*dx,y0+j*dy,reason_n[i][j]);
                }
                else
                {
                    // get mdist2
                    mdist2 = 999.;
                    lon = x0+i*dx;
                    lat = y0+j*dy;
                    for (cvi=0; cvi<nn; cvi++)
                    {
                        lon2 = lonin[cvi];
                        lat2 = latin[cvi];
                        if (fabs(lon2-lon) > cdist1/111. || fabs(lat2-lat) > cdist1/111. ) continue;
                        dist = 112.*pow((lon2-lon)*(lon2-lon)*cos(lat*pi/180.)*cos(lat*pi/180.) + (lat2-lat)*(lat2-lat),0.5);
                        if (mdist2 > dist)
                        {
                            mdist2 = dist;
                            cvii = cvi;
                        }
                    }
                    mdist2 = get_dist(lat,lon,latin[cvii],lonin[cvii]); // distance from grid point to sta point
                    //cout<<mdist2<<endl;
                    ////////////////////////////////////////////////////////////////////////////////////////////////

                    plat = y0+j*dy; // Why not lon, lat ???
                    plon = x0+i*dx;

                    /////////////////////////// get azimuth ///////////////////////////////////////////
                    baz = 0.;
                    az = 0.;
                    get_dist2(plat,plon,sta1_lat,sta1_lon,&dst,&az,&baz);

                    ///////////////////////////////////////////////////////////////////////////////////
                    az = az + 180.;
                    az = 90.-az;
                    baz = 90.-baz;
                    if (az > 180.) az = az - 360.;
                    if (az < -180.) az = az + 360.;
                    if (baz > 180.) baz = baz - 360.;
                    if (baz < -180.) baz = baz + 360.;
                    ag1 = az;
                    ag2 = atan2(temp2,temp1)/pi*180.;
                    diffa = ag2 - ag1;

                    if (diffa < -180.) diffa = diffa + 360.;
                    if (diffa > 180.) diffa = diffa - 360.;

                    /*
                    diffb = ag2 - ang_n;
                    if (diffb < -180.) diffb - diffb + 360.;
                    if (diffb > 180.) diffb = diffb - 360.;*/

                    // get another direction from temp1, temp2;
                    /*
                    lati = (90-plat)*pi/180.; loni = plon*pi/180.;

                    xi = -cos(lati)*cos(loni)*temp2 - sin(loni)*temp1;  // x component of addision
                    xj = -cos(lati)*sin(loni)*temp2 + cos(loni)*temp1;  // y component of addision
                    xk = sin(lati)*temp2;  // z component of addsion

                    yi = -cos(lati)*cos(loni);
                    yj = -cos(lati)*sin(loni);
                    yk = sin(lati); */
                    /*yi = -sin(loni);
                    yj = -cos(loni);
                    yk = 0.;*/

                    /*
                    // now get direction between 2 vectors;
                    slow_n = sqrt(xi*xi + xj*xj + xk*xk);
                    // angle between vector (xi,xj,xk) (slowness vecotr) and vector (yi,yj,yk) (North) //
                    ang_n = acos((xi*yi+xj*yj+xk*yk)/(sqrt(xi*xi+xj*xj+xk*xk) * sqrt(yi*yi+yj*yj+yk*yk)))*180./pi;

                    //fprintf(stderr,"ijks: %g %g %g north: %g %g %g latiloni %g %g %g %g\n",xi,xj,xk,yi,yj,yk,lati,loni,plat,plon);
                    //cout<<ang_n<<endl;
                    if (plon<sta1_lon) ang_n = -ang_n + 360.;
                    ang_n = 90. - ang_n;
                    if (ang_n>180.) ang_n = ang_n - 360.;
                    if (ang_n<-180.) ang_n = ang_n + 360.;

                    diffb = ang_n - ag1;
                    if (diffb < -180.) diffb = diffb + 360.;
                    if (diffb > 180.) diffb = diffb - 360.; */
                    ////////////////////////////////////////////////////////////
                    //slow_n = temp;
                    //ang_n = atan2(temp2,temp1*(cos(0.032 + 0.0016*(plat-20.))))*180./pi;
                    /*
                    ang_n = 360. - az;
                    if (ang_n > 180.) ang_n = ang_n - 360.;
                    if (ang_n < -180.) ang_n = ang_n + 360.;
                    diffb = ang_n - ag1;
                    if (diffb < -180.) diffb = diffb + 360.;
                    if (diffb > 180.) diffb = diffb - 360.;

                    if (j == 51 && i == 101) {
                    //fprintf(stderr,"ijks: %g %g %g north: %g %g %g latiloni %g %g %g %g\n",xi,xj,xk,yi,yj,yk,lati,loni,plat,plon);
                    cout<<ag1<<" "<<ag2<<" "<<ang_n<<endl;
                    cout<<plon<<" "<<plat<<endl;
                    cout<<temp<<" "<<slow_n<<endl;
                    abort();}*/

                    // grid lon, grid lat, gradient mag, gradient angle, distance to closest station point, distance to event , az, baz, off great circle diff
                    if (fabs(plat - sta1_lat)>5. || fabs(plon - sta1_lon)>5.)
                    {
		//cout<<x0+i*dx<<" "<<y0+j*dy<<" "<<temp<<" "<<atan2(temp2,temp1)/pi*180<<" "<<mdist2<<" "<<dst<<" "<< az<<" "<< baz<<" "<< diffa<<endl;
                        fprintf(fout,"%lf %lf %lf %lf %g %g %g %g %g\n",x0+i*dx, y0+j*dy, temp, atan2(temp2,temp1)/pi*180, mdist2, dst, az, baz, diffa );
                        continue;
                    }
                    tdist = get_dist(plat,plon,sta1_lat,sta1_lon); // why compute the dist again???
		    //cout<<"D:"<<tdist-dst<<endl;
                    if (tdist < cdist)  // if grid point too close to event point
                    {
		//cout<<x0+i*dx<<" "<<y0+j*dy<<" "<<tdist<<" "<<cdist<<endl;
                        fprintf(fout,"%lf %lf 0 999 5\n",plon,plat); // too close to the central station
                    }
                    else
                    {
		//cout<<x0+i*dx<<" "<<y0+j*dy<<" "<<temp<<" "<<atan2(temp2,temp1)/pi*180<<" "<<mdist2<<" "<<dst<<" "<< az<<" "<< baz<<" "<< diffa<<endl;
                        fprintf(fout,"%lf %lf %lf %lf %g %g %g %g %g\n",x0+i*dx,y0+j*dy,temp,atan2(temp2,temp1)/pi*180,mdist2,dst,az,baz,diffa);
                    }
                }
            }
        }
        fclose(fout);
        t1 = clock();
        fprintf(stderr,"now time is : %g\n",(float)(t1-t)/CLOCKS_PER_SEC);
    }
    fclose(file1);
    return 1;
}
