// based on /home/weisen/PROGS_64/EIKONAL/SCRIPT/slow_maps_to_iso_map_ani_data_v4_ANT_265_weight_robust_cv_v1_2
// reduce the weighting level for ultra high values
#define MAIN
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include <unistd.h>
#include <stdlib.h>
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
        cout<<"warning cos(theta)>1 and correct to 1!!"<<temp<<endl;
        temp=1;
    }
    if(temp<-1)
    {
        cout<<"warning cos(theta)<-1 and correct to -1!!"<<temp<<endl;
        temp=-1;
    }
    theta=fabs(acos(temp));
    return theta*radius;
}

void Slowness2IsoAni(string stalst, double min, double max, int N_bin, string out_prefix, double dx, double x0, int npts_x, double  y0, int npts_y, string pflagin, double cridist )
{
    if(na!=13)
    {
        cout<<"usage:travel_time_to_velocity_map station_morgan.lst min max N_bin out_name dx x0 nx y0 ny pflag cridist"<<endl;
        return 0;+
    }
    FILE *ff,*fin,*fout,*file1,*file_iso,*file_ani;
    int i,j,k;
    char buff1[300],sta1[10],name_iso[100],name_ani[100],name_ani_n[100],pflag[5];
    double lat,lon,t_lat,t_lon,radius,pi,sta1_lon,sta1_lat;
    int t_i,t_j,nsta;
    int ii,jj,kk,kkk,min_n;
    double d_bin;
    sprintf(name_iso,"%s.iso",arg[5]);
    sprintf(name_ani,"%s.ani",arg[5]);
    sprintf(name_ani_n,"%s_ani_n",arg[5]);
    double hist[N_bin];
    double slow_sum1[N_bin];
    double slow_un[N_bin];
    d_bin=(max-min)/N_bin;
    radius=6371.1391285;
    pi=4.0*atan(1.0);
    double x1,y1,temp,lat_temp,temp2,trash1,trash2;
//    fprintf(stderr,"Memory check!!\n");

    double slow[npts_x][npts_y][2200];
    double azi[npts_x][npts_y][2200];
    double flag[npts_x][npts_y][2200];
    double weight[2200];
    double nw[2200];
    double cdist1 = 250.;  // minimum station distance
    int idw[2200];
    double weight_sum;
    int n[npts_x][npts_y];
    double trash;
    double slow_sum[npts_x][npts_y],slow_std[npts_x][npts_y];
    double cvlon,cvlat,tdist;

    FILE *fall,*fall1;
    fall = fopen("all.measurement.dat","w");
    fall1 = fopen("all.measurement.used.dat","w");
//    fprintf(stderr,"Memory enough!!\n");

    x1=x0+(npts_x-1)*dx;
    y1=y0+(npts_y-1)*dy;
    for(i=0; i<npts_x; i++)
    {
        for(j=0; j<npts_y; j++)
        {
            slow_sum[i][j]=0;
            n[i][j]=0;
        }
    }
    char event_name[300],tstr[300];

    file1=fopen(arg[1],"r");
    nsta=0;
    cout<<"now do "<<arg[1]<<endl;
    char *pch;
    char *pch1[20];
    for(;;)
    {
        if(fscanf(file1,"%s %lf %lf",&event_name,&cvlon,&cvlat)==EOF)
            break;
        nsta++;
        sprintf(buff1,"slow_azi_%s.%s.txt.HD.2.v2",event_name,pflag);
        if((fin=fopen(buff1,"r"))==NULL)
        {
            cout<<buff1<<" not exist!!"<<endl;
            continue;
        }
        cout<<buff1<<endl;
        for(;;)
        {
            if (fgets(tstr,300,fin) == NULL ) break;
            pch = strtok(tstr," \n");
            k = 0;
            while (pch!=NULL)
            {
                pch1[k] = pch;
                pch = strtok(NULL," \n");
                k = k + 1;
            }
            if (k == 5) continue; // Skip any bad point!!!
            lon = atof(pch1[0]);
            lat = atof(pch1[1]);
            temp = atof(pch1[2]); // slowness
            if (abs(temp)<0.01 ) continue;
            temp2 = atof(pch1[3]); // angle

            tdist = get_dist(cvlat,cvlon,lat,lon);
            if (tdist < cridist + 50.)
                continue;

            if (trash > cdist1 || temp2 > 900)
                continue;

            if(lon>x1+0.01||lon<x0-0.01|lat>y1+0.01||lat<y0-0.01) // Out of boundary
                continue;
            i=int((lon-x0)/dx+0.1);
            j=int((lat-y0)/dy+0.1);
            if(temp<0.6 &&temp>0.15) // different from the Travel Time to Slowness
            {
                fprintf(fall,"%g %g %g %g 1 %s\n", lon, lat,1./temp, temp2, event_name);
                slow[i][j][n[i][j]]=temp;
                azi[i][j][n[i][j]]=temp2;
                flag[i][j][n[i][j]]=0;
                n[i][j]++;
                //	     slow_sum[i][j]+=temp;
            }
        }
        cout<<buff1<<endl;
        fclose(fin);
    } // End of Reading all slowness maps and write to all.measurement.dat
    fclose(fall);
    cout<<"ok here"<<endl;
    fclose(file1);
    file_iso=fopen(name_iso,"w"); // isotropic map file
    nsta=50; // Why???
    double w2,tave,tstd;
    double temp_slow_sum;
    int temp_n;
    for(i=0; i<npts_x; i++)
    {
        for(j=0; j<npts_y; j++)
        {
            if(n[i][j]<0.3*nsta)
            {
                fprintf(file_iso,"%lf %lf 0 9999 %d\n",x0+i*dx,y0+j*dy,n[i][j]);
                continue;
            }
            w2=0;
            weight_sum=0;
            for(k=0; k<n[i][j]; k++)
            {
                weight[k]=0;
                nw[k] = 0;
                tave = 0.;
                for(kk=0; kk<n[i][j]; kk++)
                {
                    if(fabs(azi[i][j][kk]-azi[i][j][k])<20. || fabs(azi[i][j][kk]-azi[i][j][k])>(360-20.))
                        weight[k] = weight[k]++;
                    nw[k]++;
                    idw[k] = kk;
                    tave = tave + slow[i][j][k]; /// ???
                }
                weight[k]=1/weight[k];
                weight_sum+=weight[k];
            }
            /// reduce the largest weight[k] to some value.
            tave = weight_sum/n[i][j]; /// ???
            tstd = 0.;
            for(k=0; k<n[i][j]; k++)
            {
                tstd = tstd + (weight[k] - tave)*(weight[k] - tave);
            }
            tstd = sqrt(tstd/n[i][j]);
            weight_sum = 0.;
            for(k=0; k<n[i][j]; k++)
            {
                if (weight[k] > tave + 3*tstd) weight[k] = tave + 3*tstd;
                weight_sum+=weight[k];
            }
            ///
            for(k=0; k<n[i][j]; k++)
            {
                weight[k]=weight[k]/weight_sum;
                slow_sum[i][j]+=weight[k]*slow[i][j][k];
                w2+=weight[k]*weight[k];
            }
            temp_slow_sum=slow_sum[i][j];
            temp=0;
            for(k=0; k<n[i][j]; k++)
            {
                temp+=weight[k]*(slow[i][j][k]-slow_sum[i][j])*(slow[i][j][k]-slow_sum[i][j]);
            }
            slow_std[i][j]=sqrt(temp/(1-w2));
            w2=0;
            weight_sum=0;
            slow_sum[i][j]=0;
            temp_n=0;

            for(k=0; k<n[i][j]; k++)
            {
                if(fabs(slow[i][j][k]-temp_slow_sum)>2.0*slow_std[i][j])
                    continue;
                weight_sum+=weight[k];
                temp_n++;
            }
            for(k=0; k<n[i][j]; k++)
            {
                if(fabs(slow[i][j][k]-temp_slow_sum)>2.0*slow_std[i][j])
                    continue;
                weight[k]=weight[k]/weight_sum;
                slow_sum[i][j]+=weight[k]*slow[i][j][k];
                w2+=weight[k]*weight[k];
                flag[i][j][k]=1;
                fprintf(fall1,"%g %g %g %g %g\n",x0+i*dx,y0+j*dy,1./slow[i][j][k],azi[i][j][k],weight[k]);
            }
            temp=0;
            for(k=0; k<n[i][j]; k++)
            {
                if(fabs(slow[i][j][k]-temp_slow_sum)>2.0*slow_std[i][j])
                    continue;
                temp+=weight[k]*(slow[i][j][k]-slow_sum[i][j])*(slow[i][j][k]-slow_sum[i][j]);
            }
            slow_std[i][j]=sqrt(temp/(1-w2));
            temp=slow_std[i][j]*sqrt(w2)/slow_sum[i][j]/slow_sum[i][j];
            fprintf(file_iso,"%lf %lf %lf %lf %d\n",x0+i*dx,y0+j*dy,1/slow_sum[i][j],temp,temp_n);
        }
    }
    fclose(fall1);
    fclose(file_iso);

    file_ani=fopen(name_ani,"w");
    fout=fopen(name_ani_n,"w");

    double test_lon,test_lat;
    char t_name[300];
    double tslow;
    FILE *file_point;

    for(i=0; i<npts_x; i++)
    {
        for(j=0; j<npts_y; j++)
        {
            for(k=0; k<N_bin; k++)
            {
                hist[k]=0;
                slow_sum1[k]=0;
                slow_un[k]=0;
            }
            if(i-4<0 || i+3>=npts_x || j-4<0 || j+3>=npts_y)
                continue;
            kkk=0;
            for(ii=i-4; ii<=i+4; ii+=4)
            {
                for(jj=j-4; jj<=j+4; jj+=4)
                {
                    if(n[ii][jj]<nsta*0.3)
                        continue;
                    kkk+=n[ii][jj];
                }
            }
            fprintf(fout,"%lf %lf %d\n",x0+i*dx,y0+j*dy,kkk);
            if(kkk<9*nsta*0.3||n[i][j]<nsta*0.3)
                continue;
            sprintf(t_name,"%g_%g.raw\0",x0+i*dx,y0+j*dy);
            file_point = fopen(t_name,"w");

            for(ii=i-3; ii<=i+3; ii+=3)
            {
                for(jj=j-3; jj<=j+3; jj+=3)
                {
                    if(n[ii][jj]<nsta*0.3)
                        continue;
                    for(k=0; k<n[ii][jj]; k++)
                    {
                        if (flag[ii][jj][k]<=0)
                        {
                            continue;
                        }
                        if(azi[ii][jj][k]>max||azi[ii][jj][k]<min)
                        {
                            fprintf(stderr,"out of range!!");
                            return 1;
                        }
                        hist[int((azi[ii][jj][k]-min)/d_bin)]++;
                        slow_sum1[int((azi[ii][jj][k]-min)/d_bin)]+=slow[ii][jj][k]-slow_sum[ii][jj];
                        tslow=slow_sum[i][j]+(slow[ii][jj][k]-slow_sum[ii][jj]);
                        fprintf(file_point,"%g %g %g %g %g\n",tslow,1./tslow,azi[ii][jj][k],slow[ii][jj][k],1./slow[ii][jj][k]);
                    }
                }
            }
            fclose(file_point);
            kk=0;

            for(k=0; k<N_bin; k++)
            {
                if(hist[k]>=10)
                {
                    kk++;
                }
            }
            fprintf(file_ani,"%lf %lf %d\n",x0+i*dx,y0+j*dy,kk);
            for(k=0; k<N_bin; k++)
            {
                if(hist[k]>=10)
                {
                    slow_sum1[k]=slow_sum1[k]/hist[k];
                    slow_un[k]=slow_std[i][j]/sqrt(double(hist[k]));
                    slow_un[k]=slow_un[k]/(slow_sum[i][j]+slow_sum1[k])/(slow_sum[i][j]+slow_sum1[k]);//uncertainty of vel not slow
                    fprintf(file_ani,"%lf %lf %lf\n",min+(0.5+k)*d_bin,1/(slow_sum[i][j]+slow_sum1[k]),slow_un[k]);
                }
            }
        }
    }
    fclose(file_ani);
    fclose(fout);
    return 0;
}
