
import numpy as np
import h5py
import os
from subprocess import call

class RayTomoDataSet(h5py.File):
    
    def set_input_parameters(self, minlon, maxlon, minlat, maxlat, pers=np.array([]), data_pfx='raytomo_in_', smoothpfx='N_INIT_', qcpfx='QC_'):
        if pers.size==0:
            pers=np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
        self.attrs.create(name = 'period_array', data=pers, dtype='f')
        self.attrs.create(name = 'minlon', data=minlon, dtype='f')
        self.attrs.create(name = 'maxlon', data=maxlon, dtype='f')
        self.attrs.create(name = 'minlat', data=minlat, dtype='f')
        self.attrs.create(name = 'maxlat', data=maxlat, dtype='f')
        self.attrs.create(name = 'data_pfx', data=data_pfx)
        self.attrs.create(name = 'smoothpfx', data=smoothpfx)
        self.attrs.create(name = 'qcpfx', data=qcpfx)
        return
        
    def run_smooth(self, datadir, outdir, dlon=0.5, dlat=0.5, stepInte=0.2, lengthcell=1.0, datatype='ph', channel='ZZ',
            alpha=3000, sigma=500, beta=100, runid=0):
        """
        Run Misha's Tomography Code with large regularization parameters.
        This function is designed to do an inital test run, the output can be used to discard outliers in aftan results.
        
        IsoMishaexe - Path to Misha's Tomography code executable (isotropic version)
        contourfname - Path to contour file (see the manual for detailed description)
        ----------------------------------------------------------------------------
        Input format:
        datadir/data_pre+str(per)+'_'+chpair[0]+'_'+chpair[1]+'_'+datatype+'.lst'
        e.g. datadir/MISHA_in_20.0_BHZ_BHZ_ph.lst
        
        Output format:
        e.g. 
        Prefix: outdir/10.0_ph/N_INIT_3000_500_100
        output file: outdir/10.0_ph/N_INIT_3000_500_100_10.0.1 etc. (see the manual for detailed description of output suffix)
        ----------------------------------------------------------------------------
        References:
        Barmin, M. P., M. H. Ritzwoller, and A. L. Levshin. "A fast and reliable method for surface wave tomography."
            Monitoring the Comprehensive Nuclear-Test-Ban Treaty: Surface Waves. Birkh?user Basel, 2001. 1351-1375.
        """
        IsoMishaexe='./TOMO_MISHA/itomo_sp_cu_shn'
        contourfname='./contour.ctr'
        if not os.path.isfile(IsoMishaexe): raise AttributeError('IsoMishaexe does not exist!')
        if not os.path.isfile(contourfname): raise AttributeError('Contour file does not exist!')
        pers = self.attrs['period_array']
        minlon=self.attrs['minlon']
        maxlon=self.attrs['maxlon']
        minlat=self.attrs['minlat']
        maxlat=self.attrs['maxlat']
        data_pfx=self.attrs['data_pfx']
        smoothpfx=self.attrs['smoothpfx']
        for per in pers[:1]:
            infname=datadir+'/'+data_pfx+'%g'%( per ) +'_'+channel+'_'+datatype+'.lst'
            print infname
            outper=outdir+'/'+'%g'%( per ) +'_'+datatype
            if not os.path.isdir(outper):
                os.makedirs(outper)
            outpfx=outper+'/'+smoothpfx+str(alpha)+'_'+str(sigma)+'_'+str(beta)
            temprunsh='temp_'+'%g_Smooth.sh' %(per)
            with open(temprunsh,'wb') as f:
                f.writelines('%s %s %s %g << EOF \n' %(IsoMishaexe, infname, outpfx, per ))
                # if paraFlag==False:
                #     f.writelines('me \n' );
                f.writelines('me \n4 \n5 \n%g \n6 \n%g \n%g \n%g \n' %( beta, alpha, sigma, sigma) )
                f.writelines('7 \n%g %g %g \n8 \n%g %g %g \n12 \n%g \n%g \n16 \n' %(minlat, maxlat, dlat, minlon, maxlon, dlon, stepInte, lengthcell) )
                # if paraFlag==False:
                #     f.writelines('v \n' );
                f.writelines('v \nq \ngo \nEOF \n' )
            call(['bash', temprunsh])
            os.remove(temprunsh)
            # save to hdf5 dataset
            create_group=False
            while (not create_group):
                try:
                    group=self.create_group( name = 'smooth_run_'+str(runid) )
                    create_group=True
                except:
                    runid+=1
                    continue
            
            
        return



# 
# 
# 
# ### Functions used as glue to run Misha's Tomography code
# def RunMishaSmooth(per, datadir, outdir, minlon, maxlon, minlat, maxlat, \
#         dlon=0.5, dlat=0.5, stepInte=0.2, lengthcell=1.0, datatype='ph', chpair=['BHZ', 'BHZ'], alpha=3000, sigma=500, beta=100, data_pre='MISHA_in_', outpre='N_INIT_'):
#     """
#     Run Misha's Tomography Code with large regularization parameters.
#     This function is designed to do an inital test run, the output can be used to discard outliers in aftan results.
#     
#     IsoMishaexe - Path to Misha's Tomography code executable (isotropic version)
#     contourfname - Path to contour file (see the manual for detailed description)
#     ----------------------------------------------------------------------------
#     Input format:
#     datadir/data_pre+str(per)+'_'+chpair[0]+'_'+chpair[1]+'_'+datatype+'.lst'
#     e.g. datadir/MISHA_in_20.0_BHZ_BHZ_ph.lst
#     
#     Output format:
#     e.g. 
#     Prefix: outdir/10.0_ph/N_INIT_3000_500_100
#     output file: outdir/10.0_ph/N_INIT_3000_500_100_10.0.1 etc. (see the manual for detailed description of output suffix)
#     ----------------------------------------------------------------------------
#     References:
#     Barmin, M. P., M. H. Ritzwoller, and A. L. Levshin. "A fast and reliable method for surface wave tomography."
#         Monitoring the Comprehensive Nuclear-Test-Ban Treaty: Surface Waves. Birkh?user Basel, 2001. 1351-1375.
#     """
#     IsoMishaexe='./TOMO_MISHA/itomo_sp_cu_shn';
#     contourfname='./contour.ctr';
#     if not os.path.isfile(IsoMishaexe):
#         print 'IsoMishaexe does not exist!';
#         return
#     if not os.path.isfile(contourfname):
#         print 'Contour file does not exist!';
#         return;
#     infname=datadir+'/'+data_pre+'%g'%( per ) +'_'+chpair[0]+'_'+chpair[1]+'_'+datatype+'.lst';
#     outper=outdir+'/'+'%g'%( per ) +'_'+datatype;
#     if not os.path.isdir(outper):
#         os.makedirs(outper);
#     outpre=outper+'/'+outpre+str(alpha)+'_'+str(sigma)+'_'+str(beta);
#     temprunsh='temp_'+'%g_Smooth.sh' %(per);
#     f=open(temprunsh,'wb')
#     f.writelines('%s %s %s %g << EOF \n' %(IsoMishaexe, infname, outpre, per ));
#     # if paraFlag==False:
#     #     f.writelines('me \n' );
#     f.writelines('me \n4 \n5 \n%g \n6 \n%g \n%g \n%g \n' %( beta, alpha, sigma, sigma) );
#     f.writelines('7 \n%g %g %g \n8 \n%g %g %g \n12 \n%g \n%g \n16 \n' %(minlat, maxlat, dlat, minlon, maxlon, dlon, stepInte, lengthcell) );
#     # if paraFlag==False:
#     #     f.writelines('v \n' );
#     f.writelines('v \nq \ngo \nEOF \n' );
#     f.close();
#     call(['bash', temprunsh]);
#     # os.remove(temprunsh);
#     return;
# 
# def RunMishaSmoothParallel(per_array, datadir, outdir, minlon, maxlon, minlat, maxlat, \
#         dlon=0.5, dlat=0.5, stepInte=0.2, lengthcell=1.0, datatype='ph', chpair=['BHZ', 'BHZ'], alpha=3000, sigma=500, beta=100, data_pre='MISHA_in_', outpre='N_INIT_'):
#     """
#     Parallelly run Misha's Tomography Code with large regularization parameters for a period array.
#     This function is designed to do an inital test run, the output can be used to discard outliers in aftan results.
#     
#     IsoMishaexe - Path to Misha's Tomography code executable (isotropic version)
#     contourfname - Path to contour file (see the manual for detailed description)
#     ----------------------------------------------------------------------------
#     Input format:
#     datadir/data_pre+per+'_'+chpair[0]+'_'+chpair[1]+'_'+datatype+'.lst'
#     e.g. datadir/MISHA_in_20_BHZ_BHZ_ph.lst
#     
#     Output format:
#     e.g. 
#     Prefix: outdir/10_ph/N_INIT_3000_500_100
#     output file: outdir/10_ph/N_INIT_3000_500_100_10.1 etc. (see the manual for detailed description of output suffix)
#     ----------------------------------------------------------------------------
#     References:
#     Barmin, M. P., M. H. Ritzwoller, and A. L. Levshin. "A fast and reliable method for surface wave tomography."
#         Monitoring the Comprehensive Nuclear-Test-Ban Treaty: Surface Waves. Birkh?user Basel, 2001. 1351-1375.
#     """
#     IsoMishaexe='./TOMO_MISHA/itomo_sp_cu_shn';
#     contourfname='./contour.ctr';
#     if not os.path.isfile(IsoMishaexe):
#         print 'IsoMishaexe does not exist!';
#         return
#     if not os.path.isfile(contourfname):
#         print 'Contour file does not exist!';
#         return;
#     per_list=per_array.tolist();
#     RUNMISHASMOOTH=partial(RunMishaSmooth, datadir=datadir, outdir=outdir, minlon=minlon, maxlon=maxlon, minlat=minlat, maxlat=maxlat,\
#         dlon=dlon, dlat=dlat, stepInte=stepInte, lengthcell=lengthcell, datatype=datatype, chpair=chpair, alpha=alpha, sigma=sigma, beta=beta,\
#         data_pre=data_pre, outpre=outpre); 
#     pool = mp.Pool()
#     pool.map(RUNMISHASMOOTH, per_list) #make our results with a map call
#     pool.close() #we are not adding any more processes
#     pool.join() #tell it to wait until all threads are done before going on
#     print 'End of Running Smooth Misha Tomography ( Parallel ) !'
#     return;
# 
# def RunMishaQC(per, isoFlag, datadir, outdir, minlon, maxlon, minlat, maxlat, \
#         dlon=0.5, dlat=0.5, stepInte=0.1, lengthcell=0.5, datatype='ph', Wavetype='R',\
#         alpha=850, sigma=175, beta=1, crifactor=0.5, crilimit=10., data_pre='N_INIT_', alpha1=3000, sigma1=500, beta1=100, outpre='QC_', \
#         lengthcellAni=1.0, AniparaFlag=1, xZone=2, alphaAni0=1200, betaAni0=1, sigmaAni0=200, alphaAni1=1000, sigmaAni1=100):
#     """
#     Run Misha's Tomography Code with quality control based on preliminary run of RunMishaSmooth.
#     This function is designed to discard outliers in aftan results (Quality Control), and then do tomography.
#     
#     Mishaexe - Path to Misha's Tomography code executable ( isotropic/anisotropic version, determined by isoFlag )
#     contourfname - Path to contour file (see the manual for detailed description)
#     ----------------------------------------------------------------------------
#     Input format:
#     datadir+'/'+per+'_'+datatype+'/'+data_pre+str(alpha1)+'_'+str(sigma1)+'_'+str(beta1)+'_'+per+'.resid'
#     e.g. datadir/10_ph/N_INIT_3000_500_100_10.0.resid
#     
#     Intermediate output format:
#     outdir+'/'+per+'_'+datatype+'/QC_'+per+'_'+Wavetype+'_'+datatype+'.lst'
#     e.g. outdir/10_ph/QC_10_R_ph.lst
#     
#     Output format:
#     e.g. 
#     Prefix: outdir/10_ph/QC_850_175_1  OR outdir/10_ph/QC_AZI_R_1200_200_1000_100_1
#     
#     Output file:
#     outdir/10_ph/QC_850_175_1_10.1 etc. 
#     OR
#     outdir/10_ph/QC_AZI_R_1200_200_1000_100_1_10.1 etc. (see the manual for detailed description of output suffix)
#     ----------------------------------------------------------------------------
#     References:
#     Barmin, M. P., M. H. Ritzwoller, and A. L. Levshin. "A fast and reliable method for surface wave tomography."
#         Monitoring the Comprehensive Nuclear-Test-Ban Treaty: Surface Waves. Birkh?user Basel, 2001. 1351-1375.
#     """
#     if isoFlag==True:
#         Mishaexe='./TOMO_MISHA/itomo_sp_cu_shn';
#     else:
#         Mishaexe='./TOMO_MISHA_AZI/tomo_sp_cu_s_shn-.1/tomo_sp_cu_s_shn_.1';
#         outpre=outpre+'AZI_';
#     contourfname='./contour.ctr';
#     if not os.path.isfile(Mishaexe):
#         print 'Mishaexe does not exist!';
#         return
#     if not os.path.isfile(contourfname):
#         print 'Contour file does not exist!';
#         return;
#     infname=datadir+'/'+'%g'%( per ) +'_'+datatype+'/'+data_pre+str(alpha1)+'_'+str(sigma1)+'_'+str(beta1)+'_'+'%g'%( per )+'.resid' ;
#     Inarray=np.loadtxt(infname);   
#     res_tomo=Inarray[:,7];
#     cri_res=crifactor*per;
#     if cri_res>crilimit:
#         cri_res=crilimit;
#     QC_arr= Inarray[abs(res_tomo)<cri_res,:];
#     outArray=QC_arr[:,:8];
#     outper=outdir+'/'+'%g'%( per ) +'_'+datatype ;
#     if not os.path.isdir(outper):
#         os.makedirs(outper);
#     QCfname=outper+'/QC_'+'%g'%( per ) +'_'+Wavetype+'_'+datatype+'.lst';
#     np.savetxt(QCfname, outArray, fmt='%g');
#     if isoFlag==True:
#         outpre=outper+'/'+outpre+str(alpha)+'_'+str(sigma)+'_'+str(beta);
#     else:
#         outpre=outper+'/'+outpre+Wavetype+'_'+str(alphaAni0)+'_'+str(sigmaAni0)+'_'+str(alphaAni1)+'_'+str(sigmaAni1)+'_'+str(betaAni0);
#     temprunsh='temp_'+'%g_QC.sh' %(per);
#     f=open(temprunsh,'wb')
#     f.writelines('%s %s %s %g << EOF \n' %(Mishaexe, QCfname, outpre,per ));
#     if isoFlag==True:
#         f.writelines('me \n4 \n5 \n%g \n6 \n%g \n%g \n%g \n' %( beta, alpha, sigma, sigma) );
#         f.writelines('7 \n%g %g %g \n8 \n%g %g %g \n12 \n%g \n%g \n16 \n' %(minlat, maxlat, dlat, minlon, maxlon, dlon, stepInte, lengthcell) );
#         f.writelines('v \nq \ngo \nEOF \n' );
#     else:
#         if datatype=='ph':
#             Dtype='P'
#         else:
#             Dtype='G'
#         f.writelines('me \n4 \n5 \n%g %g %g \n6 \n%g %g %g \n' %( minlat, maxlat, dlat, minlon, maxlon, dlon) );
#         f.writelines('10 \n%g \n%g \n%s \n%s \n%g \n%g \n11 \n%d \n' %(stepInte, xZone, Wavetype, Dtype, lengthcell, lengthcellAni, AniparaFlag) );
#         f.writelines('12 \n%g \n%g \n%g \n%g \n' %(alphaAni0, betaAni0, sigmaAni0, sigmaAni0) );
#         f.writelines('13 \n%g \n%g \n%g \n' %(alphaAni1, sigmaAni1, sigmaAni1) );
#         f.writelines('19 \n25 \n' );
#         f.writelines('v \nq \ngo \nEOF \n' );
#     f.close();
#     call(['bash', temprunsh]);
#     os.remove(temprunsh);
#     return;
# 
# def RunMishaQCParallel(per_array, isoFlag, datadir, outdir, minlon, maxlon, minlat, maxlat, \
#         dlon=0.5, dlat=0.5, stepInte=0.1, lengthcell=0.5, datatype='ph', Wavetype='R',\
#         alpha=850, sigma=175, beta=1, crifactor=0.5, crilimit=10., data_pre='N_INIT_', alpha1=3000, sigma1=500, beta1=100, outpre='QC_', \
#         lengthcellAni=1.0, AniparaFlag=1, xZone=2, alphaAni0=1200, betaAni0=1, sigmaAni0=200, alphaAni1=1000, sigmaAni1=100):
#     """
#     Parallelly run Misha's Tomography Code with quality control based on preliminary run of RunMishaSmooth for a period array.
#     This function is designed to discard outliers in aftan results (Quality Control), and then do tomography.
#     
#     Mishaexe - Path to Misha's Tomography code executable ( isotropic/anisotropic version, determined by isoFlag )
#     contourfname - Path to contour file (see the manual for detailed description)
#     ----------------------------------------------------------------------------
#     Input format:
#     datadir+'/'+per+'_'+datatype+'/'+data_pre+str(alpha1)+'_'+str(sigma1)+'_'+str(beta1)+'_'+per+'.resid'
#     e.g. datadir/10_ph/N_INIT_3000_500_100_10.resid
#     
#     Intermediate output format:
#     outdir+'/'+per+'_'+datatype+'/QC_'+str(per)+'_'+Wavetype+'_'+datatype+'.lst'
#     e.g. outdir/10_ph/QC_10_R_ph.lst
#     
#     Output format:
#     e.g. 
#     Prefix: outdir/10_ph/QC_850_175_1  OR outdir/10_ph/QC_AZI_R_1200_200_1000_100_1
#     
#     Output file:
#     outdir/10_ph/QC_850_175_1_10.1 etc. 
#     OR
#     outdir/10_ph/QC_AZI_R_1200_200_1000_100_1_10.1 etc. (see the manual for detailed description of output suffix)
#     ----------------------------------------------------------------------------
#     References:
#     Barmin, M. P., M. H. Ritzwoller, and A. L. Levshin. "A fast and reliable method for surface wave tomography."
#         Monitoring the Comprehensive Nuclear-Test-Ban Treaty: Surface Waves. Birkh?user Basel, 2001. 1351-1375.
#     """
#     per_list=per_array.tolist();
#     RUNMISHAQC=partial(RunMishaQC, isoFlag=isoFlag, datadir=datadir, outdir=outdir, minlon=minlon, maxlon=maxlon, minlat=minlat, maxlat=maxlat,\
#         dlon=dlon, dlat=dlat, stepInte=stepInte, lengthcell=lengthcell, datatype=datatype, Wavetype=Wavetype,\
#         alpha=alpha, sigma=sigma, beta=beta, crifactor=crifactor, crilimit=crilimit, data_pre=data_pre, alpha1=alpha1, sigma1=sigma1, beta1=beta1, outpre=outpre, \
#         lengthcellAni=lengthcellAni, AniparaFlag=AniparaFlag, xZone=xZone, alphaAni0=alphaAni0, betaAni0=betaAni0, sigmaAni0=sigmaAni0, alphaAni1=alphaAni1, sigmaAni1=sigmaAni1); 
#     pool = mp.Pool()
#     pool.map(RUNMISHAQC, per_list) #make our results with a map call
#     pool.close() #we are not adding any more processes
#     pool.join() #tell it to wait until all threads are done before going on
#     print 'End of Running Quality Controlled Misha Tomography ( Parallel ) !'
#     return;
# 
# def GetCorrectedMap(per, glbdir, regdir, outdir, reg_pre='QC_850_175_1_', glbpre='smpkolya_phv_R_', outpre='smpkolya_phv_R_'):
#     """
#     Get corrected global phave V map using a regional phase V map.
#     ----------------------------------------------------------------
#     Input format:
#     glbdir/glbpre+per - global phase V map
#     regdir/str(float(per))_ph/reg_pre+per.1 - e.g. regdir/40.0_ph/QC_850_175_1_40.1
# 
#     Output format:
#     outdir/outpre+str(int(per))
#     ----------------------------------------------------------------
#     """
#     inglobalfname=glbdir+'/'+glbpre+str(int(per));
#     inregfname=regdir+'/'+'%g'%( per ) +'_ph'+'/'+reg_pre+str(float(per))+'.1' ;
#     if not ( os.path.isfile(inglobalfname) and os.path.isfile(inregfname) ):
#         inregfname=regdir+'/'+'%g'%( per ) +'_ph'+'/'+reg_pre+'%g.1' %(per);
#         if not ( os.path.isfile(inglobalfname) and os.path.isfile(inregfname) ):
#             print 'File not exists for period: ', per;
#             print inglobalfname,inregfname
#             return
#     outfname=outdir+'/'+outpre+'%g' %(per);
#     print inglobalfname, inregfname, outfname
#     InregArr=np.loadtxt(inregfname);
#     InglbArr=np.loadtxt(inglobalfname);
#     outArr=InglbArr;
#     (Lglb, m)=InglbArr.shape;
#     (Lreg, m)=InregArr.shape;
#     for i in np.arange(Lglb):
#         lonG=InglbArr[i,0];
#         latG=InglbArr[i,1];
#         phVG=InglbArr[i,2];
#         for j in np.arange(Lreg):
#             lonR=InregArr[j,0];
#             latR=InregArr[j,1];
#             phVR=InregArr[j,2];
#             if abs(lonR-lonG)<0.05 and abs(latR-latG)<0.05:
#                 phVG=phVR;
#         outArr[i,2]=phVG
#     np.savetxt(outfname, outArr, fmt='%g %g %.4f');
#     return;
#         
# def GetCorrectedMapParallel(per_array, glbdir, regdir, outdir, reg_pre='QC_850_175_1_', glbpre='smpkolya_phv_R_', outpre='smpkolya_phv_R_'):
#     """
#     Get corrected global phave V map using a regional phase V map, do for a period array parallelly.
#     ----------------------------------------------------------------
#     Input format:
#     glbdir/glbpre+per - global phase V map
#     regdir/per_ph/reg_pre+per.1 - e.g. regdir/40_ph/QC_850_175_1_40.1
# 
#     Output format:
#     outdir/outpre+per
#     ----------------------------------------------------------------
#     """
#     if not os.path.isdir(outdir):
#         os.makedirs(outdir);
#     per_list=per_array.tolist();
#     GETCMAP=partial(GetCorrectedMap, glbdir=glbdir, regdir=regdir, outdir=outdir, reg_pre=reg_pre, glbpre=glbpre, outpre=outpre); 
#     pool = mp.Pool()
#     pool.map(GETCMAP, per_list) #make our results with a map call
#     pool.close() #we are not adding any more processes
#     pool.join() #tell it to wait until all threads are done before going on
#     print 'End of Get Corrected Global Phase V Maps( Parallel ) !'
#     return;