import ASDFDBase
import timeit
dset=ASDFDBase.noiseASDF('./COR_WUS.h5')
# 
# 
# # dset.read_stationtxt_ind('/work3/leon/ancc-1.0-0/Station.lst', chans=['LHZ'])
# # dset.read_xcorr('/work3/leon/ancc-1.0-0', pfx='COR')
# 
# 
# # dset.xcorr_prephp(outdir='/work3/leon/PRE_PHP', mapfile='./MAPS/smpkolya_phv')
# # try:
# #     del dset.auxiliary_data.DISPbasic1
# # except:
# #     pass
# # try:
# #     del dset.auxiliary_data.DISPbasic2
# # except:
# #     pass
# # try:
# #     del dset.auxiliary_data.DISPpmf1
# # except:
# #     pass
# # try:
# #     del dset.auxiliary_data.DISPpmf2
# # except:
# #     pass
# # dset.xcorr_aftan_mp(outdir='/work3/leon/WUS_workingdir', prephdir='/work3/leon/PRE_PHP_R', f77=True, nprocess=4)
# # 
# # try:
# #     del dset.auxiliary_data.DISPpmf2interp
# # except:
# #     pass
# 
# # try:
# #     del dset.auxiliary_data.DISPpmf2interp
# # except:
# #     pass
# # dset.interp_disp()
# # 
# dset.xcorr_raytomoinput(outdir='./ray_tomo_data_v2')
# dset.xcorr_raytomoinput_v2(outdir='./ray_tomo_data_v2')


import raytomo
dset=raytomo.RayTomoDataSet('./ray_tomo_WUS.h5')
# dset.set_input_parameters(minlon=235., maxlon=255., minlat=31., maxlat=50., data_pfx='raytomo_in_', smoothpfx='N_INIT_', qcpfx='QC_')
# dset.run_smooth(datadir='/home/lili/code/NoisePy/ray_tomo_data_v2', outdir='./ray_tomo_working_dir')
dset.run_qc(datadir='./ray_tomo_working_dir', outdir='./ray_tomo_working_dir')

