import ASDFDBase
import numpy as np
import timeit
import matplotlib.pyplot as plt
dset=ASDFDBase.quakeASDF('../ref_ALASKA_mp.h5')
# dset.get_events(startdate='2004-01-01', enddate='2008-01-01', Mmin=5.5, magnitudetype='mb')
# dset.get_stations(channel='BH*', station='YRT', network='XR')
# t1=timeit.default_timer()
# st=dset.get_body_waveforms()
# st=dset.get_body_waveforms_mp( outdir='./downloaded_P', verbose=False, nprocess=6)
# t2=timeit.default_timer()
# print t2-t1, 'sec'
# dset.write2sac(station='YRT', network='XR', evnumb=853)
# t1=timeit.default_timer()
# try: del dset.auxiliary_data.RefR
# except: pass
# try: del dset.auxiliary_data.RefRscaled
# except: pass
# try: del dset.auxiliary_data.RefRmoveout
# except: pass
# try: del dset.auxiliary_data.RefRstreback
# except: pass
# # # 
# # # # # 
# dset.compute_ref_mp(outdir='/work3/leon/ref_working', verbose=True, nprocess=4)
try: del dset.auxiliary_data.RefRHS
except: pass

dset.harmonic_stripping(outdir='.')
# t2=timeit.default_timer()
# print t2-t1, 'sec'
dset.plot_ref(network='XR', station='YRT', phase='P', datatype='RefRHS')