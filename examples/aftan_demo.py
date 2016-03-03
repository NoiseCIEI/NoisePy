import obspy
import noisepy as npy
import matplotlib.pyplot as plt
import numpy as np
f='./aftan_test_data/COR_SUMG_NEEM.SAC'

st=obspy.read(f)
tr=st[0]
tr1=npy.noisetrace(tr.data, tr.stats)
tr1.aftan(pmf=True)
tr1.ftanparam.writeDISP(f)
tr1.getSNR ()
tr1.SNRParam.writeAMPSNR(f)
tr1.ftanparam.FTANcomp(tr1.ftanparam)

tr1.plotftan()
plt.show()


