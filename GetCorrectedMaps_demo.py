import numpy as np
import matplotlib.pyplot as plt
import os;
import noisepy as npy
per_array=40+np.arange(5)*10
glbdir='/projects/life9360/code/devNoisePy/MAPS'
outdir='/projects/life9360/code/devNoisePy/ALASKA_MAPS'
regdir='/lustre/janus_scratch/life9360/ALASKA_tomo_test'
npy.GetCorrectedMap(per=40.0, glbdir=glbdir, regdir=regdir, outdir=outdir, reg_pre='QC_850_175_1_', glbpre='smpkolya_phv_R_', outpre='smpkolya_phv_R_')
# npy.GetCorrectedMapParallel(per_array, glbdir=glbdir, regdir=regdir, outdir=outdir, reg_pre='QC_850_175_1_', glbpre='smpkolya_phv_R_', outpre='smpkolya_phv_R_')

