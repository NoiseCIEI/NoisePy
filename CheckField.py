import ses3d_fields as sf#!/usr/bin/env python
datadir='/lustre/janus_scratch/life9360/SES3D_WorkingDir/MODELS/MODELS/'
field=sf.ses3d_fields(datadir,'earth model')
field.plot_colat_slice(component='vsv', colat='45', valmin=2.0, valmax=7.0);

# field=sf.ses3d_fields(datadir,'velocity_snapshot')
# field.plot_depth_slice('vx',0.0,-9e-8,9e-8,iteration=1000,res='i')