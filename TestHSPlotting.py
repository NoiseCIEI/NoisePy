import CURefPy as ref#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
stacode='YRT';
obsHSstream=ref.HStripStream();
diffHSstream=ref.HStripStream();
repHSstream=ref.HStripStream();
rep0HSstream=ref.HStripStream();
rep1HSstream=ref.HStripStream();
rep2HSstream=ref.HStripStream();
# indata=np.loadtxt('testdataHS');
lfadata=np.loadtxt('lfadata.dat');
drdata=np.loadtxt('drdata.dat');
rdata=np.loadtxt('rdata.dat');
A0=np.loadtxt('A0.dat');
lfrdata1=np.loadtxt('lfrdata1.dat');
lfrdata2=np.loadtxt('lfrdata2.dat');
baz=np.loadtxt('bazdataHS');

obsHSstream.GetDatafromArr(stacode=stacode, indata=lfadata, baz=baz, dt=0.02);
diffHSstream.GetDatafromArr(stacode=stacode, indata=drdata, baz=baz, dt=0.02);
repHSstream.GetDatafromArr(stacode=stacode, indata=rdata, baz=baz, dt=0.02);
rep0HSstream.GetDatafromArr(stacode=stacode, indata=A0, baz=baz, dt=0.02);
rep1HSstream.GetDatafromArr(stacode=stacode, indata=lfrdata1, baz=baz, dt=0.02);
rep2HSstream.GetDatafromArr(stacode=stacode, indata=lfrdata2, baz=baz, dt=0.02);
        

HSDataBase=ref.HarmonicStrippingDataBase(obsST=obsHSstream, diffST=diffHSstream, repST=repHSstream,\
            repST0=rep0HSstream, repST1=rep1HSstream, repST2=rep2HSstream);
# HSSTR.GetDatafromArr(stacode='YRT', indata=indata, baz=baz, dt=0.02);
HSDataBase.PlotHSStreams(outdir='/lustre/janus_scratch/life9360/Ref_TEST/my_receiverF/YRT', \
        stacode='YRT', longitude=111, latitude=45)