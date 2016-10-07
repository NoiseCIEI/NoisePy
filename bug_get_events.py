from obspy.clients.fdsn.client import Client
import obspy

client=Client('IRIS')
starttime=obspy.core.utcdatetime.UTCDateTime('2011-12-01')
endtime=obspy.core.utcdatetime.UTCDateTime('2011-12-31')
cat = client.get_events(starttime=starttime, endtime=endtime, minmagnitude=5.5, catalog='ISC', magnitudetype='mb')
cat