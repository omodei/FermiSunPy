#!/usr/bin/env python
import pyLikelihood as pl # needs fermitools
import os
from math import atan2,radians,degrees,sin,cos,asin,tan
# Helper functions for degree-radian conversion
def d(x): return degrees(x)
def r(x): return radians(x)
# Convert MET (Mission Elapsed Time) to Julian Date
def getJD(met):
    if met > 252460801: met-=1 # 2008 leap second
    if met > 157766400: met-=1 # 2005 leap second
    if met > 362793601: met-=1 # 2012 leap second    
    return  (pl.JulianDate.seconds(pl.JulianDate_missionStart())+met)/pl.JulianDate.secondsPerDay

# Convert MET to JulianDate object
def _jd_from_met(met):
    jd = pl.JulianDate(getJD(met))
    return jd 
# Get the Sun's position at a given MET
def getSunPosition( met ):
    if 'HEADAS' not in os.environ:
        raise RuntimeError('HEADAS environment variable not defined')
    
    os.environ['TIMING_DIR']=os.path.join(os.environ['HEADAS'],"refdata")
    sun=pl.SolarSystem(pl.SolarSystem.SUN)
    return sun.direction(_jd_from_met(met))

# Rotate coordinates (x,y) by angle p
def rotation(x,y,p):    
    xr=r(x)
    yr=r(y)
    pr=r(p)
    x1= d(atan2(sin(xr)*cos(pr) + tan(yr)*sin(pr), cos(xr)))
    y1 = d(asin(sin(yr)*cos(pr) - cos(yr)*sin(xr)*sin(pr)))
    return x1,y1

# Convert equatorial coordinates (RA, Dec) to ecliptic coordinates (Lon, Lat)
def equatorial2ecliptic(ra,dec):
    EPS = 23.439292 # THIS IS THE OBMIQUITY AS J2000
    (lon,lat)=rotation(ra,dec,EPS)
    if lon<0: lon+=360
    return (lon,lat)

# TEST EXAMPLE:
# 2012-12-05 00:00:00
# THE SUN Astrometric Positions  from http://aa.usno.navy.mil/cgi-bin/aa_geocentric.pl
# RA,DEC: 16 47 22.065,  -22 23 12.46
# in degrees: 251.841938, -22.386794
# converting in ecliptic (HEASARC): 253.252623,0.002996
if __name__=='__main__':
    import argparse
    argparser = argparse.ArgumentParser(description='Compute the Sun position for a given MET time.')
    argparser.add_argument('met', nargs='?', type=float, default=376358403.000,
                           help='Mission Elapsed Time (MET) in seconds (default: 376358403.000)')
    args = argparser.parse_args()
    met = args.met
    
    JD=getJD(met)
    sp=getSunPosition(met)
    sun_ra,sun_dec=sp.ra(),sp.dec()
    sun_lon,sun_lat=equatorial2ecliptic(sun_ra,sun_dec)
    print ('--------------------------------')
    print ('MET ......... :',met)
    print ('Julian Date...:',JD)
    print ('Sun R.A.,Dec..: %.4f,%.4f'%(sun_ra,sun_dec))
    print ('Sun Lon,Lat...: %.4f,%.4f'%(sun_lon,sun_lat))
    print ('--------------------------------')


