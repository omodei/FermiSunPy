#! /usr/bin/env python
import os,sys
import astropy.io.fits as pyfits
#from array import array
from math import sin,cos,atan2,asin,radians,degrees,sqrt
import sunpos
import ProgressBar
##################################################
##  Functions
##################################################

def angsep(x1,y1,x2,y2):
    """ spherical angle separation, aka haversine formula
    input and output are in degrees
    Signed: the distance is equivalent to x2-x1
    """    
    sign = +1
    if x1>x2 and x1-x2>180: sign = -1
    if x1<x2 and x2-x1>180: sign = -1
    
    dlat = radians(y2 - y1)
    dlon = radians(x2 - x1)
    y1 = radians(y1)
    y2 = radians(y2)
    a = sin(dlat/2.)*sin(dlat/2.) + cos(y1)*cos(y2)*sin(dlon/2.)*sin(dlon/2.)
    c  = 2*atan2(sqrt(a), sqrt(1.-a))
    return sign*degrees(c)

def angsep2(x1,x2):
    """ cilindrical angle separation
    input and output are in degrees  
    """
    dx=x1-x2
    if dx > 180: dx -=360
    if dx < -180: dx+=360
    return dx


def convertFT1(ft1Name):
    """
    This function converts an FT1 file to Sun-centered coordinates.

    :param ft1Name: Name of the input FT1 file.
    :return: Name of the new FT1 file with Sun-centered coordinates.

    """
    ft1NewName=ft1Name.replace('.fit','_sun.fit')
    os.system('cp %s %s' %(ft1Name,ft1NewName))
    ft1 = pyfits.open(ft1NewName,'update')
    ft1.info()
    ft1['EVENTS'].header['TLMIN2'] = -180
    ft1['EVENTS'].header['TLMAX2'] =  180
    EVENTS   = ft1['EVENTS'].data
    RA       = EVENTS.field('RA')
    DEC      = EVENTS.field('DEC')
    TIME     = EVENTS.field('TIME')
    N        = len(TIME)
    pb       = ProgressBar.progressbar(20)
    for i in range(N):
        pb.go(i,N)
        MET=TIME[i]
        #1 Get the sun position:
        s_p   = sunpos.getSunPosition(MET)
        s_ra  = s_p.ra()
        s_dec = s_p.dec()
        #2 convert
        ra    =  RA[i]
        dec   = DEC[i]
        ecl_lon, ecl_lat       = sunpos.equatorial2ecliptic(ra,dec)
        sun_ecl_lon,sun_ecl_lat= sunpos.equatorial2ecliptic(s_ra,s_dec)
        #3 New coordinates:
        RA[i]  = angsep2(ecl_lon,sun_ecl_lon) # this also remove the Sun Long.
        DEC[i] = ecl_lat
        pass
    ft1.flush()
    return ft1NewName

def convertFT2(ft2Name):
    """
    This function converts an FT2 file to Sun-centered coordinates.
    
    :param ft2Name: Name of the input FT2 file.
    :return: Name of the new FT2 file with Sun-centered coordinates.

    """

    ft2NewName=ft2Name.replace('.fit','_sun.fit')
    os.system('cp %s %s' %(ft2Name,ft2NewName))
    ft2 = pyfits.open(ft2NewName,'update')
    ft2.info()
    
    SC_DATA  = ft2['SC_DATA'].data

    RA_ZENITH       = SC_DATA.field('RA_ZENITH')
    DEC_ZENITH      = SC_DATA.field('DEC_ZENITH')

    RA_SCZ = SC_DATA.field('RA_SCZ')
    DEC_SCZ = SC_DATA.field('DEC_SCZ')

    RA_SCX = SC_DATA.field('RA_SCX')
    DEC_SCX= SC_DATA.field('DEC_SCX')
    
    START     = SC_DATA.field('START')
    STOP      = SC_DATA.field('STOP')
    TIME      = (START+STOP)/2
    N        = len(TIME)
    pb       = ProgressBar.progressbar(20)
    for i in range(N):
        pb.go(i,N)
        MET=TIME[i]

        #1 Get the sun position:
        s_p   = sunpos.getSunPosition(MET)
        s_ra  = s_p.ra()
        s_dec = s_p.dec()
        
        #2 convert
        ra_sczenith  = RA_ZENITH[i]
        dec_sczenith = DEC_ZENITH[i]

        ra_scz  = RA_SCZ[i]
        dec_scz = DEC_SCZ[i]

        ra_scx  = RA_SCX[i]
        dec_scx = DEC_SCX[i]


        ecl_lon_sczenith, ecl_lat_sczenith  = sunpos.equatorial2ecliptic(ra_sczenith,dec_sczenith)
        ecl_lon_scz, ecl_lat_scz            = sunpos.equatorial2ecliptic(ra_scz,dec_scz)
        ecl_lon_scx, ecl_lat_scx            = sunpos.equatorial2ecliptic(ra_scx,dec_scx)
        sun_ecl_lon,sun_ecl_lat             = sunpos.equatorial2ecliptic(s_ra,s_dec)
        
        RA_ZENITH[i]  = angsep2(ecl_lon_sczenith,sun_ecl_lon) # this also remove the Sun Long.
        DEC_ZENITH[i] = ecl_lat_sczenith
        
        RA_SCZ[i]  = angsep2(ecl_lon_scz,sun_ecl_lon) # this also remove the Sun Long.
        DEC_SCZ[i] = ecl_lat_scz
        
        RA_SCX[i]  = angsep2(ecl_lon_scx,sun_ecl_lon) # this also remove the Sun Long.
        DEC_SCX[i] = ecl_lat_scx        
        
        pass
    ft2.flush()
    return ft2NewName



##################################################
##  Vectorized versions (same transformation, ~1000x faster)
##################################################
import shutil
import numpy as np


def sun_radec(met, step=60.0):
    """Sun (RA, Dec) in degrees (J2000) for an array of MET.

    sunpos.getSunPosition (fermitools ephemeris) is evaluated on a grid of `step` seconds and
    interpolated (the Sun moves ~0.04 deg/hour); falls back to astropy if pyLikelihood is missing.
    """
    met = np.atleast_1d(np.asarray(met, dtype=float))
    grid = np.arange(met.min() - step, met.max() + 2 * step, step)
    try:
        radec = np.array([(p.ra(), p.dec()) for p in map(sunpos.getSunPosition, grid)])
    except Exception:
        from astropy.time import Time
        from astropy.coordinates import get_sun
        import astropy.units as u
        t = Time('2001-01-01T00:00:00', scale='utc') + grid * u.s
        s = get_sun(t)
        radec = np.column_stack([s.ra.deg, s.dec.deg])
    ra, dec = np.radians(radec[:, 0]), np.radians(radec[:, 1])
    xyz = [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]
    x, y, z = [np.interp(met, grid, c) for c in xyz]
    n = np.sqrt(x * x + y * y + z * z)
    return np.degrees(np.arctan2(y, x)) % 360.0, np.degrees(np.arcsin(z / n))


def equatorial2ecliptic_vec(ra, dec, eps=23.439292):
    """Vectorized sunpos.equatorial2ecliptic (same rotation, J2000 obliquity)."""
    x, y, p = np.radians(ra), np.radians(dec), np.radians(eps)
    lon = np.degrees(np.arctan2(np.sin(x) * np.cos(p) + np.tan(y) * np.sin(p), np.cos(x)))
    lat = np.degrees(np.arcsin(np.sin(y) * np.cos(p) - np.cos(y) * np.sin(x) * np.sin(p)))
    return lon % 360.0, lat


def _sun_frame(ra, dec, sun_lon, wrap360):
    lon, lat = equatorial2ecliptic_vec(ra, dec)
    dlon = (lon - sun_lon + 180.0) % 360.0 - 180.0      # = angsep2(lon, sun_lon)
    return (dlon % 360.0 if wrap360 else dlon), lat


def convertFT1_fast(ft1Name, outName=None, wrap360=False):
    """Sun-centred copy of an FT1 file (vectorized convertFT1).

    RA  -> ecliptic longitude minus the Sun's longitude, DEC -> ecliptic latitude.
    The Sun is at (0, ~0). wrap360=True stores RA in [0, 360) (same directions; safer for gtselect);
    otherwise RA is in [-180, 180] as in convertFT1.
    """
    outName = outName or ft1Name.replace('.fit', '_sun.fit')
    shutil.copy(ft1Name, outName)
    with pyfits.open(outName, mode='update') as ft1:
        ev = ft1['EVENTS']
        ev.header['TLMIN2'], ev.header['TLMAX2'] = (0, 360) if wrap360 else (-180, 180)
        d = ev.data
        if len(d):
            s_ra, s_dec = sun_radec(d['TIME'])
            sun_lon, _ = equatorial2ecliptic_vec(s_ra, s_dec)
            d['RA'], d['DEC'] = _sun_frame(d['RA'], d['DEC'], sun_lon, wrap360)
        ft1.flush()
    return outName


def convertFT2_fast(ft2Name, outName=None, wrap360=False):
    """Sun-centred copy of an FT2 file (vectorized convertFT2): SCZ, SCX and ZENITH directions."""
    outName = outName or ft2Name.replace('.fit', '_sun.fit')
    shutil.copy(ft2Name, outName)
    with pyfits.open(outName, mode='update') as ft2:
        d = ft2['SC_DATA'].data
        s_ra, s_dec = sun_radec(0.5 * (d['START'] + d['STOP']))
        sun_lon, _ = equatorial2ecliptic_vec(s_ra, s_dec)
        for c in ('ZENITH', 'SCZ', 'SCX'):
            d['RA_' + c], d['DEC_' + c] = _sun_frame(d['RA_' + c], d['DEC_' + c], sun_lon, wrap360)
        ft2.flush()
    return outName


if __name__=='__main__':
    import argparse
    argparser = argparse.ArgumentParser(description='Convert FT1/FT2 files to Sun-centered coordinates.')
    argparser.add_argument('-ft1', metavar='ft1', type=str, help='Input FT1 file to convert.')
    argparser.add_argument('-ft2', metavar='ft2', type=str, help='Input FT2 file to convert.')
    argparser.add_argument('--legacy', action='store_true', help='Use the original event-by-event loop (slow).')
    args = argparser.parse_args()

    c1, c2 = (convertFT1, convertFT2) if args.legacy else (convertFT1_fast, convertFT2_fast)
    if args.ft1 is not None: print(c1(args.ft1))
    if args.ft2 is not None: print(c2(args.ft2))
