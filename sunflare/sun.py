"""Sun position helpers built on FermiSunPy (sunpos / suncenter) and sunpy."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import suncenter  # noqa: E402  (FermiSunPy)
import sunpos     # noqa: E402,F401

R_SUN_DEG = 0.2666   # mean angular radius of the solar disk


def sun_radec(met):
    """(RA, Dec) of the Sun [deg, J2000] at MET (scalar or array)."""
    ra, dec = suncenter.sun_radec(met)
    if np.ndim(met) == 0:
        return float(ra[0]), float(dec[0])
    return ra, dec


def to_sun_frame(ra, dec, met):
    """Equatorial -> Sun-centred frame of suncenter.py (lon - lon_sun, ecliptic lat), lon in [-180, 180]."""
    s_ra, s_dec = suncenter.sun_radec(met)
    sun_lon, _ = suncenter.equatorial2ecliptic_vec(s_ra, s_dec)
    lon, lat = suncenter.equatorial2ecliptic_vec(ra, dec)
    return (lon - sun_lon + 180.0) % 360.0 - 180.0, lat


def from_sun_frame(dlon, lat, met):
    """Inverse of to_sun_frame (scalar MET)."""
    s_ra, s_dec = sun_radec(met)
    sun_lon, _ = suncenter.equatorial2ecliptic_vec(s_ra, s_dec)
    # suncenter uses a fixed J2000 obliquity rotation: undo it directly
    lon = np.radians((np.asarray(dlon) + float(np.atleast_1d(sun_lon)[0])) % 360.0)
    lat = np.radians(np.asarray(lat))
    eps = np.radians(23.439292)
    dec = np.arcsin(np.sin(lat) * np.cos(eps) + np.cos(lat) * np.sin(lon) * np.sin(eps))
    ra = np.arctan2(np.sin(lon) * np.cos(eps) - np.tan(lat) * np.sin(eps), np.cos(lon))
    return np.degrees(ra) % 360.0, np.degrees(dec)


def helioprojective_offset(ra, dec, met):
    """Offset of (ra, dec) from the Sun centre as seen from Earth: (Tx, Ty) in arcsec, and the distance
    in units of the solar radius. Uses sunpy's Helioprojective frame (solar north up)."""
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from astropy.time import Time
        from sunpy.coordinates import frames, get_earth
        t = Time('2001-01-01T00:00:00', scale='utc') + float(met) * u.s
        c = SkyCoord(ra * u.deg, dec * u.deg, distance=1 * u.AU, frame='gcrs', obstime=t)
        # put the point at a large distance along the line of sight; only the direction matters
        hpc = c.transform_to(frames.Helioprojective(observer=get_earth(t), obstime=t))
        r_sun = frames.Helioprojective(observer=get_earth(t), obstime=t).angular_radius
        tx, ty = float(hpc.Tx.to_value(u.arcsec)), float(hpc.Ty.to_value(u.arcsec))
        return tx, ty, float(np.hypot(tx, ty) / r_sun.to_value(u.arcsec)), float(r_sun.to_value(u.arcsec))
    except Exception:  # sunpy missing or frame problem: plain angular distance
        from astropy.coordinates import SkyCoord
        s = SkyCoord(*sun_radec(met), unit='deg')
        p = SkyCoord(ra, dec, unit='deg')
        off = s.spherical_offsets_to(p)
        return (float(off[0].arcsec), float(off[1].arcsec), float(s.separation(p).deg / R_SUN_DEG),
                R_SUN_DEG * 3600)
