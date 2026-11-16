"""Light-curve data for one flare (GOES, GBM, LLE, LAT), shared by the GUI server and the report.

Only numpy/astropy: no threeML import. All times are relative to the flare T0 (s).
"""
import os

import numpy as np
from astropy.io import fits

from sunflare import common


def _rebin(t, w, edges):
    """Sum weights w at times t into edges."""
    idx = np.digitize(t, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1) & np.isfinite(w)
    return np.bincount(idx[ok], weights=w[ok], minlength=len(edges) - 1)


def goes(fdir, edges=None):
    p = os.path.join(fdir, 'goes_xrs.csv')
    d = np.loadtxt(p, delimiter=',', skiprows=1, ndmin=2)
    t, a, b = d[:, 0], d[:, 1], d[:, 2]
    if edges is None:
        return t, a, b
    n = _rebin(t, np.isfinite(b).astype(float), edges)
    sa = _rebin(t, np.where(np.isfinite(a), a, 0), edges)
    sb = _rebin(t, np.where(np.isfinite(b), b, 0), edges)
    with np.errstate(invalid='ignore', divide='ignore'):
        return 0.5 * (edges[1:] + edges[:-1]), np.where(n > 0, sa / n, np.nan), np.where(n > 0, sb / n, np.nan)


def gbm_file(fdir, det):
    return common.find_version(os.path.join(fdir, 'GBM_data', f'glg_cspec_{det}_*.pha'))


def gbm(fdir, det, lo, hi, edges=None, t0=None):
    """Count rate of one GBM detector in [lo, hi] keV; rebinned into edges if given."""
    pha = gbm_file(fdir, det)
    with fits.open(pha) as h:
        sp, eb = h['SPECTRUM'].data, h['EBOUNDS'].data
        t0 = h[0].header['TRIGTIME'] if t0 is None else t0
        band = (eb['E_MAX'] > lo) & (eb['E_MIN'] < hi)
        tc = 0.5 * (sp['TIME'] + sp['ENDTIME']) - t0
        cnt = sp['COUNTS'][:, band].sum(1).astype(float)
        expo = sp['EXPOSURE'].astype(float)
        elo, ehi = float(eb['E_MIN'][band][0]), float(eb['E_MAX'][band][-1])
    if edges is None:
        return tc, cnt / np.maximum(expo, 1e-9), (elo, ehi)
    c, e = _rebin(tc, cnt, edges), _rebin(tc, expo, edges)
    with np.errstate(invalid='ignore', divide='ignore'):
        return 0.5 * (edges[1:] + edges[:-1]), np.where(e > 0, c / e, np.nan), (elo, ehi)


def lle(fdir, info, emin, emax, edges):
    f = info.get('lle_file')
    f = f if f and os.path.isabs(f) else os.path.join(fdir, f or '')
    with fits.open(f) as h:
        ev = h['EVENTS'].data
        m = (ev['ENERGY'] > emin) & (ev['ENERGY'] < emax)
        c, _ = np.histogram(ev['TIME'][m] - info['T0'], bins=edges)
        rng = (h[0].header['TSTART'] - info['T0'], h[0].header['TSTOP'] - info['T0'])
    rate = c / np.diff(edges)
    rate = np.where((edges[:-1] >= rng[0]) & (edges[1:] <= rng[1]), rate, np.nan)
    return 0.5 * (edges[1:] + edges[:-1]), rate


def lat_ft1(fdir, info, sun_centered=False):
    name = info['lat_name']
    sub = 'LAT_data_sun' if sun_centered else 'LAT_data'
    return os.path.join(fdir, sub, 'bn' + name, f'gll_ft1_tr_bn{name}_v00.fit')


def lat_events(fdir, info, roi=10.0, emin=100.0, zmax=100.0, sun_centered=False):
    """LAT events within `roi` of the Sun (tracking its motion, or (0, 0) in the Sun-centred file)."""
    from sunflare import lat_data
    f = lat_ft1(fdir, info, sun_centered)
    if sun_centered:
        return lat_data.events(f, info['T0'], (0.0, 0.0), roi, emin, zmax)
    return lat_data.events(f, info['T0'], None, roi, emin, zmax, sun_track=True)


def lat_track(fdir):
    return common.load_json(os.path.join(fdir, 'lat_track.json'), None)


def lat_rate(ev, edges, track=None):
    """Event rate (counts/s) in edges; NaN where the Sun is outside the visibility windows."""
    c, _ = np.histogram(ev['t'], bins=edges)
    return 0.5 * (edges[1:] + edges[:-1]), c / np.diff(edges)
