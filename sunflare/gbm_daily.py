"""GBM continuous (daily) CSPEC data and Sun-pointed responses for solar flares.

* Daily CSPEC files come from the HEASARC (always available, no trigger needed).
  write_flare_cspec() writes a copy restricted to the analysis window with TRIGTIME = T0, so that
  threeML times are relative to the flare reference time (the daily files have no trigger time).
* Sun-pointed RSP2 files are produced by the GBM solar team on hesperia for every flare
  (glg_cspec_<det>_bnYYMMDD_HHMM_FFF_v00.rsp2). merge_rsp2() joins all the files that overlap the
  analysis window into a single RSP2 whose coverage is extended to the data range.
"""
import datetime
import os
import re
import shutil
import urllib.request

import numpy as np
from astropy.io import fits

DAILY_URL = 'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/daily/{y}/{m:02d}/{d:02d}/current/'
RSP_URL = 'https://hesperia.gsfc.nasa.gov/fermi/gbm/rsp/{y}{m:02d}{d:02d}/'


def _retry(fn, tries=4, wait=3.0):
    import time
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(wait * (i + 1))


def _get(url, dest, log=print, timeout=300):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    log(f'  downloading {url}')
    tmp = dest + '.part'

    def go():
        with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, 'wb') as f:
            shutil.copyfileobj(r, f)
    _retry(go)
    os.replace(tmp, dest)
    return dest


_LISTINGS = {}


def _listing(url):
    if url not in _LISTINGS:
        def go():
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read().decode('utf-8', 'replace')
        _LISTINGS[url] = _retry(go)
    return _LISTINGS[url]


def days_between(dt0, dt1):
    d = datetime.datetime(dt0.year, dt0.month, dt0.day)
    while d <= dt1:
        yield d
        d += datetime.timedelta(days=1)


def daily_cspec(det, day, cache, log=print):
    """Path of the daily CSPEC file for one detector and day (downloaded into `cache` if needed)."""
    url = DAILY_URL.format(y=day.year, m=day.month, d=day.day)
    html = _listing(url)
    files = sorted(set(re.findall(r'glg_cspec_%s_%s_v\d+\.pha' % (det, day.strftime('%y%m%d')), html)))
    if not files:
        raise FileNotFoundError(f'No daily CSPEC for {det} on {day:%Y-%m-%d}')
    name = files[-1]
    return _get(url + name, os.path.join(cache, day.strftime('%Y/%m/%d'), name), log)


def write_flare_cspec(daily_files, out, t0_met, tmin, tmax):
    """Join daily CSPEC files, keep [t0+tmin, t0+tmax], drop bad rows, set TRIGTIME = t0."""
    cols, ebounds, prim, spec_hdr, fmts = {}, None, None, None, {}
    for f in daily_files:
        with fits.open(f) as h:
            if prim is None:
                prim, spec_hdr = h[0].header.copy(), h['SPECTRUM'].header.copy()
                ebounds = fits.BinTableHDU(h['EBOUNDS'].data.copy(), h['EBOUNDS'].header.copy())
                fmts = {c.name: (c.format, c.unit) for c in h['SPECTRUM'].columns}
            d = h['SPECTRUM'].data
            keep = (d['TIME'] >= t0_met + tmin) & (d['ENDTIME'] <= t0_met + tmax) & (d['EXPOSURE'] > 0)
            for name in fmts:                      # field access returns the scaled (TZERO/BZERO) values
                cols.setdefault(name, []).append(np.array(d[name][keep]))
    if not cols or sum(len(x) for x in cols['TIME']) == 0:
        raise ValueError('No CSPEC data in the requested window')
    data = {k: np.concatenate(v) for k, v in cols.items()}
    order = np.argsort(data['TIME'], kind='stable')
    data = {k: v[order] for k, v in data.items()}
    good = np.append(np.diff(data['TIME']) > 0, True)          # drop duplicated / overlapping rows
    good &= np.insert(data['TIME'][1:] >= data['ENDTIME'][:-1] - 1e-3, 0, True)
    data = {k: v[good] for k, v in data.items()}
    newcols = []
    for name, (fmt, unit) in fmts.items():
        if name == 'COUNTS':
            fmt = fmt.rstrip('IJ') + 'J'                # unsigned 16-bit counts -> 32-bit, no BZERO
        newcols.append(fits.Column(name=name, format=fmt, unit=unit, array=data[name]))
    spec = fits.BinTableHDU.from_columns(newcols)
    for k in ('EXTNAME', 'TELESCOP', 'INSTRUME', 'DETNAM', 'OBSERVER', 'MJDREFI', 'MJDREFF', 'TIMESYS', 'TIMEUNIT',
              'TIMEREF', 'TASSIGN', 'CLOCKAPP', 'GPS_OUT', 'HDUCLASS', 'HDUCLAS1', 'HDUCLAS2', 'HDUCLAS3', 'HDUCLAS4',
              'HDUVERS', 'CHANTYPE', 'DETCHANS', 'EXTVER', 'FILTER', 'AREASCAL', 'BACKFILE', 'BACKSCAL', 'CORRFILE',
              'CORRSCAL', 'RESPFILE', 'ANCRFILE', 'SYS_ERR', 'POISSERR', 'GROUPING', 'DATE-OBS', 'DATE-END',
              'OBJECT', 'RA_OBJ', 'DEC_OBJ', 'EQUINOX', 'RADECSYS'):
        if k in spec_hdr:
            spec.header[k] = spec_hdr[k]
    spec.header['EXTNAME'] = 'SPECTRUM'
    tstart, tstop = float(data['TIME'][0]), float(data['ENDTIME'][-1])
    for hdr in (prim, spec.header, ebounds.header):
        hdr['TRIGTIME'] = (float(t0_met), 'Flare reference time T0 (MET)')
        hdr['TSTART'], hdr['TSTOP'] = tstart, tstop
    prim['FILENAME'] = os.path.basename(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fits.HDUList([fits.PrimaryHDU(header=prim), ebounds, spec]).writeto(out, overwrite=True)
    return out, tstart, tstop


def sun_rsp2_files(det, day, cache, log=print, window=None):
    """Sun-pointed RSP2 files of one detector and day: [(path, tstart, tstop)].

    window = (datetime0, datetime1): only files whose start (encoded in the name, bnYYMMDD_HHMM_FFF)
    falls within [window0 - 30 min, window1] are downloaded (each file covers ~10 min)."""
    url = RSP_URL.format(y=day.year, m=day.month, d=day.day)
    try:
        html = _listing(url)
    except Exception as e:
        log(f'  no Sun RSP directory for {day:%Y-%m-%d}: {e}')
        return []
    out = []
    for name in sorted(set(re.findall(r'glg_cspec_%s_bn\d{6}_\d{4}_\d{3}_v\d+\.rsp2' % det, html))):
        if window is not None:
            hhmm = re.search(r'bn\d{6}_(\d{2})(\d{2})_', name)
            ts = datetime.datetime(day.year, day.month, day.day, int(hhmm.group(1)), int(hhmm.group(2)))
            if not (window[0] - datetime.timedelta(minutes=30) <= ts <= window[1]):
                continue
        p = _get(url + name, os.path.join(cache, day.strftime('%Y%m%d'), name), log)
        with fits.open(p) as h:
            m = [x for x in h if x.name == 'SPECRESP MATRIX']
            out.append((p, float(m[0].header['TSTART']), float(m[-1].header['TSTOP'])))
    return out


def merge_rsp2(files, out, data_start, data_stop):
    """Concatenate the matrices of several RSP2 files (time-ordered) and extend the coverage of the
    first/last matrix to the data range. Returns (path, [(tstart, tstop, det_angle), ...])."""
    handles = [fits.open(f) for f in files]      # keep open: the matrices are variable-length columns
    try:
        prim, ebounds = handles[0][0].header.copy(), handles[0]['EBOUNDS']
        mats = sorted([x for h in handles for x in h if x.name == 'SPECRESP MATRIX'], key=lambda m: m.header['TSTART'])
        if not mats:
            raise ValueError('no response matrices')
        clean = [mats[0]]
        for m in mats[1:]:   # drop matrices overlapping a previous file
            if m.header['TSTART'] >= clean[-1].header['TSTOP'] - 1e-3:
                clean.append(m)
        hdrs = [m.header.copy() for m in clean]
        first, last = hdrs[0], hdrs[-1]
        if first['TSTART'] > data_start:
            half = 0.5 * (first['TSTART'] + first['TSTOP'])
            first['TSTART'] = data_start
            first['TSTOP'] = max(2 * half - data_start, first['TSTOP'])
        if 0.5 * (last['TSTART'] + last['TSTOP']) < data_stop:
            last['TSTOP'] = 2 * data_stop - last['TSTART']
        newm = []
        for i, (m, h) in enumerate(zip(clean, hdrs)):
            h['EXTVER'] = i + 1
            newm.append(fits.BinTableHDU(data=m.data, header=h))
        prim['DRM_NUM'] = len(newm)
        prim['FILENAME'] = os.path.basename(out)
        fits.HDUList([fits.PrimaryHDU(header=prim), ebounds] + newm).writeto(out, overwrite=True)
        table = [(float(h['TSTART']), float(h['TSTOP']), float(h.get('DET_ANG', np.nan))) for h in hdrs]
    finally:
        for h in handles:
            h.close()
    return out, table


def det_angle_at(table, met):
    """Detector-Sun angle from merge_rsp2's table at a given MET (nearest matrix)."""
    if not table:
        return float('nan')
    c = np.array([0.5 * (a + b) for a, b, _ in table])
    return float(table[int(np.argmin(abs(c - met)))][2])


def band_rate(pha, t0_met, lo_kev, hi_kev):
    """(t_rel, rate, exposure) of a CSPEC file summed over channels within [lo, hi] keV."""
    with fits.open(pha) as h:
        sp, eb = h['SPECTRUM'].data, h['EBOUNDS'].data
        band = (eb['E_MAX'] > lo_kev) & (eb['E_MIN'] < hi_kev)
        t = 0.5 * (sp['TIME'] + sp['ENDTIME']) - t0_met
        expo = sp['EXPOSURE'].astype(float)
        cnt = sp['COUNTS'][:, band].sum(1).astype(float)
        width = sp['ENDTIME'] - sp['TIME']
    return t, cnt, expo, width
