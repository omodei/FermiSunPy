"""Fermi-LAT data for solar flares: download, Sun-centred copies, visibility and GtBurst builders."""
import glob
import os
import shutil
import sys

import numpy as np
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from sunflare import common  # noqa: E402


def lat_files(repo, lat_name):
    d = os.path.join(repo, 'bn' + lat_name)
    return (os.path.join(d, f'gll_ft1_tr_bn{lat_name}_v00.fit'), os.path.join(d, f'gll_ft2_tr_bn{lat_name}_v00.fit'),
            os.path.join(d, f'gll_cspec_tr_bn{lat_name}_v00.rsp'))


def download(sun_ra, sun_dec, t0, tmin, tmax, repo, radius=15.0, emin=30.0, emax=300000.0):
    """FT1/FT2 around the Sun (position at T0) with threeML's LAT_dataset. Returns the GtBurst name."""
    from threeML.utils.data_download.Fermi_LAT.download_LAT_data import LAT_dataset
    ds = LAT_dataset()
    ds.make_LAT_dataset(ra=sun_ra, dec=sun_dec, radius=radius, trigger_time=t0, tstart=tmin, tstop=tmax,
                        data_type='Extended', destination_directory=repo, Emin=emin, Emax=emax)
    return ds.grb_name


def import_local(ft1, ft2, sun_ra, sun_dec, t0, tmin, tmax, repo):
    """Use FT1/FT2 files already on disk (e.g. weekly files or a previous download) instead of the LAT server.
    They are copied with the GtBurst naming and the GtBurst helper files are created. Returns the name."""
    from GtBurst.dataHandling import _makeDatasetsOutOfLATdata, met2date
    name = met2date(t0, opt='grbname')
    d = os.path.join(repo, 'bn' + name)
    os.makedirs(d, exist_ok=True)
    n1, n2, _ = lat_files(repo, name)
    shutil.copy(ft1, n1)
    shutil.copy(ft2, n2)
    h = fits.getheader(n1, 0)
    tmin = max(tmin, h['TSTART'] - t0)
    tmax = min(tmax, h['TSTOP'] - t0)
    _makeDatasetsOutOfLATdata(n1, n2, name, tmin, tmax, sun_ra, sun_dec, t0, d, cspecstart=tmin, cspecstop=tmax)
    return name


def make_sun_centered(src_repo, dst_repo, lat_name, log=print):
    """Copy bn<lat_name>/ into dst_repo with FT1 and FT2 converted to the Sun-centred frame (FermiSunPy)."""
    import suncenter
    ft1, ft2, rsp = lat_files(src_repo, lat_name)
    d1, d2, drsp = lat_files(dst_repo, lat_name)
    os.makedirs(os.path.dirname(d1), exist_ok=True)
    stamp = os.path.join(os.path.dirname(d1), 'converted_from.txt')
    sig = f'{os.path.getmtime(ft1)} {os.path.getmtime(ft2)}'
    if os.path.exists(stamp) and open(stamp).read() == sig and os.path.exists(d1) and os.path.exists(d2):
        return d1, d2
    log('  converting FT1/FT2 to the Sun-centred frame (FermiSunPy suncenter)')
    suncenter.convertFT1_fast(ft1, d1, wrap360=True)
    suncenter.convertFT2_fast(ft2, d2, wrap360=True)
    for f in glob.glob(os.path.join(os.path.dirname(ft1), '*')):   # rsp / cspec helper files
        if f not in (ft1, ft2) and not os.path.exists(os.path.join(os.path.dirname(d1), os.path.basename(f))):
            shutil.copy(f, os.path.dirname(d1))
    with fits.open(d1, mode='update') as h:
        # GtBurst reads the ROI centre from the DSS keywords: make them point at the Sun (0, 0)
        for hdr in (h['EVENTS'].header,):
            for k in list(hdr.keys()):
                if k.startswith('DSVAL') and str(hdr[k]).startswith('CIRCLE('):
                    r = str(hdr[k]).split(',')[-1].rstrip(')')
                    hdr[k] = f'CIRCLE(0,0,{r})'
        h.flush()
    open(stamp, 'w').write(sig)
    return d1, d2


def visibility(ft2, t0, ra, dec, theta_max=70.0, zmax=100.0, min_len=60.0, sun_track=True):
    """Sun off-axis angle and zenith angle along the orbit, and the windows in which the Sun is observable.

    If sun_track is True the Sun position is recomputed at each FT2 time (it moves ~1 deg/day)."""
    from astropy.coordinates import SkyCoord
    with fits.open(ft2) as h:
        p = h['SC_DATA'].data
        t = 0.5 * (p['START'] + p['STOP'])
        if sun_track:
            from sunflare.sun import sun_radec
            sra, sdec = sun_radec(t)
            src = SkyCoord(sra, sdec, unit='deg')
        else:
            src = SkyCoord(ra, dec, unit='deg')
        theta = src.separation(SkyCoord(p['RA_SCZ'], p['DEC_SCZ'], unit='deg')).deg
        zen = src.separation(SkyCoord(p['RA_ZENITH'], p['DEC_ZENITH'], unit='deg')).deg
        saa = p['IN_SAA'].astype(bool) if 'IN_SAA' in p.names else np.zeros(len(t), bool)
        live = p['LIVETIME'] / np.maximum(p['STOP'] - p['START'], 1e-6)
    ok = (theta < theta_max) & (zen < zmax) & ~saa
    trel = t - t0
    wins, start = [], None
    for i, flag in enumerate(ok):
        if flag and start is None:
            start = trel[i]
        if (not flag or i == len(ok) - 1) and start is not None:
            if trel[i] - start >= min_len:
                wins.append([round(float(start), 1), round(float(trel[i]), 1)])
            start = None
    step = max(1, len(t) // 4000)
    track = {'t': np.round(trel[::step], 1).tolist(), 'theta': np.round(theta[::step], 2).tolist(),
             'zenith': np.round(zen[::step], 2).tolist(), 'saa': saa[::step].astype(int).tolist(),
             'livefrac': np.round(live[::step], 3).tolist()}
    return wins, track


def events(ft1, t0, center, roi, emin=100.0, zmax=100.0, sun_track=False):
    """Events within `roi` deg of `center`=(ra, dec) (or of the moving Sun when sun_track)."""
    from astropy.coordinates import SkyCoord
    with fits.open(ft1) as h:
        e = h['EVENTS'].data
        t = e['TIME']
        if sun_track:
            from sunflare.sun import sun_radec
            sra, sdec = sun_radec(t)
            c = SkyCoord(sra, sdec, unit='deg')
        else:
            c = SkyCoord(*center, unit='deg')
        sep = c.separation(SkyCoord(e['RA'], e['DEC'], unit='deg')).deg
        m = (sep < roi) & (e['ENERGY'] >= emin) & (e['ZENITH_ANGLE'] < zmax)
        cls = e['EVENT_CLASS'][m] if 'EVENT_CLASS' in e.names else None
        out = {'t': t[m] - t0, 'E': e['ENERGY'][m], 'sep': sep[m], 'ra': e['RA'][m], 'dec': e['DEC'][m]}
        if cls is not None:
            out['evclass'] = cls
    return out


def psf68(E):
    """Rough P8R3 SOURCE 68% PSF radius (deg) vs energy (MeV), front+back."""
    E = np.asarray(E, dtype=float)
    return np.sqrt((3.5 * (E / 100.) ** -0.8) ** 2 + 0.1 ** 2)


def builder(cfg, tstarts, tstops, outfile, source_model='PowerLaw2', emin=None, emax=None, **extra):
    """TransientLATDataBuilder for the flare (Sun position or Sun-centred frame)."""
    from threeML import TransientLATDataBuilder
    ra, dec = common.source_position(cfg)
    fmt = lambda xs: ','.join(f'{float(x):.3f}' for x in xs).replace('-', '\\-')
    kw = dict(outfile=outfile, roi=cfg.roi, tstarts=fmt(tstarts), tstops=fmt(tstops), irf=cfg.irfs, zmax=cfg.zmax,
              source_model=source_model, galactic_model='none' if cfg.sun_centered else 'template',
              particle_model='isotr template', datarepository=common.lat_dir(cfg), flemin=cfg.FLUX_EMIN_MEV,
              flemax=cfg.FLUX_EMAX_MEV, emin=emin or cfg.LAT_EMIN_MEV, emax=emax or cfg.LAT_EMAX_MEV,
              strategy=cfg.strategy, thetamax=cfg.thetamax, ra=ra, dec=dec)
    kw.update(extra)
    return TransientLATDataBuilder(cfg.info['lat_name'], **kw)


def interval_dir(base, t0, t1):
    """GtBurst writes each interval in interval<t0>-<t1>/ (number format varies): match numerically."""
    import re
    for d in glob.glob(os.path.join(base, 'interval*')):
        m = re.match(r'interval(-?[\d.]+)-(-?[\d.]+)$', os.path.basename(d))
        if m and abs(float(m.group(1)) - float(t0)) < 1e-2 and abs(float(m.group(2)) - float(t1)) < 1e-2:
            return d
    return None
