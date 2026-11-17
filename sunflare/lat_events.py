"""LAT events with the gtsrcprob probability of coming from the Sun (same table as GRBAnalyzer's report).

Reads the *_filt_prob.fit file of the LAT time-integrated interval (results/lat/interval<t0>-<t1>/) and writes
results/lat/lat_events.json: one row per event with P(Sun), P(Galactic), P(isotropic), the distance from the
Sun centre (tracking its motion, or from (0, 0) in the Sun-centred frame), the 68% PSF radius from the IRFs,
off-axis and zenith angles, conversion type, best event class, run/event ID and the time-resolved bin.

Usage: python -m sunflare.lat_events flares/<id>/config_<id>.json
"""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np  # noqa: E402

from sunflare import common  # noqa: E402

# P8R3 event-class bits, best (most restrictive) first
CLASS_BITS = [(10, 'ULTRACLEANVETO'), (9, 'ULTRACLEAN'), (8, 'CLEAN'), (7, 'SOURCE'), (6, 'TRANSIENT010'),
              (5, 'TRANSIENT010E'), (4, 'TRANSIENT020'), (3, 'TRANSIENT020E'), (2, 'TRANSIENT100'), (1, 'TRANSIENT100E')]
FIELDS = ('pSun', 'pGal', 'pIso', 'sep', 'r68', 'theta', 'zen', 'conv', 'cls', 'run', 'id', 'bin')


def best_class(bits):
    """EVENT_CLASS is a 32-element boolean array, element i = bit 31-i."""
    for b, name in CLASS_BITS:
        if bits[31 - b]:
            return name
    return ''


class PSF68:
    """68% containment radius (deg) from the Fermitools IRFs, per conversion type."""

    def __init__(self, irf_name):
        self.psf = {}
        try:
            import pyIrfLoader
            (pyIrfLoader.Loader_go if hasattr(pyIrfLoader, 'Loader_go') else pyIrfLoader.Loader.go)()
            fac = pyIrfLoader.IrfsFactory.instance()
            for conv, tag in ((0, 'FRONT'), (1, 'BACK')):
                self.psf[conv] = fac.create(f'{irf_name}::{tag}').psf()
        except Exception as e:
            print(f'  PSF from IRFs not available ({e}); no containment radii', flush=True)

    def __call__(self, energy, theta, phi, conv):
        p = self.psf.get(int(conv))
        if p is None:
            return None
        try:
            return float(p.angularContainment(float(energy), float(theta), float(phi), 0.68))
        except Exception:
            return None


def extract(cfg, log=print):
    from astropy.io import fits
    from astropy.coordinates import SkyCoord
    from sunflare import lat_data
    out = os.path.join(cfg.pwd, 'results', 'lat')
    summ = common.load_json(os.path.join(out, 'lat_summary.json'))
    if not summ:
        return []
    integ = summ['integrated']
    d = lat_data.interval_dir(out, integ['t0'], integ['t1'])
    files = glob.glob(os.path.join(d, '*_filt_prob.fit*')) if d else []
    if not files:
        log('  no *_filt_prob.fit file for the time-integrated interval')
        return []
    T0 = cfg.info['T0']
    ev = fits.getdata(files[0], 'EVENTS')
    if len(ev) == 0:
        common.save_json(os.path.join(out, 'lat_events.json'), [])
        return []
    names = ev.columns.names
    psrc = 'GRB' if 'GRB' in names else None       # GtBurst names the target source GRB
    if summ.get('sun_centered'):
        sun = SkyCoord(0.0, 0.0, unit='deg')
    else:
        from sunflare.sun import sun_radec
        sun = SkyCoord(*sun_radec(ev['TIME']), unit='deg')
    sep = sun.separation(SkyCoord(ev['RA'], ev['DEC'], unit='deg')).deg
    try:
        from GtBurst.IRFS import IRFS
        psf68 = PSF68(IRFS[cfg.irfs].name)
    except Exception:
        psf68 = lambda *a: None  # noqa: E731
    bins = [(i + 1, r['t0'], r['t1']) for i, r in enumerate(summ.get('resolved', []))]
    rows = []
    for k in range(len(ev)):
        r = ev[k]
        t = float(r['TIME'] - T0)
        b = next((str(n) for n, a, z in bins if a <= t < z), '')
        r68 = psf68(r['ENERGY'], r['THETA'], r['PHI'] if 'PHI' in names else 0.0, r['CONVERSION_TYPE'])
        rows.append({
            't': round(t, 3), 'utc': common.met2utc(r['TIME'])[11:19], 'E': round(float(r['ENERGY']), 2),
            'pSun': round(float(r[psrc]), 5) if psrc else None,
            'pGal': round(float(r['GalacticTemplate']), 5) if 'GalacticTemplate' in names else None,
            'pIso': round(float(r['IsotropicTemplate']), 5) if 'IsotropicTemplate' in names else None,
            'sep': round(float(sep[k]), 3), 'r68': None if r68 is None else round(r68, 3),
            'theta': round(float(r['THETA']), 2), 'zen': round(float(r['ZENITH_ANGLE']), 2),
            'conv': 'front' if int(r['CONVERSION_TYPE']) == 0 else 'back', 'cls': best_class(r['EVENT_CLASS']),
            'ra': round(float(r['RA']), 3), 'dec': round(float(r['DEC']), 3),
            'run': int(r['RUN_ID']), 'id': int(r['EVENT_ID']), 'bin': b})
    rows.sort(key=lambda x: x['t'])
    common.save_json(os.path.join(out, 'lat_events.json'), rows)
    n90 = sum(1 for x in rows if (x['pSun'] or 0) >= 0.9)
    log(f'  {len(rows)} LAT events in {integ["t0"]:g}-{integ["t1"]:g} s, {n90} with P(Sun) >= 0.9')
    return rows


def load_or_extract(cfg, log=print):
    """Events for the report; rebuilds files written by older versions (fewer columns)."""
    rows = common.load_json(os.path.join(cfg.pwd, 'results', 'lat', 'lat_events.json'))
    if rows and all(f in rows[0] for f in FIELDS):
        return rows
    return extract(cfg, log)


if __name__ == '__main__':
    extract(common.load_config(sys.argv[1]))
