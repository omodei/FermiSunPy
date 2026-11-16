#!/usr/bin/env python
"""LAT localization of the flare emission relative to the Sun.

1. GtBurst likelihood (with gtfindsrc, optimizeposition=yes) in LOC_interval, E > LOC_EMIN_MEV.
2. A TS map (unbinned likelihood, test source moved on a grid; same approach as GtBurst FastTSMap)
   centred on the Sun, LOC_NSIDE x LOC_NSIDE points over +-LOC_HALF_SIZE deg.
3. Best position (TS-map maximum, refined with a quadratic fit), 68/95% radii from the TS contours,
   offset from the Sun centre in degrees and in helioprojective arcsec (sunpy).
With sun_centered = true the analysis runs on the Sun-centred FT1/FT2 (FermiSunPy suncenter), where the
Sun stays at (0, 0) during the whole interval.

Usage: python -m sunflare.localization flares/<id>/config_<id>.json
Output: results/localization/{localization.json, tsmap.png, tsmap.npz}
"""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from sunflare import common, lcdata, lat_data  # noqa: E402
from sunflare.sun import sun_radec, from_sun_frame, helioprojective_offset, R_SUN_DEG  # noqa: E402

cfg = common.load_config(sys.argv[1])
info, fdir = cfg.info, cfg.pwd
T0 = info['T0']
log = lambda *a: print(*a, flush=True)
if not info.get('has_lat'):
    sys.exit('No LAT data for this flare')
if cfg.sun_centered and not os.path.isdir(common.lat_dir(cfg)):
    lat_data.make_sun_centered(common.lat_dir(cfg, False), common.lat_dir(cfg, True), info['lat_name'], log)

# ------------------------------------------------------------------ interval
iv = cfg.LOC_interval or cfg.LAT_integrated_interval
if not iv:
    ev = lcdata.lat_events(fdir, info, roi=3.0, emin=cfg.LOC_EMIN_MEV, zmax=cfg.zmax, sun_centered=cfg.sun_centered)
    best, n = None, -1
    for w in info.get('lat_windows', []):
        k = int(((ev['t'] > w[0]) & (ev['t'] < w[1])).sum())
        if w[1] > 0 and k > n:
            best, n = w, k
    iv = best or [0.0, 3600.0]
a, b = float(iv[0]), float(iv[1])
tmid = T0 + 0.5 * (a + b)
RA, DEC = common.source_position(cfg)
if not cfg.sun_centered:
    RA, DEC = sun_radec(tmid)        # Sun at the middle of the interval
log(f'### STAGE: localization interval {a:g}-{b:g} s, centre RA, Dec = {RA:.3f}, {DEC:.3f}'
    f'{" (Sun-centred frame)" if cfg.sun_centered else ""}')

out = os.path.join(fdir, 'results', 'localization')
common.ensure_fresh(out, dict(iv=[a, b], e=cfg.LOC_EMIN_MEV, irfs=cfg.irfs, roi=cfg.roi, zmax=cfg.zmax,
                              sun=cfg.sun_centered, half=cfg.LOC_HALF_SIZE, n=cfg.LOC_NSIDE))
os.chdir(out)

from threeML import update_logging_level  # noqa: E402
update_logging_level('WARNING')
log('### STAGE: GtBurst likelihood + gtfindsrc')
bld = lat_data.builder(cfg, [a], [b], outfile='loc', source_model='PowerLaw2', emin=cfg.LOC_EMIN_MEV,
                       emax=cfg.LAT_EMAX_MEV, ra=RA, dec=DEC)
obs = bld.run(include_previous_intervals=True)
idir = lat_data.interval_dir(out, a, b)
if idir is None:
    sys.exit('GtBurst did not produce the interval directory')
os.chdir(idir)                       # the GtBurst XML models use paths relative to this directory
ft1_repo, ft2, rsp = lat_data.lat_files(common.lat_dir(cfg), info['lat_name'])
findsrc = {}
try:
    # doTimeResolvedLike never optimizes the position: run GtBurst's gtdolike with gtfindsrc here
    from GtBurst.commands import gtdolike
    xmlmodel = glob.glob('*_LAT_xmlmodel_*.xml')[0]
    r = gtdolike.run(spectralfiles='no', xmlmodel=xmlmodel, liketype='unbinned',
                     filteredeventfile=glob.glob('*_filt.fit')[0], rspfile=rsp, showmodelimage='no', tsmin=9.0,
                     optimizeposition='yes', ft2file=ft2, skymap=glob.glob('*_LAT_skymap_*.fit')[0],
                     flemin=cfg.FLUX_EMIN_MEV, flemax=cfg.FLUX_EMAX_MEV)
    findsrc = {'TS': float(r[3]), 'ra': float(r[5]), 'dec': float(r[7]), 'err_deg': float(r[9]),
               'distance_deg': float(r[11])}
    log(f'  gtfindsrc: RA, Dec = {findsrc["ra"]:.3f}, {findsrc["dec"]:.3f}  error = {findsrc["err_deg"]:.3f} deg  '
        f'TS = {findsrc["TS"]:.1f}')
except Exception as e:
    log(f'  gtfindsrc failed ({type(e).__name__}: {e}); using the TS map only')

# ------------------------------------------------------------------ TS map
log('### STAGE: TS map')
import UnbinnedAnalysis as UA  # noqa: E402
import pyLikelihood  # noqa: E402

root = None
for x in glob.glob(os.path.join(idir, '*_likeRes.xml')):
    root = x[:-len('_likeRes.xml')]
evfile = root + '.fit' if os.path.exists(root + '.fit') else glob.glob(root + '*.fit')[0]
ltcube = glob.glob(root + '_ltcube.fit*')[0]
expmap = glob.glob(root + '_expomap.fit*')[0]
like_obs = UA.UnbinnedObs(evfile, ft2, expMap=expmap, expCube=ltcube)
like = UA.UnbinnedAnalysis(like_obs, root + '_likeRes.xml', 'MINUIT')
target = 'GRB' if 'GRB' in like.sourceNames() else like.sourceNames()[0]
logL = like.logLike
tsrc = logL.deleteSource(target)
test = pyLikelihood.PointSource(RA, DEC, logL.observation())
test.setSpectrum(tsrc.spectrum())
test.setName('_test_source')
like.optimize(verbosity=0)
L0 = logL.value()
null = pyLikelihood.DoubleVector()
logL.getFreeParamValues(null)

n = int(cfg.LOC_NSIDE)
half = float(cfg.LOC_HALF_SIZE)
cosd = np.cos(np.radians(DEC))


def tsgrid(xv, yv, tag):
    """TS of a test point source at local CAR offsets (xv, yv) [deg] from (RA, DEC)."""
    T = np.zeros((len(yv), len(xv)))
    for j, dy in enumerate(yv):
        for i, dx in enumerate(xv):
            test.setDir(float((RA + dx / max(cosd, 1e-3)) % 360.0), float(DEC + dy), False, False)
            logL.addSource(test)
            like.optObject.find_min_only(0, 1e-5)
            T[j, i] = 2.0 * (logL.value() - L0)
            logL.deleteSource('_test_source')
            logL.setFreeParamValues(null)
        if (j + 1) % 5 == 0 or j == len(yv) - 1:
            log(f'  {tag} row {j + 1}/{len(yv)}  max TS so far {T[:j + 1].max():.1f}')
    return T


def quad_fit(xv, yv, T):
    """Best position and 68/95% radii (2 dof) from a paraboloid fitted around the TS maximum."""
    X_, Y_ = np.meshgrid(xv, yv)
    # points within dTS = 11.8 of the maximum, or at least the 15 highest ones for very sharp peaks
    thr = max(11.8, T.max() - np.sort(T.ravel())[::-1][min(14, T.size - 1)])
    sel = T >= T.max() - thr
    if sel.sum() < 6:
        raise ValueError('too few points around the maximum')
    A = np.column_stack([X_[sel] ** 2, Y_[sel] ** 2, X_[sel] * Y_[sel], X_[sel], Y_[sel], np.ones(sel.sum())])
    c = np.linalg.lstsq(A, T[sel], rcond=None)[0]
    M = np.array([[2 * c[0], c[2]], [c[2], 2 * c[1]]])
    if np.any(np.linalg.eigvalsh(M) >= 0):
        raise ValueError('TS surface is not peaked')
    x0, y0 = np.linalg.solve(M, [-c[3], -c[4]])
    if not (xv[0] <= x0 <= xv[-1] and yv[0] <= y0 <= yv[-1]):
        raise ValueError('maximum outside the grid')
    det = np.linalg.det(np.linalg.inv(-M / 2.))
    return float(x0), float(y0), float(np.sqrt(2.30 * np.sqrt(det))), float(np.sqrt(5.99 * np.sqrt(det)))


xs = np.linspace(-half, half, n)
TS = tsgrid(xs, xs, 'coarse map')
X, Y = np.meshgrid(xs, xs)
jm, im = np.unravel_index(np.argmax(TS), TS.shape)
step = xs[1] - xs[0]
# fine map around the coarse maximum (bright flares are localized much better than the coarse pixels)
fx = np.linspace(xs[im] - 2 * step, xs[im] + 2 * step, 15)
fy = np.linspace(xs[jm] - 2 * step, xs[jm] + 2 * step, 15)
TSf = tsgrid(fx, fy, 'fine map')
np.savez(os.path.join(out, 'tsmap.npz'), x=xs, y=xs, ts=TS, fx=fx, fy=fy, tsf=TSf, ra0=RA, dec0=DEC)
try:
    bx, by, r68, r95 = quad_fit(fx, fy, TSf)
    method = 'fine-map paraboloid'
except Exception as e1:
    try:
        bx, by, r68, r95 = quad_fit(xs, xs, TS)
        method = 'coarse-map paraboloid'
    except Exception as e2:
        jf, i_f = np.unravel_index(np.argmax(TSf), TSf.shape)
        bx, by = fx[i_f], fy[jf]
        pix = (fx[1] - fx[0]) * (fy[1] - fy[0])
        r68 = float(np.sqrt(max((TSf > TSf.max() - 2.30).sum(), 1) * pix / np.pi))
        r95 = float(np.sqrt(max((TSf > TSf.max() - 5.99).sum(), 1) * pix / np.pi))
        method = f'pixel counting ({e1}; {e2})'
log(f'  position from the {method}')
TS_max = float(max(TS.max(), TSf.max()))
best_ra, best_dec = (RA + bx / max(cosd, 1e-3)) % 360.0, DEC + by
if cfg.sun_centered:
    eq_ra, eq_dec = from_sun_frame(best_ra if best_ra < 180 else best_ra - 360, best_dec, tmid)
    eq_ra, eq_dec = float(eq_ra), float(eq_dec)
else:
    eq_ra, eq_dec = float(best_ra), float(best_dec)
s_ra, s_dec = sun_radec(tmid)
from astropy.coordinates import SkyCoord  # noqa: E402
sep = float(SkyCoord(s_ra, s_dec, unit='deg').separation(SkyCoord(eq_ra, eq_dec, unit='deg')).deg)
tx, ty, rr, rsun_arcsec = helioprojective_offset(eq_ra, eq_dec, tmid)
result = {'interval': [a, b], 'emin_MeV': cfg.LOC_EMIN_MEV, 'TS_max': TS_max, 'method': method,
          'best_frame': [float(best_ra), float(best_dec)], 'frame': 'sun-centred' if cfg.sun_centered else 'equatorial',
          'best_radec': [eq_ra, eq_dec], 'sun_radec': [s_ra, s_dec], 'r68_deg': r68, 'r95_deg': r95,
          'offset_deg': sep, 'offset_sigma': sep / (r68 / 1.515) if r68 > 0 else None,
          'hpc_arcsec': [tx, ty], 'offset_rsun': rr, 'rsun_arcsec': rsun_arcsec,
          'consistent_with_disk': bool(sep - r95 <= R_SUN_DEG), 'gtfindsrc': findsrc,
          'grid': {'half_size': half, 'n': n}}
os.chdir(out)
common.save_json('localization.json', result)
log(f'  best position RA, Dec = {eq_ra:.3f}, {eq_dec:.3f}  r68 = {r68:.3f} deg  r95 = {r95:.3f} deg  '
    f'offset from the Sun centre = {sep:.3f} deg ({tx:.0f}", {ty:.0f}")  TS = {TS_max:.1f}')

# ------------------------------------------------------------------ plots
fig, axs = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw=dict(wspace=0.35))
ax = axs[0]
im_ = ax.pcolormesh(xs, xs, np.maximum(TS, 0), cmap='magma', shading='nearest')
ax.contour(fx, fy, TSf, levels=[TS_max - 9.21, TS_max - 5.99, TS_max - 2.30], colors=['w', 'c', 'lime'],
           linewidths=1)
ax.add_patch(plt.Circle((0, 0), R_SUN_DEG, fill=False, color='gold', lw=2))
if 'ra' in findsrc and not cfg.sun_centered:
    gx = ((findsrc['ra'] - RA + 180) % 360 - 180) * cosd
    ax.plot(gx, findsrc['dec'] - DEC, '+', color='w', ms=12, mew=2, label='gtfindsrc')
    ax.legend(loc='lower left', fontsize=9)
ax.plot(bx, by, 'x', color='cyan', ms=10, mew=2)
ax.add_patch(plt.Circle((bx, by), r68, fill=False, color='cyan', ls='--'))
ax.set_aspect('equal')
if not cfg.sun_centered:
    ax.invert_xaxis()            # east to the left
plt.colorbar(im_, ax=ax, label='TS')
if cfg.sun_centered:
    ax.set_xlabel('ecliptic longitude − Sun [deg]'); ax.set_ylabel('ecliptic latitude [deg]')
else:
    ax.set_xlabel(r'$\Delta$RA cos$\delta$ [deg] (east left)'); ax.set_ylabel(r'$\Delta$Dec [deg]')
ax.set_title(f'TS map {a:g}–{b:g} s, E > {cfg.LOC_EMIN_MEV:g} MeV (contours 68/95/99%)')
# helioprojective view
ax = axs[1]
try:
    from sunpy.coordinates import frames, get_earth
    from astropy.time import Time
    import astropy.units as u
    t = Time('2001-01-01T00:00:00', scale='utc') + tmid * u.s
    if cfg.sun_centered:
        gra, gdec = from_sun_frame(np.where(X > 180, X - 360, X), Y, tmid)
    else:
        gra, gdec = (RA + X / max(cosd, 1e-3)) % 360.0, DEC + Y
    hpc = SkyCoord(gra * u.deg, gdec * u.deg, distance=1 * u.AU, frame='gcrs', obstime=t).transform_to(
        frames.Helioprojective(observer=get_earth(t), obstime=t))
    Tx, Ty = hpc.Tx.to_value(u.arcsec), hpc.Ty.to_value(u.arcsec)
    ax.contourf(Tx, Ty, np.maximum(TS, 0), levels=30, cmap='magma')
    FX, FY = np.meshgrid(fx, fy)
    if cfg.sun_centered:
        fra, fdec = from_sun_frame(np.where(FX > 180, FX - 360, FX), FY, tmid)
    else:
        fra, fdec = (RA + FX / max(cosd, 1e-3)) % 360.0, DEC + FY
    fh = SkyCoord(fra * u.deg, fdec * u.deg, distance=1 * u.AU, frame='gcrs', obstime=t).transform_to(
        frames.Helioprojective(observer=get_earth(t), obstime=t))
    ax.contour(fh.Tx.to_value(u.arcsec), fh.Ty.to_value(u.arcsec), TSf,
               levels=[TS_max - 9.21, TS_max - 5.99, TS_max - 2.30], colors=['w', 'c', 'lime'])
    ax.add_patch(plt.Circle((0, 0), rsun_arcsec, fill=False, color='gold', lw=2))
    ax.plot(tx, ty, 'x', color='cyan', ms=10, mew=2, label=f'LAT best ({tx:.0f}", {ty:.0f}")')
    ax.add_patch(plt.Circle((tx, ty), r68 * 3600, fill=False, color='cyan', ls='--', label=f'r68 = {r68 * 60:.1f}′'))
    if info.get('location'):
        ax.set_title(f'Helioprojective (solar north up); flare at {info["location"]}')
    else:
        ax.set_title('Helioprojective (solar north up)')
    lim = min(half * 3600, max(2.5 * rsun_arcsec, abs(tx) + 3 * r68 * 3600, abs(ty) + 3 * r68 * 3600))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.set_xlabel('Solar X [arcsec]'); ax.set_ylabel('Solar Y [arcsec]'); ax.legend(loc='lower left', fontsize=9)
    ax.grid(alpha=0.3)
except Exception as e:
    ax.text(0.5, 0.5, f'helioprojective view unavailable:\n{e}', ha='center', transform=ax.transAxes)
fig.suptitle(f'{cfg.name} — LAT localization: offset {sep:.2f}° from the Sun centre, r68 = {r68:.2f}°, TS = {TS_max:.0f}')
fig.savefig('tsmap.png', dpi=110, bbox_inches='tight')
plt.close(fig)
log('### DONE localization')
