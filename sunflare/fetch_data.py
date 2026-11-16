#!/usr/bin/env python
"""Download GOES, GBM (daily CSPEC + Sun-pointed RSP2), LLE and LAT data for a flare and write flare_info.json.

Usage:
  python -m sunflare.fetch_data --t0 "2025-11-11 09:49:00" [--tend "2025-11-11 10:17:00"] [--peak ...]
         [--goes-class X5.1] [--tmin -3600 --tmax 14400] [--sun-centered] [--skip gbm,lat,...]
The flare directory is flares/SOLYYYYMMDD_HHMM (from T0). Times in the flare files are relative to T0.
"""
import argparse
import datetime
import glob
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np  # noqa: E402
from sunflare import common  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--t0', required=True, help='flare reference time (UTC), e.g. the GOES start')
ap.add_argument('--tend', default=None, help='flare end (UTC), e.g. the GOES end')
ap.add_argument('--peak', default=None)
ap.add_argument('--goes-class', default=None)
ap.add_argument('--location', default=None)
ap.add_argument('--ar', default=None)
ap.add_argument('--tmin', type=float, default=common.DEFAULTS['DWNL_TMIN'])
ap.add_argument('--tmax', type=float, default=common.DEFAULTS['DWNL_TMAX'])
ap.add_argument('--sun-centered', action='store_true')
ap.add_argument('--skip', default='', help='comma list of goes,gbm,lle,lat to skip')
ap.add_argument('--lat-radius', type=float, default=15.0)
ap.add_argument('--lat-ft1', default=None, help='use this FT1 file instead of querying the LAT data server')
ap.add_argument('--lat-ft2', default=None, help='FT2 file that goes with --lat-ft1')
ap.add_argument('--lle-dir', default=None, help='directory with gll_lle/gll_pt/gll_cspec(.rsp) files of a LLE trigger')
args = ap.parse_args()
skip = set(s for s in args.skip.split(',') if s)

t0_utc = datetime.datetime.fromisoformat(args.t0.replace('T', ' ').replace('Z', ''))
fid = common.flare_id(t0_utc)
d = common.flare_dir(fid)
os.makedirs(d, exist_ok=True)
os.chdir(d)
CACHE = os.path.join(common.FLARES, '_cache')
info = common.load_json(common.info_path(fid), {}) or {}
info.update({'name': fid, 'warnings': [], 't0_utc': t0_utc.strftime('%Y-%m-%d %H:%M:%S'),
             'T0': common.utc2met(t0_utc), 'tmin': args.tmin, 'tmax': args.tmax})
T0 = info['T0']
rel = lambda utc: common.utc2met(utc) - T0 if utc else None
info['tend'] = rel(args.tend) if args.tend else info.get('tend')
info['tpeak'] = rel(args.peak) if args.peak else info.get('tpeak')
for k, v in (('goes_class', args.goes_class), ('location', args.location), ('ar', args.ar)):
    if v:
        info[k] = v
win_dt = (t0_utc + datetime.timedelta(seconds=args.tmin), t0_utc + datetime.timedelta(seconds=args.tmax))
log = lambda *a: print(*a, flush=True)


def save():
    common.save_json(common.info_path(fid), info)


def warn(msg):
    log('  WARNING:', msg)
    info['warnings'].append(msg)


# ------------------------------------------------------------------ Sun
log(f'### STAGE: sun  ({fid}, T0 = {info["t0_utc"]} UTC = MET {T0:.3f})')
from sunflare.sun import sun_radec  # noqa: E402
info['sun_ra'], info['sun_dec'] = sun_radec(T0)
ra1, dec1 = sun_radec(T0 + args.tmax)
from astropy.coordinates import SkyCoord  # noqa: E402
info['sun_motion_deg'] = float(SkyCoord(info['sun_ra'], info['sun_dec'], unit='deg').separation(
    SkyCoord(ra1, dec1, unit='deg')).deg)
log(f'  Sun at RA, Dec = {info["sun_ra"]:.3f}, {info["sun_dec"]:.3f}  (moves {info["sun_motion_deg"]:.2f} deg in the window)')
save()

# ------------------------------------------------------------------ GOES
if 'goes' not in skip:
    log('### STAGE: goes  (XRS via sunpy Fido)')
    try:
        from sunpy.net import Fido, attrs as a
        import sunpy.timeseries as ts
        res = None
        for resol in ('flx1s', 'avg1m', None):
            q = [a.Time(win_dt[0], win_dt[1]), a.Instrument('XRS')]
            if resol:
                q.append(a.Resolution(resol))
            res = Fido.search(*q)
            if len(res) and len(res[0]):
                break
        tab = res[0]
        # prefer the most recent (primary) satellite
        sats = sorted(set(int(s) for s in tab['SatelliteNumber']))
        tab = tab[np.array([int(s) == sats[-1] for s in tab['SatelliteNumber']])]
        files = Fido.fetch(tab, path=os.path.join(CACHE, 'goes', '{file}'), progress=False)
        if not files:
            raise RuntimeError('GOES download returned no file')
        series = ts.TimeSeries(sorted(files), concatenate=True)
        df = series.to_dataframe()
        df = df[(df.index >= win_dt[0]) & (df.index <= win_dt[1])]
        for ch in ('xrsa', 'xrsb'):
            if ch + '_quality' in df:
                df.loc[df[ch + '_quality'] != 0, ch] = np.nan
        from astropy.time import Time
        trel = (Time(df.index.to_pydatetime(), scale='utc') - Time(t0_utc, scale='utc')).sec
        out = np.column_stack([trel, df['xrsa'].values, df['xrsb'].values])
        np.savetxt('goes_xrs.csv', out, delimiter=',', header='t_rel_s,xrsa_Wm2,xrsb_Wm2', comments='', fmt='%.3f,%.4e,%.4e')
        info['goes_satellite'] = f'GOES-{sats[-1]}'
        b = df['xrsb'].values
        if np.isfinite(b).any():
            i = int(np.nanargmax(b))
            pk = b[i]
            cls = 'ABCMX'[min(4, max(0, int(np.floor(np.log10(pk))) + 8))]
            info['goes_peak_flux'] = float(pk)
            info['goes_peak_t'] = float(trel[i])
            info.setdefault('goes_class', f'{cls}{pk / 10 ** np.floor(np.log10(pk)):.1f}')
            if info.get('tpeak') is None:
                info['tpeak'] = float(trel[i])
        log(f'  {info["goes_satellite"]}: {len(df)} samples, peak {info.get("goes_peak_flux", 0):.2e} W/m2 '
            f'(class {info.get("goes_class")}) at T0{info.get("goes_peak_t", 0):+.0f} s')
        info['has_goes'] = True
    except Exception as e:
        info['has_goes'] = False
        warn(f'No GOES data: {type(e).__name__}: {e}')
    save()

# ------------------------------------------------------------------ GBM triggers in the window
log('### STAGE: triggers  (GBM triggers in the window)')
try:
    from sunflare.search import gbm_triggers
    trig = []
    for day in {win_dt[0].date(), win_dt[1].date()}:
        trig += gbm_triggers(day)
    info['gbm_triggers'] = [dict(x, t=rel(x['utc'])) for x in trig
                            if args.tmin <= rel(x['utc']) <= args.tmax]
    for x in info['gbm_triggers']:
        log(f'  {x["name"]} at T0{x["t"]:+.0f} s')
except Exception as e:
    info['gbm_triggers'] = []
    warn(f'GBM trigger list unavailable: {e}')
save()

# ------------------------------------------------------------------ GBM
if 'gbm' not in skip:
    log('### STAGE: gbm  (daily CSPEC for 14 detectors + Sun-pointed RSP2)')
    from sunflare import gbm_daily
    os.makedirs('GBM_data', exist_ok=True)
    table = []
    days = list(gbm_daily.days_between(*win_dt))
    tstart = tstop = None
    for det in common.GBM_DETS:
        try:
            daily = [gbm_daily.daily_cspec(det, day, os.path.join(CACHE, 'gbm_daily'), log) for day in days]
            pha, tstart, tstop = gbm_daily.write_flare_cspec(daily, f'GBM_data/glg_cspec_{det}_{fid}.pha',
                                                             T0, args.tmin, args.tmax)
        except Exception as e:
            warn(f'GBM {det}: {e}')
            continue
        row = {'det': det, 'pha': os.path.basename(pha), 'angle': float('nan'), 'rsp': None}
        try:
            rsps = []
            for day in days:
                rsps += gbm_daily.sun_rsp2_files(det, day, os.path.join(CACHE, 'gbm_rsp'), log, window=win_dt)
            rsps = [r for r in rsps if r[2] > tstart and r[1] < tstop]
            if rsps:
                out, tab = gbm_daily.merge_rsp2([r[0] for r in rsps], f'GBM_data/glg_cspec_{det}_{fid}.rsp2', tstart, tstop)
                row['rsp'] = os.path.basename(out)
                row['rsp_files'] = [os.path.basename(r[0]) for r in rsps]
                row['rsp_coverage'] = [[round(r[1] - T0, 1), round(r[2] - T0, 1)] for r in rsps]
                tref = T0 + (info.get('tpeak') or 0.0)
                row['angle'] = gbm_daily.det_angle_at(tab, tref)
        except Exception as e:
            warn(f'Sun RSP2 for {det}: {e}')
        # signal-to-noise of the flare over a pre-flare background (25-100 keV NaI, 0.3-1 MeV BGO)
        lo, hi = (25, 100) if det[0] == 'n' else (300, 1000)
        t, cnt, expo, _ = gbm_daily.band_rate(pha, T0, lo, hi)
        rate = cnt / np.maximum(expo, 1e-9)
        tend = info.get('tend') or 1800.0
        s = (t > 0) & (t < tend)
        b = (t > args.tmin) & (t < -60)
        try:
            bk = np.median(rate[b]) if b.sum() > 10 else np.percentile(rate[s], 5)
            net = ((rate[s] - bk) * expo[s]).sum()
            row['sn'] = float(net / np.sqrt(max((rate[s] * expo[s]).sum(), 1)))
        except Exception:
            row['sn'] = float('nan')
        log(f'  {det}: Sun angle {row["angle"]:6.1f} deg   S/N {row["sn"]:8.1f}   rsp: {row["rsp"]}')
        table.append(row)
    info['detectors'] = table
    info['has_gbm'] = bool(table)
    if table:
        info['gbm_range'] = [tstart - T0, tstop - T0]
        nai = [x for x in table if x['det'][0] == 'n' and x['rsp']]
        bgo = [x for x in table if x['det'][0] == 'b' and x['rsp']]
        key = lambda x: (x['angle'] if np.isfinite(x['angle']) else 999)
        sug = [x['det'] for x in sorted(nai, key=key)[:2]] + [x['det'] for x in sorted(bgo, key=key)[:1]]
        info['suggested_detectors'] = sug or ['n0', 'b0']
        if not nai and not bgo:
            warn('No Sun-pointed GBM response covers this window: GBM spectral fits are not possible')
    save()

# ------------------------------------------------------------------ LLE
if 'lle' not in skip:
    log('### STAGE: lle  (LAT Low Energy data of the GBM triggers, if any)')
    info['has_lle'] = False
    if args.lle_dir:
        import shutil
        os.makedirs('LLE_data', exist_ok=True)
        for f in glob.glob(os.path.join(args.lle_dir, 'gll_lle_bn*.fit')) + glob.glob(os.path.join(args.lle_dir, 'gll_pt_bn*.fit')) + \
                glob.glob(os.path.join(args.lle_dir, 'gll_cspec_bn*.rsp')):
            shutil.copy(f, 'LLE_data')
        lle = common.find_version('LLE_data/gll_lle_bn*_v*.fit')
        if lle:
            info['gbm_triggers'] = [{'name': os.path.basename(lle).split('_')[2], 't': None}] + info.get('gbm_triggers', [])
    for trg in info.get('gbm_triggers', []):
        try:
            from threeML import download_LLE_trigger_data
            if not glob.glob(f'LLE_data/gll_lle_{trg["name"]}_v*.fit'):
                download_LLE_trigger_data(trg['name'], destination_directory='LLE_data')
            lle = common.find_version(f'LLE_data/gll_lle_{trg["name"]}_v*.fit')
            if not lle:
                raise FileNotFoundError('no LLE file')
            ver = lle.split('_')[-1].replace('.fit', '')
            from astropy.io import fits
            h = fits.getheader(lle, 0)
            info.update(has_lle=True, lle_trigger=trg['name'], lle_version=ver,
                        lle_file=lle, lle_ft2=f'LLE_data/gll_pt_{trg["name"]}_{ver}.fit',
                        lle_rsp=f'LLE_data/gll_cspec_{trg["name"]}_{ver}.rsp',
                        lle_range=[h['TSTART'] - T0, h['TSTOP'] - T0])
            log(f'  LLE data of {trg["name"]} ({ver}): T0{info["lle_range"][0]:+.0f} to {info["lle_range"][1]:+.0f} s')
            break
        except Exception as e:
            log(f'  {trg["name"]}: no LLE data ({type(e).__name__}: {str(e)[:120]})')
    save()

# ------------------------------------------------------------------ LAT
if 'lat' not in skip:
    log('### STAGE: lat  (FT1/FT2 download around the Sun)')
    try:
        from sunflare import lat_data
        repo = os.path.join(d, 'LAT_data')
        if args.lat_ft1:
            info['lat_name'] = lat_data.import_local(os.path.abspath(args.lat_ft1), os.path.abspath(args.lat_ft2),
                                                     info['sun_ra'], info['sun_dec'], T0, args.tmin, args.tmax, repo)
            info['lat_source'] = os.path.abspath(args.lat_ft1)
        else:
            info['lat_name'] = lat_data.download(info['sun_ra'], info['sun_dec'], T0, args.tmin, args.tmax, repo,
                                                 radius=args.lat_radius)
        ft1, ft2, _ = lat_data.lat_files(repo, info['lat_name'])
        log(f'  {os.path.basename(ft1)}  {os.path.basename(ft2)}')
        wins, track = lat_data.visibility(ft2, T0, info['sun_ra'], info['sun_dec'],
                                          common.DEFAULTS['LAT_VIS_THETA_MAX'], common.DEFAULTS['zmax'])
        info['lat_windows'] = wins
        common.save_json('lat_track.json', track)
        ev = lat_data.events(ft1, T0, None, 10.0, emin=100.0, sun_track=True)
        info['lat_events_100MeV'] = int(len(ev['t']))
        close = ev['sep'] < 2 * lat_data.psf68(ev['E'])
        order = np.argsort(ev['E'][close])[::-1][:10]
        info['top_photons'] = [{'E_MeV': float(ev['E'][close][i]), 't': float(ev['t'][close][i]),
                                'sep': float(ev['sep'][close][i])} for i in order]
        for w in wins:
            log(f'  Sun visible (theta < {common.DEFAULTS["LAT_VIS_THETA_MAX"]:g} deg) from T0{w[0]:+.0f} to T0{w[1]:+.0f} s')
        info['has_lat'] = True
        if args.sun_centered:
            lat_data.make_sun_centered(repo, os.path.join(d, 'LAT_data_sun'), info['lat_name'], log)
            info['has_lat_sun'] = True
    except Exception as e:
        info['has_lat'] = False
        traceback.print_exc()
        warn(f'No LAT data: {type(e).__name__}: {e}')
    save()

info['fetched'] = datetime.datetime.now().isoformat(timespec='seconds')
save()
log('### DONE', common.info_path(fid))
