#!/usr/bin/env python
"""Joint GBM (NaI/BGO) + LLE + LAT spectral fits of the flare with nuclear-line, pion and continuum models.

For every interval of spectral_intervals and every model string of spectral_models
(e.g. 'exp_co_le+nar+brd+g_2p2MeV+pion', see sunflare/sunmodel.py):
  * GBM: daily CSPEC with Sun-pointed RSP2, polynomial background from GBM_bkg_intervals
  * LLE (if available and use_lle), LAT (spec_use_lat) with the Sun as a point source
  * fit, -logL / AIC / BIC, parameters, line fluxes, component photon fluxes, count spectra, nuFnu plot
  * likelihood-ratio TS for every model that adds one component to another (e.g. significance of pion decay)

Usage: python -m sunflare.spectral flares/<id>/config_<id>.json
Output: results/spectrum_<t0>-<t1>/{summary.json, counts_<model>.png, vfv.png, results_<model>.fits}
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from sunflare import common, lat_data  # noqa: E402

cfg = common.load_config(sys.argv[1])
info, fdir = cfg.info, cfg.pwd
T0 = info['T0']
log = lambda *a: print(*a, flush=True)
if not cfg.spectral_intervals:
    log('No spectral intervals selected: nothing to do')
    sys.exit(0)

from threeML import (TimeSeriesBuilder, JointLikelihood, DataList, update_logging_level,  # noqa: E402
                     display_spectrum_model_counts)
import astropy.units as u  # noqa: E402
from sunflare.sunmodel import SunModel, ensure_templates, plot_model  # noqa: E402
update_logging_level('WARNING')
ensure_templates()

RA, DEC = common.source_position(cfg)
use_lat = bool(cfg.spec_use_lat and info.get('has_lat'))
use_lle = bool(cfg.use_lle and info.get('has_lle'))
dets = [d for d in cfg.GBM_detectors if any(x['det'] == d and x.get('rsp') for x in info.get('detectors', []))] \
    if cfg.use_gbm else []
if cfg.use_gbm and len(dets) < len(cfg.GBM_detectors):
    log(f'  WARNING: no Sun-pointed response for {sorted(set(cfg.GBM_detectors) - set(dets))}: not used')
if use_lat and cfg.sun_centered and not os.path.isdir(common.lat_dir(cfg)):
    lat_data.make_sun_centered(common.lat_dir(cfg, False), common.lat_dir(cfg, True), info['lat_name'], log)

bkg = common.interval_strings(cfg.GBM_bkg_intervals)
LLE_DT = 0.0
if use_lle:
    from astropy.io import fits as _fits
    _lf = info['lle_file'] if os.path.isabs(info['lle_file']) else os.path.join(fdir, info['lle_file'])
    LLE_DT = float(_fits.getheader(_lf, 0)['TRIGTIME']) - T0       # LLE trigger time relative to the flare T0
lle_bkg = common.interval_strings(cfg.LLE_bkg_intervals)
BANDS = [(300, 1000, '0.3–1 MeV'), (1000, 10000, '1–10 MeV'), (10000, 100000, '10–100 MeV'),
         (100000, 10000000, '0.1–10 GeV')]


SETUP_PROBLEMS = []


def with_background(make, intervals, order, name):
    """Build a TimeSeriesBuilder and fit its background; retry with a constant, else give up on it."""
    for o in ([int(order), 0] if int(order) != 0 else [0]):
        try:
            ts = make(o)
            ts.set_background_interval(*intervals, unbinned=False)
            if o != int(order):
                SETUP_PROBLEMS.append(f'{name}: background polynomial (order {order}) failed; used order 0')
            return ts
        except Exception as e:
            err = f'{type(e).__name__}: {e}'
    SETUP_PROBLEMS.append(f'{name}: background fit failed ({err}); {name} not used')
    log(f'  WARNING: {SETUP_PROBLEMS[-1]}')
    return None


def gbm_series():
    out = {}
    for det in dets:
        row = next(x for x in info['detectors'] if x['det'] == det)
        make = lambda o, row=row, det=det: TimeSeriesBuilder.from_gbm_cspec_or_ctime(
            det, cspec_or_ctime_file=os.path.join(fdir, 'GBM_data', row['pha']),
            rsp_file=os.path.join(fdir, 'GBM_data', row['rsp']), trigger_time=T0, poly_order=o, verbose=False)
        ts = with_background(make, bkg, cfg.GBM_bkg_poly_order, f'GBM {det}')
        if ts is not None:
            out[det] = ts
    return out


def lle_series():
    f = lambda k: info[k] if os.path.isabs(info[k]) else os.path.join(fdir, info[k])
    make = lambda o: TimeSeriesBuilder.from_lat_lle('LLE', lle_file=f('lle_file'), ft2_file=f('lle_ft2'),
                                                    rsp_file=f('lle_rsp'), poly_order=o, verbose=False)
    # threeML wants the reference time inside the LLE file: keep the LLE trigger time and shift our intervals
    shifted = [f'{a - LLE_DT:g}-{b - LLE_DT:g}' for a, b in cfg.LLE_bkg_intervals]
    return with_background(make, shifted, cfg.LLE_bkg_poly_order, 'LLE')


log('### STAGE: GBM/LLE background fits')
GBM = gbm_series() if dets else {}
LLE = lle_series() if use_lle else None
for d, ts in GBM.items():
    try:
        fig = ts.view_lightcurve(start=min(x for x, _ in cfg.GBM_bkg_intervals) - 100,
                                 stop=max(y for _, y in cfg.GBM_bkg_intervals) + 100, use_binner=False)
        os.makedirs(os.path.join(fdir, 'results', 'background'), exist_ok=True)
        fig.savefig(os.path.join(fdir, 'results', 'background', f'bkg_{d}.png'), dpi=90, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        log(f'  background plot {d}: {e}')


def comp_flux(fn, lo, hi):
    e = np.logspace(np.log10(lo), np.log10(hi), 400)
    return float(np.trapezoid(fn(e), e))


def nested_pairs(names):
    res = []
    for a in names:
        for b in names:
            A, B = set(a.split('+')), set(b.split('+'))
            if A < B and len(B - A) == 1:
                res.append((a, b, (B - A).pop()))
    return res


for (t0, t1) in cfg.spectral_intervals:
    t0, t1 = float(t0), float(t1)
    tag = f'{t0:.1f}-{t1:.1f}'
    log(f'### STAGE: spectrum {t0:g}-{t1:g} s')
    out = os.path.join(fdir, 'results', f'spectrum_{tag}')
    common.ensure_fresh(out, dict(dets=dets, bkg=bkg, order=cfg.GBM_bkg_poly_order, lle=use_lle, lle_bkg=lle_bkg,
                                  lat=use_lat, irfs=cfg.irfs, roi=cfg.roi, zmax=cfg.zmax, sun=cfg.sun_centered,
                                  latE=(cfg.SPEC_LAT_EMIN_MEV, cfg.SPEC_LAT_EMAX_MEV),
                                  nai=(cfg.GBM_NAI_EMIN_KEV, cfg.GBM_NAI_EMAX_KEV),
                                  bgo=(cfg.GBM_BGO_EMIN_KEV, cfg.GBM_BGO_EMAX_KEV), eac=cfg.effective_area_correction))
    os.chdir(out)
    plugins, problems = [], list(SETUP_PROBLEMS)
    first_bgo = True
    for det, ts in GBM.items():
        try:
            ts.set_active_time_interval(f'{t0:g}-{t1:g}')
            p = ts.to_spectrumlike()
            if det[0] == 'n':
                p.set_active_measurements(f'{cfg.GBM_NAI_EMIN_KEV:g}-30', f'40-{cfg.GBM_NAI_EMAX_KEV:g}')
            else:
                p.set_active_measurements(f'{cfg.GBM_BGO_EMIN_KEV:g}-{cfg.GBM_BGO_EMAX_KEV:g}')
            eac = float(cfg.effective_area_correction or 0)
            if eac > 0 and not (det[0] == 'b' and first_bgo):
                p.use_effective_area_correction(1 - eac, 1 + eac)
            first_bgo &= det[0] != 'b'
            plugins.append(p)
        except Exception as e:
            problems.append(f'GBM {det}: {type(e).__name__}: {e}')
    if LLE is not None:
        try:
            LLE.set_active_time_interval(f'{t0 - LLE_DT:g}-{t1 - LLE_DT:g}')
            p = LLE.to_spectrumlike()
            p.set_active_measurements(f'{cfg.LLE_EMIN_MEV * 1e3:g}-{cfg.LLE_EMAX_MEV * 1e3:g}')
            plugins.append(p)
        except Exception as e:
            problems.append(f'LLE: {type(e).__name__}: {e}')
    if use_lat:
        try:
            from sunflare.latfit import plugins_from_builder
            b = lat_data.builder(cfg, [t0], [t1], outfile='lat_spec', source_model='PowerLaw2',
                                 emin=cfg.SPEC_LAT_EMIN_MEV, emax=cfg.SPEC_LAT_EMAX_MEV)
            pl = plugins_from_builder(b)
            plugins += [v[0] for v in pl.values()]
        except Exception as e:
            problems.append(f'LAT: {type(e).__name__}: {e}')
    for pmsg in problems:
        log('  WARNING:', pmsg)
    log('PLUGINS:', [p.name for p in plugins])
    summary = {'interval': [t0, t1], 'plugins': [p.name for p in plugins], 'problems': problems, 'fits': {},
               'failed': {}, 'best': None, 'nested': []}
    if not plugins:
        common.save_json('summary.json', summary)
        continue
    data = DataList(*plugins)
    fitted = {}
    for name in cfg.spectral_models:
        log(f'### MODEL {name}')
        try:
            sm = SunModel(name, ra=RA, dec=DEC)
            jl = JointLikelihood(sm.model, data, verbose=False)
            for p in list(sm.model.parameters):
                if 'GalacticTemplate' in p:
                    sm.model[p].value = 1.0
                    sm.model[p].fix = True
            jl.set_minimizer('minuit')
            no_errors = False
            try:
                par_df, like_df = jl.fit(quiet=True)
            except IndexError:
                # covariance sampling gave no valid sample (parameters at their bounds): keep the best fit
                log('  covariance unusable (parameters at bounds?): refitting without errors')
                par_df, like_df = jl.fit(quiet=True, compute_covariance=False)
                no_errors = True
        except Exception as e:
            import traceback
            traceback.print_exc()
            log(f'  fit failed: {type(e).__name__}: {e}')
            summary['failed'][name] = f'{type(e).__name__}: {e}'
            continue
        res = jl.results
        stat = res.get_statistic_measure_frame().iloc[:, 0]
        r = {'logL': float(-like_df.loc['total', '-log(likelihood)']),
             'AIC': float(stat['AIC']), 'BIC': float(stat['BIC']),
             'n_free': len(sm.model.free_parameters), 'params': [], 'components': {}, 'lines': {},
             'no_errors': no_errors}
        try:
            df = res.get_data_frame()
        except Exception:
            import pandas as pd
            fp = sm.model.free_parameters
            df = pd.DataFrame({'value': [p.value for p in fp.values()], 'negative_error': np.nan,
                               'positive_error': np.nan, 'unit': [str(p.unit) for p in fp.values()]}, index=list(fp))
        for i in df.index:
            r['params'].append([i.replace('Sun.spectrum.main.composite.', '').replace('Sun.spectrum.main.', ''),
                                float(df.loc[i, 'value']), float(df.loc[i, 'negative_error']),
                                float(df.loc[i, 'positive_error']), str(df.loc[i, 'unit'])])
        funcs = sm.component_functions()
        for c, f in zip(sm.components, funcs):
            r['components'][c] = {lab: comp_flux(f, lo, hi) for lo, hi, lab in BANDS}
            if c.startswith('g_'):
                k = sm.path(c, 'F')
                if k in df.index:
                    r['lines'][c] = [float(df.loc[k, 'value']), float(df.loc[k, 'negative_error']),
                                     float(df.loc[k, 'positive_error'])]
        for lo, hi, lab in BANDS:
            try:
                fl = res.get_flux(ene_min=lo * u.keV, ene_max=hi * u.keV, flux_unit='1/(cm2 s)', use_components=False)
                r.setdefault('total_flux', {})[lab] = [float(fl['flux'].iloc[0].value), float(fl['low bound'].iloc[0].value),
                                                       float(fl['hi bound'].iloc[0].value)]
            except Exception:
                pass
        try:
            res.write_to(f'results_{name}.fits', overwrite=True)
        except Exception as e:
            log(f'  could not write the results file: {e}')
        try:
            fig = display_spectrum_model_counts(jl, step=False, figsize=(10, 9), min_rate=1e-3)
            fig.axes[0].set_title(f'{cfg.name}  {t0:g}–{t1:g} s   {name}')
            fig.savefig(f'counts_{name}.png', dpi=95, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            log(f'  count plot failed: {e}')
        summary['fits'][name] = r
        fitted[name] = sm
        log(f'  -logL = {-r["logL"]:.2f}  BIC = {r["BIC"]:.1f}  ' +
            '  '.join(f'{p[0]}={p[1]:.3g}' for p in r['params'][:8]))
        plt.close('all')
    if summary['fits']:
        summary['best'] = min(summary['fits'], key=lambda m: summary['fits'][m]['BIC'])
        for a, b, comp in nested_pairs(list(summary['fits'])):
            ts = 2 * (summary['fits'][b]['logL'] - summary['fits'][a]['logL'])
            summary['nested'].append({'base': a, 'with': b, 'component': comp, 'TS': ts})
            log(f'  adding {comp} to {a}: TS = {ts:.1f}')
        log('BEST MODEL (BIC):', summary['best'])
        fig, ax = plt.subplots(figsize=(10, 7))
        emax = cfg.SPEC_LAT_EMAX_MEV * 1e3 if use_lat else (cfg.LLE_EMAX_MEV * 1e3 if use_lle else cfg.GBM_BGO_EMAX_KEV)
        for i, (name, sm) in enumerate(fitted.items()):
            best = name == summary['best']
            plot_model(sm, ax, min(cfg.GBM_NAI_EMIN_KEV if any(d[0] == 'n' for d in dets) else 300, 300), emax,
                       color=plt.cm.tab10(i % 10), components=best, alpha=1 if best else 0.45, lw=2.5 if best else 1.3,
                       label=f'{name}  (BIC {summary["fits"][name]["BIC"]:.1f})')
        ax.legend(fontsize=8, loc='lower left'); ax.grid(alpha=0.3)
        ax.set_title(f'{cfg.name}  {t0:g}–{t1:g} s — best model: {summary["best"]}')
        fig.savefig('vfv.png', dpi=110, bbox_inches='tight')
        plt.close(fig)
    common.save_json('summary.json', summary)
log('### DONE spectral fits')
