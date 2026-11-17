#!/usr/bin/env python
"""LAT likelihood analysis of the Sun: time-integrated spectrum and time-resolved light curve.

For each interval and each model in LAT_TR_models (pion template / power law / cutoff power law):
TS, flux and energy flux in [FLUX_EMIN_MEV, FLUX_EMAX_MEV], spectral parameters (proton index s for the
pion template), 95% upper limits below LAT_TS_MIN. Bins: one per Sun visibility window ('windows'),
Bayesian blocks of the Sun photons ('bb'), N photons per bin ('const') or custom intervals.

Usage: python -m sunflare.lat_analysis flares/<id>/config_<id>.json
Output: results/lat/{lat_summary.json, lat_lightcurve.png, lat_integrated_*.png, lat_events.json}
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
import matplotlib.dates as mdates  # noqa: E402

from sunflare import common, lcdata, lat_data  # noqa: E402

cfg = common.load_config(sys.argv[1])
info, fdir = cfg.info, cfg.pwd
T0 = info['T0']
log = lambda *a: print(*a, flush=True)
if not info.get('has_lat'):
    sys.exit('No LAT data for this flare')
if cfg.sun_centered and not os.path.isdir(common.lat_dir(cfg)):
    lat_data.make_sun_centered(common.lat_dir(cfg, False), common.lat_dir(cfg, True), info['lat_name'], log)

from threeML import update_logging_level, display_spectrum_model_counts  # noqa: E402
from sunflare import latfit  # noqa: E402
from sunflare.sunmodel import ensure_templates  # noqa: E402
update_logging_level('WARNING')
ensure_templates()

out = os.path.join(fdir, 'results', 'lat')
settings = dict(irfs=cfg.irfs, roi=cfg.roi, zmax=cfg.zmax, thetamax=cfg.thetamax, e=(cfg.LAT_EMIN_MEV, cfg.LAT_EMAX_MEV),
                sun=cfg.sun_centered, strategy=cfg.strategy)
common.ensure_fresh(out, settings)
os.chdir(out)
RA, DEC = common.source_position(cfg)
E1, E2 = cfg.FLUX_EMIN_MEV, cfg.FLUX_EMAX_MEV
models = cfg.LAT_TR_models or ['pion']
windows = [w for w in info.get('lat_windows', []) if w[1] > info['tmin'] and w[0] < info['tmax']]


def clip_to_windows(edges_list):
    """Split [a, b] intervals at the visibility windows and drop the parts outside them."""
    res = []
    for a, b in edges_list:
        for w0, w1 in windows:
            lo, hi = max(a, w0), min(b, w1)
            if hi - lo >= 5:
                res.append([round(lo, 3), round(hi, 3)])
    return res


# ------------------------------------------------------------------ intervals
integ = cfg.LAT_integrated_interval
if not integ:
    after = [w for w in windows if w[1] > 0]
    integ = [max(0.0, after[0][0]), after[-1][1]] if after else [0.0, info['tmax']]
integ = [float(integ[0]), float(integ[1])]
ev = lcdata.lat_events(fdir, info, roi=cfg.roi, emin=cfg.LAT_EMIN_MEV, zmax=cfg.zmax, sun_centered=cfg.sun_centered)
near = ev['sep'] < 1.5 * lat_data.psf68(ev['E'])
t_near = np.sort(ev['t'][near & (ev['t'] >= integ[0]) & (ev['t'] <= integ[1])])
method = cfg.LAT_TR_method
if method == 'custom' and cfg.LAT_TR_custom:
    bins = [[float(a), float(b)] for a, b in cfg.LAT_TR_custom]
elif method == 'bb' and len(t_near) > 5:
    from threeML.utils import bayesian_blocks
    e = bayesian_blocks.bayesian_blocks(t_near, integ[0], integ[1], p0=float(cfg.LAT_TR_bb_p0))
    bins = clip_to_windows(list(zip(e[:-1], e[1:])))
elif method == 'const' and len(t_near) > 2 * cfg.LAT_TR_N:
    n = int(cfg.LAT_TR_N)
    e = [integ[0]] + [0.5 * (t_near[i - 1] + t_near[i]) for i in range(n, len(t_near) - n // 2, n)] + [integ[1]]
    bins = clip_to_windows(list(zip(e[:-1], e[1:])))
else:
    method = 'windows'
    bins = clip_to_windows([integ])
bins = [b for b in bins if b[1] - b[0] >= 5]
log(f'### STAGE: LAT intervals: integrated {integ[0]:g}-{integ[1]:g} s, {len(bins)} time-resolved bins ({method})')
for b in bins:
    log(f'  bin {b[0]:9.1f} - {b[1]:9.1f} s')

# ------------------------------------------------------------------ GtBurst data preparation
log('### STAGE: LAT data selection and exposure (GtBurst)')
src_model = 'ExpCutoff' if 'expco' in models else 'PowerLaw2'
all_iv = [integ] + [b for b in bins if b != integ]
plugins = latfit.plugins_from_builder(lat_data.builder(cfg, [a for a, _ in all_iv], [b for _, b in all_iv],
                                                       outfile='lat_tr', source_model=src_model))
log('  intervals ready:', ', '.join(f'{a:g}-{b:g}' for a, b in plugins))


def get_plugin(a, b):
    k = (round(a, 3), round(b, 3))
    if k in plugins:
        return plugins[k][0]
    near_k = min(plugins, key=lambda x: abs(x[0] - a) + abs(x[1] - b))
    return plugins[near_k][0]


def analyse(a, b, tag, plot=False):
    row = {'t0': a, 't1': b, 'fits': {}}
    for kind in models:
        try:
            jl, r = latfit.fit(get_plugin(a, b), kind, RA, DEC, E1, E2)
        except Exception as e:
            log(f'  {tag} {kind}: fit failed: {e}')
            row['fits'][kind] = {'error': str(e)}
            continue
        if r['TS'] < cfg.LAT_TS_MIN:
            try:
                r.update({k: v for k, v in latfit.upper_limit(get_plugin(a, b), kind, RA, DEC, E1, E2).items()
                          if k in ('flux_ul', 'eflux_ul', 'ul_shape')})
            except Exception as e:
                log(f'  {tag} {kind}: upper limit failed: {e}')
        par = ', '.join(f'{p}={v[0]:.3g}' for p, v in r['params'].items())
        log(f'  {tag} {a:9.1f}-{b:9.1f} s  {kind:6s} TS={r["TS"]:8.1f}  F={r["flux"][0]:.3e}  {par}')
        if plot:
            try:
                fig = display_spectrum_model_counts(jl, step=False, figsize=(9, 8))
                fig.axes[0].set_title(f'{cfg.name}  LAT {a:g}-{b:g} s  — {common.LAT_MODELS[kind]}')
                fig.savefig(f'lat_integrated_counts_{kind}.png', dpi=100, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                log(f'  count-spectrum plot failed: {e}')
            r['model_eval'] = sed_points(jl)
        row['fits'][kind] = r
    ok = {k: v for k, v in row['fits'].items() if 'logL' in v}
    if ok:
        row['best'] = max(ok, key=lambda k: ok[k]['logL'] - (1 if k == 'expco' else 0))
        row['TS'] = max(v['TS'] for v in ok.values())
    return row


def sed_points(jl):
    e = np.logspace(np.log10(cfg.LAT_EMIN_MEV * 1e3), np.log10(cfg.LAT_EMAX_MEV * 1e3), 60)
    f = jl.likelihood_model[latfit.LAT_SOURCE].spectrum.main.shape(e)
    return {'E_keV': e.tolist(), 'vFv_erg': (e ** 2 * f * 1.602176634e-9).tolist()}


log('### STAGE: LAT time-integrated fit')
integrated = analyse(integ[0], integ[1], 'integrated', plot=True)
log('### STAGE: LAT time-resolved fits')
resolved = [analyse(a, b, f'bin {i + 1}/{len(bins)}') for i, (a, b) in enumerate(bins)] if bins != [integ] else [integrated]

summary = {'integrated': integrated, 'resolved': resolved, 'method': method, 'models': models,
           'flux_band_MeV': [E1, E2], 'TS_MIN': cfg.LAT_TS_MIN, 'windows': windows, 'source_position': [RA, DEC],
           'sun_centered': bool(cfg.sun_centered), 'irfs': cfg.irfs}
common.save_json('lat_summary.json', summary)

# source-probability events of the time-integrated interval (gtsrcprob), for the report's event table
from sunflare.lat_events import extract as extract_events  # noqa: E402
extract_events(cfg, log)

# ------------------------------------------------------------------ plots
main = 'pion' if 'pion' in models else models[0]
utc = lambda t: common.met2datetime_array(np.asarray(t, dtype=float) + T0)
rows = [r for r in resolved if main in r['fits'] and 'flux' in r['fits'][main]]
if rows:
    npan = 3 if any(r['fits'][main]['params'] for r in rows) else 2
    fig, axs = plt.subplots(npan, 1, sharex=True, figsize=(12, 3 * npan + 1), gridspec_kw=dict(hspace=0.07))
    det = [r for r in rows if r['fits'][main]['TS'] >= cfg.LAT_TS_MIN]
    ul = [r for r in rows if r['fits'][main]['TS'] < cfg.LAT_TS_MIN]
    def xe(rs):
        c = np.array([0.5 * (r['t0'] + r['t1']) for r in rs]); h = np.array([0.5 * (r['t1'] - r['t0']) for r in rs])
        return utc(c), [utc(c) - utc(c - h), utc(c + h) - utc(c)]
    if det:
        x, xerr = xe(det)
        f = np.array([r['fits'][main]['flux'] for r in det])
        axs[0].errorbar(x, f[:, 0], xerr=xerr, yerr=[f[:, 0] - f[:, 1], f[:, 2] - f[:, 0]], fmt='o', color='tab:red',
                        label=f'LAT {E1:g}–{E2:g} MeV ({main})')
        p = list(det[0]['fits'][main]['params'])[0] if det[0]['fits'][main]['params'] else None
        if p and npan == 3:
            v = np.array([r['fits'][main]['params'][p] for r in det])
            axs[1].errorbar(x, v[:, 0], xerr=xerr, yerr=[np.abs(v[:, 1]), v[:, 2]], fmt='o', color='tab:blue')
            axs[1].set_ylabel(dict(latfit.SHAPE_PARS[main])[p])
    if ul:
        x, xerr = xe(ul)
        u = np.array([r['fits'][main].get('flux_ul', np.nan) for r in ul])
        axs[0].errorbar(x, u, xerr=xerr, yerr=0.3 * u, uplims=True, fmt='none', color='gray', label='95% UL')
    axs[0].set_yscale('log'); axs[0].set_ylabel('Flux [ph cm$^{-2}$ s$^{-1}$]'); axs[0].legend(fontsize=9)
    ts = np.array([r['fits'][main]['TS'] for r in rows])
    x, xerr = xe(rows)
    axs[-1].errorbar(x, np.maximum(ts, 0.1), xerr=xerr, fmt='s', color='k')
    axs[-1].axhline(cfg.LAT_TS_MIN, ls='--', color='tab:red'); axs[-1].set_yscale('log'); axs[-1].set_ylabel('TS')
    if info.get('has_goes'):
        ax2 = axs[0].twinx()
        t, a, b = lcdata.goes(fdir)
        lo, hi = min(r['t0'] for r in rows) - 600, max(r['t1'] for r in rows) + 600
        m = (t > lo) & (t < hi)
        ax2.plot(utc(t[m]), b[m], color='0.6', lw=0.8, zorder=0); ax2.set_yscale('log')
        ax2.set_ylabel('GOES XRS-B [W m$^{-2}$]', color='0.5')
    for ax in axs:
        ax.grid(alpha=0.3)
        for w in windows:
            ax.axvspan(utc(w[0]), utc(w[1]), color='tab:green', alpha=0.05, lw=0)
    lo_t, hi_t = min(r['t0'] for r in rows), max(r['t1'] for r in rows)
    pad = 0.05 * (hi_t - lo_t)
    axs[-1].set_xlim(utc(lo_t - pad), utc(hi_t + pad))
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axs[-1].set_xlabel(f'{common.met2utc(T0)[:10]} [UTC]')
    axs[0].set_title(f'{cfg.name} — Fermi-LAT time-resolved likelihood ({cfg.irfs}, {method} bins)')
    fig.savefig('lat_lightcurve.png', dpi=110, bbox_inches='tight')
    plt.close(fig)

# integrated SED of all models
fig, ax = plt.subplots(figsize=(8, 6))
for i, (k, r) in enumerate(integrated['fits'].items()):
    if 'model_eval' in r:
        ax.plot(np.array(r['model_eval']['E_keV']) / 1e3, r['model_eval']['vFv_erg'], lw=2, color=plt.cm.tab10(i),
                label=f'{common.LAT_MODELS[k]}  (TS={r["TS"]:.0f}, logL={r["logL"]:.1f})')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Energy [MeV]')
ax.set_ylabel(r'$E^2 dN/dE$ [erg cm$^{-2}$ s$^{-1}$]'); ax.grid(alpha=0.3); ax.legend(fontsize=9)
ax.set_title(f'LAT time-integrated {integ[0]:g}–{integ[1]:g} s')
fig.savefig('lat_integrated_sed.png', dpi=110, bbox_inches='tight')
plt.close(fig)
log('### DONE LAT analysis')
