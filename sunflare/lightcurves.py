#!/usr/bin/env python
"""Multi-instrument light curves: GOES XRS, GBM NaI/BGO bands, LLE, LAT events and Sun visibility.

Usage: python -m sunflare.lightcurves flares/<id>/config_<id>.json
Output: results/lightcurves/{overview.png, zoom.png, lightcurves.json}
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402

from sunflare import common, lcdata  # noqa: E402

cfg = common.load_config(sys.argv[1])
info, fdir = cfg.info, cfg.pwd
out = os.path.join(fdir, 'results', 'lightcurves')
os.makedirs(out, exist_ok=True)
T0 = info['T0']
log = lambda *a: print(*a, flush=True)


def utc(t):
    return common.met2datetime_array(np.asarray(t) + T0)


def goes_class_axis(ax):
    ax2 = ax.twinx()
    ax2.set_yscale('log')
    ax2.set_ylim(ax.get_ylim())
    ticks = [t for t in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4] if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
    ax2.set_yticks(ticks)
    ax2.set_yticklabels(['ABCMX'[int(round(np.log10(t))) + 8] for t in ticks])
    ax2.minorticks_off()
    for t in ticks:
        ax.axhline(t, color='0.85', lw=0.6, zorder=0)


det_nai = next((d for d in cfg.GBM_detectors if d[0] == 'n'), (info.get('suggested_detectors') or ['n0'])[0])
det_bgo = next((d for d in cfg.GBM_detectors if d[0] == 'b'), 'b0')
summary = {'nai': det_nai, 'bgo': det_bgo, 'peaks': {}}


def make(tmin, tmax, fname, dt=None, title=''):
    span = tmax - tmin
    dt = dt or max(1.024, span / 600.)
    edges = np.arange(tmin, tmax + dt, dt)
    panels = ['goes'] if info.get('has_goes') else []
    if info.get('has_gbm'):
        panels += ['nai', 'bgo']
    if info.get('has_lle') and cfg.use_lle:
        panels += ['lle']
    if info.get('has_lat'):
        panels += ['lat', 'theta']
    hr = {'goes': 1.3, 'nai': 1.3, 'bgo': 1.3, 'lle': 1, 'lat': 1.4, 'theta': 0.7}
    fig, axs = plt.subplots(len(panels), 1, sharex=True, figsize=(13, 2.1 * sum(hr[p] for p in panels) + 0.8),
                            gridspec_kw=dict(height_ratios=[hr[p] for p in panels], hspace=0.06))
    axs = np.atleast_1d(axs)
    for ax, p in zip(axs, panels):
        if p == 'goes':
            t, a, b = lcdata.goes(fdir)
            m = (t >= tmin) & (t <= tmax)
            ax.plot(utc(t[m]), b[m], color='tab:red', lw=1.2, label='XRS-B 1–8 Å')
            ax.plot(utc(t[m]), a[m], color='tab:blue', lw=1.2, label='XRS-A 0.5–4 Å')
            ax.set_yscale('log')
            lo = np.nanmin(a[m][a[m] > 0]) if np.any(a[m] > 0) else 1e-9
            ax.set_ylim(max(lo * 0.5, 1e-9), np.nanmax(b[m]) * 3 if np.isfinite(b[m]).any() else 1e-3)
            ax.set_ylabel('GOES flux\n[W m$^{-2}$]')
            goes_class_axis(ax)
            ax.legend(loc='upper right', fontsize=9)
            if np.isfinite(b[m]).any():
                summary['peaks']['GOES XRS-B'] = float(t[m][np.nanargmax(b[m])])
        elif p in ('nai', 'bgo'):
            det = det_nai if p == 'nai' else det_bgo
            bands = cfg.nai_bands_kev if p == 'nai' else cfg.bgo_bands_kev
            cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(bands)))
            for (lo, hi), c in zip(bands, cols):
                try:
                    tc, r, (elo, ehi) = lcdata.gbm(fdir, det, lo, hi, edges, T0)
                except Exception as e:
                    log(f'  {det} {lo}-{hi} keV: {e}')
                    continue
                lab = f'{lo:g}–{hi:g} keV' if hi < 1000 else f'{lo / 1000:g}–{hi / 1000:g} MeV'
                ax.step(utc(tc), r, where='mid', color=c, lw=1, label=lab)
                if np.isfinite(r).any():
                    summary['peaks'][f'{det} {lab}'] = float(tc[np.nanargmax(r)])
            ax.set_yscale('log')
            ax.set_ylabel(f'GBM {det}\n[counts s$^{{-1}}$]')
            ax.legend(loc='upper right', fontsize=8, ncol=2)
        elif p == 'lle':
            tc, r = lcdata.lle(fdir, info, cfg.LLE_EMIN_MEV, cfg.LLE_EMAX_MEV, edges)
            ax.step(utc(tc), r, where='mid', color='purple', lw=1)
            ax.set_ylabel(f'LLE {cfg.LLE_EMIN_MEV:g}–{cfg.LLE_EMAX_MEV:g} MeV\n[counts s$^{{-1}}$]')
        elif p == 'lat':
            ev = lcdata.lat_events(fdir, info, roi=min(cfg.roi, 10.0), emin=100.0, zmax=cfg.zmax)
            m = (ev['t'] >= tmin) & (ev['t'] <= tmax)
            from sunflare.lat_data import psf68
            close = m & (ev['sep'] < 1.5 * psf68(ev['E']))
            ax.scatter(utc(ev['t'][m & ~close]), ev['E'][m & ~close], s=6, color='0.6', label='ROI')
            ax.scatter(utc(ev['t'][close]), ev['E'][close], s=14, color='tab:red',
                       label=r'$<1.5\,r_{68}$ from the Sun')
            ax.set_yscale('log'); ax.set_ylim(80, max(2e3, ev['E'][m].max() * 1.5 if m.any() else 2e3))
            ax.set_ylabel('LAT energy\n[MeV]')
            ax.legend(loc='upper right', fontsize=8)
            axr = ax.twinx()
            dt_lat = max(dt, span / 120.)
            e2 = np.arange(tmin, tmax + dt_lat, dt_lat)
            c, _ = np.histogram(ev['t'][close], bins=e2)
            axr.step(utc(0.5 * (e2[1:] + e2[:-1])), c / dt_lat, where='mid', color='k', lw=0.9)
            axr.set_ylabel('rate [s$^{-1}$]', fontsize=9)
            summary['lat_events_close'] = int(close.sum())
        elif p == 'theta':
            tr = lcdata.lat_track(fdir)
            if tr:
                t = np.array(tr['t']); m = (t >= tmin) & (t <= tmax)
                ax.plot(utc(t[m]), np.array(tr['theta'])[m], color='k', lw=1, label='Sun off-axis θ')
                ax.plot(utc(t[m]), np.array(tr['zenith'])[m], color='tab:orange', lw=1, label='Sun zenith angle')
                saa = np.array(tr['saa'])[m].astype(bool)
                if saa.any():
                    ax.fill_between(utc(t[m]), 0, 180, where=saa, color='tab:purple', alpha=0.15, label='SAA', step='mid')
                ax.axhline(cfg.LAT_VIS_THETA_MAX, ls='--', color='k', lw=0.6)
                ax.axhline(cfg.zmax, ls=':', color='tab:orange', lw=0.8)
            ax.set_ylim(0, 180); ax.set_ylabel('angle\n[deg]')
            ax.legend(loc='upper right', fontsize=8, ncol=3)
    for ax in axs:
        for w in info.get('lat_windows', []):
            if w[1] > tmin and w[0] < tmax:
                ax.axvspan(utc(max(w[0], tmin)), utc(min(w[1], tmax)), color='tab:green', alpha=0.06, lw=0)
        for (a, b) in cfg.spectral_intervals:
            ax.axvspan(utc(a), utc(b), color='gold', alpha=0.18, lw=0)
        if cfg.LOC_interval:
            ax.axvspan(utc(cfg.LOC_interval[0]), utc(cfg.LOC_interval[1]), ymin=0.94, ymax=1, color='tab:red', alpha=0.5, lw=0)
        if info.get('tpeak') is not None and tmin < info['tpeak'] < tmax:
            ax.axvline(utc(info['tpeak']), color='k', ls=':', lw=0.8)
        ax.grid(alpha=0.25)
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axs[-1].set_xlabel(f'{common.met2utc(T0)[:10]} [UTC]')
    axs[-1].set_xlim(utc(tmin), utc(tmax))
    axs[0].set_title(title or f'{cfg.name}  {info.get("goes_class", "")}   (T0 = {info["t0_utc"]} UTC)')
    fig.savefig(os.path.join(out, fname), dpi=110, bbox_inches='tight')
    plt.close(fig)
    log(f'  saved {fname}  ({", ".join(panels)})')


log('### STAGE: overview light curve')
make(info['tmin'], info['tmax'], 'overview.png', title=f'{cfg.name}  {info.get("goes_class", "")}  — full data window')
lo = cfg.lc_tmin if cfg.lc_tmin is not None else -600.0
hi = cfg.lc_tmax if cfg.lc_tmax is not None else max(1800.0, (info.get('tend') or 1200.0) + 900.0)
log(f'### STAGE: zoomed light curve ({lo:g} to {hi:g} s)')
make(lo, hi, 'zoom.png')
summary['zoom'] = [lo, hi]
common.save_json(os.path.join(out, 'lightcurves.json'), summary)
log('### DONE light curves')
