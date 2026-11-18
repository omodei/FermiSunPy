"""Interactive light-curve figures of the report: data embedded at full resolution, drawn and zoomed in the browser.

build_data() collects GOES (1 s), GBM CSPEC (native bins, per band), LLE (1 s), LAT photons, the Sun track and the
LAT time-resolved likelihood results into one compact dict. The page script (JS below) rebins on the fly to the
screen resolution, so the full window shows a smooth curve and zooming reveals the native bins.
"""
import datetime
import glob
import json
import os

import numpy as np
from astropy.io import fits

from sunflare import common, lcdata
from sunflare import params as P


def _r(x, nd):
    return [None if not np.isfinite(v) else round(float(v), nd) for v in np.asarray(x, dtype=float)]


def _log1000(x):
    """log10 * 1000 as integers (None where not positive): GOES fluxes, compactly."""
    x = np.asarray(x, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        lg = np.where(np.isfinite(x) & (x > 0), np.log10(np.where(x > 0, x, 1.0)) * 1000, np.nan)
    return [None if not np.isfinite(v) else int(round(v)) for v in lg]


def _band_label(lo, hi):
    return f'{lo:.0f}–{hi:.0f} keV' if hi < 2000 else f'{lo / 1000:.2g}–{hi / 1000:.3g} MeV'


def _goes(fdir):
    t, a, b = lcdata.goes(fdir)
    d = np.diff(t)
    if len(t) > 2 and np.all(np.abs(d - np.median(d)) < 0.05 * np.median(d)):
        out = {'t0': round(float(t[0]), 3), 'dt': round(float(np.median(d)), 4), 'n': len(t)}
    else:
        out = {'t': _r(t, 2)}
    out.update(a=_log1000(a), b=_log1000(b))
    return out


def _gbm(cfg, info, fdir, summary):
    nai = [summary.get('nai')] + [d for d in cfg.GBM_detectors if d[0] == 'n'] + \
          [d for d in info.get('suggested_detectors') or [] if d[0] == 'n']
    bgo = [summary.get('bgo')] + [d for d in cfg.GBM_detectors if d[0] == 'b'] + ['b0', 'b1']
    dets = []
    for d in nai[:1] + nai[1:] + bgo:
        if d and d not in dets and lcdata.gbm_file(fdir, d):
            dets.append(d)
    dets = [d for d in dets if d[0] == 'n'][:4] + [d for d in dets if d[0] == 'b'][:2]
    if not dets:
        return None
    out = {'dets': {}, 'nai': next((d for d in dets if d[0] == 'n'), None), 'bgo': next((d for d in dets if d[0] == 'b'), None)}
    grid = None
    for det in dets:
        with fits.open(lcdata.gbm_file(fdir, det)) as h:
            sp, eb = h['SPECTRUM'].data, h['EBOUNDS'].data
            ta, tb = sp['TIME'] - info['T0'], sp['ENDTIME'] - info['T0']
            cnt, expo = sp['COUNTS'], sp['EXPOSURE'].astype(float)
            bands = cfg.nai_bands_kev if det[0] == 'n' else cfg.bgo_bands_kev
            rec = {'bands': [], 'c': [], 'x': _r(expo, 3)}
            for lo, hi in bands:
                m = (eb['E_MAX'] > lo) & (eb['E_MIN'] < hi)
                if not m.any():
                    continue
                rec['bands'].append(_band_label(float(eb['E_MIN'][m][0]), float(eb['E_MAX'][m][-1])))
                rec['c'].append(cnt[:, m].sum(1).astype(int).tolist())
        if grid is None:
            grid = (ta, tb)
            out['ta'], out['tb'] = _r(ta, 3), _r(tb, 3)
        if len(ta) != len(grid[0]) or np.abs(ta - grid[0]).max() > 1e-3:
            rec['ta'], rec['tb'] = _r(ta, 3), _r(tb, 3)
        out['dets'][det] = rec
    return out


def _lle(cfg, info, fdir):
    f = info.get('lle_file')
    if not (info.get('has_lle') and f):
        return None
    f = f if os.path.isabs(f) else os.path.join(fdir, f)
    lo, hi = float(cfg.LLE_EMIN_MEV), float(cfg.LLE_EMAX_MEV)
    with fits.open(f) as h:
        ev = h['EVENTS'].data
        m = (ev['ENERGY'] > lo) & (ev['ENERGY'] < hi)
        a, b = h[0].header['TSTART'] - info['T0'], h[0].header['TSTOP'] - info['T0']
        edges = np.arange(np.floor(a), np.ceil(b) + 1.0, 1.0)
        c, _ = np.histogram(ev['TIME'][m] - info['T0'], bins=edges)
    return {'t0': float(edges[0]), 'dt': 1.0, 'c': c.astype(int).tolist(), 'band': f'{lo:g}–{hi:g} MeV'}


def _lat(cfg, info, fdir, events):
    if not info.get('has_lat'):
        return None
    sc = bool(cfg.sun_centered and info.get('has_lat_sun'))
    roi = min(float(cfg.roi), 10.0)
    ev = lcdata.lat_events(fdir, info, roi=roi, emin=100.0, zmax=cfg.zmax, sun_centered=sc)
    from sunflare.lat_data import psf68
    o = np.argsort(ev['t'])
    t, E, sep = ev['t'][o], ev['E'][o], ev['sep'][o]
    near = sep < 1.5 * psf68(E)
    hp = [[round(r['t'], 2), round(r['E'], 1), round(r['pSun'], 3)] for r in events or []
          if (r.get('pSun') or 0) >= 0.9 and r.get('E', 0) >= 1000]
    return {'t': _r(t, 2), 'E': [int(round(x)) for x in E], 'near': near.astype(int).tolist(), 'roi': roi,
            'hp': hp}


def _track(fdir):
    tr = lcdata.lat_track(fdir)
    if not tr:
        return None
    return {'t': _r(tr['t'], 1), 'theta': _r(tr['theta'], 1), 'zenith': _r(tr['zenith'], 1),
            'saa': [int(bool(x)) for x in tr.get('saa', [0] * len(tr['t']))]}


def _like(lat):
    if not lat or not lat.get('resolved'):
        return None
    from sunflare.latfit import SHAPE_PARS
    out = {'TS_MIN': lat['TS_MIN'], 'band': lat['flux_band_MeV'], 'models': {}, 'labels': {}, 'method': lat.get('method')}
    for kind in lat['models']:
        bins = []
        for r in lat['resolved']:
            f = r['fits'].get(kind) or {}
            if 'TS' not in f:
                continue
            pname = SHAPE_PARS[kind][0][0]
            par = f.get('params', {}).get(pname)
            fixed = pname in {x[0] for x in f.get('fixed', [])}
            bins.append({'a': r['t0'], 'b': r['t1'], 'key': f'{r["t0"]:.3f}', 'TS': round(f['TS'], 2),
                         'flux': None if f['TS'] < lat['TS_MIN'] else [float(x) for x in f['flux']],
                         'ul': f.get('flux_ul'), 'par': None if (par is None or fixed or f['TS'] < lat['TS_MIN']) else par,
                         'best': r.get('best') == kind, 'noerr': bool(f.get('no_errors')),
                         'lim': P.describe_limits(P.fit_limits(f, lat_kind=kind, overrides=lat.get('param_overrides'),
                                                               ts_min=lat['TS_MIN']))})
        out['models'][kind] = bins
        out['labels'][kind] = [SHAPE_PARS[kind][0][0], SHAPE_PARS[kind][0][1], common.LAT_MODELS[kind]]
    out['main'] = 'pion' if 'pion' in out['models'] else lat['models'][0]
    return out


def spec_id(a, b):
    return f'spec-{a:g}-{b:g}'


def build_data(cfg, info, fdir, R, lat, events, log=print):
    lc = common.load_json(os.path.join(R, 'lightcurves', 'lightcurves.json'), {}) or {}
    t0 = datetime.datetime.fromisoformat(common.met2utc(info['T0']).replace('Z', '')).replace(tzinfo=datetime.timezone.utc)
    tend = info.get('tend') or 1800
    extra = [(r['t0'], r['t1']) for r in (lat or {}).get('resolved', [])]
    if lat:
        extra.append((lat['integrated']['t0'], lat['integrated']['t1']))
    tmin, tmax = common.full_window(info, cfg, fdir, extra)
    D = {'name': cfg.name, 't0_ms': t0.timestamp() * 1000, 'tmin': tmin, 'tmax': tmax,
         'data_window': [info.get('tmin', -3600), info.get('tmax', 14400)],
         'flare': lc.get('zoom') or [-900, max(1800, tend + 1200)], 'windows': info.get('lat_windows', []),
         'loc': cfg.LOC_interval, 'zmax': cfg.zmax, 'thmax': cfg.LAT_VIS_THETA_MAX, 'tpeak': info.get('tpeak'),
         'latint': [lat['integrated']['t0'], lat['integrated']['t1']] if lat else cfg.LAT_integrated_interval,
         'spectral': []}
    done = set()
    for s in sorted(glob.glob(os.path.join(R, 'spectrum_*', 'summary.json'))):
        S = common.load_json(s) or {}
        if 'interval' not in S:
            continue
        a, b = S['interval']
        pion = [n['TS'] for n in S.get('nested', []) if n['component'] == 'pion']
        D['spectral'].append({'a': a, 'b': b, 'id': spec_id(a, b), 'best': S.get('best'),
                              'pion': max(pion) if pion else None})
        done.add((round(a, 3), round(b, 3)))
    for a, b in cfg.spectral_intervals or []:
        if (round(a, 3), round(b, 3)) not in done:
            D['spectral'].append({'a': a, 'b': b, 'id': None, 'best': None, 'pion': None})
    D['spectral'].sort(key=lambda x: x['a'])
    for k, fn in (('goes', lambda: _goes(fdir) if info.get('has_goes') else None),
                  ('gbm', lambda: _gbm(cfg, info, fdir, lc) if info.get('has_gbm') else None),
                  ('lle', lambda: _lle(cfg, info, fdir) if cfg.use_lle else None),
                  ('lat', lambda: _lat(cfg, info, fdir, events)),
                  ('track', lambda: _track(fdir) if info.get('has_lat') else None),
                  ('like', lambda: _like(lat))):
        try:
            D[k] = fn()
        except Exception as e:      # a missing data set only removes its panel
            log(f'  interactive figure: no {k} data ({e})')
            D[k] = None
    return D


def script(D):
    return JS.replace('LCDATA', json.dumps(D, separators=(',', ':'), allow_nan=False).replace('</', '<\\/'))


def figure(fid, panels, title):
    """Placeholder of one interactive figure; the script fills it. panels: ids in display order."""
    return (f'<div class="ifig" id="{fid}" data-panels="{",".join(panels)}" data-title="{title}">'
            '<div class="ftool"></div><div class="fpanels"></div>'
            '<p class="fhint">Drag across a panel to zoom · Shift-drag to pan · ⌘/Ctrl + wheel or pinch to zoom around '
            'the cursor · double-click for the full window · drag a panel\'s lower edge to resize it · click a gold '
            'interval tab or a LAT point to open its results · Shift-click a LAT point for its count spectrum</p><div class="ftip" hidden></div></div>')


CSS = r'''
.ifig { display:grid; gap:8px; background:var(--surface) } .ifig:fullscreen { padding:14px; overflow:auto }
.ftool { display:flex; flex-wrap:wrap; gap:8px 12px; align-items:center; font-size:13px }
.ftool .grp { display:inline-flex; gap:0; border:1px solid var(--line); border-radius:7px; overflow:hidden }
.ftool .grp button { border:0; border-right:1px solid var(--line); border-radius:0 }
.ftool .grp button:last-child { border-right:0 }
.ftool button, .ftool select, .pc button, .pc select { font:500 12px system-ui; padding:4px 9px; border:1px solid var(--line);
  border-radius:6px; background:var(--raise); color:var(--fg); cursor:pointer }
.ftool button:hover, .pc button:hover { border-color:var(--accent) } .ftool button[aria-pressed="true"] { background:var(--accent); color:#fff }
.ftool input[type=number] { width:86px; font:12px ui-monospace,Menlo,monospace; padding:4px 6px; border:1px solid var(--line);
  border-radius:6px; background:var(--surface); color:var(--fg) }
.ftool .utcr { font:12px ui-monospace,Menlo,monospace; color:var(--muted) }
.ftool details { position:relative } .ftool summary { list-style:none; font:500 12px system-ui; padding:4px 9px;
  border:1px solid var(--line); border-radius:6px; background:var(--raise) }
.ftool details[open] { display:block; padding:0; border:0; background:none }
.ftool .pmenu { position:absolute; z-index:6; top:30px; left:0; background:var(--surface); border:1px solid var(--line);
  border-radius:8px; padding:8px 12px; display:grid; gap:4px; min-width:190px; box-shadow:0 6px 20px rgba(0,0,0,.15) }
.ftool .pmenu label { display:flex; gap:8px; align-items:center; white-space:nowrap }
.fpanels { border:1px solid var(--line); border-radius:8px; overflow:hidden; background:var(--raise); user-select:none; touch-action:none }
.fp { position:relative } .fp canvas { display:block; width:100%; cursor:crosshair }
.fp .rs { height:7px; cursor:ns-resize; border-bottom:1px solid var(--line); background:linear-gradient(var(--raise),var(--raise)) center/34px 3px no-repeat }
.fp .rs:hover { background-color:var(--accent-soft) } .fp:last-child .rs { border-bottom:0 }
.fp .tg { position:absolute; left:70px; top:3px; font:500 12px ui-monospace,Menlo,monospace; color:var(--fg); background:var(--surface);
  border:0; padding:1px 0; pointer-events:none; max-width:calc(100% - 300px); background:none; overflow:hidden;
  text-overflow:ellipsis; white-space:nowrap }
.fp .tg i { display:inline-block; width:10px; height:3px; border-radius:2px; vertical-align:middle; margin:0 4px 0 8px }
.fp .pc { position:absolute; right:58px; top:2px; display:flex; gap:4px } .fp .pc button, .fp .pc select { padding:2px 7px; font-size:11px }
.fhint { margin:0; font-size:12px; color:var(--muted) }
.ftip { position:fixed; z-index:20; pointer-events:none; background:var(--surface); border:1px solid var(--line); border-radius:8px;
  padding:8px 10px; font:12px/1.45 ui-monospace,Menlo,monospace; box-shadow:0 6px 20px rgba(0,0,0,.18); max-width:520px; white-space:pre-wrap }
.ftip b { font-family:system-ui; color:var(--accent) }
details.static { margin-top:4px }
tr.flash td { animation:flash 2.4s ease-out } @keyframes flash { 0%,40% { background:var(--accent-soft) } 100% { background:transparent } }
@media print { .ifig { display:none } details.static > * { display:block } }
'''

JS = r'''
/* ---------- interactive light curves (shared time axis for every figure) ---------- */
(function () {
const D = LCDATA;
const figs = [];
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const ML = 70, MR = 58, MT = 25;
const FULL = [D.tmin, D.tmax];
let view = { a: D.flare[0], b: D.flare[1] }, hist = [], hoverT = null, evMark = null;
let likeModel = D.like ? D.like.main : null;
const hm = location.hash.match(/t=(-?[\d.]+),(-?[\d.]+)/); if (hm && +hm[2] > +hm[1]) view = { a: +hm[1], b: +hm[2] };
const PAL = ['#2563eb', '#0d9488', '#ca8a04', '#dc2626', '#9333ea'];
const COL = () => ({ goesa: '#2563eb', goesb: '#dc2626', lle: '#9333ea', near: '#dc2626', far: css('--muted'), trace: css('--fg'),
  zen: '#ea580c', vis: 'rgba(22,163,74,.09)', spec: 'rgba(234,179,8,.12)', specE: '#ca8a04', loc: 'rgba(220,38,38,.6)',
  integ: 'rgba(37,99,235,.55)', saa: 'rgba(147,51,234,.13)', flux: '#dc2626', par: '#2563eb', ts: css('--fg'), rate: css('--fg') });

/* ---- data helpers ---- */
const F = a => a ? Float64Array.from(a, v => v == null ? NaN : v) : null;
function lb(arr, x) { let lo = 0, hi = arr.length; while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < x) lo = m + 1; else hi = m; } return lo; }
function grid(t0, dt, n) { const ta = new Float64Array(n), tb = new Float64Array(n); for (let i = 0; i < n; i++) { ta[i] = t0 + (i - .5) * dt; tb[i] = t0 + (i + .5) * dt; } return [ta, tb]; }
/* rebin native bins [ta,tb) to at most maxBins over [a,b]; value = sum(num)/sum(den) (den null: mean); gaps break groups */
function agg(ta, tb, num, den, a, b, maxBins) {
  const i0 = lb(tb, a), i1 = lb(ta, b), n = i1 - i0, k = Math.max(1, Math.ceil(n / maxBins)), out = [];
  let i = i0;
  while (i < i1) {
    let s = 0, w = 0, j = i;
    for (; j < i1 && j < i + k; j++) {
      if (j > i && ta[j] - tb[j - 1] > 1e-3 + 0.5 * (tb[j] - ta[j])) break;
      const v = num[j], d = den ? den[j] : 1;
      if (!isFinite(v) || !(d > 0)) continue;
      s += v; w += d;
    }
    out.push([ta[i], tb[j - 1], w > 0 ? s / w : NaN, k]);
    i = j;
  }
  return out;
}
const G = D.goes && (() => { const [ta, tb] = D.goes.t ? (() => { const t = F(D.goes.t); return [t.map((v, i) => i ? (v + t[i - 1]) / 2 : v - .5), t.map((v, i) => i < t.length - 1 ? (v + t[i + 1]) / 2 : v + .5)]; })() : grid(D.goes.t0, D.goes.dt, D.goes.n);
  const un = a => Float64Array.from(a, v => v == null ? NaN : 10 ** (v / 1000)); return { ta, tb, a: un(D.goes.a), b: un(D.goes.b) }; })();
const B = D.gbm && (() => { const o = {}; const ta0 = F(D.gbm.ta), tb0 = F(D.gbm.tb);
  for (const [det, r] of Object.entries(D.gbm.dets)) o[det] = { ta: r.ta ? F(r.ta) : ta0, tb: r.tb ? F(r.tb) : tb0, x: F(r.x), c: r.c.map(F), bands: r.bands };
  return o; })();
const L = D.lle && (() => { const n = D.lle.c.length, [ta, tb] = grid(D.lle.t0 + .5, 1, n); return { ta, tb, c: F(D.lle.c), dt: new Float64Array(n).fill(1) }; })();
const P = D.lat && { t: F(D.lat.t), E: F(D.lat.E), near: D.lat.near };
const TR = D.track && { t: F(D.track.t), th: F(D.track.theta), ze: F(D.track.zenith), saa: D.track.saa };
let det = { nai: D.gbm && D.gbm.nai, bgo: D.gbm && D.gbm.bgo };

/* ---- formatting ---- */
const pad2 = n => String(n).padStart(2, '0');
const utc = (t, sec) => { const d = new Date(D.t0_ms + t * 1000); return `${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())}` + (sec ? ':' + pad2(d.getUTCSeconds()) : ''); };
const g3 = v => !isFinite(v) ? '–' : Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-2 && v !== 0) ? v.toExponential(2) : +v.toPrecision(3) + '';
const goesClass = v => { if (!(v > 0)) return ''; const e = Math.min(-4, Math.floor(Math.log10(v))); return ('ABCMX'[e + 8] || 'A') + (v / 10 ** e).toFixed(1); };
const sgn = t => (t >= 0 ? '+' : '') + (Math.abs(t) >= 100 ? t.toFixed(0) : t.toFixed(1));
function niceStep(span) { const c = [1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800]; return c.find(s => span / s <= 9) || 21600; }

/* ---- panel definitions ---- */
const PANELS = {
  goes: { title: 'GOES XRS', h: 150, log: true, ok: () => !!G, legend: c => [['XRS-B 1–8 Å', c.goesb], ['XRS-A 0.5–4 Å', c.goesa]] },
  nai: { title: 'GBM NaI', h: 140, log: true, ok: () => !!(B && det.nai), det: 'nai' },
  bgo: { title: 'GBM BGO', h: 140, log: true, ok: () => !!(B && det.bgo), det: 'bgo' },
  lle: { title: 'LAT LLE', h: 110, log: false, ok: () => !!L, legend: c => [[`LLE ${D.lle.band}`, c.lle]] },
  lat: { title: 'LAT photons', h: 160, log: true, ok: () => !!P, modes: ['photons', 'rate', 'both'] },
  theta: { title: 'Sun angles', h: 90, nolog: true, ok: () => !!TR, legend: c => [['off-axis θ', c.trace], ['zenith', c.zen], ['SAA', '#9333ea']] },
  flux: { title: 'LAT flux', h: 140, log: true, ok: () => !!D.like, like: true },
  par: { title: 'LAT spectral parameter', h: 110, log: false, ok: () => !!D.like, like: true },
  ts: { title: 'LAT TS', h: 100, log: true, ok: () => !!D.like, like: true },
};
const NAMES = { goes: 'GOES X-rays', nai: 'GBM NaI', bgo: 'GBM BGO', lle: 'LLE', lat: 'LAT photons', theta: 'Sun angles', flux: 'LAT likelihood flux', par: 'LAT spectral parameter', ts: 'LAT TS' };

/* ---- view ---- */
function setView(a, b, push = true) {
  if (!(b > a)) return;
  if (b - a < 2) { const m = (a + b) / 2; a = m - 1; b = m + 1; }
  const span = Math.min(b - a, FULL[1] - FULL[0]);
  if (a < FULL[0]) { a = FULL[0]; b = a + span; } if (b > FULL[1]) { b = FULL[1]; a = b - span; }
  if (push && (Math.abs(a - view.a) > 1e-6 || Math.abs(b - view.b) > 1e-6)) { hist.push({ ...view }); if (hist.length > 60) hist.shift(); }
  view = { a, b }; redraw(); syncTools(); writeHash(); linkTable();
}
window.LCVIEW = { get: () => view, set: setView, hashPart: () => `t=${+view.a.toFixed(1)},${+view.b.toFixed(1)}` };
function writeHash() { try { const tab = location.hash.slice(1).split('&')[0] || (document.querySelector('nav button[aria-selected="true"]') || {}).dataset?.t || ''; history.replaceState(null, '', '#' + tab + '&' + window.LCVIEW.hashPart()); } catch (e) {} }
function linkTable() {
  for (const f of figs) {
    const cb = f.el.querySelector('.lnk'); if (!cb || !cb.checked) continue;
    const a = document.getElementById('evtmin'), b = document.getElementById('evtmax'); if (!a) return;
    a.value = +view.a.toFixed(1); b.value = +view.b.toFixed(1); a.dispatchEvent(new Event('input')); return;
  }
}
const redraw = () => figs.forEach(f => f.draw());

/* ---- figure ---- */
function Figure(el) {
  const F_ = { el, panels: [] };
  const ids = el.dataset.panels.split(',').filter(id => PANELS[id] && PANELS[id].ok());
  const tool = el.querySelector('.ftool'), box = el.querySelector('.fpanels'), tip = el.querySelector('.ftip');
  const presets = [['full', 'Full window'], ['flare', 'Flare']];
  if (D.spectral.length) presets.push(['spec', 'Spectral intervals']);
  if (D.like) presets.push(['bins', 'LAT bins']);
  tool.innerHTML = `<span class="grp">${presets.map(([k, l]) => `<button type="button" data-v="${k}">${l}</button>`).join('')}</span>` +
    (D.windows.length ? `<select class="vw" aria-label="LAT visibility window"><option value="">LAT window…</option>${D.windows.map((w, i) => `<option value="${i}">#${i + 1} ${utc(w[0])}–${utc(w[1])}</option>`).join('')}</select>` : '') +
    `<label>T − T0 <input type="number" class="va" step="any"> to <input type="number" class="vb" step="any"> s</label><span class="utcr"></span>` +
    `<span class="grp"><button type="button" data-z="0.5" title="Zoom in">+</button><button type="button" data-z="2" title="Zoom out">−</button><button type="button" class="back" title="Previous view">↶ Back</button></span>` +
    `<details><summary>Panels</summary><div class="pmenu">${ids.map(id => `<label><input type="checkbox" data-p="${id}" checked> ${NAMES[id]}</label>`).join('')}</div></details>` +
    (document.getElementById('evtable') ? `<label title="Show in the events table only the events of the zoomed time range"><input type="checkbox" class="lnk"> Filter events table</label>` : '') +
    `<span class="grp" style="margin-left:auto"><button type="button" class="fs">⛶ Full screen</button><button type="button" class="png">Save PNG</button><button type="button" class="cp">Copy link</button></span>`;
  for (const id of ids) {
    const d = PANELS[id], p = { id, d, h: d.h, log: !!d.log, vis: true, mode: d.modes ? d.modes[0] : null };
    const div = document.createElement('div'); div.className = 'fp';
    let pc = '';
    if (d.det) pc += `<select data-a="det">${Object.keys(B).filter(k => k[0] === (d.det === 'nai' ? 'n' : 'b')).map(k => `<option${k === det[d.det] ? ' selected' : ''}>${k}</option>`).join('')}</select>`;
    if (id === 'flux' && Object.keys(D.like.models).length > 1) pc += `<select data-a="model">${Object.keys(D.like.models).map(k => `<option value="${k}"${k === likeModel ? ' selected' : ''}>${({ pion: 'pion', pow: 'power law', expco: 'cutoff PL' })[k] || k}</option>`).join('')}</select>`;
    if (d.modes) pc += `<select data-a="mode">${d.modes.map(m => `<option>${m}</option>`).join('')}</select>`;
    if (!d.nolog) pc += `<button type="button" data-a="log">${p.log ? 'log' : 'lin'}</button>`;
    pc += `<button type="button" data-a="hide" title="Hide panel">✕</button>`;
    div.innerHTML = `<canvas></canvas><span class="tg"></span><div class="pc">${pc}</div><div class="rs" title="Drag to resize"></div>`;
    box.appendChild(div); p.el = div; p.cv = div.querySelector('canvas'); p.tg = div.querySelector('.tg');
    F_.panels.push(p); attach(F_, p);
    div.querySelector('.pc').addEventListener('change', e => {
      const a = e.target.dataset.a;
      if (a === 'det') { det[d.det] = e.target.value; figs.forEach(f => f.panels.filter(q => q.id === id).forEach(q => { const s = q.el.querySelector('[data-a=det]'); if (s) s.value = det[d.det]; })); }
      if (a === 'model') { likeModel = e.target.value; figs.forEach(f => f.el.querySelectorAll('[data-a=model]').forEach(s => s.value = likeModel)); }
      if (a === 'mode') p.mode = e.target.value;
      redraw();
    });
    div.querySelector('.pc').addEventListener('click', e => {
      const a = e.target.dataset.a;
      if (a === 'log') { p.log = !p.log; e.target.textContent = p.log ? 'log' : 'lin'; F_.draw(); }
      if (a === 'hide') { p.vis = false; tool.querySelector(`[data-p="${id}"]`).checked = false; F_.layout(); }
    });
  }
  tool.addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    if (b.dataset.v === 'full') setView(...FULL);
    if (b.dataset.v === 'flare') setView(...D.flare);
    if (b.dataset.v === 'spec') { const a = Math.min(...D.spectral.map(s => s.a)), z = Math.max(...D.spectral.map(s => s.b)), m = 0.15 * (z - a) + 10; setView(a - m, z + m); }
    if (b.dataset.v === 'bins') { const bs = D.like.models[likeModel]; const a = Math.min(...bs.map(s => s.a)), z = Math.max(...bs.map(s => s.b)), m = 0.05 * (z - a) + 10; setView(a - m, z + m); }
    if (b.dataset.z) { const c = (view.a + view.b) / 2, h = (view.b - view.a) * +b.dataset.z / 2; setView(c - h, c + h); }
    if (b.classList.contains('back') && hist.length) { const v = hist.pop(); setView(v.a, v.b, false); }
    if (b.classList.contains('fs')) { if (document.fullscreenElement) document.exitFullscreen(); else el.requestFullscreen && el.requestFullscreen(); }
    if (b.classList.contains('png')) savePNG(F_);
    if (b.classList.contains('cp')) { writeHash(); const u = location.href; (navigator.clipboard ? navigator.clipboard.writeText(u) : Promise.reject()).then(() => { b.textContent = 'Copied'; setTimeout(() => b.textContent = 'Copy link', 1500); }, () => prompt('Link to this view:', u)); }
  });
  tool.addEventListener('change', e => {
    const t = e.target;
    if (t.dataset.p) { F_.panels.find(p => p.id === t.dataset.p).vis = t.checked; F_.layout(); }
    if (t.classList.contains('vw') && t.value !== '') { const w = D.windows[+t.value], m = 0.05 * (w[1] - w[0]); setView(w[0] - m, w[1] + m); t.value = ''; }
    if (t.classList.contains('va') || t.classList.contains('vb')) setView(+tool.querySelector('.va').value, +tool.querySelector('.vb').value);
    if (t.classList.contains('lnk')) { if (t.checked) linkTable(); else { const a = document.getElementById('evtmin'), b = document.getElementById('evtmax'); a.value = b.value = ''; a.dispatchEvent(new Event('input')); } }
  });
  F_.tool = tool; F_.tip = tip;
  F_.layout = () => {
    const dpr = window.devicePixelRatio || 1, w = box.clientWidth; if (!w) return;
    const vis = F_.panels.filter(p => p.vis);
    let k = 1;
    if (document.fullscreenElement === el) { const avail = innerHeight - tool.offsetHeight - 80, tot = vis.reduce((s, p) => s + p.h + 7, 0); k = Math.max(1, avail / tot); }
    F_.panels.forEach(p => { p.el.hidden = !p.vis; p.last = p === vis[vis.length - 1]; const h = Math.round(p.h * k) + (p.last ? 18 : 0); p.w = w; p.H = h; p.dpr = dpr; p.cv.width = w * dpr; p.cv.height = h * dpr; p.cv.style.height = h + 'px'; });
    F_.draw();
  };
  F_.draw = () => F_.panels.forEach(p => p.vis && p.w && drawPanel(F_, p));
  new ResizeObserver(() => F_.layout()).observe(box);
  document.addEventListener('fullscreenchange', () => setTimeout(F_.layout, 50));
  figs.push(F_);
  return F_;
}
function syncTools() {
  for (const f of figs) {
    f.tool.querySelector('.va').value = +view.a.toFixed(1); f.tool.querySelector('.vb').value = +view.b.toFixed(1);
    const sec = view.b - view.a < 600;
    f.tool.querySelector('.utcr').textContent = `${utc(view.a, sec)} – ${utc(view.b, sec)} UTC · ${(view.b - view.a) >= 120 ? ((view.b - view.a) / 60).toFixed(1) + ' min' : (view.b - view.a).toFixed(1) + ' s'}`;
    f.tool.querySelector('.back').disabled = !hist.length;
  }
}

/* ---- drawing ---- */
function geom(p) { const x0 = ML, x1 = p.w - MR, y0 = MT, y1 = p.H - (p.last ? 24 : 6); return { x0, x1, y0, y1, xs: t => x0 + (x1 - x0) * (t - view.a) / (view.b - view.a), ts: x => view.a + (x - x0) / (x1 - x0) * (view.b - view.a) }; }
function yAxis(c, g, lo, hi, log, fmt, right) {
  let ys;
  c.fillStyle = css('--muted'); c.textAlign = right ? 'left' : 'right';
  const xl = right ? g.x1 + 6 : g.x0 - 6;
  if (log) {
    lo = Math.max(lo, 1e-30); if (hi <= lo) hi = lo * 10;
    const L0 = Math.log10(lo), L1 = Math.log10(hi);
    ys = v => g.y1 - (g.y1 - g.y0) * (Math.log10(v) - L0) / (L1 - L0);
    let e0 = Math.ceil(L0), e1 = Math.floor(L1), st = Math.max(1, Math.ceil((e1 - e0 + 1) / Math.max(2, (g.y1 - g.y0) / 22)));
    const tick = (v, e) => { const y = ys(v); c.fillText(fmt(v, e), xl, y + 4); if (!right) { c.strokeStyle = css('--line'); c.beginPath(); c.moveTo(g.x0, y + .5); c.lineTo(g.x1, y + .5); c.stroke(); } };
    if (e1 - e0 < 1) {          // less than two decades in view: label 1, 2, 5 x 10^n
      for (let e = Math.floor(L0); e <= Math.ceil(L1); e++) for (const m of [1, 2, 5]) { const v = m * 10 ** e; if (v >= lo && v <= hi) tick(v, m === 1 ? e : null); }
    } else for (let e = e0; e <= e1; e += st) tick(10 ** e, e);
  } else {
    if (hi <= lo) { hi = lo + 1; }
    ys = v => g.y1 - (g.y1 - g.y0) * (v - lo) / (hi - lo);
    const n = Math.max(2, Math.min(5, Math.floor((g.y1 - g.y0) / 26)));
    for (let i = 0; i <= n; i++) { const v = lo + (hi - lo) * i / n; c.fillText(fmt(v), xl, ys(v) + 4); }
  }
  return ys;
}
const fmtN = v => v >= 1e4 || (v < 1e-2 && v > 0) ? v.toExponential(0).replace('e+', 'e') : +v.toPrecision(3) + '';
function range(vals, log, padLin = 0.08) {
  const v = vals.filter(x => isFinite(x) && (!log || x > 0)).sort((a, b) => a - b);
  if (!v.length) return null;
  let lo = v[Math.floor(v.length * 0.005)], hi = v[v.length - 1];
  if (log) return [lo / 1.6, hi * 1.8];
  const pd = (hi - lo) * padLin || Math.abs(hi) * 0.1 || 1; return [lo - pd, hi + pd];
}
function steps(c, g, bins, ys, col, lw = 1.3) {
  c.strokeStyle = col; c.lineWidth = lw; c.beginPath(); let pen = false, last = null;
  for (const [a, b, v] of bins) {
    const y = ys(v);
    if (!isFinite(y)) { pen = false; continue; }
    const yy = Math.min(g.y1, Math.max(g.y0, y)), xa = Math.max(g.x0, g.xs(a)), xb = Math.min(g.x1, g.xs(b));
    if (!pen || (last !== null && a - last > 1e-3 + 0.5 * (b - a))) { c.moveTo(xa, yy); pen = true; } else c.lineTo(xa, yy);
    c.lineTo(xb, yy); last = b;
  }
  c.stroke(); c.lineWidth = 1;
}
function overlays(c, g, p) {
  const C = COL(), band = (a, b, col, top, h = 5) => { const xa = Math.max(g.x0, g.xs(a)), xb = Math.min(g.x1, g.xs(b)); if (xb <= xa) return null; c.fillStyle = col; if (top) c.fillRect(xa, g.y0, xb - xa, h); else c.fillRect(xa, g.y0, xb - xa, g.y1 - g.y0); return [xa, xb]; };
  D.windows.forEach(w => band(w[0], w[1], C.vis));
  if (['goes', 'nai', 'bgo'].includes(p.id) && D.data_window) {        // outside the GOES/GBM download window
    const [da, db] = D.data_window;
    for (const [a, b] of [[view.a, da], [db, view.b]]) {
      const r = b > a ? band(a, b, 'rgba(128,128,140,.12)') : null;
      if (r && r[1] - r[0] > 120) { c.fillStyle = css('--muted'); c.textAlign = 'center'; c.fillText('not downloaded', (r[0] + r[1]) / 2, (g.y0 + g.y1) / 2); }
    }
  }
  if (p.id === 'theta' && TR) { c.fillStyle = C.saa; for (let i = 0; i < TR.t.length - 1; i++) if (TR.saa[i]) { const xa = Math.max(g.x0, g.xs(TR.t[i])), xb = Math.min(g.x1, g.xs(TR.t[i + 1])); if (xb > xa) c.fillRect(xa, g.y0, xb - xa, g.y1 - g.y0); } }
  D.spectral.forEach((s, i) => {
    const r = band(s.a, s.b, C.spec); if (!r) return;
    c.strokeStyle = C.specE; c.beginPath(); c.moveTo(r[0] + .5, g.y0); c.lineTo(r[0] + .5, g.y1); c.moveTo(r[1] - .5, g.y0); c.lineTo(r[1] - .5, g.y1); c.stroke();
    if (p.first) { c.fillStyle = C.specE; c.fillRect(r[0], g.y0, Math.max(r[1] - r[0], 14), 13); c.fillStyle = '#fff'; c.font = 'bold 10px system-ui'; c.textAlign = 'left'; c.fillText('S' + (i + 1), r[0] + 3, g.y0 + 10); c.font = '11px ui-monospace,Menlo,monospace'; }
  });
  if (D.like && PANELS[p.id].like) { c.strokeStyle = css('--line'); c.setLineDash([2, 3]); for (const b of D.like.models[likeModel]) { const x = g.xs(b.a); if (x > g.x0 && x < g.x1) { c.beginPath(); c.moveTo(x + .5, g.y0); c.lineTo(x + .5, g.y1); c.stroke(); } } c.setLineDash([]); }
  if (p.id === 'lat' || PANELS[p.id].like) { if (D.latint) band(D.latint[0], D.latint[1], C.integ, true, 4); if (D.loc) { const r = band(D.loc[0], D.loc[1], C.loc, true, 4); } }
}
function xAxis(c, g, p) {
  const span = view.b - view.a, st = niceStep(span), off = ((D.t0_ms / 1000) % st + st) % st;
  c.textAlign = 'center';
  for (let t = Math.ceil((view.a + off) / st) * st - off; t <= view.b; t += st) {
    const x = g.xs(t); c.strokeStyle = css('--line'); c.beginPath(); c.moveTo(x + .5, g.y0); c.lineTo(x + .5, g.y1); c.stroke();
    if (p.last) { c.fillStyle = css('--muted'); c.fillText(utc(t, st < 60), x, g.y1 + 15); }
  }
  if (p.last) { c.textAlign = 'right'; c.fillStyle = css('--muted'); c.fillText('UTC', g.x1 + MR - 6, g.y1 + 15); }
  if (view.a < 0 && view.b > 0) { c.strokeStyle = css('--muted'); c.setLineDash([3, 3]); c.beginPath(); c.moveTo(g.xs(0) + .5, g.y0); c.lineTo(g.xs(0) + .5, g.y1); c.stroke(); c.setLineDash([]); }
}
function legend(p, items, extra) { p.tg.innerHTML = `<b>${p.d.title}</b>` + items.map(([l, col]) => `<i style="background:${col}"></i>${l}`).join('') + (extra ? ` <span style="color:var(--muted)">· ${extra}</span>` : ''); }
const binInfo = bins => { if (!bins.length) return ''; const k = bins[0][3], w = bins[0][1] - bins[0][0]; return k > 1 ? `${w >= 10 ? w.toFixed(0) : w.toFixed(1)} s bins` : 'native bins'; };
function drawPanel(F_, p) {
  const c = p.cv.getContext('2d'), g = geom(p), C = COL(), nb = Math.max(50, Math.floor((g.x1 - g.x0) / 1.5));
  p.first = F_.panels.find(q => q.vis) === p;
  c.setTransform(p.dpr, 0, 0, p.dpr, 0, 0); c.clearRect(0, 0, p.w, p.H);
  c.fillStyle = css('--surface'); c.fillRect(0, 0, p.w, p.H);
  c.font = '11px ui-monospace,Menlo,monospace'; c.lineWidth = 1;
  overlays(c, g, p); xAxis(c, g, p);
  p.cur = null;
  const id = p.id;
  if (id === 'goes') {
    const A = agg(G.ta, G.tb, G.a, null, view.a, view.b, nb), Bb = agg(G.ta, G.tb, G.b, null, view.a, view.b, nb);
    const r = range([...A, ...Bb].map(x => x[2]), p.log) || [1e-8, 1e-4];
    const ys = yAxis(c, g, p.log ? r[0] / 1.3 : r[0], p.log ? r[1] * 1.3 : r[1], p.log, (v, e) => e != null ? `${'ABCMX'[e + 8] || ''} 1e${e}` : g3(v));
    steps(c, g, A, ys, C.goesa); steps(c, g, Bb, ys, C.goesb);
    p.cur = { A, B: Bb }; legend(p, PANELS.goes.legend(C), binInfo(A));
  } else if (id === 'nai' || id === 'bgo') {
    const R = B[det[id]], series = R.c.map(cn => agg(R.ta, R.tb, cn, R.x, view.a, view.b, nb));
    const r = range(series.flat().map(x => x[2]), p.log) || [1, 10];
    const ys = yAxis(c, g, r[0], r[1], p.log, v => fmtN(v));
    series.forEach((s, i) => steps(c, g, s, ys, PAL[i % PAL.length]));
    p.cur = { series, bands: R.bands }; legend(p, R.bands.map((b, i) => [b, PAL[i % PAL.length]]), `${det[id]} · counts/s · ${binInfo(series[0] || [])}`);
  } else if (id === 'lle') {
    const s = agg(L.ta, L.tb, L.c, L.dt, view.a, view.b, nb), r = range(s.map(x => x[2]), p.log) || [0, 1];
    const ys = yAxis(c, g, p.log ? r[0] : Math.max(0, r[0]), r[1], p.log, v => fmtN(v));
    steps(c, g, s, ys, C.lle); p.cur = { s }; legend(p, PANELS.lle.legend(C), `counts/s · ${binInfo(s)}`);
  } else if (id === 'lat') {
    const i0 = lb(P.t, view.a), i1 = lb(P.t, view.b);
    let nNear = 0; for (let i = i0; i < i1; i++) nNear += P.near[i];
    if (p.mode !== 'rate') {
      let emax = 2000; for (let i = i0; i < i1; i++) emax = Math.max(emax, P.E[i]);
      const ys = yAxis(c, g, 80, emax * 1.4, p.log, v => v >= 1000 ? +(v / 1000).toPrecision(2) + ' GeV' : +v.toPrecision(2) + ' MeV');
      p.ysE = ys;
      const dense = i1 - i0 > 4000;
      for (let pass = 0; pass < 2; pass++) for (let i = i0; i < i1; i++) {
        if (P.near[i] !== pass) continue;
        c.globalAlpha = pass ? (dense ? .6 : .85) : .3; c.fillStyle = pass ? C.near : C.far;
        c.beginPath(); c.arc(g.xs(P.t[i]), ys(Math.max(P.E[i], 80)), pass ? (dense ? 1.8 : 2.6) : 1.5, 0, 7); c.fill();
      }
      c.globalAlpha = 1;
      c.strokeStyle = css('--fg'); c.lineWidth = 1; c.globalAlpha = .8;
      for (const [t, E] of D.lat.hp) { if (t < view.a || t > view.b) continue; c.beginPath(); c.arc(g.xs(t), ys(E), 4.5, 0, 7); c.stroke(); }
      c.lineWidth = 1; c.globalAlpha = 1;
      if (evMark && evMark.t >= view.a && evMark.t <= view.b) { c.strokeStyle = css('--accent'); c.lineWidth = 2.5; c.beginPath(); c.arc(g.xs(evMark.t), ys(Math.max(evMark.E, 80)), 8, 0, 7); c.stroke(); c.lineWidth = 1; }
    }
    if (p.mode !== 'photons') {
      const span = view.b - view.a, w = [1, 2, 5, 10, 20, 30, 60, 120, 300, 600].find(x => span / x <= 150) || 1200;
      const e0 = Math.floor(view.a / w) * w, n = Math.ceil((view.b - e0) / w), cnt = new Float64Array(n);
      for (let i = i0; i < i1; i++) if (P.near[i]) cnt[Math.floor((P.t[i] - e0) / w)]++;
      const bins = Array.from(cnt, (v, k) => [e0 + k * w, e0 + (k + 1) * w, v / w]);
      const both = p.mode === 'both', r = range(bins.map(x => x[2]), p.log && !both) || [0, 1];
      const ys = yAxis(c, g, both || !p.log ? 0 : r[0], r[1] * (both ? 1.15 : 1), p.log && !both, v => fmtN(v), both);
      if (both) { c.strokeStyle = css('--surface'); c.lineWidth = 5; steps(c, g, bins, ys, css('--surface'), 5); }
      steps(c, g, bins, ys, C.rate, 1.6);
      p.cur = { bins, w };
    }
    legend(p, p.mode === 'rate' ? [[`< 1.5 r68 of the Sun, ${D.lat.roi}° ROI`, C.rate]] : [['< 1.5 r68 of the Sun', C.near], [`${D.lat.roi}° ROI`, C.far]],
      `${i1 - i0} photons (${nNear} near the Sun)${p.mode !== 'photons' ? ` · rate in ${p.cur.w} s bins` : ''}${D.lat.hp.length && p.mode !== 'rate' ? ' · ○ P(Sun) ≥ 0.9 and E ≥ 1 GeV' : ''}`);
  } else if (id === 'theta') {
    const ys = v => g.y1 - (g.y1 - g.y0) * v / 180;
    c.fillStyle = css('--muted'); c.textAlign = 'right'; [0, 90, 180].forEach(v => c.fillText(v + '°', g.x0 - 6, ys(v) + 4));
    const i0 = Math.max(0, lb(TR.t, view.a) - 1), i1 = Math.min(TR.t.length, lb(TR.t, view.b) + 1);
    const line = (arr, col) => { c.strokeStyle = col; c.lineWidth = 1.3; c.beginPath(); let pen = false; for (let i = i0; i < i1; i++) { if (!isFinite(arr[i])) { pen = false; continue; } const x = g.xs(TR.t[i]), y = ys(arr[i]); if (!pen) { c.moveTo(x, y); pen = true; } else c.lineTo(x, y); } c.stroke(); c.lineWidth = 1; };
    c.save(); c.beginPath(); c.rect(g.x0, g.y0, g.x1 - g.x0, g.y1 - g.y0); c.clip();
    line(TR.th, C.trace); line(TR.ze, C.zen);
    c.setLineDash([4, 3]); c.strokeStyle = C.zen; c.beginPath(); c.moveTo(g.x0, ys(D.zmax)); c.lineTo(g.x1, ys(D.zmax)); c.stroke();
    c.strokeStyle = C.trace; c.beginPath(); c.moveTo(g.x0, ys(D.thmax)); c.lineTo(g.x1, ys(D.thmax)); c.stroke(); c.setLineDash([]); c.restore();
    legend(p, PANELS.theta.legend(C), `dashed: θ ${D.thmax}°, zenith ${D.zmax}° limits · green: Sun visibility`);
  } else if (PANELS[id].like) {
    const bins = D.like.models[likeModel], vis = bins.filter(b => b.b > view.a && b.a < view.b), lab = D.like.labels[likeModel];
    let vals, ys, unit;
    if (id === 'flux') vals = vis.flatMap(b => b.flux ? [b.flux[0], b.flux[1], b.flux[2]] : b.ul ? [b.ul, b.ul / 3] : []);
    if (id === 'par') vals = vis.flatMap(b => b.par ? [b.par[0] + b.par[1], b.par[0] + b.par[2]] : []);
    if (id === 'ts') vals = vis.map(b => Math.max(b.TS, 0.1)).concat([D.like.TS_MIN]);
    const r = range(vals, p.log, 0.12);
    if (r) {
      ys = yAxis(c, g, r[0], r[1], p.log, v => fmtN(v));
      if (id === 'ts') { c.strokeStyle = C.flux; c.setLineDash([5, 4]); const y = ys(D.like.TS_MIN); c.beginPath(); c.moveTo(g.x0, y); c.lineTo(g.x1, y); c.stroke(); c.setLineDash([]); }
      const col = id === 'flux' ? C.flux : id === 'par' ? C.par : C.ts;
      c.save(); c.beginPath(); c.rect(g.x0, g.y0, g.x1 - g.x0, g.y1 - g.y0); c.clip();
      for (const b of vis) {
        const xa = g.xs(b.a), xb = g.xs(b.b), xm = (xa + xb) / 2;
        let v = null, lo, hi, ul = false;
        if (id === 'flux') { if (b.flux) [v, lo, hi] = b.flux; else if (b.ul) { v = b.ul; ul = true; } }
        if (id === 'par' && b.par) { v = b.par[0]; lo = v + b.par[1]; hi = v + b.par[2]; }
        if (id === 'ts') { v = Math.max(b.TS, 0.1); lo = hi = v; }
        if (v == null || !isFinite(ys(v))) continue;
        const y = ys(v), cc = ul ? css('--muted') : col;
        c.strokeStyle = cc; c.fillStyle = cc; c.lineWidth = 1.4;
        c.beginPath(); c.moveTo(xa, y); c.lineTo(xb, y); c.stroke();
        if (ul) { c.beginPath(); c.moveTo(xm, y); c.lineTo(xm, y + 16); c.stroke(); c.beginPath(); c.moveTo(xm - 4, y + 11); c.lineTo(xm, y + 17); c.lineTo(xm + 4, y + 11); c.fill(); }
        else { if (lo !== hi) { c.beginPath(); c.moveTo(xm, ys(Math.max(lo, p.log ? 1e-30 : -1e30))); c.lineTo(xm, ys(hi)); c.stroke(); }
          c.beginPath(); id === 'ts' ? c.rect(xm - 3, y - 3, 6, 6) : c.arc(xm, y, 3.2, 0, 7); c.fill(); if (b.noerr && id !== 'ts') { c.fillStyle = css('--surface'); c.beginPath(); c.arc(xm, y, 1.6, 0, 7); c.fill(); }
          if (b.lim && id === 'par') { c.strokeStyle = '#d97706'; c.lineWidth = 2; c.beginPath(); c.moveTo(xm, y - 11); c.lineTo(xm + 7, y + 1); c.lineTo(xm - 7, y + 1); c.closePath(); c.stroke(); c.lineWidth = 1.4; } }
      }
      c.restore(); c.lineWidth = 1;
    } else { c.fillStyle = css('--muted'); c.textAlign = 'center'; c.fillText(id === 'par' ? 'no detected bin in view (or parameter fixed)' : 'no LAT bin in view', (g.x0 + g.x1) / 2, (g.y0 + g.y1) / 2); }
    const L_ = { flux: [[`${D.like.band[0]}–${D.like.band[1]} MeV flux [ph cm⁻² s⁻¹]`, C.flux], ['95% UL', css('--muted')]], par: [[lab[1], C.par]], ts: [['TS', C.ts], [`detection TS ${D.like.TS_MIN}`, C.flux]] }[id];
    legend(p, L_, `${lab[2]} · ${({ bb: 'Bayesian-block', windows: 'visibility-window', const: 'constant-counts', custom: 'custom' })[D.like.method] || D.like.method} bins`);
  }
  c.strokeStyle = css('--muted'); c.strokeRect(g.x0 + .5, g.y0 + .5, g.x1 - g.x0, g.y1 - g.y0);
  if (sel && sel.f === F_) { const xa = g.xs(Math.min(sel.a, sel.b)), xb = g.xs(Math.max(sel.a, sel.b)); c.fillStyle = 'rgba(120,120,140,.2)'; c.fillRect(xa, g.y0, xb - xa, g.y1 - g.y0); }
  if (hoverT != null) { const x = g.xs(hoverT); if (x >= g.x0 && x <= g.x1) { c.strokeStyle = css('--accent'); c.beginPath(); c.moveTo(x + .5, g.y0); c.lineTo(x + .5, g.y1); c.stroke(); } }
}

/* ---- readout ---- */
const inBin = (bins, t) => bins && bins.find(b => t >= b[0] && t < b[1]);
function readout(F_, t, px) {
  const out = [`<b>${utc(t, true)} UTC</b>  T0 ${sgn(t)} s`];
  for (const p of F_.panels) {
    if (!p.vis) continue;
    const cu = p.cur;
    if (p.id === 'goes' && cu) { const a = inBin(cu.A, t), b = inBin(cu.B, t); if (b) out.push(`GOES   B ${g3(b[2])} W/m² (${goesClass(b[2])})  A ${g3(a ? a[2] : NaN)}`); }
    if ((p.id === 'nai' || p.id === 'bgo') && cu) out.push(`${det[p.id]} [c/s]` + cu.series.map((s, i) => { const b = inBin(s, t); return (i % 2 ? '   ' : '\n  ') + `${cu.bands[i]}: ${b ? g3(b[2]) : '–'}`; }).join(''));
    if (p.id === 'lle' && cu) { const b = inBin(cu.s, t); if (b) out.push(`LLE    ${g3(b[2])} c/s`); }
    if (p.id === 'lat') {
      if (cu && cu.bins) { const b = inBin(cu.bins, t); if (b) out.push(`LAT    ${g3(b[2] * cu.w)} photons near the Sun in ${cu.w} s`); }
      if (p.mode !== 'rate') { const dt = 6 * (view.b - view.a) / (p.w - ML - MR), i0 = lb(P.t, t - dt), i1 = lb(P.t, t + dt); let best = -1; for (let i = i0; i < i1; i++) if (best < 0 || P.E[i] > P.E[best]) best = i; if (best >= 0) out.push(`photon ${P.E[best] >= 1000 ? (P.E[best] / 1000).toFixed(2) + ' GeV' : P.E[best] + ' MeV'} at ${sgn(P.t[best])} s${P.near[best] ? ' (near the Sun)' : ''}`); }
    }
    if (p.id === 'theta' && TR) { const i = Math.min(TR.t.length - 1, Math.max(0, lb(TR.t, t))); if (Math.abs(TR.t[i] - t) < 60) out.push(`Sun    θ ${g3(TR.th[i])}°  zenith ${g3(TR.ze[i])}°${TR.saa[i] ? '  SAA' : ''}`); }
  }
  if (D.like && F_.panels.some(p => p.vis && PANELS[p.id].like)) {
    const b = D.like.models[likeModel].find(b => t >= b.a && t < b.b), lab = D.like.labels[likeModel];
    if (b) out.push(`<b>LAT bin ${b.a.toFixed(0)}–${b.b.toFixed(0)} s</b>  ${lab[2]}\n  TS ${b.TS.toFixed(1)}  ` + (b.flux ? `flux ${g3(b.flux[0])} (${g3(b.flux[1])}–${g3(b.flux[2])})` : `flux < ${g3(b.ul)} (95% UL)`) +
      (b.par ? `\n  ${lab[0]} = ${g3(b.par[0])} −${g3(Math.abs(b.par[1]))} +${g3(b.par[2])}` : '') + (b.best ? '  ★ best model' : '') + (b.noerr ? '\n  no errors: parameters at their bounds' : '') + (b.lim ? `\n  ⚠ ${b.lim}` : ''));
  }
  D.spectral.forEach((s, i) => { if (t >= s.a && t <= s.b) out.push(`<b>Spectral interval S${i + 1}</b>  ${s.a}–${s.b} s\n  ` + (s.best ? `best model ${s.best}` + (s.pion != null ? `\n  adding pion decay: TS = ${s.pion.toFixed(1)}` : '') : 'not fitted yet')); });
  D.windows.forEach((w, i) => { if (t >= w[0] && t <= w[1]) out.push(`LAT visibility window #${i + 1}`); });
  return out.join('\n');
}

/* ---- mouse ---- */
let sel = null;
function attach(F_, p) {
  const tOf = ev => { const r = p.cv.getBoundingClientRect(); return geom(p).ts(ev.clientX - r.left); };
  let down = null;
  p.cv.addEventListener('pointerdown', ev => {
    if (ev.button !== 0) return; p.cv.setPointerCapture(ev.pointerId);
    down = { x: ev.clientX, y: ev.clientY, t: tOf(ev), pan: ev.shiftKey, v: { ...view } };
    if (!down.pan) sel = { f: F_, a: down.t, b: down.t };
  });
  p.cv.addEventListener('pointermove', ev => {
    const t = tOf(ev);
    if (down && down.pan) { const dt = (ev.clientX - down.x) / (p.w - ML - MR) * (down.v.b - down.v.a); view = { a: down.v.a - dt, b: down.v.b - dt }; redraw(); syncTools(); }
    else if (down) sel.b = t;
    hoverT = t; redraw();
    const tip = F_.tip; tip.innerHTML = readout(F_, t); tip.hidden = false;
    const r = tip.getBoundingClientRect(); let x = ev.clientX + 16, y = ev.clientY + 14;
    if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - 16; if (y + r.height > innerHeight - 8) y = innerHeight - r.height - 8;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  });
  p.cv.addEventListener('pointerleave', () => { hoverT = null; F_.tip.hidden = true; if (!down) redraw(); });
  p.cv.addEventListener('pointerup', ev => {
    if (!down) return; const d = down; down = null;
    const moved = Math.abs(ev.clientX - d.x) > 4;
    if (d.pan && !moved && PANELS[p.id].like && window.openCounts) {     // shift-click on a LAT bin: its count spectrum
      view = d.v; const b = D.like.models[likeModel].find(b => d.t >= b.a && d.t < b.b); if (b) window.openCounts(b.key, likeModel); return;
    }
    if (d.pan) { const v = { ...view }; view = d.v; setView(v.a, v.b); return; }
    const s = sel; sel = null;
    if (moved) { setView(Math.min(s.a, s.b), Math.max(s.a, s.b)); return; }
    redraw(); click(F_, p, ev, d.t);
  });
  p.cv.addEventListener('dblclick', () => setView(...FULL));
  p.cv.addEventListener('wheel', ev => {
    if (!(ev.ctrlKey || ev.metaKey || ev.altKey)) return;
    ev.preventDefault();
    const t = tOf(ev), k = Math.exp(Math.max(-0.5, Math.min(0.5, ev.deltaY * 0.01)));
    setView(t - (t - view.a) * k, t + (view.b - t) * k);
  }, { passive: false });
  const rs = p.el.querySelector('.rs');
  rs.addEventListener('pointerdown', ev => {
    rs.setPointerCapture(ev.pointerId); const y0 = ev.clientY, h0 = p.h;
    const mv = e => { p.h = Math.max(50, Math.min(800, h0 + e.clientY - y0)); F_.layout(); };
    const up = () => { rs.removeEventListener('pointermove', mv); rs.removeEventListener('pointerup', up); };
    rs.addEventListener('pointermove', mv); rs.addEventListener('pointerup', up);
  });
}
function openTab(t) { const b = document.querySelector(`nav button[data-t="${t}"]`); if (b) b.click(); }
function click(F_, p, ev, t) {
  const g = geom(p), r = p.cv.getBoundingClientRect(), y = ev.clientY - r.top;
  if (p.first && y < g.y0 + 15) {
    const s = D.spectral.find(s => t >= s.a && t <= s.b && s.id);
    if (s) { openTab('spec'); const el = document.getElementById(s.id); if (el) { el.scrollIntoView({ behavior: 'smooth', block: 'start' }); } return; }
  }
  if (PANELS[p.id].like) {
    const b = D.like.models[likeModel].find(b => t >= b.a && t < b.b);
    if (b) { openTab('lat'); const row = document.querySelector(`tr[data-bin="${b.key}"]`); if (row) { row.scrollIntoView({ behavior: 'smooth', block: 'center' }); document.querySelectorAll(`tr[data-bin="${b.key}"]`).forEach(r => { r.classList.remove('flash'); void r.offsetWidth; r.classList.add('flash'); }); } }
  }
}

/* ---- PNG export ---- */
function savePNG(F_) {
  const vis = F_.panels.filter(p => p.vis), dpr = vis[0].dpr, W = vis[0].w, head = 30;
  const H = head + vis.reduce((s, p) => s + p.H, 0);
  const cv = document.createElement('canvas'); cv.width = W * dpr; cv.height = H * dpr;
  const c = cv.getContext('2d'); c.scale(dpr, dpr); c.fillStyle = css('--surface'); c.fillRect(0, 0, W, H);
  c.fillStyle = css('--fg'); c.font = '600 14px system-ui'; c.textAlign = 'left';
  const sec = view.b - view.a < 600;
  c.fillText(`${D.name} — ${F_.el.dataset.title}   ${utc(view.a, sec)}–${utc(view.b, sec)} UTC (T0 ${sgn(view.a)} to ${sgn(view.b)} s)`, ML, 20);
  let y = head;
  for (const p of vis) { c.drawImage(p.cv, 0, y, p.w, p.H); c.fillStyle = css('--fg'); c.font = '600 11px system-ui'; c.fillText(p.tg.textContent.slice(0, 140), ML + 6, y + 17); y += p.H; }
  const a = document.createElement('a'); a.download = `${D.name}_${F_.el.id}_${Math.round(view.a)}_${Math.round(view.b)}.png`;
  a.href = cv.toDataURL('image/png'); document.body.appendChild(a); a.click(); a.remove();
}

/* ---- events table hover marks the photon ---- */
const evt = document.getElementById('evtable');
if (evt) {
  evt.addEventListener('mouseover', e => { const tr = e.target.closest('tr[data-t]'); if (!tr) return; evMark = { t: +tr.dataset.t, E: +tr.dataset.e }; redraw(); });
  evt.addEventListener('mouseleave', () => { evMark = null; redraw(); });
}
window.addEventListener('beforeprint', () => document.querySelectorAll('details.static').forEach(d => d.open = true));
document.querySelectorAll('.ifig').forEach(el => Figure(el));
syncTools();
})();
'''
