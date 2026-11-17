#!/usr/bin/env python
"""Collect all results of one flare into a self-contained results/report.html.

Usage: python -m sunflare.make_report flares/<id>/config_<id>.json
"""
import base64
import datetime
import glob
import html
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from sunflare import common  # noqa: E402

cfg = common.load_config(sys.argv[1])
info, fdir = cfg.info, cfg.pwd
R = os.path.join(fdir, 'results')
E = html.escape


def img(path, alt, cls='fig'):
    if not path or not os.path.exists(path):
        return ''
    with open(path, 'rb') as f:
        b = base64.b64encode(f.read()).decode()
    return f'<figure class="{cls}"><img alt="{E(alt)}" src="data:image/png;base64,{b}"><figcaption>{E(alt)}</figcaption></figure>'


def fmt(v, p=3):
    if v is None:
        return '–'
    try:
        v = float(v)
    except (TypeError, ValueError):
        return E(str(v))
    if v != v:
        return '–'
    if v != 0 and (abs(v) < 1e-2 or abs(v) >= 1e5):
        m, e = f'{v:.{p - 1}e}'.split('e')
        return f'{m}×10<sup>{int(e)}</sup>'
    return f'{v:.{p}g}'


def pm(v, lo, hi, p=3):
    return f'{fmt(v, p)} <span class="err">−{fmt(abs(lo), 2)} +{fmt(abs(hi), 2)}</span>'


def pmb(v, lo, hi, p=3):
    """value with [low, high] bounds"""
    return pm(v, v - lo, hi - v, p)


def utc(t):
    return common.met2utc(info['T0'] + float(t))[11:19]


def table(head, rows, cls=''):
    h = ''.join(f'<th>{x}</th>' for x in head)
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<div class="tablebox"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


sections = []
EVENTS = []

# ------------------------------------------------------------------ summary
lc = common.load_json(os.path.join(R, 'lightcurves', 'lightcurves.json'), {})
loc = common.load_json(os.path.join(R, 'localization', 'localization.json'))
lat = common.load_json(os.path.join(R, 'lat', 'lat_summary.json'))
facts = [('Flare', cfg.name), ('GOES class', info.get('goes_class', '–')),
         ('T0 (UTC)', info.get('t0_utc')), ('MET of T0', f'{info["T0"]:.3f}'),
         ('GOES peak', f'{utc(info["goes_peak_t"])} UTC (T0{info["goes_peak_t"]:+.0f} s)' if info.get('goes_peak_t') is not None else '–'),
         ('Active region', info.get('ar') or '–'), ('Location', info.get('location') or '–'),
         ('Sun RA, Dec at T0', f'{info["sun_ra"]:.3f}°, {info["sun_dec"]:.3f}°'),
         ('GBM triggers', ', '.join(t['name'] for t in info.get('gbm_triggers', [])) or 'none'),
         ('LLE', info.get('lle_trigger', 'not available') if info.get('has_lle') else 'not available'),
         ('LAT frame', 'Sun-centred (FermiSunPy)' if cfg.sun_centered else 'equatorial (J2000)'),
         ('LAT event class', cfg.irfs)]
key = []
if lat and lat.get('integrated', {}).get('fits'):
    integ = lat['integrated']
    m = 'pion' if 'pion' in integ['fits'] else list(integ['fits'])[0]
    f = integ['fits'][m]
    if 'flux' in f:
        key.append(f'LAT time-integrated ({integ["t0"]:g}–{integ["t1"]:g} s, {common.LAT_MODELS[m].lower()}): '
                   f'TS = {f["TS"]:.0f}, flux {lat["flux_band_MeV"][0]:g}–{lat["flux_band_MeV"][1]:g} MeV = '
                   f'{pmb(*f["flux"])} ph cm<sup>−2</sup> s<sup>−1</sup>' +
                   (f', proton index s = {pm(*f["params"]["s"])}' if 's' in f.get('params', {}) else ''))
if loc:
    key.append(f'LAT position {loc["offset_deg"]:.2f}° from the Sun centre (r68 = {loc["r68_deg"]:.2f}°, '
               f'helioprojective {loc["hpc_arcsec"][0]:.0f}″, {loc["hpc_arcsec"][1]:.0f}″): '
               + ('consistent with the solar disk' if loc['offset_deg'] - loc['r95_deg'] <= 0.2666
                  else 'outside the solar disk at 95% c.l.') + '.'
               + (f' gtfindsrc: {loc["gtfindsrc"]["ra"]:.3f}°, {loc["gtfindsrc"]["dec"]:.3f}° '
                  f'(±{loc["gtfindsrc"]["err_deg"]:.3f}°).' if 'ra' in (loc.get('gtfindsrc') or {}) else ''))
for s in sorted(glob.glob(os.path.join(R, 'spectrum_*', 'summary.json'))):
    S = common.load_json(s)
    if S and S.get('best'):
        piontest = [n for n in S['nested'] if n['component'] == 'pion']
        extra = f'; adding pion decay: TS = {max(n["TS"] for n in piontest):.1f}' if piontest else ''
        key.append(f'Joint fit {S["interval"][0]:g}–{S["interval"][1]:g} s: best model (BIC) <code>{E(S["best"])}</code>{extra}')
summary_html = '<div class="facts">' + ''.join(f'<div><span>{E(k)}</span><b>{E(str(v))}</b></div>' for k, v in facts) + '</div>'
if key:
    summary_html += '<h3>Key results</h3><ul class="key">' + ''.join(f'<li>{k}</li>' for k in key) + '</ul>'
if cfg.notes:
    summary_html += f'<h3>Notes</h3><p>{E(cfg.notes)}</p>'
if info.get('warnings'):
    summary_html += '<h3>Data warnings</h3><ul class="warn">' + ''.join(f'<li>{E(w)}</li>' for w in info['warnings']) + '</ul>'
sections.append(('summary', 'Summary', summary_html))

# ------------------------------------------------------------------ light curves
h = img(os.path.join(R, 'lightcurves', 'zoom.png'), 'GOES, GBM, LAT light curves around the flare (yellow: spectral intervals, green: LAT Sun visibility)')
h += img(os.path.join(R, 'lightcurves', 'overview.png'), 'Full data window')
if lc.get('peaks'):
    h += '<h3>Peak times</h3>' + table(['Band', 'Peak (UTC)', 'T − T0 [s]'],
                                       [[E(k), utc(v), f'{v:.0f}'] for k, v in lc['peaks'].items()])
if info.get('lat_windows'):
    h += '<h3>LAT Sun visibility windows</h3>' + table(
        ['#', 'From (UTC)', 'To (UTC)', 'T − T0 [s]', 'Duration'],
        [[i + 1, utc(a), utc(b), f'{a:.0f} – {b:.0f}', f'{(b - a) / 60:.1f} min'] for i, (a, b) in enumerate(info['lat_windows'])])
bk = sorted(glob.glob(os.path.join(R, 'background', 'bkg_*.png')))
if bk:
    h += '<details><summary>GBM background fits</summary>' + ''.join(img(p, os.path.basename(p)[4:-4], 'fig small') for p in bk) + '</details>'
sections.append(('lc', 'Light curves', h))

# ------------------------------------------------------------------ localization
if loc:
    h = img(os.path.join(R, 'localization', 'tsmap.png'), 'LAT TS map (left: sky offsets from the Sun; right: helioprojective, gold circle: solar disk)')
    rows = [['Interval', f'{loc["interval"][0]:g} – {loc["interval"][1]:g} s ({utc(loc["interval"][0])}–{utc(loc["interval"][1])} UTC)'],
            ['Energy', f'> {loc["emin_MeV"]:g} MeV'], ['Maximum TS', f'{loc["TS_max"]:.1f}'],
            ['Best RA, Dec (J2000)', f'{loc["best_radec"][0]:.3f}°, {loc["best_radec"][1]:.3f}°'],
            ['Sun RA, Dec (mid interval)', f'{loc["sun_radec"][0]:.3f}°, {loc["sun_radec"][1]:.3f}°'],
            ['Error radius 68% / 95%', f'{loc["r68_deg"]:.3f}° / {loc["r95_deg"]:.3f}°'],
            ['Offset from Sun centre', f'{loc["offset_deg"]:.3f}° = {loc["offset_rsun"]:.2f} R<sub>☉</sub>'],
            ['Helioprojective (Tx, Ty)', f'{loc["hpc_arcsec"][0]:.0f}″, {loc["hpc_arcsec"][1]:.0f}″'],
            ['Frame used', loc['frame']]]
    g = loc.get('gtfindsrc') or {}
    if 'ra' in g:
        rows.append(['gtfindsrc (GtBurst)', f'RA, Dec = {g["ra"]:.3f}°, {g["dec"]:.3f}°, error {g["err_deg"]:.3f}° '
                     f'(TS = {g["TS"]:.0f})'])
    h += table(['', ''], rows, 'kv')
    sections.append(('loc', 'Localization', h))

# ------------------------------------------------------------------ LAT likelihood
if lat:
    h = img(os.path.join(R, 'lat', 'lat_lightcurve.png'), 'LAT time-resolved likelihood: flux, spectral parameter and TS')
    band = f'{lat["flux_band_MeV"][0]:g}–{lat["flux_band_MeV"][1]:g} MeV'

    def rows_for(rs):
        out = []
        for r in rs:
            for k, f in r['fits'].items():
                if 'TS' not in f:
                    out.append([f'{utc(r["t0"])}–{utc(r["t1"])}', E(k), '–', '–', E(f.get('error', 'failed')), '–'])
                    continue
                par = '; '.join(f'{p} = {pm(*v)}' for p, v in f['params'].items())
                if f['TS'] >= lat['TS_MIN']:
                    flux = pmb(*f['flux'])
                else:
                    flux = f'&lt; {fmt(f.get("flux_ul"))} <span class="err">(95% UL)</span>'
                out.append([f'{utc(r["t0"])}–{utc(r["t1"])}<br><span class="err">{r["t0"]:.0f}–{r["t1"]:.0f} s</span>',
                            E(k) + (' ★' if r.get('best') == k else ''), f'{f["TS"]:.1f}', flux, par, f'{f["logL"]:.2f}'])
        return out
    head = ['Interval (UTC)', 'Model', 'TS', f'Flux {band} [ph cm⁻² s⁻¹]', 'Parameters', 'log L']
    h += '<h3>Time-integrated</h3>' + table(head, rows_for([lat['integrated']]))
    h += img(os.path.join(R, 'lat', 'lat_integrated_sed.png'), 'LAT time-integrated spectral models', 'fig half')
    for k in lat['models']:
        h += img(os.path.join(R, 'lat', f'lat_integrated_counts_{k}.png'), f'Count spectrum, {common.LAT_MODELS[k]}', 'fig half')
    h += f'<h3>Time-resolved ({E(lat["method"])} bins)</h3>' + table(head, rows_for(lat['resolved']))
    from sunflare.lat_events import load_or_extract
    EVENTS = load_or_extract(cfg, log=lambda *a: print(*a, flush=True)) or []
    if EVENTS:
        it = lat['integrated']
        h += (f'<h3 id="evblock">Events with source probability</h3><p class="muted">Events of the time-integrated '
              f'interval ({it["t0"]:g}–{it["t1"]:g} s) from the <code>*_filt_prob.fit</code> file written by '
              '<code>gtsrcprob</code>: P(Sun), P(Galactic) and P(isotropic) are the probabilities that each event comes '
              'from that model component. Distance from the Sun centre at the event time; 68% PSF radius from the '
              f'{E(cfg.irfs)} IRFs at the event energy, off-axis angle and conversion type. Click a header to sort.</p>'
              '<div class="evctl"><label>P(Sun) ≥ <input type="number" id="evpmin" value="0" min="0" max="1" step="0.05"></label>'
              '<label>E ≥ <input type="number" id="evemin" value="0" min="0" step="any"> MeV</label>'
              '<label>T − T0 from <input type="number" id="evtmin" step="any" placeholder="min"> to '
              '<input type="number" id="evtmax" step="any" placeholder="max"> s</label>'
              '<span class="muted" id="evcount"></span>'
              '<button type="button" class="evbtn" id="evcsv">Download CSV</button></div>'
              '<div class="tablebox evbox"><table id="evtable"></table></div>')
    sections.append(('lat', 'LAT likelihood', h))

# ------------------------------------------------------------------ joint spectra
spec_h = ''
for s in sorted(glob.glob(os.path.join(R, 'spectrum_*', 'summary.json')), key=lambda p: common.load_json(p)['interval'][0]):
    S = common.load_json(s)
    d = os.path.dirname(s)
    a, b = S['interval']
    spec_h += f'<section class="spec"><h3>{a:g} – {b:g} s &nbsp;<span class="muted">({utc(a)} – {utc(b)} UTC)</span></h3>'
    spec_h += f'<p class="muted">Data: {E(", ".join(S["plugins"]))}</p>'
    for p in S.get('problems', []):
        spec_h += f'<p class="warnline">{E(p)}</p>'
    if S['fits']:
        best_bic = min(v['BIC'] for v in S['fits'].values())
        rows = []
        for n, v in sorted(S['fits'].items(), key=lambda kv: kv[1]['BIC']):
            rows.append([f'<code>{E(n)}</code>' + (' ★' if n == S['best'] else '') +
                         (' <span class="err">(no errors: parameters at bounds)</span>' if v.get('no_errors') else ''), f'{-v["logL"]:.2f}', v['n_free'],
                         f'{v["AIC"]:.1f}', f'{v["BIC"]:.1f}', f'{v["BIC"] - best_bic:.1f}'])
        spec_h += table(['Model', '−log L', 'free par.', 'AIC', 'BIC', 'ΔBIC'], rows)
        if S['nested']:
            spec_h += '<h4>Component significance (likelihood ratio)</h4>' + table(
                ['Added component', 'to model', 'TS = 2ΔlogL'],
                [[f'<code>{E(n["component"])}</code>', f'<code>{E(n["base"])}</code>', f'{n["TS"]:.1f}'] for n in S['nested']])
        spec_h += img(os.path.join(d, 'vfv.png'), 'νFν of the fitted models (best model with its components)', 'fig half')
        for n, v in S['fits'].items():
            spec_h += f'<details><summary><code>{E(n)}</code> — parameters, fluxes and count spectrum</summary>'
            spec_h += table(['Parameter', 'Value', 'Unit'], [[f'<code>{E(p[0])}</code>', pm(p[1], p[2], p[3]), E(p[4])] for p in v['params']])
            if v.get('lines'):
                spec_h += table(['Line', 'Flux [ph cm⁻² s⁻¹]'], [[E(common.COMPONENTS[k]['label']), pm(*x)] for k, x in v['lines'].items()])
            bands = list(next(iter(v['components'].values())).keys()) if v['components'] else []
            if bands:
                spec_h += table(['Component (photon flux, ph cm⁻² s⁻¹)'] + bands,
                                [[E(c)] + [fmt(x[bd]) for bd in bands] for c, x in v['components'].items()] +
                                ([['<b>total</b>'] + [pmb(*v['total_flux'][bd]) if bd in v.get('total_flux', {}) else '–' for bd in bands]]))
            spec_h += img(os.path.join(d, f'counts_{n}.png'), f'Count spectra and residuals: {n}', 'fig half')
            spec_h += '</details>'
    for n, why in S.get('failed', {}).items():
        spec_h += f'<p class="warnline">{E(n)}: {E(why)}</p>'
    spec_h += '</section>'
if spec_h:
    sections.append(('spec', 'Joint spectral fits', spec_h))

# ------------------------------------------------------------------ config
cfg_public = {k: v for k, v in cfg.items() if k not in ('info',)}
sections.append(('cfg', 'Configuration', '<pre>' + E(json.dumps(cfg_public, indent=1, default=str)) + '</pre>'))

# ------------------------------------------------------------------ page
nav = ''.join(f'<button role="tab" data-t="{k}" aria-selected="{str(i == 0).lower()}">{E(t)}</button>' for i, (k, t, _) in enumerate(sections))
body = ''.join(f'<section class="tab" id="t-{k}" {"" if i == 0 else "hidden"}><h2>{E(t)}</h2>{c}</section>'
               for i, (k, t, c) in enumerate(sections))
page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{E(cfg.name)} Fermi report</title>
<style>
:root {{ --bg:#f4f5f8; --surface:#fff; --raise:#fafbfd; --fg:#17202c; --muted:#5b6577; --line:#dde1e9; --accent:#c2410c; --accent-soft:#fdeee5; --warn:#a2611a; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0f131a; --surface:#171c25; --raise:#1c2230; --fg:#e6e9f0; --muted:#9aa3b5; --line:#2b3343; --accent:#fb923c; --accent-soft:#3a2516; --warn:#e0a35b; }} }}
:root[data-theme="dark"] {{ --bg:#0f131a; --surface:#171c25; --raise:#1c2230; --fg:#e6e9f0; --muted:#9aa3b5; --line:#2b3343; --accent:#fb923c; --accent-soft:#3a2516; --warn:#e0a35b; }}
* {{ box-sizing:border-box }} [hidden] {{ display:none !important }} body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif }}
header.top {{ padding:22px max(16px,4vw) 10px; display:grid; gap:4px }} header.top h1 {{ margin:0; font-size:24px }}
.muted, .err {{ color:var(--muted) }} .err {{ font-size:12px }}
nav {{ position:sticky; top:0; z-index:5; background:var(--bg); padding:8px max(16px,4vw); display:flex; gap:6px; flex-wrap:wrap; border-bottom:1px solid var(--line) }}
nav button {{ font:500 14px system-ui; border:1px solid var(--line); background:var(--surface); color:var(--fg); padding:6px 13px; border-radius:99px; cursor:pointer }}
nav button[aria-selected="true"] {{ background:var(--accent); border-color:var(--accent); color:#fff }}
main {{ padding:18px max(16px,4vw) 60px; max-width:1400px }}
.tab {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:20px; display:grid; gap:14px }}
.tab h2 {{ margin:0; font-size:20px }} h3 {{ margin:10px 0 0; font-size:16px }} h4 {{ margin:6px 0 0; font-size:14px; color:var(--muted) }}
.facts {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:1px; background:var(--line); border:1px solid var(--line); border-radius:8px; overflow:hidden }}
.facts div {{ background:var(--raise); padding:10px 12px; display:grid }} .facts span {{ font-size:12px; color:var(--muted) }} .facts b {{ font-weight:600; overflow-wrap:anywhere }}
ul.key li {{ margin:4px 0 }} ul.warn, .warnline {{ color:var(--warn) }}
.tablebox {{ overflow-x:auto; border:1px solid var(--line); border-radius:8px }}
table {{ border-collapse:collapse; width:100%; font-size:14px }} th,td {{ padding:6px 11px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; font-variant-numeric:tabular-nums }}
th {{ background:var(--raise); font-size:12px; color:var(--muted) }} tbody tr:last-child td {{ border-bottom:0 }}
table.kv td:first-child {{ color:var(--muted); width:240px }}
figure {{ margin:0; display:grid; gap:4px }} figure img {{ width:100%; height:auto; background:#fff; border-radius:6px; border:1px solid var(--line) }}
figcaption {{ font-size:13px; color:var(--muted) }} figure.half {{ max-width:760px }} figure.small {{ max-width:520px; display:inline-grid; margin:6px }}
details {{ border:1px solid var(--line); border-radius:8px; padding:8px 12px; background:var(--raise) }} details[open] {{ display:grid; gap:10px }}
summary {{ cursor:pointer }} section.spec {{ display:grid; gap:10px; padding-bottom:16px; border-bottom:1px dashed var(--line) }}
pre {{ background:var(--raise); border:1px solid var(--line); border-radius:8px; padding:12px; overflow:auto; font-size:12px }}
code {{ font-family:ui-monospace,Menlo,monospace; font-size:13px }}
th.sort {{ cursor:pointer; user-select:none; white-space:nowrap }} th.sort:hover {{ color:var(--fg) }}
th.sort[aria-sort="ascending"]::after {{ content:" ▲"; color:var(--accent) }} th.sort[aria-sort="descending"]::after {{ content:" ▼"; color:var(--accent) }}
.evbox {{ max-height:560px; overflow:auto }} .evbox thead th {{ position:sticky; top:0; z-index:1 }}
.evbox td {{ white-space:nowrap }} .evbox td.num {{ font-family:ui-monospace,Menlo,monospace; font-size:13px; text-align:right }}
.pbar {{ display:inline-grid; grid-template-columns:52px 44px; gap:6px; align-items:center }}
.pbar i {{ height:6px; border-radius:3px; background:var(--line); position:relative; overflow:hidden }}
.pbar i b {{ position:absolute; inset:0 auto 0 0; background:var(--accent) }}
tr.hi td {{ background:var(--accent-soft) }}
.evctl {{ display:flex; flex-wrap:wrap; gap:10px 16px; align-items:center; font-size:14px }}
.evctl input[type=number] {{ width:84px; font:13px ui-monospace,Menlo,monospace; padding:4px 6px; border:1px solid var(--line); border-radius:6px; background:var(--surface); color:var(--fg) }}
.evbtn {{ font:500 13px system-ui; padding:5px 12px; border:1px solid var(--line); border-radius:6px; background:var(--surface); color:var(--fg); cursor:pointer; margin-left:auto }}
</style></head><body>
<header class="top"><span class="muted">Fermi solar-flare analysis report</span><h1>{E(cfg.name)} &nbsp;{E(info.get("goes_class", ""))}</h1>
<span class="muted">Generated {datetime.datetime.now():%Y-%m-%d %H:%M} · T0 = {E(info.get("t0_utc", ""))} UTC · FermiSunPy / threeML</span></header>
<nav role="tablist">{nav}</nav><main>{body}</main>
<script>
document.querySelectorAll('nav button').forEach(b => b.addEventListener('click', () => {{
  document.querySelectorAll('nav button').forEach(x => x.setAttribute('aria-selected', x === b));
  document.querySelectorAll('.tab').forEach(s => s.hidden = s.id !== 't-' + b.dataset.t);
  try {{ history.replaceState(null, '', '#' + b.dataset.t); }} catch (e) {{}}
}}));
EVJS
const h = location.hash.slice(1); const tb = document.querySelector(`nav button[data-t="${{h}}"]`); if (tb) tb.click();
</script></body></html>'''
EVJS = r'''
/* LAT events (gtsrcprob): filters + sortable table, as in the GRBAnalyzer report */
const EV = EVDATA;
const $e = s => document.querySelector(s);
const f3 = (v, n) => v == null ? '–' : v.toFixed(n);
const EVCOLS = [
  ['bin', 'Bin', r => r.bin || '–', 'txt'], ['utc', 'UTC', r => r.utc, 'txt'], ['t', 'T − T0 (s)', r => r.t.toFixed(2)],
  ['E', 'Energy (MeV)', r => r.E >= 1000 ? r.E.toFixed(0) : r.E.toFixed(1)],
  ['pSun', 'P(Sun)', r => r.pSun == null ? '–' : `<span class="pbar"><i><b style="width:${(100 * r.pSun).toFixed(1)}%"></b></i>${r.pSun.toFixed(3)}</span>`],
  ['pGal', 'P(Galactic)', r => f3(r.pGal, 3)], ['pIso', 'P(isotropic)', r => f3(r.pIso, 3)],
  ['sep', 'Sun dist. (°)', r => f3(r.sep, 2)], ['r68', 'PSF r68 (°)', r => f3(r.r68, 2)],
  ['theta', 'θ (°)', r => f3(r.theta, 1)], ['zen', 'Zenith (°)', r => f3(r.zen, 1)],
  ['conv', 'Conv.', r => r.conv, 'txt'], ['cls', 'Best class', r => r.cls, 'txt'],
  ['ra', 'RA (°)', r => f3(r.ra, 3)], ['dec', 'Dec (°)', r => f3(r.dec, 3)], ['run', 'Run', r => r.run], ['id', 'Event ID', r => r.id]];
const evSort = { key: 't', dir: 1 };
function evRows() {
  const pmin = +$e('#evpmin').value || 0, emin = +$e('#evemin').value || 0;
  const tmin = $e('#evtmin').value === '' ? -Infinity : +$e('#evtmin').value, tmax = $e('#evtmax').value === '' ? Infinity : +$e('#evtmax').value;
  const rows = EV.filter(r => (r.pSun ?? 0) >= pmin && r.E >= emin && r.t >= tmin && r.t <= tmax);
  const k = evSort.key, d = evSort.dir;
  rows.sort((a, b) => { const x = a[k], y = b[k]; if (typeof x === 'string' || typeof y === 'string') return d * String(x).localeCompare(String(y), undefined, { numeric: true }); return d * ((x ?? -Infinity) - (y ?? -Infinity)); });
  return rows;
}
function evRender() {
  const rows = evRows();
  const head = '<thead><tr><th>#</th>' + EVCOLS.map(([k, l]) => `<th class="sort" data-k="${k}" aria-sort="${evSort.key === k ? (evSort.dir > 0 ? 'ascending' : 'descending') : 'none'}">${l}</th>`).join('') + '</tr></thead>';
  const body = rows.map((r, i) => `<tr class="${(r.pSun ?? 0) >= 0.9 && r.E >= 1000 ? 'hi' : ''}"><td class="num">${i + 1}</td>` + EVCOLS.map(([k, , f, t]) => `<td class="${t ? '' : 'num'}">${f(r)}</td>`).join('') + '</tr>').join('');
  $e('#evtable').innerHTML = head + '<tbody>' + body + '</tbody>';
  const n90 = rows.filter(r => (r.pSun ?? 0) >= 0.9).length;
  $e('#evcount').textContent = `${rows.length} of ${EV.length} events shown · ${n90} with P(Sun) ≥ 0.9 · highlighted: P(Sun) ≥ 0.9 and E ≥ 1 GeV`;
  $e('#evtable').querySelectorAll('th.sort').forEach(th => th.addEventListener('click', () => {
    const k = th.dataset.k;
    if (evSort.key === k) evSort.dir *= -1; else { evSort.key = k; evSort.dir = ['E', 'pSun'].includes(k) ? -1 : 1; }
    evRender();
  }));
}
if (EV.length && $e('#evtable')) {
  ['#evpmin', '#evemin', '#evtmin', '#evtmax'].forEach(s => $e(s).addEventListener('input', evRender));
  $e('#evcsv').addEventListener('click', () => {
    const keys = ['t', 'utc', 'E', 'pSun', 'pGal', 'pIso', 'sep', 'r68', 'theta', 'zen', 'conv', 'cls', 'ra', 'dec', 'run', 'id', 'bin'];
    const csv = [keys.join(',')].concat(evRows().map(r => keys.map(k => r[k] ?? '').join(','))).join('\n');
    const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = 'FLARENAME_lat_events.csv'; document.body.appendChild(a); a.click(); a.remove();
  });
  evRender();
}
'''.replace('EVDATA', json.dumps(EVENTS).replace('</', '<\\/')).replace('FLARENAME', cfg.name)
page = page.replace('EVJS', EVJS, 1)
out = os.path.join(R, 'report.html')
os.makedirs(R, exist_ok=True)
with open(out, 'w') as f:
    f.write(page)
print('### DONE report', out, flush=True)
