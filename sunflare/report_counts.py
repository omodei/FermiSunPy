"""Count spectra (data, model, components, residuals) and nuFnu of every fit, viewed in a pop-up of the report.

LAT fits: latfit.count_spectrum stores counts and model counts per likelihood component in lat_summary.json.
Joint fits: spectral.py writes spectrum_<t0>-<t1>/counts.json (countdata.py) with, per detector, the observed counts,
background and channel selection, and, per model, the predicted counts and the nuFnu curve of each component.
build() turns both into "spectrum sets" that the page script draws on demand.
"""
import glob
import html
import json
import os

from sunflare import common

E = html.escape


def _g(v, p=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return '–'
    if v != v:
        return '–'
    return f'{v:.{p - 1}e}' if v != 0 and (abs(v) < 1e-2 or abs(v) >= 1e5) else f'{v:.{p}g}'


def _lat_info(f, ts_min, band):
    det = f['TS'] >= ts_min
    s = f'TS {f["TS"]:.1f} · log L {f["logL"]:.2f}'
    if det and f.get('flux'):
        s += f' · flux {band[0]:g}–{band[1]:g} MeV {_g(f["flux"][0])} ({_g(f["flux"][1])}–{_g(f["flux"][2])}) ph cm⁻² s⁻¹'
    elif f.get('flux_ul') is not None:
        s += f' · flux &lt; {_g(f["flux_ul"])} (95% UL, TS &lt; {ts_min:g})'
    fixed = {x[0] for x in f.get('fixed', [])}
    par = '; '.join(f'{p} = {v[0]:.3g} (fixed)' if p in fixed else f'{p} = {v[0]:.3g} −{abs(v[1]):.2g} +{abs(v[2]):.2g}'
                    for p, v in f.get('params', {}).items()) if det else ''
    notes = (['no errors: parameters at their bounds'] if f.get('no_errors') else []) + \
            ([f'started from the {f["start"]}'] if f.get('start', 'configured') != 'configured' else [])
    return s, E(par), E(' · '.join(notes))


def _lat_set(r, key, title, lat):
    models, data = {}, None
    for kind, f in r['fits'].items():
        c = f.get('counts')
        if not c:
            continue
        if data is None:
            data = {'LAT': {'kind': 'lat', 'E': [e * 1e3 for e in c['E']], 'n': c['n'], 't': r['t1'] - r['t0']}}
        s, par, notes = _lat_info(f, lat['TS_MIN'], lat['flux_band_MeV'])
        models[kind] = {'label': common.LAT_MODELS[kind], 'short': {'pion': 'pion', 'pow': 'power law', 'expco': 'cutoff PL'}.get(kind, kind),
                        'info': s, 'par': par, 'notes': notes, 'logL': f['logL'], 'ds': {'LAT': c['m']}}
    if not models:
        return None
    return {'key': key, 'group': 'lat', 'title': f'LAT count spectrum — {title}', 'a': r['t0'], 'b': r['t1'],
            'best': r.get('best'), 'models': models, 'data': data}


def _joint_set(S, C, key):
    fits = S.get('fits', {})
    if not fits or not C or not C.get('models'):
        return None
    best_bic = min(v['BIC'] for v in fits.values())
    models = {}
    for name, m in C['models'].items():
        if name not in fits:
            continue
        v = fits[name]
        info = (f'−log L {-v["logL"]:.2f} · BIC {v["BIC"]:.1f} (ΔBIC {v["BIC"] - best_bic:.1f}) · '
                f'{v["n_free"]} free parameters')
        par = '; '.join(f'{p[0]} = {_g(p[1])}' for p in v.get('params', []) if not p[0].startswith(('cons_', 'Isotropic', 'Galactic')))
        notes = (['no errors: parameters at their bounds'] if v.get('no_errors') else []) + \
                ([f'started from the {v["start"]}'] if v.get('start', 'configured') != 'configured' else []) + \
                [f'fixed: {", ".join(x[0] for x in v["fixed"])}'] * bool(v.get('fixed'))
        models[name] = {'label': name, 'short': name, 'info': info, 'par': E(par), 'notes': E(' · '.join(notes)),
                        'logL': v['logL'], 'ds': m['ds'], 'vfv': m.get('vfv')}
    if not models:
        return None
    a, b = S['interval']
    return {'key': key, 'group': 'joint', 'title': f'Joint fit {a:g}–{b:g} s', 'a': a, 'b': b, 'best': S.get('best'),
            'models': models, 'data': C['data'], 'vfvE': C.get('vfv_E')}


def build(lat, R, spec_id):
    sets = []
    if lat:
        s = _lat_set(lat['integrated'], 'integ', 'time-integrated', lat)
        if s:
            sets.append(s)
        n = len(lat['resolved'])
        for i, r in enumerate(lat['resolved']):
            s = _lat_set(r, f'{r["t0"]:.3f}', f'bin {i + 1} of {n}', lat)
            if s:
                sets.append(s)
    for p in sorted(glob.glob(os.path.join(R, 'spectrum_*', 'summary.json')), key=lambda p: (common.load_json(p) or {}).get('interval', [0])[0]):
        S = common.load_json(p) or {}
        C = common.load_json(os.path.join(os.path.dirname(p), 'counts.json'))
        if 'interval' in S:
            s = _joint_set(S, C, spec_id(*S['interval']))
            if s:
                sets.append(s)
    return {'sets': sets} if sets else None


def keys(D):
    return {s['key']: set(s['models']) for s in (D or {}).get('sets', [])}


def button(key, kind, label='Counts'):
    return (f'<button type="button" class="cbtn" data-ckey="{key}" data-kind="{E(kind)}" '
            f'title="Count spectrum and residuals of this fit">{label}</button>')


def script(D):
    return JS.replace('CSPECDATA', json.dumps(D, separators=(',', ':'), allow_nan=False).replace('</', '<\\/'))


CSS = r'''
.cbtn { font:500 12px system-ui; padding:3px 9px; border:1px solid var(--line); border-radius:6px; background:var(--raise); color:var(--fg); cursor:pointer; white-space:nowrap }
.cbtn:hover { border-color:var(--accent); color:var(--accent) }
dialog.cdlg { width:min(1060px, calc(100vw - 32px)); max-height:calc(100vh - 24px); border:1px solid var(--line); border-radius:12px; padding:0;
  background:var(--surface); color:var(--fg); box-shadow:0 20px 60px rgba(0,0,0,.35); overflow:auto }
dialog.cdlg:fullscreen { width:100vw; max-height:none; height:100vh; border-radius:0 }
dialog.cdlg::backdrop { background:rgba(10,14,22,.55) }
.cdlg .ch { display:flex; gap:10px; align-items:center; padding:12px 16px; border-bottom:1px solid var(--line); flex-wrap:wrap }
.cdlg .ch b { font-size:16px } .cdlg .ch .cs { color:var(--muted); font:13px ui-monospace,Menlo,monospace }
.cdlg .ch .nav { margin-left:auto; display:flex; gap:6px; flex-wrap:wrap }
.cdlg button.b { font:500 12px system-ui; padding:5px 10px; border:1px solid var(--line); border-radius:6px; background:var(--raise); color:var(--fg); cursor:pointer }
.cdlg button.b:hover:not(:disabled) { border-color:var(--accent) } .cdlg button.b:disabled { opacity:.4; cursor:default }
.cdlg button.b[aria-pressed="true"] { background:var(--accent); border-color:var(--accent); color:#fff }
.cdlg .cm { display:flex; gap:8px 14px; align-items:center; flex-wrap:wrap; padding:10px 16px 0; font-size:13px }
.cdlg .cm .grp { display:inline-flex; gap:4px; flex-wrap:wrap } .cdlg .cm label { display:inline-flex; gap:5px; align-items:center; white-space:nowrap }
.cdlg .cm select { font:12px system-ui; padding:3px 6px; border:1px solid var(--line); border-radius:6px; background:var(--surface); color:var(--fg) }
.cdlg .cm i { display:inline-block; width:16px; border-top:2px solid; vertical-align:middle }
.cdlg .cm .sep { width:1px; height:18px; background:var(--line) }
.cdlg canvas { display:block; width:100%; touch-action:none } .cdlg .cvbox { padding:4px 8px 0 }
.cdlg .ci { padding:4px 16px 14px; font:13px/1.55 ui-monospace,Menlo,monospace; color:var(--fg) }
.cdlg .ci .muted { color:var(--muted) } .cdlg .cr { padding:0 16px; min-height:20px; font:12px ui-monospace,Menlo,monospace; color:var(--muted) }
'''

JS = r'''
/* ---------- count spectra and nuFnu (pop-up) ---------- */
(function () {
const D = CSPECDATA;
if (!D) return;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const MPAL = ['#dc2626', '#2563eb', '#0d9488', '#ea580c', '#9333ea', '#ca8a04', '#db2777', '#0891b2'];
const DPAL = ['#2563eb', '#dc2626', '#0d9488', '#ca8a04', '#9333ea', '#ea580c', '#db2777', '#4b5563'];
const LATCOMP = { Sun: [], Isotropic: [6, 4], Galactic: [2, 3] }, COMPCOL = { Isotropic: '#9333ea', Galactic: '#ca8a04', background: '#64748b' };
const DASH = [[7, 3], [2, 3], [9, 3, 2, 3], [4, 4], [1, 2]];
const byKey = Object.fromEntries(D.sets.map(s => [s.key, s]));
const S = { key: null, kind: null, cmp: false, minc: 5, hide: {}, dsOff: {}, view: 'counts', er: null, split: 0.66 };
const g3 = v => v == null || !isFinite(v) ? '–' : Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-2 && v !== 0) ? v.toExponential(2) : +v.toPrecision(3) + '';
const fe = e => e >= 1e6 ? +(e / 1e6).toPrecision(3) + ' GeV' : e >= 1e3 ? +(e / 1e3).toPrecision(3) + ' MeV' : +e.toPrecision(3) + ' keV';

const dlg = document.createElement('dialog'); dlg.className = 'cdlg';
dlg.innerHTML = `<div class="ch"><b class="ct"></b><span class="cs"></span><span class="nav">
  <span class="views"></span>
  <button type="button" class="b prev" title="Previous (←)">‹ Previous</button><button type="button" class="b next" title="Next (→)">Next ›</button>
  <button type="button" class="b fs" title="Full screen">⛶</button><button type="button" class="b png" title="Save the plot as PNG">PNG</button>
  <button type="button" class="b x" title="Close (Esc)">✕</button></span></div>
  <div class="cm"><span class="grp mods"></span><label><input type="checkbox" class="cmp"> compare models</label>
  <label class="gl">group to ≥ <select class="minc">${[1, 3, 5, 10, 20, 50].map(v => `<option${v === 5 ? ' selected' : ''}>${v}</option>`).join('')}</select> counts</label>
  <span class="sep"></span><span class="grp dss"></span><span class="grp comps"></span></div>
  <div class="cvbox"><canvas></canvas></div><div class="cr"></div><div class="ci"></div>`;
document.body.appendChild(dlg);
const cv = dlg.querySelector('canvas'), $d = s => dlg.querySelector(s);

function setOf() { return byKey[S.key]; }
function mcolOf(set, k) { return MPAL[Object.keys(set.models).indexOf(k) % MPAL.length]; }
function open(key, kind) {
  const s = byKey[key]; if (!s) return;
  if (!S.key || byKey[S.key].group !== s.group) { S.er = null; S.view = 'counts'; }
  S.key = key; S.kind = s.models[kind] ? kind : (s.models[s.best] ? s.best : Object.keys(s.models)[0]);
  if (!dlg.open) dlg.showModal();
  render();
}
window.openCounts = open;
const groupKeys = () => D.sets.filter(s => s.group === setOf().group).map(s => s.key);
function step(d) { const ks = groupKeys(), i = ks.indexOf(S.key) + d; if (i >= 0 && i < ks.length) open(ks[i], S.kind); }

function render() {
  const s = setOf(), mods = Object.keys(s.models), dsn = Object.keys(s.data);
  $d('.ct').textContent = s.title;
  $d('.cs').textContent = `${s.a.toFixed(1)}–${s.b.toFixed(1)} s from T0 (${(s.b - s.a) >= 120 ? ((s.b - s.a) / 60).toFixed(1) + ' min' : (s.b - s.a).toFixed(1) + ' s'})`;
  const ks = groupKeys(), i = ks.indexOf(S.key); $d('.prev').disabled = i <= 0; $d('.next').disabled = i >= ks.length - 1;
  const hasV = mods.some(k => s.models[k].vfv); if (!hasV) S.view = 'counts';
  $d('.views').innerHTML = hasV ? ['counts', 'vfv'].map(v => `<button type="button" class="b" data-v="${v}" aria-pressed="${S.view === v}">${v === 'counts' ? 'Counts' : 'νFν'}</button>`).join(' ') : '';
  $d('.mods').innerHTML = mods.map(k => `<button type="button" class="b" data-k="${k}" aria-pressed="${k === S.kind}" title="${s.models[k].label}">${s.models[k].short}${k === s.best ? ' ★' : ''}</button>`).join('');
  $d('.cmp').disabled = mods.length < 2; $d('.cmp').checked = S.cmp;
  $d('.gl').hidden = S.view !== 'counts';
  $d('.dss').innerHTML = dsn.length > 1 && S.view === 'counts' ? dsn.map((n, j) => `<label><input type="checkbox" data-ds="${n}"${S.dsOff[n] ? '' : ' checked'}><i style="border-color:${DPAL[j % DPAL.length]}"></i>${n}</label>`).join('') : '';
  let comps = [];
  if (S.view === 'counts') { const seen = new Set(); for (const n of dsn) { for (const c of Object.keys(s.models[S.kind].ds[n] || {})) if (c !== 'source') seen.add(c); if (s.data[n].bkg) seen.add('background'); } comps = [...seen]; }
  else comps = Object.keys((s.models[S.kind].vfv || {}).comps || {});
  $d('.comps').innerHTML = comps.map((c, j) => `<label><input type="checkbox" data-c="${c}"${S.hide[c] ? '' : ' checked'}><i style="border-top-style:${c === 'Sun' ? 'solid' : 'dashed'};border-color:${COMPCOL[c] || (S.view === 'vfv' ? mcolOf(s, S.kind) : mcolOf(s, S.kind))}"></i>${c}</label>`).join('');
  const shown = mods.filter(k => S.cmp || k === S.kind);
  const lines = shown.map(k => { const m = s.models[k]; return `<span style="color:${mcolOf(s, k)}">■</span> <b>${m.label}</b>${k === s.best ? ' ★ best' : ''} · ${m.info}` + (m.par ? `<br><span class="muted">   ${m.par}</span>` : '') + (m.notes ? `<br><span class="muted">   ${m.notes}</span>` : ''); });
  if (S.cmp && mods.length > 1) { const L = mods.map(k => s.models[k].logL); lines.push(`<span class="muted">largest Δ log L between models: ${(Math.max(...L) - Math.min(...L)).toFixed(2)}</span>`); }
  $d('.ci').innerHTML = lines.join('<br>');
  draw();
}

/* group adjacent selected channels until each group has >= minc observed counts */
function groups(n, mask, minc) {
  const g = []; let s = 0, i0 = -1;
  for (let i = 0; i <= n.length; i++) {
    const on = i < n.length && (!mask || mask[i]);
    if (!on) { if (i0 >= 0) { if (s >= minc || !g.length || g[g.length - 1][1] !== i0 - 1) g.push([i0, i - 1]); else g[g.length - 1][1] = i - 1; } i0 = -1; s = 0; continue; }
    if (i0 < 0) i0 = i;
    s += n[i];
    if (s >= minc) { g.push([i0, i]); i0 = -1; s = 0; }
  }
  return g;
}
function totals(set, k, name) {
  const m = set.models[k].ds[name] || {}, d = set.data[name], out = new Array(d.n.length).fill(0);
  for (const arr of Object.values(m)) arr.forEach((v, i) => out[i] += v);
  if (d.bkg) d.bkg.forEach((v, i) => out[i] += v);
  return out;
}
let G = null, dragE = null, dragSplit = null;
function frame(W, H) {
  const x0 = 78, x1 = W - 22, ya = 12, yd = H - 34, yb = Math.round(ya + (yd - ya) * S.split), yc = yb + 12;
  return { x0, x1, ya, yb, yc, yd };
}
function eRange(s) {
  let lo = Infinity, hi = 0;
  if (S.view === 'vfv') { lo = s.vfvE[0]; hi = s.vfvE[s.vfvE.length - 1]; }
  else for (const [n, d] of Object.entries(s.data)) { if (S.dsOff[n]) continue; const on = d.mask ? d.mask.map((m, i) => m ? i : -1).filter(i => i >= 0) : [0, d.n.length - 1]; if (!on.length) continue; lo = Math.min(lo, d.E[on[0]]); hi = Math.max(hi, d.E[on[on.length - 1] + 1]); }
  if (!(hi > lo)) { lo = 10; hi = 1e6; }
  return S.er ? [Math.max(S.er[0], lo / 10), Math.min(S.er[1], hi * 10)] : [lo / 1.15, hi * 1.15];
}
function xAxis(c, F, xs, lx0, lx1) {
  c.textAlign = 'center';
  const dec = lx1 - lx0, ms = dec > 4 ? [1] : dec > 1.5 ? [1, 3] : [1, 2, 5];
  for (let e = Math.floor(lx0); e <= Math.ceil(lx1); e++) for (const m of ms) { const v = m * 10 ** e; if (Math.log10(v) < lx0 || Math.log10(v) > lx1) continue; const x = xs(v); c.strokeStyle = css('--line'); c.beginPath(); c.moveTo(x + .5, F.ya); c.lineTo(x + .5, F.yd); c.stroke(); c.fillStyle = css('--muted'); c.fillText(fe(v), x, F.yd + 16); }
}
function logY(c, F, lo, hi, y0, y1) {
  const L0 = Math.log10(lo), L1 = Math.log10(hi), ys = v => y1 - (y1 - y0) * (Math.log10(v) - L0) / (L1 - L0);
  c.textAlign = 'right'; const st = Math.max(1, Math.ceil((Math.floor(L1) - Math.ceil(L0) + 1) / Math.max(2, (y1 - y0) / 24)));
  for (let e = Math.ceil(L0); e <= Math.floor(L1); e += st) { const y = ys(10 ** e); c.strokeStyle = css('--line'); c.beginPath(); c.moveTo(F.x0, y + .5); c.lineTo(F.x1, y + .5); c.stroke(); c.fillStyle = css('--muted'); c.fillText(e >= -2 && e <= 3 ? String(+(10 ** e).toPrecision(1)) : `1e${e}`, F.x0 - 6, y + 4); }
  return ys;
}
function ylabel(c, text, y) { c.save(); c.fillStyle = css('--muted'); c.translate(16, y); c.rotate(-Math.PI / 2); c.textAlign = 'center'; c.fillText(text, 0, 0); c.restore(); }
function draw() {
  const s = setOf(), dpr = devicePixelRatio || 1, W = cv.clientWidth || 900;
  const H = document.fullscreenElement === dlg ? Math.max(420, innerHeight - 260) : Math.min(560, Math.max(380, innerHeight - 320));
  cv.width = W * dpr; cv.height = H * dpr; cv.style.height = H + 'px';
  const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.fillStyle = css('--surface'); c.fillRect(0, 0, W, H); c.font = '11px ui-monospace,Menlo,monospace'; c.lineWidth = 1;
  const F = frame(W, H), [elo, ehi] = eRange(s), lx0 = Math.log10(elo), lx1 = Math.log10(ehi);
  const xs = e => F.x0 + (F.x1 - F.x0) * (Math.log10(e) - lx0) / (lx1 - lx0), es = x => 10 ** (lx0 + (x - F.x0) / (F.x1 - F.x0) * (lx1 - lx0));
  G = { F, xs, es, pts: [] };
  if (S.view === 'vfv') drawVfv(c, s, F, xs, lx0, lx1); else drawCounts(c, s, F, xs, lx0, lx1);
  if (dragE) { const xa = xs(Math.min(dragE.a, dragE.b)), xb = xs(Math.max(dragE.a, dragE.b)); c.fillStyle = 'rgba(120,120,140,.2)'; c.fillRect(xa, F.ya, xb - xa, F.yd - F.ya); }
}
function drawCounts(c, s, F, xs, lx0, lx1) {
  const dsn = Object.keys(s.data).filter(n => !S.dsOff[n]), multi = Object.keys(s.data).length > 1;
  const mods = Object.keys(s.models).filter(k => S.cmp || k === S.kind);
  const dcol = n => multi ? DPAL[Object.keys(s.data).indexOf(n) % DPAL.length] : css('--fg');
  const lcol = (n, k) => multi ? dcol(n) : mcolOf(s, k);
  const per = []; const vals = [];
  for (const n of dsn) {
    const d = s.data[n], t = d.t > 0 ? d.t : 1, dE = d.E.slice(1).map((v, i) => v - d.E[i]);
    const tot = Object.fromEntries(mods.map(k => [k, totals(s, k, n)]));
    const gr = groups(d.n, d.mask, S.minc).map(([i, j]) => { let N = 0, de = 0; const M = {}; for (let q = i; q <= j; q++) { N += d.n[q]; de += dE[q]; } for (const k of mods) { M[k] = 0; for (let q = i; q <= j; q++) M[k] += tot[k][q]; } return { n, i, j, e0: d.E[i], e1: d.E[j + 1], N, de, M, t }; })
      .filter(p => p.e1 > 10 ** lx0 && p.e0 < 10 ** lx1);
    per.push({ n, d, t, dE, tot, gr });
    for (const p of gr) { if (p.N > 0) vals.push(p.N / p.de / t); for (const k of mods) if (p.M[k] > 0) vals.push(p.M[k] / p.de / t); }
    if (d.bkg && !S.hide.background) for (const p of gr) { let b = 0; for (let q = p.i; q <= p.j; q++) b += d.bkg[q]; if (b > 0) vals.push(b / p.de / t); }
  }
  G.pts = per.flatMap(x => x.gr);
  if (!vals.length) { c.fillStyle = css('--muted'); c.textAlign = 'center'; c.fillText('no data in this energy range', (F.x0 + F.x1) / 2, (F.ya + F.yb) / 2); return; }
  vals.sort((a, b) => a - b);
  let lo = vals[Math.floor(vals.length * 0.01)] / 3, hi = vals[vals.length - 1] * 3;
  const ys = logY(c, F, lo, hi, F.ya, F.yb);
  xAxis(c, F, xs, lx0, lx1);
  ylabel(c, multi || s.group === 'joint' ? 'counts s⁻¹ keV⁻¹' : 'counts s⁻¹ keV⁻¹', (F.ya + F.yb) / 2);
  ylabel(c, '(data − model) / σ', (F.yc + F.yd) / 2);
  c.save(); c.beginPath(); c.rect(F.x0, F.ya, F.x1 - F.x0, F.yb - F.ya); c.clip();
  const stepLine = (gr, val, col, dash, lw) => { c.strokeStyle = col; c.lineWidth = lw; c.setLineDash(dash); c.beginPath(); let pen = false, last = null;
    for (const p of gr) { const v = val(p); const y = v > 0 ? Math.min(F.yb + 2, Math.max(F.ya - 2, ys(v))) : null; if (y == null) { pen = false; continue; } const xa = xs(p.e0), xb = xs(p.e1); if (!pen || (last != null && Math.abs(last - p.e0) > 1e-6 * p.e0)) { c.moveTo(xa, y); pen = true; } else c.lineTo(xa, y); c.lineTo(xb, y); last = p.e1; }
    c.stroke(); c.setLineDash([]); c.lineWidth = 1; };
  for (const P of per) {
    const sum = (arr, p) => { let v = 0; for (let q = p.i; q <= p.j; q++) v += arr[q]; return v / p.de / P.t; };
    const md = s.models[S.kind].ds[P.n] || {};
    for (const [comp, arr] of Object.entries(md)) { if (comp === 'source' || S.hide[comp]) continue; stepLine(P.gr, p => sum(arr, p), comp === 'Sun' ? lcol(P.n, S.kind) : COMPCOL[comp] || '#888', comp === 'Sun' ? [7, 3] : LATCOMP[comp] || [3, 3], 1.3); }
    if (P.d.bkg && !S.hide.background) stepLine(P.gr, p => sum(P.d.bkg, p), multi ? dcol(P.n) : COMPCOL.background, [2, 3], 1.2);
    mods.forEach((k, mi) => stepLine(P.gr, p => p.M[k] / p.de / P.t, lcol(P.n, k), k === S.kind ? [] : (multi ? DASH[mi % DASH.length] : []), k === S.kind ? 2.2 : 1.4));
    c.strokeStyle = dcol(P.n); c.fillStyle = dcol(P.n); c.lineWidth = 1.2;
    for (const p of P.gr) { const xa = xs(p.e0), xb = xs(p.e1), xm = xs(Math.sqrt(p.e0 * p.e1));
      if (p.N > 0) { const r = p.N / p.de / P.t, er = Math.sqrt(p.N) / p.de / P.t, y = ys(r); c.beginPath(); c.moveTo(xa, y); c.lineTo(xb, y); c.moveTo(xm, ys(r + er)); c.lineTo(xm, r - er > 0 ? ys(r - er) : F.yb); c.stroke(); c.beginPath(); c.arc(xm, y, 2.6, 0, 7); c.fill(); }
      else { const y = ys(1.84 / p.de / P.t); c.beginPath(); c.moveTo(xa, y); c.lineTo(xb, y); c.moveTo(xm, y); c.lineTo(xm, y + 12); c.stroke(); c.beginPath(); c.moveTo(xm - 3.5, y + 8); c.lineTo(xm, y + 13); c.lineTo(xm + 3.5, y + 8); c.fill(); } }
  }
  c.restore();
  // residuals
  let rmax = 3; const R = [];
  for (const P of per) for (const p of P.gr) for (const k of mods) { const M = p.M[k]; if (!(M > 0)) continue; const r = (p.N - M) / Math.sqrt(M), e = (p.N > 0 ? Math.sqrt(p.N) : 1) / Math.sqrt(M); R.push([P.n, p, k, r, e]); rmax = Math.max(rmax, Math.abs(r) + e); }
  rmax = Math.min(rmax * 1.1, 40);
  const yr = v => (F.yc + F.yd) / 2 - (F.yd - F.yc) / 2 * v / rmax;
  c.textAlign = 'right'; c.fillStyle = css('--muted');
  (rmax > 20 ? [-20, 0, 20] : rmax > 10 ? [-10, 0, 10] : rmax > 5 ? [-5, 0, 5] : [-3, 0, 3]).forEach(v => { if (Math.abs(v) <= rmax) c.fillText(String(v), F.x0 - 6, yr(v) + 4); });
  c.strokeStyle = css('--muted'); c.beginPath(); c.moveTo(F.x0, yr(0) + .5); c.lineTo(F.x1, yr(0) + .5); c.stroke();
  c.setLineDash([3, 4]); c.strokeStyle = css('--line'); [-2, 2].forEach(v => { c.beginPath(); c.moveTo(F.x0, yr(v) + .5); c.lineTo(F.x1, yr(v) + .5); c.stroke(); }); c.setLineDash([]);
  c.save(); c.beginPath(); c.rect(F.x0, F.yc, F.x1 - F.x0, F.yd - F.yc); c.clip();
  for (const [n, p, k, r, e] of R.sort((a, b) => (a[2] === S.kind) - (b[2] === S.kind))) {
    const main = k === S.kind, col = multi ? dcol(n) : (main ? css('--fg') : mcolOf(s, k)), xm = xs(Math.sqrt(p.e0 * p.e1)) + (main ? 0 : 3 + 3 * mods.indexOf(k));
    c.strokeStyle = col; c.fillStyle = col; c.globalAlpha = main ? 1 : .6;
    if (main) { c.beginPath(); c.moveTo(xs(p.e0), yr(r)); c.lineTo(xs(p.e1), yr(r)); c.stroke(); }
    c.beginPath(); c.moveTo(xm, yr(r - e)); c.lineTo(xm, yr(r + e)); c.stroke();
    c.beginPath(); main ? c.arc(xm, yr(r), 2.6, 0, 7) : c.rect(xm - 2, yr(r) - 2, 4, 4); c.fill();
  }
  c.globalAlpha = 1; c.restore();
  c.strokeStyle = css('--muted'); c.strokeRect(F.x0 + .5, F.ya + .5, F.x1 - F.x0, F.yb - F.ya); c.strokeRect(F.x0 + .5, F.yc + .5, F.x1 - F.x0, F.yd - F.yc);
  const nObs = per.reduce((a, P) => a + P.gr.reduce((b, p) => b + p.N, 0), 0), nMod = per.reduce((a, P) => a + P.gr.reduce((b, p) => b + p.M[S.kind], 0), 0);
  G.summary = `${G.pts.length} energy groups in view · ${nObs} counts observed, ${nMod.toFixed(1)} predicted by ${s.models[S.kind].short} (incl. background) · drag to zoom in energy, double-click to reset, drag the gap between the panels to resize them`;
  $d('.cr').textContent = G.summary;
}
function drawVfv(c, s, F, xs, lx0, lx1) {
  const E = s.vfvE, mods = Object.keys(s.models).filter(k => (S.cmp || k === S.kind) && s.models[k].vfv);
  const vals = [];
  for (const k of mods) { const v = s.models[k].vfv; v.total.forEach((y, i) => { if (y > 0 && E[i] >= 10 ** lx0 && E[i] <= 10 ** lx1) vals.push(y); }); }
  if (!vals.length) return;
  vals.sort((a, b) => a - b);
  const top = vals[Math.floor(vals.length * 0.97)], lo = top * 1e-4, hi = top * 6;
  const F2 = { ...F, yb: F.yd };
  const ys = logY(c, F2, lo, hi, F.ya, F.yd);
  xAxis(c, F2, xs, lx0, lx1);
  ylabel(c, 'E² dN/dE  [erg cm⁻² s⁻¹]', (F.ya + F.yd) / 2);
  c.save(); c.beginPath(); c.rect(F.x0, F.ya, F.x1 - F.x0, F.yd - F.ya); c.clip();
  const line = (arr, col, dash, lw, alpha) => { c.strokeStyle = col; c.lineWidth = lw; c.setLineDash(dash); c.globalAlpha = alpha; c.beginPath(); let pen = false; arr.forEach((v, i) => { if (!(v > 0)) { pen = false; return; } const x = xs(E[i]), y = Math.max(F.ya - 5, Math.min(F.yd + 5, ys(v))); if (!pen) { c.moveTo(x, y); pen = true; } else c.lineTo(x, y); }); c.stroke(); c.setLineDash([]); c.globalAlpha = 1; c.lineWidth = 1; };
  for (const k of mods) {
    const v = s.models[k].vfv, col = mcolOf(s, k), main = k === S.kind;
    if (main) Object.entries(v.comps).forEach(([cn, arr], i) => { if (!S.hide[cn]) line(arr, col, DASH[i % DASH.length], 1.3, .85); });
    line(v.total, col, [], main ? 2.6 : 1.4, main ? 1 : .6);
  }
  c.restore();
  // component labels at their peaks
  const v = s.models[S.kind].vfv; c.font = '11px system-ui'; c.textAlign = 'left'; c.fillStyle = mcolOf(s, S.kind);
  Object.entries(v.comps).forEach(([cn, arr]) => { if (S.hide[cn]) return; let im = -1; arr.forEach((y, i) => { if (E[i] >= 10 ** lx0 && E[i] <= 10 ** lx1 && (im < 0 || y > arr[im])) im = i; }); if (im >= 0 && arr[im] > lo) c.fillText(cn, xs(E[im]) + 4, Math.max(F.ya + 12, ys(arr[im]) - 6)); });
  c.font = '11px ui-monospace,Menlo,monospace';
  c.strokeStyle = css('--muted'); c.strokeRect(F.x0 + .5, F.ya + .5, F.x1 - F.x0, F.yd - F.ya);
  G.summary = `νFν of ${mods.length > 1 ? 'the fitted models' : s.models[S.kind].short} (best-fit parameters; dashed: components) · drag to zoom in energy, double-click to reset`;
  $d('.cr').textContent = G.summary;
}

/* ---- interaction ---- */
const posOf = ev => { const r = cv.getBoundingClientRect(); return [ev.clientX - r.left, ev.clientY - r.top]; };
cv.addEventListener('pointerdown', ev => {
  if (!G || ev.button !== 0) return; const [x, y] = posOf(ev); cv.setPointerCapture(ev.pointerId);
  if (S.view === 'counts' && y > G.F.yb - 2 && y < G.F.yc + 2) { dragSplit = { y0: y, s0: S.split }; return; }
  if (x >= G.F.x0 && x <= G.F.x1) dragE = { a: G.es(x), b: G.es(x), x };
});
cv.addEventListener('pointermove', ev => {
  if (!G) return; const [x, y] = posOf(ev);
  cv.style.cursor = (S.view === 'counts' && y > G.F.yb - 2 && y < G.F.yc + 2) || dragSplit ? 'ns-resize' : 'crosshair';
  if (dragSplit) { S.split = Math.max(0.3, Math.min(0.85, dragSplit.s0 + (y - dragSplit.y0) / (G.F.yd - G.F.ya))); draw(); return; }
  if (dragE) { dragE.b = G.es(Math.max(G.F.x0, Math.min(G.F.x1, x))); draw(); return; }
  if (S.view !== 'counts') { $d('.cr').textContent = `${fe(G.es(x))} · ` + G.summary; return; }
  const s = setOf(), hits = G.pts.filter(p => x >= G.xs(p.e0) && x < G.xs(p.e1));
  if (!hits.length) { $d('.cr').textContent = G.summary; return; }
  $d('.cr').textContent = hits.map(p => `${Object.keys(s.data).length > 1 ? p.n + ' ' : ''}${fe(p.e0)}–${fe(p.e1)}: ${p.N} observed, ` + Object.entries(p.M).map(([k, v]) => `${s.models[k].short} ${v.toFixed(1)}`).join(', ') +
    (p.M[S.kind] > 0 ? ` (${((p.N - p.M[S.kind]) / Math.sqrt(p.M[S.kind])).toFixed(1)} σ)` : '')).join('  |  ');
});
cv.addEventListener('pointerup', ev => {
  if (dragSplit) { dragSplit = null; return; }
  if (!dragE) return; const d = dragE; dragE = null;
  const [x] = posOf(ev);
  if (Math.abs(x - d.x) > 5) S.er = [Math.min(d.a, d.b), Math.max(d.a, d.b)];
  draw();
});
cv.addEventListener('dblclick', () => { S.er = null; draw(); });
cv.addEventListener('wheel', ev => {
  if (!(ev.ctrlKey || ev.metaKey || ev.altKey) || !G) return; ev.preventDefault();
  const [x] = posOf(ev), e = Math.log10(G.es(x)), k = Math.exp(Math.max(-0.5, Math.min(0.5, ev.deltaY * 0.01)));
  const [a, b] = eRange(setOf()).map(Math.log10); S.er = [10 ** (e - (e - a) * k), 10 ** (e + (b - e) * k)]; draw();
}, { passive: false });
dlg.addEventListener('click', e => {
  if (e.target === dlg) { dlg.close(); return; }
  const t = e.target.closest('button'); if (!t) return;
  if (t.classList.contains('x')) { if (document.fullscreenElement) document.exitFullscreen(); dlg.close(); }
  if (t.classList.contains('prev')) step(-1);
  if (t.classList.contains('next')) step(1);
  if (t.classList.contains('fs')) { if (document.fullscreenElement) document.exitFullscreen(); else dlg.requestFullscreen && dlg.requestFullscreen(); }
  if (t.classList.contains('png')) { const a = document.createElement('a'); a.download = `${S.key}_${S.kind}_${S.view}.png`.replace(/[^\w.+-]/g, '_'); a.href = cv.toDataURL('image/png'); document.body.appendChild(a); a.click(); a.remove(); }
  if (t.dataset.k) { S.kind = t.dataset.k; render(); }
  if (t.dataset.v) { S.view = t.dataset.v; S.er = null; render(); }
});
dlg.addEventListener('change', e => {
  const t = e.target;
  if (t.classList.contains('cmp')) S.cmp = t.checked;
  if (t.classList.contains('minc')) S.minc = +t.value;
  if (t.dataset.c) S.hide[t.dataset.c] = !t.checked;
  if (t.dataset.ds) S.dsOff[t.dataset.ds] = !t.checked;
  render();
});
dlg.addEventListener('keydown', e => { if (e.target.tagName === 'SELECT') return; if (e.key === 'ArrowLeft') { step(-1); e.preventDefault(); } if (e.key === 'ArrowRight') { step(1); e.preventDefault(); } });
document.addEventListener('fullscreenchange', () => dlg.open && setTimeout(draw, 60));
addEventListener('resize', () => dlg.open && draw());
document.addEventListener('click', e => { const b = e.target.closest('.cbtn'); if (b) open(b.dataset.ckey, b.dataset.kind); });
})();
'''
