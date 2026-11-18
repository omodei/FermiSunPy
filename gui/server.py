#!/usr/bin/env python
"""Solar-flare analysis console: local web GUI (run it inside the FermiSunPy conda env).

    ./gui/start_gui.sh            # or: conda activate FermiSunPy && python gui/server.py --port 8766

Then open http://localhost:8766
"""
import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

GUI = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(GUI)
sys.path.insert(0, REPO)
from sunflare import common, lcdata  # noqa: E402

import numpy as np  # noqa: E402

JOBS = {}
LOCK = threading.Lock()
ENV = dict(os.environ, PYTHONUNBUFFERED='1', MPLBACKEND='Agg',
           PYTHONPATH=os.pathsep.join([REPO, os.environ.get('PYTHONPATH', '')]))   # keeps sunflare/_mpfix


# ------------------------------------------------------------------ jobs
def start_job(kind, fid, cmd):
    with LOCK:
        for j in JOBS.values():
            if j['fid'] == fid and j['proc'].poll() is None:
                raise RuntimeError(f'A {j["kind"]} job is already running for {fid}')
        jid = uuid.uuid4().hex[:8]
        logdir = os.path.join(common.flare_dir(fid), 'logs')
        os.makedirs(logdir, exist_ok=True)
        log = os.path.join(logdir, f'{kind}_{time.strftime("%Y%m%d_%H%M%S")}.log')
        fh = open(log, 'w')
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, env=ENV, start_new_session=True)
        JOBS[jid] = dict(id=jid, kind=kind, fid=fid, cmd=cmd, proc=proc, log=log, fh=fh, started=time.time())
    return jid


def job_status(jid, offset=0):
    j = JOBS.get(jid)
    if not j:
        raise KeyError('Unknown job')
    rc = j['proc'].poll()
    with open(j['log'], 'rb') as f:
        f.seek(offset)
        chunk = f.read(400_000)
    return dict(id=jid, kind=j['kind'], fid=j['fid'], running=rc is None, returncode=rc,
                elapsed=round(time.time() - j['started']), offset=offset + len(chunk),
                text=chunk.decode('utf-8', 'replace'), log=j['log'])


def stop_job(jid):
    j = JOBS[jid]
    if j['proc'].poll() is None:
        os.killpg(os.getpgid(j['proc'].pid), signal.SIGTERM)
    return {'stopped': True}


# ------------------------------------------------------------------ light curves
def _clean(a, nd=4):
    return [None if not np.isfinite(v) else float(f'{v:.{nd}g}') for v in np.asarray(a, dtype=float)]


EV_CACHE = {}


def lc_data(fid, src, tmin, tmax, dt, sun_centered=False, roi=10.0):
    d = common.flare_dir(fid)
    info = common.load_json(common.info_path(fid), {})
    edges = np.arange(tmin, tmax + dt, dt)
    if len(edges) > 6000:
        edges = np.linspace(tmin, tmax, 6001)
    if src == 'goes':
        tc, a, b = lcdata.goes(d, edges)
        return dict(kind='goes', edges=edges.tolist(), a=_clean(a), b=_clean(b), label='GOES XRS (A: 0.5–4 Å, B: 1–8 Å)')
    if src.startswith('gbm:'):
        _, det, lo, hi = src.split(':')
        tc, r, (elo, ehi) = lcdata.gbm(d, det, float(lo), float(hi), edges, info['T0'])
        lab = f'{det}  {elo:.0f}–{ehi:.0f} keV' if ehi < 2000 else f'{det}  {elo / 1000:.2g}–{ehi / 1000:.3g} MeV'
        return dict(kind='rate', edges=edges.tolist(), y=_clean(r, 5), label=lab)
    if src == 'lle':
        st = common.load_json(common.config_path(fid), {}) or {}
        lo, hi = float(st.get('LLE_EMIN_MEV', 30)), float(st.get('LLE_EMAX_MEV', 300))
        tc, r = lcdata.lle(d, info, lo, hi, edges)
        return dict(kind='rate', edges=edges.tolist(), y=_clean(r, 5), label=f'LLE  {lo:g}–{hi:g} MeV')
    if src == 'lat':
        key = (fid, bool(sun_centered), float(roi))
        if key not in EV_CACHE:
            EV_CACHE[key] = lcdata.lat_events(d, info, roi=roi, emin=100.0, sun_centered=bool(sun_centered))
        ev = EV_CACHE[key]
        from sunflare.lat_data import psf68
        m = (ev['t'] >= tmin) & (ev['t'] <= tmax)
        near = ev['sep'][m] < 1.5 * psf68(ev['E'][m])
        return dict(kind='events', t=np.round(ev['t'][m], 2).tolist(), E=np.round(ev['E'][m], 1).tolist(),
                    near=near.astype(int).tolist(), label=f'LAT  > 100 MeV within {roi:g}° (red: < 1.5 r68 of the Sun)')
    if src == 'theta':
        tr = lcdata.lat_track(d) or {'t': [], 'theta': [], 'zenith': []}
        t = np.array(tr['t'])
        m = (t >= tmin - 60) & (t <= tmax + 60)
        step = max(1, int(m.sum() // 1500))
        return dict(kind='angles', t=t[m][::step].tolist(), theta=np.array(tr['theta'])[m][::step].tolist(),
                    zenith=np.array(tr['zenith'])[m][::step].tolist(), windows=info.get('lat_windows', []),
                    label='LAT: Sun off-axis angle θ (black) and zenith angle (orange)')
    raise ValueError(f'Unknown source {src}')


# ------------------------------------------------------------------ flare summary
def flare_summary(name):
    fid = common.normalize_name(name)
    d = common.flare_dir(fid)
    rep = os.path.join(d, 'results', 'report.html')
    return dict(name=fid, exists=os.path.isdir(d), info=common.load_json(common.info_path(fid)),
                state=common.load_json(common.config_path(fid)),
                config=common.config_path(fid) if os.path.exists(common.config_path(fid)) else None,
                report=os.path.exists(rep), report_time=os.path.getmtime(rep) if os.path.exists(rep) else None,
                status=common.load_json(os.path.join(d, 'results', 'run_status.json')),
                jobs=[{'id': j['id'], 'kind': j['kind'], 'running': j['proc'].poll() is None}
                      for j in JOBS.values() if j['fid'] == fid])


def meta():
    try:
        from GtBurst.IRFS import IRFS
        irfs = [k for k in IRFS if k.startswith('p8_')] or common.IRFS_DEFAULT
    except Exception:
        irfs = common.IRFS_DEFAULT
    flares = []
    for p in sorted(glob.glob(os.path.join(common.FLARES, 'SOL*', 'flare_info.json'))):
        i = common.load_json(p, {})
        flares.append({'name': os.path.basename(os.path.dirname(p)), 'class': i.get('goes_class', ''), 't0': i.get('t0_utc', '')})
    return dict(irfs=irfs, defaults=common.DEFAULTS, components=common.COMPONENTS, presets=common.MODEL_PRESETS,
                lat_models=common.LAT_MODELS, flares=flares, python=sys.executable, data_dir=common.FLARES)


SEARCH_CACHE = {}


def search(date, min_class):
    k = (date, min_class)
    if k not in SEARCH_CACHE:
        from sunflare.search import search as do_search
        SEARCH_CACHE[k] = do_search(date, min_class)
    return SEARCH_CACHE[k]


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        pass

    def send(self, code, body, ctype='application/json'):
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body, default=common._json_default).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        n = int(self.headers.get('Content-Length') or 0)
        return json.loads(self.rfile.read(n) or b'{}')

    def handle_any(self, method):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        p = u.path
        try:
            if method == 'GET' and p in ('/', '/index.html'):
                return self.send(200, open(os.path.join(GUI, 'index.html'), 'rb').read(), 'text/html; charset=utf-8')
            if method == 'GET' and p.startswith('/report/'):
                fid = common.normalize_name(p.split('/')[2])
                f = os.path.join(common.flare_dir(fid), 'results', 'report.html')
                if not os.path.exists(f):
                    return self.send(404, b'No report yet for this flare. Run the analysis first.', 'text/plain')
                return self.send(200, open(f, 'rb').read(), 'text/html; charset=utf-8')
            if method == 'GET' and p == '/api/meta':
                return self.send(200, meta())
            if method == 'GET' and p == '/api/search':
                return self.send(200, search(q['date'], q.get('min_class', 'M')))
            if method == 'GET' and p == '/api/flare':
                return self.send(200, flare_summary(q['name']))
            if method == 'GET' and p == '/api/flare_id':
                return self.send(200, {'name': common.flare_id(q['t0'])})
            if method == 'GET' and p == '/api/lc':
                fid = common.normalize_name(q['name'])
                return self.send(200, lc_data(fid, q['src'], float(q['tmin']), float(q['tmax']), max(float(q['dt']), 1.0),
                                              q.get('sun') == '1', float(q.get('roi', 10))))
            if method == 'GET' and p == '/api/config_text':
                fid = common.normalize_name(q['name'])
                return self.send(200, {'path': common.config_path(fid), 'text': open(common.config_path(fid)).read()})
            if method == 'GET' and p == '/api/config_file':
                fid = common.normalize_name(q['name'])
                body = open(common.config_path(fid), 'rb').read()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Disposition', f'attachment; filename="config_{fid}.json"')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
            if method == 'GET' and p == '/api/job':
                return self.send(200, job_status(q['id'], int(q.get('offset', 0))))
            if method == 'POST' and p == '/api/job/stop':
                return self.send(200, stop_job(self.body()['id']))
            if method == 'POST' and p == '/api/fetch':
                b = self.body()
                fid = common.flare_id(b['t0'])
                cmd = [sys.executable, '-u', '-m', 'sunflare.fetch_data', '--t0', b['t0'],
                       '--tmin', str(float(b.get('tmin', -3600))), '--tmax', str(float(b.get('tmax', 14400)))]
                for k, flag in (('tend', '--tend'), ('peak', '--peak'), ('goes_class', '--goes-class'),
                                ('location', '--location'), ('ar', '--ar')):
                    if b.get(k):
                        cmd += [flag, str(b[k])]
                if b.get('sun_centered'):
                    cmd.append('--sun-centered')
                for k, flag in (('lat_ft1', '--lat-ft1'), ('lat_ft2', '--lat-ft2'), ('lle_dir', '--lle-dir')):
                    if b.get(k):
                        if not os.path.exists(os.path.expanduser(b[k])):
                            raise FileNotFoundError(f'{b[k]} does not exist')
                        cmd += [flag, os.path.expanduser(b[k])]
                if bool(b.get('lat_ft1')) != bool(b.get('lat_ft2')):
                    raise ValueError('Give both the local FT1 and FT2 files, or neither')
                os.makedirs(common.flare_dir(fid), exist_ok=True)
                for k in list(EV_CACHE):
                    if k[0] == fid:
                        EV_CACHE.pop(k)
                return self.send(200, {'id': start_job('fetch', fid, cmd), 'name': fid})
            if method == 'POST' and p == '/api/lat_bins':
                # Bayesian blocks of the Sun photons (same code as the LAT step), or rates of the given bins
                b = self.body()
                cfg = dict(common.DEFAULTS)
                cfg.update(b['config'])
                fid = common.normalize_name(cfg['name'])
                info = common.load_json(common.info_path(fid), {})
                if not info.get('has_lat'):
                    raise ValueError('No LAT data for this flare')
                from sunflare import lat_bins
                if b.get('bins') is not None:
                    bins, used, integ = [[float(x), float(y)] for x, y in b['bins']], 'custom', None
                else:
                    bins, used, integ = lat_bins.compute(cfg, info, common.flare_dir(fid), method=b.get('method', 'bb'),
                                                         t0=b.get('t0'), t1=b.get('t1'))
                rates = lat_bins.block_rates(cfg, info, common.flare_dir(fid), bins)
                return self.send(200, {'bins': bins, 'rates': rates, 'method': used, 'interval': integ})
            if method == 'POST' and p == '/api/config':
                path = common.write_config(self.body())
                return self.send(200, {'path': path, 'text': open(path).read()})
            if method == 'POST' and p == '/api/run':
                b = self.body()
                path = common.write_config(b['config'])
                fid = common.normalize_name(b['config']['name'])
                cmd = [sys.executable, '-u', '-m', 'sunflare.run_analysis', path, '--steps',
                       ','.join(b.get('steps') or ['lightcurves', 'localization', 'lat', 'spectral', 'report'])]
                return self.send(200, {'id': start_job('analysis', fid, cmd), 'config': path})
            return self.send(404, {'error': f'No route {method} {p}'})
        except (ValueError, KeyError, RuntimeError, FileNotFoundError, TypeError, OSError) as e:
            return self.send(400, {'error': f'{type(e).__name__}: {e}'})
        except Exception as e:
            traceback.print_exc()
            return self.send(500, {'error': f'{type(e).__name__}: {e}'})

    def do_GET(self):
        self.handle_any('GET')

    def do_POST(self):
        self.handle_any('POST')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8766)
    ap.add_argument('--host', default='127.0.0.1')
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f'Solar-flare console on http://{a.host}:{a.port}  (python: {sys.executable}, data: {common.FLARES})', flush=True)
    srv.serve_forever()
