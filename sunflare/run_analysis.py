#!/usr/bin/env python
"""Run the analysis steps for one flare and build results/report.html.

Usage: python -m sunflare.run_analysis flares/<id>/config_<id>.json [--steps lightcurves,localization,lat,spectral,report]
"""
import argparse
import datetime
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from sunflare import common  # noqa: E402

STEPS = {'lightcurves': 'lightcurves', 'localization': 'localization', 'lat': 'lat_analysis',
         'spectral': 'spectral', 'report': 'make_report'}
ALL = list(STEPS)

ap = argparse.ArgumentParser()
ap.add_argument('config')
ap.add_argument('--steps', default=None, help='default: run_steps saved in the config, else all steps')
args = ap.parse_args()
cfg = common.load_config(args.config)
steps = [s for s in (args.steps.split(',') if args.steps else (cfg.run_steps or ALL)) if s in STEPS]
if not cfg.info.get('has_lat') or not cfg.use_lat:
    steps = [s for s in steps if s not in ('localization', 'lat')]
if not cfg.do_localization and 'localization' in steps:
    steps.remove('localization')

status_file = os.path.join(cfg.pwd, 'results', 'run_status.json')
status = {'started': datetime.datetime.now().isoformat(timespec='seconds'), 'config': cfg.config_file,
          'steps': {s: {'state': 'pending'} for s in steps}}
common.save_json(status_file, status)
failed = False
for name in steps:
    print(f'### STEP {name} START', flush=True)
    status['steps'][name] = {'state': 'running', 'start': time.time()}
    common.save_json(status_file, status)
    p = subprocess.run([sys.executable, '-u', '-m', f'sunflare.{STEPS[name]}', cfg.config_file],
                       cwd=common.REPO, env=dict(os.environ, PYTHONPATH=os.pathsep.join([common.REPO, os.environ.get('PYTHONPATH', '')]),
                                                 MPLBACKEND='Agg'))
    st = status['steps'][name]
    st['state'] = 'done' if p.returncode == 0 else 'failed'
    st['seconds'] = round(time.time() - st.pop('start'), 1)
    failed |= p.returncode != 0
    common.save_json(status_file, status)
    print(f'### STEP {name} {st["state"].upper()} ({st["seconds"]} s)', flush=True)
status['finished'] = datetime.datetime.now().isoformat(timespec='seconds')
common.save_json(status_file, status)
sys.exit(1 if failed else 0)
