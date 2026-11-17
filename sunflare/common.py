"""Shared helpers for the solar-flare pipeline and GUI.

Nothing here imports threeML or the fermitools, so the GUI server can use it without the slow import.
Times inside a flare directory are seconds relative to the flare reference time T0 (MET).
"""
import datetime
import glob
import hashlib
import json
import os
import re
import shutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLARES = os.environ.get('SUNFLARE_DATA', os.path.join(REPO, 'flares'))
TEMPLATE_DIR = os.environ.get('SUNFLARE_TEMPLATES',
                              os.path.join(os.path.dirname(REPO), 'solarflares', 'Sun', 'data'))
for _cand in (TEMPLATE_DIR, os.path.expanduser('~/gitrepository/solarflares/Sun/data')):
    if os.path.isdir(_cand):
        TEMPLATE_DIR = _cand
        break

MET_EPOCH = '2001-01-01T00:00:00'
GBM_DETS = ['n%d' % i for i in range(10)] + ['na', 'nb', 'b0', 'b1']

# ------------------------------------------------------------------ defaults
DEFAULTS = {
    # data download (relative to T0)
    'DWNL_TMIN': -3600.0,
    'DWNL_TMAX': 14400.0,
    # GBM
    'use_gbm': True,
    'spec_use_nai': True,         # NaI detectors in the joint spectral fit
    'spec_use_bgo': True,         # BGO detectors in the joint spectral fit
    'GBM_detectors': [],
    'GBM_bkg_intervals': [],
    'GBM_bkg_poly_order': -1,
    'GBM_NAI_EMIN_KEV': 10.0,
    'GBM_NAI_EMAX_KEV': 900.0,
    'GBM_BGO_EMIN_KEV': 300.0,
    'GBM_BGO_EMAX_KEV': 30000.0,
    'effective_area_correction': 0.0,
    # LLE (only for GBM/LAT triggers that have LLE data)
    'use_lle': False,
    'LLE_bkg_intervals': [],
    'LLE_bkg_poly_order': -1,
    'LLE_EMIN_MEV': 30.0,
    'LLE_EMAX_MEV': 300.0,
    # LAT
    'use_lat': True,
    'sun_centered': False,
    'irfs': 'p8_source',
    'roi': 10.0,
    'zmax': 100.0,
    'thetamax': 180.0,
    'strategy': 'time',
    'LAT_EMIN_MEV': 60.0,
    'LAT_EMAX_MEV': 10000.0,
    'FLUX_EMIN_MEV': 100.0,
    'FLUX_EMAX_MEV': 1000.0,
    'LAT_TS_MIN': 9.0,
    'LAT_VIS_THETA_MAX': 70.0,
    # localization
    'do_localization': True,
    'LOC_interval': None,
    'LOC_EMIN_MEV': 100.0,
    'LOC_HALF_SIZE': 1.5,
    'LOC_NSIDE': 21,
    # LAT time-resolved likelihood
    'LAT_integrated_interval': None,
    'LAT_TR_method': 'windows',   # windows | bb | const | custom
    'LAT_TR_bb_p0': 0.05,
    'LAT_TR_N': 20,
    'LAT_TR_custom': [],
    'LAT_TR_models': ['pion', 'pow', 'expco'],
    # joint spectral analysis
    'spectral_intervals': [],
    'spectral_models': ['exp_co_le+nar+brd+g_2p2MeV', 'exp_co_le+nar+brd+g_2p2MeV+pion'],
    'spec_use_lat': True,
    'SPEC_LAT_EMIN_MEV': 60.0,
    'SPEC_LAT_EMAX_MEV': 10000.0,
    # light curves in the report
    'lc_tmin': None,
    'lc_tmax': None,
    'nai_bands_kev': [[10, 25], [25, 50], [50, 100], [100, 300]],
    'bgo_bands_kev': [[300, 1000], [1000, 10000], [10000, 40000]],
    'notes': '',
}

IRFS_DEFAULT = ['p8_source', 'p8_transient015s', 'p8_transient020', 'p8_transient010', 'p8_transient020e',
                'p8_transient010e', 'p8_clean', 'p8_ultraclean']

# Spectral components of the joint fit (keys are what the model strings are made of, e.g. 'exp_co_le+nar+pion').
COMPONENTS = {
    'pow_le':    {'label': 'Power law (electron bremsstrahlung, pivot 1 MeV)', 'group': 'Continuum'},
    'exp_co_le': {'label': 'Cutoff power law (electron bremsstrahlung, pivot 1 MeV)', 'group': 'Continuum'},
    'pow_he':    {'label': 'Power law (high energy, pivot 30 MeV)', 'group': 'Continuum'},
    'exp_co_he': {'label': 'Cutoff power law (high energy, pivot 30 MeV)', 'group': 'Continuum'},
    'nar':       {'label': 'Narrow nuclear de-excitation lines (template, index s)', 'group': 'Nuclear lines'},
    'brd':       {'label': 'Broad nuclear de-excitation lines (template, index s)', 'group': 'Nuclear lines'},
    'g_2p2MeV':  {'label': '2.223 MeV neutron-capture line (Gaussian)', 'group': 'Nuclear lines'},
    'g_511keV':  {'label': '511 keV positron-annihilation line (Gaussian)', 'group': 'Nuclear lines'},
    'g_1p7MeV':  {'label': '1.7 MeV line (Gaussian)', 'group': 'Nuclear lines'},
    'pion':      {'label': 'Pion-decay template (proton index s)', 'group': 'Pion decay'},
}
MODEL_PRESETS = [
    'exp_co_le+nar+brd+g_2p2MeV',
    'exp_co_le+nar+brd+g_2p2MeV+pion',
    'pow_le+nar+brd+g_2p2MeV+g_511keV+pion',
    'exp_co_le+nar+brd+exp_co_he',
    'exp_co_le+pion',
    'pow_le+exp_co_he',
]
LAT_MODELS = {'pion': 'Pion-decay template (proton index s)', 'pow': 'Power law (photon index)',
              'expco': 'Power law with exponential cutoff'}


# ------------------------------------------------------------------ time
def utc2met(utc):
    """'2025-11-11 09:30:00' / ISO / datetime -> Fermi MET (leap seconds included)."""
    from astropy.time import Time
    return float((Time(utc, scale='utc') - Time(MET_EPOCH, scale='utc')).sec)


def met2utc(met, fmt='iso'):
    from astropy.time import Time
    import astropy.units as u
    t = Time(MET_EPOCH, scale='utc') + float(met) * u.s
    return t.iso if fmt == 'iso' else t.datetime


def met2datetime_array(met):
    import numpy as np
    from astropy.time import Time
    import astropy.units as u
    return (Time(MET_EPOCH, scale='utc') + np.asarray(met, dtype=float) * u.s).datetime


# ------------------------------------------------------------------ names / paths
def flare_id(utc):
    """Reference time -> 'SOL20251111_0930'."""
    if not isinstance(utc, datetime.datetime):
        utc = datetime.datetime.fromisoformat(str(utc).replace('Z', '').replace('T', ' ').strip()[:19])
    return utc.strftime('SOL%Y%m%d_%H%M')


def normalize_name(name):
    s = re.sub(r'\s', '', str(name))
    m = re.match(r'^SOL(\d{8})_(\d{4})$', s, re.I)
    if m:
        return 'SOL' + m.group(1) + '_' + m.group(2)
    raise ValueError(f'Flare names look like SOL20251111_0930, not "{name}"')


def flare_dir(fid):
    return os.path.join(FLARES, fid)


def config_path(fid):
    return os.path.join(flare_dir(fid), f'config_{fid}.json')


def info_path(fid):
    return os.path.join(flare_dir(fid), 'flare_info.json')


def results_dir(fid):
    return os.path.join(flare_dir(fid), 'results')


def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save_json(path, obj):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


class Config(dict):
    """dict with attribute access and defaults filled in."""
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


def load_config(path_or_id):
    p = path_or_id if os.path.exists(str(path_or_id)) else config_path(normalize_name(path_or_id))
    c = dict(DEFAULTS)
    c.update(load_json(p, {}))
    cfg = Config(c)
    cfg['config_file'] = os.path.abspath(p)
    cfg['pwd'] = os.path.dirname(os.path.abspath(p))
    cfg['info'] = load_json(os.path.join(cfg['pwd'], 'flare_info.json'), {})
    return cfg


def write_config(cfg):
    """Validate and write config_<id>.json from a dict sent by the GUI. Returns the path."""
    fid = normalize_name(cfg['name'])
    c = dict(DEFAULTS)
    c.update({k: v for k, v in cfg.items() if k not in ('info', 'pwd', 'config_file')})
    for k in ('GBM_bkg_intervals', 'LLE_bkg_intervals', 'spectral_intervals', 'LAT_TR_custom'):
        c[k] = sorted([[float(a), float(b)] for a, b in c.get(k) or [] if float(b) > float(a)])
    for k in ('LOC_interval', 'LAT_integrated_interval'):
        v = c.get(k)
        c[k] = [float(v[0]), float(v[1])] if v and float(v[1]) > float(v[0]) else None
    gbm_in_fit = c['use_gbm'] and any((d[0] == 'n' and c['spec_use_nai']) or (d[0] == 'b' and c['spec_use_bgo'])
                                      for d in c.get('GBM_detectors') or [])
    if c['spectral_intervals'] and gbm_in_fit and not c['GBM_bkg_intervals']:
        raise ValueError('The joint spectral fit needs at least one GBM background interval')
    if c['use_lle'] and c['spectral_intervals'] and not c['LLE_bkg_intervals']:
        raise ValueError('Select at least one LLE background interval, or switch LLE off')
    for m in c['spectral_models']:
        bad = [x for x in m.split('+') if x not in COMPONENTS]
        if bad:
            raise ValueError(f'Unknown spectral component(s) {bad} in model "{m}"')
    bad = [m for m in c['LAT_TR_models'] if m not in LAT_MODELS]
    if bad:
        raise ValueError(f'Unknown LAT model(s) {bad}')
    c['name'] = fid
    c['written'] = datetime.datetime.now().isoformat(timespec='seconds')
    path = config_path(fid)
    save_json(path, c)
    return path


def settings_hash(d):
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]


def ensure_fresh(directory, settings):
    """Make `directory`; wipe it if it was produced with different settings."""
    os.makedirs(directory, exist_ok=True)
    stamp = os.path.join(directory, 'settings.json')
    old = load_json(stamp)
    if old is not None and settings_hash(old) != settings_hash(settings):
        for p in glob.glob(os.path.join(directory, '*')):
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    with open(stamp, 'w') as f:
        json.dump(settings, f, indent=1, default=str)


def find_version(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def interval_strings(pairs):
    """[(a, b), ...] -> ['a-b', ...] as threeML expects."""
    return [f'{float(a):g}-{float(b):g}' for a, b in pairs]


def lat_dir(cfg_or_info, sun_centered=None):
    """Directory used as GtBurst 'datarepository' (contains bn<name>/gll_ft1_tr_bn<name>_v00.fit)."""
    d = cfg_or_info.get('pwd') or os.path.dirname(cfg_or_info.get('config_file', ''))
    sc = cfg_or_info.get('sun_centered') if sun_centered is None else sun_centered
    return os.path.join(d, 'LAT_data_sun' if sc else 'LAT_data')


def source_position(cfg):
    """Position of the Sun in the frame used by the LAT analysis."""
    if cfg.get('sun_centered'):
        return 0.0, 0.0
    return float(cfg['info']['sun_ra']), float(cfg['info']['sun_dec'])
