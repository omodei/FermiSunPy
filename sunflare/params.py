"""User control of the model parameters: catalog, overrides and chaining between intervals.

Overrides live in the config as `param_overrides`, keyed by component parameter:

  "pion.s":                                   {"value": 4.0, "fix": true}            every joint-fit model with pion
  "exp_co_le+nar+brd+pion::exp_co_le.xc":     {"value": 1e4, "min": 1e3, "max": 5e4}  that model only (wins)
  "lat:pion.s":                               {"value": 3.5, "min": 2.5, "max": 5}    LAT-only likelihood model

Each entry may set any of value, min, max, fix. Parameters of the templates cannot leave their grids (hard limits).
With `chain_intervals`, each interval starts from the best fit of the previous interval (same model) instead of
from the configured starting values; fixed parameters keep their configured value.
"""
import functools
import math

# parameters never shown / edited: reference energies and template bookkeeping
HIDDEN = {'piv', 'scale', 'redshift'}
# the templates are interpolated on a grid of the index: they cannot be evaluated outside it
HARD = {('pion', 's'): (2.0, 6.0), ('nar', 's'): (2.6, 4.2), ('brd', 's'): (2.6, 4.2),
        ('lat:pion', 's'): (2.0, 6.0)}
FIELDS = ('value', 'min', 'max', 'fix')


def _describe(par, comp, prefix=''):
    lo, hi = par.bounds
    hard = HARD.get((prefix + comp, par.name))
    return {'name': par.name, 'value': float(par.value), 'min': None if lo is None else float(lo),
            'max': None if hi is None else float(hi), 'fix': bool(par.fix), 'unit': str(par.unit),
            'desc': par.description, 'hard': list(hard) if hard else None}


@functools.lru_cache(maxsize=1)
def catalog():
    """{'components': {comp: [param, ...]}, 'lat': {kind: [param, ...]}} with the default settings."""
    from sunflare.sunmodel import make_component
    from sunflare.common import COMPONENTS, LAT_MODELS
    from sunflare import latfit
    out = {'components': {}, 'lat': {}}
    for comp in COMPONENTS:
        f = make_component(comp)
        out['components'][comp] = [_describe(p, comp) for p in f.parameters.values() if p.name not in HIDDEN]
    for kind in LAT_MODELS:
        shape = latfit.lat_model(kind, 0.0, 0.0)[latfit.LAT_SOURCE].spectrum.main.shape
        out['lat'][kind] = [_describe(p, kind, 'lat:') for p in shape.parameters.values() if p.name not in HIDDEN]
    return out


def _num(x):
    return x is not None and isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate(overrides, models=()):
    """List of problems with a param_overrides dict (empty if fine)."""
    cat = catalog()
    errs = []
    for key, o in (overrides or {}).items():
        if not isinstance(o, dict) or not o:
            errs.append(f'{key}: empty setting')
            continue
        model = None
        if key.startswith('lat:'):
            comp, _, par = key[4:].partition('.')
            pars = {p['name']: p for p in cat['lat'].get(comp, [])}
            if comp not in cat['lat']:
                errs.append(f'{key}: unknown LAT model "{comp}"')
                continue
            hard = HARD.get(('lat:' + comp, par))
        else:
            if '::' in key:
                model, key2 = key.split('::', 1)
            else:
                key2 = key
            comp, _, par = key2.partition('.')
            if comp not in cat['components']:
                errs.append(f'{key}: unknown component "{comp}"')
                continue
            if model is not None and comp not in model.split('+'):
                errs.append(f'{key}: model {model} has no {comp} component')
            pars = {p['name']: p for p in cat['components'][comp]}
            hard = HARD.get((comp, par))
        if par not in pars:
            errs.append(f'{key}: {comp} has no editable parameter "{par}" (has {", ".join(pars)})')
            continue
        bad = [k for k in o if k not in FIELDS]
        if bad:
            errs.append(f'{key}: unknown field(s) {bad}')
        for k in ('value', 'min', 'max'):
            if k in o and o[k] is not None and not _num(o[k]):
                errs.append(f'{key}: {k} must be a number')
        d = pars[par]
        lo = o.get('min', d['min'])
        hi = o.get('max', d['max'])
        v = o.get('value', d['value'])
        if _num(lo) and _num(hi) and lo >= hi:
            errs.append(f'{key}: min ({lo:g}) must be below max ({hi:g})')
        if _num(v) and ((_num(lo) and v < lo) or (_num(hi) and v > hi)):
            errs.append(f'{key}: value {v:g} outside [{lo}, {hi}]')
        if hard:
            for k in ('value', 'min', 'max'):
                x = o.get(k)
                if _num(x) and not (hard[0] - 1e-9 <= x <= hard[1] + 1e-9):
                    errs.append(f'{key}: {k} {x:g} outside the template grid [{hard[0]:g}, {hard[1]:g}]')
    return errs


def resolve(overrides, comp, par, model=None, prefix=''):
    """Merged setting for one parameter: component level, then model level (wins). Returns (dict, sources)."""
    o, src = {}, []
    a = (overrides or {}).get(f'{prefix}{comp}.{par}')
    if a:
        o.update(a)
        src.append('all models')
    if model:
        b = (overrides or {}).get(f'{model}::{comp}.{par}')
        if b:
            o.update(b)
            src.append('this model')
    return o, src


def apply(param, o):
    """Apply one merged setting to an astromodels Parameter (bounds first, then value, then fix)."""
    if not o:
        return
    lo, hi = param.bounds
    lo = o.get('min', lo) if o.get('min') is not None else lo
    hi = o.get('max', hi) if o.get('max') is not None else hi
    param.bounds = (lo, hi)
    if o.get('value') is not None:
        param.value = float(o['value'])
    elif (lo is not None and param.value < lo) or (hi is not None and param.value > hi):
        param.value = min(max(param.value, lo if lo is not None else param.value), hi if hi is not None else param.value)
    if 'fix' in o and o['fix'] is not None:
        param.fix = bool(o['fix'])


def free_values(model, source):
    """{path: value} of the free spectral parameters of `source` (for chaining)."""
    return {k: float(p.value) for k, p in model.free_parameters.items() if k.startswith(source + '.')}


def start_from(model, values):
    """Set the free parameters of `model` to `values` (clipped to their bounds). Returns the number set."""
    n = 0
    for k, v in (values or {}).items():
        if k in model.free_parameters:
            p = model[k]
            lo, hi = p.bounds
            p.value = min(max(v, lo if lo is not None else v), hi if hi is not None else v)
            n += 1
    return n


def describe_fixed(pairs):
    """[[label, value, unit], ...] for the fixed parameters among (label, Parameter) pairs."""
    return [[lab, float(p.value), str(p.unit)] for lab, p in pairs if p.fix]


def _near(v, b, lo, hi):
    """True if v sits on bound b (relative tolerance for parameters spanning many decades)."""
    if b is None or v is None:
        return False
    if lo is not None and hi is not None and lo > 0 and hi / lo > 1e3:
        return abs(v - b) <= 2e-3 * abs(b)
    span = (hi - lo) if (lo is not None and hi is not None) else max(abs(b), 1.0)
    return abs(v - b) <= 1e-3 * span


def limit_hit(label, v, lo, hi, hard=None):
    """{'param', 'value', 'side', 'bound', 'grid'} if v is at one of its bounds, else None."""
    for side, b in (('lower', lo), ('upper', hi)):
        if _near(v, b, lo, hi):
            grid = bool(hard) and abs(b - hard[0 if side == 'lower' else 1]) < 1e-6
            return {'param': label, 'value': float(v), 'side': side, 'bound': float(b), 'grid': grid}
    return None


def limits_of_model(sm):
    """Free parameters of a fitted SunModel at their bounds (spectral.py)."""
    out = []
    for path, p in sm.model.free_parameters.items():
        lab = sm.short(path) if path.startswith(sm.source_name + '.') else path.split('.')[-1]
        comp, _, par = lab.partition('.')
        h = limit_hit(lab, float(p.value), *p.bounds, hard=HARD.get((comp, par)))
        if h:
            out.append(h)
    return out


def limits_of_lat(model, kind, source):
    """Free parameters of a fitted LAT model at their bounds (latfit.py)."""
    out = []
    for path, p in model.free_parameters.items():
        lab = p.name if path.startswith(source + '.') else path.split('.')[0]
        h = limit_hit(lab, float(p.value), *p.bounds, hard=HARD.get(('lat:' + kind, p.name)))
        if h:
            out.append(h)
    return out


def limits_from_values(values, model=None, overrides=None, lat_kind=None):
    """Fallback for results saved without 'at_limits': compare fitted values with the default bounds of the
    catalog, the user's settings and the template grids. values: {label: value} ('comp.par' or LAT 'par')."""
    cat = catalog()
    out = []
    for lab, v in values.items():
        if lat_kind:
            comp, par, pre = lat_kind, lab, 'lat:'
            d = {p['name']: p for p in cat['lat'].get(lat_kind, [])}.get(par)
        else:
            comp, _, par = lab.partition('.')
            pre = ''
            d = {p['name']: p for p in cat['components'].get(comp, [])}.get(par)
        if d is None:
            continue
        o, _ = resolve(overrides, comp, par, model=model, prefix=pre)
        lo = o.get('min') if o.get('min') is not None else d['min']
        hi = o.get('max') if o.get('max') is not None else d['max']
        h = limit_hit(f'{comp}.{par}' if lat_kind else lab, v, lo, hi, hard=HARD.get((pre + comp, par)))
        if h:
            out.append(h)
    return out


def describe_limits(hits):
    """Short text: 'pion.s = 6 at the upper end of the template grid; nar.s = 4.2 at its upper limit'."""
    return '; '.join(f'{h["param"]} = {h["value"]:.4g} at ' +
                     (f'the {h["side"]} end of the template grid' if h.get('grid') else f'its {h["side"]} limit')
                     for h in hits)


def fit_limits(f, model=None, lat_kind=None, overrides=None, ts_min=None):
    """Parameters at limits of one saved fit result (joint: params list; LAT: params dict). Uses the
    'at_limits' recorded at fit time, else limits_from_values. LAT bins below ts_min are not checked."""
    if not f or 'logL' not in f or (ts_min is not None and f.get('TS', 0) < ts_min):
        return []
    if 'at_limits' in f:
        return f['at_limits']
    if lat_kind:
        vals = {p: v[0] for p, v in (f.get('params') or {}).items()}
    else:
        vals = {p[0]: p[1] for p in f.get('params') or []}
    try:
        return limits_from_values(vals, model=model, overrides=overrides, lat_kind=lat_kind)
    except Exception:
        return []
