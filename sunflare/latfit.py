"""LAT-only likelihood fits of the Sun with threeML (pion template, power law, cutoff power law).

Follows the analysis used in solarflares/SolarFlare-*/timeresolved_analysis.py:
GtBurst (TransientLATDataBuilder) selects the data and computes exposures, threeML fits the Sun as a
point source with the Galactic template normalisation fixed to 1 and the isotropic template free.
"""
import numpy as np

LAT_SOURCE = 'Sun'


def lat_model(kind, ra, dec):
    from astromodels import PointSource, Model, Powerlaw, Cutoff_powerlaw, TemplateModel
    if kind == 'pion':
        f = TemplateModel('pion')
        f.K = 1e-4; f.K.bounds = (1e-14, 1e4); f.s = 4.0
        f.scale = 1.0; f.scale.fix = True
        if 'redshift' in f.parameters:
            f.redshift.fix = True
        shape = f
    elif kind == 'pow':
        f = Powerlaw(); f.piv = 2e5; f.K = 1e-9; f.K.bounds = (1e-20, 1e-2); f.index = -2.5; f.index.bounds = (-8, 0)
        shape = f
    elif kind == 'expco':
        f = Cutoff_powerlaw(); f.piv = 1e5; f.K = 1e-8; f.K.bounds = (1e-20, 1e-2)
        f.index = -1.0; f.index.bounds = (-5, 3); f.xc = 1e5; f.xc.bounds = (5e3, 1e8)
        shape = f
    else:
        raise ValueError(kind)
    return Model(PointSource(LAT_SOURCE, ra=ra, dec=dec, spectral_shape=shape))


SHAPE_PARS = {'pion': [('s', 'proton index s')], 'pow': [('index', 'photon index')],
              'expco': [('index', 'photon index'), ('xc', 'cutoff [keV]')]}


def _main(model):
    return model[LAT_SOURCE].spectrum.main.shape


def par_path(model, p):
    return f'{LAT_SOURCE}.spectrum.main.{_main(model).name}.{p}'


def fix_templates(model, fix_galactic=True):
    for p in list(model.parameters):
        if 'GalacticTemplate' in p and fix_galactic:
            model[p].value = 1.0
            model[p].fix = True


def apply_overrides(model, kind, overrides):
    """param_overrides with keys 'lat:<kind>.<par>'. Returns the settings applied."""
    from sunflare import params as _params
    applied = []
    shape = _main(model)
    for p in shape.parameters.values():
        if p.name in _params.HIDDEN:
            continue
        o, src = _params.resolve(overrides, kind, p.name, prefix='lat:')
        if o:
            _params.apply(model[par_path(model, p.name)], o)
            applied.append({'param': f'{kind}.{p.name}', 'setting': o, 'scope': 'LAT model'})
    return applied


def fit(plugin, kind, ra, dec, emin_mev, emax_mev, fix_shape=None, quiet=True, overrides=None, start=None):
    """Fit one LATLike plugin. Returns (jl, dict of results).

    overrides: param_overrides (keys 'lat:<kind>.<par>'); start: {path: value} from a previous interval."""
    from threeML import JointLikelihood, DataList
    import astropy.units as u
    from sunflare import params as _params
    model = lat_model(kind, ra, dec)
    applied = apply_overrides(model, kind, overrides)
    n_start = _params.start_from(model, start) if start else 0
    if fix_shape:
        for p, v in fix_shape.items():
            model[par_path(model, p)].value = v
            model[par_path(model, p)].fix = True
    jl = JointLikelihood(model, DataList(plugin), verbose=False)
    fix_templates(model)
    jl.set_minimizer('minuit')
    no_errors = False
    try:
        pars, like = jl.fit(quiet=quiet, compute_covariance=True)
    except IndexError:         # covariance sampling gave no valid sample (parameters at their bounds)
        pars, like = jl.fit(quiet=quiet, compute_covariance=False)
        no_errors = True
    except Exception:          # retry from a different starting point
        for p in model.free_parameters.values():
            if p.name in ('K', 'F'):
                p.value = p.value * 10
        try:
            pars, like = jl.fit(quiet=quiet, compute_covariance=True)
        except IndexError:
            pars, like = jl.fit(quiet=quiet, compute_covariance=False)
            no_errors = True
    ts = float(jl.compute_TS(LAT_SOURCE, like)['TS'].iloc[0])
    jl.restore_best_fit()
    out = {'model': kind, 'TS': ts, 'logL': float(-like['-log(likelihood)'].sum()), 'params': {},
           'overrides': applied, 'start': 'previous bin' if n_start else 'configured',
           'best': _params.free_values(model, LAT_SOURCE),
           'fixed': _params.describe_fixed([(p.name, p) for p in _main(model).parameters.values()
                                            if p.name not in _params.HIDDEN])}
    out['no_errors'] = no_errors
    try:
        df = jl.results.get_data_frame()
    except Exception:
        import pandas as pd
        df = pd.DataFrame(columns=['value', 'negative_error', 'positive_error'])
    for p, lab in SHAPE_PARS[kind]:
        k = par_path(model, p)
        if k in df.index:
            out['params'][p] = [float(df.loc[k, 'value']), float(df.loc[k, 'negative_error']),
                                float(df.loc[k, 'positive_error'])]
        else:
            out['params'][p] = [float(model[k].value), 0.0, 0.0]
    e1, e2 = emin_mev * u.MeV, emax_mev * u.MeV
    for unit, tag in (('1/(cm2 s)', 'flux'), ('erg/(cm2 s)', 'eflux')):
        try:
            f = jl.results.get_flux(ene_min=e1, ene_max=e2, flux_unit=unit, sources=[LAT_SOURCE], use_components=False)
            out[tag] = [float(f['flux'].iloc[0].value), float(f['low bound'].iloc[0].value), float(f['hi bound'].iloc[0].value)]
        except Exception as e:
            out[tag] = [np.nan, np.nan, np.nan]
            out['flux_error'] = str(e)
    return jl, out


def upper_limit(plugin, kind, ra, dec, emin_mev, emax_mev, overrides=None):
    """95% c.l. (one-sided) flux upper limit with the spectral shape fixed (pion s=4, PL index -2, or the
    value the user fixed in param_overrides)."""
    import astropy.units as u
    from sunflare import params as _params
    shape = {'pion': {'s': 4.0}, 'pow': {'index': -2.0}, 'expco': {'index': -1.0, 'xc': 2e5}}[kind]
    for p in shape:
        o, _ = _params.resolve(overrides, kind, p, prefix='lat:')
        if o.get('fix') and o.get('value') is not None:
            shape[p] = float(o['value'])
    jl, r = fit(plugin, kind, ra, dec, emin_mev, emax_mev, fix_shape=shape, overrides=overrides)
    for unit, tag in (('1/(cm2 s)', 'flux_ul'), ('erg/(cm2 s)', 'eflux_ul')):
        try:
            f = jl.results.get_flux(ene_min=emin_mev * u.MeV, ene_max=emax_mev * u.MeV, flux_unit=unit,
                                    sources=[LAT_SOURCE], use_components=False, confidence_level=0.90)
            r[tag] = float(f['hi bound'].iloc[0].value)
        except Exception:
            r[tag] = np.nan
    r['ul_shape'] = shape
    return r


def plugins_from_builder(b):
    """Run a TransientLATDataBuilder and return {(t0, t1): (LATLike plugin, observation)}."""
    obs = b.run(include_previous_intervals=True)
    out = {}
    for o in obs:
        key = (round(float(o.tstart), 3), round(float(o.tstop), 3))
        out[key] = (o.to_LATLike(), o)
    return out
