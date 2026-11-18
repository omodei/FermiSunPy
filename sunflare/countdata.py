"""Count spectra of fitted threeML plugins, stored for the report's interactive spectrum viewer.

data(plugin)   observed counts, background and selection of a plugin (the same for every model)
model(plugin)  predicted counts of the current (best-fit) model
Energies are channel edges in keV. GBM/LLE (SpectrumLike): background counts scaled to the source exposure and
source-model counts. LAT (LATLike): model counts per likelihood component (Sun, isotropic, Galactic).
"""
import numpy as np


def _sig(x, n=4):
    return [float(f'{v:.{n}g}') for v in np.asarray(x, dtype=float)]


def is_lat(plugin):
    return hasattr(plugin, 'like') and hasattr(plugin.like, '_srcCnts')


def data(plugin):
    if is_lat(plugin):
        from sunflare.latfit import count_spectrum
        c = count_spectrum(plugin)
        return {'kind': 'lat', 'E': _sig(np.asarray(c['E']) * 1e3, 6), 'n': c['n'],
                't': float(getattr(plugin, '_FermiLATLike__observation_duration', 0) or 0)}
    sp = plugin._observed_spectrum
    expo = float(sp.exposure)
    out = {'kind': 'spec', 'E': _sig(sp.edges, 6), 'n': [int(round(v)) for v in np.asarray(plugin.observed_counts)],
           't': expo, 'mask': [int(bool(m)) for m in np.asarray(plugin.mask)]}
    try:
        b = plugin.background_counts
        if b is not None:
            out['bkg'] = _sig(np.asarray(b, dtype=float) / plugin._background_exposure * plugin._area_ratio * expo)
    except Exception:
        pass
    return out


def model(plugin):
    if is_lat(plugin):
        from sunflare.latfit import count_spectrum
        return count_spectrum(plugin)['m']
    sp = plugin._observed_spectrum
    return {'source': _sig(np.asarray(plugin.expected_model_rate, dtype=float) * float(sp.exposure))}


def vfv(sun_model, emin_kev, emax_kev, n=240):
    """E^2 dN/dE [erg cm-2 s-1] of the fitted model and of its components on a log grid."""
    e = np.logspace(np.log10(emin_kev), np.log10(emax_kev), n)
    sc = e ** 2 * 1.602176634e-9
    out = {'total': _sig(sun_model.model[sun_model.source_name].spectrum.main.shape(e) * sc), 'comps': {}}
    if sun_model.composite:
        for c, f in zip(sun_model.components, sun_model.component_functions()):
            out['comps'][c] = _sig(f(e) * sc)
    return _sig(e, 5), out
