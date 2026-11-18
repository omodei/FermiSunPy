"""Solar-flare spectral models for threeML (port of solarflares/scripts/SunModel.py and SolarTemplates.py).

A model is a '+'-separated string of components, e.g. 'exp_co_le+nar+brd+g_2p2MeV+pion':

  pow_le / exp_co_le   power law / cutoff power law, pivot 1 MeV (electron bremsstrahlung)
  pow_he / exp_co_he   power law / cutoff power law, pivot 30 MeV
  nar / brd            narrow / broad nuclear de-excitation line templates (ion spectral index s)
  pion                 pion-decay template (proton spectral index s)
  g_2p2MeV, g_511keV, g_1p7MeV   Gaussian lines (2.223 MeV neutron capture, 511 keV annihilation, 1.7 MeV)

Energies are in keV (threeML convention); templates are in ph cm^-2 s^-1 keV^-1.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from sunflare.common import TEMPLATE_DIR, COMPONENTS  # noqa: E402

TEMPLATE_NAMES = {'pion': 'pion', 'nar': 'NAR', 'brd': 'BRD'}


# ------------------------------------------------------------------ templates
def _read_template(path):
    data = np.genfromtxt(path)[::10].transpose()
    data[1][-1] = 0
    data[data == 0] = 1e-50
    return data


def build_pion_template(template_dir, idxmin=20, idxmax=60):
    import astropy.units as u
    from astromodels import TemplateModelFactory
    ids = np.arange(idxmin, idxmax + 1)
    data = {i: _read_template(os.path.join(template_dir, 'piontemplate', f'pi_ap010_s{i}_b300_nh1e15_MeV.dat')) for i in ids}
    t = TemplateModelFactory('pion', 'Pion-decay template (proton index s)', data[idxmin][0] * u.MeV, ['s'])
    t.define_parameter_grid('s', ids / 10.)
    for i in ids:
        t.add_interpolation_data(data[i][1] / 1000., s=i / 10.)      # per MeV -> per keV
    t.save_data(overwrite=True)


def build_nuclear_template(template_dir, kind='NAR', idxmin=260, idxmax=420):
    import astropy.units as u
    from astromodels import TemplateModelFactory
    ids = np.arange(idxmin, idxmax + 1, 40)
    data = {i: _read_template(os.path.join(template_dir, 'NuclearLines', f'{kind}_ap022_s{i}_theta60.txt')) for i in ids}
    t = TemplateModelFactory(kind, f'{kind} nuclear-line template (ion index s)', data[idxmin][0] * u.keV, ['s'])
    t.define_parameter_grid('s', ids / 100.)
    for i in ids:
        t.add_interpolation_data(data[i][1], s=i / 100.)
    t.save_data(overwrite=True)


def ensure_templates(template_dir=TEMPLATE_DIR, log=print):
    """Build the pion / NAR / BRD astromodels templates if they are not in ~/.astromodels/data yet."""
    from astromodels.functions.template_model import get_user_data_path
    base = get_user_data_path()
    for kind in ('pion', 'NAR', 'BRD'):
        if os.path.exists(os.path.join(base, f'{kind}.h5')):
            continue
        if not os.path.isdir(template_dir):
            raise FileNotFoundError(f'Template {kind} missing and template data not found in {template_dir} '
                                    '(set SUNFLARE_TEMPLATES to solarflares/Sun/data)')
        log(f'building the {kind} template from {template_dir}')
        if kind == 'pion':
            build_pion_template(template_dir)
        else:
            build_nuclear_template(template_dir, kind)


# ------------------------------------------------------------------ components
def make_component(name):
    """New astromodels function for a component, with starting values and bounds."""
    from astromodels import Powerlaw, Cutoff_powerlaw, Gaussian, TemplateModel
    if name == 'pow_le':
        f = Powerlaw(); f.piv = 1000.; f.K = 1e-3; f.K.bounds = (1e-12, 1e3); f.index = -2.4; f.index.bounds = (-7, 1)
    elif name == 'exp_co_le':
        f = Cutoff_powerlaw(); f.piv = 1000.; f.K = 1e-3; f.K.bounds = (1e-12, 1e3)
        f.index = -2.4; f.index.bounds = (-7, 2); f.xc = 1e4; f.xc.bounds = (30., 1e6)
    elif name == 'pow_he':
        f = Powerlaw(); f.piv = 30000.; f.K = 1e-6; f.K.bounds = (1e-14, 1e1); f.index = -2.0; f.index.bounds = (-7, 1)
    elif name == 'exp_co_he':
        f = Cutoff_powerlaw(); f.piv = 30000.; f.K = 1e-6; f.K.bounds = (1e-14, 1e1)
        f.index = -1.0; f.index.bounds = (-6, 3); f.xc = 1e5; f.xc.bounds = (1e4, 1e7)
    elif name in TEMPLATE_NAMES:
        f = TemplateModel(TEMPLATE_NAMES[name])
        f.scale = 1.0; f.scale.fix = True
        if 'redshift' in f.parameters:
            f.redshift.fix = True
        if name == 'pion':
            f.K = 1e-4; f.K.bounds = (1e-12, 1e4); f.s = 4.0
        else:
            f.K = 1.0; f.K.bounds = (1e-8, 1e5); f.s = 3.4
    elif name == 'g_2p2MeV':
        f = Gaussian(); f.F = 1e-2; f.F.bounds = (1e-8, 1e2); f.mu = 2223.; f.mu.bounds = (2150., 2300.)
        f.sigma = 10.; f.sigma.fix = True
    elif name == 'g_511keV':
        f = Gaussian(); f.F = 1e-2; f.F.bounds = (1e-8, 1e2); f.mu = 511.; f.mu.fix = True; f.sigma = 5.; f.sigma.fix = True
    elif name == 'g_1p7MeV':
        f = Gaussian(); f.F = 1e-3; f.F.bounds = (1e-8, 1e2); f.mu = 1700.; f.mu.fix = True; f.sigma = 10.; f.sigma.fix = True
    else:
        raise ValueError(f'Unknown component {name}; known: {", ".join(COMPONENTS)}')
    return f


def make_component_params(name):
    """Editable parameters of a component (cached default instances)."""
    from sunflare.params import HIDDEN
    if name not in _PARAMS_CACHE:
        _PARAMS_CACHE[name] = [p for p in make_component(name).parameters.values() if p.name not in HIDDEN]
    return _PARAMS_CACHE[name]


_PARAMS_CACHE = {}


class SunModel:
    """Composite Sun model built from a component string."""

    def __init__(self, model_string, ra=0.0, dec=0.0, link_line_index=True, source_name='Sun', overrides=None):
        from astromodels import PointSource, Model
        self.name = model_string
        self.components = model_string.split('+')
        self.source_name = source_name
        funcs = [make_component(c) for c in self.components]
        shape = funcs[0]
        for f in funcs[1:]:
            shape = shape + f
        self.composite = len(funcs) > 1
        self.model = Model(PointSource(source_name, ra=ra, dec=dec, spectral_shape=shape))
        self.linked = bool(link_line_index and 'nar' in self.components and 'brd' in self.components)
        # user settings (sunflare.params): component level, then model level
        from sunflare import params as _params
        self.applied = []
        for c in self.components:
            for pname in [p.name for p in make_component_params(c)]:
                if self.linked and c == 'brd' and pname == 's':
                    continue                      # follows nar.s
                o, src = _params.resolve(overrides, c, pname, model_string)
                if o:
                    _params.apply(self.par(c, pname), o)
                    self.applied.append({'param': f'{c}.{pname}', 'setting': o, 'scope': ' + '.join(src)})
        if self.linked:
            self.model.link(self.par('brd', 's'), self.par('nar', 's'))

    def path(self, comp, par):
        i = self.components.index(comp)
        if self.composite:
            return f'{self.source_name}.spectrum.main.composite.{par}_{i + 1}'
        fname = self.model[self.source_name].spectrum.main.shape.name
        return f'{self.source_name}.spectrum.main.{fname}.{par}'

    def par(self, comp, par):
        return self.model[self.path(comp, par)]

    def component_functions(self):
        sh = self.model[self.source_name].spectrum.main.shape
        return list(sh.functions) if self.composite else [sh]

    def editable(self):
        """[(label, Parameter)] for the user-editable parameters ('brd.s' omitted when linked to nar.s)."""
        return [(f'{c}.{p.name}', self.par(c, p.name)) for c in self.components for p in make_component_params(c)
                if not (self.linked and c == 'brd' and p.name == 's')]

    def short(self, path):
        """'Sun.spectrum.main.composite.s_3' -> 'brd.s'."""
        for c in self.components:
            for p in make_component_params(c):
                if self.path(c, p.name) == path:
                    return f'{c}.{p.name}'
        return path.replace(f'{self.source_name}.spectrum.main.composite.', '').replace(f'{self.source_name}.spectrum.main.', '')

    def describe(self):
        return ' + '.join(COMPONENTS[c]['label'] for c in self.components)


def plot_model(sun_model, ax, emin=100., emax=1e7, n=600, vfv=True, label=None, color='k', components=True,
               alpha=1.0, lw=2):
    """Plot a (fitted) SunModel in E^2 dN/dE [erg cm^-2 s^-1] or dN/dE."""
    e = np.logspace(np.log10(emin), np.log10(emax), n)
    kev2erg = 1.602176634e-9
    sc = (e ** 2 * kev2erg) if vfv else 1.0
    tot = sun_model.model[sun_model.source_name].spectrum.main.shape(e)
    ax.plot(e / 1000., tot * sc, color=color, lw=lw, alpha=alpha, label=label or sun_model.name)
    if components and sun_model.composite:
        ls = ['--', '-.', ':', (0, (5, 1)), (0, (3, 1, 1, 1))]
        for i, (c, f) in enumerate(zip(sun_model.components, sun_model.component_functions())):
            ax.plot(e / 1000., f(e) * sc, ls=ls[i % len(ls)], lw=1.2, color=color, alpha=0.8 * alpha, label=f'  {c}')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Energy [MeV]')
    ax.set_ylabel(r'$E^2\,dN/dE$ [erg cm$^{-2}$ s$^{-1}$]' if vfv else r'dN/dE [ph cm$^{-2}$ s$^{-1}$ keV$^{-1}$]')
    vals = tot * sc
    good = vals[np.isfinite(vals) & (vals > 0)]
    top = np.percentile(good, 97) if good.size else 1      # ignore the spikes of narrow Gaussian lines
    lo, hi = ax.get_ylim() if ax.has_data() and getattr(ax, '_sf_ylim', False) else (np.inf, 0)
    lo, hi = min(lo, top * 1e-4), max(hi, top * 6)
    ax.set_ylim(lo, hi)
    ax._sf_ylim = True


if __name__ == '__main__':
    import argparse
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = argparse.ArgumentParser(description='Display a solar-flare model')
    p.add_argument('model', nargs='?', default='exp_co_le+nar+brd+g_2p2MeV+pion')
    p.add_argument('--out', default='sun_model.png')
    a = p.parse_args()
    ensure_templates()
    m = SunModel(a.model)
    m.model.display(complete=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_model(m, ax, 100, 1e7)
    ax.legend(fontsize=9)
    fig.savefig(a.out, dpi=110, bbox_inches='tight')
    print('saved', a.out)
