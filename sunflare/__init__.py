"""FermiSunPy solar-flare analysis pipeline (GOES + GBM + LLE + LAT) used by the analysis console in gui/."""
import os
import sys


def _fermitools_env():
    """Set the variables of the fermitools conda activation script when Python was started without
    `conda activate` (otherwise pyLikelihood aborts looking for the CALDB)."""
    prefix = os.environ.get('CONDA_PREFIX') or sys.prefix
    inst = os.path.join(prefix, 'share', 'fermitools')
    if not os.path.isdir(inst) or os.environ.get('CALDB'):
        return
    env = {'FERMI_DIR': inst, 'FERMI_INST_DIR': inst, 'INST_DIR': inst, 'BASE_DIR': inst,
           'EXTFILESSYS': os.path.join(inst, 'refdata', 'fermi'),
           'GENERICSOURCESDATAPATH': os.path.join(inst, 'data', 'genericSources'),
           'TIMING_DIR': os.path.join(inst, 'refdata', 'fermi', 'jplephem'),
           'CALDBALIAS': os.path.join(inst, 'data', 'caldb', 'software', 'tools', 'alias_config.fits'),
           'CALDBCONFIG': os.path.join(inst, 'data', 'caldb', 'software', 'tools', 'caldb.config'),
           'CALDBROOT': os.path.join(inst, 'data', 'caldb'), 'CALDB': os.path.join(inst, 'data', 'caldb'),
           'PFILES': '.:' + os.path.expanduser('~/pfiles') + ';' + os.path.join(inst, 'syspfiles')}
    for k, v in env.items():
        os.environ.setdefault(k, v)
    os.environ['PATH'] = os.path.join(prefix, 'bin') + os.pathsep + os.environ.get('PATH', '')
    os.environ.setdefault('CONDA_PREFIX', prefix)


_fermitools_env()
