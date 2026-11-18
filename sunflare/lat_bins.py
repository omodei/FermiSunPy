"""LAT time bins for the time-resolved likelihood (shared by sunflare.lat_analysis and the GUI).

Photons used: within `roi` of the Sun (tracking its motion, or of (0, 0) in the Sun-centred frame), E >= LAT_EMIN_MEV,
zenith < zmax, and closer than 1.5 x r68(E) to the Sun. Bins are split at the Sun visibility windows and the parts
outside them (Sun occulted, SAA) are dropped.
"""
import numpy as np

from sunflare import common, lcdata, lat_data

MIN_BIN_S = 5.0


def windows(info):
    return [w for w in info.get('lat_windows', []) if w[1] > info['tmin'] and w[0] < info['tmax']]


def integrated_interval(cfg, info):
    integ = cfg.get('LAT_integrated_interval')
    if not integ:
        after = [w for w in windows(info) if w[1] > 0]
        integ = [max(0.0, after[0][0]), after[-1][1]] if after else [0.0, info['tmax']]
    return [float(integ[0]), float(integ[1])]


def clip_to_windows(intervals, wins, min_len=MIN_BIN_S):
    """Split [a, b] intervals at the visibility windows and drop the parts outside them."""
    res = []
    for a, b in intervals:
        for w0, w1 in wins:
            lo, hi = max(a, w0), min(b, w1)
            if hi - lo >= min_len:
                res.append([round(float(lo), 3), round(float(hi), 3)])
    return res


def sun_photon_times(fdir, info, cfg, t0, t1):
    ev = lcdata.lat_events(fdir, info, roi=float(cfg.get('roi', 10.0)), emin=float(cfg.get('LAT_EMIN_MEV', 60.0)),
                           zmax=float(cfg.get('zmax', 100.0)), sun_centered=bool(cfg.get('sun_centered')))
    near = ev['sep'] < 1.5 * lat_data.psf68(ev['E'])
    return np.sort(ev['t'][near & (ev['t'] >= t0) & (ev['t'] <= t1)])


def bayesian_blocks(t, t0, t1, p0, wins):
    from threeML.utils import bayesian_blocks as bb
    if len(t) < 5:
        return clip_to_windows([[t0, t1]], wins)
    e = bb.bayesian_blocks(t, t0, t1, p0=float(p0))
    return clip_to_windows(list(zip(e[:-1], e[1:])), wins)


def constant_counts(t, t0, t1, n, wins):
    n = int(n)
    if len(t) <= 2 * n:
        return clip_to_windows([[t0, t1]], wins)
    e = [t0] + [0.5 * (t[i - 1] + t[i]) for i in range(n, len(t) - n // 2, n)] + [t1]
    return clip_to_windows(list(zip(e[:-1], e[1:])), wins)


def compute(cfg, info, fdir, method=None, t0=None, t1=None):
    """(bins, method, integrated interval). `method` overrides cfg.LAT_TR_method; t0/t1 the integrated interval."""
    integ = integrated_interval(cfg, info) if t0 is None else [float(t0), float(t1)]
    wins = windows(info)
    method = method or cfg.get('LAT_TR_method', 'windows')
    if method == 'custom' and cfg.get('LAT_TR_custom'):
        bins = [[float(a), float(b)] for a, b in cfg['LAT_TR_custom']]
    elif method == 'bb':
        bins = bayesian_blocks(sun_photon_times(fdir, info, cfg, *integ), integ[0], integ[1],
                               cfg.get('LAT_TR_bb_p0', 0.05), wins)
    elif method == 'const':
        bins = constant_counts(sun_photon_times(fdir, info, cfg, *integ), integ[0], integ[1],
                               cfg.get('LAT_TR_N', 20), wins)
    else:
        method = 'windows'
        bins = clip_to_windows([integ], wins)
    return [b for b in bins if b[1] - b[0] >= MIN_BIN_S], method, integ


def block_rates(cfg, info, fdir, bins):
    """Sun-photon rate (counts/s) in each bin, for display."""
    if not bins:
        return []
    t = sun_photon_times(fdir, info, cfg, min(a for a, _ in bins), max(b for _, b in bins))
    return [float(((t >= a) & (t < b)).sum() / max(b - a, 1e-9)) for a, b in bins]


if __name__ == '__main__':
    import sys
    c = common.load_config(sys.argv[1])
    b, m, i = compute(c, c.info, c.pwd, method=sys.argv[2] if len(sys.argv) > 2 else None)
    print(m, i, len(b))
    for x in b:
        print(x)
